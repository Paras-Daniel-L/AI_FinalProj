"""
LLM calls: RAG-grounded answers, checked by a second, different model.

Pulled out of api.py so the model names/temperatures live in one place. The
actual HTTP calls (thinking off, output caps, retries, cost logging) live in
app/llm_client.py; models are reached through OpenRouter by default.

Sagot AI has no open-domain/conversational generation path anymore — see
prompts.SYSTEM_PROMPT and app/greetings.py. The only LLM call left is the
grounded, retrieval-anchored one below.
"""

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from . import answerability, grounding, llm_client, progress
from .language import LanguageResult, detect_language
from .prompts import (
    ACTIVE_VERIFY_PROMPT,
    NO_ANSWER_SENTINEL,
    RAG_PROMPT,
    SYSTEM_PROMPT,
    VERIFIER_RELEVANCE,
    VERIFIER_SYSTEM_PROMPT,
    get_no_answer_message,
)
from .retrieval import format_history
from .schemas import ConvMessage

load_dotenv()  # the model settings below are read at import time

# The generator and the verifier are deliberately DIFFERENT models from
# different vendors. A verifier that is the same model as the generator shares
# its blind spots: whatever fact it confidently invents, it will tend to
# confidently approve. Both are overridable in .env so the thesis ablations
# (same model vs. different model) need no code change.
# Model ids are OpenRouter ids. Generator: DeepSeek V4.1 Flash. Verifier:
# Alibaba's Qwen 3.8 27B (a different vendor and family from the generator).
# Thinking is turned OFF for both in llm_client.py; _parse_verdict() below
# still tolerates thinking text in case a host ignores that.
GENERATOR_MODEL = os.environ.get("GENERATOR_MODEL", "deepseek/deepseek-v4.1-flash").strip()
VERIFIER_MODEL = os.environ.get("VERIFIER_MODEL", "qwen/qwen3.8-27b").strip()

# Hard output caps per call (tokens). They bound the worst-case cost of one
# call and stop a runaway reply. A reply that hits the cap is treated as a
# failed attempt, never returned as an answer.
GENERATOR_MAX_TOKENS = int(os.environ.get("GENERATOR_MAX_TOKENS", "1500"))
VERIFIER_MAX_TOKENS = int(os.environ.get("VERIFIER_MAX_TOKENS", "600"))


def _providers(name: str) -> List[str]:
    """Optional OpenRouter host pinning, e.g. VERIFIER_PROVIDERS=DeepInfra.
    Different hosts serve the same model at different prices and precisions
    (some run 4-bit), so pinning also makes thesis results reproducible."""
    return [p.strip() for p in os.environ.get(name, "").split(",") if p.strip()]


GENERATOR_PROVIDERS = _providers("GENERATOR_PROVIDERS")
VERIFIER_PROVIDERS = _providers("VERIFIER_PROVIDERS")
MODEL_NAME = GENERATOR_MODEL  # kept for any code that still imports the old name

# Both calls are grounded/audit tasks, so they run (near-)deterministic.
# Kept separate so they can be tuned independently for the thesis ablations.
# GENERATION_TEMPERATURE was a fixed 0.1; in a re-run of the 60 Subset B
# questions only 8/58 answers came back identical (the retrieved context was
# identical for 57/58), so per-question scores moved by up to a full point
# between runs. 0.0 removes the sampling part of that variance; host-side
# nondeterminism remains (pin providers + LLM_SEED for evaluation runs).
GENERATION_TEMPERATURE = float(os.environ.get("GENERATION_TEMPERATURE", "0.0"))
VERIFIER_TEMPERATURE = float(os.environ.get("VERIFIER_TEMPERATURE", "0.0"))
# Generate -> verify attempts before falling back to the canned refusal.
# Worst case = 2 * MAX_RAG_RETRIES model calls (3 attempts = 6 calls).
MAX_RAG_RETRIES = int(os.environ.get("MAX_RAG_RETRIES", "3"))

if GENERATOR_MODEL == VERIFIER_MODEL:
    print(
        f"⚠️  [LLM] GENERATOR_MODEL and VERIFIER_MODEL are both '{GENERATOR_MODEL}'. "
        "The verifier shares the generator's blind spots; set VERIFIER_MODEL to a "
        "different model in .env."
    )

# Matches the sentinel with optional surrounding markdown emphasis/punctuation
# the model might add despite being told to reply with exactly one word
# (e.g. "**NO_ANSWER**", "NO_ANSWER.").
_NO_ANSWER_RE = re.compile(
    rf"^[\s*_`]*{re.escape(NO_ANSWER_SENTINEL)}[\s*_`.!]*$", re.IGNORECASE
)


def _is_no_answer(draft: str) -> bool:
    return bool(_NO_ANSWER_RE.match((draft or "").strip()))


def _chat(
    user_prompt: str,
    system_prompt: str,
    temperature: float,
    model_name: str,
    max_tokens: int,
    providers: List[str],
    role: str,
) -> llm_client.ChatResult:
    return llm_client.chat(
        system_prompt,
        user_prompt,
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        provider_order=providers,
        role=role,
    )


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK_RE = re.compile(r"<think>.*\Z", re.DOTALL | re.IGNORECASE)
_NO_VERDICT = "The verifier returned no parseable verdict."


def _find_verdict_object(text: str) -> Optional[dict]:
    """
    The LAST JSON object in `text` that has a "verdict" key, or None.

    A reasoning model may write thinking text — possibly containing braces or
    a draft JSON — before its final answer, so grabbing "first { to last }"
    can produce something that isn't valid JSON, which would wrongly refuse a
    good answer. Instead, try to decode an object at every "{" and keep the
    last one that has a verdict; the final answer is the last thing written.
    """
    decoder = json.JSONDecoder()
    found = None
    pos = text.find("{")
    while pos != -1:
        try:
            obj, _end = decoder.raw_decode(text, pos)
            if isinstance(obj, dict) and "verdict" in obj:
                found = obj
        except ValueError:
            pass
        pos = text.find("{", pos + 1)
    return found


# Verdict kinds, kept apart because they mean different things in the thesis:
# UNSUPPORTED = the verifier found a problem (a "catch"); UNPARSEABLE = the
# verifier failed to answer in the required format (a verifier failure, not
# evidence that the draft was wrong). Both still fail closed.
VERDICT_SUPPORTED = "SUPPORTED"
VERDICT_UNSUPPORTED = "UNSUPPORTED"
VERDICT_SUPPORTED_WITH_ISSUES = "SUPPORTED_WITH_ISSUES"
VERDICT_UNPARSEABLE = "UNPARSEABLE"
# System v1.6 (VERIFIER_RELEVANCE=1): every claim is supported, but the draft
# does not answer the question that was asked. Also fails closed.
VERDICT_OFF_TOPIC = "OFF_TOPIC"


@dataclass
class Verdict:
    passed: bool
    kind: str                  # one of the VERDICT_* values above
    critique: str = ""         # joined issues ("" when passed)
    issues: List[str] = field(default_factory=list)
    off_topic: bool = False    # the verifier said the draft doesn't answer the question
    relevance_issue: str = ""


def _off_topic(data: dict) -> bool:
    value = data.get("answers_question", True)
    if isinstance(value, str):
        return value.strip().lower() in ("false", "no", "0")
    return value is False


def parse_verdict(raw: str, relevance: Optional[bool] = None) -> Verdict:
    """
    Parse the verifier's JSON reply.

    Fail-closed: anything other than a well-formed object whose "verdict" is
    exactly "SUPPORTED" (case-insensitive) with no issues is a FAIL — malformed
    JSON, a missing key, "UNSUPPORTED", or free text like "NOT VERIFIED". The
    old `"VERIFIED" in text.upper()` check passed all of the latter.
    Thinking text (<think>...</think>, or an unclosed <think> block) is
    ignored, and the last verdict object wins.
    """
    text = _THINK_RE.sub("", raw or "")
    # Some reasoning models emit only the closing tag; the answer is what follows it.
    text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]
    # An opening tag that never closes = the reply was cut off mid-thought: no verdict.
    text = _UNCLOSED_THINK_RE.sub("", text).strip()
    data = _find_verdict_object(text)
    if data is None:
        return Verdict(False, VERDICT_UNPARSEABLE, _NO_VERDICT)

    issues = data.get("issues") or []
    if isinstance(issues, str):
        issues = [issues]
    if not isinstance(issues, list):
        issues = [str(issues)]
    issues = [str(i) for i in issues]
    critique = "; ".join(issues) or "Some claims are not supported by the context."

    relevance = VERIFIER_RELEVANCE if relevance is None else relevance
    # Relevance (verifier v2) is a separate field: a missing "answers_question"
    # counts as "answers it", so only an explicit "false" rejects a draft on
    # relevance. Grounding stays fail-closed as before.
    off = relevance and _off_topic(data)
    why = str(data.get("relevance_issue") or "").strip() if off else ""

    verdict = str(data.get("verdict", "")).strip().upper()
    if verdict == "SUPPORTED":
        if not issues:
            if off:
                note = f"The draft does not answer the question asked{': ' + why if why else '.'}"
                return Verdict(False, VERDICT_OFF_TOPIC, note, [], True, why)
            return Verdict(True, VERDICT_SUPPORTED, "", [])
        return Verdict(False, VERDICT_SUPPORTED_WITH_ISSUES, critique, issues, off, why)
    if verdict == "UNSUPPORTED":
        return Verdict(False, VERDICT_UNSUPPORTED, critique, issues, off, why)
    return Verdict(False, VERDICT_UNPARSEABLE, f"Unknown verdict {verdict!r}. {critique}", issues, off, why)


def _parse_verdict(raw: str) -> Tuple[bool, str]:
    """Old (passed, critique) form, kept for any code that still calls it."""
    v = parse_verdict(raw)
    return v.passed, v.critique


def _verify(context_text: str, draft: str, question: str = "") -> Tuple[Verdict, llm_client.ChatResult]:
    """Audit `draft` against `context_text` with the dedicated auditor persona,
    running on VERIFIER_MODEL (a different model from the generator). With
    VERIFIER_RELEVANCE=1 the verifier also sees the question and says whether
    the draft answers it (prompts.VERIFY_PROMPT_V2)."""
    result = _chat(
        ACTIVE_VERIFY_PROMPT.format(context=context_text, draft=draft, question=question),
        system_prompt=VERIFIER_SYSTEM_PROMPT,
        temperature=VERIFIER_TEMPERATURE,
        model_name=VERIFIER_MODEL,
        max_tokens=VERIFIER_MAX_TOKENS,
        providers=VERIFIER_PROVIDERS,
        role="verifier",
    )
    return parse_verdict(result.text), result


# ── Structured result (what happened, for evaluation) ────────────────────
#
# Outcomes of run_rag():
#   verified             a draft passed verification on attempt n (returned)
#   generator_no_answer  the generator replied NO_ANSWER (canned refusal)
#   verification_failed  no attempt passed within MAX_RAG_RETRIES (canned refusal)
#   error                a model call failed (LLMError etc.); see `error`
#
# Attempt statuses:
#   supported     verifier approved the draft
#   rejected      verifier said UNSUPPORTED, or SUPPORTED but listed issues
#   unparseable   verifier reply had no usable verdict (format failure)
#   no_answer     generator replied NO_ANSWER (not verified — nothing to check)
#   non_answer    the draft opened by saying the excerpts don't answer it
#                 (app/answerability.py); treated like NO_ANSWER (v1.6)
#   grounding_failed  the deterministic check (app/grounding.py) found a
#                 figure/issuance/citation the cited excerpts don't support,
#                 and repair wasn't possible; regenerated, no verifier call (v1.6)
#   off_topic     grounded, but the verifier said it doesn't answer the question (v1.6)
#   repaired      the flagged sentences were removed and the rest passed a
#                 second verifier call (v1.6); the repaired text is returned
#   empty         generator reply was empty (not verified)
#   truncated     generator reply hit the output cap (not verified)
#   error         the call raised; `error` on the result says why

OUTCOME_VERIFIED = "verified"
OUTCOME_GENERATOR_NO_ANSWER = "generator_no_answer"
OUTCOME_VERIFICATION_FAILED = "verification_failed"
OUTCOME_ERROR = "error"


@dataclass
class Attempt:
    number: int                              # 1-based
    status: str                              # see "Attempt statuses" above
    audit_notice: str = ""                   # feedback given to the generator for THIS attempt
    draft: str = ""                          # full generator reply (unverified unless status == supported)
    generator: Optional[Dict[str, Any]] = None   # ChatResult.meta() of the generator call
    verdict: Optional[str] = None            # VERDICT_* kind, if the verifier ran
    issues: List[str] = field(default_factory=list)
    critique: str = ""
    verifier_raw: Optional[str] = None       # full verifier reply, if it ran
    verifier: Optional[Dict[str, Any]] = None    # ChatResult.meta() of the verifier call
    grounding: Optional[Dict[str, Any]] = None   # deterministic check report (v1.6)
    repair: Optional[Dict[str, Any]] = None      # claim repair: what was removed, the result, its verdict (v1.6)
    off_topic: bool = False


@dataclass
class RagResult:
    outcome: str                             # OUTCOME_* value
    answer: str                              # what the user sees (draft or canned refusal)
    language: str                            # language label used for the answer
    attempts: List[Attempt] = field(default_factory=list)
    max_attempts: int = MAX_RAG_RETRIES
    error: Optional[str] = None
    latency_ms: int = 0

    @property
    def verified(self) -> bool:
        return self.outcome == OUTCOME_VERIFIED

    @property
    def calls(self) -> List[Dict[str, Any]]:
        out = []
        for a in self.attempts:
            out += [m for m in (a.generator, a.verifier) if m]
            if a.repair and a.repair.get("verifier"):
                out.append(a.repair["verifier"])
        return out

    @property
    def cost(self) -> Optional[float]:
        """Sum of reported costs (None if the API reported none)."""
        costs = [c["cost"] for c in self.calls if isinstance(c.get("cost"), (int, float))]
        return round(sum(costs), 8) if costs else None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["n_attempts"] = len(self.attempts)
        d["n_calls"] = len(self.calls)
        d["cost"] = self.cost
        d["models"] = {"generator": GENERATOR_MODEL, "verifier": VERIFIER_MODEL}
        return d


def run_rag(
    query: str,
    history: List[ConvMessage],
    context_text: str,
    language: Optional[LanguageResult] = None,
    max_retries: int = MAX_RAG_RETRIES,
) -> RagResult:
    """
    Generate an answer grounded in the retrieved context, verify it with a
    DIFFERENT model, retry up to `max_retries` times, and report everything
    that happened as a RagResult (outcome, every draft, every verdict, and
    which host served each call). Never raises: a failing model call becomes
    outcome "error" with the attempts made so far.

    Design rules (unchanged from earlier sessions):

    1. Language: detected ONCE, before the loop; `query` is never mutated.
       The verifier's critique goes to the next attempt as a separate
       `audit_notice` section, never glued onto the question (an English
       critique next to the question pushed answers toward English).
    2. Grounding: the generator replies exactly NO_ANSWER when the excerpts
       don't support an answer; the user then gets a fixed, localized refusal,
       never the model's own refusal text.
    3. Fail-closed verification: every draft (including the last) is audited
       by the verifier (near-zero temperature, JSON verdict). If no attempt is
       SUPPORTED, the canned refusal is returned — never an unverified draft.
       Empty or cut-off drafts are never verified or returned.
    """
    started = time.monotonic()
    lang = language or detect_language(query)
    refusal = get_no_answer_message(lang.label)
    history_str = format_history(history)
    result = RagResult(OUTCOME_VERIFICATION_FAILED, refusal, lang.label, max_attempts=max_retries)
    audit_notice = ""

    def done(outcome: str, answer: str) -> RagResult:
        result.outcome, result.answer = outcome, answer
        result.latency_ms = int((time.monotonic() - started) * 1000)
        return result

    for number in range(1, max_retries + 1):
        attempt = Attempt(number=number, status="error", audit_notice=audit_notice.strip())
        result.attempts.append(attempt)
        try:
            prompt = RAG_PROMPT.format(
                user_language=lang.display_name,
                no_answer_sentinel=NO_ANSWER_SENTINEL,
                context=context_text,
                history=history_str,
                audit_notice=audit_notice,
                question=query,  # always the ORIGINAL question — never mutated
            )
            progress.emit("generate", "running", attempt=number, max_attempts=max_retries,
                          model=GENERATOR_MODEL, language=lang.display_name,
                          retry_reason=audit_notice.strip() or None)
            gen = _chat(
                prompt, SYSTEM_PROMPT, GENERATION_TEMPERATURE, GENERATOR_MODEL,
                GENERATOR_MAX_TOKENS, GENERATOR_PROVIDERS, "generator",
            )
            attempt.generator = gen.meta()
            attempt.draft = draft = gen.text
            progress.emit("generate", attempt=number, max_attempts=max_retries, model=GENERATOR_MODEL,
                          draft=draft, no_answer=_is_no_answer(draft), truncated=gen.truncated)

            if _is_no_answer(draft):
                attempt.status = "no_answer"
                print(f"🙅 [Generator] attempt {number}/{max_retries}: NO_ANSWER")
                return done(OUTCOME_GENERATOR_NO_ANSWER, refusal)

            # An empty draft has no claims, so a verifier could "support" it; a
            # draft cut off by the output cap is an incomplete answer. Neither
            # may be returned — count it as a failed attempt and try again.
            if not draft.strip() or gen.truncated:
                attempt.status = "empty" if not draft.strip() else "truncated"
                why = "empty" if attempt.status == "empty" else "cut off for being too long"
                print(f"⚠️  [LLM] generator reply was {why}; retrying.")
                audit_notice = (
                    f"\n[AUDIT NOTICE — the previous reply was {why}. Write a "
                    "complete, concise answer in the user's language, citing "
                    "excerpt numbers, using only facts from the excerpts]\n"
                )
                continue

            # v1.6: a refusal in the model's own words ("Hindi direktang
            # sinasagot ng mga excerpt...") is a no-answer, not an answer.
            if answerability.is_non_answer(draft):
                attempt.status = "non_answer"
                print(f"🙅 [Generator] attempt {number}/{max_retries}: draft opens by saying the excerpts don't answer")
                progress.emit("non_answer", attempt=number, first_sentence=answerability.first_sentence(draft))
                return done(OUTCOME_GENERATOR_NO_ANSWER, refusal)

            # v1.6: deterministic grounding check (no model call). Figures,
            # issuance/RA/form numbers and contacts must be in the excerpts the
            # sentence cites. Flagged sentences outside the opening answer are
            # removed (claim repair); otherwise the draft is regenerated.
            if grounding.GROUNDING_CHECK:
                report = grounding.check(draft, context_text, query)
                attempt.grounding = report.to_dict()
                fixed = None
                if not report.passed and grounding.CLAIM_REPAIR:
                    fixed = grounding.repair(draft, {f.index for f in report.flagged})
                    if fixed and not grounding.check(fixed, context_text, query).passed:
                        fixed = None
                progress.emit("grounding", attempt=number, passed=report.passed, issues=report.issues,
                              tokens=report.n_tokens, repaired=bool(fixed))
                if not report.passed:
                    if fixed:
                        attempt.repair = {"source": "grounding_check",
                                          "removed": [f.sentence for f in report.flagged], "draft": fixed}
                        print(f"🩹 [Grounding] attempt {number}: removed {len(report.flagged)} unsupported sentence(s)")
                        draft = fixed
                    else:
                        attempt.status = "grounding_failed"
                        print(f"❌ [Grounding] attempt {number}/{max_retries}: {'; '.join(report.issues)}")
                        audit_notice = (
                            "\n[AUDIT NOTICE — a prior attempt stated facts that the cited excerpts do not "
                            "contain. For each item below, either DELETE the claim or cite the excerpt that "
                            "states it; do not add new figures or calculate anything. Keep the same language "
                            f"and keep citing excerpt numbers]: {'; '.join(report.issues[:5])}\n"
                        )
                        continue

            progress.emit("verify", "running", attempt=number, max_attempts=max_retries, model=VERIFIER_MODEL)
            verdict, ver = _verify(context_text, draft, query)
            progress.emit("verify", attempt=number, max_attempts=max_retries, model=VERIFIER_MODEL,
                          passed=verdict.passed, verdict=verdict.kind, issues=verdict.issues,
                          off_topic=verdict.off_topic, relevance_issue=verdict.relevance_issue,
                          will_retry=(not verdict.passed and number < max_retries))
            attempt.verifier = ver.meta()
            attempt.verifier_raw = ver.text
            attempt.verdict, attempt.issues, attempt.critique = verdict.kind, verdict.issues, verdict.critique
            attempt.off_topic = verdict.off_topic

            if verdict.passed:
                attempt.status = "repaired" if attempt.repair else "supported"
                print(f"✅ [Verifier] attempt {number}/{max_retries}: {attempt.status}")
                return done(OUTCOME_VERIFIED, draft)

            # v1.6: claim repair after the verifier's rejection. Its issues are
            # mapped to sentences; if they are all outside the opening answer,
            # those sentences are removed and the rest is verified once more.
            if (grounding.CLAIM_REPAIR and not verdict.off_topic and attempt.repair is None
                    and verdict.kind in (VERDICT_UNSUPPORTED, VERDICT_SUPPORTED_WITH_ISSUES)):
                drop = grounding.map_issues_to_sentences(draft, verdict.issues)
                fixed = grounding.repair(draft, drop) if drop else None
                if fixed:
                    progress.emit("repair", "running", attempt=number, removed=len(drop))
                    v2, ver2 = _verify(context_text, fixed, query)
                    flat = [x for line in grounding.split_draft(draft) for x in line]
                    attempt.repair = {"source": "verifier", "removed": [flat[i] for i in sorted(drop)],
                                      "draft": fixed, "verdict": v2.kind, "issues": v2.issues,
                                      "off_topic": v2.off_topic, "verifier": ver2.meta(),
                                      "verifier_raw": ver2.text}
                    progress.emit("repair", attempt=number, removed=len(drop), passed=v2.passed, verdict=v2.kind)
                    if v2.passed:
                        attempt.status = "repaired"
                        print(f"🩹 [Verifier] attempt {number}/{max_retries}: removed {len(drop)} sentence(s); "
                              "the rest is supported")
                        return done(OUTCOME_VERIFIED, fixed)

            if verdict.kind == VERDICT_UNPARSEABLE:
                attempt.status = "unparseable"
            elif verdict.kind == VERDICT_OFF_TOPIC:
                attempt.status = "off_topic"
            else:
                attempt.status = "rejected"
            print(f"❌ [Verifier] attempt {number}/{max_retries}: {attempt.status} — {verdict.critique}")
            if verdict.kind == VERDICT_OFF_TOPIC:
                audit_notice = (
                    "\n[AUDIT NOTICE — a prior attempt did not answer the question that was asked"
                    f"{' (' + verdict.relevance_issue + ')' if verdict.relevance_issue else ''}. Answer the "
                    "question itself, from the excerpts. If no excerpt answers it, reply with exactly "
                    f"{NO_ANSWER_SENTINEL}]\n"
                )
            else:
                audit_notice = (
                    "\n[AUDIT NOTICE — a prior attempt failed grounding verification. "
                    "For each flagged claim below, either DELETE it or restate it using "
                    "only what the excerpts literally say; do not replace it with a new "
                    "claim. Keep the supported parts of the answer, keep the same "
                    f"language, and keep citing excerpt numbers]: {verdict.critique}\n"
                )
        except Exception as e:  # LLMError, network, anything: record, stop, fail closed
            attempt.status = "error"
            result.error = f"{type(e).__name__}: {e}"
            progress.emit("generate", "failed", attempt=number, error=result.error)
            print(f"💥 [LLM] attempt {number}/{max_retries} failed: {result.error}")
            return done(OUTCOME_ERROR, refusal)

    # Fail closed: every attempt failed, so no draft is trustworthy.
    return done(OUTCOME_VERIFICATION_FAILED, refusal)


def rag_answer(
    query: str,
    history: List[ConvMessage],
    context_text: str,
    language: Optional[LanguageResult] = None,
    max_retries: int = MAX_RAG_RETRIES,
) -> str:
    """Old interface: just the answer text (a verified draft or the canned
    refusal). Raises if a model call failed, as before. New code should call
    run_rag(), which says WHICH of those happened."""
    result = run_rag(query, history, context_text, language, max_retries)
    if result.outcome == OUTCOME_ERROR:
        raise llm_client.LLMError(result.error or "model call failed")
    return result.answer
