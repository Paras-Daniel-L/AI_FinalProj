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
from typing import List, Optional, Tuple

from dotenv import load_dotenv

from . import llm_client
from .language import LanguageResult, detect_language
from .prompts import (
    NO_ANSWER_SENTINEL,
    RAG_PROMPT,
    SYSTEM_PROMPT,
    VERIFIER_SYSTEM_PROMPT,
    VERIFY_PROMPT,
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
GENERATION_TEMPERATURE = 0.1
VERIFIER_TEMPERATURE = 0.0
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


def _parse_verdict(raw: str) -> Tuple[bool, str]:
    """
    Parse the verifier's JSON reply into (passed, critique).

    Fail-closed: anything other than a well-formed object whose "verdict" is
    exactly "SUPPORTED" (case-insensitive) is a FAIL — malformed JSON, a
    missing key, "UNSUPPORTED", or free text like "NOT VERIFIED". The old
    `"VERIFIED" in text.upper()` check passed all of the latter.
    A "SUPPORTED" verdict that still lists issues is also a fail.
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
        return False, _NO_VERDICT

    issues = data.get("issues") or []
    if isinstance(issues, str):
        issues = [issues]
    critique = "; ".join(str(i) for i in issues) or "Some claims are not supported by the context."

    verdict = str(data.get("verdict", "")).strip().upper()
    if verdict == "SUPPORTED" and not issues:
        return True, ""
    return False, critique


def _verify(context_text: str, draft: str) -> Tuple[bool, str]:
    """Audit `draft` against `context_text` with the dedicated auditor persona,
    running on VERIFIER_MODEL (a different model from the generator)."""
    result = _chat(
        VERIFY_PROMPT.format(context=context_text, draft=draft),
        system_prompt=VERIFIER_SYSTEM_PROMPT,
        temperature=VERIFIER_TEMPERATURE,
        model_name=VERIFIER_MODEL,
        max_tokens=VERIFIER_MAX_TOKENS,
        providers=VERIFIER_PROVIDERS,
        role="verifier",
    )
    return _parse_verdict(result.text)


def rag_answer(
    query: str,
    history: List[ConvMessage],
    context_text: str,
    language: Optional[LanguageResult] = None,
    max_retries: int = MAX_RAG_RETRIES,
) -> str:
    """
    Generate an answer grounded in retrieved document context, with an
    automated verification/retry loop and a language-anchored, grounding-
    enforced prompt.

    Two fixes versus the previous version:

    1. Language: detected ONCE from the ORIGINAL `query`, before the retry
       loop starts, and `query` itself is never mutated. The previous
       implementation appended the verification critique directly onto
       `query` on retry (`current_query = f"{query}\\n\\nNote: ..."`), which
       silently pushed the model toward English regardless of what
       language the user actually asked in, since the critique text is
       always English and sits as the most recent text before the answer.
       Here the critique is passed as a separate `audit_notice` prompt
       section instead — the question text, and its language, stay intact
       across every attempt.

    2. Grounding: the model is instructed (via RAG_PROMPT) to reply with
       exactly NO_ANSWER when the retrieved context doesn't support an
       answer. This function detects that sentinel and substitutes a
       fixed, localized refusal — the model's raw output on the no-answer
       path is never returned to the caller, so wording is deterministic
       and no unrelated retrieved content can leak through.

    3. Verification is fail-closed: every draft (including the last) is
       audited by a separate, near-zero-temperature auditor call returning a
       JSON verdict, run on a DIFFERENT model (VERIFIER_MODEL) than the
       generator (GENERATOR_MODEL). If no attempt is verified as SUPPORTED
       (up to MAX_RAG_RETRIES attempts), the canned refusal is returned —
       never an unverified draft.
    """
    lang = language or detect_language(query)
    history_str = format_history(history)
    audit_notice = ""

    for _attempt in range(max_retries):
        prompt = RAG_PROMPT.format(
            user_language=lang.display_name,
            no_answer_sentinel=NO_ANSWER_SENTINEL,
            context=context_text,
            history=history_str,
            audit_notice=audit_notice,
            question=query,  # always the ORIGINAL question — never mutated
        )
        result = _chat(
            prompt, SYSTEM_PROMPT, GENERATION_TEMPERATURE, GENERATOR_MODEL,
            GENERATOR_MAX_TOKENS, GENERATOR_PROVIDERS, "generator",
        )
        draft = result.text

        if _is_no_answer(draft):
            return get_no_answer_message(lang.label)

        # An empty draft has no claims, so a verifier could "support" it; a
        # draft cut off by the output cap is an incomplete answer. Neither may
        # be returned — count it as a failed attempt and try again.
        if not draft.strip() or result.truncated:
            why = "empty" if not draft.strip() else "cut off for being too long"
            print(f"⚠️  [LLM] generator reply was {why}; retrying.")
            audit_notice = (
                f"\n[AUDIT NOTICE — the previous reply was {why}. Write a "
                "complete, concise answer in the user's language, citing "
                "excerpt numbers, using only facts from the excerpts]\n"
            )
            continue

        # EVERY draft is verified before it can be returned — including the
        # last attempt (the old code returned the final draft unverified).
        passed, critique = _verify(context_text, draft)
        if passed:
            print(f"✅ [Verifier] attempt {_attempt + 1}/{max_retries}: supported")
            return draft
        print(f"❌ [Verifier] attempt {_attempt + 1}/{max_retries}: rejected — {critique}")

        # Feed the critique back as its OWN labeled prompt section — never
        # glued onto `query` — so the question text (and the language
        # signal derived from it) is identical on every attempt.
        audit_notice = (
            "\n[AUDIT NOTICE — a prior attempt failed grounding verification "
            "with this critique; fix the issue below without changing your "
            f"answer's language or inventing new facts]: {critique}\n"
        )

    # Fail closed: every attempt failed verification, so no draft is
    # trustworthy. Return the canned refusal (api.py maps it to mode
    # "no_answer") rather than an unverified, possibly hallucinated answer.
    return get_no_answer_message(lang.label)