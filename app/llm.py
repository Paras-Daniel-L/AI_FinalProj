"""
Groq LLM calls: RAG-grounded answers.

Pulled out of api.py so the model name/temperature live in one place.

Sagot AI has no open-domain/conversational generation path anymore — see
prompts.SYSTEM_PROMPT and app/greetings.py. The only LLM call left is the
grounded, retrieval-anchored one below.
"""

import json
import re
from typing import List, Optional, Tuple

from langchain_groq import ChatGroq

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

MODEL_NAME = "qwen/qwen3.8-27b"
# Both calls are grounded/audit tasks, so they run (near-)deterministic.
# Kept separate so they can be tuned independently for the thesis ablations.
GENERATION_TEMPERATURE = 0.1
VERIFIER_TEMPERATURE = 0.0
MAX_RAG_RETRIES = 2

# Matches the sentinel with optional surrounding markdown emphasis/punctuation
# the model might add despite being told to reply with exactly one word
# (e.g. "**NO_ANSWER**", "NO_ANSWER.").
_NO_ANSWER_RE = re.compile(
    rf"^[\s*_`]*{re.escape(NO_ANSWER_SENTINEL)}[\s*_`.!]*$", re.IGNORECASE
)


def _is_no_answer(draft: str) -> bool:
    return bool(_NO_ANSWER_RE.match((draft or "").strip()))


def _chat(user_prompt: str, system_prompt: str, temperature: float) -> str:
    model = ChatGroq(model=MODEL_NAME, temperature=temperature)
    response = model.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ])
    return response.content


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_verdict(raw: str) -> Tuple[bool, str]:
    """
    Parse the verifier's JSON reply into (passed, critique).

    Fail-closed: anything other than a well-formed object whose "verdict" is
    exactly "SUPPORTED" (case-insensitive) is a FAIL — malformed JSON, a
    missing key, "UNSUPPORTED", or free text like "NOT VERIFIED". The old
    `"VERIFIED" in text.upper()` check passed all of the latter.
    A "SUPPORTED" verdict that still lists issues is also a fail.
    """
    text = _THINK_RE.sub("", raw or "").strip()
    match = _JSON_OBJ_RE.search(text)
    if not match:
        return False, "The verifier returned no parseable verdict."
    try:
        data = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return False, "The verifier returned no parseable verdict."
    if not isinstance(data, dict):
        return False, "The verifier returned no parseable verdict."

    issues = data.get("issues") or []
    if isinstance(issues, str):
        issues = [issues]
    critique = "; ".join(str(i) for i in issues) or "Some claims are not supported by the context."

    verdict = str(data.get("verdict", "")).strip().upper()
    if verdict == "SUPPORTED" and not issues:
        return True, ""
    return False, critique


def _verify(context_text: str, draft: str) -> Tuple[bool, str]:
    """Audit `draft` against `context_text` with the dedicated auditor persona."""
    raw = _chat(
        VERIFY_PROMPT.format(context=context_text, draft=draft),
        system_prompt=VERIFIER_SYSTEM_PROMPT,
        temperature=VERIFIER_TEMPERATURE,
    )
    return _parse_verdict(raw)


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
       JSON verdict. If no attempt is verified as SUPPORTED, the canned
       refusal is returned — never an unverified draft.
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
        draft = _chat(prompt, SYSTEM_PROMPT, GENERATION_TEMPERATURE)

        if _is_no_answer(draft):
            return get_no_answer_message(lang.label)

        # EVERY draft is verified before it can be returned — including the
        # last attempt (the old code returned the final draft unverified).
        passed, critique = _verify(context_text, draft)
        if passed:
            return draft

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