"""
Groq LLM calls: conversational answers and RAG-grounded answers.

Pulled out of api.py so the model name/temperature live in one place
and the two generation paths are easy to compare side by side.
"""

import re
from typing import List, Optional

from langchain_groq import ChatGroq

from .language import LanguageResult, detect_language
from .prompts import (
    CONV_PROMPT,
    NO_ANSWER_SENTINEL,
    RAG_PROMPT,
    SYSTEM_PROMPT,
    get_no_answer_message,
)
from .retrieval import format_history
from .schemas import ConvMessage

MODEL_NAME = "qwen/qwen3.8-27b"
TEMPERATURE = 0.6
MAX_RAG_RETRIES = 2

# Matches the sentinel with optional surrounding markdown emphasis/punctuation
# the model might add despite being told to reply with exactly one word
# (e.g. "**NO_ANSWER**", "NO_ANSWER.").
_NO_ANSWER_RE = re.compile(
    rf"^[\s*_`]*{re.escape(NO_ANSWER_SENTINEL)}[\s*_`.!]*$", re.IGNORECASE
)


def _is_no_answer(draft: str) -> bool:
    return bool(_NO_ANSWER_RE.match((draft or "").strip()))


def _chat(user_prompt: str) -> str:
    model = ChatGroq(model=MODEL_NAME, temperature=TEMPERATURE)
    response = model.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ])
    return response.content


def conversational_answer(
    query: str,
    history: List[ConvMessage],
    language: Optional[LanguageResult] = None,
) -> str:
    """Generate a conversational response without RAG context."""
    lang = language or detect_language(query)
    history_str = format_history(history)
    prompt = CONV_PROMPT.format(
        user_language=lang.display_name, history=history_str, question=query
    )
    return _chat(prompt)


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
    """
    lang = language or detect_language(query)
    history_str = format_history(history)
    audit_notice = ""

    draft = ""
    for attempt in range(max_retries):
        prompt = RAG_PROMPT.format(
            user_language=lang.display_name,
            no_answer_sentinel=NO_ANSWER_SENTINEL,
            context=context_text,
            history=history_str,
            audit_notice=audit_notice,
            question=query,  # always the ORIGINAL question — never mutated
        )
        draft = _chat(prompt)

        if _is_no_answer(draft):
            return get_no_answer_message(lang.label)

        if attempt == max_retries - 1:
            return draft

        verification_prompt = (
            f"Context: {context_text}\n\n"
            f"Draft Answer: {draft}\n\n"
            "Analyze the Draft Answer. Does it hallucinate any details, numbers, or rules not explicitly found in the Context? "
            "If it is 100% grounded in the Context, reply exactly with 'VERIFIED'. "
            "If it contains hallucinations or fabricated information, briefly point out the specific error."
        )
        verification = _chat(verification_prompt)

        if "VERIFIED" in verification.upper():
            return draft

        # Feed the critique back as its OWN labeled prompt section — never
        # glued onto `query` — so the question text (and the language
        # signal derived from it) is identical on every attempt.
        audit_notice = (
            "\n[AUDIT NOTICE — a prior attempt failed grounding verification "
            "with this critique; fix the issue below without changing your "
            f"answer's language or inventing new facts]: {verification}\n"
        )

    return draft