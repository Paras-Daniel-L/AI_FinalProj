"""
Groq LLM calls: conversational answers and verified RAG-grounded answers.

Pulled out of api.py so the model name/temperature live in one place
and the two generation paths are easy to compare side by side.
"""

from typing import List, NamedTuple, Tuple

from langchain_groq import ChatGroq

from .prompts import (
    CONV_PROMPT,
    RAG_PROMPT,
    SAFE_FALLBACK_RESPONSE,
    SYSTEM_PROMPT,
    VERIFICATION_PROMPT,
)
from .retrieval import format_history
from .schemas import ConvMessage

MODEL_NAME = "openai/gpt-oss-120b"
TEMPERATURE = 0.0  # Enforces highly deterministic, direct generation

# Matches the "Retry Count <= 3?" node in the automated fallback loop.
MAX_VERIFICATION_RETRIES = 3


class RagResult(NamedTuple):
    answer: str
    verified: bool       # True only if a verification pass actually passed
    retries_used: int    # how many regenerate-and-reverify cycles ran
    degraded: bool       # True if we returned SAFE_FALLBACK_RESPONSE instead of a real draft


def _chat(user_prompt: str) -> str:
    model = ChatGroq(model=MODEL_NAME, temperature=TEMPERATURE)
    response = model.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ])
    return response.content


def conversational_answer(query: str, history: List[ConvMessage]) -> str:
    """Generate a conversational response without RAG context."""
    history_str = format_history(history)
    prompt = CONV_PROMPT.format(history=history_str, question=query)
    return _chat(prompt)


def _verify_answer(query: str, context_text: str, draft_answer: str) -> Tuple[bool, str]:
    """
    Ask the LLM to check the draft against the retrieved context.
    Returns (is_supported, raw_verdict_text).
    """
    prompt = VERIFICATION_PROMPT.format(
        context=context_text, question=query, draft_answer=draft_answer
    )
    verdict_text = _chat(prompt)

    verdict_line = ""
    for line in verdict_text.splitlines():
        if line.strip().upper().startswith("VERDICT:"):
            verdict_line = line.strip().upper()
            break

    if "UNSUPPORTED" in verdict_line:
        is_supported = False
    elif "SUPPORTED" in verdict_line:
        is_supported = True
    else:
        is_supported = False

    return is_supported, verdict_text


def rag_answer(
    query: str,
    history: List[ConvMessage],
    context_text: str,
    max_retries: int = MAX_VERIFICATION_RETRIES,
) -> RagResult:
    """
    Generate an answer grounded in retrieved context, verify it against that
    context, and retry with the critique fed back into the prompt if it
    isn't grounded. 
    """
    history_str = format_history(history)
    current_query = query

    # Strict negative constraint dynamically appended to force direct answers
    strict_constraint = (
        "\n\nAnswer the question directly using ONLY the provided context. "
        "Do not include conversational filler, outside knowledge, or training memory. "
        "If the context does not contain the exact answer, state that the information is missing."
    )

    for attempt in range(1, max_retries + 1):
        # Inject the constraint into the base RAG prompt
        prompt = RAG_PROMPT.format(context=context_text, history=history_str, question=current_query)
        prompt += strict_constraint
        
        draft = _chat(prompt)

        is_supported, verdict_text = _verify_answer(query, context_text, draft)

        if is_supported:
            return RagResult(
                answer=draft,
                verified=True,
                retries_used=attempt - 1,
                degraded=False,
            )

        print(f"⚠️ [Verification] Attempt {attempt}/{max_retries} failed: {verdict_text.strip()}")

        current_query = (
            f"{query}\n\n"
            f"Note: your previous answer failed a groundedness check for this reason: "
            f"{verdict_text.strip()}\n"
            "Rewrite your answer using ONLY the retrieved documents above. If the documents "
            "don't contain enough information, say so plainly instead of filling the gap. "
            "Do not include conversational filler."
        )

    print(f"🛑 [Fallback] {max_retries} verification attempts failed. Returning safe fallback response.")
    return RagResult(
        answer=SAFE_FALLBACK_RESPONSE,
        verified=False,
        retries_used=max_retries,
        degraded=True,
    )