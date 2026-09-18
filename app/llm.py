"""
Groq LLM calls: conversational answers and RAG-grounded answers.

Pulled out of api.py so the model name/temperature live in one place
and the two generation paths are easy to compare side by side.
"""

from typing import List

from langchain_groq import ChatGroq

from .prompts import CONV_PROMPT, RAG_PROMPT, SYSTEM_PROMPT
from .retrieval import format_history
from .schemas import ConvMessage

MODEL_NAME = "openai/gpt-oss-120b"
TEMPERATURE = 0.6


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


def rag_answer(query: str, history: List[ConvMessage], context_text: str, max_retries: int = 2) -> str:
    """Generate an answer grounded in retrieved document context, with an automated verification loop."""
    history_str = format_history(history)
    current_query = query
    
    for attempt in range(max_retries):
        # 1. Draft the grounded answer
        prompt = RAG_PROMPT.format(context=context_text, history=history_str, question=current_query)
        draft = _chat(prompt)
        
        # If this is the last attempt, return the draft directly
        if attempt == max_retries - 1:
            return draft
            
        # 2. Verification Step
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
            
        # 3. Apply correction if hallucination was detected
        current_query = (
            f"{query}\n\n"
            f"Note: Your previous attempt failed validation with this critique: {verification}\n"
            "Please rewrite your answer to fix these errors. Rely STRICTLY on the provided context and admit if the context is insufficient."
        )

    return draft