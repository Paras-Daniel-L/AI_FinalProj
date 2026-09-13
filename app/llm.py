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


def rag_answer(query: str, history: List[ConvMessage], context_text: str) -> str:
    """Generate an answer grounded in retrieved document context."""
    history_str = format_history(history)
    prompt = RAG_PROMPT.format(context=context_text, history=history_str, question=query)
    return _chat(prompt)
