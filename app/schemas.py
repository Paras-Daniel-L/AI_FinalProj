"""
Request/response models for the API.

Pulled out of api.py so routes and data shapes are easy to tell apart at a glance.
"""

from typing import List

from pydantic import BaseModel


class ConvMessage(BaseModel):
    role: str          # "user" or "assistant"
    content: str


class QueryRequest(BaseModel):
    query: str
    history: List[ConvMessage] = []   # conversation turns before this message


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    classification: str
    predicted_class: int
    mode: str                # "rag" | "rag_fallback" | "conversational"
    verified: bool = True    # False only when a RAG answer failed groundedness verification
    retries: int = 0         # how many regenerate-and-reverify cycles the RAG path ran
    degraded: bool = False   # True if the safe fallback response was returned instead of a draft
    language: str = "english"  # "taglish" | "english" - what detect_taglish() found in the raw query


class StatusResponse(BaseModel):
    document_count: int
    chroma_path: str
    data_path: str