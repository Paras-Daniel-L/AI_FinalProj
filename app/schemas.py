"""
Request/response models for the API.

Pulled out of api.py so routes and data shapes are easy to tell apart at a glance.
"""

from typing import List, Optional

from pydantic import BaseModel


class ConvMessage(BaseModel):
    role: str          # "user" or "assistant"
    content: str


class QueryRequest(BaseModel):
    query: str
    history: List[ConvMessage] = []   # conversation turns before this message
    bypass_cache: bool = False        # skip the answer cache (evaluation runs, debugging)


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    classification: str   # source categories of the retrieved chunks, e.g. "BIR Tax Query (Source Year: 2024)"
    mode: str          # "rag" | "no_answer" | "greeting"
    language: Optional[str] = None   # "english" | "filipino" | "taglish"
    cached: bool = False             # True when served from the answer cache (already verified)


class StatusResponse(BaseModel):
    document_count: int
    chroma_path: str
    data_path: str