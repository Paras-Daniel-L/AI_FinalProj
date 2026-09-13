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
    mode: str          # "rag" | "conversational"


class StatusResponse(BaseModel):
    document_count: int
    chroma_path: str
    data_path: str
