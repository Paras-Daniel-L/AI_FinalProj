"""
Request/response models for the API.

Pulled out of api.py so routes and data shapes are easy to tell apart at a glance.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


# The max_length values below are only HARD CEILINGS that reject absurd payloads
# before any work is done (HTTP 422). The real, friendlier limits (a question
# that fits the 5-excerpt budget) live in app/sanitize.py.
class ConvMessage(BaseModel):
    role: str = Field(max_length=20)          # "user" or "assistant"
    content: str = Field(max_length=20000)


class QueryRequest(BaseModel):
    query: str = Field(max_length=2000)
    history: List[ConvMessage] = Field(default_factory=list, max_length=100)   # turns before this message
    bypass_cache: bool = False        # skip the answer cache (evaluation runs, debugging)


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    classification: str   # source categories of the retrieved chunks, e.g. "BIR Tax Query (Source Year: 2024)"
    mode: str          # "rag" | "no_answer" | "greeting" | "rejected"
    language: Optional[str] = None   # "english" | "filipino" | "taglish"
    cached: bool = False             # True when served from the answer cache (already verified)
    # What happened (see api.run_query): verified | generator_no_answer |
    # verification_failed | no_retrieval | no_index | cache_hit | greeting |
    # rejected_input. Drafts and critiques are NOT sent to the browser (they
    # are unverified); they are in the local trace log under request_id.
    outcome: Optional[str] = None
    attempts: int = 0                # generate->verify attempts used (0 = no model call)
    request_id: Optional[str] = None # matches the line in logs/queries.jsonl


class StatusResponse(BaseModel):
    document_count: int
    chroma_path: str
    data_path: str
