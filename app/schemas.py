"""
Request/response models for the API.

Pulled out of api.py so routes and data shapes are easy to tell apart at a glance.
"""

import re
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# The max_length values below are only HARD CEILINGS that reject absurd payloads
# before any work is done (HTTP 422). The real, friendlier limits (a question
# that fits the 5-excerpt budget) live in app/sanitize.py.
class ConvMessage(BaseModel):
    role: str = Field(max_length=20)          # "user" or "assistant"
    content: str = Field(max_length=20000)


# Inputs a computation may hold. Anything else the client sends is dropped.
COMPUTATION_VALUE_KEYS = {
    "tax_year", "taxable_income", "gross_sales_receipts", "non_operating_income",
    "tax_withheld", "regime", "kind",
    "gross_pay", "pay_period", "benefits", "contributions",       # employee (gross pay) mode
}
_AMOUNT_VALUE_RE = re.compile(r"^\d{1,13}(?:\.\d{1,2})?$")
_ENUM_VALUES = {
    "regime": {"graduated", "8_percent"},
    "kind": {"employee", "self_employed", "mixed"},
    "pay_period": {"monthly", "semi_monthly", "weekly", "annual"},
}


class ComputationState(BaseModel):
    """
    The values collected so far in a multi-turn tax computation. The server
    returns it with every computation reply and the browser sends it back with
    the next message, so the server keeps no per-user session. It is
    re-validated here on every request: unknown keys and malformed values are
    dropped (the user is then simply asked again). It never holds rates; those
    only come from tax_rules/.
    """
    tax_type: Optional[str] = Field(default=None, max_length=64)
    values: Dict[str, str] = Field(default_factory=dict)
    status: str = Field(default="collecting", max_length=20)        # collecting | completed
    awaiting: List[str] = Field(default_factory=list, max_length=8)
    language: Optional[str] = Field(default=None, max_length=20)
    notes: List[str] = Field(default_factory=list, max_length=8)

    @field_validator("values")
    @classmethod
    def _clean_values(cls, v: Dict[str, str]) -> Dict[str, str]:
        clean: Dict[str, str] = {}
        for key, value in (v or {}).items():
            value = str(value).strip()
            if key not in COMPUTATION_VALUE_KEYS or len(value) > 20:
                continue
            if key == "tax_year":
                if re.fullmatch(r"(?:19|20)\d{2}", value):
                    clean[key] = value
            elif key in _ENUM_VALUES:
                if value in _ENUM_VALUES[key]:
                    clean[key] = value
            elif _AMOUNT_VALUE_RE.match(value):
                clean[key] = value
        return clean

    @field_validator("awaiting", "notes")
    @classmethod
    def _short_strings(cls, v: List[str]) -> List[str]:
        return [s for s in (v or []) if isinstance(s, str) and len(s) <= 40]


class QueryRequest(BaseModel):
    query: str = Field(max_length=2000)
    history: List[ConvMessage] = Field(default_factory=list, max_length=100)   # turns before this message
    bypass_cache: bool = False        # skip the answer cache (evaluation runs, debugging)
    computation_state: Optional[ComputationState] = None   # echoed back from the previous computation reply


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    classification: str   # source categories of the retrieved chunks, e.g. "BIR Tax Query (Source Year: 2024)"
    # "rag" | "guidance" (no evidence: guided follow-up) | "no_answer" (RECOVERY_ENABLED=0) | "clarify" | "chat" |
    # "computation" | "computation_input" | "greeting" | "rejected"
    mode: str
    language: Optional[str] = None   # "english" | "filipino" | "taglish"
    cached: bool = False             # True when served from the answer cache (already verified)
    # What happened (see api.run_query): verified | generator_no_answer |
    # verification_failed | no_retrieval | no_index | cache_hit | greeting |
    # rejected_input. Drafts and critiques are NOT sent to the browser (they
    # are unverified); they are in the local trace log under request_id.
    outcome: Optional[str] = None
    attempts: int = 0                # generate->verify attempts used (0 = no model call)
    request_id: Optional[str] = None # matches the line in logs/queries.jsonl
    intent: Optional[str] = None     # CHAT | TAX_INFORMATION | TAX_COMPUTATION | OUT_OF_SCOPE
    # The computation in progress (or just completed). The client sends it back
    # as `computation_state`; None means there is nothing to continue.
    computation: Optional[ComputationState] = None


class StatusResponse(BaseModel):
    document_count: int
    chroma_path: str
    data_path: str
