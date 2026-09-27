"""
Robust JSON extraction from an LLM reply — shared by every judge prompt in
rag_eval/. Mirrors the pattern app/llm.py already uses for the production
verifier's verdict, for the same reason: a judge model may emit thinking
text, markdown fences, or stray braces before its real answer, and grabbing
"first { to last }" over the whole reply can decode something that parses as
JSON but isn't the model's actual answer.

This module trusts nothing an LLM says about its own arithmetic: callers
extract a LIST (of claims, sentences, questions — whatever the judge was
asked to itemize) and this module never sums, divides, or averages anything.
Every ratio reported anywhere in rag_eval/ is computed in Python from that
list's length, not asked of the model.
"""

import json
import re
from typing import Any, List, Optional

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK_RE = re.compile(r"<think>.*\Z", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    text = _THINK_RE.sub("", text or "")
    text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]
    return _UNCLOSED_THINK_RE.sub("", text).strip()


def extract_json_value(raw: str, key: str) -> Optional[Any]:
    """
    The value of `key` from the LAST well-formed JSON object in `raw` that
    contains it (a code-fenced block is tried first, since a model that
    fences its JSON usually means exactly that block). Returns None if
    nothing decodes — callers must treat that as a judge failure, not a
    zero/empty result, since those mean different things for every metric
    here (a failed judge call is missing data; an empty list is a real
    "found nothing" answer).
    """
    text = _strip_thinking(raw)
    candidates: List[str] = [m.group(1) for m in _FENCE_RE.finditer(text)] or [text]

    found = None
    for candidate in candidates:
        decoder = json.JSONDecoder()
        pos = candidate.find("{")
        while pos != -1:
            try:
                obj, _end = decoder.raw_decode(candidate, pos)
                if isinstance(obj, dict) and key in obj:
                    found = obj[key]
            except ValueError:
                pass
            pos = candidate.find("{", pos + 1)
    return found


def as_list(value: Any) -> Optional[List[Any]]:
    """`value` as a list, or None if it plainly isn't one (a judge failure)."""
    if isinstance(value, list):
        return value
    return None
