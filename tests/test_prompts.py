"""
Tests for the generation-prompt versions (app/prompts.py).

Version 3 (default, system v1.3) is concise but combines every relevant
excerpt; version 2 is the v1.2 concise prompt; version 1 is the prompt the
thesis evaluation in rag_eval/results_v2 was run with. Both must keep what the
architecture depends on, and neither may contain evaluation questions.
"""

import importlib
import json
import os
from pathlib import Path

import pytest

from app import prompts

FILL = dict(user_language="Filipino", no_answer_sentinel="NO_ANSWER", context="[1] RMC No. 1-2026, p.1 (2026)\ntext",
            history="(No previous conversation)", audit_notice="\n[AUDIT NOTICE — fix this]\n", question="Q?")


@pytest.mark.parametrize("template", [prompts.RAG_PROMPT_V1, prompts.RAG_PROMPT_V2, prompts.RAG_PROMPT_V3])
def test_both_versions_keep_what_the_pipeline_needs(template):
    out = template.format(**FILL)
    assert "NO_ANSWER" in out                    # fail-closed path (generator_no_answer)
    assert "Filipino" in out                     # language trigger
    assert "[1] RMC No. 1-2026" in out           # numbered excerpts for citations / verifier
    assert "AUDIT NOTICE" in out                 # verifier feedback on retries
    assert "USER QUESTION: Q?" in out


@pytest.mark.parametrize("text", [prompts.RAG_PROMPT_V1, prompts.RAG_PROMPT_V2, prompts.RAG_PROMPT_V3,
                                  prompts.SYSTEM_PROMPT_V1, prompts.SYSTEM_PROMPT_V2])
def test_no_evaluation_question_leaks_into_any_prompt(text):
    ted = json.loads((Path(__file__).parent.parent / "rag_eval" / "ted_dataset.json").read_text(encoding="utf-8"))
    for row in ted:
        assert row["query"][:40] not in text, f"T-TED question {row['qid']} leaked into a prompt"


def test_v2_asks_for_concise_cited_answers():
    assert "concise" in prompts.RAG_PROMPT_V2 and "60 words" in prompts.RAG_PROMPT_V2
    assert "cite" in prompts.RAG_PROMPT_V2
    assert len(prompts.RAG_PROMPT_V2) < len(prompts.RAG_PROMPT_V1) / 2


def test_v3_is_the_default_and_combines_excerpts():
    importlib.reload(prompts)
    assert prompts.RAG_PROMPT_VERSION == "3" and prompts.RAG_PROMPT is prompts.RAG_PROMPT_V3
    assert "combine" in prompts.RAG_PROMPT_V3 and "80 words" in prompts.RAG_PROMPT_V3


def test_version_switch(monkeypatch):
    try:
        monkeypatch.setenv("RAG_PROMPT_VERSION", "1")
        p = importlib.reload(prompts)
        assert p.RAG_PROMPT is p.RAG_PROMPT_V1 and p.SYSTEM_PROMPT is p.SYSTEM_PROMPT_V1
        monkeypatch.setenv("RAG_PROMPT_VERSION", "2")
        p = importlib.reload(prompts)
        assert p.RAG_PROMPT is p.RAG_PROMPT_V2 and p.SYSTEM_PROMPT is p.SYSTEM_PROMPT_V2
        monkeypatch.setenv("RAG_PROMPT_VERSION", "3")
        p = importlib.reload(prompts)
        assert p.RAG_PROMPT is p.RAG_PROMPT_V3
        monkeypatch.setenv("RAG_PROMPT_VERSION", "banana")    # bad value: falls back to the default
        assert importlib.reload(prompts).RAG_PROMPT_VERSION == "3"
    finally:
        monkeypatch.delenv("RAG_PROMPT_VERSION", raising=False)
        importlib.reload(prompts)
