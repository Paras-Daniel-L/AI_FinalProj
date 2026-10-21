"""
The RAGAS judge — an LLM used ONLY to grade Sagot AI's (and REVIE's) answers,
never to answer questions itself. Every prompt asks for a list (claims,
sentences, reverse-questions); the RATIO each metric reports is always
computed afterward in Python (see ragas_metrics.py), never by the judge.

Why a separate judge model, and why this file refuses to run without one
being set explicitly: Sagot AI's generator (DeepSeek) and verifier (Qwen)
already judge each other inside the production pipeline. Using either one
again here would mean the system grading its own homework — a validity
problem, since a model's own blind spots don't show up to itself. JUDGE_MODEL
must be a third model, uninvolved anywhere in app/llm.py, and this module
enforces that at import/call time rather than silently allowing it.

Settings (env vars):
    JUDGE_MODEL          REQUIRED. An OpenRouter model id, different from
                          both GENERATOR_MODEL and VERIFIER_MODEL.
    JUDGE_MAX_TOKENS      default 1200 (claim/sentence lists can be long for
                          a padded answer or a long excerpt)
    JUDGE_TEMPERATURE     default 0.0 (grading should be as deterministic as
                          the model allows, same convention as app/llm.py's
                          verifier)
    JUDGE_ANSWER_QUESTIONS  how many reverse-engineered questions to generate
                          for Answer Relevance (SOP's formula sums over N of
                          them). Default 3: RAGAS's own reference
                          implementation typically uses 3-5; more is more
                          stable but costs more judge calls per answer.
    JUDGE_DISABLE_THINKING  default 0. The production generator/verifier are
                          reasoning models, so app/llm_client.py sends
                          "reasoning: enabled=false" + require_parameters on
                          every call. A NON-reasoning judge (e.g.
                          openai/gpt-4o-mini) has no such parameter, and
                          OpenRouter then refuses to route the call (HTTP
                          404 "No endpoints found that can handle the
                          requested parameters"). Leave 0 for a
                          non-reasoning judge; set 1 only if you pick a
                          reasoning model as the judge (llm_client still
                          prints a loud warning if reasoning tokens are
                          billed, so a wrong setting is never silent).
"""

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

from dotenv import load_dotenv

from app import llm_client
from app.textproc import clean_pdf_text, split_sentences
from .json_util import as_list, extract_json_value

load_dotenv()

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "").strip()
# Was 1200: too small when the Context Relevance judge had to quote every
# sentence of a 4,000-character context back verbatim — 4 of 100 Sagot AI
# units (A8-EN, B3, B42, B53) lost their score to a cut-off JSON reply.
# Context Relevance now returns sentence NUMBERS (short), but the other
# judges still return lists, so the cap is raised too.
JUDGE_MAX_TOKENS = int(os.environ.get("JUDGE_MAX_TOKENS", "2500"))
JUDGE_TEMPERATURE = float(os.environ.get("JUDGE_TEMPERATURE", "0.0"))
JUDGE_ANSWER_QUESTIONS = int(os.environ.get("JUDGE_ANSWER_QUESTIONS", "3"))
JUDGE_DISABLE_THINKING = os.environ.get("JUDGE_DISABLE_THINKING", "0").strip() in ("1", "true", "True", "yes")


class JudgeConfigError(RuntimeError):
    """JUDGE_MODEL is missing or collides with a model already in the pipeline under test."""


def require_independent_judge() -> str:
    """
    JUDGE_MODEL, after confirming it's set and isn't GENERATOR_MODEL or
    VERIFIER_MODEL. Called at the start of every judge call in this module
    (not just once at import) so a `.env` edit mid-run can't silently start
    grading with a circular model.
    """
    if not JUDGE_MODEL:
        raise JudgeConfigError(
            "JUDGE_MODEL is not set. RAGAS scoring needs a model that is NOT "
            "part of the pipeline being evaluated. Set JUDGE_MODEL in .env to "
            "a third OpenRouter model (not GENERATOR_MODEL or VERIFIER_MODEL)."
        )
    from app.llm import GENERATOR_MODEL, VERIFIER_MODEL
    if JUDGE_MODEL in (GENERATOR_MODEL, VERIFIER_MODEL):
        raise JudgeConfigError(
            f"JUDGE_MODEL ({JUDGE_MODEL!r}) is the same model already used as the "
            f"generator or verifier in app/llm.py. Grading a system with a model "
            f"that is also part of that system is circular — pick a third model."
        )
    return JUDGE_MODEL


def _chat(prompt: str, system: str, role: str) -> str:
    model = require_independent_judge()
    result = llm_client.chat(
        system, prompt, model=model, temperature=JUDGE_TEMPERATURE,
        max_tokens=JUDGE_MAX_TOKENS, role=f"ragas:{role}",
        thinking_off=JUDGE_DISABLE_THINKING,
    )
    return result.text


def preflight() -> str:
    """One tiny judge call (a fraction of a cent) to prove JUDGE_MODEL is set,
    independent, and actually routable on OpenRouter with these settings —
    run BEFORE any paid generator/verifier call, so a judge misconfiguration
    can't waste a pipeline run whose answer then can't be scored. Returns the
    model id; raises JudgeConfigError or llm_client.LLMError on failure."""
    model = require_independent_judge()
    raw = _chat('Reply with exactly this JSON object: {"ok": true}',
                "You are a connectivity check. Reply with JSON only.", "preflight")
    if extract_json_value(raw, "ok") is not True:
        raise JudgeConfigError(f"{model} answered the preflight but not with the expected JSON: {raw[:200]!r}")
    return model


# ── Groundedness (Faithfulness): claims in the answer vs. the context ─────

_GROUNDEDNESS_SYSTEM = """You are a strict fact-checking auditor for a Philippine BIR tax Q&A system \
under evaluation. You do not answer questions or use outside knowledge. Your only job is to \
break an ANSWER into its individual factual claims and check each one against the CONTEXT it \
was supposedly based on.

A "claim" is one discrete, checkable statement of fact — a rate, date, deadline, form number, \
condition, definition, or rule. Do NOT list: greetings, hedges ("please note that..."), \
requests to consult a professional, or the citation markers like [1] themselves (those are not \
claims, they are labels).

A claim is "supported" only if the CONTEXT states it or something that directly entails it — not \
if it merely sounds plausible or matches general tax knowledge. Judge facts, not wording: a \
correct paraphrase or translation of something in the CONTEXT counts as supported.

Reply with a single JSON object and nothing else."""

_GROUNDEDNESS_PROMPT = """CONTEXT (the retrieved excerpts the answer was supposed to be grounded in):
{context}

ANSWER TO CHECK:
{answer}

List every factual claim in the ANSWER and whether the CONTEXT supports it.

Reply with ONLY this JSON object:
{{"claims": [{{"claim": "<the claim, quoted or closely paraphrased>", "supported": true or false}}, ...]}}

If the ANSWER contains no checkable factual claims (e.g. it is a refusal, a greeting, or purely \
a request for clarification), reply {{"claims": []}}."""


@dataclass
class ClaimJudgement:
    claim: str
    supported: bool


@dataclass
class GroundednessResult:
    claims: List[ClaimJudgement] = field(default_factory=list)
    raw: str = ""
    error: Optional[str] = None

    @property
    def score(self) -> Optional[float]:
        """supported / total, or None if there were no claims to judge (an
        answer with nothing checkable in it — a refusal or a greeting — has
        no faithfulness score, not a score of 0 or 1; exclude it from any
        average rather than picking a value that isn't really measured."""
        if self.error or not self.claims:
            return None
        return sum(1 for c in self.claims if c.supported) / len(self.claims)


def judge_groundedness(context: str, answer: str) -> GroundednessResult:
    raw = _chat(
        _GROUNDEDNESS_PROMPT.format(context=context, answer=answer),
        _GROUNDEDNESS_SYSTEM, "groundedness",
    )
    items = as_list(extract_json_value(raw, "claims"))
    if items is None:
        return GroundednessResult(raw=raw, error="judge did not return a parseable claims list")
    claims = []
    for item in items:
        if isinstance(item, dict) and "claim" in item:
            claims.append(ClaimJudgement(str(item["claim"]), bool(item.get("supported"))))
    return GroundednessResult(claims=claims, raw=raw)


# ── Context Relevance: sentences in the retrieved context vs. the query ───
#
# Scoring version 3 (rag_eval/run_evaluation.py SCORING_VERSION): the
# sentences are split IN PYTHON (app/textproc.split_sentences — the same
# splitter ingestion and evidence compression use), numbered, and the judge
# only returns the numbers of the relevant ones. Before (v1/v2) the judge
# both split the context AND judged it, quoting every sentence back:
#   * the denominator was whatever the judge decided — the quoted sentences
#     covered between 39% and 123% of the actual context text across units,
#     so two runs over the same context could have different totals;
#   * quoting 20-40 sentences verbatim overflowed the 1200-token cap and lost
#     4 units' scores entirely.
# Source label lines ("[1] RMC No. 116-2024 Digest, p.1 (2024)") are citation
# headers added by format_context(), not retrieved content, so they are not
# counted as sentences.

_CONTEXT_RELEVANCE_SYSTEM = """You are grading a retrieval system for a Philippine BIR tax Q&A \
service under evaluation. You do not answer the query. You are given the QUERY and the \
RETRIEVED CONTEXT already split into numbered sentences. Decide, for each sentence, whether it \
is relevant to answering the QUERY.

A sentence is relevant if it states information that would help answer the QUERY, even \
partially (a rule, condition, figure, date, exception or definition the answer needs). A \
sentence that is on-topic (about BIR/tax matters generally, or about the same issuance) but does \
not bear on THIS query is not relevant — judge against the specific query, not the general \
subject area. The QUERY may be in English, Filipino or Taglish while the context is English: \
judge meaning, not wording.

Reply with a single JSON object and nothing else."""

_CONTEXT_RELEVANCE_PROMPT = """QUERY:
{query}

RETRIEVED CONTEXT (numbered sentences):
{numbered}

Reply with ONLY this JSON object, listing the numbers of the relevant sentences (an empty list \
if none is relevant):
{{"relevant": [<sentence numbers>]}}"""

_LABEL_LINE_RE = re.compile(r"^\s*\[\d+\]\s")


def context_sentences(context: str) -> List[str]:
    """
    The retrieved context as the list of sentences Context Relevance is
    computed over: each "---"-separated excerpt block loses its "[n] source"
    label line, is cleaned of PDF line wraps (so old, hard-wrapped chunks
    split the same way as new ones), and is split with the shared sentence
    splitter.
    """
    sentences: List[str] = []
    for block in re.split(r"\n\s*---\s*\n", context or ""):
        lines = block.strip().split("\n")
        if lines and _LABEL_LINE_RE.match(lines[0]):
            lines = lines[1:]
        text = clean_pdf_text("\n".join(lines))
        sentences.extend(s for s in split_sentences(text) if s.strip() != "…")
    return sentences


@dataclass
class SentenceJudgement:
    text: str
    relevant: bool


@dataclass
class ContextRelevanceResult:
    sentences: List[SentenceJudgement] = field(default_factory=list)
    raw: str = ""
    error: Optional[str] = None

    @property
    def score(self) -> Optional[float]:
        """relevant / total, or None if the context was empty (nothing was
        retrieved at all — a different failure than "retrieved but useless",
        which would score 0.0)."""
        if self.error or not self.sentences:
            return None
        return sum(1 for s in self.sentences if s.relevant) / len(self.sentences)


def _parse_indices(value, n: int) -> Optional[List[int]]:
    """Valid 1-based sentence numbers from the judge's list (ints or numeric
    strings); out-of-range numbers are ignored. None = not a list at all."""
    items = as_list(value)
    if items is None:
        return None
    out = set()
    for item in items:
        try:
            k = int(str(item).strip().strip("[]"))
        except (TypeError, ValueError):
            continue
        if 1 <= k <= n:
            out.add(k)
    return sorted(out)


def judge_context_relevance(query: str, context: str) -> ContextRelevanceResult:
    sentences = context_sentences(context)
    if not sentences:
        return ContextRelevanceResult(error="the context contains no sentences")
    numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences, start=1))
    raw = _chat(
        _CONTEXT_RELEVANCE_PROMPT.format(query=query, numbered=numbered),
        _CONTEXT_RELEVANCE_SYSTEM, "context_relevance",
    )
    indices = _parse_indices(extract_json_value(raw, "relevant"), len(sentences))
    if indices is None:
        return ContextRelevanceResult(raw=raw, error="judge did not return a parseable list of sentence numbers")
    chosen = set(indices)
    return ContextRelevanceResult(
        sentences=[SentenceJudgement(s, i in chosen) for i, s in enumerate(sentences, start=1)],
        raw=raw,
    )


# ── Answer Relevance: reverse-engineer questions from the answer ─────────

_REVERSE_QUESTION_SYSTEM = """You reverse-engineer questions from answers, for evaluating a \
Philippine BIR tax Q&A system. You do not answer anything or use outside knowledge about tax law.

Given only an ANSWER (not the original question), write plausible questions that this ANSWER \
would be a direct, complete response to. Base every question ONLY on what the ANSWER actually \
says — do not invent details the answer doesn't contain, and do not use any outside tax \
knowledge to guess what "should" have been asked.

If the ANSWER is a refusal, a greeting, or otherwise does not actually answer any tax question, \
write questions that reflect THAT (e.g. "What can you help me with?" for a greeting) rather than \
inventing a tax question it never addressed.

Write the questions in the SAME language and style as the ANSWER (English, Filipino or Taglish). \
They are compared with the user's original question by embedding similarity, and a question in a \
different language scores lower for reasons that have nothing to do with relevance.

Reply with a single JSON object and nothing else."""

_REVERSE_QUESTION_PROMPT = """ANSWER:
{answer}

Write exactly {n} different questions that this ANSWER would be a good, direct response to.

Reply with ONLY this JSON object:
{{"questions": ["<question 1>", "<question 2>", ...]}}"""


@dataclass
class AnswerRelevanceResult:
    original_query: str = ""
    generated_questions: List[str] = field(default_factory=list)
    similarities: List[float] = field(default_factory=list)  # same order as generated_questions
    raw: str = ""
    error: Optional[str] = None

    @property
    def score(self) -> Optional[float]:
        """Mean cosine similarity, or None if nothing could be generated/embedded."""
        if self.error or not self.similarities:
            return None
        return sum(self.similarities) / len(self.similarities)


def generate_reverse_questions(answer: str, n: int = JUDGE_ANSWER_QUESTIONS) -> List[str]:
    """The LLM half of Answer Relevance. Embedding + cosine similarity happen
    in embeddings_eval.py, kept separate so this stays testable without Jina."""
    raw = _chat(
        _REVERSE_QUESTION_PROMPT.format(answer=answer, n=n),
        _REVERSE_QUESTION_SYSTEM, "answer_relevance",
    )
    items = as_list(extract_json_value(raw, "questions"))
    if items is None:
        return []
    return [str(q) for q in items if str(q).strip()]


# ── Answer Correctness: the answer vs. the GROUND TRUTH (supplementary) ───
#
# Not one of the SOP's three RAGAS metrics — added because the evaluation's
# fourth data point, the ground truth ("possible correct answer"), is only
# used here. Groundedness can't do this job: it checks an answer against the
# excerpts the system itself retrieved, so an answer faithfully built from an
# outdated excerpt scores 1.0; and a chatbot with no retrieval at all (REVIE)
# has no Groundedness score. Answer Correctness checks every system against
# the same human-written reference, so it is the fair Sagot AI vs. REVIE
# comparison of "did it get the facts right".
#
# Direction: the judge lists the FACTS IN THE GROUND TRUTH and checks each
# against the answer (recall-style), rather than listing the answer's claims.
# A ground truth is a short sentence; a detailed, correct answer would be
# penalized for every extra correct detail the reference doesn't mention if
# the answer's claims were scored instead — a bias against longer answers.
# Contradictions are counted separately: a fact the answer states WRONGLY
# is the direct, reference-based evidence of a hallucination.

_CORRECTNESS_SYSTEM = """You are grading a Philippine BIR tax Q&A system against a reference \
answer written by a human expert. You do not use outside knowledge: the GROUND TRUTH is \
treated as correct.

Break the GROUND TRUTH into its key facts (a rate, amount, date, deadline, form, law or \
issuance number, condition, yes/no conclusion). For EACH fact, decide what the ANSWER does with it:
  "correct"      - the answer states this fact, or something equivalent in meaning
  "contradicted" - the answer states something incompatible with it (a different rate, the \
opposite conclusion, a different law, a wrong amount)
  "missing"      - the answer does not address this fact (including when the answer refuses)

The two texts may be in different languages (English, Filipino, Taglish): judge meaning, not \
wording. Extra details in the answer that the ground truth does not mention are NOT \
penalized here.

Reply with a single JSON object and nothing else."""

_CORRECTNESS_PROMPT = """GROUND TRUTH (the correct answer):
{ground_truth}

ANSWER TO GRADE:
{answer}

Reply with ONLY this JSON object:
{{"facts": [{{"fact": "<one key fact from the ground truth>", "verdict": "correct" | "contradicted" | "missing", "evidence": "<the answer's words that support or contradict it, or empty>"}}, ...]}}"""

CORRECTNESS_VERDICTS = ("correct", "contradicted", "missing")


@dataclass
class FactJudgement:
    fact: str
    verdict: str          # "correct" | "contradicted" | "missing"
    evidence: str = ""


@dataclass
class AnswerCorrectnessResult:
    facts: List[FactJudgement] = field(default_factory=list)
    raw: str = ""
    error: Optional[str] = None

    @property
    def score(self) -> Optional[float]:
        """correct / total ground-truth facts, or None if nothing was judged
        (no ground truth given, or the judge failed)."""
        if self.error or not self.facts:
            return None
        return sum(1 for f in self.facts if f.verdict == "correct") / len(self.facts)

    @property
    def n_contradicted(self) -> int:
        return sum(1 for f in self.facts if f.verdict == "contradicted")


def judge_answer_correctness(ground_truth: str, answer: str) -> AnswerCorrectnessResult:
    raw = _chat(
        _CORRECTNESS_PROMPT.format(ground_truth=ground_truth, answer=answer),
        _CORRECTNESS_SYSTEM, "answer_correctness",
    )
    items = as_list(extract_json_value(raw, "facts"))
    if items is None:
        return AnswerCorrectnessResult(raw=raw, error="judge did not return a parseable facts list")
    facts = []
    for item in items:
        if isinstance(item, dict) and "fact" in item:
            verdict = str(item.get("verdict", "")).strip().lower()
            if verdict not in CORRECTNESS_VERDICTS:
                verdict = "missing"  # unknown label: never count it as correct
            facts.append(FactJudgement(str(item["fact"]), verdict, str(item.get("evidence") or "")))
    return AnswerCorrectnessResult(facts=facts, raw=raw)
