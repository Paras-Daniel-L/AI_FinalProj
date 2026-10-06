"""
Intent routing — rule-based, runs before retrieval, after app/greetings.py.

Labels
------
  CHAT               "what can you do?", "sino ka?", "help": fixed capability
                     reply, no retrieval, no model call.
  TAX_COMPUTATION    the user wants a number computed for THEIR situation
                     ("Magkano tax ko kung 800k ang taxable income ko?",
                     "Can you compute my income tax?", "How much tax do I
                     owe?", "Calculate this for me"). Goes to app/computation.
  TAX_INFORMATION    everything with a tax signal (tax words, an issuance
                     number, a BIR form, a year...). Goes to RAG, unchanged.
  OUT_OF_SCOPE       no tax signal at all ("what's the weather?").

Design choices (defensible at a thesis defense)
------------------------------------------------
1. Computation needs a COMPUTE cue: a compute verb (compute, calculate,
   i-compute, kwentahin, ...) or "how much tax / magkano ... tax" together
   with something personal ("my", "ko", "I owe") or an amount. A how-to
   question without either ("How is income tax computed?", "Paano i-compute
   ang income tax?") stays TAX_INFORMATION: the knowledge base explains the
   steps (BIR FAQ Q10).
2. OUT_OF_SCOPE does NOT skip retrieval. A lexical scope test is too brittle
   for this corpus (e.g. "Is POGO now banned?" has no tax word but is answered
   by a BIR issuance), so the knowledge base itself is the scope test: the
   question still goes through RAG, and the label only chooses which guidance
   message is shown if no evidence is found.
3. `vague` marks a message too short to retrieve on ("tax?", "VAT",
   "deadline?", "income tax"): the bot asks a clarifying question instead of
   retrieving on one word.

A regression test checks that no T-TED evaluation question is routed to
computation, chat or clarification, so the evaluated RAG path is unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

CHAT = "CHAT"
TAX_INFORMATION = "TAX_INFORMATION"
TAX_COMPUTATION = "TAX_COMPUTATION"
OUT_OF_SCOPE = "OUT_OF_SCOPE"

_WORD_RE = re.compile(r"[A-Za-zÀ-ÿñÑ0-9'%₱-]+")

# ── Chat / capability questions (whole message) ─────────────────────────
_CHAT_RE = re.compile(
    r"""^\s*(?:hi|hello|hey)?[\s,!]*(
        help | help\s+me | can\s+you\s+help(\s+me)? |
        what\s+can\s+you\s+do | what\s+do\s+you\s+do | what\s+are\s+you | who\s+are\s+you |
        how\s+do\s+(i|you)\s+use\s+(this|you) | what\s+can\s+i\s+ask(\s+you)? |
        ano(ng)?\s+(ang\s+)?(kaya|magagawa)\s+mo(ng\s+gawin)? | ano(ng)?\s+pwede(ng)?\s+(kong\s+)?(itanong|i-?ask) |
        sino\s+ka | ano\s+ka | paano\s+(ka\s+)?(gamitin|gumamit) | tulungan\s+mo\s+(ako|ko) |
        may\s+tanong\s+ako | i\s+have\s+a\s+question | pwede\s+(ba\s+)?magtanong
    )(\s+(po|ba|please))*\s*[?!.]*\s*$""",
    re.IGNORECASE | re.VERBOSE,
)

# ── Computation cues ────────────────────────────────────────────────────
_COMPUTE_VERB_RE = re.compile(
    r"\b(compute|calculate|calc|kalkulahin|kwentahin|kuwentahin|i-?compute|i-?calculate|icompute|"
    r"kompyutin|compyutin|estimate|tantyahin)\b",
    re.I,
)
_HOW_TO_RE = re.compile(
    r"\b(how\s+(do|does|is|are|to|should|can|would)\b|paano\b|what\s+is\s+the\s+formula|ano\s+ang\s+formula)", re.I)
_HOW_MUCH_TAX_RE = re.compile(
    r"\bhow\s+much\s+(?:income\s+)?tax(?:es)?\b|\bhow\s+much\s+(?:is|will\s+be|would\s+be)\s+(?:my|the)\s+(?:income\s+)?tax\b|\bhow\s+much\s+(?:do|will|would|should|must)\s+(?:i|we)\s+(?:owe|pay)\b|"
    r"\bmagkano\b(?:\W+\w+){0,4}?\W+(?:tax|buwis|babayaran|income\s+tax)\b",
    re.I,
)
# "₱35,000 a month salary, tax for 2025?": a pay amount WITH a pay period and a
# tax word is a computation even without a compute verb. All four are needed,
# so "Is income of ₱250,000 exempt?" stays an information question.
_PAY_WORD_RE = re.compile(r"\b(salary|salaries|sahod|sweldo|suweldo|pay|wages?|income|kita|earn\w*)\b", re.I)
_PAY_PERIOD_RE = re.compile(
    r"a month|per month|monthly|semi-?monthly|kinsenas|a week|per week|weekly|a year|per year|annual(?:ly)?|yearly|"
    r"kada buwan|bawat buwan|buwanan|kada linggo|lingguhan|kada taon|taun-taon", re.I)
_TAX_WORD_RE = re.compile(r"\b(tax|taxes|buwis)\b", re.I)

_OWE_RE = re.compile(r"\b(?:do|will|would|should)\s+(?:i|we)\s+owe\b|\bi\s+owe\b|\bbabayaran\s+(?:ko|kong|namin|naming)\b|\bmy\s+tax\s+due\b", re.I)
_PERSONAL_RE = re.compile(r"\b(my|me|mine|our|ko|kong|akin|aking|namin|naming|natin|i\s+(?:owe|pay|earn|make|made|have))\b", re.I)
_AMOUNT_HINT_RE = re.compile(r"₱\s?\d|\bp(?:hp)?\s?\d|\d{1,3}(?:,\d{3})+|\b\d+(?:\.\d+)?\s?(?:k|m|million|thousand|milyon|libo)\b|\b\d{5,}\b", re.I)

# ── Tax signal (for OUT_OF_SCOPE and vagueness) ─────────────────────────
_TAX_WORDS = {
    "tax", "taxes", "taxable", "taxpayer", "taxpayers", "taxation", "buwis", "bir", "vat", "income", "withholding",
    "withheld", "kaltas", "deduction", "deductions", "exemption", "exemptions", "exempt", "itr", "return", "returns",
    "filing", "revenue", "invoice", "invoicing", "receipt", "receipts", "resibo", "registration", "tin",
    "penalty", "penalties", "multa", "surcharge", "estate", "donor", "excise", "payroll", "compensation",
    "sahod", "sweldo", "de minimis", "percentage", "cgt", "dst", "mcit", "rr", "rmc", "rmo", "rao", "rdao",
    "rdo", "eopt", "train", "create", "pogo", "nirc", "ruling", "rulings", "issuance", "regulation",
    "regulations", "circular", "memorandum", "form", "1701", "1700", "1702", "2316", "2307", "2550",
    "ocr", "e-invoicing", "e-receipt", "eis", "efps", "ebirforms", "orus", "assessment", "audit", "refund",
    "kita", "negosyo", "benta", "deadline", "dividend", "dividends", "interest", "royalty", "royalties",
    "gross", "net", "sales", "seller", "sellers", "employer", "employee", "freelancer", "professional",
    "fiscal", "incentive", "incentives", "ecozone", "peza", "boi", "import", "imported", "customs", "duty",
    "duties", "landed", "doctrine", "eatrig", "batas", "law",
}
_ISSUANCE_RE = re.compile(r"\b\d{1,3}[A-Za-z]?\s?-\s?(?:19|20)\d{2}\b")

# Words that add nothing to a retrieval query. A message whose content words
# are all generic ("income tax", "tax?", "help with VAT") is too vague.
_STOP = {
    "the", "a", "an", "is", "are", "was", "what", "about", "on", "for", "of", "to", "in", "and", "or", "me",
    "my", "i", "you", "can", "could", "please", "tell", "know", "explain", "info", "information",
    "question", "questions", "ask", "help", "with", "how", "do", "does", "any", "some", "ba", "po", "ang",
    "ng", "sa", "na", "mga", "yung", "ano", "anong", "tungkol", "paano", "pa", "lang", "naman", "ko", "mo",
    "ako", "ka", "si", "kay", "ni", "may", "meron", "mayroon", "about", "regarding", "re", "it", "this",
    "that", "there", "here", "need", "want", "gusto", "kailangan", "pwede", "puwede", "ito", "iyan",
    "something", "stuff", "things", "thing", "topic", "details", "detail",
}
_GENERIC = {"tax", "taxes", "buwis", "income", "kita", "bir", "rule", "rules", "law", "laws", "regulation",
            "regulations", "general"}

# ── Topics (to tailor guidance) ─────────────────────────────────────────
_TOPICS = [
    ("computation", re.compile(r"\b(compute|calculate|magkano|how much|kwenta|i-?compute)\b", re.I)),
    ("withholding", re.compile(r"withholding|withheld|kaltas|2307|2316|creditable", re.I)),
    ("vat", re.compile(r"\bvat\b|value[-\s]added|digital services", re.I)),
    ("corporate", re.compile(r"\bmcit\b|corporat|korporasyon|company", re.I)),
    ("filing", re.compile(r"\bfil(e|ing)\b|\bitr\b|return|deadline|mag-?file|ihain", re.I)),
    ("registration", re.compile(r"regist|\btin\b|magparehistro|rehistro", re.I)),
    ("invoicing", re.compile(r"invoice|receipt|resibo|e-?invoic|ocr\b", re.I)),
    ("income", re.compile(r"income|kita|sahod|sweldo|salary|compensation|earn|kumita|kinikita|money|pera", re.I)),
]


@dataclass
class Intent:
    label: str
    vague: bool = False
    tax_signal: bool = False
    topic: str = "general"
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"label": self.label, "vague": self.vague, "tax_signal": self.tax_signal,
                "topic": self.topic, "reasons": self.reasons}


def _words(text: str) -> List[str]:
    return [w.lower().strip("'") for w in _WORD_RE.findall(text or "")]


def has_tax_signal(text: str) -> bool:
    words = _words(text)
    if any(w in _TAX_WORDS or w.startswith("tax") or "buwis" in w for w in words):  # "pagbubuwis", "mabuwisan"
        return True
    return bool(_ISSUANCE_RE.search(text or "") or re.search(r"\b(?:19|20)\d{2}\b", text or ""))


def detect_topic(text: str) -> str:
    for name, pat in _TOPICS:
        if pat.search(text or ""):
            return name
    return "general"


def is_chat(text: str) -> bool:
    return bool(_CHAT_RE.match(text or ""))


def is_computation_request(text: str) -> List[str]:
    """Reasons the message asks for a computation ([] = it doesn't)."""
    t = text or ""
    personal = bool(_PERSONAL_RE.search(t))
    amount = bool(_AMOUNT_HINT_RE.search(t))
    reasons = []
    if _COMPUTE_VERB_RE.search(t):
        if _HOW_TO_RE.search(t) and not personal and not amount:
            return []  # "How is income tax computed?" -> the knowledge base explains it
        reasons.append("compute_verb")
    if _HOW_MUCH_TAX_RE.search(t) and (personal or amount or re.search(r"\b(i|we)\b", t, re.I)):
        reasons.append("how_much_tax")
    if _OWE_RE.search(t):
        reasons.append("owe")
    if amount and _PAY_WORD_RE.search(t) and _PAY_PERIOD_RE.search(t) and _TAX_WORD_RE.search(t):
        reasons.append("pay_amount_with_period")
    return reasons


def is_vague(text: str) -> bool:
    if _ISSUANCE_RE.search(text or "") or _AMOUNT_HINT_RE.search(text or ""):
        return False
    content = [w for w in _words(text) if w not in _STOP and not re.fullmatch(r"[?!.,]+", w)]
    if not content:
        return True
    if len(content) == 1:
        return True
    return len(content) <= 2 and all(w in _GENERIC for w in content)


def classify(text: str) -> Intent:
    """Pre-retrieval intent. See the module docstring for what each label does."""
    topic = detect_topic(text)
    signal = has_tax_signal(text)
    if is_chat(text):
        return Intent(CHAT, tax_signal=signal, topic=topic, reasons=["capability_question"])
    reasons = is_computation_request(text)
    if reasons:
        return Intent(TAX_COMPUTATION, tax_signal=True, topic="computation", reasons=reasons)
    vague = is_vague(text)
    if signal:
        return Intent(TAX_INFORMATION, vague=vague, tax_signal=True, topic=topic, reasons=["tax_signal"])
    return Intent(OUT_OF_SCOPE, vague=vague, tax_signal=False, topic=topic, reasons=["no_tax_signal"])
