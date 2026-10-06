"""
Rule-based extraction of computation inputs from English, Filipino and
Taglish messages: peso amounts (and what each amount is), tax years, the
taxpayer's situation, cancel requests, and taxes we cannot compute.

Why rules and not the LLM: a misread amount becomes a wrong tax. Regexes are
exact, testable and explainable at a defense; every value they produce is
echoed back to the user in the breakdown, so a misparse is visible. Anything
the rules can't read is asked again, never guessed.

Amounts understood: 800000, 800,000, 800,000.50, ₱800,000, P800k, Php 1.2M,
800k, 1.2 million, 2 milyon, 800 thousand, 500 libo. Not treated as amounts:
years (2025), issuance numbers (RMC 34-2024), BIR form numbers (Form 2316),
percentages (8%), and small bare numbers ("3 kids", "13th month").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import List, Optional, Tuple

# ── Amount pattern ──────────────────────────────────────────────────────

_CURRENCY = r"(?:₱|php|piso|p(?=\s?\d))"
_NUMBER = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_SUFFIX = r"k|thousand|libo|m|mil|million|milyon|pesos?|piso|php"
_AMOUNT_RE = re.compile(
    rf"(?<![\w.,])(?P<neg>-\s?)?(?P<cur>{_CURRENCY})?\s?(?P<num>{_NUMBER})(?:\s?(?P<suf>{_SUFFIX}))?(?![\w%])",
    re.IGNORECASE,
)
# Comma groups that are not thousands groups: "800,00", "8,0000".
_MALFORMED_RE = re.compile(r"(?<![\w.,])₱?\s?\d+(?:,\d+)+(?![\w])")
_WELLFORMED_GROUPS = re.compile(r"^\d{1,3}(?:,\d{3})+(?:\.\d+)?$")

_MULTIPLIERS = {
    "k": 1_000, "thousand": 1_000, "libo": 1_000,
    "m": 1_000_000, "mil": 1_000_000, "million": 1_000_000, "milyon": 1_000_000,
}

# Spans that contain digits but are never amounts. Masked before matching.
_NOT_AMOUNT_RES = [
    re.compile(r"\b\d{1,3}[A-Za-z]?\s?-\s?(?:19|20)\d{2}\b"),                     # issuance no. 34-2024
    re.compile(r"\b(?:bir\s+)?form\s*(?:no\.?\s*)?\d{4}[a-z]{0,3}\b", re.I),      # Form 2316, form no. 1701Q
    re.compile(r"\b\d+(?:\.\d+)?\s?(?:%|percent|porsyento|porsiyento)", re.I),     # 8%, 8 percent
    re.compile(r"\b\d+(?:st|nd|rd|th)\b", re.I),                                    # 13th, 4th
    re.compile(r"\b(?:sec(?:tion)?\.?|rr|rmc|rmo|rao|rdao|ra|no\.?)\s*\d+[a-z]?\b", re.I),  # Sec. 24, RA 10963
]

_YEAR_RE = re.compile(r"(?<![\w,.-])(?:ty|fy|cy)?\s?((?:19|20)\d{2})(?![\w%-]|[,.]\d|\s?(?:k|m|thousand|million)\b)", re.I)
_TAX_YEAR_MIN, _TAX_YEAR_MAX = 1990, 2099

# ── Role cues (what an amount is) ───────────────────────────────────────

_ROLE_CUES: List[Tuple[str, re.Pattern]] = [
    ("tax_withheld", re.compile(
        r"withheld|withholding|kinaltas|kaltas|kinaltasan|binawas na tax|tax credit|creditable|2316|2307|nabawas", re.I)),
    ("non_operating_income", re.compile(
        r"non-?\s?operating|other income|iba pang kita|interest income", re.I)),
    ("benefits", re.compile(
        r"13th[-\s]?month|thirteenth[-\s]month|christmas bonus|bonus(?:es)?|productivity incentive|"
        r"other benefits|benepisyo", re.I)),
    ("contributions", re.compile(
        r"contributions?|kontribusyon|\bsss\b|philhealth|pag-?ibig|\bhdmf\b|mandatory deductions", re.I)),
    ("taxable_income", re.compile(r"taxable(?:\s+(?:income|compensation|pay|salary))?", re.I)),
    # Employee pay (before contributions and tax). Listed before "gross" so
    # "gross salary" is pay, not gross sales.
    ("gross_compensation", re.compile(
        r"gross\s+(?:salary|pay|compensation)|basic\s+(?:salary|pay)|monthly\s+pay|salary|salaries|sahod|"
        r"sweldo|suweldo|compensation|\bwages?\b", re.I)),
    ("gross_sales_receipts", re.compile(
        r"gross(?:\s+(?:sales|receipts|income))?|sales|receipts|benta|bentahan|kabuuang kita sa negosyo|revenue", re.I)),
    # "income" only where no more specific cue claims the same words.
    ("income", re.compile(
        r"(?<!taxable )(?<!gross )(?<!other )(?<!interest )(?<!operating )\bincome|\bkita\b|kinikita|kumikita|"
        r"\bearn\w*", re.I)),
]

# Pay periods, most specific first ("semi-monthly" also contains "monthly").
_PERIOD_RES: List[Tuple[str, re.Pattern]] = [
    ("semi_monthly", re.compile(
        r"semi[-\s]?monthly|kinsenas|kinsenahan|twice a month|every (?:15 days|kinsenas|two weeks)|"
        r"(?:per|a|every|kada) cut-?off", re.I)),
    ("monthly", re.compile(
        r"per\s?month|a month|monthly|/\s?mo(?:nth)?\b|kada buwan|bawat buwan|buwan-buwan|buwanan|isang buwan|"
        r"every month", re.I)),
    ("weekly", re.compile(r"per\s?week|a week|weekly|/\s?wk\b|kada linggo|bawat linggo|lingguhan|linggo-linggo", re.I)),
    ("daily", re.compile(r"per\s?day|a day|daily|/\s?day\b|kada araw|bawat araw|arawan|araw-araw", re.I)),
    ("annual", re.compile(
        r"per\s?year|a year|yearly|annual(?:ly)?|per annum|/\s?yr\b|kada taon|bawat taon|taun-taon|taunan\w*|"
        r"sa isang taon|for the (?:whole )?year", re.I)),
]


def detect_period(segment: str) -> Optional[str]:
    for name, pat in _PERIOD_RES:
        if pat.search(segment or ""):
            return name
    return None

# ── Situation / regime / scope ──────────────────────────────────────────

_REGIME_8_RE = re.compile(r"(?<!\d)8\s?%|\b8\s?(?:percent|porsyento|porsiyento)\b|\beight percent\b|\bwalong porsyento\b|\b8-percent\b", re.I)
_REGIME_GRAD_RE = re.compile(r"\bgraduated\b", re.I)
_SELF_EMPLOYED_RE = re.compile(
    r"self[-\s]?employed|freelanc\w*|\bprofessional\b|\bnegosyo\b|\bnegosyante\b|\bbusiness\b|sole proprietor\w*|"
    r"online seller|\bseller\b|\bpractitioner\b|\bconsultant\b",
    re.I,
)
_EMPLOYEE_RE = re.compile(r"\bemployee\b|\bempleyado\b|\bemployed\b|\bsahod\b|\bsweldo\b|\bsuweldo\b|\bsalary\b|\bcompensation\b|minimum wage", re.I)

# Taxes with no computation rule yet. Display keys are localized in messages.py.
_UNSUPPORTED_TAX_RES: List[Tuple[str, re.Pattern]] = [
    ("minimum_wage", re.compile(r"minimum[-\s]wage|\bmwe\b", re.I)),
    ("vat", re.compile(r"(?<!non-)(?<!non )\bvat\b|value[-\s]added", re.I)),
    ("estate", re.compile(r"\bestate tax\b|\bpamana\b|\bestate\b", re.I)),
    ("donor", re.compile(r"\bdonor'?s? tax\b|\bdonation\b|\bdonasyon\b", re.I)),
    ("corporate", re.compile(r"\bcorporate\b|\bcorporation\b|\bkorporasyon\b|\bmcit\b|\bcompany tax\b", re.I)),
    ("percentage", re.compile(r"\bpercentage tax\b", re.I)),
    ("capital_gains", re.compile(r"\bcapital gains?\b|\bcgt\b", re.I)),
    ("dst", re.compile(r"\bdocumentary stamp\b|\bdst\b", re.I)),
    ("excise", re.compile(r"\bexcise\b", re.I)),
    ("real_property", re.compile(r"\breal property tax\b|\bamilyar\b", re.I)),
    ("mixed_income", re.compile(r"\bmixed[-\s]income\b", re.I)),
]

_CANCEL_RE = re.compile(
    r"^\s*(?:ok(?:ay)?,?\s*)?(?:cancel|never\s?mind|forget it|stop|reset|huwag na|wag na|'?wag na|"
    r"hindi na|kalimutan mo na|tama na|ayoko na)\b",
    re.I,
)


@dataclass
class Amount:
    value: Decimal
    role: str            # tax_withheld | non_operating_income | benefits | contributions | taxable_income |
                         # gross_compensation | gross_sales_receipts | income | unlabeled
    raw: str
    period: Optional[str] = None   # monthly | semi_monthly | weekly | daily | annual | None (not stated)
    start: int = 0
    end: int = 0

    @property
    def monthly(self) -> bool:
        return self.period == "monthly"


@dataclass
class Extraction:
    amounts: List[Amount] = field(default_factory=list)
    invalid_amounts: List[str] = field(default_factory=list)   # malformed or negative, shown back to the user
    years: List[int] = field(default_factory=list)
    regime: Optional[str] = None          # "8_percent" | "graduated"
    kind: Optional[str] = None            # "self_employed" | "employee" | "mixed"
    unsupported_tax: Optional[str] = None
    cancel: bool = False
    period: Optional[str] = None          # a pay period named in the message ("monthly", "buwanan")
    spans: List[Tuple[int, int]] = field(default_factory=list)  # text recognized as slot values

    @property
    def has_values(self) -> bool:
        return bool(self.amounts or self.years or self.regime or self.kind or self.invalid_amounts or self.period)


def _mask(text: str, patterns) -> str:
    out = text
    for pat in patterns:
        out = pat.sub(lambda m: " " * len(m.group(0)), out)
    return out


def _to_decimal(num: str, suffix: Optional[str]) -> Optional[Decimal]:
    try:
        value = Decimal(num.replace(",", ""))
    except InvalidOperation:
        return None
    mult = _MULTIPLIERS.get((suffix or "").lower())
    return value * mult if mult else value


def _nearest_cue(segment: str, from_end: bool) -> Optional[str]:
    """The role cue closest to the amount: the LAST cue before it, or the FIRST after it."""
    best = None
    for role, pat in _ROLE_CUES:
        for m in pat.finditer(segment):
            pos = m.end() if from_end else m.start()
            if best is None or (from_end and pos > best[0]) or (not from_end and pos < best[0]):
                best = (pos, role)
    return best[1] if best else None


def parse_amounts(text: str) -> Tuple[List[Amount], List[str]]:
    masked = _mask(text, _NOT_AMOUNT_RES)
    only_number = bool(re.fullmatch(r"\s*(?:₱|php|p)?\s?[\d,.]+\s?(?:k|m|million|thousand)?\s*[.!]?\s*", text, re.I))

    invalid: List[str] = []
    for m in _MALFORMED_RE.finditer(masked):
        digits = m.group(0).lstrip("₱").strip()
        if not _WELLFORMED_GROUPS.match(digits):
            invalid.append(m.group(0).strip())
            masked = masked[: m.start()] + " " * (m.end() - m.start()) + masked[m.end():]

    found: List[Amount] = []
    matches = list(_AMOUNT_RE.finditer(masked))
    for i, m in enumerate(matches):
        num, suffix, cur = m.group("num"), m.group("suf"), m.group("cur")
        plain = not cur and not (suffix and suffix.lower() in _MULTIPLIERS) and "," not in num
        # A bare 4-digit 19xx/20xx is a year, not an amount.
        if plain and re.fullmatch(r"(?:19|20)\d{2}", num):
            continue
        value = _to_decimal(num, suffix)
        if value is None:
            continue

        prev_end = matches[i - 1].end() if i else 0
        next_start = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        before = text[max(prev_end, m.start() - 45): m.start()]
        after = text[m.end(): min(next_start, m.end() + 40)]
        role = _nearest_cue(before, from_end=True) or _nearest_cue(after, from_end=False)

        # Small bare numbers ("3 kids", "1 employer") are not amounts unless the
        # whole message is the number or a cue sits right next to it.
        if plain and value < 1000 and not only_number and not _nearest_cue(text[max(0, m.start() - 25): m.start()], True):
            continue

        if m.group("neg"):
            invalid.append(m.group(0).strip())
            continue
        if value > Decimal("1e12"):
            invalid.append(m.group(0).strip())
            continue

        window = text[max(prev_end, m.start() - 20): min(next_start, m.end() + 25)]
        found.append(Amount(
            value=value, role=role or "unlabeled", raw=m.group(0).strip(),
            period=detect_period(window), start=m.start(), end=m.end(),
        ))
    return found, invalid


def parse_years(text: str) -> List[Tuple[int, int, int]]:
    masked = _mask(text, _NOT_AMOUNT_RES[:2])  # issuance and form numbers are not tax years
    out = []
    for m in _YEAR_RE.finditer(masked):
        year = int(m.group(1))
        if _TAX_YEAR_MIN <= year <= _TAX_YEAR_MAX:
            out.append((year, m.start(), m.end()))
    return out


def extract(text: str) -> Extraction:
    text = text or ""
    ex = Extraction()
    ex.cancel = bool(_CANCEL_RE.search(text))

    ex.amounts, ex.invalid_amounts = parse_amounts(text)
    ex.spans += [(a.start, a.end) for a in ex.amounts]

    years = parse_years(text)
    ex.years = sorted({y for y, _s, _e in years})
    ex.spans += [(s, e) for _y, s, e in years]

    if _REGIME_8_RE.search(text):
        ex.regime = "8_percent"
    elif _REGIME_GRAD_RE.search(text):
        ex.regime = "graduated"

    self_emp, employee = bool(_SELF_EMPLOYED_RE.search(text)), bool(_EMPLOYEE_RE.search(text))
    if self_emp and employee:
        ex.kind = "mixed"
    elif self_emp:
        ex.kind = "self_employed"
    elif employee:
        ex.kind = "employee"

    for key, pat in _UNSUPPORTED_TAX_RES:
        if pat.search(text):
            ex.unsupported_tax = key
            break

    periods = {name for name, pat in _PERIOD_RES if pat.search(text)}
    if "semi_monthly" in periods:
        periods.discard("monthly")
    if len(periods) == 1:
        ex.period = periods.pop()
    return ex


# Words that carry no meaning beyond "here is the value you asked for". Used to
# tell a slot reply ("actually it's 850k po") from a new question that merely
# contains a number or year ("what is the deadline for 2025?").
_FILLER = {
    "my", "it", "its", "it's", "is", "was", "the", "a", "an", "and", "of", "for", "in", "to", "on", "at",
    "actually", "sorry", "oops", "correction", "make", "change", "update", "use", "instead", "please",
    "ok", "okay", "yes", "yeah", "sure", "no", "not", "just", "only", "about", "around", "approximately",
    "roughly", "income", "taxable", "annual", "yearly", "year", "tax", "gross", "sales", "receipts",
    "withheld", "withholding", "ko", "ang", "ng", "na", "ay", "po", "yung", "iyong", "aking", "akin",
    "pala", "lang", "naman", "talaga", "siguro", "mga", "sa", "taon", "noong", "nung", "this", "that",
    "i", "me", "am", "im", "i'm", "earn", "earned", "make", "made", "kita", "kinita", "sahod", "sweldo",
    "salary", "compensation", "employee", "self", "employed", "self-employed", "freelancer", "option",
    "monthly", "semi-monthly", "weekly", "annually", "yearly", "basic", "pay", "contribution", "contributions",
    "sss", "philhealth", "pag-ibig", "pagibig", "bonus", "bonuses", "13th", "benefits", "week", "kinsenas",
    "buwanan", "taunan", "taunang", "lingguhan", "gross", "wage", "wages", "suweldo", "every", "whole",
    "graduated", "percent", "rate", "rates", "eight", "use", "using", "chose", "choose", "pinili", "ako",
    "mo", "ba", "kung", "si", "ni", "be", "will", "should", "would", "go", "with", "ty", "fy", "cy",
    "pesos", "peso", "php", "piso", "k", "thousand", "million", "libo", "milyon", "non-operating",
    "other", "per", "month", "monthly", "kada", "buwan", "already", "na-withhold", "kinaltas",
    "business", "negosyo", "professional", "seller", "online", "compute", "calculate", "recompute",
}
_QUESTION_WORDS = re.compile(
    r"\b(what|when|where|who|why|which|how|ano|anong|kailan|saan|sino|bakit|paano|alin|is there|are there|may|meron)\b", re.I)
_WORD_RE = re.compile(r"[A-Za-zÀ-ÿñÑ'-]+")


def unexplained_words(text: str, ex: Extraction) -> List[str]:
    """Content words left after removing recognized values and filler."""
    chars = list(text)
    for s, e in ex.spans:
        for i in range(s, min(e, len(chars))):
            chars[i] = " "
    remaining = "".join(chars)
    return [w.lower() for w in _WORD_RE.findall(remaining) if w.lower() not in _FILLER and len(w) > 1]


def looks_like_new_question(text: str, ex: Extraction) -> bool:
    """True when a message with some values in it is really a different question."""
    leftover = unexplained_words(text, ex)
    return bool(_QUESTION_WORDS.search(text)) and len(leftover) >= 2
