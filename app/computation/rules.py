"""
Tax rule registry: the ONLY place rates, brackets and thresholds come from.

Each file in tax_rules/*.json describes one tax type (one computation method)
and one or more year-bounded VERSIONS of its rule. A version carries its own
legal basis, sources and verification record. The calculator never sees a
number that isn't in one of these files, and the LLM never sees them at all.

Why JSON files and not PDFs in the vector store
-----------------------------------------------
A rate table extracted from a PDF chunk is text: a retrieval miss, an OCR
misread digit or a table split across two chunks would silently produce a
wrong tax. A computation needs the table as DATA — exact numbers, exact year
range, checked once by a person — so it is stored as structured data and
cited back to its legal source. The PDFs stay in the knowledge base for the
explanations (see each rule's `kb_references`).

Validation (load time, fail closed)
-----------------------------------
A rule file is rejected — and that tax type simply becomes "not available" —
when any check fails:
  * required fields missing, unknown method, bad numbers;
  * graduated brackets not starting at 0, not contiguous, or not ending in an
    open top bracket;
  * a bracket's base tax that does not equal the tax at its lower bound
    computed from the bracket below (this catches typos such as the
    "not over P5,000,000" misprint in the published RA 10963 text);
  * two versions of the same tax covering the same year.
A version whose verification.status is not "verified" is kept but never used
for a computation (set ALLOW_UNVERIFIED_RULES=1 only for development).

CLI (project root):
    python -m app.computation.rules          # validate and print every rule
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, List, Optional, Tuple

RULES_DIR = Path(os.environ.get("TAX_RULES_DIR", Path(__file__).resolve().parents[2] / "tax_rules"))
ALLOW_UNVERIFIED_RULES = os.environ.get("ALLOW_UNVERIFIED_RULES", "0").strip() in ("1", "true", "True", "yes")

METHODS = {"graduated_schedule", "flat_rate_over_threshold", "compensation_from_gross", "employee_contributions"}

# Nested parameters of the employee_contributions method: section -> required keys.
_CONTRIBUTION_KEYS = {
    "sss": ("employee_rate", "msc_min", "msc_max", "msc_step"),
    "philhealth": ("premium_rate", "employee_share", "income_floor", "income_ceiling"),
    "pagibig": ("employee_rate", "max_fund_salary", "auto_min_monthly_compensation"),
}
_RATE_KEYS = {"employee_rate", "premium_rate", "employee_share"}


class RuleError(ValueError):
    """A rule file is malformed or internally inconsistent."""


def _dec(value, where: str) -> Decimal:
    try:
        d = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        raise RuleError(f"{where}: {value!r} is not a number")
    if not d.is_finite():
        raise RuleError(f"{where}: {value!r} is not a finite number")
    return d


@dataclass(frozen=True)
class Bracket:
    over: Decimal
    not_over: Optional[Decimal]      # None = open top bracket
    base_tax: Decimal
    rate: Decimal

    def contains(self, amount: Decimal) -> bool:
        # "Over X but not over Y": X < amount <= Y. The first bracket also
        # includes 0 itself ("not over P250,000").
        lower_ok = amount > self.over or (self.over == 0 and amount >= 0)
        upper_ok = self.not_over is None or amount <= self.not_over
        return lower_ok and upper_ok


@dataclass
class RuleVersion:
    version_id: str
    tax_year_from: int
    tax_year_to: int
    label: Dict[str, str]
    legal_basis: str
    sources: List[Dict[str, str]]
    verification: Dict[str, str]
    brackets: List[Bracket] = field(default_factory=list)          # graduated_schedule
    params: Dict[str, Decimal] = field(default_factory=dict)       # flat_rate_over_threshold

    @property
    def verified(self) -> bool:
        return self.verification.get("status") == "verified"

    def covers(self, year: int) -> bool:
        return self.tax_year_from <= year <= self.tax_year_to

    def label_for(self, language: str) -> str:
        return self.label.get(language) or self.label.get("english") or self.version_id


@dataclass
class TaxRule:
    rule_id: str
    tax_type: str
    method: str
    name: Dict[str, str]
    inputs: Dict[str, Dict]
    kb_references: List[Dict]
    versions: List[RuleVersion]
    path: str = ""
    document_type: str = ""
    jurisdiction: str = ""
    uses: Dict[str, str] = field(default_factory=dict)   # other rules this one depends on

    def name_for(self, language: str) -> str:
        return self.name.get(language) or self.name.get("english") or self.tax_type

    @property
    def required_inputs(self) -> List[str]:
        return [k for k, spec in self.inputs.items() if spec.get("required")]

    @property
    def supported_years(self) -> Tuple[int, int]:
        return (min(v.tax_year_from for v in self.versions), max(v.tax_year_to for v in self.versions))

    def version_for_year(self, year: int) -> Optional[RuleVersion]:
        """The ONE version that covers `year`, or None. Never borrows a
        neighbouring year's rule."""
        matches = [v for v in self.versions if v.covers(year)]
        return matches[0] if len(matches) == 1 else None


# ── Parsing + validation ────────────────────────────────────────────────

def _parse_brackets(raw: list, where: str) -> List[Bracket]:
    if not isinstance(raw, list) or not raw:
        raise RuleError(f"{where}: 'brackets' must be a non-empty list")
    out = []
    for i, b in enumerate(raw):
        w = f"{where} bracket {i + 1}"
        not_over = b.get("not_over")
        out.append(Bracket(
            over=_dec(b.get("over"), f"{w} over"),
            not_over=None if not_over is None else _dec(not_over, f"{w} not_over"),
            base_tax=_dec(b.get("base_tax"), f"{w} base_tax"),
            rate=_dec(b.get("rate"), f"{w} rate"),
        ))
    return out


def validate_brackets(brackets: List[Bracket], where: str = "schedule") -> None:
    """Structural and arithmetic consistency of a graduated schedule."""
    if brackets[0].over != 0:
        raise RuleError(f"{where}: the first bracket must start at 0")
    if brackets[-1].not_over is not None:
        raise RuleError(f"{where}: the last bracket must be open-ended (not_over: null)")
    if brackets[0].base_tax != 0:
        raise RuleError(f"{where}: the first bracket's base tax must be 0")
    for i, b in enumerate(brackets):
        if not (Decimal(0) <= b.rate <= Decimal(1)):
            raise RuleError(f"{where} bracket {i + 1}: rate {b.rate} is outside 0..1")
        if b.not_over is not None and b.not_over <= b.over:
            raise RuleError(f"{where} bracket {i + 1}: not_over must be greater than over")
        if i == 0:
            continue
        prev = brackets[i - 1]
        if prev.not_over != b.over:
            raise RuleError(
                f"{where} bracket {i + 1}: starts at {b.over} but the previous bracket ends at {prev.not_over} "
                "(brackets must be contiguous)"
            )
        expected = prev.base_tax + prev.rate * (prev.not_over - prev.over)
        if expected != b.base_tax:
            raise RuleError(
                f"{where} bracket {i + 1}: base tax {b.base_tax} does not match the tax at its lower bound "
                f"computed from the bracket below ({expected}). Check the table for a typo."
            )


def parse_rule(data: dict, path: str = "") -> TaxRule:
    where = path or data.get("rule_id", "rule")
    for key in ("rule_id", "tax_type", "method", "name", "inputs", "versions"):
        if key not in data:
            raise RuleError(f"{where}: missing '{key}'")
    method = data["method"]
    if method not in METHODS:
        raise RuleError(f"{where}: unknown method {method!r} (known: {', '.join(sorted(METHODS))})")
    if "tax_year" not in data["inputs"]:
        raise RuleError(f"{where}: every rule must take 'tax_year' as an input (rules are year-dependent)")

    versions: List[RuleVersion] = []
    for raw in data["versions"]:
        vid = raw.get("version_id") or "?"
        w = f"{where} [{vid}]"
        try:
            y_from, y_to = int(raw["tax_year_from"]), int(raw["tax_year_to"])
        except (KeyError, TypeError, ValueError):
            raise RuleError(f"{w}: tax_year_from / tax_year_to must be integers")
        if y_from > y_to:
            raise RuleError(f"{w}: tax_year_from is after tax_year_to")
        if not raw.get("legal_basis") or not raw.get("sources"):
            raise RuleError(f"{w}: every version needs a legal_basis and at least one source")
        version = RuleVersion(
            version_id=vid,
            tax_year_from=y_from,
            tax_year_to=y_to,
            label=raw.get("label") or {},
            legal_basis=raw["legal_basis"],
            sources=raw["sources"],
            verification=raw.get("verification") or {"status": "unverified"},
        )
        if method == "graduated_schedule":
            version.brackets = _parse_brackets(raw.get("brackets"), w)
            validate_brackets(version.brackets, w)
        elif method == "employee_contributions":
            for section, keys in _CONTRIBUTION_KEYS.items():
                block = raw.get(section) or {}
                for key in keys:
                    if key not in block:
                        raise RuleError(f"{w}: missing '{section}.{key}'")
                    value = _dec(block[key], f"{w} {section}.{key}")
                    if key in _RATE_KEYS and not (Decimal(0) < value <= Decimal(1)):
                        raise RuleError(f"{w}: {section}.{key} must be in (0, 1]")
                    if key not in _RATE_KEYS and value < 0:
                        raise RuleError(f"{w}: {section}.{key} must not be negative")
                    version.params[f"{section}_{key}"] = value
            p = version.params
            if not (p["sss_msc_step"] > 0 and p["sss_msc_min"] < p["sss_msc_max"]):
                raise RuleError(f"{w}: SSS MSC range or step is invalid")
            if p["sss_msc_min"] % p["sss_msc_step"] or p["sss_msc_max"] % p["sss_msc_step"]:
                raise RuleError(f"{w}: SSS minimum and maximum MSC must be multiples of the MSC step")
            if p["philhealth_income_floor"] >= p["philhealth_income_ceiling"]:
                raise RuleError(f"{w}: PhilHealth floor must be below the ceiling")
        elif method == "compensation_from_gross":
            version.params["benefits_exclusion_cap"] = _dec(raw.get("benefits_exclusion_cap"), f"{w} benefits_exclusion_cap")
            periods = raw.get("periods_per_year") or {}
            if "annual" not in periods or _dec(periods["annual"], f"{w} annual") != 1:
                raise RuleError(f"{w}: periods_per_year must include annual = 1")
            for name, n in periods.items():
                n = _dec(n, f"{w} periods_per_year.{name}")
                if n <= 0 or n != n.to_integral_value():
                    raise RuleError(f"{w}: periods_per_year.{name} must be a positive whole number")
                version.params[f"periods_{name}"] = n
        else:
            for key in ("rate", "threshold_deduction", "eligibility_max_gross"):
                if key not in raw:
                    raise RuleError(f"{w}: missing '{key}'")
                version.params[key] = _dec(raw[key], f"{w} {key}")
            if not (Decimal(0) <= version.params["rate"] <= Decimal(1)):
                raise RuleError(f"{w}: rate outside 0..1")
        versions.append(version)

    # No two versions may claim the same year: a computation must never have
    # to choose between two rules.
    ordered = sorted(versions, key=lambda v: v.tax_year_from)
    for a, b in zip(ordered, ordered[1:]):
        if b.tax_year_from <= a.tax_year_to:
            raise RuleError(f"{where}: versions {a.version_id} and {b.version_id} overlap in years")

    return TaxRule(
        rule_id=data["rule_id"],
        tax_type=data["tax_type"],
        method=method,
        name=data["name"],
        inputs=data["inputs"],
        kb_references=data.get("kb_references") or [],
        versions=ordered,
        path=path,
        document_type=data.get("document_type", ""),
        jurisdiction=data.get("jurisdiction", ""),
        uses=data.get("uses") or {},
    )


# ── Registry ────────────────────────────────────────────────────────────

@dataclass
class Registry:
    rules: Dict[str, TaxRule]
    errors: List[str]

    def get(self, tax_type: str) -> Optional[TaxRule]:
        return self.rules.get(tax_type)

    def lookup(self, tax_type: str, year: int) -> Tuple[Optional[TaxRule], Optional[RuleVersion], str]:
        """
        (rule, version, status) where status is one of:
          ok              a verified version covers this exact year
          no_rule         no (valid) rule file for this tax type
          unsupported_year  the rule exists but no version covers the year
          unverified      the covering version is not marked verified
        """
        rule = self.rules.get(tax_type)
        if rule is None:
            return None, None, "no_rule"
        version = rule.version_for_year(year)
        if version is None:
            return rule, None, "unsupported_year"
        if not version.verified and not ALLOW_UNVERIFIED_RULES:
            return rule, version, "unverified"
        return rule, version, "ok"


def load_registry(rules_dir: Path = RULES_DIR) -> Registry:
    rules: Dict[str, TaxRule] = {}
    errors: List[str] = []
    for path in sorted(Path(rules_dir).glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            rule = parse_rule(data, path.name)
            if rule.tax_type in rules:
                raise RuleError(f"{path.name}: tax_type {rule.tax_type!r} is already defined in {rules[rule.tax_type].path}")
            rules[rule.tax_type] = rule
        except (RuleError, json.JSONDecodeError, OSError) as e:
            errors.append(str(e))
            print(f"🚫 [TaxRules] rejected {path.name}: {e}")
    # A rule that depends on another (the employee mode uses the graduated
    # table) is unusable if that rule is missing or was rejected.
    for tax_type in list(rules):
        missing = [dep for dep in rules[tax_type].uses.values() if dep not in rules]
        if missing:
            errors.append(f"{rules[tax_type].path}: depends on missing rule(s) {', '.join(missing)}")
            print(f"🚫 [TaxRules] rejected {rules[tax_type].path}: depends on missing {', '.join(missing)}")
            del rules[tax_type]
    if rules:
        print(f"📐 [TaxRules] loaded: {', '.join(sorted(rules))}")
    return Registry(rules, errors)


_registry: Optional[Registry] = None


def get_registry() -> Registry:
    global _registry
    if _registry is None:
        _registry = load_registry()
    return _registry


def main() -> None:
    reg = load_registry()
    for rule in reg.rules.values():
        lo, hi = rule.supported_years
        print(f"\n{rule.tax_type}  ({rule.path})  years {lo}-{hi}")
        for v in rule.versions:
            print(f"  {v.version_id}: {v.tax_year_from}-{v.tax_year_to}  verified={v.verified}")
            for b in v.brackets:
                top = "and above" if b.not_over is None else f"to {b.not_over:,}"
                print(f"    over {b.over:,} {top}: {b.base_tax:,} + {b.rate * 100:g}% of excess over {b.over:,}")
            for k, val in v.params.items():
                print(f"    {k} = {val}")
    if reg.errors:
        print("\nERRORS:")
        for e in reg.errors:
            print(f"  - {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
