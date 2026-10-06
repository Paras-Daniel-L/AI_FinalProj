"""
Deterministic tax calculator. Pure functions, Decimal arithmetic, no I/O,
no model calls: the same inputs and the same rule always give the same
answer, and every number in the result can be traced to an input or to the
rule version that was applied.

Adding a tax type = add a rule file in tax_rules/ and (if its method is new)
one function here registered in METHODS. Nothing else in the pipeline changes.

Each result carries `steps`: a language-neutral list of what was computed.
app/computation/messages.py turns them into English / Filipino / Taglish text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from typing import Callable, Dict, List, Optional

from .rules import Bracket, RuleVersion, TaxRule

CENT = Decimal("0.01")


def money(value: Decimal) -> Decimal:
    """Round to centavos, half-up (how amounts are shown and compared)."""
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


class NotEligible(Exception):
    """The inputs fall outside the rule's eligibility condition."""

    def __init__(self, reason: str, **details):
        super().__init__(reason)
        self.reason = reason
        self.details = details


@dataclass
class ComputationResult:
    tax_type: str
    tax_year: int
    rule_id: str
    version_id: str
    inputs: Dict[str, Decimal]
    tax_due: Decimal
    steps: List[Dict] = field(default_factory=list)
    tax_withheld: Optional[Decimal] = None
    tax_payable: Optional[Decimal] = None      # tax_due - withheld; negative = overwithheld
    bracket: Optional[Bracket] = None
    per_period: Optional[Dict] = None          # employee mode: tax and take-home per pay period

    def to_dict(self) -> Dict:
        def conv(v):
            return str(v) if isinstance(v, Decimal) else v
        return {
            "tax_type": self.tax_type,
            "tax_year": self.tax_year,
            "rule_id": self.rule_id,
            "version_id": self.version_id,
            "inputs": {k: conv(v) for k, v in self.inputs.items()},
            "tax_due": str(self.tax_due),
            "tax_withheld": conv(self.tax_withheld),
            "tax_payable": conv(self.tax_payable),
            "per_period": {k: conv(v) for k, v in self.per_period.items()} if self.per_period else None,
            "steps": [{k: conv(v) for k, v in s.items()} for s in self.steps],
        }


def find_bracket(brackets: List[Bracket], amount: Decimal) -> Bracket:
    for b in brackets:
        if b.contains(amount):
            return b
    raise ValueError(f"no bracket contains {amount}")  # impossible for a validated schedule


def _apply_withholding(result: ComputationResult, withheld: Optional[Decimal]) -> None:
    if withheld is None:
        return
    result.tax_withheld = money(withheld)
    result.tax_payable = money(result.tax_due - withheld)
    result.steps.append({"kind": "less_withheld", "tax_due": result.tax_due,
                         "withheld": result.tax_withheld, "payable": result.tax_payable})


def compute_graduated(rule: TaxRule, version: RuleVersion, year: int, values: Dict[str, Decimal]) -> ComputationResult:
    income = values["taxable_income"]
    b = find_bracket(version.brackets, income)
    excess = income - b.over
    rate_part = excess * b.rate
    tax_due = money(b.base_tax + rate_part)

    steps: List[Dict] = [{"kind": "bracket", "over": b.over, "not_over": b.not_over,
                          "base_tax": b.base_tax, "rate": b.rate}]
    if b.rate == 0:
        steps.append({"kind": "zero_bracket", "income": income, "not_over": b.not_over})
    else:
        steps.append({"kind": "excess", "income": income, "over": b.over, "excess": money(excess)})
        steps.append({"kind": "rate_times_excess", "rate": b.rate, "excess": money(excess), "result": money(rate_part)})
        steps.append({"kind": "add_base", "base_tax": b.base_tax, "rate_part": money(rate_part), "tax_due": tax_due})

    result = ComputationResult(
        tax_type=rule.tax_type, tax_year=year, rule_id=rule.rule_id, version_id=version.version_id,
        inputs={"taxable_income": income}, tax_due=tax_due, steps=steps, bracket=b,
    )
    _apply_withholding(result, values.get("tax_withheld"))
    return result


def compute_flat_over_threshold(rule: TaxRule, version: RuleVersion, year: int, values: Dict[str, Decimal]) -> ComputationResult:
    p = version.params
    sales = values["gross_sales_receipts"]
    other = values.get("non_operating_income") or Decimal(0)
    gross = sales + other
    if gross > p["eligibility_max_gross"]:
        raise NotEligible("over_vat_threshold", gross=gross, limit=p["eligibility_max_gross"])

    net = max(gross - p["threshold_deduction"], Decimal(0))
    tax_due = money(net * p["rate"])
    steps: List[Dict] = [
        {"kind": "gross_total", "sales": sales, "other": other, "gross": gross},
        {"kind": "less_threshold", "gross": gross, "deduction": p["threshold_deduction"], "net": net},
        {"kind": "flat_rate", "rate": p["rate"], "net": net, "tax_due": tax_due},
    ]
    result = ComputationResult(
        tax_type=rule.tax_type, tax_year=year, rule_id=rule.rule_id, version_id=version.version_id,
        inputs={"gross_sales_receipts": sales, "non_operating_income": other}, tax_due=tax_due, steps=steps,
    )
    _apply_withholding(result, values.get("tax_withheld"))
    return result


# ── Employees: from gross pay ────────────────────────────────────────────

@dataclass
class Contributions:
    """Monthly employee shares, as computed from a contribution schedule."""
    monthly_compensation: Decimal
    sss_msc: Decimal
    sss: Decimal
    philhealth_base: Decimal
    philhealth: Decimal
    pagibig_base: Decimal
    pagibig: Decimal

    @property
    def monthly_total(self) -> Decimal:
        return self.sss + self.philhealth + self.pagibig


class ContributionsUnavailable(Exception):
    """The schedule can't be applied to this pay (e.g. at or below the
    Pag-IBIG minimum the rule file auto-computes for)."""


def compute_contributions(version: RuleVersion, monthly: Decimal) -> Contributions:
    """Employee shares for one month of compensation, per the verified schedule."""
    p = version.params
    if monthly <= p["pagibig_auto_min_monthly_compensation"]:
        raise ContributionsUnavailable("monthly compensation at or below the auto-compute minimum")
    # SSS: MSC brackets are msc_step wide and centred on each MSC (P5,250-P5,749.99 -> P5,500).
    step = p["sss_msc_step"]
    msc = (monthly / step).quantize(Decimal(1), rounding=ROUND_HALF_UP) * step
    msc = min(max(msc, p["sss_msc_min"]), p["sss_msc_max"])
    sss = money(msc * p["sss_employee_rate"])
    # PhilHealth: rate x salary clamped to [floor, ceiling], shared with the employer.
    ph_base = min(max(monthly, p["philhealth_income_floor"]), p["philhealth_income_ceiling"])
    philhealth = money(ph_base * p["philhealth_premium_rate"] * p["philhealth_employee_share"])
    # Pag-IBIG: rate x compensation up to the maximum Fund Salary.
    pi_base = min(monthly, p["pagibig_max_fund_salary"])
    pagibig = money(pi_base * p["pagibig_employee_rate"])
    return Contributions(monthly, msc, sss, ph_base, philhealth, pi_base, pagibig)


def compute_compensation(
    rule: TaxRule, version: RuleVersion, year: int, values: Dict[str, Decimal], pay_period: str,
    schedule: TaxRule, schedule_version: RuleVersion, contributions_version: Optional[RuleVersion] = None,
) -> ComputationResult:
    """
    Employee mode: annualize gross pay, exclude the employee's mandatory
    contributions (from the verified schedule, or the user's own annual
    figure), add only the 13th-month pay and benefits above the exclusion cap,
    then apply the graduated table for the same tax year.
    """
    p = version.params
    periods = p.get(f"periods_{pay_period}")
    if periods is None:
        raise ValueError(f"unsupported pay period {pay_period!r}")
    gross = values["gross_pay"]
    annual_pay = gross * periods
    steps: List[Dict] = []
    if periods != 1:
        steps.append({"kind": "annualize", "gross": gross, "periods": periods, "period": pay_period, "annual": money(annual_pay)})

    if values.get("contributions") is not None:
        contrib_annual = values["contributions"]
        contrib_monthly = None
        steps.append({"kind": "contrib_manual", "annual": money(contrib_annual)})
    else:
        if contributions_version is None:
            raise ValueError("no contribution schedule and no contributions given")
        c = compute_contributions(contributions_version, annual_pay / 12)
        contrib_monthly = c.monthly_total
        contrib_annual = money(c.monthly_total * 12)
        steps.append({"kind": "contrib_auto", "monthly": money(c.monthly_compensation), "msc": c.sss_msc,
                      "sss": c.sss, "sss_rate": contributions_version.params["sss_employee_rate"],
                      "philhealth": c.philhealth, "ph_base": money(c.philhealth_base),
                      "ph_rate": contributions_version.params["philhealth_premium_rate"]
                      * contributions_version.params["philhealth_employee_share"],
                      "pagibig": c.pagibig, "pi_base": money(c.pagibig_base),
                      "pi_rate": contributions_version.params["pagibig_employee_rate"],
                      "monthly_total": money(c.monthly_total), "annual": contrib_annual})

    benefits = values.get("benefits") or Decimal(0)
    cap = p["benefits_exclusion_cap"]
    taxable_benefits = max(benefits - cap, Decimal(0))
    if benefits > 0:
        steps.append({"kind": "benefits", "benefits": benefits, "cap": cap, "taxable": money(taxable_benefits)})

    taxable = max(annual_pay - contrib_annual + taxable_benefits, Decimal(0))
    steps.append({"kind": "taxable_compensation", "annual": money(annual_pay), "contributions": money(contrib_annual),
                  "taxable_benefits": money(taxable_benefits), "taxable": money(taxable)})

    table = compute_graduated(schedule, schedule_version, year, {"taxable_income": money(taxable)})
    steps += table.steps

    result = ComputationResult(
        tax_type=rule.tax_type, tax_year=year, rule_id=rule.rule_id, version_id=version.version_id,
        inputs={"gross_pay": gross, "annual_pay": money(annual_pay), "benefits": benefits,
                "contributions": money(contrib_annual), "taxable_income": money(taxable)},
        tax_due=table.tax_due, steps=steps, bracket=table.bracket,
    )
    if periods != 1:
        per_period_tax = money(table.tax_due / periods)
        per_period_contrib = money(contrib_annual / periods)
        take_home = money(gross - per_period_contrib - per_period_tax)
        result.steps.append({"kind": "per_period", "period": pay_period, "periods": periods, "tax_due": table.tax_due,
                             "tax": per_period_tax, "gross": gross, "contributions": per_period_contrib,
                             "take_home": take_home})
        result.per_period = {"period": pay_period, "tax": per_period_tax, "take_home": take_home}
    _apply_withholding(result, values.get("tax_withheld"))
    return result


METHODS: Dict[str, Callable[[TaxRule, RuleVersion, int, Dict[str, Decimal]], ComputationResult]] = {
    "graduated_schedule": compute_graduated,
    "flat_rate_over_threshold": compute_flat_over_threshold,
}


def compute(rule: TaxRule, version: RuleVersion, year: int, values: Dict[str, Decimal]) -> ComputationResult:
    """Run the rule's method. Callers must have checked that `version` covers
    `year` and is verified (Registry.lookup does both) and that every required
    input is present and valid (dialogue.py does that)."""
    if not version.covers(year):
        raise ValueError(f"rule version {version.version_id} does not cover {year}")
    missing = [k for k in rule.required_inputs if k != "tax_year" and values.get(k) is None]
    if missing:
        raise ValueError(f"missing inputs: {missing}")
    return METHODS[rule.method](rule, version, year, values)
