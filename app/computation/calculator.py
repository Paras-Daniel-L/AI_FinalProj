"""
Deterministic tax calculator. Pure functions, Decimal arithmetic, no I/O,
no model calls: the same inputs and the same rule always give the same
answer, and every number in the result can be traced to an input or to the
rule version that was applied.

Adding a tax type = add a rule file in tax_rules/ and (if its method is new)
one function here registered in METHODS. Nothing else in the pipeline changes.

Each result carries `steps`: a language-neutral list of what was computed.
app/computation/messages.py turns them into English / Filipino / Taglish text.

No rounding (system v1.5.2). Every figure is exact Decimal arithmetic on the
user's amounts and the rule file's constants: the taxable income, every
contribution, the tax due, the balance after withholding. Nothing is rounded
to centavos or to whole pesos along the way or at the end. Multiplying by the
law's rates (5%, 2.5%, 15%, 8%...) always gives a finite decimal, so these
figures are exact. The only division is spreading the annual tax over pay
periods (÷ 12, 24 or 52); `divide()` returns that quotient exactly when it
terminates, and otherwise flags it as a repeating decimal (exact=False)
instead of rounding it.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from decimal import ROUND_DOWN, Decimal, Inexact, InvalidOperation, Rounded, localcontext
from fractions import Fraction
from typing import Callable, Dict, List, Optional, Tuple

from .rules import Bracket, RuleVersion, TaxRule

CENT = Decimal("0.01")

# Digits kept for a quotient that never terminates (e.g. ₱1,000 ÷ 12 =
# ₱83.333…). It is cut off here (ROUND_DOWN, never rounded up) and flagged
# exact=False, so the display marks it with "…".
REPEATING_DIGITS = 30


def exact_arithmetic(fn):
    """
    Run a calculator function where Decimal can never round silently: 60
    significant digits (real tax figures need far fewer) and the Inexact and
    Rounded signals turned into errors. If any operation would have to drop a
    digit, the computation stops with an exception instead of returning a
    rounded number.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with localcontext() as ctx:
            ctx.prec = 60
            ctx.traps[Inexact] = True
            ctx.traps[Rounded] = True
            return fn(*args, **kwargs)
    return wrapper


def centavos(value: Decimal) -> Optional[Decimal]:
    """`value` written to the centavo (e.g. 800000 -> 800000.00) when that
    changes nothing, or None when it has digits finer than a centavo. Peso
    INPUTS must be exact to the centavo: an amount like 800,000.555 is asked
    about again, never rounded to fit."""
    with localcontext() as ctx:
        ctx.prec = 60
        ctx.traps[Inexact] = ctx.traps[Rounded] = False
        try:
            q = value.quantize(CENT)
        except InvalidOperation:
            return None
    return q if q == value else None


def divide(numerator: Decimal, periods) -> Tuple[Decimal, bool]:
    """
    numerator ÷ periods without rounding. Returns (quotient, exact).

    exact=True: the decimal terminates and the quotient is the exact value
    (₱20,415 ÷ 12 = ₱1,701.25). exact=False: it repeats (₱3,765 ÷ 52 =
    ₱72.403846153846…); the quotient holds the first REPEATING_DIGITS decimals,
    cut off (never rounded), and the caller must show it as non-terminating.
    """
    frac = Fraction(numerator) / Fraction(periods)
    den = frac.denominator
    for p in (2, 5):
        while den % p == 0:
            den //= p
    exact = den == 1
    with localcontext() as ctx:
        # The one place a digit may be dropped, on purpose and only for a
        # repeating decimal (flagged exact=False): the traps are off here.
        ctx.prec = 80
        ctx.traps[Inexact] = ctx.traps[Rounded] = False
        q = Decimal(frac.numerator) / Decimal(frac.denominator)
        if not exact:
            q = q.quantize(Decimal(1).scaleb(-REPEATING_DIGITS), rounding=ROUND_DOWN)
    return q, exact


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
    result.tax_withheld = withheld
    result.tax_payable = result.tax_due - withheld
    result.steps.append({"kind": "less_withheld", "tax_due": result.tax_due,
                         "withheld": result.tax_withheld, "payable": result.tax_payable})


@exact_arithmetic
def compute_graduated(rule: TaxRule, version: RuleVersion, year: int, values: Dict[str, Decimal]) -> ComputationResult:
    income = values["taxable_income"]
    b = find_bracket(version.brackets, income)
    excess = income - b.over
    rate_part = excess * b.rate
    tax_due = b.base_tax + rate_part

    steps: List[Dict] = [{"kind": "bracket", "over": b.over, "not_over": b.not_over,
                          "base_tax": b.base_tax, "rate": b.rate}]
    if b.rate == 0:
        steps.append({"kind": "zero_bracket", "income": income, "not_over": b.not_over})
    else:
        steps.append({"kind": "excess", "income": income, "over": b.over, "excess": excess})
        steps.append({"kind": "rate_times_excess", "rate": b.rate, "excess": excess, "result": rate_part})
        steps.append({"kind": "add_base", "base_tax": b.base_tax, "rate_part": rate_part, "tax_due": tax_due})

    result = ComputationResult(
        tax_type=rule.tax_type, tax_year=year, rule_id=rule.rule_id, version_id=version.version_id,
        inputs={"taxable_income": income}, tax_due=tax_due, steps=steps, bracket=b,
    )
    _apply_withholding(result, values.get("tax_withheld"))
    return result


@exact_arithmetic
def compute_flat_over_threshold(rule: TaxRule, version: RuleVersion, year: int, values: Dict[str, Decimal]) -> ComputationResult:
    p = version.params
    sales = values["gross_sales_receipts"]
    other = values.get("non_operating_income") or Decimal(0)
    gross = sales + other
    if gross > p["eligibility_max_gross"]:
        raise NotEligible("over_vat_threshold", gross=gross, limit=p["eligibility_max_gross"])

    net = max(gross - p["threshold_deduction"], Decimal(0))
    tax_due = net * p["rate"]
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

def to_decimal(x: Fraction) -> Tuple[Decimal, bool]:
    """An exact fraction as (Decimal, exact): see divide()."""
    return divide(Decimal(x.numerator), x.denominator)


@dataclass
class Contributions:
    """
    Monthly employee shares, as computed from a contribution schedule.

    The arithmetic is done on exact fractions (compute_contributions), so a
    monthly pay that never terminates (weekly pay × 52 ÷ 12) loses nothing.
    The Decimal fields hold those values; each is exact unless its decimal
    repeats (then `exact[name]` is False and it is cut off, never rounded).
    The ANNUAL total always terminates (the ÷ 12 cancels the × 12) and is exact.
    """
    monthly_compensation: Decimal
    sss_msc: Decimal
    sss: Decimal
    philhealth_base: Decimal
    philhealth: Decimal
    pagibig_base: Decimal
    pagibig: Decimal
    monthly_total: Decimal
    annual_total: Decimal
    exact: Dict[str, bool] = field(default_factory=dict)   # field name -> False if the decimal repeats


class ContributionsUnavailable(Exception):
    """The schedule can't be applied to this pay (e.g. at or below the
    Pag-IBIG minimum the rule file auto-computes for)."""


@exact_arithmetic
def compute_contributions(version: RuleVersion, monthly) -> Contributions:
    """Employee shares for one month of compensation, per the verified schedule.
    `monthly` may be a Decimal or an exact Fraction (annual pay ÷ 12)."""
    p = {k: Fraction(v) if isinstance(v, Decimal) else v for k, v in version.params.items()}
    m = Fraction(monthly)
    if m <= p["pagibig_auto_min_monthly_compensation"]:
        raise ContributionsUnavailable("monthly compensation at or below the auto-compute minimum")
    # SSS: the Monthly Salary Credit is the schedule's bracket for this pay,
    # not a rounding of the pay. Each bracket is msc_step wide and centred on
    # its MSC (P5,250.00-P5,749.99 -> MSC P5,500), so the MSC is found from the
    # bracket the pay falls in, then held within the schedule's min and max.
    step = p["sss_msc_step"]
    msc = ((m + step / 2) // step) * step
    msc = min(max(msc, p["sss_msc_min"]), p["sss_msc_max"])
    sss = msc * p["sss_employee_rate"]
    # PhilHealth: rate x salary held within [floor, ceiling], shared with the employer.
    ph_base = min(max(m, p["philhealth_income_floor"]), p["philhealth_income_ceiling"])
    philhealth = ph_base * p["philhealth_premium_rate"] * p["philhealth_employee_share"]
    # Pag-IBIG: rate x compensation up to the maximum Fund Salary.
    pi_base = min(m, p["pagibig_max_fund_salary"])
    pagibig = pi_base * p["pagibig_employee_rate"]
    total = sss + philhealth + pagibig

    parts = {"monthly_compensation": m, "sss_msc": msc, "sss": sss, "philhealth_base": ph_base,
             "philhealth": philhealth, "pagibig_base": pi_base, "pagibig": pagibig,
             "monthly_total": total, "annual_total": total * 12}
    values, exact = {}, {}
    for name, frac in parts.items():
        values[name], exact[name] = to_decimal(frac)
    return Contributions(**values, exact=exact)


@exact_arithmetic
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
        steps.append({"kind": "annualize", "gross": gross, "periods": periods, "period": pay_period, "annual": annual_pay})

    if values.get("contributions") is not None:
        contrib_annual = values["contributions"]
        steps.append({"kind": "contrib_manual", "annual": contrib_annual})
    else:
        if contributions_version is None:
            raise ValueError("no contribution schedule and no contributions given")
        # Monthly pay = annual ÷ 12, kept as an exact fraction: weekly pay
        # (× 52 ÷ 12) can give a repeating decimal, and nothing is rounded.
        c = compute_contributions(contributions_version, Fraction(annual_pay) / 12)
        if not c.exact["annual_total"]:  # cannot happen: the ÷ 12 always cancels the × 12
            raise ValueError("annual contributions did not come out exact")
        contrib_annual = c.annual_total
        steps.append({"kind": "contrib_auto", "monthly": c.monthly_compensation, "msc": c.sss_msc,
                      "sss": c.sss, "sss_rate": contributions_version.params["sss_employee_rate"],
                      "philhealth": c.philhealth, "ph_base": c.philhealth_base,
                      "ph_rate": contributions_version.params["philhealth_premium_rate"]
                      * contributions_version.params["philhealth_employee_share"],
                      "pagibig": c.pagibig, "pi_base": c.pagibig_base,
                      "pi_rate": contributions_version.params["pagibig_employee_rate"],
                      "monthly_total": c.monthly_total, "annual": contrib_annual,
                      # False = that monthly figure is a repeating decimal (shown with "…")
                      "exact": {"monthly": c.exact["monthly_compensation"], "sss": c.exact["sss"],
                                "philhealth": c.exact["philhealth"], "ph_base": c.exact["philhealth_base"],
                                "pagibig": c.exact["pagibig"], "pi_base": c.exact["pagibig_base"],
                                "monthly_total": c.exact["monthly_total"]}})

    benefits = values.get("benefits") or Decimal(0)
    cap = p["benefits_exclusion_cap"]
    taxable_benefits = max(benefits - cap, Decimal(0))
    if benefits > 0:
        steps.append({"kind": "benefits", "benefits": benefits, "cap": cap, "taxable": taxable_benefits})

    taxable = max(annual_pay - contrib_annual + taxable_benefits, Decimal(0))
    steps.append({"kind": "taxable_compensation", "annual": annual_pay, "contributions": contrib_annual,
                  "taxable_benefits": taxable_benefits, "taxable": taxable})

    table = compute_graduated(schedule, schedule_version, year, {"taxable_income": taxable})
    steps += table.steps

    result = ComputationResult(
        tax_type=rule.tax_type, tax_year=year, rule_id=rule.rule_id, version_id=version.version_id,
        inputs={"gross_pay": gross, "annual_pay": annual_pay, "benefits": benefits,
                "contributions": contrib_annual, "taxable_income": taxable},
        tax_due=table.tax_due, steps=steps, bracket=table.bracket,
    )
    if periods != 1:
        # The annual tax and contributions spread evenly over the pay periods.
        # Each is ONE exact division of an annual figure (no rounded figure is
        # reused): take-home = (annual pay − contributions − tax) ÷ periods.
        per_period_tax, tax_exact = divide(table.tax_due, periods)
        per_period_contrib, contrib_exact = divide(contrib_annual, periods)
        take_home, take_home_exact = divide(annual_pay - contrib_annual - table.tax_due, periods)
        exact = {"tax": tax_exact, "contributions": contrib_exact, "take_home": take_home_exact}
        result.steps.append({"kind": "per_period", "period": pay_period, "periods": periods, "tax_due": table.tax_due,
                             "tax": per_period_tax, "gross": gross, "contributions": per_period_contrib,
                             "take_home": take_home, "exact": exact})
        result.per_period = {"period": pay_period, "periods": periods, "tax": per_period_tax, "take_home": take_home,
                             "exact": exact}
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
