"""
No rounding in the tax computation (system v1.5.2).

Every figure the calculator produces must be the exact result of the law's
arithmetic on the user's amounts: nothing is rounded to centavos or whole
pesos, at any step. These tests check that against an independent
calculation done with Python's exact fractions (Fraction), written from the
rule values (TRAIN 2023+ table; 2025 SSS / PhilHealth / Pag-IBIG schedules),
not by calling app/computation.
"""

import os
import random
import sys
from decimal import Decimal
from fractions import Fraction as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.computation import dialogue  # noqa: E402
from app.computation.calculator import compute_compensation, divide  # noqa: E402
from app.computation.messages import amount, peso  # noqa: E402
from app.computation.rules import get_registry  # noqa: E402

# ── Independent exact reference (fractions, no Decimal, no rounding) ─────

TABLE_2023_ON = [(0, 0, F(0)), (250000, 0, F(15, 100)), (400000, 22500, F(20, 100)),
                 (800000, 102500, F(25, 100)), (2000000, 402500, F(30, 100)), (8000000, 2202500, F(35, 100))]


def ref_graduated(taxable: F) -> F:
    tax = F(0)
    for over, base, rate in TABLE_2023_ON:
        if taxable > over:
            tax = base + (taxable - over) * rate
    return tax


def ref_contributions_month(m: F) -> F:
    if m < F(5250):
        msc = F(5000)
    elif m >= F(34750):
        msc = F(35000)
    else:
        msc = ((m - 4750) // 500) * 500 + 5000          # 5,250-5,749.99 -> 5,500
    sss = msc * F(5, 100)
    philhealth = min(max(m, F(10000)), F(100000)) * F(25, 1000)
    pagibig = min(m, F(10000)) * F(2, 100)
    return sss + philhealth + pagibig


def ref_employee(pay: F, periods: int) -> dict:
    annual = pay * periods
    contrib = ref_contributions_month(annual / 12) * 12
    taxable = annual - contrib
    tax = ref_graduated(taxable)
    return {"annual": annual, "contrib": contrib, "taxable": taxable, "tax": tax,
            "per_period_tax": tax / periods, "take_home": (annual - contrib - tax) / periods}


def _employee(pay: str, period: str, year: int = 2025):
    reg = get_registry()
    rule, version, _ = reg.lookup("income_tax_compensation", year)
    sched, sched_v, _ = reg.lookup("income_tax_graduated", year)
    _c, contrib_v, _ = reg.lookup("employee_contributions", year)
    return compute_compensation(rule, version, year, {"gross_pay": Decimal(pay)}, period, sched, sched_v, contrib_v)


# ── The calculator keeps every digit ─────────────────────────────────────

def test_sub_centavo_contribution_flows_through_unrounded():
    # 2.5% × ₱34,749.99 = ₱868.74975. Rounding it to ₱868.75 (as before
    # v1.5.2) changed the taxable income and the tax by a fraction of a centavo.
    r = _employee("34749.99", "monthly")
    auto = next(s for s in r.steps if s["kind"] == "contrib_auto")
    assert auto["philhealth"] == Decimal("868.74975")
    assert r.inputs["contributions"] == Decimal("33524.997")
    assert r.inputs["taxable_income"] == Decimal("383474.883")
    assert r.tax_due == Decimal("20021.23245")
    assert r.per_period["tax"] == Decimal("1668.4360375") and r.per_period["exact"]["tax"]


def test_graduated_tax_on_centavos_is_exact_and_so_is_the_balance():
    t = dialogue.handle("Compute my income tax for 2025. My taxable income is ₱800,000.55. "
                        "My employer withheld ₱100,000.10.", "english")
    assert t.outcome == "computation_done"
    assert Decimal(t.result["tax_due"]) == Decimal("102500.1375")        # ₱102,500 + 25% × ₱0.55
    assert Decimal(t.result["tax_payable"]) == Decimal("2500.0375")
    assert "**Income tax due for 2025: ₱102,500.1375**" in t.answer
    assert "**₱2,500.0375 still payable**" in t.answer


def test_eight_percent_is_exact():
    t = dialogue.handle("I'm self-employed, 8% option, gross sales ₱1,234,567.89 for 2025, compute my tax", "english")
    assert t.outcome == "computation_done"
    assert Decimal(t.result["tax_due"]) == Decimal("78765.4312")         # 8% × (₱1,234,567.89 − ₱250,000)


def test_random_salaries_match_the_exact_reference():
    rng = random.Random(20261007)
    for _ in range(300):
        pay = F(rng.randint(600000, 50000000), 100)      # ₱6,000.00 – ₱500,000.00, any centavos
        period, periods = rng.choice([("monthly", 12), ("semi_monthly", 24), ("weekly", 52)])
        if period == "weekly":
            pay = pay / 4
            pay = F(int(pay * 100), 100)
        ref = ref_employee(pay, periods)
        r = _employee(str(Decimal(pay.numerator) / Decimal(pay.denominator)), period)
        # Annual figures always terminate and must match EXACTLY.
        assert F(r.inputs["annual_pay"]) == ref["annual"]
        assert F(r.inputs["contributions"]) == ref["contrib"]
        assert F(r.inputs["taxable_income"]) == ref["taxable"]
        assert F(r.tax_due) == ref["tax"]
        # Per-period figures: exact when they terminate; otherwise cut off,
        # never rounded up, and within 10^-30 of the true value.
        for key, ref_key in (("tax", "per_period_tax"), ("take_home", "take_home")):
            got, exact = F(r.per_period[key]), r.per_period["exact"][key]
            if exact:
                assert got == ref[ref_key]
            else:
                assert got <= ref[ref_key] < got + F(1, 10 ** 30)


# ── Division: exact when it terminates, flagged (never rounded) when not ─

def test_divide_terminating_and_repeating():
    assert divide(Decimal("20415"), 12) == (Decimal("1701.25"), True)
    q, exact = divide(Decimal("3765"), 52)
    assert not exact
    assert str(q).startswith("72.403846153846153846")
    assert F(q) <= F(3765, 52) < F(q) + F(1, 10 ** 30)    # cut off, not rounded up
    q, exact = divide(Decimal("2"), 3)                    # 0.666…: rounding would end in 7
    assert not exact and str(q).endswith("6")


def test_repeating_per_period_figure_is_marked_not_rounded():
    t = dialogue.handle("Compute my income tax for 2025. I am an employee and my weekly salary is ₱10,001.", "english")
    assert t.outcome == "computation_done"
    assert "₱39,230.14 a year (₱754.42576923… per week)" in t.answer
    assert "repeating decimal" in t.answer
    # The annual contributions are exact even though the monthly figures repeat.
    assert "× 12 = **₱36,401.30** a year" in t.answer
    assert "PhilHealth 2.5% × ₱43,337.66666666… = ₱1,083.44166666…" in t.answer


# ── Inputs are never rounded to fit ──────────────────────────────────────

def test_amount_finer_than_a_centavo_is_asked_again():
    t = dialogue.handle("Compute my income tax for 2025. My taxable income is ₱800,000.555.", "english")
    assert t.outcome == "computation_invalid_input"
    assert "**₱800,000.555**" in t.answer and "won't round" in t.answer
    assert "taxable_income" not in t.state.values


# ── Display shows every digit ────────────────────────────────────────────

def test_amount_and_peso_formatting_never_round():
    assert amount(Decimal("3765")) == "₱3,765.00"
    assert amount(Decimal("868.74975")) == "₱868.74975"
    assert amount(Decimal("0.1375")) == "₱0.1375"
    assert amount(Decimal("-2500.0375")) == "₱-2,500.0375"
    assert amount(Decimal("0.666666666666"), exact=False) == "₱0.66666666…"   # cut off, not ₱0.66666667
    assert amount(Decimal("72.403846153846"), exact=False) == "₱72.40384615…"
    assert peso(Decimal("400000")) == "₱400,000"
    assert peso(Decimal("1234.567")) == "₱1,234.567"


# ── Two input-reading bugs found while testing (fixed in v1.5.2) ─────────

def test_semi_monthly_pay_is_not_read_as_monthly():
    # The period window used to start mid-word ("-monthly"), so ₱17,500
    # semi-monthly was taxed as ₱17,500 a month: ₱0 instead of ₱20,415.00.
    t = dialogue.handle("Compute my income tax for 2025. I am an employee and my semi-monthly salary is ₱17,500.",
                        "english")
    assert t.outcome == "computation_done"
    assert t.state.values["pay_period"] == "semi_monthly"
    assert Decimal(t.result["tax_due"]) == Decimal("20415") and "₱17,500.00 × 24 = ₱420,000.00" in t.answer


def test_self_employed_is_not_mixed_income():
    # "self-employed" used to match the employee pattern ("employed") too, so it
    # was refused as mixed income.
    t = dialogue.handle("I am self-employed. Compute my income tax for 2025 under the 8% option. "
                        "My gross sales are ₱1,000,000.", "english")
    assert t.outcome == "computation_done" and Decimal(t.result["tax_due"]) == Decimal("60000")


def test_calculator_refuses_to_round_silently():
    # The calculator runs with Decimal's Inexact/Rounded signals trapped, so an
    # operation that would drop a digit raises instead of returning a rounded
    # number. (The 300 random salaries above all pass with the traps on.)
    import decimal
    import pytest
    from app.computation.calculator import exact_arithmetic

    @exact_arithmetic
    def one_third():
        return Decimal(1) / Decimal(3)

    with pytest.raises(decimal.Inexact):
        one_third()
