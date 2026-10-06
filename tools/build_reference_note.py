"""
Builds data/reference/Employee_Compensation_Tax_Reference.pdf, a knowledge-
base note on income tax on employee compensation, so the RAG path can answer
"are SSS contributions taxable?", "how much 13th-month pay is tax-free?",
"what are the 2026 tax brackets?" and similar questions with citations.

Every statement in the note comes from an official source named in the same
answer (the NIRC as amended by RA 10963, RA 8424, RA 11223, SSS, PhilHealth
and HDMF issuances). The worked example's figures come from the project's own
calculator (app/computation), so the note and the computation always agree.
None of the questions comes from the T-TED evaluation set.

Run from the project root, then index the new file:
    python tools/build_reference_note.py
    python -m app.database
"""

import os
import sys
from decimal import Decimal

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = os.path.join("data", "reference", "Employee_Compensation_Tax_Reference.pdf")


def worked_example():
    """₱35,000 a month in 2025, computed by the project's calculator."""
    from app.computation.calculator import compute_compensation
    from app.computation.rules import load_registry

    reg = load_registry()
    rule, version, _ = reg.lookup("income_tax_compensation", 2025)
    sched, sched_v, _ = reg.lookup("income_tax_graduated", 2025)
    _c, contrib_v, _ = reg.lookup("employee_contributions", 2025)
    r = compute_compensation(rule, version, 2025, {"gross_pay": Decimal("35000")}, "monthly", sched, sched_v, contrib_v)
    auto = next(s for s in r.steps if s["kind"] == "contrib_auto")
    return {
        "annual": r.inputs["annual_pay"], "sss": auto["sss"], "ph": auto["philhealth"], "pi": auto["pagibig"],
        "monthly_contrib": auto["monthly_total"], "annual_contrib": r.inputs["contributions"],
        "taxable": r.inputs["taxable_income"], "tax": r.tax_due, "per_month": r.per_period["tax"],
        "take_home": r.per_period["take_home"],
    }


def p(amount) -> str:
    d = Decimal(str(amount))
    return f"P{int(d):,}" if d == d.to_integral_value() else f"P{d:,.2f}"


def entries():
    ex = worked_example()
    return [
        ("What amounts are excluded from an employee's taxable compensation income?",
         "Employee compensation",
         "Two exclusions apply to employees under the National Internal Revenue Code (NIRC). First, the employee's "
         "own mandatory contributions: GSIS, SSS, Medicare (now PhilHealth) and Pag-ibig contributions, and union dues "
         "(NIRC Sec. 32(B)(7)(f)). Second, 13th month pay and other benefits, up to a combined total of P90,000 a year "
         "(NIRC Sec. 32(B)(7)(e), as amended by RA 10963). Only the part of the 13th month pay and other benefits "
         "above P90,000 is added to taxable compensation."),
        ("How much 13th month pay and other benefits is tax-exempt?",
         "Employee compensation",
         "Under NIRC Sec. 32(B)(7)(e), as amended by RA 10963 (TRAIN Law), gross benefits received by officials and "
         "employees of public and private entities are excluded from gross income, provided that the total exclusion "
         "does not exceed P90,000. The ceiling covers, combined: benefits under RA 6686, benefits under PD 851 (13th "
         "month pay), benefits received by officials and employees not covered by PD 851, and productivity "
         "incentives and Christmas bonuses. Any amount above "
         "P90,000 is taxable compensation."),
        ("Are SSS, PhilHealth and Pag-IBIG contributions deducted before income tax is computed?",
         "Employee compensation",
         "Yes. NIRC Sec. 32(B)(7)(f) excludes from gross income \"GSIS, SSS, Medicare and Pag-ibig contributions, and "
         "union dues of individuals.\" The employee's share of these mandatory contributions is therefore subtracted "
         "from compensation before the graduated income tax rates are applied."),
        ("Can an employee still claim personal or dependent exemptions?",
         "Employee compensation",
         "No. Section 12 of RA 10963 (TRAIN Law) repealed Section 35 of the NIRC, which had provided the personal and "
         "additional (dependent) exemptions. Under the graduated rates in NIRC Sec. 24(A)(2)(a), as amended by "
         "RA 10963, the first P250,000 of annual taxable income is taxed at 0% for every individual, so the number of "
         "dependents no longer changes the income tax."),
        ("What are the graduated income tax rates for individuals for tax years 2023 onwards?",
         "Tax rates",
         "Under NIRC Sec. 24(A)(2)(a), as amended by RA 10963 (schedule effective January 1, 2023 and onwards): "
         "not over P250,000: 0%; over P250,000 but not over P400,000: 15% of the excess over P250,000; over P400,000 "
         "but not over P800,000: P22,500 + 20% of the excess over P400,000; over P800,000 but not over P2,000,000: "
         "P102,500 + 25% of the excess over P800,000; over P2,000,000 but not over P8,000,000: P402,500 + 30% of the "
         "excess over P2,000,000; over P8,000,000: P2,202,500 + 35% of the excess over P8,000,000."),
        ("What were the graduated income tax rates for individuals for tax years 2018 to 2022?",
         "Tax rates",
         "Under NIRC Sec. 24(A)(2)(a), as amended by RA 10963 (schedule effective January 1, 2018 until December 31, "
         "2022): not over P250,000: 0%; over P250,000 but not over P400,000: 20% of the excess over P250,000; over "
         "P400,000 but not over P800,000: P30,000 + 25% of the excess over P400,000; over P800,000 but not over "
         "P2,000,000: P130,000 + 30% of the excess over P800,000; over P2,000,000 but not over P8,000,000: P490,000 + "
         "32% of the excess over P2,000,000; over P8,000,000: P2,410,000 + 35% of the excess over P8,000,000. (The "
         "P8,000,000 bound follows the Senate Tax Study and Research Office TRAIN primer.)"),
        ("How much is the employee's SSS contribution in 2025 and 2026?",
         "Mandatory contributions",
         "Under SSS Circular No. 2024-006 (effective January 2025, pursuant to RA 11199), the SSS contribution rate is "
         "15% of the Monthly Salary Credit (MSC): 10% paid by the employer and 5% by the employee. The MSC ranges from "
         "P5,000 (compensation below P5,250) to P35,000 (compensation of P34,750 and over), in P500 steps. The "
         "employee share therefore ranges from P250 to P1,750 a month. No later SSS contribution schedule has been "
         "issued, so the same schedule applies in 2026."),
        ("How much is the employee's PhilHealth premium in 2025 and 2026?",
         "Mandatory contributions",
         "The PhilHealth premium rate is 5% of the monthly basic salary, with an income floor of P10,000 and an income "
         "ceiling of P100,000 (RA 11223 Sec. 10; PhilHealth Advisory No. 2025-0002 for 2025 and Advisory No. 2026-0042 "
         "for 2026). The premium is shared equally by the employer and the employee (PhilHealth Circular No. "
         "2020-0005, Rev. 1), so the employee pays 2.5% of basic salary: P250 a month at the floor and P2,500 a month "
         "at the ceiling. Allowances, overtime, commissions and bonuses are not part of the monthly basic salary."),
        ("How much is the employee's Pag-IBIG contribution?",
         "Mandatory contributions",
         "Under HDMF Circular No. 460, effective February 2024, the contribution rate is 2% of the Fund Salary and the "
         "maximum Fund Salary is P10,000, so the employee's personal share is at most P200 a month. Source: HDMF Circular No. 460 as implemented in OCA Circular No. 25-2024."),
        ("How is income tax on an employee's salary computed from gross pay?",
         "Employee compensation",
         "Step 1: convert the pay to an annual amount (monthly pay x 12, semi-monthly pay x 24, weekly pay x 52). "
         "Step 2: subtract the employee's mandatory SSS, PhilHealth and Pag-IBIG contributions for the year "
         "(NIRC Sec. 32(B)(7)(f)). Step 3: add only the part of 13th month pay and other benefits above P90,000 "
         "(NIRC Sec. 32(B)(7)(e)). The result is taxable compensation. Step 4: apply the graduated rates for that tax "
         "year (NIRC Sec. 24(A)(2)(a)). Step 5: subtract tax already withheld by the employer (BIR Form 2316) to get "
         "the tax still payable or the excess withheld."),
        ("What is the income tax on a P35,000 monthly salary in 2025?",
         "Worked example",
         f"Annual gross pay: P35,000 x 12 = {p(ex['annual'])}. Monthly employee contributions: SSS 5% of the P35,000 "
         f"MSC = {p(ex['sss'])}; PhilHealth 2.5% of P35,000 = {p(ex['ph'])}; Pag-IBIG 2% of the P10,000 maximum Fund "
         f"Salary = {p(ex['pi'])}; total {p(ex['monthly_contrib'])} a month, or {p(ex['annual_contrib'])} a year. "
         f"Taxable compensation: {p(ex['annual'])} - {p(ex['annual_contrib'])} = {p(ex['taxable'])}, assuming 13th "
         f"month pay and other benefits of P90,000 or less. Tax (2023-onwards table): 15% of the excess over P250,000 "
         f"= {p(ex['tax'])} a year, about {p(ex['per_month'])} a month. Monthly take-home pay: P35,000 - "
         f"{p(ex['monthly_contrib'])} - {p(ex['per_month'])} = {p(ex['take_home'])}."),
        ("Which taxpayers does this note not cover?",
         "Scope",
         "This note covers purely compensation earners (employees). It does not cover minimum wage earners, "
         "self-employed individuals and professionals, mixed-income earners (salary plus business income), "
         "non-resident aliens, or the income of overseas Filipino workers from abroad. For self-employed individuals, "
         "see the 8% option and the graduated rates on net taxable income."),
    ]


def build(path: str = OUT) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    styles = getSampleStyleSheet()
    q_style = ParagraphStyle("Q", parent=styles["Heading3"], spaceBefore=10, spaceAfter=2)
    t_style = ParagraphStyle("T", parent=styles["Normal"], textColor="#555555", fontSize=9, spaceAfter=2)
    a_style = ParagraphStyle("A", parent=styles["Normal"], leading=14)

    doc = SimpleDocTemplate(
        path, pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
        title="Employee Compensation Tax Reference (Philippines)",
        author="Sagot AI project - compiled from official sources",
        subject="Philippine income tax on employee compensation",
    )
    story = [
        Paragraph("Employee Compensation Tax Reference (Philippines)", styles["Title"]),
        Paragraph(
            "This reference note explains how income tax on employee compensation is computed in the Philippines, "
            "with the legal source of every rule. Amounts written with the letter P (for example P90,000) are "
            "Philippine pesos (PHP). NIRC means the National Internal Revenue Code (Tax Code). Compiled on "
            "October 6, 2026 from the NIRC as amended by RA 10963 (TRAIN Law), RA 8424, RA 11223, SSS Circular "
            "No. 2024-006, PhilHealth Circular No. 2020-0005 and Advisories 2025-0002 and 2026-0042, and HDMF "
            "Circular No. 460.", a_style),
        Spacer(1, 6),
    ]
    for i, (question, topic, answer) in enumerate(entries(), start=1):
        story.append(KeepTogether([
            Paragraph(f"Question {i}: {question}", q_style),
            Paragraph(f"Topic: {topic}", t_style),
            Paragraph(f"Answer: {answer}", a_style),
        ]))
    doc.build(story)
    return path


if __name__ == "__main__":
    print(f"Wrote {build()}")
