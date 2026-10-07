"""
Every sentence the computation path can say, in English, Filipino and
Taglish. Fixed templates, no model call: the numbers come from the
calculator and the rule file, the wording from here. That way a computation
reply can never contain a rate, threshold or figure that isn't in a rule file.
"""

from __future__ import annotations

from decimal import ROUND_DOWN, Decimal
from typing import Dict, List, Optional

LANGS = ("english", "filipino", "taglish")

# Decimal places shown for a repeating quotient (e.g. ₱1,000 ÷ 12), cut off
# without rounding and followed by "…". See calculator.divide().
REPEATING_SHOWN = 8


def _digits(d: Decimal, min_decimals: int, strip: bool = True) -> str:
    """Every digit of `d` (no rounding), thousands separators, at least
    `min_decimals` decimals. format(d, "f") writes the exact value."""
    sign = "-" if d < 0 else ""
    whole, _, frac = format(abs(d), "f").partition(".")
    if strip:
        frac = frac.rstrip("0")
    frac = frac.ljust(min_decimals, "0")
    return f"{sign}{int(whole):,}" + (f".{frac}" if frac else "")


def peso(value) -> str:
    """A figure from the law or a rule file, or an amount the user typed, as
    written: ₱400,000 / ₱1,234.50. Never rounded: every decimal is shown."""
    return "₱" + _digits(Decimal(str(value)), 0)


def amount(value, exact: bool = True) -> str:
    """
    A COMPUTED figure, never rounded: always shown to at least the centavo
    and with every further digit the arithmetic produced (₱3,765.00,
    ₱868.74975). exact=False marks a repeating decimal from dividing an annual
    figure by the pay periods: its first REPEATING_SHOWN decimals are shown,
    cut off (not rounded), then "…" (₱72.40384615…).
    """
    d = Decimal(str(value))
    if exact:
        return "₱" + _digits(d, 2)
    d = d.quantize(Decimal(1).scaleb(-REPEATING_SHOWN), rounding=ROUND_DOWN)
    return "₱" + _digits(d, REPEATING_SHOWN, strip=False) + "…"


def _ex(step: dict, key: str) -> bool:
    """Whether a step's value is exact (False only for a repeating decimal)."""
    return (step.get("exact") or {}).get(key, True)


def pct(rate) -> str:
    r = Decimal(str(rate)) * 100
    return f"{r.normalize():f}%" if r != r.to_integral_value() else f"{int(r)}%"


def t(key: str, lang: str, **kw) -> str:
    table = _T[key]
    template = table.get(lang) or table["english"]
    return template.format(**kw) if kw else template


# Field names as shown in questions and acknowledgements.
FIELD = {
    "tax_year": {"english": "Tax year", "filipino": "Taon ng buwis (tax year)", "taglish": "Tax year"},
    "taxable_income": {"english": "Annual taxable income", "filipino": "Taunang taxable income",
                       "taglish": "Annual taxable income"},
    "gross_sales_receipts": {"english": "Gross sales/receipts for the year",
                             "filipino": "Kabuuang gross sales/receipts sa taon",
                             "taglish": "Gross sales/receipts for the year"},
    "non_operating_income": {"english": "Other non-operating income", "filipino": "Iba pang non-operating income",
                             "taglish": "Other non-operating income"},
    "tax_withheld": {"english": "Tax already withheld", "filipino": "Buwis na na-withhold na",
                     "taglish": "Tax na na-withhold na"},
    "regime": {"english": "Tax option (graduated rates or 8%)", "filipino": "Tax option (graduated o 8%)",
               "taglish": "Tax option (graduated o 8%)"},
    "gross_pay": {"english": "Gross pay (before deductions)", "filipino": "Gross na sahod (bago ang mga kaltas)",
                  "taglish": "Gross pay mo (before deductions)"},
    "pay_period": {"english": "Pay period (monthly, semi-monthly, weekly or annual)",
                   "filipino": "Pay period (buwanan, kinsenas, lingguhan o taunan)",
                   "taglish": "Pay period (monthly, semi-monthly, weekly o annual)"},
    "benefits": {"english": "13th-month pay and other benefits for the year",
                 "filipino": "13th-month pay at iba pang benepisyo sa taon",
                 "taglish": "13th-month pay at other benefits for the year"},
    "contributions": {"english": "Total SSS, PhilHealth and Pag-IBIG employee contributions for the year",
                      "filipino": "Kabuuang kontribusyon mo sa SSS, PhilHealth at Pag-IBIG sa taon",
                      "taglish": "Total SSS, PhilHealth at Pag-IBIG contributions mo for the year"},
    "annual_pay": {"english": "Annual gross pay", "filipino": "Taunang gross na sahod", "taglish": "Annual gross pay"},
    "taxable_compensation": {"english": "Taxable compensation", "filipino": "Taxable na sahod",
                             "taglish": "Taxable compensation"},
}
FIELD_EXAMPLE = {"tax_year": "2025", "taxable_income": "₱800,000", "gross_sales_receipts": "₱1,500,000",
                 "non_operating_income": "₱0", "tax_withheld": "₱90,000", "regime": "8%",
                 "gross_pay": "₱35,000 a month", "pay_period": "monthly", "benefits": "₱35,000",
                 "contributions": "₱33,900"}

PERIOD = {
    "monthly": {"english": "month", "filipino": "buwan", "taglish": "month"},
    "semi_monthly": {"english": "half-month (semi-monthly)", "filipino": "kinsenas", "taglish": "kinsenas (semi-monthly)"},
    "weekly": {"english": "week", "filipino": "linggo", "taglish": "week"},
    "annual": {"english": "year", "filipino": "taon", "taglish": "year"},
}


def period_name(period: str, lang: str) -> str:
    return PERIOD.get(period, {}).get(lang) or PERIOD.get(period, {}).get("english") or period

UNSUPPORTED_TAX = {
    "vat": {"english": "VAT", "filipino": "VAT", "taglish": "VAT"},
    "estate": {"english": "estate tax", "filipino": "estate tax", "taglish": "estate tax"},
    "donor": {"english": "donor's tax", "filipino": "donor's tax", "taglish": "donor's tax"},
    "corporate": {"english": "corporate income tax / MCIT", "filipino": "corporate income tax / MCIT",
                  "taglish": "corporate income tax / MCIT"},
    "percentage": {"english": "percentage tax", "filipino": "percentage tax", "taglish": "percentage tax"},
    "capital_gains": {"english": "capital gains tax", "filipino": "capital gains tax", "taglish": "capital gains tax"},
    "dst": {"english": "documentary stamp tax", "filipino": "documentary stamp tax", "taglish": "documentary stamp tax"},
    "excise": {"english": "excise tax", "filipino": "excise tax", "taglish": "excise tax"},
    "real_property": {"english": "real property tax", "filipino": "real property tax (amilyar)",
                      "taglish": "real property tax (amilyar)"},
    "minimum_wage": {"english": "income tax for minimum wage earners",
                     "filipino": "income tax ng minimum wage earner",
                     "taglish": "income tax for minimum wage earners"},
    "mixed_income": {"english": "income tax for mixed-income earners (salary + business)",
                     "filipino": "income tax ng mixed-income earner (sahod + negosyo)",
                     "taglish": "income tax for mixed-income earners (sahod + business)"},
}

_T: Dict[str, Dict[str, str]] = {
    # ── Asking for inputs ─────────────────────────────────────────────
    "ask_intro": {
        "english": "Sure, I can calculate that. I just need a few details first:",
        "filipino": "Sige, kaya kong i-compute iyan. Kailangan ko lang muna ang mga sumusunod:",
        "taglish": "Sure, kaya kong i-compute 'yan. Kailangan ko lang muna ng ilang details:",
    },
    "ask_one": {
        "english": "Got it. One more thing: what is your **{field}**? (e.g. {example})",
        "filipino": "Sige. Isa pa: ano ang iyong **{field}**? (hal. {example})",
        "taglish": "Got it. Isa pa: ano ang **{field}** mo? (e.g. {example})",
    },
    "ask_year_one": {
        "english": "Got it. Which **tax year** should I use? (e.g. 2025) The rates depend on the year, so I won't assume one.",
        "filipino": "Sige. Anong **taon ng buwis (tax year)** ang gagamitin ko? (hal. 2025) Nakadepende sa taon ang rates, kaya hindi ako manghuhula.",
        "taglish": "Got it. Anong **tax year** ang gagamitin ko? (e.g. 2025) Depende sa year ang rates, kaya hindi ako mag-a-assume.",
    },
    "hint_8pct": {
        "english": "If you're purely self-employed and chose the 8% option, tell me your gross sales/receipts instead.",
        "filipino": "Kung purely self-employed ka at pinili mo ang 8% option, ibigay mo na lang ang iyong gross sales/receipts.",
        "taglish": "Kung purely self-employed ka at nag-8% option ka, ibigay mo na lang yung gross sales/receipts mo.",
    },
    "have_so_far": {
        "english": "So far I have:",
        "filipino": "Ito na ang nakuha ko:",
        "taglish": "Ito na yung meron ako:",
    },
    "ask_regime": {
        "english": ("Since you're self-employed, which option are you using for {year_text}?\n"
                    "- **Graduated rates**: tell me your annual taxable income (after deductions)\n"
                    "- **8% option**: tell me your gross sales/receipts"),
        "filipino": ("Dahil self-employed ka, aling option ang gamit mo para sa {year_text}?\n"
                     "- **Graduated rates**: ibigay ang taunang taxable income (pagkatapos ng deductions)\n"
                     "- **8% option**: ibigay ang gross sales/receipts"),
        "taglish": ("Since self-employed ka, anong option ang gamit mo for {year_text}?\n"
                    "- **Graduated rates**: ibigay ang annual taxable income mo (after deductions)\n"
                    "- **8% option**: ibigay ang gross sales/receipts mo"),
    },
    "year_text_unknown": {"english": "that year", "filipino": "taong iyon", "taglish": "that year"},
    "gross_not_taxable": {
        "english": ("You gave {amount} as a **gross** amount (before deductions). The graduated rates apply to **taxable income** "
                    "(gross income minus allowable deductions), which I can't work out for you. What is your annual taxable income? "
                    "If you elected the 8% option instead, just say \"8%\"."),
        "filipino": ("Ibinigay mo ang {amount} bilang **gross** na halaga (bago ang deductions). Ang graduated rates ay para sa **taxable income** "
                     "(gross income bawas ang allowable deductions), na hindi ko kayang tantyahin para sa iyo. Magkano ang iyong taunang taxable income? "
                     "Kung 8% option ang pinili mo, sabihin lang ang \"8%\"."),
        "taglish": ("Ang binigay mo na {amount} ay **gross** amount (before deductions). Ang graduated rates ay para sa **taxable income** "
                    "(gross income minus allowable deductions), na hindi ko ma-compute para sa'yo. Magkano ang annual taxable income mo? "
                    "Kung 8% option ang pinili mo, sabihin mo lang \"8%\"."),
    },
    # ── Validation ───────────────────────────────────────────────────
    "invalid_amount": {
        "english": "I couldn't read **{raw}** as a valid peso amount. Please type it like **800,000**, **800000** or **800k** (no negative values).",
        "filipino": "Hindi ko mabasa ang **{raw}** bilang tamang halaga. Pakisulat ito tulad ng **800,000**, **800000** o **800k** (walang negatibong halaga).",
        "taglish": "Hindi ko ma-read ang **{raw}** as a valid na amount. Pakitype like **800,000**, **800000** o **800k** (walang negative values).",
    },
    "sub_centavo_amount": {
        "english": "**{raw}** has digits smaller than a centavo. Peso amounts go down to the centavo (2 decimal places), and I won't round your figure. What is the exact amount (e.g. **800,000.55**)?",
        "filipino": "May decimal na mas maliit sa sentimo ang **{raw}**. Hanggang sentimo (2 decimal places) lang ang halaga ng piso, at hindi ko ira-round ang halagang ibinigay mo. Ano ang eksaktong halaga (hal. **800,000.55**)?",
        "taglish": "May digits na mas maliit sa centavo ang **{raw}**. Hanggang centavo (2 decimal places) lang ang peso amounts, at hindi ko ira-round ang figure mo. Ano ang exact na amount (e.g. **800,000.55**)?",
    },
    "need_number": {
        "english": "I need a number for your **{field}**, e.g. **{example}**. What is it?",
        "filipino": "Kailangan ko ng numero para sa iyong **{field}**, hal. **{example}**. Magkano ito?",
        "taglish": "Kailangan ko ng number para sa **{field}** mo, e.g. **{example}**. Magkano ito?",
    },
    "monthly_amount": {
        "english": ("{amount} looks like a **monthly** amount. Income tax uses the **annual** figure, and annual taxable income "
                    "isn't always 12 × monthly pay (13th-month pay, other benefits and deductions change it). What is your annual taxable income?"),
        "filipino": ("Mukhang **buwanang** halaga ang {amount}. Ang income tax ay batay sa **taunang** halaga, at hindi palaging 12 × buwanang sahod "
                     "ang taunang taxable income (dahil sa 13th-month pay, iba pang benepisyo at deductions). Magkano ang iyong taunang taxable income?"),
        "taglish": ("Mukhang **monthly** amount ang {amount}. Annual ang basis ng income tax, at hindi laging 12 × monthly pay ang annual taxable income "
                    "(dahil sa 13th-month pay, other benefits at deductions). Magkano ang annual taxable income mo?"),
    },
    "conflicting_amounts": {
        "english": "You mentioned more than one amount for your **{field}** ({values}). Which one should I use?",
        "filipino": "Higit sa isang halaga ang binanggit mo para sa **{field}** ({values}). Alin ang gagamitin ko?",
        "taglish": "More than one amount ang binigay mo for **{field}** ({values}). Alin ang gagamitin ko?",
    },
    "conflicting_years": {
        "english": "You mentioned more than one year ({values}). Which **tax year** should I compute for?",
        "filipino": "Higit sa isang taon ang binanggit mo ({values}). Para sa aling **tax year** ako mag-compute?",
        "taglish": "More than one year ang binanggit mo ({values}). Para sa anong **tax year** ako mag-compute?",
    },
    # ── Refusals that keep the conversation going ───────────────────
    "unsupported_year": {
        "english": ("I don't have a verified rule for **{tax_name}** in **{year}**, so I won't compute it with another year's rates. "
                    "I can compute tax years **{lo} to {hi}**. Which of those should I use?"),
        "filipino": ("Wala akong verified na rule para sa **{tax_name}** sa taong **{year}**, kaya hindi ko ito ico-compute gamit ang rates ng ibang taon. "
                     "Kaya kong mag-compute para sa taong **{lo} hanggang {hi}**. Alin doon ang gagamitin ko?"),
        "taglish": ("Wala akong verified na rule for **{tax_name}** sa **{year}**, kaya hindi ko 'to ico-compute gamit ang rates ng ibang year. "
                    "Kaya kong mag-compute for tax years **{lo} to {hi}**. Alin doon ang gagamitin ko?"),
    },
    "no_rule": {
        "english": ("I can't compute **{tax_name}** yet: no validated computation rule for it has been loaded, and I won't make up rates. "
                    "Right now I can compute **individual income tax**: for employees from their gross salary, from annual taxable income, or the 8% option for purely self-employed individuals. "
                    "I can still answer questions about {tax_name} from my documents. Just ask."),
        "filipino": ("Hindi ko pa kayang i-compute ang **{tax_name}**: wala pang validated na computation rule para dito, at hindi ako mag-iimbento ng rates. "
                     "Sa ngayon, kaya kong i-compute ang **income tax ng indibidwal**: para sa empleyado mula sa gross na sahod, mula sa taunang taxable income, o ang 8% option para sa purely self-employed. "
                     "Pero puwede mo pa rin akong tanungin tungkol sa {tax_name} mula sa aking mga dokumento."),
        "taglish": ("Hindi ko pa ma-compute ang **{tax_name}**: wala pang validated na computation rule for it, at hindi ako mag-iimbento ng rates. "
                    "Sa ngayon, kaya kong i-compute ang **individual income tax**: for employees from gross salary, from annual taxable income, o 8% option for purely self-employed. "
                    "Pero puwede mo pa rin akong tanungin about {tax_name} from my documents."),
    },
    "unverified": {
        "english": "The rule for **{tax_name}** in **{year}** hasn't been verified yet, so I won't use it for a computation.",
        "filipino": "Hindi pa verified ang rule para sa **{tax_name}** sa **{year}**, kaya hindi ko ito gagamitin sa computation.",
        "taglish": "Hindi pa verified ang rule for **{tax_name}** sa **{year}**, kaya hindi ko 'to gagamitin sa computation.",
    },
    "not_eligible_8pct": {
        "english": ("The 8% option is only for purely self-employed individuals whose gross sales/receipts plus other non-operating income "
                    "**do not exceed {limit}** (the VAT threshold). Yours add up to **{gross}**, so the 8% option doesn't apply and I won't compute it. "
                    "I can compute it with the **graduated rates** instead. What is your annual taxable income (after deductions)?"),
        "filipino": ("Ang 8% option ay para lang sa purely self-employed na ang gross sales/receipts at iba pang non-operating income ay "
                     "**hindi lalampas sa {limit}** (ang VAT threshold). Ang sa iyo ay **{gross}**, kaya hindi puwede ang 8% option at hindi ko ito ico-compute. "
                     "Puwede kong gamitin ang **graduated rates**. Magkano ang iyong taunang taxable income (pagkatapos ng deductions)?"),
        "taglish": ("Ang 8% option ay para lang sa purely self-employed na ang gross sales/receipts plus other non-operating income ay "
                    "**hindi lalampas sa {limit}** (VAT threshold). Ang total mo ay **{gross}**, kaya hindi applicable ang 8% option at hindi ko 'to ico-compute. "
                    "Puwede kong gamitin ang **graduated rates** instead. Magkano ang annual taxable income mo (after deductions)?"),
    },
    "cancelled": {
        "english": "Okay, I've cancelled that computation. What else can I help you with?",
        "filipino": "Sige, kinansela ko na ang computation. Ano pa ang maitutulong ko?",
        "taglish": "Okay, cancelled na ang computation. Ano pa ang maitutulong ko?",
    },
    "updated": {
        "english": "Updated your {field}: {old} → **{new}**.",
        "filipino": "In-update ko ang iyong {field}: {old} → **{new}**.",
        "taglish": "In-update ko ang {field} mo: {old} → **{new}**.",
    },
    # ── Result ──────────────────────────────────────────────────────
    "result_title": {
        "english": "**Income tax due for {year}: {tax_due}**",
        "filipino": "**Income tax due para sa {year}: {tax_due}**",
        "taglish": "**Income tax due for {year}: {tax_due}**",
    },
    "label_rule": {"english": "Rule applied", "filipino": "Rule na ginamit", "taglish": "Rule na ginamit"},
    "label_basis": {"english": "Legal basis", "filipino": "Batayang legal", "taglish": "Legal basis"},
    "label_bracket": {"english": "Bracket", "filipino": "Bracket", "taglish": "Bracket"},
    "label_calc": {"english": "**Calculation**", "filipino": "**Kompyutasyon**", "taglish": "**Computation**"},
    "bracket_text": {
        "english": "over {over}{upper}: {base} + {rate} of the excess over {over}",
        "filipino": "lampas {over}{upper}: {base} + {rate} ng labis sa {over}",
        "taglish": "over {over}{upper}: {base} + {rate} ng excess over {over}",
    },
    "bracket_upper": {"english": " but not over {not_over}", "filipino": " pero hindi lalampas sa {not_over}",
                      "taglish": " pero hindi lalampas sa {not_over}"},
    "bracket_first": {
        "english": "not over {not_over}: 0%",
        "filipino": "hindi lalampas sa {not_over}: 0%",
        "taglish": "not over {not_over}: 0%",
    },
    "step_zero": {
        "english": "{income} is not over {not_over}, so the rate is 0% and the tax due is **₱0.00**.",
        "filipino": "Ang {income} ay hindi lumalampas sa {not_over}, kaya 0% ang rate at **₱0.00** ang tax due.",
        "taglish": "Ang {income} ay hindi lalampas sa {not_over}, kaya 0% ang rate at **₱0.00** ang tax due.",
    },
    "step_excess": {
        "english": "Excess over {over}: {income} − {over} = {excess}",
        "filipino": "Labis sa {over}: {income} − {over} = {excess}",
        "taglish": "Excess over {over}: {income} − {over} = {excess}",
    },
    "step_rate": {
        "english": "{rate} × {excess} = {result}",
        "filipino": "{rate} × {excess} = {result}",
        "taglish": "{rate} × {excess} = {result}",
    },
    "step_add_base": {
        "english": "Tax due: {base} + {rate_part} = **{tax_due}**",
        "filipino": "Tax due: {base} + {rate_part} = **{tax_due}**",
        "taglish": "Tax due: {base} + {rate_part} = **{tax_due}**",
    },
    "step_gross_total": {
        "english": "Gross sales/receipts {sales} + other non-operating income {other} = {gross}",
        "filipino": "Gross sales/receipts {sales} + iba pang non-operating income {other} = {gross}",
        "taglish": "Gross sales/receipts {sales} + other non-operating income {other} = {gross}",
    },
    "step_less_threshold": {
        "english": "Less the {deduction} allowed by law: {gross} − {deduction} = {net} (not below ₱0)",
        "filipino": "Bawas ang {deduction} na pinapayagan ng batas: {gross} − {deduction} = {net} (hindi bababa sa ₱0)",
        "taglish": "Less {deduction} na allowed by law: {gross} − {deduction} = {net} (hindi bababa sa ₱0)",
    },
    "step_flat_rate": {
        "english": "Tax due: {rate} × {net} = **{tax_due}**",
        "filipino": "Tax due: {rate} × {net} = **{tax_due}**",
        "taglish": "Tax due: {rate} × {net} = **{tax_due}**",
    },
    "step_withheld_payable": {
        "english": "Less tax already withheld: {tax_due} − {withheld} = **{payable} still payable**",
        "filipino": "Bawas ang buwis na na-withhold na: {tax_due} − {withheld} = **{payable} na babayaran pa**",
        "taglish": "Less tax na na-withhold na: {tax_due} − {withheld} = **{payable} na babayaran pa**",
    },
    "step_withheld_over": {
        "english": "Less tax already withheld: {tax_due} − {withheld} = **{excess} withheld in excess** (possibly refundable or creditable; ask me how excess withholding is treated)",
        "filipino": "Bawas ang buwis na na-withhold na: {tax_due} − {withheld} = **{excess} na sobrang na-withhold** (maaaring ma-refund o ma-credit; puwede mo akong tanungin kung paano ito tinatrato)",
        "taglish": "Less tax na na-withhold na: {tax_due} − {withheld} = **{excess} na sobra ang na-withhold** (possibly refundable o creditable; tanungin mo ako kung paano ito tina-treat)",
    },
    "step_withheld_zero": {
        "english": "Less tax already withheld: {tax_due} − {withheld} = **₱0.00 still payable**",
        "filipino": "Bawas ang buwis na na-withhold na: {tax_due} − {withheld} = **₱0.00 na babayaran pa**",
        "taglish": "Less tax na na-withhold na: {tax_due} − {withheld} = **₱0.00 na babayaran pa**",
    },
    "note_income_assumed": {
        "english": "I treated {amount} as your **annual taxable income**, meaning after deductions and non-taxable items (such as 13th-month pay and other benefits up to ₱90,000, and GSIS, SSS, Medicare and other contributions). If it's your gross pay, tell me your taxable income instead.",
        "filipino": "Itinuring ko ang {amount} bilang iyong **taunang taxable income**, ibig sabihin pagkatapos ng deductions at non-taxable items (tulad ng 13th-month pay at iba pang benepisyo hanggang ₱90,000, at mga kontribusyon sa GSIS, SSS, Medicare at iba pa). Kung gross pay mo ito, ibigay mo na lang ang iyong taxable income.",
        "taglish": "Tinreat ko ang {amount} as **annual taxable income** mo, ibig sabihin after deductions at non-taxable items (like 13th-month pay at other benefits up to ₱90,000, at GSIS, SSS, Medicare at other contributions). Kung gross pay mo 'to, ibigay mo na lang ang taxable income mo.",
    },
    "note_gross_assumed": {
        "english": "I treated {amount} as your **gross sales/receipts** for the year.",
        "filipino": "Itinuring ko ang {amount} bilang iyong **gross sales/receipts** sa taon.",
        "taglish": "Tinreat ko ang {amount} as **gross sales/receipts** mo for the year.",
    },
    "note_8pct_scope": {
        "english": "The 8% option applies only if you are **purely** self-employed (no salary) and elected it for the year; the 8% replaces both the graduated income tax and the percentage tax.",
        "filipino": "Ang 8% option ay para lang kung **purely** self-employed ka (walang sahod) at pinili mo ito para sa taon; pinapalitan ng 8% ang graduated income tax at percentage tax.",
        "taglish": "Applicable lang ang 8% option kung **purely** self-employed ka (walang sahod) at in-elect mo ito for the year; pinapalitan ng 8% ang graduated income tax at percentage tax.",
    },
    # ── Employee (gross pay) mode ───────────────────────────────────
    "ask_period_one": {
        "english": "Is {amount} your **monthly**, **semi-monthly**, **weekly** or **annual** gross pay?",
        "filipino": "Ang {amount} ba ay iyong **buwanan**, **kinsenas**, **lingguhan** o **taunang** gross na sahod?",
        "taglish": "Ang {amount} ba ay **monthly**, **semi-monthly**, **weekly** o **annual** gross pay mo?",
    },
    "daily_pay": {
        "english": ("I can't annualize daily pay reliably, because the number of paid days in a year differs between employers. "
                    "What is your **monthly** (or annual) gross pay?"),
        "filipino": ("Hindi ko maaasahang gawing taunan ang arawang sahod, dahil iba-iba ang bilang ng bayad na araw sa isang taon depende sa employer. "
                     "Magkano ang iyong **buwanang** (o taunang) gross na sahod?"),
        "taglish": ("Hindi ko ma-annualize nang tama ang daily pay, kasi iba-iba ang number of paid days per year depende sa employer. "
                    "Magkano ang **monthly** (o annual) gross pay mo?"),
    },
    "ask_contributions_no_table": {
        "english": ("I only have verified SSS, PhilHealth and Pag-IBIG schedules for **{lo} to {hi}**, so I won't estimate your {year} contributions "
                    "with another year's rates. What were your **total employee contributions for {year}** (SSS + PhilHealth + Pag-IBIG)? "
                    "Your payslips or BIR Form 2316 show them; a monthly figure is fine too (e.g. \"2,825 a month\")."),
        "filipino": ("May verified na SSS, PhilHealth at Pag-IBIG schedule lang ako para sa **{lo} hanggang {hi}**, kaya hindi ko tatantyahin ang kontribusyon mo sa {year} "
                     "gamit ang rates ng ibang taon. Magkano ang **kabuuang kontribusyon mo noong {year}** (SSS + PhilHealth + Pag-IBIG)? "
                     "Makikita ito sa payslip o sa BIR Form 2316; puwede rin ang buwanang halaga (hal. \"2,825 kada buwan\")."),
        "taglish": ("Verified SSS, PhilHealth at Pag-IBIG schedules lang ang meron ako for **{lo} to {hi}**, kaya hindi ko i-e-estimate ang {year} contributions mo "
                    "gamit ang rates ng ibang year. Magkano ang **total employee contributions mo for {year}** (SSS + PhilHealth + Pag-IBIG)? "
                    "Nasa payslip o BIR Form 2316 mo 'yan; okay din ang monthly (e.g. \"2,825 a month\")."),
    },
    "ask_contributions_no_rule": {
        "english": "I don't have a verified contribution schedule loaded. What were your **total SSS, PhilHealth and Pag-IBIG employee contributions for {year}**?",
        "filipino": "Wala akong verified na contribution schedule. Magkano ang **kabuuang kontribusyon mo sa SSS, PhilHealth at Pag-IBIG noong {year}**?",
        "taglish": "Wala akong verified na contribution schedule. Magkano ang **total SSS, PhilHealth at Pag-IBIG contributions mo for {year}**?",
    },
    "ask_contributions_low_pay": {
        "english": "At this pay level I can't compute the contributions from the schedules I have. What were your **total SSS, PhilHealth and Pag-IBIG employee contributions for the year**? (See your payslips.)",
        "filipino": "Sa ganitong halaga ng sahod, hindi ko makukwenta ang kontribusyon gamit ang mga schedule na meron ako. Magkano ang **kabuuang kontribusyon mo sa SSS, PhilHealth at Pag-IBIG sa taon**? (Tingnan ang payslip.)",
        "taglish": "Sa ganitong pay level, hindi ko ma-compute ang contributions from my schedules. Magkano ang **total SSS, PhilHealth at Pag-IBIG contributions mo for the year**? (Check your payslips.)",
    },
    "result_title_period": {
        "english": "**Income tax for {year}: {tax_due} a year ({per} per {period})**",
        "filipino": "**Income tax para sa {year}: {tax_due} sa isang taon ({per} kada {period})**",
        "taglish": "**Income tax for {year}: {tax_due} a year ({per} per {period})**",
    },
    "label_table": {"english": "Tax table", "filipino": "Tax table", "taglish": "Tax table"},
    "label_gross_pay": {"english": "Gross pay", "filipino": "Gross na sahod", "taglish": "Gross pay"},
    "per": {"english": "per", "filipino": "kada", "taglish": "per"},
    "step_annualize": {
        "english": "Annual gross pay: {gross} × {periods} = {annual}",
        "filipino": "Taunang gross na sahod: {gross} × {periods} = {annual}",
        "taglish": "Annual gross pay: {gross} × {periods} = {annual}",
    },
    "step_contrib_auto": {
        "english": ("Employee contributions on {monthly} a month: SSS {sss_rate} × {msc} salary credit = {sss}; "
                    "PhilHealth {ph_rate} × {ph_base} = {ph}; Pag-IBIG {pi_rate} × {pi_base} = {pi}; "
                    "total {monthly_total} a month × 12 = **{annual}** a year"),
        "filipino": ("Kontribusyon ng empleyado sa {monthly} kada buwan: SSS {sss_rate} × {msc} salary credit = {sss}; "
                     "PhilHealth {ph_rate} × {ph_base} = {ph}; Pag-IBIG {pi_rate} × {pi_base} = {pi}; "
                     "kabuuang {monthly_total} kada buwan × 12 = **{annual}** sa isang taon"),
        "taglish": ("Employee contributions sa {monthly} a month: SSS {sss_rate} × {msc} salary credit = {sss}; "
                    "PhilHealth {ph_rate} × {ph_base} = {ph}; Pag-IBIG {pi_rate} × {pi_base} = {pi}; "
                    "total {monthly_total} a month × 12 = **{annual}** a year"),
    },
    "step_contrib_manual": {
        "english": "Mandatory contributions you gave (for the year): **{annual}**",
        "filipino": "Kontribusyong ibinigay mo (para sa taon): **{annual}**",
        "taglish": "Contributions na binigay mo (for the year): **{annual}**",
    },
    "step_benefits": {
        "english": "13th-month pay and other benefits {benefits}: the first {cap} is tax-exempt, so **{taxable}** is taxable",
        "filipino": "13th-month pay at iba pang benepisyo {benefits}: libre sa buwis ang unang {cap}, kaya **{taxable}** ang taxable",
        "taglish": "13th-month pay at other benefits {benefits}: tax-exempt ang first {cap}, kaya **{taxable}** ang taxable",
    },
    "step_taxable_comp": {
        "english": "Taxable compensation: {annual} − {contributions}{plus_benefits} = **{taxable}**",
        "filipino": "Taxable na sahod: {annual} − {contributions}{plus_benefits} = **{taxable}**",
        "taglish": "Taxable compensation: {annual} − {contributions}{plus_benefits} = **{taxable}**",
    },
    "step_per_period": {
        "english": "Per {period}: tax {tax_due} ÷ {periods} = {tax}; take-home = {gross} − {contributions} contributions − {tax} tax = **{take_home}**",
        "filipino": "Kada {period}: buwis {tax_due} ÷ {periods} = {tax}; take-home = {gross} − {contributions} kontribusyon − {tax} buwis = **{take_home}**",
        "taglish": "Per {period}: tax {tax_due} ÷ {periods} = {tax}; take-home = {gross} − {contributions} contributions − {tax} tax = **{take_home}**",
    },
    "note_repeating": {
        "english": "Dividing by {periods} doesn't come out even here, so a figure marked \"…\" is a repeating decimal: it is shown to 8 decimal places and cut off there, not rounded. The annual figures are exact.",
        "filipino": "Hindi pantay ang hatian sa {periods} dito, kaya ang halagang may \"…\" ay umuulit na decimal: ipinapakita ito hanggang 8 decimal places at pinuputol doon, hindi ira-round. Eksakto ang mga taunang halaga.",
        "taglish": "Hindi pantay ang division sa {periods} dito, kaya ang figure na may \"…\" ay repeating decimal: naka-show ito up to 8 decimal places at pinutol doon, hindi ni-round. Exact ang annual figures.",
    },
    "note_benefits_assumed": {
        "english": "I assumed your 13th-month pay and other benefits for the year total {cap} or less, which is tax-exempt. If they're higher, tell me the total (e.g. \"13th month 120,000\") and I'll recompute.",
        "filipino": "Inakala kong {cap} o mas mababa ang kabuuan ng 13th-month pay at iba pang benepisyo mo sa taon, na libre sa buwis. Kung mas mataas, sabihin ang kabuuan (hal. \"13th month 120,000\") at ico-compute ko ulit.",
        "taglish": "In-assume ko na {cap} o less ang total ng 13th-month pay at other benefits mo for the year, na tax-exempt. Kung mas mataas, sabihin mo ang total (e.g. \"13th month 120,000\") at ire-recompute ko.",
    },
    "note_contributions_auto": {
        "english": "Contributions were computed from the official schedules for {year}. Your payslip can differ slightly; for example, PhilHealth is based on basic salary only. If you know your actual annual total, tell me and I'll use it.",
        "filipino": "Kinwenta ang kontribusyon mula sa opisyal na schedule para sa {year}. Maaaring bahagyang iba ang nasa payslip mo; halimbawa, basic salary lang ang batayan ng PhilHealth. Kung alam mo ang aktwal na kabuuan sa taon, sabihin mo at iyon ang gagamitin ko.",
        "taglish": "Computed ang contributions from the official schedules for {year}. Puwedeng medyo iba sa payslip mo; halimbawa, basic salary lang ang basis ng PhilHealth. Kung alam mo ang actual annual total, sabihin mo at 'yon ang gagamitin ko.",
    },
    "note_per_period": {
        "english": "The per-period tax is the annual tax spread evenly. Actual payslip withholding can differ, for example after a pay change or the employer's year-end adjustment.",
        "filipino": "Ang buwis kada pay period ay ang taunang buwis na hinati nang pantay. Maaaring iba ang aktwal na kaltas sa payslip, halimbawa kapag nagbago ang sahod o sa year-end adjustment ng employer.",
        "taglish": "Ang tax per pay period ay ang annual tax na hinati nang pantay. Puwedeng iba ang actual withholding sa payslip, halimbawa kapag nagbago ang sahod o sa year-end adjustment ng employer.",
    },
    "note_employee_scope": {
        "english": "This covers purely compensation income. Minimum wage earners, and anyone who also has business or freelance income, need a different computation.",
        "filipino": "Para lang ito sa kita mula sa sahod. Iba ang kompyutasyon para sa minimum wage earner at sa may kita rin mula sa negosyo o freelance.",
        "taglish": "Para lang ito sa purely compensation income. Iba ang computation for minimum wage earners at sa may business o freelance income din.",
    },
    "disclaimer": {
        "english": "_Every figure above is computed exactly from the amounts you gave and the rule shown above, with no rounding. It doesn't cover penalties, other income types or special cases, so check your actual return or ask a BIR officer or tax professional before filing._",
        "filipino": "_Eksaktong kinompyut ang bawat halaga sa itaas mula sa mga halagang ibinigay mo at sa rule na nakasaad sa itaas, nang walang pag-round. Hindi kasama ang penalties, ibang uri ng kita o mga espesyal na kaso, kaya i-check ang iyong aktwal na return o magtanong sa BIR o sa isang tax professional bago mag-file._",
        "taglish": "_Exact na computed ang bawat figure sa itaas from the amounts na binigay mo at sa rule sa itaas, walang rounding. Hindi kasama ang penalties, other income types o special cases, kaya i-check ang actual return mo o magtanong sa BIR o tax professional bago mag-file._",
    },
    "can_update": {
        "english": "You can change any value (e.g. \"actually it's ₱850,000\" or \"use 2024\") and I'll recompute.",
        "filipino": "Puwede mong baguhin ang kahit anong halaga (hal. \"850,000 pala\" o \"2024 na lang\") at ico-compute ko ulit.",
        "taglish": "Puwede mong palitan ang kahit anong value (e.g. \"850k pala\" o \"use 2024\") at ire-recompute ko.",
    },
}


def field_name(key: str, lang: str) -> str:
    return FIELD.get(key, {}).get(lang) or FIELD.get(key, {}).get("english") or key


def tax_display(key: str, lang: str) -> str:
    return UNSUPPORTED_TAX.get(key, {}).get(lang) or key


def format_value(key: str, value: str, lang: str = "english") -> str:
    if key == "tax_year":
        return str(value)
    if key == "regime":
        return "8%" if value == "8_percent" else "graduated"
    if key == "pay_period":
        return {"monthly": "monthly", "semi_monthly": "semi-monthly", "weekly": "weekly", "annual": "annual"}.get(value, value)
    if key == "kind":
        return value
    return peso(value)


def collected_lines(values: Dict[str, str], lang: str) -> List[str]:
    order = ["tax_year", "gross_pay", "pay_period", "benefits", "contributions", "taxable_income",
             "gross_sales_receipts", "non_operating_income", "tax_withheld", "regime"]
    return [f"- {field_name(k, lang)}: {format_value(k, values[k], lang)}" for k in order if k in values]


def ask_for(missing: List[str], values: Dict[str, str], lang: str, first_turn: bool, hint_8pct: bool,
            tax_type: Optional[str] = None) -> str:
    """Grouped request on the first turn (or when several fields are missing),
    a single focused question afterwards."""
    if len(missing) == 1 and not first_turn:
        if missing[0] == "tax_year":
            return t("ask_year_one", lang)
        return t("ask_one", lang, field=field_name(missing[0], lang), example=FIELD_EXAMPLE[missing[0]])
    lines = [t("ask_intro", lang)]
    shown = [k for k in missing if not (k == "pay_period" and "gross_pay" in missing)]  # "₱35,000 a month" covers both
    lines += [f"- **{field_name(k, lang)}** ({'e.g.' if lang != 'filipino' else 'hal.'} {FIELD_EXAMPLE[k]})" for k in shown]
    have = collected_lines(values, lang)
    if have:
        lines += ["", t("have_so_far", lang)] + have
    if hint_8pct:
        lines += ["", t("hint_8pct", lang)]
    return "\n".join(lines)


def render_result(result, rule, version, lang: str, notes: List[str], changes: List[str],
                  schedule=None, schedule_version=None, pay_period: Optional[str] = None) -> str:
    """The full breakdown: headline, rule + legal basis, steps, notes."""
    out: List[str] = []
    out += changes
    if changes:
        out.append("")
    if result.per_period:
        pp = result.per_period
        out.append(t("result_title_period", lang, year=result.tax_year, tax_due=amount(result.tax_due),
                     per=amount(pp["tax"], (pp.get("exact") or {}).get("tax", True)),
                     period=period_name(pp["period"], lang)))
    else:
        out.append(t("result_title", lang, year=result.tax_year, tax_due=amount(result.tax_due)))
    out.append("")
    inputs = result.inputs
    out.append(f"- {field_name('tax_year', lang)}: {result.tax_year}")
    if "gross_pay" in inputs:
        per = f" {t('per', lang)} {period_name(pay_period, lang)}" if pay_period and pay_period != "annual" else ""
        out.append(f"- {t('label_gross_pay', lang)}: {amount(inputs['gross_pay'])}{per}")
        if inputs.get("benefits"):
            out.append(f"- {field_name('benefits', lang)}: {amount(inputs['benefits'])}")
        out.append(f"- {field_name('taxable_compensation', lang)}: {amount(inputs['taxable_income'])}")
    else:
        for key in ("taxable_income", "gross_sales_receipts", "non_operating_income"):
            if key in inputs and not (key == "non_operating_income" and inputs[key] == 0):
                out.append(f"- {field_name(key, lang)}: {amount(inputs[key])}")
    if result.tax_withheld is not None:
        out.append(f"- {field_name('tax_withheld', lang)}: {amount(result.tax_withheld)}")
    out.append(f"- {t('label_rule', lang)}: {rule.name_for(lang)} ({version.label_for(lang)})")
    out.append(f"- {t('label_basis', lang)}: {version.legal_basis}")
    if schedule is not None and schedule_version is not None:
        out.append(f"- {t('label_table', lang)}: {schedule.name_for(lang)} ({schedule_version.label_for(lang)})")
    if result.bracket is not None:
        b = result.bracket
        if b.over == 0 and b.rate == 0:
            text = t("bracket_first", lang, not_over=peso(b.not_over))
        else:
            upper = t("bracket_upper", lang, not_over=peso(b.not_over)) if b.not_over is not None else ""
            text = t("bracket_text", lang, over=peso(b.over), upper=upper, base=peso(b.base_tax), rate=pct(b.rate))
        out.append(f"- {t('label_bracket', lang)}: {text}")

    out += ["", t("label_calc", lang)]
    n = 0
    for s in result.steps:
        kind = s["kind"]
        line: Optional[str] = None
        # Law/rule constants (bracket bounds, base tax, caps, salary credit) use
        # peso(); every computed figure and every input uses amount(): exact,
        # to the centavo or finer, never rounded.
        if kind == "zero_bracket":
            line = t("step_zero", lang, income=amount(s["income"]), not_over=peso(s["not_over"]))
        elif kind == "excess":
            line = t("step_excess", lang, over=peso(s["over"]), income=amount(s["income"]), excess=amount(s["excess"]))
        elif kind == "rate_times_excess":
            line = t("step_rate", lang, rate=pct(s["rate"]), excess=amount(s["excess"]), result=amount(s["result"]))
        elif kind == "add_base":
            line = t("step_add_base", lang, base=peso(s["base_tax"]), rate_part=amount(s["rate_part"]),
                     tax_due=amount(s["tax_due"]))
        elif kind == "gross_total":
            line = t("step_gross_total", lang, sales=amount(s["sales"]), other=amount(s["other"]), gross=amount(s["gross"]))
        elif kind == "less_threshold":
            line = t("step_less_threshold", lang, gross=amount(s["gross"]), deduction=peso(s["deduction"]),
                     net=amount(s["net"]))
        elif kind == "flat_rate":
            line = t("step_flat_rate", lang, rate=pct(s["rate"]), net=amount(s["net"]), tax_due=amount(s["tax_due"]))
        elif kind == "annualize":
            line = t("step_annualize", lang, gross=amount(s["gross"]), periods=s["periods"], annual=amount(s["annual"]))
        elif kind == "contrib_auto":
            line = t("step_contrib_auto", lang, monthly=amount(s["monthly"], _ex(s, "monthly")),
                     sss_rate=pct(s["sss_rate"]), msc=peso(s["msc"]), sss=amount(s["sss"], _ex(s, "sss")),
                     ph_rate=pct(s["ph_rate"]), ph_base=amount(s["ph_base"], _ex(s, "ph_base")),
                     ph=amount(s["philhealth"], _ex(s, "philhealth")), pi_rate=pct(s["pi_rate"]),
                     pi_base=amount(s["pi_base"], _ex(s, "pi_base")), pi=amount(s["pagibig"], _ex(s, "pagibig")),
                     monthly_total=amount(s["monthly_total"], _ex(s, "monthly_total")), annual=amount(s["annual"]))
        elif kind == "contrib_manual":
            line = t("step_contrib_manual", lang, annual=amount(s["annual"]))
        elif kind == "benefits":
            line = t("step_benefits", lang, benefits=amount(s["benefits"]), cap=peso(s["cap"]), taxable=amount(s["taxable"]))
        elif kind == "taxable_compensation":
            plus = f" + {amount(s['taxable_benefits'])}" if Decimal(str(s["taxable_benefits"])) > 0 else ""
            line = t("step_taxable_comp", lang, annual=amount(s["annual"]), contributions=amount(s["contributions"]),
                     plus_benefits=plus, taxable=amount(s["taxable"]))
        elif kind == "per_period":
            line = t("step_per_period", lang, period=period_name(s["period"], lang), tax_due=amount(s["tax_due"]),
                     periods=s["periods"], tax=amount(s["tax"], _ex(s, "tax")), gross=amount(s["gross"]),
                     contributions=amount(s["contributions"], _ex(s, "contributions")),
                     take_home=amount(s["take_home"], _ex(s, "take_home")))
        elif kind == "less_withheld":
            payable = Decimal(str(s["payable"]))
            args = dict(tax_due=amount(s["tax_due"]), withheld=amount(s["withheld"]))
            if payable > 0:
                line = t("step_withheld_payable", lang, payable=amount(payable), **args)
            elif payable < 0:
                line = t("step_withheld_over", lang, excess=amount(-payable), **args)
            else:
                line = t("step_withheld_zero", lang, **args)
        if line:
            n += 1
            out.append(f"{n}. {line}")

    if notes:
        out.append("")
        out += [f"_{x}_" if not x.startswith("_") else x for x in notes]
    out += ["", t("can_update", lang), "", t("disclaimer", lang)]
    return "\n".join(out)
