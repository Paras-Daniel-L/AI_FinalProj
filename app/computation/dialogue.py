"""
The multi-turn computation conversation.

    user message ─► extract() ─► merge into state (newest value wins)
                                     │
             cancel / no rule / invalid / conflicting / unsupported year
                                     │            └─► say so, keep the conversation open
                         missing inputs? ─► ask for exactly those
                                     │
                  Registry.lookup(tax_type, year) ─► the ONE verified rule version for that year
                                     │
                       calculator ─► breakdown + legal basis + sources

Three computations:
  income_tax_graduated     from annual TAXABLE income
  income_tax_8_percent     purely self-employed, from gross sales/receipts
  income_tax_compensation  employees, from GROSS pay: annualize, exclude the
                           mandatory SSS/PhilHealth/Pag-IBIG employee shares
                           (computed from the verified schedule for that
                           year, or the user's own annual total), add 13th-
                           month pay and benefits above ₱90,000, then the
                           graduated table for the same year

State lives with the client: every computation reply returns the state, and
the browser sends it back with the next message (`computation_state`). The
server re-validates it on every turn (schemas.ComputationState), so a
tampered or stale state can at worst cause a question to be asked again; it
can never inject a rate, because rates only come from tax_rules/.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Callable, Dict, List, Optional

from ..schemas import ComputationState
from . import messages as M
from .calculator import NotEligible, centavos, compute, compute_compensation
from .extract import Extraction, extract
from .rules import Registry, get_registry

GRADUATED = "income_tax_graduated"
EIGHT_PCT = "income_tax_8_percent"
COMPENSATION = "income_tax_compensation"
CONTRIBUTIONS = "employee_contributions"

PRIMARY_AMOUNT = {GRADUATED: "taxable_income", EIGHT_PCT: "gross_sales_receipts", COMPENSATION: "gross_pay"}
AMOUNT_SLOTS = ("taxable_income", "gross_sales_receipts", "non_operating_income", "tax_withheld",
                "gross_pay", "benefits", "contributions")
PERIODIC = ("monthly", "semi_monthly", "weekly", "daily")
COMPENSATION_ROLES = {"gross_compensation", "benefits", "contributions"}

# Trace outcomes of this path (alongside the RAG outcomes in api.run_query).
OUT_DONE = "computation_done"
OUT_NEEDS_INPUT = "computation_needs_input"
OUT_INVALID = "computation_invalid_input"
OUT_UNSUPPORTED_YEAR = "computation_unsupported_year"
OUT_NO_RULE = "computation_no_rule"
OUT_NOT_ELIGIBLE = "computation_not_eligible"
OUT_UNVERIFIED = "computation_rule_unverified"
OUT_CANCELLED = "computation_cancelled"


@dataclass
class Turn:
    answer: str
    mode: str                                   # "computation" (a result) | "computation_input" (anything else)
    outcome: str
    state: Optional[ComputationState]
    classification: str = "Tax Computation"
    sources: List[str] = field(default_factory=list)
    result: Optional[Dict] = None               # calculator output, for the trace


KbLookup = Callable[[str, int], Optional[str]]


def _classification(rule_name: str = "", year: Optional[str] = None) -> str:
    bits = [b for b in (rule_name, year) if b]
    return "Tax Computation" + (f" ({', '.join(bits)})" if bits else "")


def _slot_for(role: str, tax_type: Optional[str], awaiting: List[str]) -> tuple:
    """(slot, assumed?) for an extracted amount."""
    if tax_type == COMPENSATION:
        if role in ("tax_withheld", "benefits", "contributions"):
            return role, False
        if role in ("gross_compensation", "gross_sales_receipts"):
            return "gross_pay", False
    if role in ("tax_withheld", "non_operating_income", "gross_sales_receipts", "taxable_income"):
        return role, False
    waiting_amount = [s for s in awaiting if s in AMOUNT_SLOTS]
    if role == "unlabeled" and waiting_amount:
        return waiting_amount[0], False          # a direct answer to the question we asked
    slot = PRIMARY_AMOUNT.get(tax_type or GRADUATED, "taxable_income")
    if tax_type == COMPENSATION:
        return slot, False                       # pay is pay: no taxable-income assumption is made
    return slot, True                            # "income"/bare number: we assume what it is, and say so


def _choose_tax_type(ex: Extraction, values: Dict[str, str], current: Optional[str]) -> Optional[str]:
    """Which computation the conversation is now about (newest statement wins)."""
    roles = {a.role for a in ex.amounts}
    self_employed = values.get("kind") == "self_employed"
    periodic_income = any(a.role in ("income", "unlabeled") and (a.period or ex.period) in PERIODIC
                          for a in ex.amounts)
    employee_signal = bool(roles & COMPENSATION_ROLES) or (periodic_income and not self_employed and current != EIGHT_PCT)

    if ex.regime == "8_percent":
        return EIGHT_PCT
    if "taxable_income" in roles:
        return GRADUATED                         # the user already knows their taxable income
    if employee_signal:
        values.pop("regime", None)
        return COMPENSATION
    if ex.regime == "graduated":
        return COMPENSATION if current == COMPENSATION else GRADUATED
    if current is None:
        if values.get("regime") == "8_percent":
            return EIGHT_PCT
        if values.get("regime") == "graduated":
            return GRADUATED
        if self_employed:
            return None                          # self-employed: ask which option (graduated vs 8%)
        if values.get("kind") == "employee":
            return COMPENSATION                  # "the tax on my salary": ask for gross pay
        return GRADUATED                         # taxable income, or nothing yet
    return current


def _rule_sources(rule, version) -> List[str]:
    out = [f"Rule: {rule.name_for('english')}, {version.label_for('english')}. {version.legal_basis}"]
    out += [f"Reference: {s.get('title')} ({s.get('url')})" for s in version.sources]
    return out


def handle(
    query: str,
    lang: str,
    state: Optional[ComputationState] = None,
    kb_lookup: Optional[KbLookup] = None,
    registry: Optional[Registry] = None,
) -> Turn:
    reg = registry or get_registry()
    ex: Extraction = extract(query)
    st = state.model_copy(deep=True) if state else ComputationState(language=lang)
    first_turn = state is None
    values: Dict[str, str] = dict(st.values)
    notes = set(st.notes)

    def reply(text: str, outcome: str, keep: bool = True, mode: str = "computation_input",
              classification: str = "", **extra) -> Turn:
        new_state = None
        if keep:
            st.values, st.notes = values, sorted(notes)
            new_state = st
        return Turn(text, mode, outcome, new_state, classification or _classification(), **extra)

    # 1. Cancel, or a tax we have no rule for.
    if ex.cancel and not ex.amounts and not ex.years:
        return reply(M.t("cancelled", lang), OUT_CANCELLED, keep=False)
    unsupported = ex.unsupported_tax or ("mixed_income" if ex.kind == "mixed" else None)
    if unsupported:
        return reply(M.t("no_rule", lang, tax_name=M.tax_display(unsupported, lang)), OUT_NO_RULE, keep=False)

    # 2. Malformed input is never used; ask for it again.
    if ex.invalid_amounts:
        return reply(M.t("invalid_amount", lang, raw=ex.invalid_amounts[0]), OUT_INVALID)
    if len(ex.years) > 1:
        return reply(M.t("conflicting_years", lang, values=", ".join(map(str, ex.years))), OUT_INVALID)

    # 3. Situation, regime and which computation this is (newest statement wins).
    if ex.regime:
        values["regime"] = ex.regime
    if ex.kind:
        values["kind"] = ex.kind
    tax_type = _choose_tax_type(ex, values, st.tax_type)
    if tax_type != st.tax_type and st.tax_type is not None:
        st.awaiting = []                         # a different computation asks its own questions

    # 4. Amounts -> slots. Periodic figures, daily pay and contradictions are asked about, not guessed.
    assigned: Dict[str, List[Decimal]] = {}
    assumed: Dict[str, bool] = {}
    pay_period_from_amount: Optional[str] = None
    for a in ex.amounts:
        period = a.period or (ex.period if len(ex.amounts) == 1 else None)
        slot, is_assumed = _slot_for(a.role, tax_type, st.awaiting)
        # Amounts are used exactly as typed. One finer than a centavo
        # (800,000.555) is asked about again, never rounded to fit.
        value = centavos(a.value)
        if value is None:
            return reply(M.t("sub_centavo_amount", lang, raw=a.raw), OUT_INVALID)
        if tax_type == COMPENSATION and slot == "gross_pay":
            if period == "daily":
                return reply(M.t("daily_pay", lang), OUT_NEEDS_INPUT)
            if period:
                pay_period_from_amount = period
        elif tax_type == COMPENSATION and slot == "contributions" and period in PERIODIC:
            if period == "daily":
                return reply(M.t("daily_pay", lang), OUT_NEEDS_INPUT)
            value = value * {"monthly": 12, "semi_monthly": 24, "weekly": 52}[period]   # exact, no rounding
        elif tax_type != COMPENSATION and period in PERIODIC and slot in ("taxable_income", "gross_sales_receipts"):
            return reply(M.t("monthly_amount", lang, amount=M.peso(a.value)), OUT_INVALID)
        # Correcting a value the user already stated explicitly ("actually it's
        # 850k") is not a new assumption.
        prior_note = "assumed_taxable" if slot == "taxable_income" else "assumed_gross"
        if slot in values and prior_note not in notes:
            is_assumed = False
        assigned.setdefault(slot, []).append(value)
        assumed[slot] = is_assumed and a.role in ("income", "unlabeled")
    for slot, vals in assigned.items():
        if len(set(vals)) > 1:
            shown = ", ".join(M.peso(v) for v in dict.fromkeys(vals))
            return reply(M.t("conflicting_amounts", lang, field=M.field_name(slot, lang), values=shown), OUT_INVALID)

    changes: List[str] = []

    def set_value(slot: str, new: str) -> None:
        old = values.get(slot)
        if old is not None and old != new:
            changes.append(M.t("updated", lang, field=M.field_name(slot, lang).lower(),
                               old=M.format_value(slot, old, lang), new=M.format_value(slot, new, lang)))
        values[slot] = new

    for slot, vals in assigned.items():
        set_value(slot, str(vals[0]))
        note = "assumed_taxable" if slot == "taxable_income" else "assumed_gross" if slot == "gross_sales_receipts" else None
        if note:
            if assumed.get(slot):
                notes.add(note)
            else:
                notes.discard(note)
    if ex.years:
        set_value("tax_year", str(ex.years[0]))
    if tax_type == COMPENSATION:
        period = pay_period_from_amount or (ex.period if (not ex.amounts or "pay_period" in st.awaiting) else None)
        if period == "daily":
            return reply(M.t("daily_pay", lang), OUT_NEEDS_INPUT)
        if period:
            set_value("pay_period", period)

    # A reply that carried nothing usable, to a question we asked.
    if not first_turn and not ex.has_values and st.awaiting:
        want = st.awaiting[0]
        if want == "tax_year":
            return reply(M.t("ask_year_one", lang), OUT_NEEDS_INPUT)
        if want == "pay_period" and "gross_pay" in values:
            return reply(M.t("ask_period_one", lang, amount=M.peso(values["gross_pay"])), OUT_NEEDS_INPUT)
        if want in AMOUNT_SLOTS:
            return reply(M.t("need_number", lang, field=M.field_name(want, lang), example=M.FIELD_EXAMPLE[want]), OUT_NEEDS_INPUT)

    prefix = "\n".join(changes) + ("\n\n" if changes else "")
    st.tax_type = tax_type

    # 5. Which option? (self-employed without a stated regime)
    if tax_type is None:
        year_text = values.get("tax_year") or M.t("year_text_unknown", lang)
        st.awaiting = ["regime"]
        return reply(prefix + M.t("ask_regime", lang, year_text=year_text), OUT_NEEDS_INPUT)

    rule = reg.get(tax_type)
    if rule is None:
        return reply(M.t("no_rule", lang, tax_name=tax_type.replace("_", " ")), OUT_NO_RULE, keep=False)
    rule_name = rule.name_for(lang)
    label = _classification(rule.name_for("english"), values.get("tax_year"))

    # 6. The year decides the rule. Never borrow another year's rule.
    version = None
    if "tax_year" in values:
        year = int(values["tax_year"])
        rule, version, status = reg.lookup(tax_type, year)
        if status == "unsupported_year":
            lo, hi = rule.supported_years
            values.pop("tax_year")
            st.awaiting = ["tax_year"]
            return reply(prefix + M.t("unsupported_year", lang, tax_name=rule_name, year=year, lo=lo, hi=hi),
                         OUT_UNSUPPORTED_YEAR, classification=_classification(rule.name_for("english")))
        if status == "unverified":
            return reply(M.t("unverified", lang, tax_name=rule_name, year=year), OUT_UNVERIFIED, keep=False)

    # 7. Missing inputs: ask for exactly the ones this rule needs.
    missing = [k for k in rule.required_inputs if k not in values]
    if tax_type == GRADUATED and "taxable_income" in missing and "gross_sales_receipts" in values:
        st.awaiting = ["taxable_income"] + [k for k in missing if k != "taxable_income"]
        return reply(prefix + M.t("gross_not_taxable", lang, amount=M.peso(values["gross_sales_receipts"])),
                     OUT_NEEDS_INPUT, classification=label)
    if tax_type == COMPENSATION and missing == ["pay_period"]:
        st.awaiting = ["pay_period"]
        return reply(prefix + M.t("ask_period_one", lang, amount=M.peso(values["gross_pay"])), OUT_NEEDS_INPUT,
                     classification=label)

    # Employee mode: contributions come from the verified schedule for that
    # year, or from the user when there is none (or the pay is too low for it).
    contrib_rule = contrib_version = None
    if tax_type == COMPENSATION and "tax_year" in values and "contributions" not in values:
        contrib_rule, contrib_version, c_status = reg.lookup(rule.uses.get("contributions", CONTRIBUTIONS), int(values["tax_year"]))
        too_low = False
        if c_status == "ok" and "gross_pay" in values and "pay_period" in values:
            periods = version.params.get(f"periods_{values['pay_period']}") if version else None
            if periods:
                monthly = Decimal(values["gross_pay"]) * periods / 12
                too_low = monthly <= contrib_version.params["pagibig_auto_min_monthly_compensation"]
        if c_status != "ok" or too_low:
            contrib_version = None
            if not missing:
                st.awaiting = ["contributions"]
                if too_low:
                    text = M.t("ask_contributions_low_pay", lang)
                else:
                    lo, hi = contrib_rule.supported_years if contrib_rule else (None, None)
                    key = "ask_contributions_no_table" if contrib_rule else "ask_contributions_no_rule"
                    text = M.t(key, lang, year=values["tax_year"], lo=lo, hi=hi)
                return reply(prefix + text, OUT_NEEDS_INPUT, classification=label)

    if missing:
        st.awaiting = missing
        st.status = "collecting"
        hint = tax_type == GRADUATED and "taxable_income" in missing and values.get("kind") != "employee"
        return reply(prefix + M.ask_for(missing, values, lang, first_turn, hint_8pct=hint, tax_type=tax_type),
                     OUT_NEEDS_INPUT, classification=label)

    # 8. Compute with the validated values and the one rule version for that year.
    year = int(values["tax_year"])
    inputs = {k: Decimal(values[k]) for k in AMOUNT_SLOTS if k in values}
    sources: List[str] = []
    schedule = schedule_version = None
    try:
        if tax_type == COMPENSATION:
            schedule, schedule_version, s_status = reg.lookup(rule.uses.get("schedule", GRADUATED), year)
            if s_status != "ok":
                lo, hi = (schedule.supported_years if schedule else (None, None))
                values.pop("tax_year")
                st.awaiting = ["tax_year"]
                return reply(prefix + M.t("unsupported_year", lang, tax_name=rule_name, year=year, lo=lo, hi=hi),
                             OUT_UNSUPPORTED_YEAR, classification=label)
            result = compute_compensation(rule, version, year, inputs, values["pay_period"],
                                          schedule, schedule_version, contrib_version)
        else:
            result = compute(rule, version, year, inputs)
    except NotEligible as e:
        values.pop("regime", None)
        st.tax_type = GRADUATED
        st.awaiting = ["taxable_income"]
        text = M.t("not_eligible_8pct", lang, limit=M.peso(e.details["limit"]), gross=M.peso(e.details["gross"]))
        return reply(prefix + text, OUT_NOT_ELIGIBLE, classification=_classification(rule.name_for("english"), str(year)))

    note_lines = []
    if tax_type == GRADUATED and "assumed_taxable" in notes:
        note_lines.append(M.t("note_income_assumed", lang, amount=M.peso(inputs["taxable_income"])))
    if tax_type == EIGHT_PCT:
        if "assumed_gross" in notes:
            note_lines.append(M.t("note_gross_assumed", lang, amount=M.peso(inputs["gross_sales_receipts"])))
        note_lines.append(M.t("note_8pct_scope", lang))
    if result.per_period and not all(result.per_period["exact"].values()):
        note_lines.append(M.t("note_repeating", lang, periods=result.per_period["periods"]))
    if tax_type == COMPENSATION:
        if "benefits" not in values:
            note_lines.append(M.t("note_benefits_assumed", lang, cap=M.peso(version.params["benefits_exclusion_cap"])))
        if contrib_version is not None:
            note_lines.append(M.t("note_contributions_auto", lang, year=year))
        if result.per_period:
            note_lines.append(M.t("note_per_period", lang))
        note_lines.append(M.t("note_employee_scope", lang))

    sources += _rule_sources(rule, version)
    if schedule is not None:
        sources += _rule_sources(schedule, schedule_version)
    if contrib_version is not None:
        sources += _rule_sources(contrib_rule, contrib_version)
    kb_refs = list(rule.kb_references) + (list(schedule.kb_references) if schedule is not None else [])
    for ref in kb_refs:
        kb = kb_lookup(ref.get("title", ""), int(ref.get("page", 0))) if kb_lookup else None
        if kb:
            sources.append(f"Knowledge base: {kb}. {ref.get('what', '')}".strip())

    st.status, st.awaiting = "completed", []
    text = M.render_result(result, rule, version, lang, note_lines, changes,
                           schedule=schedule, schedule_version=schedule_version,
                           pay_period=values.get("pay_period"))
    return reply(text, OUT_DONE, mode="computation",
                 classification=_classification(rule.name_for("english"), str(year)),
                 sources=sources, result={**result.to_dict(), "rule_version": version.version_id})
