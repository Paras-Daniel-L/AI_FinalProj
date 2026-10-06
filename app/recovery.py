"""
Conversational recovery: what the bot says when it cannot answer, so the
conversation continues instead of stopping at "no answer found".

Every reply here has the same three parts, kept deliberately separate:

  1. An honest status. "I don't have enough information in my documents to
     answer that yet" (or: the question is outside scope / too short / the
     documents found couldn't be verified). Never an answer.
  2. A way forward. A clarifying question for the user's topic, a few example
     questions phrased the way the system can answer them, and, when retrieval
     DID find something, the titles of the closest documents.
  3. What the bot can do: BIR tax questions from its documents, and
     individual income tax computations.

No model call is made and no tax fact is stated: the example questions are
questions (not answers), document titles come from the chunks' own metadata,
and the capability text is fixed. So the recovery path can't hallucinate a
rate or a rule, and it costs nothing.

It replaces the fixed one-line refusal that api._no_answer_response() used to
return. The trace OUTCOMES (no_retrieval, generator_no_answer,
verification_failed, no_index) are unchanged, so rag_eval still counts these
as refusals. RECOVERY_ENABLED=0 restores the old one-line refusal for an
ablation run.
"""

from __future__ import annotations

import os
from typing import List, Optional

from langchain_core.documents import Document

from .prompts import get_no_answer_message
from .textproc import issuance_title

RECOVERY_ENABLED = os.environ.get("RECOVERY_ENABLED", "1").strip() not in ("0", "false", "False", "")

REASON_NO_EVIDENCE = "no_evidence"          # no_retrieval / generator_no_answer
REASON_UNVERIFIED = "unverified"            # verification_failed
REASON_NO_INDEX = "no_index"
REASON_OUT_OF_SCOPE = "out_of_scope"
REASON_VAGUE = "vague"

_STATUS = {
    REASON_NO_EVIDENCE: {
        "english": "I don't have enough information in my documents to answer that yet.",
        "filipino": "Wala pa akong sapat na impormasyon sa aking mga dokumento para masagot iyan.",
        "taglish": "Wala pa akong enough na information sa documents ko para masagot 'yan.",
    },
    REASON_UNVERIFIED: {
        "english": "I found some related documents, but I couldn't confirm an answer from them, so I won't guess.",
        "filipino": "May nahanap akong kaugnay na mga dokumento, pero hindi ko makumpirma ang sagot mula sa mga ito, kaya hindi ako manghuhula.",
        "taglish": "May nahanap akong related documents, pero hindi ko ma-confirm ang sagot from them, kaya hindi ako manghuhula.",
    },
    REASON_NO_INDEX: {
        "english": "My document index isn't available right now, so I can't look that up at the moment.",
        "filipino": "Hindi available ang aking mga dokumento sa ngayon, kaya hindi ko iyan mahahanap sa ngayon.",
        "taglish": "Hindi available ang document index ko ngayon, kaya hindi ko 'yan ma-look up sa ngayon.",
    },
    REASON_OUT_OF_SCOPE: {
        "english": "That looks outside what I can help with. I answer questions about Philippine BIR taxes using my documents.",
        "filipino": "Mukhang labas iyan sa kaya kong sagutin. Sumasagot ako ng mga tanong tungkol sa buwis ng BIR gamit ang aking mga dokumento.",
        "taglish": "Mukhang outside 'yan sa kaya kong i-help. Sumasagot ako ng questions about Philippine BIR taxes gamit ang documents ko.",
    },
    REASON_VAGUE: {
        "english": "I'd like to help. Could you tell me a bit more about what you want to know?",
        "filipino": "Gusto kitang tulungan. Puwede mo bang sabihin nang mas detalyado kung ano ang gusto mong malaman?",
        "taglish": "Gusto kitang i-help. Puwede mo bang i-explain nang mas specific kung ano ang gusto mong malaman?",
    },
}

# One clarifying question per topic (app/intent.detect_topic).
_CLARIFY = {
    "income": {
        "english": "What kind of income is it (salary, business or professional income), and which tax year do you mean?",
        "filipino": "Anong klase ng kita ito (sahod, negosyo o propesyon), at para sa aling taon?",
        "taglish": "Anong klase ng income ito (sahod, business o professional income), at para sa anong tax year?",
    },
    "computation": {
        "english": "Do you want me to compute your income tax, or to explain a rule? If it's a computation, tell me the type of income (salary, business or professional), the amount and the tax year.",
        "filipino": "Gusto mo bang i-compute ko ang iyong income tax, o ipaliwanag ang isang rule? Kung computation, sabihin ang uri ng kita (sahod, negosyo o propesyon), ang halaga at ang taon.",
        "taglish": "Gusto mo bang i-compute ko ang income tax mo, o i-explain ang isang rule? Kung computation, sabihin mo ang type ng income (sahod, business o professional), ang amount at ang tax year.",
    },
    "withholding": {
        "english": "Is this about tax withheld from salary, or from business/professional income? Which year?",
        "filipino": "Tungkol ba ito sa buwis na kinaltas sa sahod, o sa kita sa negosyo/propesyon? Para sa aling taon?",
        "taglish": "About ba ito sa tax na kinaltas sa sahod, o sa business/professional income? Anong year?",
    },
    "vat": {
        "english": "What about VAT: a specific transaction (like digital services or sales), registration, or a particular issuance?",
        "filipino": "Ano tungkol sa VAT: isang partikular na transaksyon (tulad ng digital services o benta), pagpaparehistro, o isang partikular na issuance?",
        "taglish": "Ano about VAT: specific na transaction (like digital services o sales), registration, o isang specific issuance?",
    },
    "corporate": {
        "english": "Is this about the Minimum Corporate Income Tax (MCIT), the regular corporate income tax, or a specific issuance?",
        "filipino": "Tungkol ba ito sa Minimum Corporate Income Tax (MCIT), sa regular na corporate income tax, o sa isang partikular na issuance?",
        "taglish": "About ba ito sa Minimum Corporate Income Tax (MCIT), regular corporate income tax, o specific issuance?",
    },
    "filing": {
        "english": "Which return or form is it, and for which year? For example, the annual income tax return for 2025.",
        "filipino": "Aling return o form ito, at para sa aling taon? Halimbawa, ang annual income tax return para sa 2025.",
        "taglish": "Anong return o form ito, at para sa anong year? Halimbawa, annual income tax return for 2025.",
    },
    "registration": {
        "english": "Is this about registering a business, getting a TIN, or updating registration? Which issuance or year, if you know it?",
        "filipino": "Tungkol ba ito sa pagpaparehistro ng negosyo, pagkuha ng TIN, o pag-update ng rehistro? Aling issuance o taon, kung alam mo?",
        "taglish": "About ba ito sa pag-register ng business, pagkuha ng TIN, o pag-update ng registration? Anong issuance o year, kung alam mo?",
    },
    "invoicing": {
        "english": "Is this about issuing invoices or receipts, e-invoicing, or a specific issuance?",
        "filipino": "Tungkol ba ito sa pag-isyu ng invoice o resibo, e-invoicing, o isang partikular na issuance?",
        "taglish": "About ba ito sa pag-issue ng invoices o receipts, e-invoicing, o specific na issuance?",
    },
    "general": {
        "english": "Which tax or topic is it about, and is there a specific year or BIR issuance (RR, RMC, RMO...) involved?",
        "filipino": "Tungkol saang buwis o paksa ito, at may partikular bang taon o BIR issuance (RR, RMC, RMO...)?",
        "taglish": "Anong tax o topic ito, at may specific bang year o BIR issuance (RR, RMC, RMO...)?",
    },
}

# Example questions, phrased the way the system answers well. Drawn from the
# subjects the knowledge base covers (the BIR income tax FAQ and the issuance
# folders), never from the T-TED evaluation set.
_EXAMPLES = {
    "income": {
        "english": ["Who is not required to file an income tax return?",
                    "What are the allowable deductions from gross income?",
                    "Compute my income tax: taxable income ₱800,000 for 2025"],
        "filipino": ["Sino ang hindi kailangang mag-file ng income tax return?",
                     "Ano ang mga allowable deductions mula sa gross income?",
                     "I-compute ang income tax ko: taxable income ₱800,000 para sa 2025"],
        "taglish": ["Sino ang hindi need mag-file ng income tax return?",
                    "Ano ang allowable deductions from gross income?",
                    "I-compute ang income tax ko: taxable income ₱800,000 for 2025"],
    },
    "computation": {
        "english": ["My salary is ₱35,000 a month. How much is my income tax for 2025?",
                    "Compute my income tax: taxable income ₱800,000 for 2025",
                    "I'm self-employed on the 8% option with gross receipts of ₱1,200,000 in 2024. How much is my tax?",
                    "How is income tax computed for individuals?"],
        "filipino": ["₱35,000 ang sahod ko kada buwan. Magkano ang income tax ko para sa 2025?",
                     "I-compute ang income tax ko: taxable income ₱800,000 para sa 2025",
                     "Self-employed ako na naka-8% option, gross receipts ₱1,200,000 noong 2024. Magkano ang tax ko?",
                     "Paano kinukwenta ang income tax ng isang indibidwal?"],
        "taglish": ["₱35,000 a month ang sahod ko. Magkano ang income tax ko for 2025?",
                    "I-compute ang income tax ko: taxable income ₱800,000 for 2025",
                    "Self-employed ako, 8% option, gross receipts ₱1,200,000 sa 2024. Magkano tax ko?",
                    "Paano kino-compute ang income tax ng individuals?"],
    },
    "withholding": {
        "english": ["What are the withholding rates on business or professional income?",
                    "How is income tax paid through withholding?"],
        "filipino": ["Ano ang withholding rates sa kita mula sa negosyo o propesyon?",
                     "Paano binabayaran ang income tax sa pamamagitan ng withholding?"],
        "taglish": ["Ano ang withholding rates sa business o professional income?",
                    "Paano binabayaran ang income tax through withholding?"],
    },
    "vat": {
        "english": ["Is VAT imposed on digital services like video streaming and online platforms?",
                    "What does a specific RMC say about VAT? (include its number, e.g. RMC No. <number>-<year>)"],
        "filipino": ["May VAT ba sa mga digital services tulad ng video streaming at online platforms?",
                     "Ano ang sinasabi ng isang partikular na RMC tungkol sa VAT? (isama ang numero, hal. RMC No. <numero>-<taon>)"],
        "taglish": ["May VAT ba sa digital services like video streaming at online platforms?",
                    "Ano ang sinasabi ng isang specific na RMC about VAT? (isama ang number, e.g. RMC No. <number>-<year>)"],
    },
    "corporate": {
        "english": ["When does a corporation start to be covered by the MCIT?",
                    "How is the MCIT computed?"],
        "filipino": ["Kailan nagsisimulang saklaw ng MCIT ang isang korporasyon?",
                     "Paano kinukwenta ang MCIT?"],
        "taglish": ["Kailan nagsisimulang covered ng MCIT ang isang corporation?",
                    "Paano kino-compute ang MCIT?"],
    },
    "filing": {
        "english": ["Where do I file a \"no payment\" income tax return?",
                    "Can I pay my income tax in two installments?"],
        "filipino": ["Saan ako magfa-file ng \"no payment\" na income tax return?",
                     "Puwede ko bang bayaran ang income tax nang dalawang hulog?"],
        "taglish": ["Saan ako magfa-file ng \"no payment\" income tax return?",
                    "Puwede ko bang bayaran ang income tax in two installments?"],
    },
}
_EXAMPLES["general"] = {
    lang: [_EXAMPLES["income"][lang][0], _EXAMPLES["vat"][lang][0], _EXAMPLES["computation"][lang][0]]
    for lang in ("english", "filipino", "taglish")
}
for _topic in ("registration", "invoicing"):
    _EXAMPLES[_topic] = {
        "english": ["What does RMC No. <number>-<year> say about this? (the issuance number helps me find it)",
                    _EXAMPLES["income"]["english"][0]],
        "filipino": ["Ano ang sinasabi ng RMC No. <numero>-<taon> tungkol dito? (nakakatulong ang numero ng issuance)",
                     _EXAMPLES["income"]["filipino"][0]],
        "taglish": ["Ano ang sinasabi ng RMC No. <number>-<year> about dito? (makakatulong ang issuance number)",
                    _EXAMPLES["income"]["taglish"][0]],
    }

_LABELS = {
    "try": {"english": "You could ask, for example:", "filipino": "Halimbawa, puwede mong itanong:",
            "taglish": "For example, puwede mong itanong:"},
    "closest": {"english": "These documents came closest, but they don't clearly answer it:",
                "filipino": "Ito ang pinakamalapit na mga dokumento, pero hindi nila malinaw na nasasagot ito:",
                "taglish": "Ito yung pinakamalapit na documents, pero hindi nila clearly nasasagot:"},
    "closest_tip": {"english": "You can ask about one of them directly.",
                    "filipino": "Puwede mong itanong nang direkta ang tungkol sa isa sa mga ito.",
                    "taglish": "Puwede mong i-ask directly about isa sa mga 'yan."},
    "can_do": {
        "english": ("**What I can help with:** BIR tax rules and the revenue issuances in my documents (RR, RMC, RMO, RAO, RDAO "
                    "from {years}), income tax basics from the BIR FAQ, and **computing individual income tax** "
                    "(from your gross salary, your taxable income, or the 8% option) if you give me the amount and the tax year."),
        "filipino": ("**Ang kaya kong itulong:** mga patakaran ng BIR at mga revenue issuance sa aking mga dokumento (RR, RMC, RMO, RAO, RDAO "
                     "mula {years}), mga batayan ng income tax mula sa BIR FAQ, at **pag-compute ng income tax ng indibidwal** "
                     "(mula sa gross na sahod, taxable income, o 8% option) kung ibibigay mo ang halaga at ang taon."),
        "taglish": ("**Kaya kong i-help:** BIR tax rules at revenue issuances sa documents ko (RR, RMC, RMO, RAO, RDAO "
                    "from {years}), income tax basics from the BIR FAQ, at **pag-compute ng individual income tax** "
                    "(from gross salary, taxable income, o 8% option) kung ibibigay mo ang amount at ang tax year."),
    },
    "chat_intro": {
        "english": "I'm Sagot AI, an assistant for Philippine BIR tax questions.",
        "filipino": "Ako si Sagot AI, isang assistant para sa mga tanong tungkol sa buwis ng BIR.",
        "taglish": "Ako si Sagot AI, assistant para sa mga BIR tax questions mo.",
    },
    "chat_close": {
        "english": "What would you like to ask?",
        "filipino": "Ano ang gusto mong itanong?",
        "taglish": "Ano ang gusto mong i-ask?",
    },
    "continue_computation": {
        "english": "Your income tax computation is still open. Send the missing value whenever you're ready, or say \"cancel\".",
        "filipino": "Bukas pa ang iyong income tax computation. Ipadala ang kulang na halaga kapag handa ka na, o sabihin ang \"cancel\".",
        "taglish": "Open pa ang income tax computation mo. I-send mo lang ang kulang na value kapag ready ka na, o sabihin mo \"cancel\".",
    },
}


def _pick(table: dict, lang: str):
    return table.get(lang) or table["english"]


def _corpus_years() -> str:
    from .classifier import KNOWN_YEARS
    years = sorted(KNOWN_YEARS)
    early = [y for y in years if y < "2010"]
    late = [y for y in years if y >= "2010"]
    parts = []
    if early:
        parts.append(f"{early[0]}–{early[-1]}" if len(early) > 1 else early[0])
    if late:
        parts.append(f"{late[0]}–{late[-1]}" if len(late) > 1 else late[0])
    return " and ".join(parts) if parts else "several years"


def capabilities(lang: str) -> str:
    return _pick(_LABELS["can_do"], lang).format(years=_corpus_years())


def closest_documents(docs: Optional[List[Document]], limit: int = 3) -> List[str]:
    """Distinct issuance titles of what retrieval found, best first, with the
    issuance's own subject line when it is short and clean (metadata, not a
    claim made by the bot)."""
    out: List[str] = []
    seen = set()
    for doc in docs or []:
        meta = doc.metadata or {}
        title = meta.get("title") or issuance_title(str(meta.get("source") or ""))
        if not title or title in seen or title == "Unknown source":
            continue
        seen.add(title)
        summary = (meta.get("doc_summary") or "").strip()
        if summary and len(summary) <= 160 and summary.lower() != title.lower():
            out.append(f"{title}: {summary}")
        else:
            out.append(title)
        if len(out) >= limit:
            break
    return out


def guidance(lang: str, reason: str, topic: str = "general", docs: Optional[List[Document]] = None) -> str:
    """The full recovery message (status, way forward, capabilities)."""
    if not RECOVERY_ENABLED:
        return get_no_answer_message(lang)
    topic = topic if topic in _CLARIFY else "general"
    lines = [_pick(_STATUS[reason], lang)]
    if reason != REASON_OUT_OF_SCOPE:
        lines += ["", _pick(_CLARIFY[topic], lang)]

    closest = closest_documents(docs) if reason in (REASON_NO_EVIDENCE, REASON_UNVERIFIED) else []
    if closest:
        lines += ["", _pick(_LABELS["closest"], lang)] + [f"- {c}" for c in closest]
        lines.append(_pick(_LABELS["closest_tip"], lang))

    examples = _pick(_EXAMPLES.get(topic, _EXAMPLES["general"]), lang)
    lines += ["", _pick(_LABELS["try"], lang)] + [f"- *{e}*" for e in examples]
    lines += ["", capabilities(lang)]
    return "\n".join(lines)


def chat_reply(lang: str) -> str:
    """Fixed answer to "what can you do?" / "sino ka?" (CHAT intent)."""
    examples = _pick(_EXAMPLES["general"], lang)
    return "\n".join(
        [_pick(_LABELS["chat_intro"], lang), "", capabilities(lang), "", _pick(_LABELS["try"], lang)]
        + [f"- *{e}*" for e in examples]
        + ["", _pick(_LABELS["chat_close"], lang)]
    )


def continue_computation_note(lang: str) -> str:
    return _pick(_LABELS["continue_computation"], lang)


def mode_for(reason: str) -> str:
    if not RECOVERY_ENABLED:
        return "no_answer"
    return "clarify" if reason == REASON_VAGUE else "guidance"
