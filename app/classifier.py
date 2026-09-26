import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline

# --- Class Mapping ---
# No chit-chat class: greetings are intercepted before classification by
# app/greetings.py's rule-based check (see api.py), and general/open-domain
# chat has no home in this system at all anymore. Every class below is a
# real BIR tax bucket, so the classifier's whole training budget goes to
# the thing it actually needs to be good at: telling these years apart.
CLASS_NAMES = {
    0: "BIR Tax Query (Source Year: 2001)",
    1: "BIR Tax Query (Source Year: 2002)",
    2: "BIR Tax Query (Source Year: 2003)",
    3: "BIR Tax Query (Source Year: 2022)",
    4: "BIR Tax Query (Source Year: 2023)",
    5: "BIR Tax Query (Source Year: 2024)",
    6: "BIR Tax Query (Source Year: 2025)",
    7: "BIR Tax Query (Source Year: 2026)",
    8: "FAQ",
}

# --- Training Data ---
TRAINING_QUERIES = [
    # Class 0: BIR Tax 2001
    "RULING  NO.  1-2001",
    "paano mag file ng ITR for 2001",
    "may changes ba sa income tax return ko nung 2001",
    "2001 tax deadline",
    "2001 bir circular",

    # Class 1: BIR Tax 2002
    "saan makikita ang 2002 tax table",
    "update sa vat nitong 2002",
    "RULING NO. 2-2002",
    "ano ang bagong batas sa tax ngayong 2002",
    "2002 bir revenue regulation",

    # Class 2: BIR Tax 2003
    "REVENUE BULLETIN NO. 2-2003",
    "Ano ba ang nakasulat sa Revenue Bulletin No. 2-2003?",
    "Paano i-compute ang estate tax under RR 2-2003 or bulletin 2003?",
    "Active pa ba yung legal guidelines ng Revenue Bulletin nung 2003?",
    "Saan makakakuha ng kopya ng BIR Revenue Bulletin 2-2003?",

    # Class 3: BIR Tax 2022
    "REVENUE DELEGATION AUTHORITY ORDER NO. 1-2022",
    "Ano ang nakasaad sa RDAO No. 1-2022 ng BIR?",
    "Paano i-apply yung Revenue Delegation Authority Order nung 2022?",
    "Sino ang authorized mag-sign under RDAO 1-2022?",
    "May circular ba nung 2022 tungkol sa signature authorities?",

    # Class 4: BIR Tax 2023
    # (RMO 14-2023, RMC 11-2023, RMC 18-2023, RDAO 2-2023, RDAO 4-2023)
    "REVENUE MEMORANDUM ORDER NO. 14-2023",
    "Ano ang Online Citizen/Client Satisfaction Survey (CCSS) ng BIR nung 2023?",
    "Kailan ipapasa ng RDO ang monthly Summary Report on Total Number of Transactions per critical service under RMO 14-2023?",
    "REVENUE MEMORANDUM CIRCULAR NO. 11-2023",
    "Ano ang theme ng BIR Data Privacy Month 2023?",
    "REVENUE MEMORANDUM CIRCULAR NO. 18-2023",
    "Pwede bang mag 100% work from home ang IT-BPM registered business enterprises ayon sa RMC 18-2023?",
    "Ano ang FIRB Administrative Order 001-2023 tungkol sa BOI registration ng RBEs?",
    "REVENUE DELEGATION AUTHORITY ORDER NO. 2-2023",
    "Sino ang binigyan ng authority mag-sign ng Certificate of Acceptance para sa BIR Digital Transformation consultant nung 2023?",
    "REVENUE DELEGATION AUTHORITY ORDER NO. 4-2023",
    "Sino ang OIC-Assistant Regional Director ng RR 7A Quezon City under RDAO 4-2023?",

    # Class 5: BIR Tax 2024
    # (RMC 3-2024, RR 8-2024, RMC 1-2024, RMC 34-2024)
    "REVENUE MEMORANDUM CIRCULAR NO. 3-2024",
    "Ano ang nakasaad sa RMC 3-2024 tungkol sa EOPT Act o RA 11976?",
    "Anong mga section ng NIRC ang binago ng Ease of Paying Taxes Act nung 2024?",
    "Vetoed ba ng Presidente ang withholding tax exemption ng micro-enterprises sa EOPT Act 2024?",
    "REVENUE REGULATIONS NO. 8-2024",
    "Ano ang pagkakaiba ng micro, small, medium at large taxpayer under RR 8-2024?",
    "Magkano ang gross sales para maging micro taxpayer ayon sa 2024 revenue regulations?",
    "REVENUE MEMORANDUM CIRCULAR NO. 1-2024",
    "Pwede pa bang gamitin ng mga NGA ang eTRA para sa payment ng penalties nung 2024?",
    "REVENUE MEMORANDUM CIRCULAR NO. 34-2024",
    "Anong mga gamot ang VAT-exempt sa updated FDA list ng 2024?",
    "Kasama ba ang gamot sa cancer, hypertension at mental illness sa VAT exemption under RMC 34-2024?",

    # Class 6: BIR Tax 2025
    # (RAO 1-2025, RAO 2-2025, RDAO 22-2025)
    "REVENUE ADMINISTRATIVE ORDER NO. 1-2025",
    "Ano ang bagong pangalan ng Revenue Region No. 12 Bacolod City ayon sa RAO 1-2025?",
    "Anong mga probinsya ang sakop ng RR 12 Negros Island Region?",
    "REVENUE ADMINISTRATIVE ORDER NO. 2-2025",
    "Ano na ang pangalan ng RR No. 1 Calasiao nung 2025?",
    "Bakit pinalitan ang Revenue Region 1 Calasiao ng Ilocos Region?",
    "REVENUE DELEGATION AUTHORITY ORDER NO. 22-2025",
    "Sino ang authorized mag-sign sa RR 11 Iloilo City matapos mag-retire ang OIC-Regional Director nung May 2025?",
    "Ano ang nakasaad sa RDAO No. 22-2025 ng BIR?",
    "Sino si OIC-Assistant Regional Director Jona Ruth Alonte ng Iloilo?",
    "May revenue administrative order ba nung 2025 tungkol sa pagpapalit ng pangalan ng revenue region?",
    "Revenue Delegation Authority Order 2025 Iloilo City",

    # Class 7: BIR Tax 2026
    # (RMO 5-2026, RDAO 8-2026, RMC 20-2026)
    "REVENUE MEMORANDUM ORDER NO. 5-2026",
    "Kailan ang deadline ng SALN submission sa RCC ng BIR ayon sa RMO 5-2026?",
    "Ano ang penalty sa hindi pag-file ng SALN ng empleyado ng BIR?",
    "Sino ang bumubuo ng Review and Compliance Committee (RCC) para sa SALN?",
    "REVENUE DELEGATION AUTHORITY ORDER NO. 8-2026",
    "Sino ang OIC ng Assistant Commissioner ng HRDS mula March 23 hanggang April 4, 2026?",
    "REVENUE MEMORANDUM CIRCULAR NO. 20-2026",
    "Paano mag-file ng annual income tax return para sa calendar year 2025 ayon sa RMC 20-2026?",
    "Ano ang deadline ng 2025 AITR na ifa-file sa April 15, 2026?",
    "Pwede bang mag-file ng BIR Form 1701-MS ang micro at small taxpayers ngayong 2026?",
    "Anong ePayment gateways ang pwedeng gamitin sa pagbabayad ng AITR nung 2026?",
    "Ano ang mga attachments sa AITR at kailan isasubmit sa eAFS ngayong 2026?",

    # Class 8: FAQ
    "paano ba magbayad ng buwis",
    "kailangan ko ba ng resibo",
    "ano ang ibig sabihin ng vat",
    "paano mag compute ng tax",
    "saan magbabayad ng income tax",
]

LABELS = [
    0, 0, 0, 0, 0,                        # Tax 2001
    1, 1, 1, 1, 1,                        # Tax 2002
    2, 2, 2, 2, 2,                        # Tax 2003
    3, 3, 3, 3, 3,                        # Tax 2022
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,   # Tax 2023 (12)
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,   # Tax 2024 (12)
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,   # Tax 2025 (12)
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,   # Tax 2026 (12)
    8, 8, 8, 8, 8,                        # FAQ
]

assert len(TRAINING_QUERIES) == len(LABELS), (
    f"TRAINING_QUERIES ({len(TRAINING_QUERIES)}) and LABELS ({len(LABELS)}) "
    "must be the same length"
)

# Class -> the year a *correct* prediction for that class implies. This is
# now used ONLY for the "ℹ️" comparison log in classify_query(), never as
# the actual Chroma filter — see the module docstring below for why.
# (Class 8, FAQ, deliberately has no entry — it isn't a year.)
_CLASS_TO_YEAR = {0: "2001", 1: "2002", 2: "2003", 3: "2022",
                  4: "2023", 5: "2024", 6: "2025", 7: "2026"}

# Years that actually exist as folders/metadata in this corpus (see
# app/database.py's ingestion log). An explicit citation for a year
# outside this set can't be satisfied by anything in the DB, so it must
# NOT be turned into a Chroma filter — that would just guarantee a
# no-answer instead of letting hybrid retrieval try the rest of the corpus
# (e.g. a typo'd year, or a real citation from a year not yet ingested).
KNOWN_YEARS = {"2001", "2002", "2003", "2022", "2023", "2024", "2025", "2026"}

# Matches the "<sequence number>-<year>" suffix every BIR issuance number
# in this corpus and in TRAINING_QUERIES uses: "1-2001", "14-2023",
# "34-2024", "001-2023", etc. Capped at 3 digits for the sequence number
# (the corpus's highest is "105-2023") specifically so it does NOT match
# a 4-digit year range like "2024-2025" (school year, fiscal year, a date
# range) — "2024" would overflow the {1,3} bound.
_ISSUANCE_YEAR_RE = re.compile(r"\b\d{1,3}[A-Za-z]?-((?:20)\d{2})\b")
_BARE_YEAR_RE = re.compile(r"\b(20\d{2})\b")


def extract_explicit_year(query: str) -> str | None:
    """
    Deterministically pull a source year out of the query text — no
    guessing, no model. Two tiers, checked in order:

    1. A formal issuance citation, e.g. "RMC 34-2024", "RR 2-2003",
       "RULING NO. 1-2001", "RDAO No. 22-2025" — the "<number>-<year>"
       pattern every issuance number in this corpus uses. Unambiguous by
       construction: if "14-2023" is in the query, the year is 2023.

    2. A bare year mention with nothing else competing for it, e.g.
       "2001 tax deadline", "nung 2023" — exactly one KNOWN_YEARS year
       appears anywhere in the query and no citation matched. This
       covers the (common) case where someone names a year without
       quoting the issuance number. It intentionally backs off to None
       the moment a SECOND distinct year also appears in the query (e.g.
       "yung 2025 ruling, still valid ba ngayong 2026?", or a range like
       "2024-2025") rather than picking one arbitrarily — two named
       years is a real ambiguity a regex shouldn't resolve by guessing,
       and unfiltered search still has both years available to it.

    Returns None — not a guess — when neither tier fires, or when a year
    is found but isn't in KNOWN_YEARS (mistyped year, or a real citation
    for a year not yet in the corpus): a filter that can only ever match
    zero documents is strictly worse than no filter.
    """
    for m in _ISSUANCE_YEAR_RE.finditer(query):
        year = m.group(1)
        if year in KNOWN_YEARS:
            return year

    bare_years = {y for y in _BARE_YEAR_RE.findall(query) if y in KNOWN_YEARS}
    if len(bare_years) == 1:
        return next(iter(bare_years))
    return None


def build_classifier(model_type: str = "naive_bayes"):
    """
    Build and train classifier.
    model_type: 'naive_bayes' or 'svm'
    """
    if model_type == "svm":
        model = make_pipeline(TfidfVectorizer(), LinearSVC())
        print("🤖 Using SVM Classifier")
    else:
        model = make_pipeline(TfidfVectorizer(), MultinomialNB())
        print("🤖 Using Naive Bayes Classifier")

    model.fit(TRAINING_QUERIES, LABELS)
    return model


def classify_query(model, query: str):
    """
    Classify a query and return (class_id, label, year_filter).

    IMPORTANT — where year_filter comes from:

    Every class here is a real BIR tax bucket (no chit-chat class —
    greetings are intercepted before this function is ever called, see
    app/greetings.py and api.py). predicted_class/label are still
    returned (useful for thesis reporting/ablations on router behavior),
    but year_filter is NOT derived from predicted_class. Every training
    example above that implies a specific year does so via an explicit
    issuance citation ("14-2023", "1-2001", ...) — i.e. the ONLY reliable
    signal the classifier is actually learning from for those classes is
    a substring regex could extract directly, with 100% precision,
    instead of guessing from a 9-way TF-IDF/Naive-Bayes decision trained
    on 5-12 examples per class.

    That guess matters because year_filter becomes a HARD Chroma `where`
    filter in retrieval.py: if the classifier predicts the wrong year for
    a query that paraphrases an issuance without citing it by number
    (very plausible — most real questions don't quote "RMC 34-2024"
    verbatim), the correct chunks are excluded from semantic search
    entirely, not just left unranked. A wrong hard filter is worse than
    no filter, so:

      1. If the query contains an explicit, in-corpus issuance citation,
         that year is used as the filter — no ambiguity possible.
      2. Otherwise, year_filter is None: hybrid retrieval searches the
         whole corpus unfiltered rather than betting a hard exclusion on
         an 8-way guess. The classifier's year guess is only logged, for
         debugging/eval, never enforced.
    """
    predicted_class = model.predict([query])[0]
    label           = CLASS_NAMES[predicted_class]

    explicit_year = extract_explicit_year(query)
    ml_guessed_year = _CLASS_TO_YEAR.get(predicted_class)

    if explicit_year:
        year_filter = explicit_year
    else:
        year_filter = None

    print(f"\n🔍 [Classifier] Query: '{query}'")
    print(f"📌 [Classified as]: {label}")
    if explicit_year:
        print(f"📅 [Year Filter Applied]: {explicit_year}  (explicit citation in query)")
        if ml_guessed_year and ml_guessed_year != explicit_year:
            print(
                f"⚠️  [Classifier] ML guessed year={ml_guessed_year} but the query "
                f"explicitly cites {explicit_year} — citation wins."
            )
    elif ml_guessed_year:
        print(
            f"ℹ️  [Classifier] ML guessed year={ml_guessed_year} from class '{label}', "
            f"but no explicit issuance citation was found in the query — NOT applying "
            f"as a hard filter; searching the whole corpus instead."
        )
    else:
        print("📅 [Year Filter]: none")

    return predicted_class, label, year_filter