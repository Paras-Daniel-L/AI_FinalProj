from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline

# --- Class Mapping ---
CLASS_NAMES = {
    0: "Board Games (Monopoly / Ticket to Ride)",
    1: "General Taglish Tax Query",
    2: "Non-Tax / Chit-Chat",
    3: "BIR Tax Query (Source Year: 2001)",
    4: "BIR Tax Query (Source Year: 2002)",
    5: "BIR Tax Query (Source Year: 2003)",
    6: "BIR Tax Query (Source Year: 2022)",
    7: "BIR Tax Query (Source Year: 2023)",
    8: "BIR Tax Query (Source Year: 2024)",
    9: "BIR Tax Query (Source Year: 2025)",
    10: "BIR Tax Query (Source Year: 2026)",
    11: "Frequently Asked Questions (FAQ)"
}

# --- Training Data (10 samples per class to maintain class balance) ---
TRAINING_QUERIES = [
    # Class 0: Board Games
    "strategy sa ticket to ride",
    "monopoly property trading rules",
    "ticket to ride longest train route",
    "how do I get out of jail in monopoly",
    "How many points does the longest continuous train get in Ticket to Ride?",
    "paano manalo sa monopoly game",
    "pwede ba mag-mortgage ng property sa monopoly",
    "how much cash do you receive when passing GO in monopoly",
    "paano gamitin ang locomotives sa ticket to ride board game",
    "ano ang gagawin kapag na-bankrupt sa monopoly",

    # Class 1: General Taglish Tax Query
    "paano ba magbayad ng buwis sa Pilipinas",
    "kailangan ko ba ng official receipt para sa negosyo ko",
    "ano ang ibig sabihin ng value added tax o vat",
    "paano mag compute ng income tax ng regular employee",
    "saan magbabayad ng taunang income tax",
    "kailangan ba mag-register sa BIR kapag freelance worker",
    "ano ang mga documents na kailangan para makakuha ng TIN number",
    "magkano ang surcharge o penalty kapag late filing ng tax return",
    "paano mag-register ng bagong business sa Revenue District Office",
    "ano ang ibig sabihin ng tax exemption sa BIR",

    # Class 2: Chit-Chat / Non-Tax
    "magandang araw sayo Sagot AI",
    "kamusta ka naman ngayong araw",
    "hello how are you doing today",
    "kumain ka na ba kanina",
    "ano ang pangalan mo at ano ang silbi mo",
    "hello pwede ba magtanong tungkol sa kung ano-ano",
    "I wanna ask something about life in general",
    "maraming salamat sa tulong mo ha",
    "ano ang paborito mong kulay chatbot",
    "sige mag-ingat ka bye bye",

    # Class 3: BIR Tax 2001
    "RULING NO. 1-2001",
    "paano mag file ng ITR for 2001 taxable year",
    "may changes ba sa income tax return guidelines nung 2001",
    "2001 tax deadline and compliance requirements",
    "2001 bir circular and ruling regulations",
    "ano ang nakasaad sa BIR ruling noong 2001",
    "mga revenue issuances para sa taong 2001",
    "BIR Ruling 1-2001 official digest",
    "mga revenue regulations nung 2001 tungkol sa withholding",
    "source documents and tax issuances for year 2001",

    # Class 4: BIR Tax 2002
    "saan makikita ang 2002 tax table",
    "update sa vat regulations nitong 2002",
    "RULING NO. 2-2002",
    "ano ang bagong batas sa tax noong 2002",
    "2002 bir revenue regulation guidelines",
    "ano ang nakasulat sa ruling 2-2002 ng BIR",
    "mga tax circulars and table changes para sa taong 2002",
    "may memo ba nung 2002 tungkol sa final withholding tax",
    "BIR ruling no. 2-2002 digest summary",
    "mga memorandum circulars issued during 2002",

    # Class 5: BIR Tax 2003
    "REVENUE BULLETIN NO. 2-2003",
    "Ano ba ang nakasulat sa Revenue Bulletin No. 2-2003?",
    "Paano i-compute ang estate tax under RR 2-2003 or bulletin 2003?",
    "Active pa ba yung legal guidelines ng Revenue Bulletin nung 2003?",
    "Saan makakakuha ng kopya ng BIR Revenue Bulletin 2-2003?",
    "ano ang mga probisyon sa Revenue Bulletin 2-2003",
    "tax rules and updates under RR 2-2003",
    "may issuance ba ang BIR nung 2003 regarding estate tax assessment",
    "Revenue Bulletin 2003 official copy digest",
    "guidelines ng BIR para sa 2003 tax issuances",

    # Class 6: BIR Tax 2022
    "REVENUE DELEGATION AUTHORITY ORDER NO. 1-2022",
    "Ano ang nakasaad sa RDAO No. 1-2022 ng BIR?",
    "Paano i-apply yung Revenue Delegation Authority Order nung 2022?",
    "Sino ang authorized mag-sign under RDAO 1-2022?",
    "May circular ba nung 2022 tungkol sa signature authorities?",
    "RDAO No. 1-2022 official digest and summary",
    "authority to sign revenue tax documents under RDAO 1-2022",
    "ano ang mga update sa BIR delegated authorities noong 2022",
    "sino ang authorized signers sa BIR batay sa 2022 order",
    "RDAO 1-2022 guidelines for regional directors and officers",

    # Class 7: BIR Tax 2023
    "RMC No. 30-2023",
    "Ano ang mga bagong guidelines sa ilalim ng RMC No. 33-2023?",
    "Paano i-comply ang requirements ng RMC No. 49-2023v3?",
    "Meron ba kayong digest ng RMO No. 16-2023 Digest FINAL?",
    "Ano ang nakasaad sa RMO No. 26-2023 Final Digest?",
    "Paano ang VAT refund rules ayon sa RR 9-2023?",
    "Ano ang bagong withholding tax requirement sa RR 16-2023 nitong 2023?",
    "Ano ang circular ng BIR nung 2023 tungkol sa online sellers withholding tax?",
    "Pakipaliwanag naman ang mga tax changes sa ilalim ng revenue regulation 16-2023",
    "Ano-ano ang mga importanteng tax circulars at issuances para sa taong 2023?",

    # Class 8: BIR Tax 2024
    "RMC No. 1-2024",
    "Ano ang nilalaman at implementing rules ng RMC No. 3-2024?",
    "Saan makikita ang digest para sa RMO No. 43-2024 Digest?",
    "Ano ang mga binago sa audit guidelines ayon sa RMO No. 46-2024 Digest?",
    "Paano ang bagong filing rules at deadlines sa ilalim ng RR 5-2024 final?",
    "Ano ang mga tax classification updates under RR 8-2024?",
    "May bagong revenue regulation ba nitong 2024 para sa Ease of Paying Taxes Act?",
    "Paano i-implement ang taxpayer classification at invoicing rules under RR 8-2024?",
    "Ano ang latest issuances ng BIR para sa taxable year 2024?",
    "RR No. 8-2024 implementing guidelines para sa micro at small taxpayers",

    # Class 9: BIR Tax 2025
    "RAO No. 5-2025 (Digest)",
    "Ano ang nakasaad sa administrative order RAO No. 5-2025 Digest?",
    "May summary ba kayo para sa RMC No. 11-2025 Digest?",
    "Ano ang mga bagong probisyon sa ilalim ng RMO No. 30-2025 DIgest?",
    "Ano ang revenue audit order guidelines noong 2025 under RAO 5-2025?",
    "Paano mag-comply sa documentary requirements na nakasaad sa RMC No. 11-2025?",
    "Ano ang bagong memorandum order ng BIR na inilabas nitong 2025?",
    "May update ba sa operational tax guidelines sa ilalim ng RMO 30-2025?",
    "Ano ang bagong issuance ng BIR para sa tax compliance nitong 2025?",
    "RAO No. 5-2025 digest summary tungkol sa audit procedures",

    # Class 10: BIR Tax 2026
    "RAO No. 1-2026 Digest",
    "Ano ang mga patakaran at panuntunan sa RMC No. 4-2026 Digest?",
    "Paano ipapatupad ang administrative guidelines ng RMO NO. 5-2026?",
    "Ano ang mga bagong tax rate o compliance rules sa ilalim ng RR No. 2-2026?",
    "Meron bang bagong revenue regulation 2-2026 para sa taong 2026?",
    "Ano ang nakasaad sa delegation order na RAO No. 1-2026 Digest?",
    "Paano mag-comply sa circular guidelines ng BIR para sa taong 2026?",
    "Ano ang bagong memo circular na RMO 5-2026 nitong 2026?",
    "May circular ba ang BIR nitong 2026 tungkol sa electronic tax filing?",
    "RR No. 2-2026 revenue regulations overview and summary",

    # Class 11: Frequently Asked Questions (FAQ)
    "Taxable ba ang 13th month pay at bonus kapag lumagpas sa 90,000 pesos?",
    "Paano ba i-compute ang 8% preferential tax rate para sa purely self-employed?",
    "Sino-sino ang qualified sa substituted filing ng ITR gamit ang BIR Form 2316?",
    "Ano ang pagkakaiba ng 40% Optional Standard Deduction OSD sa Itemized Deductions?",
    "Exempted ba sa Philippine income tax ang kinikita ng mga OFW at overseas seamen?",
    "Kailan nagiging covered ang isang korporasyon sa 2% Minimum Corporate Income Tax o MCIT?",
    "Paano ang computation ng 2% MCIT para sa sale of services at cost of services?",
    "Ano ang tamang proseso kapag no payment ITR ang ifi-file sa Revenue District Office?",
    "Pwede pa ba mag-amend ng income tax return kapag may Letter of Authority na mula sa BIR?",
    "Pwede ko pa ba i-claim bilang dependent ang senior citizen kong magulang under sa TRAIN Law?"
]

# Generate labels programmatically: exactly 10 entries per class (120 total)
LABELS = [class_id for class_id in range(12) for _ in range(10)]

assert len(TRAINING_QUERIES) == len(LABELS), (
    f"Mismatch: {len(TRAINING_QUERIES)} queries vs {len(LABELS)} labels"
)


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
    year_filter is used to filter ChromaDB by year metadata.
    """
    predicted_class = model.predict([query])[0]
    label           = CLASS_NAMES[predicted_class]

    # Map class to optional year filter for ChromaDB
    year_filter = None
    if predicted_class == 3:
        year_filter = "2001"
    elif predicted_class == 4:
        year_filter = "2002"
    elif predicted_class == 5:
        year_filter = "2003"
    elif predicted_class == 6:
        year_filter = "2022"
    elif predicted_class == 7:
        year_filter = "2023"
    elif predicted_class == 8:
        year_filter = "2024"
    elif predicted_class == 9:
        year_filter = "2025"
    elif predicted_class == 10:
        year_filter = "2026"
    elif predicted_class == 11:
        year_filter = "faq"

    print(f"\n🔍 [Classifier] Query: '{query}'")
    print(f"📌 [Classified as]: {label}")
    if year_filter:
        print(f"📅 [Year Filter Applied]: {year_filter}")

    return predicted_class, label, year_filter