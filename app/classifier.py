from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline

# --- Class Mapping ---
CLASS_NAMES = {
    0: "Non-Tax / Chit-Chat",
    1: "BIR Tax Query (Source Year: 2001)",
    2: "BIR Tax Query (Source Year: 2002)",
    3: "BIR Tax Query (Source Year: 2003)",
    4: "BIR Tax Query (Source Year: 2022)",
    5: "BIR Tax Query (Source Year: 2023)",
    6: "BIR Tax Query (Source Year: 2024)",
    7: "BIR Tax Query (Source Year: 2025)",
    8: "BIR Tax Query (Source Year: 2026)",
    9: "FAQ",
}

# --- Training Data ---
TRAINING_QUERIES = [
    # Class 0: Chit-chat
    "magandang araw sayo",
    "kamusta ka",
    "hello how are you",
    "kumain ka na ba",
    "ano ang pangalan mo",
    "hello",
    "I wanna ask something",

    # Class 1: BIR Tax 2001
    "RULING  NO.  1-2001",
    "paano mag file ng ITR for 2001",
    "may changes ba sa income tax return ko nung 2001",
    "2001 tax deadline",
    "2001 bir circular",

    # Class 2: BIR Tax 2002
    "saan makikita ang 2002 tax table",
    "update sa vat nitong 2002",
    "RULING NO. 2-2002",
    "ano ang bagong batas sa tax ngayong 2002",
    "2002 bir revenue regulation",

    # Class 3: BIR Tax 2003
    "REVENUE BULLETIN NO. 2-2003",
    "Ano ba ang nakasulat sa Revenue Bulletin No. 2-2003?",
    "Paano i-compute ang estate tax under RR 2-2003 or bulletin 2003?",
    "Active pa ba yung legal guidelines ng Revenue Bulletin nung 2003?",
    "Saan makakakuha ng kopya ng BIR Revenue Bulletin 2-2003?",

    # Class 4: BIR Tax 2022
    "REVENUE DELEGATION AUTHORITY ORDER NO. 1-2022",
    "Ano ang nakasaad sa RDAO No. 1-2022 ng BIR?",
    "Paano i-apply yung Revenue Delegation Authority Order nung 2022?",
    "Sino ang authorized mag-sign under RDAO 1-2022?",
    "May circular ba nung 2022 tungkol sa signature authorities?",

    # Class 5: BIR Tax 2023
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

    # Class 6: BIR Tax 2024
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

    # Class 7: BIR Tax 2025
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

    # Class 8: BIR Tax 2026
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

    # Class 9: FAQ
    "paano ba magbayad ng buwis",
    "kailangan ko ba ng resibo",
    "ano ang ibig sabihin ng vat",
    "paano mag compute ng tax",
    "saan magbabayad ng income tax",
]

LABELS = [
    0, 0, 0, 0, 0, 0, 0,                  # Chit-chat
    1, 1, 1, 1, 1,                        # Tax 2001
    2, 2, 2, 2, 2,                        # Tax 2002
    3, 3, 3, 3, 3,                        # Tax 2003
    4, 4, 4, 4, 4,                        # Tax 2022
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,   # Tax 2023 (12)
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,   # Tax 2024 (12)
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,   # Tax 2025 (12)
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,   # Tax 2026 (12)
    9, 9, 9, 9, 9,                        # FAQ
]

assert len(TRAINING_QUERIES) == len(LABELS), (
    f"TRAINING_QUERIES ({len(TRAINING_QUERIES)}) and LABELS ({len(LABELS)}) "
    "must be the same length"
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
    if predicted_class == 1:
        year_filter = "2001"
    elif predicted_class == 2:
        year_filter = "2002"
    elif predicted_class == 3:
        year_filter = "2003"
    elif predicted_class == 4:
        year_filter = "2022"
    elif predicted_class == 5:
        year_filter = "2023"
    elif predicted_class == 6:
        year_filter = "2024"
    elif predicted_class == 7:
        year_filter = "2025"
    elif predicted_class == 8:
        year_filter = "2026"

    print(f"\n🔍 [Classifier] Query: '{query}'")
    print(f"📌 [Classified as]: {label}")
    if year_filter:
        print(f"📅 [Year Filter Applied]: {year_filter}")

    return predicted_class, label, year_filter