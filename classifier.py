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
    5: "General Taglish Tax Query",
    6: "Board Games (Monopoly / Ticket to Ride)"
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

    # Class 5: General Taglish Tax
    "paano ba magbayad ng buwis",
    "kailangan ko ba ng resibo",
    "ano ang ibig sabihin ng vat",
    "paano mag compute ng tax",
    "saan magbabayad ng income tax",

    # Class 6: Board Games
    "strategy sa ticket to ride",
    "monopoly",
    "ticket to ride",
    "how do I get out of jail in monopoly",
    "How many points does the longest continuous train get in Ticket to Ride?",
]

LABELS = [
    0, 0, 0, 0, 0, 0, 0,  # Chit-chat
    1, 1, 1, 1, 1,  # Tax 2001
    2, 2, 2, 2, 2,  # Tax 2002
    3, 3, 3, 3, 3,  # Tax 2003
    4, 4, 4, 4, 4,  # Tax 2022
    5, 5, 5, 5, 5,  # General Tax
    6, 6, 6, 6, 6,  # Board Games
]


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

    print(f"\n🔍 [Classifier] Query: '{query}'")
    print(f"📌 [Classified as]: {label}")
    if year_filter:
        print(f"📅 [Year Filter Applied]: {year_filter}")

    return predicted_class, label, year_filter