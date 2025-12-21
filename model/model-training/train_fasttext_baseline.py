import os
import fasttext
import fasttext.util
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ===============================
# CONFIG
# ===============================
FASTTEXT_DIR = "models/fasttext"
DATA_DIR = "data"

TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")

LANG_MODELS = {
    "en": "cc.en.300.bin",
    "hi": "cc.hi.300.bin"
}

REDUCED_DIM = 100   # IMPORTANT: reduces RAM usage


# ===============================
# SETUP
# ===============================
os.makedirs(FASTTEXT_DIR, exist_ok=True)
os.chdir(FASTTEXT_DIR)

print("Working directory:", os.getcwd())


# ===============================
# DOWNLOAD MODELS (ONE-TIME)
# ===============================
print("Downloading FastText models (only if not present)...")

for lang in LANG_MODELS:
    fasttext.util.download_model(lang, if_exists="ignore")

print("FastText models downloaded")


# ===============================
# LOAD & REDUCE MODELS
# ===============================
print("Loading and reducing FastText models...")

models = {}

for lang, model_name in LANG_MODELS.items():
    if not os.path.exists(model_name.replace(".300.", ".100.")):
        fasttext.util.reduce_model(model_name, REDUCED_DIM)

    reduced_model = model_name.replace(".300.", ".100.")
    models[lang] = fasttext.load_model(reduced_model)

print("Models ready")


# ===============================
# HELPER FUNCTIONS
# ===============================
def detect_language(text):
    """Very lightweight heuristic"""
    if any('\u0900' <= ch <= '\u097F' for ch in text):
        return "hi"
    return "en"


def sentence_embedding(text):
    lang = detect_language(text)
    model = models[lang]

    words = text.lower().split()
    vectors = [model.get_word_vector(w) for w in words if w.strip()]

    if len(vectors) == 0:
        return np.zeros(REDUCED_DIM)

    return np.mean(vectors, axis=0)


# ===============================
# LOAD DATA
# ===============================
print("Loading dataset...")

train_df = pd.read_csv(os.path.join("..", "..", TRAIN_FILE))
test_df = pd.read_csv(os.path.join("..", "..", TEST_FILE))

print("Train size:", len(train_df))
print("Test size:", len(test_df))


# ===============================
# VECTORIZE
# ===============================
print("Generating sentence embeddings...")

X_train = np.vstack([
    sentence_embedding(text)
    for text in tqdm(train_df["text"])
])

X_test = np.vstack([
    sentence_embedding(text)
    for text in tqdm(test_df["text"])
])

y_train = train_df["label"].values
y_test = test_df["label"].values


# ===============================
# TRAIN CLASSIFIER
# ===============================
print("Training Logistic Regression...")

clf = LogisticRegression(
    max_iter=1000,
    n_jobs=-1
)

clf.fit(X_train, y_train)


# ===============================
# EVALUATION
# ===============================
print("Evaluating model...")

y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
