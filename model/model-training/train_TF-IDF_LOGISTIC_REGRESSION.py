import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

# ======================
# CONFIG
# ======================
DATA_PATH = "C:/Users/somas/PycharmProjects/EXPLAIN/preprocessing/processed_data/combined_preprocessed_with_synthetic.csv"   # change if local
MAX_FEATURES = 50000
NGRAM_RANGE = (1, 2)
TEST_SIZE = 0.15
RANDOM_STATE = 42

# ======================
# LOAD DATA
# ======================
df = pd.read_csv(DATA_PATH)

df = df.dropna(subset=["text", "label"])
df["label"] = df["label"].astype(int)

X = df["text"].astype(str)
y = df["label"]

# ======================
# TRAIN / TEST SPLIT
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

print("Train size:", len(X_train))
print("Test size :", len(X_test))

# ======================
# TF-IDF VECTORIZATION
# ======================
vectorizer = TfidfVectorizer(
    max_features=MAX_FEATURES,
    ngram_range=NGRAM_RANGE,
    stop_words="english",
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ======================
# LOGISTIC REGRESSION
# ======================
clf = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    n_jobs=-1
)

clf.fit(X_train_tfidf, y_train)

# ======================
# EVALUATION
# ======================
y_pred = clf.predict(X_test_tfidf)

acc = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average="macro", zero_division=0
)

print("\n=== TF-IDF + LOGISTIC REGRESSION RESULTS ===")
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"Macro-F1  : {f1:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
