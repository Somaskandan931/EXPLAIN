# train_fasttext_baseline.py
# ==========================

import os
import pandas as pd
import numpy as np
import fasttext
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# -----------------------
# Step 1: Load dataset
# -----------------------
DATA_PATH = (
    "C:/Users/somas/PycharmProjects/EXPLAIN/"
    "preprocessing/processed_data/combined_preprocessed.csv"
)

df = pd.read_csv(DATA_PATH)
print(f"✓ Loaded dataset: {df.shape}")

required_cols = {"text", "label", "language"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"CSV must contain {required_cols}")

texts = df["text"].astype(str).tolist()
labels = df["label"].astype(int).tolist()
languages = df["language"].astype(str).tolist()

# -----------------------
# Step 2: Load FastText models
# -----------------------
print("🔹 Loading FastText models...")

FT_MODEL_DIR = "C:/Users/somas/PycharmProjects/EXPLAIN/models/"

EN_MODEL_PATH = os.path.join(FT_MODEL_DIR, "cc.en.300.bin")
HI_MODEL_PATH = os.path.join(FT_MODEL_DIR, "cc.hi.300.bin")

# Fail fast if models are missing
if not os.path.exists(EN_MODEL_PATH):
    raise FileNotFoundError(f"Missing FastText model: {EN_MODEL_PATH}")

if not os.path.exists(HI_MODEL_PATH):
    raise FileNotFoundError(f"Missing FastText model: {HI_MODEL_PATH}")

ft_models = {
    "en": fasttext.load_model(EN_MODEL_PATH),
    "hi": fasttext.load_model(HI_MODEL_PATH),
}

print("✅ FastText models loaded: en, hi")

# -----------------------
# Step 3: Text → Embedding
# -----------------------
def get_embedding(text, lang):
    # Fallback to English for unsupported languages
    ft = ft_models.get(lang, ft_models["en"])

    words = text.split()
    if not words:
        return np.zeros(ft.get_dimension())

    return np.mean([ft.get_word_vector(w) for w in words], axis=0)

print("🔹 Converting texts to FastText embeddings...")
X = np.array([get_embedding(t, l) for t, l in zip(texts, languages)])
y = np.array(labels)
print("✅ Embedding generation completed")

# -----------------------
# Step 4: Train–Test split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"🔹 Train size: {len(X_train)} | Test size: {len(X_test)}")

# -----------------------
# Step 5: Train classifier
# -----------------------
print("🔹 Training Logistic Regression classifier...")
clf = LogisticRegression(
    max_iter=1000,
    n_jobs=-1,
    solver="lbfgs"
)
clf.fit(X_train, y_train)
print("✅ Training completed")

# -----------------------
# Step 6: Evaluation
# -----------------------
y_pred = clf.predict(X_test)

print("\n📊 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))
