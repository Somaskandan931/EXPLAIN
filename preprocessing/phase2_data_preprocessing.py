"""
Explainable Multilingual Fake News Detection System
Phase 2: Language-Aware Preprocessing (Enhanced with Synthetic Data)
Input  : combined_dataset.csv (from Phase 1)
         indicbert_synthetic.csv (synthetic data)
Output : combined_preprocessed.csv
         combined_preprocessed_with_synthetic.csv
"""

import pandas as pd
import re
from langdetect import detect
from tqdm import tqdm

tqdm.pandas()

# ======================================================
# PATHS
# ======================================================
INPUT_PATH = "C:/Users/somas/PycharmProjects/EXPLAIN/preprocessing/processed_data/combined_dataset.csv"
SYNTHETIC_PATH = "C:/Users/somas/PycharmProjects/EXPLAIN/preprocessing/processed_data/indicbert_synthetic.csv"
OUTPUT_PATH = "C:/Users/somas/PycharmProjects/EXPLAIN/preprocessing/processed_data/combined_preprocessed.csv"
OUTPUT_WITH_SYNTHETIC = "C:/Users/somas/PycharmProjects/EXPLAIN/preprocessing/processed_data/combined_preprocessed_with_synthetic.csv"

MAX_CHARS = 5000   # safety cap for transformers
MIN_WORDS = 3      # avoid garbage samples


# ======================================================
# TEXT CLEANING (TRANSFORMER-SAFE)
# ======================================================
def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.strip()
    text = re.sub(r"<.*?>", " ", text)                 # HTML tags
    text = re.sub(r"http\S+|www\S+", "<URL>", text)    # URLs
    text = re.sub(r"\s+", " ", text)                   # extra spaces

    return text[:MAX_CHARS]


# ======================================================
# SCRIPT DETECTION
# ======================================================
def contains_devanagari(text):
    return bool(re.search(r"[\u0900-\u097F]", text))

def contains_latin(text):
    return bool(re.search(r"[A-Za-z]", text))

def detect_script(text):
    if contains_devanagari(text):
        return "devanagari"
    elif contains_latin(text):
        return "latin"
    else:
        return "other"


# ======================================================
# LANGUAGE DETECTION
# ======================================================
def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"


def normalize_language(lang, script):
    """
    Normalize to project-level language tags
    """
    if lang == "hi" and script == "latin":
        return "hinglish"
    if lang in ["en", "hi"]:
        return lang
    if lang == "unknown":
        return "unknown"
    return lang   # keep other Indic languages as-is


# ======================================================
# MODEL ROUTING LOGIC (CORE CONTRIBUTION)
# ======================================================
def assign_route(language):
    if language in ["en", "hinglish"]:
        return "xlmr"
    elif language == "hi":
        return "indicbert"
    else:
        return "translate_xlmr"


# ======================================================
# PROCESS SYNTHETIC DATA
# ======================================================
def process_synthetic_data(synthetic_path):
    """
    Process synthetic data with same pipeline as original data
    """
    print("\n" + "=" * 70)
    print("PROCESSING SYNTHETIC DATA FOR INDICBERT")
    print("=" * 70)

    try:
        synthetic_df = pd.read_csv(synthetic_path)
        print(f"✓ Loaded synthetic dataset: {synthetic_df.shape}")

        # Clean text
        print("\n🔹 Cleaning synthetic text...")
        synthetic_df["text"] = synthetic_df["text"].progress_apply(clean_text)

        # Remove very short samples
        synthetic_df["word_count"] = synthetic_df["text"].str.split().str.len()
        synthetic_df = synthetic_df[synthetic_df["word_count"] >= MIN_WORDS]
        print(f"✓ After cleaning: {synthetic_df.shape}")

        # Script detection
        print("\n🔹 Detecting script...")
        synthetic_df["script"] = synthetic_df["text"].progress_apply(detect_script)

        # Language detection
        print("\n🔹 Detecting language...")
        synthetic_df["raw_language"] = synthetic_df["text"].progress_apply(detect_language)
        synthetic_df["language"] = synthetic_df.apply(
            lambda x: normalize_language(x["raw_language"], x["script"]), axis=1
        )

        # Force route to IndicBERT (since this is synthetic data for IndicBERT)
        synthetic_df["route"] = "indicbert"

        # Add unique IDs with synthetic prefix
        synthetic_df["id"] = ["synthetic_" + str(i) for i in range(len(synthetic_df))]

        # Ensure source column exists
        if "source" not in synthetic_df.columns:
            synthetic_df["source"] = "synthetic_indicbert"

        # Final column order
        synthetic_df = synthetic_df[
            ["id", "text", "label", "language", "script", "route", "source"]
        ]

        print(f"\n✅ Processed {len(synthetic_df)} synthetic samples")
        return synthetic_df

    except FileNotFoundError:
        print(f"⚠️  Synthetic data file not found: {synthetic_path}")
        print("   Proceeding without synthetic data...")
        return None


# ======================================================
# MAIN PIPELINE
# ======================================================
def main():
    print("=" * 70)
    print("PHASE 2: PREPROCESSING COMBINED DATASET")
    print("=" * 70)

    # ----------------------------
    # Load Phase 1 output
    # ----------------------------
    df = pd.read_csv(INPUT_PATH)
    print(f"✓ Loaded dataset: {df.shape}")

    # ----------------------------
    # Clean text
    # ----------------------------
    print("\n🔹 Cleaning text...")
    df["text"] = df["text"].progress_apply(clean_text)

    # Remove very short / bad samples
    df["word_count"] = df["text"].str.split().str.len()
    df = df[df["word_count"] >= MIN_WORDS]
    print(f"✓ After cleaning: {df.shape}")

    # ----------------------------
    # Script detection
    # ----------------------------
    print("\n🔹 Detecting script...")
    df["script"] = df["text"].progress_apply(detect_script)

    # ----------------------------
    # Language detection
    # ----------------------------
    print("\n🔹 Detecting language...")
    df["raw_language"] = df["text"].progress_apply(detect_language)
    df["language"] = df.apply(
        lambda x: normalize_language(x["raw_language"], x["script"]), axis=1
    )

    # ----------------------------
    # Routing
    # ----------------------------
    print("\n🔹 Assigning model routes...")
    df["route"] = df["language"].apply(assign_route)

    # ----------------------------
    # Add unique IDs
    # ----------------------------
    df["id"] = ["sample_" + str(i) for i in range(len(df))]

    # ----------------------------
    # Final column order
    # ----------------------------
    df = df[
        ["id", "text", "label", "language", "script", "route", "source"]
    ]

    # ----------------------------
    # Save original preprocessed data
    # ----------------------------
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ Saved preprocessed dataset: {OUTPUT_PATH}")

    # ----------------------------
    # Process and merge synthetic data
    # ----------------------------
    synthetic_df = process_synthetic_data(SYNTHETIC_PATH)

    if synthetic_df is not None:
        print("\n" + "=" * 70)
        print("MERGING SYNTHETIC DATA")
        print("=" * 70)

        # Combine original and synthetic
        combined_df = pd.concat([df, synthetic_df], ignore_index=True)

        # Shuffle the combined dataset
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Save combined dataset
        combined_df.to_csv(OUTPUT_WITH_SYNTHETIC, index=False)
        print(f"\n✅ Saved combined dataset with synthetic data: {OUTPUT_WITH_SYNTHETIC}")

        # Summary stats for combined dataset
        print("\n" + "=" * 70)
        print("COMBINED DATASET STATISTICS")
        print("=" * 70)

        print(f"\nTotal samples: {len(combined_df)}")
        print(f"  - Original samples: {len(df)}")
        print(f"  - Synthetic samples: {len(synthetic_df)}")

        print("\n📊 Language distribution:")
        print(combined_df["language"].value_counts())

        print("\n📊 Routing distribution:")
        print(combined_df["route"].value_counts())

        print("\n📊 IndicBERT route breakdown:")
        indicbert_data = combined_df[combined_df["route"] == "indicbert"]
        print(f"  Total IndicBERT samples: {len(indicbert_data)}")
        print(f"  - Original: {len(indicbert_data[indicbert_data['source'] != 'synthetic_indicbert'])}")
        print(f"  - Synthetic: {len(indicbert_data[indicbert_data['source'] == 'synthetic_indicbert'])}")

        print("\n📊 Source distribution:")
        print(combined_df["source"].value_counts())

        print("\n📊 Label distribution:")
        print(combined_df["label"].value_counts())

    else:
        print("\n⚠️  No synthetic data merged. Only original preprocessed data saved.")

    # ----------------------------
    # Summary stats (original data only)
    # ----------------------------
    print("\n" + "=" * 70)
    print("ORIGINAL DATASET STATISTICS")
    print("=" * 70)

    print("\n📊 Language distribution:")
    print(df["language"].value_counts())

    print("\n📊 Routing distribution:")
    print(df["route"].value_counts())

    print("\n📊 Source distribution:")
    print(df["source"].value_counts())

    print("\n📊 Label distribution:")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()