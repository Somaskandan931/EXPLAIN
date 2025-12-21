"""
Phase 3.2: Tokenization for XLM-R and IndicBERT
"""

from datasets import Dataset
from transformers import AutoTokenizer
import pandas as pd
from pathlib import Path

MAX_LENGTH = 256

SPLITS_DIR = Path("processed_data/splits")
OUTPUT_DIR = Path("processed_data/tokenized")
OUTPUT_DIR.mkdir(exist_ok=True)

TOKENIZERS = {
    "xlmr": "xlm-roberta-base",
    "indicbert": "ai4bharat/indic-bert"
}


def tokenize_and_save(route):
    print(f"\nTokenizing route: {route}")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZERS[route])
    route_out = OUTPUT_DIR / route
    route_out.mkdir(exist_ok=True)

    for split in ["train", "val", "test"]:
        csv_path = SPLITS_DIR / route / f"{split}.csv"
        df = pd.read_csv(csv_path)

        dataset = Dataset.from_pandas(df)

        def tokenize(batch):
            return tokenizer(
                batch["text"],
                truncation=True,
                padding="max_length",
                max_length=MAX_LENGTH
            )

        tokenized = dataset.map(tokenize, batched=True)
        tokenized = tokenized.remove_columns(
            ["text", "language", "script", "route", "source", "id"]
        )

        tokenized.save_to_disk(str(route_out / split))
        print(f"Saved {route}/{split}")


def main():
    tokenize_and_save("xlmr")
    tokenize_and_save("indicbert")
    print("\nTranslation route will be tokenized after translation step.")


if __name__ == "__main__":
    main()
