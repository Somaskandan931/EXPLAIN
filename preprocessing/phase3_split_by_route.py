"""
Phase 3.1: Route-wise Train / Validation / Test Split
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

INPUT_PATH = "C:/Users/somas/PycharmProjects/EXPLAIN/preprocessing/processed_data/combined_preprocessed_with_synthetic.csv"
OUTPUT_DIR = Path("processed_data/splits")

TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42


def split_and_save(df, route_name):
    route_dir = OUTPUT_DIR / route_name
    route_dir.mkdir(parents=True, exist_ok=True)

    X = df
    y = df["label"]

    # Train + temp split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=TEST_SIZE + VAL_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # Val + test split
    val_ratio = VAL_SIZE / (TEST_SIZE + VAL_SIZE)
    X_val, X_test, _, _ = train_test_split(
        X_temp, y_temp,
        test_size=1 - val_ratio,
        stratify=y_temp,
        random_state=RANDOM_STATE
    )

    X_train.to_csv(route_dir / "train.csv", index=False)
    X_val.to_csv(route_dir / "val.csv", index=False)
    X_test.to_csv(route_dir / "test.csv", index=False)

    print(f"{route_name}: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")


def main():
    df = pd.read_csv(INPUT_PATH)
    OUTPUT_DIR.mkdir(exist_ok=True)

    for route in df["route"].unique():
        print(f"\nProcessing route: {route}")
        route_df = df[df["route"] == route]
        split_and_save(route_df, route)


if __name__ == "__main__":
    main()
