import torch
import numpy as np
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, Trainer
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ===== CONFIG =====
MODEL_DIR = "./models/xlmr_lora"
DATA_DIR = "/content/tokenized/xlmr"   # Colab
NUM_LABELS = 2

def main():
    # Load test dataset
    test_ds = load_from_disk(f"{DATA_DIR}/test")

    # Load trained model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR,
        num_labels=NUM_LABELS
    )

    trainer = Trainer(model=model)

    # Predict
    predictions = trainer.predict(test_ds)
    logits = predictions.predictions
    labels = predictions.label_ids
    preds = np.argmax(logits, axis=1)

    # Metrics
    f1 = f1_score(labels, preds, average="macro")
    precision = precision_score(labels, preds, average="macro")
    recall = recall_score(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds)

    print("\n=== XLM-R TEST RESULTS ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro-F1 : {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(labels, preds, digits=4))

if __name__ == "__main__":
    main()
