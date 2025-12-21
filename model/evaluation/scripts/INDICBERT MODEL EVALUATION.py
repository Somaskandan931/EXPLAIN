import numpy as np
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, Trainer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ===== CONFIG =====
MODEL_DIR = "./models/indicbert_lora"
DATA_DIR = "/content/tokenized/indicbert"   # Colab
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

    predictions = trainer.predict(test_ds)
    logits = predictions.predictions
    labels = predictions.label_ids
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="macro")
    recall = recall_score(labels, preds, average="macro")
    f1 = f1_score(labels, preds, average="macro")
    cm = confusion_matrix(labels, preds)

    print("\n=== INDICBERT TEST RESULTS ===")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"Macro-F1  : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(labels, preds, digits=4))

if __name__ == "__main__":
    main()
