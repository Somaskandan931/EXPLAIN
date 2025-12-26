import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
import evaluate

# ===== CONFIG =====
MODEL_NAME = "ai4bharat/indic-bert"
DATA_DIR = "/content/tokenized/indicbert"   # Colab
OUTPUT_DIR = "../../app/models/indicbert_lora"
NUM_LABELS = 2

# ===== METRIC =====
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return f1_metric.compute(
        predictions=preds,
        references=labels,
        average="macro"
    )

def main():
    # Load datasets
    train_ds = load_from_disk(f"{DATA_DIR}/train")
    val_ds = load_from_disk(f"{DATA_DIR}/val")

    # Load base model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS
    )

    # ===== LoRA CONFIG (ALBERT NEEDS EXPLICIT TARGETS) =====
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["query", "key", "value"],
        bias="none"
    )

    model = get_peft_model(model, lora_config)

    # ===== UNFREEZE CLASSIFIER HEAD (CRITICAL) =====
    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True

    model.print_trainable_parameters()

    # ===== CLASS-WEIGHTED LOSS =====
    class_weights = torch.tensor([1.0, 1.5]).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss

    # ===== TRAINING ARGUMENTS (COLAB SAFE) =====
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        num_train_epochs=4,
        weight_decay=0.01,
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    # ===== TRAINER =====
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)

if __name__ == "__main__":
    main()
