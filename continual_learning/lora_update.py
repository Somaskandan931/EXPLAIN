from peft import PeftModel

def update_lora(base_model, replay_dataset, lora_path):
    model = PeftModel.from_pretrained(base_model, lora_path)

    trainer.train_dataset = replay_dataset
    trainer.train()

    model.save_pretrained(lora_path)
