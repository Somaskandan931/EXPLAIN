import torch
from captum.attr import IntegratedGradients

class IGExplainer:
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        self.ig = IntegratedGradients(self.forward_func)

    def forward_func(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.logits

    def explain(self, text):
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        with torch.no_grad():
            logits = self.model(**encoding).logits
            pred_label = torch.argmax(logits, dim=1).item()
            confidence = torch.softmax(logits, dim=1)[0][pred_label].item()

        attributions = self.ig.attribute(
            inputs=input_ids,
            additional_forward_args=(attention_mask,),
            target=pred_label,
            n_steps=50
        )

        token_importance = attributions.sum(dim=-1).squeeze(0)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        explanation = [
            {"token": tok, "score": float(score)}
            for tok, score in zip(tokens, token_importance)
            if tok not in ["<s>", "</s>", "<pad>"]
        ]

        return {
            "prediction": int(pred_label),
            "confidence": confidence,
            "method": "IntegratedGradients",
            "tokens": explanation
        }
