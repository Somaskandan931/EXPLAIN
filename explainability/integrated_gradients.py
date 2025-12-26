import torch
from captum.attr import IntegratedGradients

class IGExplainer:
    def __init__(self, model, tokenizer, device=None):
        self.device = device or next(model.parameters()).device
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.model.eval()
        self.ig = IntegratedGradients(self.forward_func)

    def forward_func(self, inputs_embeds, attention_mask):
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )
        return outputs.logits

    def explain(self, text):
        # -------------------------------
        # Normalize input
        # -------------------------------
        if isinstance(text, list):
            text = text[0] if len(text) > 0 else ""
        elif not isinstance(text, str):
            text = str(text)

        # -------------------------------
        # Tokenize and move to device
        # -------------------------------
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        # -------------------------------
        # Prediction
        # -------------------------------
        with torch.no_grad():
            logits = self.model(**encoding).logits
            probs = torch.softmax(logits, dim=1)[0]
            raw_label = torch.argmax(probs).item()

            if logits.size(1) == 3:
                pred_label = 1 if raw_label in [1, 2] else 0
                confidence = probs[1:].sum().item()
            else:
                pred_label = raw_label
                confidence = probs[pred_label].item()

        # -------------------------------
        # Integrated Gradients
        # -------------------------------
        embeddings = self.model.get_input_embeddings()(input_ids)
        attributions = self.ig.attribute(
            inputs=embeddings,
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
            "confidence": float(confidence),
            "method": "IntegratedGradients",
            "tokens": explanation
        }
