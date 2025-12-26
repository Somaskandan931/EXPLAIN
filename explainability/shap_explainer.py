import shap
import torch
import pickle

# ==================================================
# SHAP for Transformer Models (CORRECT + STABLE)
# ==================================================
class SHAPTransformerExplainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.model.eval()

        # ✅ CORRECT masker
        self.masker = shap.maskers.Text(tokenizer)

        # ✅ CORRECT explainer
        self.explainer = shap.Explainer(
            self.predict,
            self.masker,
            output_names=["Real", "Fake"]
        )

    def predict(self, texts):
        """
        texts: list[str]
        returns: np.ndarray [batch_size, num_classes]
        """

        # 🔒 HARD GUARANTEE: list[str]
        if isinstance(texts, str):
            texts = [texts]
        else:
            texts = [str(t) for t in texts]

        # Tokenize
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # 🔥 SHAP REQUIRES: batch preserved
        return logits.detach().cpu().numpy()

    def explain(self, text):
        if not isinstance(text, str):
            text = str(text)

        return self.explainer([text])


# ==================================================
# SHAP for TF-IDF (CPU only – correct)
# ==================================================
class SHAPTFIDFExplainer:
    def __init__(self, model_path, vectorizer_path, X_train):
        self.model = pickle.load(open(model_path, "rb"))
        self.vectorizer = pickle.load(open(vectorizer_path, "rb"))
        self.explainer = shap.LinearExplainer(self.model, X_train)

    def explain(self, text):
        X = self.vectorizer.transform([str(text)])
        return self.explainer.shap_values(X)
