import shap
import torch

class SHAPTransformerExplainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.explainer = shap.Explainer(self.predict, tokenizer)

    def predict(self, texts):
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.logits.numpy()

    def explain(self, text):
        shap_values = self.explainer([text])
        return shap_values

import shap
import pickle

class SHAPTFIDFExplainer:
    def __init__(self, model_path, vectorizer_path, X_train):
        self.model = pickle.load(open(model_path, "rb"))
        self.vectorizer = pickle.load(open(vectorizer_path, "rb"))
        self.explainer = shap.LinearExplainer(self.model, X_train)

    def explain(self, text):
        X = self.vectorizer.transform([text])
        shap_values = self.explainer.shap_values(X)
        return shap_values
