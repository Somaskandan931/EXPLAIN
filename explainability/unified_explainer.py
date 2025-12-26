import logging
import torch
from explainability.integrated_gradients import IGExplainer
from explainability.shap_explainer import SHAPTransformerExplainer

logger = logging.getLogger(__name__)


class UnifiedExplainer:
    """
    Unified explainer returning:
    - Integrated Gradients (IG)
    - SHAP
    SINGLE TEXT ONLY (FastAPI handles batching)
    """

    def __init__(self, xlmr_model, xlmr_tokenizer, indic_model, indic_tokenizer, tfidf_explainer=None):
        logger.info("Initializing UnifiedExplainer")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.xlmr_model = xlmr_model.to(self.device).eval()
        self.indic_model = indic_model.to(self.device).eval()

        # IG
        self.xlmr_ig = IGExplainer(self.xlmr_model, xlmr_tokenizer, device=self.device)
        self.indic_ig = IGExplainer(self.indic_model, indic_tokenizer, device=self.device)

        # SHAP (SINGLE INPUT ONLY)
        self.xlmr_shap = SHAPTransformerExplainer(self.xlmr_model, xlmr_tokenizer)
        self.indic_shap = SHAPTransformerExplainer(self.indic_model, indic_tokenizer)

        self.tfidf = tfidf_explainer

    # -------------------------------
    # Explain ONE text
    # -------------------------------
    def explain(self, text: str, model_name: str) -> dict:
        text = str(text)

        try:
            if model_name == "xlmr":
                ig = self.xlmr_ig.explain(text)
                shap_vals = self.xlmr_shap.explain(text)

            elif model_name == "indicbert":
                ig = self.indic_ig.explain(text)
                shap_vals = self.indic_shap.explain(text)

            elif model_name == "tfidf":
                if self.tfidf is None:
                    raise RuntimeError("TF-IDF explainer not available")
                return self.tfidf.explain(text)

            else:
                raise ValueError(f"Unknown model: {model_name}")

            return {
                "prediction": int(ig["prediction"]),
                "confidence": float(ig["confidence"]),
                "model": model_name,
                "explanations": {
                    "integrated_gradients": ig["tokens"],
                    "shap": self._format_shap(shap_vals),
                },
            }

        except Exception as e:
            logger.error(f"Unified explanation failed [{model_name}]: {e}")
            raise

    # -------------------------------
    # Ensemble explain (FastAPI endpoint)
    # -------------------------------
    def ensemble_explain(self, text: str) -> dict:
        text = str(text)

        xlmr = self.explain(text, "xlmr")
        indic = self.explain(text, "indicbert")

        fake_score = (
            xlmr["prediction"] * xlmr["confidence"]
            + indic["prediction"] * indic["confidence"]
        )
        real_score = (
            (1 - xlmr["prediction"]) * xlmr["confidence"]
            + (1 - indic["prediction"]) * indic["confidence"]
        )

        final_pred = 1 if fake_score > real_score else 0
        final_conf = max(fake_score, real_score) / 2

        return {
            "prediction": final_pred,
            "confidence": final_conf,
            "method": "ensemble_xlmr_indicbert",
            "details": {
                "xlmr": xlmr,
                "indicbert": indic,
            },
        }

    # -------------------------------
    # SHAP formatting (SAFE)
    # -------------------------------
    def _format_shap(self, shap_values):
        return {
            "tokens": shap_values.data[0].tolist(),
            "scores": shap_values.values[0].tolist(),
        }
