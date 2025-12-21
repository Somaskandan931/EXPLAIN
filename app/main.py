from fastapi import FastAPI
from app.schemas import InputText
from explainability.unified_explainer import UnifiedExplainer

app = FastAPI()

# Load models once
explainer = UnifiedExplainer(
    xlmr_model=xlmr_model,
    xlmr_tokenizer=xlmr_tokenizer,
    indic_model=indic_model,
    indic_tokenizer=indic_tokenizer,
    tfidf_explainer=tfidf_explainer
)

@app.post("/predict")
def predict(input: InputText):
    explanation = explainer.explain(input.text, input.model)
    return explanation
