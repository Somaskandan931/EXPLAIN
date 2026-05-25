import os
from pathlib import Path
import torch

# =====================================================
# Base paths  — derived from this file's location so
# the project works on any machine / drive / user
# =====================================================
# config.py lives at  <project_root>/app/config.py
# so two .parent calls get us to <project_root>
PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR      = PROJECT_ROOT / "app"
MODELS_DIR   = APP_DIR / "models"
DATA_DIR     = PROJECT_ROOT / "data"

# =====================================================
# Base models (can be HF ID or local folder)
# =====================================================
XLMR_BASE_MODEL      = "xlm-roberta-base"   # pulled from HuggingFace Hub

# Local IndicBERT — resolved to str so AutoTokenizer/AutoModel never
# receive a Path object (which some HF versions reject).
INDICBERT_BASE_MODEL = str(MODELS_DIR / "Yousuf-Islam" / "indicBERTv2_Model_v2")

# =====================================================
# LoRA adapter / checkpoint paths (optional)
# =====================================================
XLMR_MODEL_PATH      = MODELS_DIR / "xlmr_lora"
INDICBERT_MODEL_PATH = MODELS_DIR / "indicbert_lora"

# =====================================================
# API Configuration
# =====================================================
API_HOST = "0.0.0.0"
API_PORT = 8000

# =====================================================
# MongoDB Configuration
# =====================================================
MONGODB_URI     = os.environ.get("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DB_NAME = "fake_news_db"

# =====================================================
# Device Configuration
# =====================================================
DEVICE = "cuda" if torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

# =====================================================
# Model Configuration
# =====================================================
MAX_LENGTH = 512
BATCH_SIZE = 16

# =====================================================
# NewsAPI Configuration
# =====================================================
NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "59593215cd46458c9214ba33b88c2831")

# =====================================================
# Ensure directories exist
# =====================================================
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# Helper: normalise model identifier → str
# Works for both HF Hub IDs (str) and local Path objects.
# =====================================================
def get_model_path(model) -> str:
    if isinstance(model, Path):
        return str(model.resolve())
    return str(model)