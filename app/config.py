import os
from pathlib import Path
import torch  # needed for DEVICE detection

# =====================================================
# Base paths (hardcoded)
# =====================================================
MODELS_DIR = Path(r"C:\Users\somas\PycharmProjects\EXPLAIN\app\models")
DATA_DIR = Path(r"C:\Users\somas\PycharmProjects\EXPLAIN\data")

# =====================================================
# Base models (can be HF ID or local folder)
# =====================================================
XLMR_BASE_MODEL = "xlm-roberta-base"  # HF ID
INDICBERT_BASE_MODEL = Path(r"C:\Users\somas\PycharmProjects\EXPLAIN\app\models\Yousuf-Islam\indicBERTv2_Model_v2")  # local folder

# =====================================================
# LoRA adapter paths (optional)
# =====================================================
XLMR_MODEL_PATH = MODELS_DIR / "xlmr_lora"
INDICBERT_MODEL_PATH = MODELS_DIR / "indicbert_lora"

# =====================================================
# API Configuration
# =====================================================
API_HOST = "0.0.0.0"
API_PORT = 8000

# =====================================================
# MongoDB Configuration
# =====================================================
MONGODB_URI = "mongodb://localhost:27017/"
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
NEWS_API_KEY = "59593215cd46458c9214ba33b88c2831"

# =====================================================
# Ensure directories exist
# =====================================================
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# Helper function to get string path (for AutoModel)
# =====================================================
def get_model_path(model):
    """
    Convert model config to string path if it's a Path object.
    Keeps HF model IDs unchanged.
    """
    if isinstance(model, Path):
        return str(model.resolve())
    return model
