import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# Model paths (Transformer models only)
XLMR_MODEL_PATH = MODELS_DIR / "xlmr_model"
INDIC_MODEL_PATH = MODELS_DIR / "indicbert_model"

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000

# MongoDB Configuration
MONGODB_URI = "mongodb://localhost:27017/"
MONGODB_DB_NAME = "fake_news_db"

# Device Configuration
DEVICE = "cuda"  # Change to "cpu" if no GPU available

# Model Configuration
MAX_LENGTH = 512
BATCH_SIZE = 16

# NewsAPI Configuration
NEWS_API_KEY = "59593215cd46458c9214ba33b88c2831"

# Create directories if they don't exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)