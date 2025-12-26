import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
)
from peft import PeftModel
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelLoader:
    """Handles loading of (optional LoRA) transformer models for fake news detection"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and "cuda" in config.DEVICE else "cpu"
        )
        logger.info(f"Using device: {self.device}")

    # --------------------------------------------------
    # Utility: find latest checkpoint
    # --------------------------------------------------
    def _get_latest_checkpoint(self, model_path: Path):
        if not model_path.exists():
            return None

        checkpoints = sorted(
            [
                d for d in model_path.iterdir()
                if d.is_dir() and d.name.startswith("checkpoint-")
            ]
        )
        return checkpoints[-1] if checkpoints else None

    # --------------------------------------------------
    # Generic SAFE model loader
    # --------------------------------------------------
    def _load_model(
        self,
        base_model_name: str,
        lora_path: Path = None,
        num_labels: int = None,
        fix_mistral_regex: bool = False,
    ):
        """
        Load base model and optionally merge LoRA adapter.
        LoRA is skipped automatically if label mismatch occurs.
        """
        try:
            logger.info(f"Loading base model: {base_model_name}")

            # --------------------------------------------------
            # Infer num_labels safely
            # --------------------------------------------------
            if num_labels is None:
                base_config = AutoConfig.from_pretrained(str(base_model_name))
                num_labels = getattr(base_config, "num_labels", 2)
                logger.info(f"Inferred num_labels={num_labels} from base model config")

            model = AutoModelForSequenceClassification.from_pretrained(
                str(base_model_name),
                num_labels=num_labels,
            )

            tokenizer = AutoTokenizer.from_pretrained(
                str(base_model_name),
                fix_mistral_regex=fix_mistral_regex,
            )

            # --------------------------------------------------
            # Load LoRA adapter (SAFE)
            # --------------------------------------------------
            if lora_path:
                latest_checkpoint = self._get_latest_checkpoint(lora_path)
                if latest_checkpoint:
                    logger.info(f"Found LoRA checkpoint: {latest_checkpoint}")
                    try:
                        model = PeftModel.from_pretrained(
                            model,
                            str(latest_checkpoint),
                            is_trainable=False,
                        )
                        model = model.merge_and_unload()
                        logger.info("LoRA adapter merged successfully")
                    except RuntimeError as e:
                        logger.error(f"LoRA merge failed: {e}")
                        logger.warning(
                            "LoRA skipped due to label mismatch or incompatible head"
                        )
                else:
                    logger.warning(f"No LoRA checkpoints found in {lora_path}")

            model.to(self.device)
            model.eval()

            return model, tokenizer

        except Exception as e:
            logger.error(f"Error loading model {base_model_name}: {e}")
            raise

    # --------------------------------------------------
    # XLM-RoBERTa (Binary)
    # --------------------------------------------------
    def load_xlmr_model(self):
        return self._load_model(
            base_model_name=self.config.XLMR_BASE_MODEL,
            lora_path=Path(self.config.XLMR_MODEL_PATH),
            num_labels=2,          # binary
            fix_mistral_regex=False,
        )

    # --------------------------------------------------
    # IndicBERT (AUTO labels + regex fix)
    # --------------------------------------------------
    def load_indicbert_model(self):
        return self._load_model(
            base_model_name=self.config.INDICBERT_BASE_MODEL,
            lora_path=Path(self.config.INDICBERT_MODEL_PATH),
            num_labels=None,       # auto-detect
            fix_mistral_regex=True,
        )

    # --------------------------------------------------
    # Load all models
    # --------------------------------------------------
    def load_all_models(self):
        xlmr_model, xlmr_tokenizer = self.load_xlmr_model()
        indic_model, indic_tokenizer = self.load_indicbert_model()

        return {
            "xlmr_model": xlmr_model,
            "xlmr_tokenizer": xlmr_tokenizer,
            "indic_model": indic_model,
            "indic_tokenizer": indic_tokenizer,
        }


# --------------------------------------------------
# Model file checker
# --------------------------------------------------
def check_model_files(config):
    """Check if required LoRA model files exist, warn if missing"""

    required_paths = [
        (config.XLMR_MODEL_PATH, "XLM-RoBERTa LoRA model"),
        (config.INDICBERT_MODEL_PATH, "IndicBERT LoRA model"),
    ]

    missing = []
    for path, name in required_paths:
        path = Path(path)
        if not path.exists() or not any(path.iterdir()):
            missing.append(f"  - {name}: missing or empty at {path}")
        else:
            checkpoints = [
                d for d in path.iterdir()
                if d.is_dir() and d.name.startswith("checkpoint-")
            ]
            if not checkpoints:
                missing.append(f"  - {name}: no checkpoints found in {path}")

    if missing:
        msg = "Warning: Some LoRA model files are missing:\n" + "\n".join(missing)
        logger.warning(msg)
        return False, msg

    logger.info("All required model files found")
    return True, "All model files found"
