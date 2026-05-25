import json
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    XLMRobertaTokenizerFast,
)
from peft import PeftModel
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _is_full_model(path: Path) -> bool:
    """Return True if the directory contains a full HuggingFace model (not a LoRA adapter)."""
    has_weights = (path / "pytorch_model.bin").exists() or any(path.glob("model*.safetensors"))
    has_adapter  = (path / "adapter_config.json").exists()
    return has_weights and not has_adapter


def _is_lora_adapter(path: Path) -> bool:
    """Return True if the directory contains a LoRA adapter."""
    return (path / "adapter_config.json").exists()


class ModelLoader:
    """Handles loading of transformer models for fake news detection.

    Supports three checkpoint layouts automatically:
      1. Full model in <model_path>/checkpoint-pretrained  (or any checkpoint-* dir)
      2. LoRA adapter in a checkpoint-* sub-directory
      3. Base model only (no LoRA, no local full model)
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and "cuda" in config.DEVICE else "cpu"
        )
        logger.info(f"Using device: {self.device}")

    # --------------------------------------------------
    # Utility: find latest checkpoint sub-directory
    # --------------------------------------------------
    def _get_latest_checkpoint(self, model_path: Path):
        if not model_path.exists():
            return None
        checkpoints = sorted(
            [d for d in model_path.iterdir()
             if d.is_dir() and d.name.startswith("checkpoint-")]
        )
        return checkpoints[-1] if checkpoints else None

    # --------------------------------------------------
    # Core loader — handles full model OR LoRA adapter
    # --------------------------------------------------
    def _load_model(
        self,
        base_model_name: str,
        lora_path: Path = None,
        num_labels: int = None,
        fix_mistral_regex: bool = False,
    ):
        """
        Load a model using the best strategy detected from the checkpoint directory:

          • Full model checkpoint  → loaded directly (no base model needed from HF Hub)
          • LoRA adapter           → base model loaded from HF Hub, adapter merged on top
          • No checkpoint found    → base model loaded from HF Hub only
        """
        try:
            # --------------------------------------------------
            # Step 1: Detect what we have in lora_path
            # --------------------------------------------------
            checkpoint_dir  = None
            checkpoint_type = "none"   # "full" | "lora" | "none"

            if lora_path:
                lora_path = Path(lora_path)

                # Check the path itself first
                if _is_full_model(lora_path):
                    checkpoint_dir  = lora_path
                    checkpoint_type = "full"
                elif _is_lora_adapter(lora_path):
                    checkpoint_dir  = lora_path
                    checkpoint_type = "lora"
                else:
                    # Look one level deeper for checkpoint-* sub-dirs
                    latest = self._get_latest_checkpoint(lora_path)
                    if latest:
                        if _is_full_model(latest):
                            checkpoint_dir  = latest
                            checkpoint_type = "full"
                        elif _is_lora_adapter(latest):
                            checkpoint_dir  = latest
                            checkpoint_type = "lora"
                        else:
                            logger.warning(
                                f"Checkpoint found at {latest} but contains neither a full model "
                                f"nor a LoRA adapter — falling back to base model only."
                            )
                    else:
                        logger.warning(f"No checkpoints found in {lora_path} — using base model only.")

            logger.info(f"Checkpoint type detected: {checkpoint_type}"
                        + (f" at {checkpoint_dir}" if checkpoint_dir else ""))

            # --------------------------------------------------
            # Step 2: Load tokenizer  (with legacy-class fallback)
            # --------------------------------------------------
            # Some HuggingFace checkpoints (e.g. hamzb/roberta-fake-news-classification)
            # ship a tokenizer_config.json that declares "tokenizer_class": "XLMTokenizer"
            # even though the actual vocab/merges are RoBERTa-style.  XLMTokenizer
            # requires the optional `sacremoses` package; if that import fails we:
            #   1. Try again with use_fast=True (loads XLMRobertaTokenizerFast directly)
            #   2. If the checkpoint src itself fails, fall back to base_model_name
            tokenizer_src = str(checkpoint_dir) if checkpoint_dir else str(base_model_name)
            logger.info(f"Loading tokenizer from: {tokenizer_src}")

            def _load_tokenizer(src: str):
                """Try AutoTokenizer first; fall back to XLMRobertaTokenizerFast."""
                try:
                    return AutoTokenizer.from_pretrained(src, use_fast=True)
                except (ImportError, OSError) as e:
                    logger.warning(
                        f"AutoTokenizer failed for '{src}' ({e}); "
                        "retrying with XLMRobertaTokenizerFast directly."
                    )
                    return XLMRobertaTokenizerFast.from_pretrained(src)

            try:
                tokenizer = _load_tokenizer(tokenizer_src)
            except Exception as e:
                if tokenizer_src != str(base_model_name):
                    logger.warning(
                        f"Tokenizer load failed from checkpoint '{tokenizer_src}' ({e}); "
                        f"falling back to base model tokenizer: {base_model_name}"
                    )
                    tokenizer = _load_tokenizer(str(base_model_name))
                else:
                    raise

            # --------------------------------------------------
            # Step 3: Load model weights
            # --------------------------------------------------
            if checkpoint_type == "full":
                # Infer num_labels from the checkpoint's own config so the
                # classification head size is always correct.
                ckpt_cfg   = AutoConfig.from_pretrained(str(checkpoint_dir))
                num_labels = getattr(ckpt_cfg, "num_labels", num_labels or 2)
                logger.info(
                    f"Loading full model from checkpoint: {checkpoint_dir} "
                    f"(num_labels={num_labels})"
                )
                model = AutoModelForSequenceClassification.from_pretrained(
                    str(checkpoint_dir),
                    num_labels=num_labels,
                )

            else:
                # Need the base model from HF Hub (or local path)
                if num_labels is None:
                    base_cfg   = AutoConfig.from_pretrained(str(base_model_name))
                    num_labels = getattr(base_cfg, "num_labels", 2)
                    logger.info(f"Inferred num_labels={num_labels} from base model config")

                logger.info(f"Loading base model: {base_model_name} (num_labels={num_labels})")
                model = AutoModelForSequenceClassification.from_pretrained(
                    str(base_model_name),
                    num_labels=num_labels,
                )

                if checkpoint_type == "lora":
                    logger.info(f"Merging LoRA adapter from: {checkpoint_dir}")
                    try:
                        model = PeftModel.from_pretrained(
                            model,
                            str(checkpoint_dir),
                            is_trainable=False,
                        )
                        model = model.merge_and_unload()
                        logger.info("LoRA adapter merged successfully")
                    except RuntimeError as e:
                        logger.error(f"LoRA merge failed: {e}")
                        logger.warning(
                            "LoRA skipped due to label mismatch or incompatible head — "
                            "falling back to base model weights."
                        )

            model.to(self.device)
            model.eval()
            return model, tokenizer

        except Exception as e:
            logger.error(f"Error loading model '{base_model_name}': {e}")
            raise

    # --------------------------------------------------
    # XLM-RoBERTa (Binary)
    # --------------------------------------------------
    def load_xlmr_model(self):
        return self._load_model(
            base_model_name=self.config.XLMR_BASE_MODEL,
            lora_path=Path(self.config.XLMR_MODEL_PATH),
            num_labels=2,
            fix_mistral_regex=False,
        )

    # --------------------------------------------------
    # IndicBERT (AUTO labels)
    # --------------------------------------------------
    def load_indicbert_model(self):
        return self._load_model(
            base_model_name=self.config.INDICBERT_BASE_MODEL,
            lora_path=Path(self.config.INDICBERT_MODEL_PATH),
            num_labels=None,    # auto-detect from checkpoint or base config
            fix_mistral_regex=True,
        )

    # --------------------------------------------------
    # Load all models
    # --------------------------------------------------
    def load_all_models(self):
        xlmr_model,  xlmr_tokenizer  = self.load_xlmr_model()
        indic_model, indic_tokenizer = self.load_indicbert_model()

        return {
            "xlmr_model":       xlmr_model,
            "xlmr_tokenizer":   xlmr_tokenizer,
            "indic_model":      indic_model,
            "indic_tokenizer":  indic_tokenizer,
        }


# --------------------------------------------------
# Model file checker  (used by main.py at startup)
# --------------------------------------------------
def check_model_files(config):
    """
    Verify that each model path has something loadable.
    Accepts: a full-model checkpoint dir, a LoRA adapter dir,
             or a parent dir containing checkpoint-* sub-dirs of either type.
    """
    paths = [
        (Path(config.XLMR_MODEL_PATH),      "XLM-RoBERTa model"),
        (Path(config.INDICBERT_MODEL_PATH),  "IndicBERT model"),
    ]

    missing = []
    for path, name in paths:
        if not path.exists():
            missing.append(f"  - {name}: path does not exist → {path}")
            continue

        # Direct full model or adapter
        if _is_full_model(path) or _is_lora_adapter(path):
            logger.info(f"{name}: found at {path}")
            continue

        # Look for checkpoint sub-dirs
        checkpoints = [
            d for d in path.iterdir()
            if d.is_dir() and d.name.startswith("checkpoint-")
        ]
        valid = [d for d in checkpoints if _is_full_model(d) or _is_lora_adapter(d)]

        if valid:
            logger.info(f"{name}: found checkpoint(s) in {path}")
        else:
            missing.append(
                f"  - {name}: no loadable checkpoint found in {path} "
                f"(checked {len(checkpoints)} sub-dir(s))"
            )

    if missing:
        msg = "Warning: Some model files are missing or unrecognised:\n" + "\n".join(missing)
        logger.warning(msg)
        return False, msg

    logger.info("All required model files found")
    return True, "All model files found"