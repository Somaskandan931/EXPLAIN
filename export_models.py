"""
Helper script to export your trained transformer models in the correct format
Run this script to save your XLM-RoBERTa and IndicBERT models after training
"""

from pathlib import Path

# Create models directory if it doesn't exist
MODELS_DIR = Path( "app/models" )
MODELS_DIR.mkdir(exist_ok=True)

print("="*60)
print("  Transformer Model Export Helper")
print("="*60)

# ============================================================================
# EXPORT TRANSFORMER MODELS (XLM-RoBERTa and IndicBERT)
# ============================================================================

def export_transformer_model(model, tokenizer, model_name):
    """
    Export a transformer model and its tokenizer

    Args:
        model: Your trained model (from transformers library)
        tokenizer: The tokenizer used with the model
        model_name: Name to save as (e.g., 'xlmr_model', 'indicbert_model')
    """
    save_path = MODELS_DIR / model_name
    save_path.mkdir(exist_ok=True)

    print(f"\nExporting {model_name}...")

    # Save model
    model.save_pretrained(save_path)
    print(f"  ✓ Model saved to: {save_path}")

    # Save tokenizer
    tokenizer.save_pretrained(save_path)
    print(f"  ✓ Tokenizer saved to: {save_path}")

    return save_path

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

# EXAMPLE 1: Export XLM-RoBERTa Model
# Uncomment and modify with your actual model and tokenizer objects

"""
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load or use your trained XLM-RoBERTa model
xlmr_model = AutoModelForSequenceClassification.from_pretrained("your_xlmr_path")
xlmr_tokenizer = AutoTokenizer.from_pretrained("your_xlmr_tokenizer_path")

# Export
export_transformer_model(xlmr_model, xlmr_tokenizer, "xlmr_model")
"""

# EXAMPLE 2: Export IndicBERT Model
# Uncomment and modify with your actual model and tokenizer objects

"""
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load or use your trained IndicBERT model
indic_model = AutoModelForSequenceClassification.from_pretrained("your_indicbert_path")
indic_tokenizer = AutoTokenizer.from_pretrained("your_indicbert_tokenizer_path")

# Export
export_transformer_model(indic_model, indic_tokenizer, "indicbert_model")
"""

# ============================================================================
# COMPLETE EXPORT EXAMPLE
# ============================================================================

def export_all_models(xlmr_model=None, xlmr_tokenizer=None,
                      indic_model=None, indic_tokenizer=None):
    """
    Export all models at once

    Pass your trained models as arguments. Set to None if you don't have that model yet.
    """
    print("="*60)
    print("  Exporting All Models")
    print("="*60)

    exported = []

    # Export XLM-R if provided
    if xlmr_model is not None and xlmr_tokenizer is not None:
        export_transformer_model(xlmr_model, xlmr_tokenizer, "xlmr_model")
        exported.append("XLM-RoBERTa")
    else:
        print("\n⚠️  XLM-RoBERTa model not provided, skipping...")

    # Export IndicBERT if provided
    if indic_model is not None and indic_tokenizer is not None:
        export_transformer_model(indic_model, indic_tokenizer, "indicbert_model")
        exported.append("IndicBERT")
    else:
        print("\n⚠️  IndicBERT model not provided, skipping...")

    print("\n" + "="*60)
    if exported:
        print(f"  Export Complete!")
        print(f"  Models exported: {', '.join(exported)}")
    else:
        print("  No models exported.")
        print("  Please provide your trained models to export.")
    print("="*60)

# ============================================================================
# VERIFICATION
# ============================================================================

def verify_exported_models():
    """Check if all required model files exist"""
    print("\n" + "="*60)
    print("  Verifying Exported Models")
    print("="*60)

    required_files = {
        "XLM-RoBERTa": [
            MODELS_DIR / "xlmr_model" / "config.json",
            MODELS_DIR / "xlmr_model" / "pytorch_model.bin",
        ],
        "IndicBERT": [
            MODELS_DIR / "indicbert_model" / "config.json",
            MODELS_DIR / "indicbert_model" / "pytorch_model.bin",
        ]
    }

    all_good = True
    for model_name, files in required_files.items():
        print(f"\n{model_name}:")
        model_complete = True
        for file_path in files:
            if file_path.exists():
                print(f"  ✓ {file_path.name}")
            else:
                print(f"  ❌ {file_path.name} - MISSING")
                model_complete = False
                all_good = False

        if model_complete:
            print(f"  ✅ {model_name} is ready!")

    if all_good:
        print("\n" + "="*60)
        print("  ✅ All models exported successfully!")
        print("  You can now run the application.")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("  ⚠️  Some models are missing.")
        print("  Export them using the functions above.")
        print("="*60)

    return all_good

# ============================================================================
# STEP-BY-STEP GUIDE
# ============================================================================

def print_export_guide():
    """Print step-by-step guide for exporting models"""
    print("\n" + "="*60)
    print("  Step-by-Step Export Guide")
    print("="*60)
    print("""
STEP 1: Load Your Trained Models
---------------------------------
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load XLM-RoBERTa
xlmr_model = AutoModelForSequenceClassification.from_pretrained("path/to/your/xlmr")
xlmr_tokenizer = AutoTokenizer.from_pretrained("path/to/your/xlmr")

# Load IndicBERT
indic_model = AutoModelForSequenceClassification.from_pretrained("path/to/your/indicbert")
indic_tokenizer = AutoTokenizer.from_pretrained("path/to/your/indicbert")

STEP 2: Export Models
----------------------
# Option A: Export individually
export_transformer_model(xlmr_model, xlmr_tokenizer, "xlmr_model")
export_transformer_model(indic_model, indic_tokenizer, "indicbert_model")

# Option B: Export all at once
export_all_models(
    xlmr_model=xlmr_model,
    xlmr_tokenizer=xlmr_tokenizer,
    indic_model=indic_model,
    indic_tokenizer=indic_tokenizer
)

STEP 3: Verify Export
----------------------
verify_exported_models()

STEP 4: Run Application
------------------------
# Start API
python -m uvicorn app.main:app --reload

# Start UI
streamlit run ui/streamlit_app.py
    """)
    print("="*60)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print(__doc__)

    # Print guide
    print_export_guide()

    # Verify what's been exported
    print("\nChecking current model files...")
    verify_exported_models()

    print("\n" + "="*60)
    print("  Next Steps:")
    print("="*60)
    print("""
1. Uncomment the relevant section above
2. Replace placeholder variables with your actual model objects
3. Run this script: python export_models.py
4. Verify all models are exported correctly
5. Run quick_start.py to test your setup
6. Start the application
    """)