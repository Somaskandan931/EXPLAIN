"""
Helper script to export your trained models in the correct format
Run this script to save your models after training
"""

import pickle
from pathlib import Path
import torch

# Create models directory if it doesn't exist
MODELS_DIR = Path( "models" )
MODELS_DIR.mkdir( exist_ok=True )

print( "=" * 60 )
print( "  Model Export Helper" )
print( "=" * 60 )


# ============================================================================
# 1. EXPORT TRANSFORMER MODELS (XLM-R, IndicBERT)
# ============================================================================

def export_transformer_model ( model, tokenizer, model_name ) :
    """
    Export a transformer model and its tokenizer

    Args:
        model: Your trained model (e.g., from transformers)
        tokenizer: The tokenizer used with the model
        model_name: Name to save as (e.g., 'xlmr_model', 'indicbert_model')
    """
    save_path = MODELS_DIR / model_name
    save_path.mkdir( exist_ok=True )

    print( f"\nExporting {model_name}..." )

    # Save model
    model.save_pretrained( save_path )
    print( f"  ✓ Model saved to: {save_path}" )

    # Save tokenizer
    tokenizer.save_pretrained( save_path )
    print( f"  ✓ Tokenizer saved to: {save_path}" )

    return save_path


# EXAMPLE USAGE FOR TRANSFORMER MODELS:
# Uncomment and modify with your actual model and tokenizer objects

"""
# For XLM-RoBERTa:
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load or use your trained model
xlmr_model = AutoModelForSequenceClassification.from_pretrained("your_model_path")
xlmr_tokenizer = AutoTokenizer.from_pretrained("your_tokenizer_path")

# Export
export_transformer_model(xlmr_model, xlmr_tokenizer, "xlmr_model")

# For IndicBERT (similar process):
indic_model = AutoModelForSequenceClassification.from_pretrained("your_indicbert_path")
indic_tokenizer = AutoTokenizer.from_pretrained("your_indicbert_path")

export_transformer_model(indic_model, indic_tokenizer, "indicbert_model")
"""


# ============================================================================
# 2. EXPORT TF-IDF MODEL
# ============================================================================

def export_tfidf_model ( model, vectorizer, X_train=None ) :
    """
    Export TF-IDF model and vectorizer

    Args:
        model: Your trained sklearn model (e.g., LogisticRegression, SVM)
        vectorizer: The TfidfVectorizer used for feature extraction
        X_train: Optional - Training data in vectorized form (for SHAP explanations)
    """
    print( "\nExporting TF-IDF model..." )

    # Save model
    model_path = MODELS_DIR / "tfidf_model.pkl"
    with open( model_path, 'wb' ) as f :
        pickle.dump( model, f )
    print( f"  ✓ Model saved to: {model_path}" )

    # Save vectorizer
    vectorizer_path = MODELS_DIR / "tfidf_vectorizer.pkl"
    with open( vectorizer_path, 'wb' ) as f :
        pickle.dump( vectorizer, f )
    print( f"  ✓ Vectorizer saved to: {vectorizer_path}" )

    # Save training data if provided (optional, for SHAP)
    if X_train is not None :
        train_data_path = MODELS_DIR / "tfidf_train_data.pkl"
        with open( train_data_path, 'wb' ) as f :
            pickle.dump( X_train, f )
        print( f"  ✓ Training data saved to: {train_data_path}" )

    return model_path, vectorizer_path


# EXAMPLE USAGE FOR TF-IDF MODEL:
# Uncomment and modify with your actual objects

"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Your trained model and vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_vectorized = tfidf_vectorizer.fit_transform(train_texts)

tfidf_model = LogisticRegression()
tfidf_model.fit(X_train_vectorized, train_labels)

# Export (X_train_vectorized is optional)
export_tfidf_model(tfidf_model, tfidf_vectorizer, X_train_vectorized)
"""


# ============================================================================
# 3. COMPLETE EXPORT EXAMPLE
# ============================================================================

def export_all_models ( xlmr_model=None, xlmr_tokenizer=None,
                        indic_model=None, indic_tokenizer=None,
                        tfidf_model=None, tfidf_vectorizer=None,
                        X_train=None ) :
    """
    Export all models at once

    Pass your trained models as arguments. Set to None if you don't have that model yet.
    """
    print( "=" * 60 )
    print( "  Exporting All Models" )
    print( "=" * 60 )

    exported = []

    # Export XLM-R if provided
    if xlmr_model is not None and xlmr_tokenizer is not None :
        export_transformer_model( xlmr_model, xlmr_tokenizer, "xlmr_model" )
        exported.append( "XLM-RoBERTa" )

    # Export IndicBERT if provided
    if indic_model is not None and indic_tokenizer is not None :
        export_transformer_model( indic_model, indic_tokenizer, "indicbert_model" )
        exported.append( "IndicBERT" )

    # Export TF-IDF if provided
    if tfidf_model is not None and tfidf_vectorizer is not None :
        export_tfidf_model( tfidf_model, tfidf_vectorizer, X_train )
        exported.append( "TF-IDF" )

    print( "\n" + "=" * 60 )
    print( f"  Export Complete!" )
    print( f"  Models exported: {', '.join( exported ) if exported else 'None'}" )
    print( "=" * 60 )


# ============================================================================
# 4. VERIFICATION
# ============================================================================

def verify_exported_models () :
    """Check if all required model files exist"""
    print( "\n" + "=" * 60 )
    print( "  Verifying Exported Models" )
    print( "=" * 60 )

    required_files = {
        "XLM-RoBERTa" : [
            MODELS_DIR / "xlmr_model" / "config.json",
            MODELS_DIR / "xlmr_model" / "pytorch_model.bin",
        ],
        "IndicBERT" : [
            MODELS_DIR / "indicbert_model" / "config.json",
            MODELS_DIR / "indicbert_model" / "pytorch_model.bin",
        ],
        "TF-IDF" : [
            MODELS_DIR / "tfidf_model.pkl",
            MODELS_DIR / "tfidf_vectorizer.pkl",
        ]
    }

    all_good = True
    for model_name, files in required_files.items() :
        print( f"\n{model_name}:" )
        model_complete = True
        for file_path in files :
            if file_path.exists() :
                print( f"  ✓ {file_path.name}" )
            else :
                print( f"  ❌ {file_path.name} - MISSING" )
                model_complete = False
                all_good = False

        if model_complete :
            print( f"  ✅ {model_name} is ready!" )

    if all_good :
        print( "\n" + "=" * 60 )
        print( "  ✅ All models exported successfully!" )
        print( "  You can now run the application." )
        print( "=" * 60 )
    else :
        print( "\n" + "=" * 60 )
        print( "  ⚠️  Some models are missing." )
        print( "  Export them using the functions above." )
        print( "=" * 60 )

    return all_good


# ============================================================================
# MAIN - UNCOMMENT TO USE
# ============================================================================

if __name__ == "__main__" :
    print( __doc__ )

    # OPTION 1: Export models individually as shown in examples above

    # OPTION 2: Export all at once (uncomment and provide your models)
    """
    export_all_models(
        xlmr_model=your_xlmr_model,
        xlmr_tokenizer=your_xlmr_tokenizer,
        indic_model=your_indic_model,
        indic_tokenizer=your_indic_tokenizer,
        tfidf_model=your_tfidf_model,
        tfidf_vectorizer=your_tfidf_vectorizer,
        X_train=your_X_train_vectorized  # Optional
    )
    """

    # Verify what's been exported
    verify_exported_models()

    print( "\n" + "=" * 60 )
    print( "  Instructions:" )
    print( "=" * 60 )
    print( """
1. Uncomment the relevant section above
2. Replace placeholder variables with your actual model objects
3. Run this script: python export_models.py
4. Verify all models are exported correctly
5. Run quick_start.py to test your setup
6. Start the application with uvicorn and streamlit
    """ )