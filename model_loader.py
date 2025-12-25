import torch
import pickle
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path
import logging

logging.basicConfig( level=logging.INFO )
logger = logging.getLogger( __name__ )


class ModelLoader :
    """Handles loading of all models for the fake news detection system"""

    def __init__ ( self, config ) :
        self.config = config
        self.device = torch.device( config.DEVICE if torch.cuda.is_available() else "cpu" )
        logger.info( f"Using device: {self.device}" )

    def load_xlmr_model ( self ) :
        """Load XLM-RoBERTa model and tokenizer"""
        try :
            logger.info( "Loading XLM-RoBERTa model..." )
            model = AutoModelForSequenceClassification.from_pretrained(
                str( self.config.XLMR_MODEL_PATH )
            )
            tokenizer = AutoTokenizer.from_pretrained(
                str( self.config.XLMR_MODEL_PATH )
            )
            model.to( self.device )
            model.eval()
            logger.info( "✓ XLM-RoBERTa model loaded successfully" )
            return model, tokenizer
        except Exception as e :
            logger.error( f"Error loading XLM-RoBERTa model: {e}" )
            raise

    def load_indicbert_model ( self ) :
        """Load IndicBERT model and tokenizer"""
        try :
            logger.info( "Loading IndicBERT model..." )
            model = AutoModelForSequenceClassification.from_pretrained(
                str( self.config.INDIC_MODEL_PATH )
            )
            tokenizer = AutoTokenizer.from_pretrained(
                str( self.config.INDIC_MODEL_PATH )
            )
            model.to( self.device )
            model.eval()
            logger.info( "✓ IndicBERT model loaded successfully" )
            return model, tokenizer
        except Exception as e :
            logger.error( f"Error loading IndicBERT model: {e}" )
            raise

    def load_tfidf_model ( self ) :
        """Load TF-IDF model, vectorizer, and training data"""
        try :
            logger.info( "Loading TF-IDF model..." )

            # Load the trained model
            with open( self.config.TFIDF_MODEL_PATH, 'rb' ) as f :
                model = pickle.load( f )

            # Load the vectorizer
            with open( self.config.TFIDF_VECTORIZER_PATH, 'rb' ) as f :
                vectorizer = pickle.load( f )

            # Load training data for SHAP explainer (if available)
            X_train = None
            if self.config.TFIDF_TRAIN_DATA_PATH.exists() :
                with open( self.config.TFIDF_TRAIN_DATA_PATH, 'rb' ) as f :
                    X_train = pickle.load( f )

            logger.info( "✓ TF-IDF model loaded successfully" )
            return model, vectorizer, X_train
        except Exception as e :
            logger.error( f"Error loading TF-IDF model: {e}" )
            raise

    def load_all_models ( self ) :
        """Load all models at once"""
        try :
            xlmr_model, xlmr_tokenizer = self.load_xlmr_model()
            indic_model, indic_tokenizer = self.load_indicbert_model()
            tfidf_model, tfidf_vectorizer, X_train = self.load_tfidf_model()

            return {
                'xlmr_model' : xlmr_model,
                'xlmr_tokenizer' : xlmr_tokenizer,
                'indic_model' : indic_model,
                'indic_tokenizer' : indic_tokenizer,
                'tfidf_model' : tfidf_model,
                'tfidf_vectorizer' : tfidf_vectorizer,
                'X_train' : X_train
            }
        except Exception as e :
            logger.error( f"Error loading models: {e}" )
            raise


def check_model_files ( config ) :
    """Check if all required model files exist"""
    required_files = [
        (config.XLMR_MODEL_PATH, "XLM-RoBERTa model"),
        (config.INDIC_MODEL_PATH, "IndicBERT model"),
        (config.TFIDF_MODEL_PATH, "TF-IDF model"),
        (config.TFIDF_VECTORIZER_PATH, "TF-IDF vectorizer"),
    ]

    missing_files = []
    for file_path, name in required_files :
        if not Path( file_path ).exists() :
            missing_files.append( f"  - {name}: {file_path}" )

    if missing_files :
        error_msg = "Missing required model files:\n" + "\n".join( missing_files )
        logger.error( error_msg )
        return False, error_msg

    return True, "All model files found"