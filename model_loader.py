import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path
import logging

logging.basicConfig( level=logging.INFO )
logger = logging.getLogger( __name__ )


class ModelLoader :
    """Handles loading of transformer models for fake news detection"""

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

    def load_all_models ( self ) :
        """Load all transformer models at once"""
        try :
            xlmr_model, xlmr_tokenizer = self.load_xlmr_model()
            indic_model, indic_tokenizer = self.load_indicbert_model()

            return {
                'xlmr_model' : xlmr_model,
                'xlmr_tokenizer' : xlmr_tokenizer,
                'indic_model' : indic_model,
                'indic_tokenizer' : indic_tokenizer
            }
        except Exception as e :
            logger.error( f"Error loading models: {e}" )
            raise


def check_model_files ( config ) :
    """Check if all required model files exist"""
    required_files = [
        (config.XLMR_MODEL_PATH, "XLM-RoBERTa model"),
        (config.INDIC_MODEL_PATH, "IndicBERT model")
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