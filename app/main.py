from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
import threading
from langdetect import detect

# Project imports
from app import config
from model_loader import ModelLoader, check_model_files
from explainability.unified_explainer import UnifiedExplainer

# --------------------------------------------------
# Logging Configuration
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger( __name__ )

# --------------------------------------------------
# FastAPI Application
# --------------------------------------------------
app = FastAPI(
    title="Multilingual Fake News Detection API",
    description="Explainable fake news detection using XLM-RoBERTa and IndicBERT with ensemble capabilities",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# --------------------------------------------------
# CORS Middleware
# --------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Request/Response Models
# --------------------------------------------------
class InputText( BaseModel ) :
    text: str
    model: Optional[str] = None


class PredictionResponse( BaseModel ) :
    label: str
    confidence: float
    model: str
    explanations: Dict[str, Any]


class EnsemblePredictionResponse( BaseModel ) :
    label: str
    confidence: float
    method: str
    details: Dict[str, Any]


class HealthResponse( BaseModel ) :
    status: str
    models_loaded: bool


class RootResponse( BaseModel ) :
    status: str
    models_loaded: bool
    available_models: list
    version: str


# --------------------------------------------------
# Global Variables
# --------------------------------------------------
explainer: Optional[UnifiedExplainer] = None
models_loaded: bool = False
loading_lock = threading.Lock()
loading_error: Optional[str] = None


# --------------------------------------------------
# Model Loading Function (Thread-safe)
# --------------------------------------------------
def load_models () :
    """Load ML models in background with error handling"""
    global explainer, models_loaded, loading_error

    with loading_lock :
        if models_loaded :
            return

        try :
            logger.info( "🚀 Background ML model loading started..." )

            # Check if model files exist
            files_exist, message = check_model_files( config )
            if not files_exist :
                loading_error = message
                logger.error( f"❌ {message}" )
                return

            # Load models
            loader = ModelLoader( config )
            models = loader.load_all_models()

            # Initialize explainer
            explainer = UnifiedExplainer(
                xlmr_model=models["xlmr_model"],
                xlmr_tokenizer=models["xlmr_tokenizer"],
                indic_model=models["indic_model"],
                indic_tokenizer=models["indic_tokenizer"],
            )

            models_loaded = True
            logger.info( "✅ ML models loaded successfully" )

        except Exception as e :
            loading_error = str( e )
            logger.exception( f"❌ Error loading models: {e}" )


# --------------------------------------------------
# Startup Event
# --------------------------------------------------
@app.on_event( "startup" )
def startup_event () :
    """Initialize model loading on startup"""
    threading.Thread( target=load_models, daemon=True ).start()
    logger.info( "⚡ Model loading initiated in background" )


# --------------------------------------------------
# Middleware: Request Logging
# --------------------------------------------------
@app.middleware( "http" )
async def log_requests ( request, call_next ) :
    """Log all incoming requests"""
    logger.info( f"➡️ {request.method} {request.url.path}" )
    response = await call_next( request )
    logger.info( f"✅ {request.method} {request.url.path} - Status: {response.status_code}" )
    return response


# --------------------------------------------------
# Utility Functions
# --------------------------------------------------
def auto_select_model ( text: str ) -> str :
    """Auto-detect language and select appropriate model"""
    try :
        lang = detect( text )
        logger.info( f"Detected language: {lang}" )

        # European languages -> XLM-RoBERTa
        european_langs = ["en", "fr", "de", "es", "it", "pt", "nl", "pl", "ru"]

        return "xlmr" if lang in european_langs else "indicbert"
    except Exception as e :
        logger.warning( f"Language detection failed: {e}. Defaulting to xlmr" )
        return "xlmr"


# --------------------------------------------------
# API Routes
# --------------------------------------------------

@app.get( "/", response_model=RootResponse )
def root () :
    """Root endpoint with API information"""
    return {
        "status" : "running",
        "models_loaded" : models_loaded,
        "available_models" : ["xlmr", "indicbert", "ensemble"],
        "version" : "2.0.0"
    }


@app.get( "/health", response_model=HealthResponse )
def health () :
    """Health check endpoint"""
    if loading_error and not models_loaded :
        raise HTTPException(
            status_code=503,
            detail=f"Model loading failed: {loading_error}"
        )

    return {
        "status" : "healthy" if models_loaded else "loading",
        "models_loaded" : models_loaded
    }


@app.post( "/predict", response_model=PredictionResponse )
def predict ( data: InputText ) :
    """
    Predict if news is fake or real using single model

    Args:
        data: InputText containing text and optional model selection

    Returns:
        PredictionResponse with label, confidence, model, and explanations
    """
    global explainer

    # Ensure models are loaded
    if not models_loaded :
        load_models()

    if not explainer :
        raise HTTPException(
            status_code=503,
            detail="Models not loaded yet. Please try again in a few moments."
        )

    # Validate input
    if not data.text.strip() :
        raise HTTPException(
            status_code=400,
            detail="Text input cannot be empty"
        )

    # Select model
    model_name = data.model or auto_select_model( data.text )

    if model_name not in ["xlmr", "indicbert"] :
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model: {model_name}. Choose 'xlmr' or 'indicbert'"
        )

    try :
        logger.info( f"Processing prediction with model: {model_name}" )
        result = explainer.explain( data.text, model_name )

        return {
            "label" : "Fake" if result["prediction"] == 1 else "Real",
            "confidence" : result["confidence"],
            "model" : model_name,
            "explanations" : result["explanations"]
        }

    except Exception as e :
        logger.exception( "Prediction error occurred" )
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str( e )}"
        )


@app.post( "/predict/ensemble", response_model=EnsemblePredictionResponse )
def ensemble_predict ( data: InputText ) :
    """
    Predict using ensemble of both models

    Args:
        data: InputText containing text to analyze

    Returns:
        EnsemblePredictionResponse with combined prediction and individual model details
    """
    global explainer

    # Ensure models are loaded
    if not models_loaded :
        load_models()

    if not explainer :
        raise HTTPException(
            status_code=503,
            detail="Models not loaded yet. Please try again in a few moments."
        )

    # Validate input
    if not data.text.strip() :
        raise HTTPException(
            status_code=400,
            detail="Text input cannot be empty"
        )

    try :
        logger.info( "Processing ensemble prediction" )
        result = explainer.ensemble_explain( data.text )

        return {
            "label" : "Fake" if result["prediction"] == 1 else "Real",
            "confidence" : result["confidence"],
            "method" : result["method"],
            "details" : result.get( "details", {} )
        }

    except Exception as e :
        logger.exception( "Ensemble prediction error occurred" )
        raise HTTPException(
            status_code=500,
            detail=f"Ensemble prediction failed: {str( e )}"
        )


@app.get( "/models/info" )
def models_info () :
    """Get information about loaded models"""
    return {
        "models_loaded" : models_loaded,
        "available_models" : {
            "xlmr" : {
                "name" : "XLM-RoBERTa",
                "description" : "Multilingual model for 100+ languages",
                "best_for" : ["English", "European languages", "Code-mixed text"]
            },
            "indicbert" : {
                "name" : "IndicBERT",
                "description" : "Specialized model for Indian languages",
                "best_for" : ["Hindi", "Tamil", "Telugu", "Other Indian languages"]
            },
            "ensemble" : {
                "name" : "Ensemble",
                "description" : "Combines both models for enhanced accuracy",
                "best_for" : ["Maximum accuracy", "Uncertain language detection"]
            }
        },
        "loading_error" : loading_error if loading_error else None
    }


@app.post( "/validate" )
def validate_text ( data: InputText ) :
    """Validate text and detect language"""
    if not data.text.strip() :
        raise HTTPException( status_code=400, detail="Text cannot be empty" )

    try :
        detected_lang = detect( data.text )
        recommended_model = auto_select_model( data.text )

        return {
            "text_length" : len( data.text ),
            "word_count" : len( data.text.split() ),
            "detected_language" : detected_lang,
            "recommended_model" : recommended_model,
            "valid" : True
        }
    except Exception as e :
        return {
            "text_length" : len( data.text ),
            "word_count" : len( data.text.split() ),
            "detected_language" : "unknown",
            "recommended_model" : "xlmr",
            "valid" : True,
            "note" : "Language detection failed, defaulting to XLM-R"
        }


# --------------------------------------------------
# Error Handlers
# --------------------------------------------------
@app.exception_handler( 404 )
async def not_found_handler ( request, exc ) :
    return {
        "error" : "Endpoint not found",
        "path" : request.url.path,
        "available_endpoints" : [
            "/",
            "/health",
            "/predict",
            "/predict/ensemble",
            "/models/info",
            "/validate"
        ]
    }


@app.exception_handler( 500 )
async def internal_error_handler ( request, exc ) :
    logger.exception( "Internal server error" )
    return {
        "error" : "Internal server error",
        "detail" : "An unexpected error occurred. Please try again."
    }


# --------------------------------------------------
# Main Entry Point
# --------------------------------------------------
if __name__ == "__main__" :
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )