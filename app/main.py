from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

# Import configuration and utilities
import config
from model_loader import ModelLoader, check_model_files
from explainability.unified_explainer import UnifiedExplainer

# Setup logging
logging.basicConfig( level=logging.INFO )
logger = logging.getLogger( __name__ )

# Initialize FastAPI
app = FastAPI(
    title="Multilingual Fake News Detection API",
    description="Explainable fake news detection using XLM-RoBERTa and IndicBERT",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class InputText( BaseModel ) :
    text: str
    model: str  # "xlmr" or "indicbert"


class PredictionResponse( BaseModel ) :
    prediction: int
    confidence: float
    method: str
    tokens: list


# Global variables for models
explainer = None
models_loaded = False


@app.on_event( "startup" )
async def startup_event () :
    """Load models on startup"""
    global explainer, models_loaded

    try :
        logger.info( "Starting model loading..." )

        # Check if model files exist
        files_exist, message = check_model_files( config )
        if not files_exist :
            logger.error( message )
            logger.warning( "API will start but predictions will fail until models are available" )
            return

        # Load all models
        loader = ModelLoader( config )
        models = loader.load_all_models()

        # Initialize unified explainer
        explainer = UnifiedExplainer(
            xlmr_model=models['xlmr_model'],
            xlmr_tokenizer=models['xlmr_tokenizer'],
            indic_model=models['indic_model'],
            indic_tokenizer=models['indic_tokenizer']
        )

        models_loaded = True
        logger.info( "✓ All models loaded successfully!" )

    except Exception as e :
        logger.error( f"Failed to load models: {e}" )
        logger.warning( "API will start but predictions will fail until models are loaded" )


@app.get( "/" )
async def root () :
    """Root endpoint"""
    return {
        "message" : "Multilingual Fake News Detection API",
        "status" : "running",
        "models_loaded" : models_loaded,
        "available_models" : ["xlmr", "indicbert"],
        "description" : "XLM-RoBERTa and IndicBERT for multilingual fake news detection"
    }


@app.get( "/health" )
async def health_check () :
    """Health check endpoint"""
    return {
        "status" : "healthy" if models_loaded else "models_not_loaded",
        "models_loaded" : models_loaded,
        "available_models" : {
            "xlmr" : models_loaded,
            "indicbert" : models_loaded
        }
    }


@app.post( "/predict", response_model=PredictionResponse )
async def predict ( input_data: InputText ) :
    """
    Predict whether text is fake news

    Args:
        input_data: InputText object with text and model name

    Returns:
        PredictionResponse with prediction, confidence, and explanation
    """
    if not models_loaded :
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please check server logs and ensure model files are available."
        )

    try :
        # Validate model name
        valid_models = ["xlmr", "indicbert"]
        if input_data.model not in valid_models :
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model. Choose from: {valid_models}"
            )

        # Get explanation
        explanation = explainer.explain( input_data.text, input_data.model )

        # Map prediction to label if needed
        prediction_label = explanation['prediction']
        if isinstance( prediction_label, int ) :
            # Assuming 0 = Real, 1 = Fake
            prediction_label = "Fake" if prediction_label == 1 else "Real"

        # Return response in expected format
        return {
            "prediction" : prediction_label,
            "confidence" : explanation['confidence'],
            "method" : explanation['method'],
            "tokens" : explanation.get( 'tokens', [] )
        }

    except Exception as e :
        logger.error( f"Prediction error: {e}" )
        raise HTTPException( status_code=500, detail=str( e ) )


@app.post( "/batch_predict" )
async def batch_predict ( texts: list[str], model: str = "xlmr" ) :
    """Batch prediction endpoint"""
    if not models_loaded :
        raise HTTPException(
            status_code=503,
            detail="Models not loaded"
        )

    # Validate model
    valid_models = ["xlmr", "indicbert"]
    if model not in valid_models :
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model. Choose from: {valid_models}"
        )

    results = []
    for text in texts :
        try :
            explanation = explainer.explain( text, model )
            prediction_label = "Fake" if explanation['prediction'] == 1 else "Real"
            results.append( {
                "text" : text[:100] + "..." if len( text ) > 100 else text,
                "prediction" : prediction_label,
                "confidence" : explanation['confidence']
            } )
        except Exception as e :
            results.append( {
                "text" : text[:100] + "..." if len( text ) > 100 else text,
                "error" : str( e )
            } )

    return {"results" : results, "model_used" : model}


@app.get( "/models" )
async def get_models_info () :
    """Get information about available models"""
    return {
        "models" : [
            {
                "name" : "xlmr",
                "full_name" : "XLM-RoBERTa",
                "description" : "Cross-lingual RoBERTa model trained on 100+ languages",
                "strengths" : [
                    "Excellent multilingual performance",
                    "Strong on code-mixed content (Hinglish)",
                    "Robust cross-lingual transfer"
                ],
                "languages" : "100+ including English, Hindi, and regional Indian languages"
            },
            {
                "name" : "indicbert",
                "full_name" : "IndicBERT",
                "description" : "BERT model specialized for Indian languages",
                "strengths" : [
                    "Optimized for 12 major Indian languages",
                    "Better performance on Hindi and regional content",
                    "Trained on Indic language corpora"
                ],
                "languages" : "12 major Indian languages including Hindi, Bengali, Tamil, Telugu"
            }
        ],
        "loaded" : models_loaded
    }


if __name__ == "__main__" :
    import uvicorn

    uvicorn.run(
        "main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True
    )