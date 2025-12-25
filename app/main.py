from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

# Import configuration and utilities
import config
from model_loader import ModelLoader, check_model_files
from explainability.unified_explainer import UnifiedExplainer
from explainability.tfidf_explainer import TFIDFExplainer

# Setup logging
logging.basicConfig( level=logging.INFO )
logger = logging.getLogger( __name__ )

# Initialize FastAPI
app = FastAPI(
    title="Fake News Detection API",
    description="Explainable fake news detection system with multiple models",
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
    model: str  # "xlmr", "indicbert", or "tfidf"


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

        # Create TF-IDF explainer wrapper
        tfidf_explainer = TFIDFExplainer(
            models['tfidf_model'],
            models['tfidf_vectorizer']
        )

        # Initialize unified explainer
        explainer = UnifiedExplainer(
            xlmr_model=models['xlmr_model'],
            xlmr_tokenizer=models['xlmr_tokenizer'],
            indic_model=models['indic_model'],
            indic_tokenizer=models['indic_tokenizer'],
            tfidf_explainer=tfidf_explainer
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
        "message" : "Fake News Detection API",
        "status" : "running",
        "models_loaded" : models_loaded,
        "available_models" : ["xlmr", "indicbert", "tfidf"]
    }


@app.get( "/health" )
async def health_check () :
    """Health check endpoint"""
    return {
        "status" : "healthy" if models_loaded else "models_not_loaded",
        "models_loaded" : models_loaded
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
        valid_models = ["xlmr", "indicbert", "tfidf"]
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

    results = []
    for text in texts :
        try :
            explanation = explainer.explain( text, model )
            results.append( {
                "text" : text[:100] + "...",
                "prediction" : explanation['prediction'],
                "confidence" : explanation['confidence']
            } )
        except Exception as e :
            results.append( {
                "text" : text[:100] + "...",
                "error" : str( e )
            } )

    return {"results" : results}


if __name__ == "__main__" :
    import uvicorn

    uvicorn.run(
        "main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True
    )