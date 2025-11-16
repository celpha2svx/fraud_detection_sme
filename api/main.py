from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import TransactionRequest, FraudPrediction
from api.predictor import FraudPredictor
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize app
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time e-commerce fraud detection for Nigerian SMEs",
    version="1.0.0"
)

# CORS (allow frontend to call API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor
predictor = FraudPredictor()


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    logger.info("Loading fraud detection model...")
    predictor.load_model()
    logger.info("✅ API ready!")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "Fraud Detection API",
        "version": "1.0.0"
    }


@app.post("/predict", response_model=FraudPrediction)
async def predict_fraud(transaction: TransactionRequest):
    """
    Predict fraud probability for a transaction.

    Returns fraud score, risk level, and recommendation.
    """
    try:
        result = predictor.predict(transaction.dict())
        return FraudPrediction(**result)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "model_loaded": predictor.model is not None,
        "device": str(predictor.device)
    }