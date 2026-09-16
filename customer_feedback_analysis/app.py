import os
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator

from src.model_loader import load_pipeline_components
from src.predictor import SpamPredictor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("spam_guard.app")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Global predictor instance
predictor: Optional[SpamPredictor] = None
model_metadata: Dict[str, Any] = {
    "status": "uninitialized",
    "model_path": None,
    "vectorizer_path": None,
    "error": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context to safely load model and vectorizer on startup."""
    global predictor, model_metadata
    logger.info("Initializing SpamGuard AI application...")
    try:
        components = load_pipeline_components()
        predictor = SpamPredictor(
            model=components["model"],
            vectorizer=components["vectorizer"],
        )
        model_metadata["status"] = "ready"
        model_metadata["model_path"] = components["model_path"]
        model_metadata["vectorizer_path"] = components["vectorizer_path"]
        logger.info(
            "SpamGuard AI pipeline initialized successfully. Model: %s, Vectorizer: %s",
            components["model_path"],
            components["vectorizer_path"],
        )
    except Exception as e:
        logger.error("Failed to initialize ML pipeline: %s", e, exc_info=True)
        model_metadata["status"] = "error"
        model_metadata["error"] = str(e)
    yield
    logger.info("Shutting down SpamGuard AI application.")


app = FastAPI(
    title="SpamGuard AI",
    description="Intelligent Message Classification System powered by Multinomial Naive Bayes",
    version="1.0.0",
    lifespan=lifespan,
)

# Mount static assets
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Templates
if not os.path.exists(TEMPLATES_DIR):
    os.makedirs(TEMPLATES_DIR, exist_ok=True)
templates = Jinja2Templates(directory=TEMPLATES_DIR)


# Pydantic Schemas
class MessageRequest(BaseModel):
    message: str = Field(
        ...,
        description="The email or SMS message to classify.",
        min_length=1,
        max_length=5000,
        example="Congratulations! You won a free lottery ticket.",
    )

    @field_validator("message")
    @classmethod
    def check_non_empty(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("Message cannot be blank or contain only whitespace.")
        return trimmed


class PredictionResponse(BaseModel):
    message: str
    prediction: str
    confidence: float
    confidence_percentage: float
    is_spam: bool
    label: str
    probabilities: Dict[str, float]
    cleaned_text: str
    explanation: str


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    """Render the main SpamGuard AI web dashboard."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "system_status": model_metadata["status"],
        },
    )


@app.get("/health")
async def health_check():
    """Health check endpoint exposing system and ML pipeline status."""
    return {
        "status": "healthy" if model_metadata["status"] == "ready" else "degraded",
        "pipeline": model_metadata,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_message(payload: MessageRequest):
    """
    Classify a message as Spam or Not Spam (Ham).
    Returns the prediction label, confidence score, class probabilities, and explanation.
    """
    global predictor
    if predictor is None or model_metadata["status"] != "ready":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Machine Learning model is not available: {model_metadata.get('error', 'Model not initialized')}",
        )

    try:
        result = predictor.predict(payload.message)
        return result
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )
    except Exception as e:
        logger.error("Prediction error for message: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal prediction failure: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
