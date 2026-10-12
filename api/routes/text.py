"""
Text Emotion Recognition API Route.
Receives raw text snippets and runs real deep RoBERTa transformer inference.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from core.schemas import ModalityResult, TextPredictionRequest
from api.dependencies import get_text_model, TextEmotionModel, sanitize_modality_result

router = APIRouter(prefix="/predict", tags=["Prediction"])

MAX_TEXT_LENGTH = 10000


@router.post("/text", response_model=ModalityResult, summary="Classify Emotion from Text")
def predict_text_emotion(
    request: TextPredictionRequest,
    text_model: TextEmotionModel = Depends(get_text_model),
) -> ModalityResult:
    """
    Classifies natural language text into the 7-class Ekman emotion ontology.
    Uses deep transformer sequence classification (j-hartmann/emotion-english-distilroberta-base).
    """
    raw_text = request.text.strip()
    if not raw_text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Text field cannot be empty or contain only whitespace.",
        )

    if len(raw_text) > MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Text length exceeds maximum allowed limit of {MAX_TEXT_LENGTH} characters.",
        )

    result = text_model.predict(raw_text)

    if not result.available:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Text model inference failed: {result.status}",
        )

    return sanitize_modality_result(result)
