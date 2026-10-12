"""
Health and System Status API Route.
Exposes readiness and operational status across Text, Speech, and Vision pipelines.
"""

from fastapi import APIRouter, Depends
from core.schemas import HealthStatus
from api.dependencies import (
    get_text_model,
    get_speech_model,
    get_vision_model,
    TextEmotionModel,
    SpeechEmotionModel,
    VisionEmotionModel,
)

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthStatus, summary="Check API and Model Readiness")
def check_health(
    text_model: TextEmotionModel = Depends(get_text_model),
    speech_model: SpeechEmotionModel = Depends(get_speech_model),
    vision_model: VisionEmotionModel = Depends(get_vision_model),
) -> HealthStatus:
    """
    Returns structured system health status and per-modality model readiness.
    Accurately reflects the current operational state:
    - Text: Operational (HuggingFace RoBERTa)
    - Speech: Operational (Deep CNN H5 model)
    - Vision: Operational (Trained AlexNet model_image_new50.pth)
    """
    text_ok = bool(text_model._is_loaded)
    speech_ok = bool(speech_model._is_loaded)
    vision_ok = bool(vision_model._is_loaded)

    # If all modalities are operational, overall status is ok
    overall_status = "ok" if (text_ok and speech_ok and vision_ok) else "degraded"

    models_info = {
        "text": {
            "operational": text_ok,
            "model_name": text_model.model_name,
            "framework": "transformers",
            "supported_classes": 7,
        },
        "audio": {
            "operational": speech_ok,
            "model_architecture": "CNN",
            "weights": "best_model_CNN.h5",
            "supported_classes": ["neutral", "angry", "happy", "sad"],
        },
        "vision": {
            "operational": vision_ok,
            "status": "ok" if vision_ok else "VISION INFERENCE BLOCKED — TRAINED CHECKPOINT MISSING",
            "reason": None if vision_ok else (vision_model.load_error or "Checkpoint missing"),
            "face_detection_available": True,
        },
    }

    return HealthStatus(
        status=overall_status,
        service="emotion-recognition-api",
        version="1.0.0",
        device=text_model.device,
        models=models_info,
    )
