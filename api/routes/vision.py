"""
Vision Emotion Recognition API Route.
Processes uploaded facial images and runs face detection/alignment.
Enforces the Zero Fake AI policy: classifies 7 core emotions using genuine
trained AlexNet checkpoint weights (models/vision/model_image_new50.pth).
If weights are absent, returns HTTP 503 rather than fabricating random guesses.
"""

import os
import tempfile
import logging
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse

from core.schemas import ModalityResult
from api.dependencies import get_vision_model, VisionEmotionModel, sanitize_modality_result

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["Prediction"])

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


@router.post("/vision", summary="Classify Emotion from Facial Image")
async def predict_vision_emotion(
    file: UploadFile = File(..., description="Facial image (JPG, PNG, or WebP)"),
    vision_model: VisionEmotionModel = Depends(get_vision_model),
) -> JSONResponse:
    """
    Evaluates facial image input.
    Detects faces, normalizes crops to 224x224, and performs 7-class emotion inference.
    ZERO FAKE AI POLICY:
    If trained visual weights are missing, this endpoint returns HTTP 503 Service
    Unavailable. No random guesses or heuristics are fabricated.
    """
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image file was not provided.",
        )

    _, ext = os.path.splitext(file.filename.lower())
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported image format '{ext}'. Allowed extensions: {list(ALLOWED_IMAGE_EXTENSIONS)}",
        )

    temp_fd, temp_path = tempfile.mkstemp(suffix=ext, prefix="vision_upload_")
    try:
        total_bytes = 0
        with os.fdopen(temp_fd, "wb") as temp_file:
            while chunk := await file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > MAX_IMAGE_SIZE_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"Image file exceeds maximum allowed limit of {MAX_IMAGE_SIZE_BYTES // (1024 * 1024)} MB.",
                    )
                temp_file.write(chunk)

        if total_bytes == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded image file is empty (0 bytes).",
            )

        # Run face preprocessing and checkpoint check
        result = vision_model.predict(temp_path)
        sanitized = sanitize_modality_result(result).model_dump()

        if not result.available:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    **sanitized,
                    "detail": "Vision emotion inference unavailable because the trained checkpoint (models/vision/model_image_new50.pth) is missing.",
                },
            )

        return JSONResponse(status_code=status.HTTP_200_OK, content=sanitized)

    finally:
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning("Failed to remove temporary image file %s: %s", temp_path, e)
