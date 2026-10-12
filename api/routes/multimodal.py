"""
Multi-Modal Emotion Recognition API Route.
Receives arbitrary combinations of Text, Audio, and Image inputs, invokes the respective
models, and executes confidence-aware late decision fusion.
"""

import os
import tempfile
import logging
from typing import Optional
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status
from fastapi.responses import JSONResponse

from core.schemas import ModalityResult, FusedEmotionResult, FusionEvaluateRequest
from api.dependencies import (
    get_text_model,
    get_speech_model,
    get_vision_model,
    get_fusion_engine,
    TextEmotionModel,
    SpeechEmotionModel,
    VisionEmotionModel,
    MultiModalFusionEngine,
    sanitize_fused_result,
)
from api.routes.audio import ALLOWED_AUDIO_EXTENSIONS, MAX_AUDIO_SIZE_BYTES
from api.routes.vision import ALLOWED_IMAGE_EXTENSIONS, MAX_IMAGE_SIZE_BYTES
from api.routes.text import MAX_TEXT_LENGTH

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post("/multimodal", response_model=FusedEmotionResult, summary="Multi-Modal Emotion Recognition")
async def predict_multimodal_emotion(
    text: Optional[str] = Form(None, description="Optional raw text string"),
    audio: Optional[UploadFile] = File(None, description="Optional audio file (WAV, MP3, OGG, FLAC)"),
    image: Optional[UploadFile] = File(None, description="Optional facial image file (JPG, PNG, WebP)"),
    text_model: TextEmotionModel = Depends(get_text_model),
    speech_model: SpeechEmotionModel = Depends(get_speech_model),
    vision_model: VisionEmotionModel = Depends(get_vision_model),
    fusion_engine: MultiModalFusionEngine = Depends(get_fusion_engine),
):
    """
    Primary Multi-Modal endpoint.
    Accepts any combination of Text, Audio, and Image inputs.
    - At least one input must be provided.
    - Runs genuine deep models for each supplied input.
    - Fuses available signals using confidence-weighted late decision fusion.
    - If Vision is supplied, its unavailable status is honestly reported without failing the request.
    """
    has_text = bool(text and text.strip())
    has_audio = bool(audio and audio.filename)
    has_image = bool(image and image.filename)

    if not (has_text or has_audio or has_image):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one modality input ('text', 'audio', or 'image') must be provided.",
        )

    text_result: Optional[ModalityResult] = None
    audio_result: Optional[ModalityResult] = None
    vision_result: Optional[ModalityResult] = None

    temp_audio_path: Optional[str] = None
    temp_image_path: Optional[str] = None

    try:
        # 1. Process Text
        if has_text:
            cleaned_text = text.strip()
            if len(cleaned_text) > MAX_TEXT_LENGTH:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Text length exceeds maximum allowed limit of {MAX_TEXT_LENGTH} characters.",
                )
            text_result = text_model.predict(cleaned_text)

        # 2. Process Audio
        if has_audio:
            _, ext = os.path.splitext(audio.filename.lower())
            if ext not in ALLOWED_AUDIO_EXTENSIONS:
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail=f"Unsupported audio format '{ext}'. Allowed extensions: {list(ALLOWED_AUDIO_EXTENSIONS)}",
                )

            temp_fd, temp_audio_path = tempfile.mkstemp(suffix=ext, prefix="multi_audio_")
            total_audio_bytes = 0
            with os.fdopen(temp_fd, "wb") as temp_file:
                while chunk := await audio.read(1024 * 1024):
                    total_audio_bytes += len(chunk)
                    if total_audio_bytes > MAX_AUDIO_SIZE_BYTES:
                        raise HTTPException(
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail=f"Audio file exceeds maximum allowed limit of {MAX_AUDIO_SIZE_BYTES // (1024 * 1024)} MB.",
                        )
                    temp_file.write(chunk)

            if total_audio_bytes > 0:
                audio_result = speech_model.predict(temp_audio_path)

        # 3. Process Image / Vision
        if has_image:
            _, ext = os.path.splitext(image.filename.lower())
            if ext not in ALLOWED_IMAGE_EXTENSIONS:
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail=f"Unsupported image format '{ext}'. Allowed extensions: {list(ALLOWED_IMAGE_EXTENSIONS)}",
                )

            temp_fd, temp_image_path = tempfile.mkstemp(suffix=ext, prefix="multi_vision_")
            total_image_bytes = 0
            with os.fdopen(temp_fd, "wb") as temp_file:
                while chunk := await image.read(1024 * 1024):
                    total_image_bytes += len(chunk)
                    if total_image_bytes > MAX_IMAGE_SIZE_BYTES:
                        raise HTTPException(
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail=f"Image file exceeds maximum allowed limit of {MAX_IMAGE_SIZE_BYTES // (1024 * 1024)} MB.",
                        )
                    temp_file.write(chunk)

            if total_image_bytes > 0:
                vision_result = vision_model.predict(temp_image_path)

        # 4. Execute Multi-Modal Fusion
        fused = fusion_engine.fuse(
            text_result=text_result,
            audio_result=audio_result,
            vision_result=vision_result,
        )

        sanitized_fused = sanitize_fused_result(fused)

        # If no valid modality was available (e.g. only vision provided and checkpoint missing)
        if not sanitized_fused.available:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=sanitized_fused.model_dump(),
            )

        return sanitized_fused

    finally:
        # Guarantee cleanup of temporary upload files
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
            except Exception as e:
                logger.warning("Failed to remove temp audio file %s: %s", temp_audio_path, e)

        if temp_image_path and os.path.exists(temp_image_path):
            try:
                os.unlink(temp_image_path)
            except Exception as e:
                logger.warning("Failed to remove temp image file %s: %s", temp_image_path, e)


@router.post("/fuse", response_model=FusedEmotionResult, summary="Fuse Pre-Evaluated Modality Results")
async def fuse_evaluated_modalities(
    payload: FusionEvaluateRequest,
    fusion_engine: MultiModalFusionEngine = Depends(get_fusion_engine),
) -> FusedEmotionResult:
    """
    Directly invokes the existing MultiModalFusionEngine.fuse_dict on a dictionary of
    genuine ModalityResult objects.
    Enables low-latency real-time multimodal fusion without redundant re-inference.
    """
    if not payload.results:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one modality result must be provided for fusion.",
        )
    fused = fusion_engine.fuse_dict(payload.results)
    sanitized_fused = sanitize_fused_result(fused)
    return sanitized_fused

