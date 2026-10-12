"""
Audio / Speech Emotion Recognition API Route.
Processes uploaded audio files using deep CNN acoustic inference with safe temporary file handling.
"""

import os
import shutil
import tempfile
import logging
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status

from core.schemas import ModalityResult
from api.dependencies import get_speech_model, SpeechEmotionModel, sanitize_modality_result

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["Prediction"])

ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac"}
MAX_AUDIO_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB


@router.post("/audio", response_model=ModalityResult, summary="Classify Emotion from Audio")
async def predict_audio_emotion(
    file: UploadFile = File(..., description="Audio file (WAV, MP3, OGG, or FLAC)"),
    speech_model: SpeechEmotionModel = Depends(get_speech_model),
) -> ModalityResult:
    """
    Classifies speech emotion using the trained deep CNN model (best_model_CNN.h5).
    Extracts 39-channel MFCC features from 16 kHz audio and maps to Ekman core emotions.
    Unsupported classes (disgust, fear, surprise) remain strictly 0.0.
    """
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio file was not provided.",
        )

    # Validate file extension
    _, ext = os.path.splitext(file.filename.lower())
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported audio format '{ext}'. Allowed extensions: {list(ALLOWED_AUDIO_EXTENSIONS)}",
        )

    # Temporary file storage workflow
    temp_fd, temp_path = tempfile.mkstemp(suffix=ext, prefix="audio_upload_")
    try:
        total_bytes = 0
        with os.fdopen(temp_fd, "wb") as temp_file:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                total_bytes += len(chunk)
                if total_bytes > MAX_AUDIO_SIZE_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"Audio file exceeds maximum allowed limit of {MAX_AUDIO_SIZE_BYTES // (1024 * 1024)} MB.",
                    )
                temp_file.write(chunk)

        if total_bytes == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded audio file is empty (0 bytes).",
            )

        # Run genuine model inference
        result = speech_model.predict(temp_path)

        if not result.available and result.status != "SILENT_OR_NEAR_SILENT_AUDIO":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Audio inference processing failed: {result.status}",
            )

        return sanitize_modality_result(result)

    finally:
        # Guarantee temporary file cleanup even if error occurs
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning("Failed to remove temporary audio file %s: %s", temp_path, e)
