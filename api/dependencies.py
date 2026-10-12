"""
FastAPI Dependency Injection and Model Lifecycle Management.
Provides application-level singletons for Text, Speech, Vision, and Fusion engines.
Ensures zero redundant model re-instantiation across HTTP requests.
Enforces file safety and prevents filesystem paths from leaking in API responses.
"""

import os
import re
import logging
from typing import Optional, Dict, Any

from core.text.model import TextEmotionModel
from core.speech.model import SpeechEmotionModel
from core.vision.model import VisionEmotionModel
from core.fusion.engine import MultiModalFusionEngine
from core.schemas import ModalityResult, FusedEmotionResult

logger = logging.getLogger(__name__)

# Singleton holders
_text_model: Optional[TextEmotionModel] = None
_speech_model: Optional[SpeechEmotionModel] = None
_vision_model: Optional[VisionEmotionModel] = None
_fusion_engine: Optional[MultiModalFusionEngine] = None

# Regex to detect and redact absolute Windows or Unix paths
_PATH_PATTERN = re.compile(r"[a-zA-Z]:\\[^\"\';\n\r\t<>]+|/(?:home|Users|tmp|var|etc)/[^\"\';\n\r\t<>]+")


def get_text_model() -> TextEmotionModel:
    """Return the cached TextEmotionModel singleton instance."""
    global _text_model
    if _text_model is None:
        logger.info("Initializing TextEmotionModel singleton...")
        device = os.getenv("DEVICE", "cpu")
        model_name = os.getenv("TEXT_MODEL_NAME", "j-hartmann/emotion-english-distilroberta-base")
        _text_model = TextEmotionModel(model_name=model_name, device=device)
    return _text_model


def get_speech_model() -> SpeechEmotionModel:
    """Return the cached SpeechEmotionModel singleton instance."""
    global _speech_model
    if _speech_model is None:
        logger.info("Initializing SpeechEmotionModel singleton...")
        device = os.getenv("DEVICE", "cpu")
        weights_path = os.getenv("SPEECH_MODEL_PATH", "models/speech/best_model_CNN.h5")
        _speech_model = SpeechEmotionModel(weights_path=weights_path, device=device)
    return _speech_model


def get_vision_model() -> VisionEmotionModel:
    """Return the cached VisionEmotionModel singleton instance."""
    global _vision_model
    if _vision_model is None:
        logger.info("Initializing VisionEmotionModel singleton...")
        device = os.getenv("DEVICE", "cpu")
        weights_path = os.getenv("VISION_MODEL_PATH", "models/vision/model_image_new50.pth")
        _vision_model = VisionEmotionModel(weights_path=weights_path, device=device)
    return _vision_model


def get_fusion_engine() -> MultiModalFusionEngine:
    """Return the cached MultiModalFusionEngine singleton instance."""
    global _fusion_engine
    if _fusion_engine is None:
        logger.info("Initializing MultiModalFusionEngine singleton...")
        _fusion_engine = MultiModalFusionEngine()
    return _fusion_engine


def sanitize_metadata(data: Any) -> Any:
    """
    Recursively sanitize dictionaries/lists to ensure no internal filesystem paths leak to clients.
    """
    if isinstance(data, dict):
        sanitized = {}
        for k, v in data.items():
            if k in ("weights_path", "temp_path", "file_path", "source_path"):
                sanitized[k] = os.path.basename(str(v)) if v else ""
            else:
                sanitized[k] = sanitize_metadata(v)
        return sanitized
    elif isinstance(data, list):
        return [sanitize_metadata(item) for item in data]
    elif isinstance(data, str):
        return _PATH_PATTERN.sub("[REDACTED_PATH]", data)
    return data


def sanitize_modality_result(res: ModalityResult) -> ModalityResult:
    """Sanitize metadata in ModalityResult to prevent path disclosure."""
    res_dict = res.model_dump()
    res_dict["metadata"] = sanitize_metadata(res_dict.get("metadata", {}))
    return ModalityResult(**res_dict)


def sanitize_fused_result(fused: FusedEmotionResult) -> FusedEmotionResult:
    """Sanitize metadata across all sub-modalities in FusedEmotionResult."""
    fused_dict = fused.model_dump()
    for mod_name, mod_res in fused_dict.get("modality_results", {}).items():
        if "metadata" in mod_res:
            mod_res["metadata"] = sanitize_metadata(mod_res["metadata"])
    if "explainability" in fused_dict and "key_features" in fused_dict["explainability"]:
        fused_dict["explainability"]["key_features"] = sanitize_metadata(
            fused_dict["explainability"]["key_features"]
        )
    return FusedEmotionResult(**fused_dict)
