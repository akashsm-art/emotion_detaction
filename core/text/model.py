"""
Text Emotion Recognition Model.
Executes genuine deep Transformer inference using RoBERTa sequence classification.
Standardizes predictions to the Ekman 7 Core Emotions schema.
ZERO FAKE AI: Exposes model loading errors transparently; no heuristic masquerading as ML.
"""

import logging
from typing import Dict, Any, Optional
import numpy as np

from core.ontology import (
    CORE_EMOTIONS,
    EMOTION_VALENCE_AROUSAL,
    normalize_to_core_ontology,
    create_empty_distribution,
)
from core.schemas import ModalityResult
from core.text.preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)

DEFAULT_TEXT_MODEL = "j-hartmann/emotion-english-distilroberta-base"


class TextEmotionModel:
    """Classifies emotions from natural language using deep RoBERTa transformers."""

    def __init__(
        self,
        model_name: str = DEFAULT_TEXT_MODEL,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.device = device
        self.preprocessor = TextPreprocessor()
        self.pipeline = None
        self._is_loaded = False
        self.load_error: Optional[str] = None

        self._load_transformer()

    def _load_transformer(self) -> None:
        """Attempt to load HuggingFace transformer sequence classification pipeline."""
        try:
            from transformers import pipeline

            device_idx = 0 if self.device == "cuda" else -1
            self.pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                top_k=None,
                device=device_idx,
            )
            self._is_loaded = True
            self.load_error = None
            logger.info("Successfully loaded Transformer text model: %s", self.model_name)
        except Exception as e:
            self.load_error = str(e)
            self.pipeline = None
            self._is_loaded = False
            logger.error("Failed to load Transformer model (%s): %s", self.model_name, e)

    def predict(self, raw_text: str) -> ModalityResult:
        """Classify emotion of input text string using deep transformer pipeline."""
        cleaned_text = self.preprocessor.clean(raw_text)
        salient_keywords = self.preprocessor.extract_salient_keywords(cleaned_text)

        if not self._is_loaded or self.pipeline is None:
            empty_probs = create_empty_distribution()
            return ModalityResult(
                modality="text",
                available=False,
                status=f"TEXT_INFERENCE_BLOCKED — MODEL_NOT_LOADED: {self.load_error}",
                emotion="none",
                confidence=0.0,
                probabilities=empty_probs,
                supported_emotions=CORE_EMOTIONS,
                valence=0.0,
                arousal=0.0,
                metadata={"error": self.load_error},
            )

        if not cleaned_text.strip():
            empty_probs = create_empty_distribution()
            return ModalityResult(
                modality="text",
                available=False,
                status="EMPTY_TEXT_INPUT",
                emotion="none",
                confidence=0.0,
                probabilities=empty_probs,
                supported_emotions=CORE_EMOTIONS,
                valence=0.0,
                arousal=0.0,
                metadata={"reason": "Input text was empty or whitespace only"},
            )

        try:
            # Truncate text to max 512 characters for sequence classification
            truncated = cleaned_text[:512]
            raw_outputs = self.pipeline(truncated)[0]

            raw_probs: Dict[str, float] = {}
            for item in raw_outputs:
                label = item["label"].lower()
                score = float(item["score"])
                raw_probs[label] = score

            # Normalize to 7 core emotions
            normalized_probs = normalize_to_core_ontology(
                raw_probs, supported_emotions=CORE_EMOTIONS
            )
            top_emotion = max(normalized_probs, key=normalized_probs.get)
            confidence = float(normalized_probs[top_emotion])

            # Compute circumplex coordinates (Valence / Arousal)
            valence = float(
                sum(normalized_probs[e] * EMOTION_VALENCE_AROUSAL[e]["valence"] for e in CORE_EMOTIONS)
            )
            arousal = float(
                sum(normalized_probs[e] * EMOTION_VALENCE_AROUSAL[e]["arousal"] for e in CORE_EMOTIONS)
            )

            return ModalityResult(
                modality="text",
                available=True,
                status="ok",
                emotion=top_emotion,
                confidence=round(confidence, 4),
                probabilities=normalized_probs,
                supported_emotions=CORE_EMOTIONS,
                valence=round(valence, 4),
                arousal=round(arousal, 4),
                metadata={
                    "char_length": len(raw_text),
                    "word_count": len(cleaned_text.split()),
                    "salient_keywords": salient_keywords,
                    "model_name": self.model_name,
                    "engine": "transformer_deep_inference",
                },
            )

        except Exception as e:
            logger.error("Inference execution error: %s", e)
            empty_probs = create_empty_distribution()
            return ModalityResult(
                modality="text",
                available=False,
                status=f"INFERENCE_EXECUTION_ERROR: {e}",
                emotion="none",
                confidence=0.0,
                probabilities=empty_probs,
                supported_emotions=CORE_EMOTIONS,
                valence=0.0,
                arousal=0.0,
                metadata={"error": str(e)},
            )
