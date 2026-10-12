"""
Facial Vision Emotion Recognition Model.
Processes images and video frames, extracts facial bounding boxes, and prepares normalized 224x224 crops.
ZERO FAKE AI POLICY:
Because 'model_visual_alexnet50.pth' was not provided in the Multimodal RAVDESS repository,
this module STRICTLY REFUSES to fabricate random outputs or heuristic guesses.
When trained checkpoint weights are missing, it returns:
    modality: "vision"
    available: False
    status: "VISION INFERENCE BLOCKED — TRAINED CHECKPOINT MISSING"
"""

import os
import logging
from typing import Dict, Any, List, Optional
import numpy as np

from core.ontology import (
    CORE_EMOTIONS,
    EMOTION_VALENCE_AROUSAL,
    normalize_to_core_ontology,
    create_empty_distribution,
)
from core.schemas import ModalityResult
from core.vision.face_detector import FaceDetector

logger = logging.getLogger(__name__)

DEFAULT_VISION_WEIGHTS = "models/vision/model_image_new50.pth"
LEGACY_VISION_WEIGHTS = "models/vision/model_visual_alexnet50.pth"


class VisionEmotionModel:
    """Prepares facial crops and checks for trained visual network weights."""

    def __init__(
        self,
        weights_path: str = DEFAULT_VISION_WEIGHTS,
        device: str = "cpu",
    ):
        self.weights_path = weights_path
        self.device = device
        self.detector = FaceDetector()
        self.model = None
        self._is_loaded = False
        self.load_error: Optional[str] = None

        self._check_and_load_weights()

    def _check_and_load_weights(self) -> None:
        """
        Inspect whether trained weights exist.
        Checks models/vision/model_image_new50.pth (or fallback model_visual_alexnet50.pth).
        If missing, logs explicit blocked warning without faking weights.
        """
        resolved_path = self.weights_path
        if not os.path.isabs(resolved_path):
            if not os.path.exists(resolved_path):
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
                cand = os.path.join(project_root, resolved_path)
                if os.path.exists(cand):
                    resolved_path = cand
                else:
                    cand_legacy = os.path.join(project_root, LEGACY_VISION_WEIGHTS)
                    if os.path.exists(cand_legacy):
                        resolved_path = cand_legacy

        if not os.path.exists(resolved_path):
            self.load_error = (
                f"Trained weights checkpoint missing at: {resolved_path}. "
                "Multimodal-Emotion-Recognition-master contains video frames and architecture code, "
                "but no trained .pth weights file was included in the repository."
            )
            self._is_loaded = False
            logger.warning(
                "VISION INFERENCE BLOCKED — TRAINED CHECKPOINT MISSING: %s",
                resolved_path,
            )
            return

        try:
            import torch
            import torch.nn as nn
            import torchvision.models as models

            # AlexNet backbone matching Multimodal-Emotion-Recognition-master
            self.model = models.alexnet(weights=None)
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Sequential(
                nn.Linear(num_ftrs, len(CORE_EMOTIONS)),
                nn.Softmax(dim=1),
            )

            state_dict = torch.load(resolved_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            self._is_loaded = True
            self.weights_path = resolved_path
            self.load_error = None
            logger.info("Successfully loaded custom visual weights from %s", resolved_path)

        except Exception as e:
            self.load_error = f"Failed to load visual weights: {e}"
            self._is_loaded = False
            self.model = None
            logger.error(self.load_error)

    def predict(self, image_source: Any) -> ModalityResult:
        """
        Prepares face crops and evaluates emotion.
        If weights are missing, returns honest UNAVAILABLE status with zero fabrication.
        """
        try:
            image = self.detector.load_image(image_source)
        except Exception as e:
            empty_probs = create_empty_distribution()
            return ModalityResult(
                modality="vision",
                available=False,
                status=f"IMAGE_LOAD_ERROR: {e}",
                emotion="none",
                confidence=0.0,
                probabilities=empty_probs,
                supported_emotions=CORE_EMOTIONS,
                valence=0.0,
                arousal=0.0,
                metadata={"error": str(e)},
            )

        crops, boxes = self.detector.detect_and_crop_faces(image)
        face_count = sum(1 for b in boxes if b.get("is_face", False))

        metadata = {
            "image_resolution": [int(image.shape[1]), int(image.shape[0])],
            "faces_detected": face_count,
            "bounding_boxes": boxes,
            "crop_dimensions": [list(c.shape) for c in crops],
            "weights_path": self.weights_path,
            "weights_found": self._is_loaded,
        }

        # ZERO FAKE AI ENFORCEMENT: If no valid weights, return BLOCKED status
        if not self._is_loaded or self.model is None:
            empty_probs = create_empty_distribution()
            return ModalityResult(
                modality="vision",
                available=False,
                status="VISION INFERENCE BLOCKED — TRAINED CHECKPOINT MISSING",
                emotion="none",
                confidence=0.0,
                probabilities=empty_probs,
                supported_emotions=CORE_EMOTIONS,
                valence=0.0,
                arousal=0.0,
                metadata={
                    **metadata,
                    "reason": self.load_error or "Trained checkpoint missing",
                    "policy": "Zero Fake AI: No random or heuristic prediction allowed when model weights are missing.",
                },
            )

        # If a real checkpoint is provided in the future:
        try:
            import torch

            primary_crop = crops[0]
            tensor = torch.from_numpy(primary_crop).float().permute(2, 0, 1) / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            tensor = (tensor - mean) / std
            tensor = tensor.unsqueeze(0).to(self.device)

            with torch.no_grad():
                out = self.model(tensor)
                probs = out.squeeze().cpu().numpy()

            raw_probs = {CORE_EMOTIONS[i]: float(probs[i]) for i in range(len(CORE_EMOTIONS))}
            normalized_probs = normalize_to_core_ontology(raw_probs, supported_emotions=CORE_EMOTIONS)
            top_emotion = max(normalized_probs, key=normalized_probs.get)
            confidence = float(normalized_probs[top_emotion])

            valence = float(
                sum(normalized_probs[e] * EMOTION_VALENCE_AROUSAL[e]["valence"] for e in CORE_EMOTIONS)
            )
            arousal = float(
                sum(normalized_probs[e] * EMOTION_VALENCE_AROUSAL[e]["arousal"] for e in CORE_EMOTIONS)
            )

            return ModalityResult(
                modality="vision",
                available=True,
                status="ok",
                emotion=top_emotion,
                confidence=round(confidence, 4),
                probabilities=normalized_probs,
                supported_emotions=CORE_EMOTIONS,
                valence=round(valence, 4),
                arousal=round(arousal, 4),
                metadata={
                    **metadata,
                    "engine": "alexnet_visual_cnn",
                },
            )

        except Exception as e:
            empty_probs = create_empty_distribution()
            return ModalityResult(
                modality="vision",
                available=False,
                status=f"INFERENCE_EXECUTION_ERROR: {e}",
                emotion="none",
                confidence=0.0,
                probabilities=empty_probs,
                supported_emotions=CORE_EMOTIONS,
                valence=0.0,
                arousal=0.0,
                metadata={**metadata, "error": str(e)},
            )
