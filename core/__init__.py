"""
Multi-Modal Emotion Recognition System Core Package.
Integrates Text, Speech, and Vision modality pipelines with unified ontology and late fusion.
"""

from core.ontology import CORE_EMOTIONS, EMOTION_VALENCE_AROUSAL
from core.schemas import ModalityResult, FusedEmotionResult, ConflictAnalysis, ExplainabilityReport

__all__ = [
    "CORE_EMOTIONS",
    "EMOTION_VALENCE_AROUSAL",
    "ModalityResult",
    "FusedEmotionResult",
    "ConflictAnalysis",
    "ExplainabilityReport",
]
