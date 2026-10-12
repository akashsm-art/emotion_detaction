"""
Centralized Emotion Ontology and Label Mapping System.
Standardizes emotion representations across Text, Speech, and Vision modalities
to the universal Ekman 7-emotion schema, while rigorously tracking mapping fidelity:
- DIRECT: Exact semantic equivalence
- APPROXIMATE: Close emotional alignment
- LOSSY: Significant nuance reduced to broader category
- UNSUPPORTED: Modality does not support or predict this emotion
"""

from enum import Enum
from typing import Dict, Any, List, Optional, Tuple


class MappingType(str, Enum):
    DIRECT = "direct"
    APPROXIMATE = "approximate"
    LOSSY = "lossy"
    UNSUPPORTED = "unsupported"


# Universal 7 Core Emotions (Ekman 6 + Neutral)
CORE_EMOTIONS: List[str] = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

# Russell's Circumplex Model (Valence: -1.0 to +1.0, Arousal: -1.0 to +1.0)
EMOTION_VALENCE_AROUSAL: Dict[str, Dict[str, float]] = {
    "angry": {"valence": -0.65, "arousal": 0.80},
    "disgust": {"valence": -0.60, "arousal": 0.35},
    "fear": {"valence": -0.70, "arousal": 0.75},
    "happy": {"valence": 0.85, "arousal": 0.60},
    "neutral": {"valence": 0.00, "arousal": 0.00},
    "sad": {"valence": -0.80, "arousal": -0.45},
    "surprise": {"valence": 0.25, "arousal": 0.85},
}

# Modality & Dataset Specific Mappings with Mapping Type Fidelity
# Format: raw_label_lowercase -> (core_emotion, MappingType)
LABEL_MAPPINGS: Dict[str, Tuple[str, MappingType]] = {
    # Direct mappings (100% semantic identity)
    "angry": ("angry", MappingType.DIRECT),
    "anger": ("angry", MappingType.DIRECT),
    "disgust": ("disgust", MappingType.DIRECT),
    "disgusted": ("disgust", MappingType.DIRECT),
    "fear": ("fear", MappingType.DIRECT),
    "fearful": ("fear", MappingType.DIRECT),
    "happy": ("happy", MappingType.DIRECT),
    "happiness": ("happy", MappingType.DIRECT),
    "joy": ("happy", MappingType.DIRECT),
    "neutral": ("neutral", MappingType.DIRECT),
    "sad": ("sad", MappingType.DIRECT),
    "sadness": ("sad", MappingType.DIRECT),
    "surprise": ("surprise", MappingType.DIRECT),
    "surprised": ("surprise", MappingType.DIRECT),

    # German Emo-DB labels (speech-emotion-recognition-master)
    "wut": ("angry", MappingType.DIRECT),
    "ekel": ("disgust", MappingType.DIRECT),
    "angst": ("fear", MappingType.DIRECT),
    "freude": ("happy", MappingType.DIRECT),
    "trauer": ("sad", MappingType.DIRECT),
    "langeweile": ("neutral", MappingType.APPROXIMATE),

    # Polish dataset abbreviations
    "an": ("angry", MappingType.DIRECT),
    "di": ("disgust", MappingType.DIRECT),
    "fe": ("fear", MappingType.DIRECT),
    "ha": ("happy", MappingType.DIRECT),
    "ne": ("neutral", MappingType.DIRECT),
    "sa": ("sad", MappingType.DIRECT),
    "su": ("surprise", MappingType.DIRECT),

    # RAVDESS numeric codes
    "01": ("neutral", MappingType.DIRECT),       # neutral
    "02": ("neutral", MappingType.APPROXIMATE),  # calm -> neutral
    "03": ("happy", MappingType.DIRECT),         # happy
    "04": ("sad", MappingType.DIRECT),           # sad
    "05": ("angry", MappingType.DIRECT),         # angry
    "06": ("fear", MappingType.DIRECT),          # fearful
    "07": ("disgust", MappingType.DIRECT),       # disgust
    "08": ("surprise", MappingType.DIRECT),      # surprised

    # Twitter 6-class / BERT 12-class extensions
    "love": ("happy", MappingType.APPROXIMATE),
    "optimism": ("happy", MappingType.APPROXIMATE),
    "anticipation": ("surprise", MappingType.APPROXIMATE),
    "pessimism": ("sad", MappingType.APPROXIMATE),
    "trust": ("neutral", MappingType.LOSSY),
    "calm": ("neutral", MappingType.APPROXIMATE),
}


def create_empty_distribution() -> Dict[str, float]:
    """Return an empty probability distribution initialized to 0.0 for all 7 emotions."""
    return {emotion: 0.0 for emotion in CORE_EMOTIONS}


def map_label(raw_label: str) -> Tuple[str, MappingType]:
    """
    Map an arbitrary raw emotion label to the standard 7 core emotions,
    returning the mapped emotion and its MappingType fidelity.
    """
    if not raw_label:
        return "neutral", MappingType.LOSSY

    cleaned = str(raw_label).strip().lower()
    if cleaned in LABEL_MAPPINGS:
        return LABEL_MAPPINGS[cleaned]

    # Substring search fallback
    for key, (mapped, mtype) in LABEL_MAPPINGS.items():
        if key in cleaned:
            return mapped, MappingType.APPROXIMATE

    return "neutral", MappingType.LOSSY


def normalize_to_core_ontology(
    raw_probs: Dict[str, float],
    supported_emotions: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Maps raw probability distribution to the standard 7 core emotions.
    CRITICAL RULE: Unsupported emotions are strictly set to 0.0 (NO artificial mass).
    The mapped probabilities for supported emotions are normalized to sum to 1.0.
    """
    distribution = create_empty_distribution()

    for raw_label, prob in raw_probs.items():
        core_emotion, _ = map_label(raw_label)
        distribution[core_emotion] += float(max(0.0, prob))

    if supported_emotions is not None:
        # Zero out any emotion that this specific model does NOT support
        supported_set = set(supported_emotions)
        for emotion in CORE_EMOTIONS:
            if emotion not in supported_set:
                distribution[emotion] = 0.0

    total = sum(distribution.values())
    if total > 0.0:
        for emotion in distribution:
            distribution[emotion] = round(distribution[emotion] / total, 4)

    return distribution


def validate_probability_distribution(
    probs: Any,
    supported_emotions: Optional[List[str]] = None,
    tolerance: float = 0.05,
) -> Tuple[bool, Optional[str]]:
    """
    Validates that a probability distribution satisfies:
    1. It is a dictionary containing all 7 core emotions.
    2. All probability values are numeric, finite (no NaN or Inf), and >= 0.0.
    3. The sum of probabilities is approximately 1.0 (within tolerance).
    4. If supported_emotions is given, unsupported emotions must have exactly 0.0 probability.
    Returns (is_valid, error_message).
    """
    import math

    if not isinstance(probs, dict):
        return False, f"Expected dictionary of probabilities, got {type(probs).__name__}"

    missing_keys = [e for e in CORE_EMOTIONS if e not in probs]
    if missing_keys:
        return False, f"Missing core emotion keys: {missing_keys}"

    for e in CORE_EMOTIONS:
        val = probs[e]
        if not isinstance(val, (int, float)):
            return False, f"Non-numeric probability for '{e}': {val}"
        if math.isnan(val) or math.isinf(val):
            return False, f"Non-finite probability (NaN/Inf) for '{e}': {val}"
        if val < 0.0:
            return False, f"Negative probability for '{e}': {val}"

    # Verify unsupported emotions are strictly 0.0
    if supported_emotions is not None:
        supported_set = set(supported_emotions)
        for e in CORE_EMOTIONS:
            if e not in supported_set and probs[e] != 0.0:
                return False, f"Unsupported emotion '{e}' must have 0.0 probability, got {probs[e]}"

    total = sum(probs[e] for e in CORE_EMOTIONS)
    if abs(total - 1.0) > tolerance:
        return False, f"Probability distribution does not sum to 1.0 (sum={total:.4f}, tolerance={tolerance})"

    return True, None

