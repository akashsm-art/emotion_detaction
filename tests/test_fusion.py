"""
Unit and Integration Test Suite for MultiModalFusionEngine.
Implements the 12 mandatory validation tests specified in Phase F.
"""

import pytest
import math
from typing import Dict

from core.ontology import CORE_EMOTIONS, create_empty_distribution, validate_probability_distribution
from core.schemas import ModalityResult, FusedEmotionResult
from core.fusion.engine import MultiModalFusionEngine


def _make_mock_modality_result(
    modality: str,
    available: bool = True,
    status: str = "ok",
    emotion: str = "neutral",
    confidence: float = 0.80,
    probabilities: Dict[str, float] = None,
    supported_emotions: list = None,
) -> ModalityResult:
    """Helper to construct controlled ModalityResult objects for deterministic testing."""
    if probabilities is None:
        probs = create_empty_distribution()
        probs[emotion] = round(confidence, 4)
        remainder = round((1.0 - confidence) / max(1, len(CORE_EMOTIONS) - 1), 4)
        for e in CORE_EMOTIONS:
            if e != emotion:
                probs[e] = remainder
        # Adjust sum to exactly 1.0
        diff = round(1.0 - sum(probs.values()), 4)
        probs[emotion] = round(probs[emotion] + diff, 4)
    else:
        probs = dict(probabilities)

    return ModalityResult(
        modality=modality,
        available=available,
        status=status,
        emotion=emotion if available else "none",
        confidence=confidence if available else 0.0,
        probabilities=probs,
        supported_emotions=supported_emotions or CORE_EMOTIONS,
        valence=0.5 if emotion == "happy" else (-0.5 if emotion == "sad" else 0.0),
        arousal=0.5 if emotion in ("happy", "angry") else 0.0,
        metadata={"test": True},
    )


# ---------------------------------------------------------------------------
# Test 1: Text Only
# ---------------------------------------------------------------------------
def test_1_text_only():
    engine = MultiModalFusionEngine()
    text_res = _make_mock_modality_result("text", emotion="happy", confidence=0.95)

    fused = engine.fuse(text_result=text_res)

    assert fused.available is True
    assert fused.status == "ok"
    assert fused.fusion_mode == "single_modality"
    assert fused.active_modalities == ["text"]
    assert fused.emotion == "happy"
    assert fused.confidence == 0.95
    assert fused.probabilities == text_res.probabilities
    assert fused.explainability.primary_modality == "text"
    assert fused.explainability.modality_weights == {"text": 1.0}


# ---------------------------------------------------------------------------
# Test 2: Speech Only
# ---------------------------------------------------------------------------
def test_2_speech_only():
    engine = MultiModalFusionEngine()
    speech_probs = create_empty_distribution()
    speech_probs["sad"] = 0.92
    speech_probs["angry"] = 0.05
    speech_probs["neutral"] = 0.02
    speech_probs["happy"] = 0.01
    speech_res = _make_mock_modality_result(
        "audio",
        emotion="sad",
        confidence=0.92,
        probabilities=speech_probs,
        supported_emotions=["neutral", "angry", "happy", "sad"],
    )

    fused = engine.fuse(audio_result=speech_res)

    assert fused.available is True
    assert fused.status == "ok"
    assert fused.fusion_mode == "single_modality"
    assert fused.active_modalities == ["audio"]
    assert fused.emotion == "sad"
    assert fused.confidence == 0.92
    assert fused.probabilities["sad"] == 0.92
    assert fused.probabilities["disgust"] == 0.0
    assert fused.probabilities["fear"] == 0.0
    assert fused.probabilities["surprise"] == 0.0


# ---------------------------------------------------------------------------
# Test 3: Text + Speech Controlled Mathematical Fusion
# ---------------------------------------------------------------------------
def test_3_text_plus_speech_math():
    engine = MultiModalFusionEngine(modality_weights={"text": 1.0, "audio": 1.0})

    # Text: confidence 0.80 -> weight = 1.0 * 0.80 = 0.80
    text_probs = create_empty_distribution()
    text_probs["happy"] = 0.70
    text_probs["neutral"] = 0.30
    text_res = _make_mock_modality_result("text", emotion="happy", confidence=0.80, probabilities=text_probs)

    # Audio: confidence 0.40 -> weight = 1.0 * 0.40 = 0.40
    audio_probs = create_empty_distribution()
    audio_probs["happy"] = 0.40
    audio_probs["neutral"] = 0.60
    audio_res = _make_mock_modality_result("audio", emotion="neutral", confidence=0.40, probabilities=audio_probs)

    # Total weight = 0.80 + 0.40 = 1.20
    # Normalized: alpha_text = 0.80 / 1.20 = 2/3 (~0.6667), alpha_audio = 0.40 / 1.20 = 1/3 (~0.3333)
    # Expected fused happy = (2/3 * 0.70) + (1/3 * 0.40) = 0.4667 + 0.1333 = 0.6000
    # Expected fused neutral = (2/3 * 0.30) + (1/3 * 0.60) = 0.2000 + 0.2000 = 0.4000
    fused = engine.fuse(text_result=text_res, audio_result=audio_res)

    assert fused.available is True
    assert fused.fusion_mode == "multimodal"
    assert set(fused.active_modalities) == {"text", "audio"}
    assert fused.emotion == "happy"
    assert math.isclose(fused.probabilities["happy"], 0.60, abs_tol=0.01)
    assert math.isclose(fused.probabilities["neutral"], 0.40, abs_tol=0.01)


# ---------------------------------------------------------------------------
# Test 4: Vision Unavailable Contributes Zero
# ---------------------------------------------------------------------------
def test_4_vision_unavailable_ignored():
    engine = MultiModalFusionEngine()
    empty_probs = create_empty_distribution()
    vision_res = _make_mock_modality_result(
        "vision",
        available=False,
        status="VISION INFERENCE BLOCKED — TRAINED CHECKPOINT MISSING",
        emotion="none",
        confidence=0.0,
        probabilities=empty_probs,
    )

    fused = engine.fuse(vision_result=vision_res)

    assert fused.available is False
    assert fused.status == "NO_VALID_MODALITIES_AVAILABLE"
    assert "vision" in fused.unavailable_modalities
    assert fused.active_modalities == []
    assert fused.confidence == 0.0


# ---------------------------------------------------------------------------
# Test 5: Text + Speech + Unavailable Vision
# ---------------------------------------------------------------------------
def test_5_text_speech_plus_unavailable_vision():
    engine = MultiModalFusionEngine()
    text_res = _make_mock_modality_result("text", emotion="happy", confidence=0.85)
    speech_res = _make_mock_modality_result("audio", emotion="happy", confidence=0.90)
    vision_res = _make_mock_modality_result(
        "vision",
        available=False,
        status="VISION INFERENCE BLOCKED — TRAINED CHECKPOINT MISSING",
        probabilities=create_empty_distribution(),
    )

    fused = engine.fuse(text_result=text_res, audio_result=speech_res, vision_result=vision_res)

    assert fused.available is True
    assert set(fused.active_modalities) == {"text", "audio"}
    assert "vision" in fused.unavailable_modalities
    assert "vision" not in fused.explainability.modality_weights
    assert fused.emotion == "happy"


# ---------------------------------------------------------------------------
# Test 6: Conflicting Modalities
# ---------------------------------------------------------------------------
def test_6_conflicting_modalities():
    engine = MultiModalFusionEngine(conflict_threshold=0.30)
    text_res = _make_mock_modality_result("text", emotion="happy", confidence=0.95)
    speech_res = _make_mock_modality_result("audio", emotion="sad", confidence=0.95)

    fused = engine.fuse(text_result=text_res, audio_result=speech_res)

    assert fused.available is True
    assert fused.conflict.has_conflict is True
    assert fused.conflict.conflict_score > 0.30
    assert len(fused.conflict.divergent_pairs) == 1
    assert "Prediction conflict" in fused.conflict.summary


# ---------------------------------------------------------------------------
# Test 7: Modality Agreement (High Concordance)
# ---------------------------------------------------------------------------
def test_7_modality_agreement():
    engine = MultiModalFusionEngine()
    text_res = _make_mock_modality_result("text", emotion="happy", confidence=0.95)
    speech_res = _make_mock_modality_result("audio", emotion="happy", confidence=0.90)

    fused = engine.fuse(text_result=text_res, audio_result=speech_res)

    assert fused.available is True
    assert fused.conflict.has_conflict is False
    assert fused.explainability.concordance_score >= 0.80
    assert len(fused.conflict.divergent_pairs) == 0
    assert "concordance" in fused.conflict.summary.lower()


# ---------------------------------------------------------------------------
# Test 8: Invalid Probability Vector Handled Gracefully
# ---------------------------------------------------------------------------
def test_8_invalid_probability_vector():
    engine = MultiModalFusionEngine()
    # Invalid: missing 5 core emotion keys and sums to 0.25 (fails probability distribution validation)
    bad_probs = {"happy": 0.15, "sad": 0.10}
    bad_res = _make_mock_modality_result("text", probabilities=bad_probs)

    fused = engine.fuse(text_result=bad_res)

    # Should be gracefully rejected, resulting in no valid modalities
    assert fused.available is False
    assert fused.status == "NO_VALID_MODALITIES_AVAILABLE"
    assert "text" in fused.unavailable_modalities


# ---------------------------------------------------------------------------
# Test 9: Missing Modality (Continues with Remaining Valid Modalities)
# ---------------------------------------------------------------------------
def test_9_missing_modality_continuation():
    engine = MultiModalFusionEngine()
    text_res = _make_mock_modality_result("text", emotion="angry", confidence=0.88)

    # Audio and Vision are omitted (None)
    fused = engine.fuse(text_result=text_res, audio_result=None, vision_result=None)

    assert fused.available is True
    assert fused.fusion_mode == "single_modality"
    assert fused.active_modalities == ["text"]
    assert fused.emotion == "angry"


# ---------------------------------------------------------------------------
# Test 10: No Modalities Supplied
# ---------------------------------------------------------------------------
def test_10_no_modalities_supplied():
    engine = MultiModalFusionEngine()

    fused = engine.fuse(text_result=None, audio_result=None, vision_result=None)

    assert fused.available is False
    assert fused.status == "NO_VALID_MODALITIES_AVAILABLE"
    assert fused.fusion_mode == "unavailable"
    assert fused.emotion == "none"
    assert fused.confidence == 0.0
    assert all(v == 0.0 for v in fused.probabilities.values())
    assert fused.active_modalities == []


# ---------------------------------------------------------------------------
# Test 11: Probability Normalization
# ---------------------------------------------------------------------------
def test_11_probability_normalization():
    engine = MultiModalFusionEngine()
    text_res = _make_mock_modality_result("text", emotion="fear", confidence=0.80)
    speech_res = _make_mock_modality_result("audio", emotion="surprise", confidence=0.60)

    fused = engine.fuse(text_result=text_res, audio_result=speech_res)

    prob_sum = sum(fused.probabilities.values())
    assert math.isclose(prob_sum, 1.0, abs_tol=0.01)
    for e in CORE_EMOTIONS:
        assert 0.0 <= fused.probabilities[e] <= 1.0


# ---------------------------------------------------------------------------
# Test 12: Unsupported Speech Classes Remain Exactly 0.0
# ---------------------------------------------------------------------------
def test_12_unsupported_speech_classes():
    engine = MultiModalFusionEngine()
    speech_probs = create_empty_distribution()
    speech_probs["angry"] = 0.85
    speech_probs["neutral"] = 0.15
    # disgust, fear, surprise are strictly 0.0
    speech_res = _make_mock_modality_result(
        "audio",
        emotion="angry",
        confidence=0.85,
        probabilities=speech_probs,
        supported_emotions=["neutral", "angry", "happy", "sad"],
    )

    fused = engine.fuse(audio_result=speech_res)

    # In single modality speech, unsupported classes must be strictly 0.0
    assert fused.probabilities["disgust"] == 0.0
    assert fused.probabilities["fear"] == 0.0
    assert fused.probabilities["surprise"] == 0.0
