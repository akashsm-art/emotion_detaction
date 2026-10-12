"""
Multi-Modal Emotion Recognition Fusion Engine.
Implements principled Confidence-Aware Weighted Late Fusion across Text, Speech/Audio,
and Vision modalities with mathematical concordance analysis and structured explainability.

ZERO FAKE AI POLICY:
- Only modalities with genuine inference (available == True and status == 'ok') are fused.
- Modalities without trained weights (e.g., Vision) are excluded and contribute 0 weight.
- Model output confidence is treated as a weighting prior, NEVER as clinical accuracy.
"""

from datetime import datetime, timezone
import logging
import math
from typing import Dict, Any, List, Optional, Tuple

from core.ontology import (
    CORE_EMOTIONS,
    EMOTION_VALENCE_AROUSAL,
    validate_probability_distribution,
    create_empty_distribution,
)
from core.schemas import (
    ModalityResult,
    FusedEmotionResult,
    ConflictAnalysis,
    ExplainabilityReport,
)

logger = logging.getLogger(__name__)

# Default base priors for modalities (configurable via constructor)
DEFAULT_CONFIGURED_WEIGHTS: Dict[str, float] = {
    "text": 1.0,
    "audio": 1.0,
    "speech": 1.0,
    "vision": 1.0,
}


class MultiModalFusionEngine:
    """
    Confidence-aware late decision fusion engine for multi-modal emotion recognition.
    Dynamically fuses available, valid modalities while cleanly handling missing or
    unavailable modalities (such as Vision when checkpoint weights are absent).
    """

    def __init__(
        self,
        modality_weights: Optional[Dict[str, float]] = None,
        conflict_threshold: float = 0.40,
    ):
        """
        Initialize the fusion engine.
        Args:
            modality_weights: Optional dictionary of base configured weights.
                              Defaults to W_text=1.0, W_audio=1.0, W_vision=1.0.
            conflict_threshold: Divergence threshold (0.0 to 1.0) above which
                                modalities are considered to have a prediction conflict.
        """
        self.configured_weights = dict(DEFAULT_CONFIGURED_WEIGHTS)
        if modality_weights:
            for k, v in modality_weights.items():
                self.configured_weights[k.lower()] = float(v)

        self.conflict_threshold = float(conflict_threshold)

    def fuse(
        self,
        text_result: Optional[ModalityResult] = None,
        audio_result: Optional[ModalityResult] = None,
        vision_result: Optional[ModalityResult] = None,
        speech_result: Optional[ModalityResult] = None,
        **kwargs: Any,
    ) -> FusedEmotionResult:
        """
        Execute multi-modal fusion over the provided modality prediction results.
        Accepts results from individual modality models and deterministically computes:
        1. Active vs unavailable modalities
        2. Confidence-adjusted effective weights
        3. Fused 7-class emotion probability distribution
        4. Cross-modal concordance and conflict metrics
        5. Structured explainability report
        """
        # Map inputs into normalized dictionary
        raw_modalities: Dict[str, ModalityResult] = {}
        if text_result is not None:
            raw_modalities["text"] = text_result
        # Support both 'audio_result' and 'speech_result'
        if audio_result is not None:
            raw_modalities["audio"] = audio_result
        elif speech_result is not None:
            raw_modalities["audio"] = speech_result
        if vision_result is not None:
            raw_modalities["vision"] = vision_result

        # Handle any additional kwargs passed
        for k, v in kwargs.items():
            if isinstance(v, ModalityResult):
                norm_key = "audio" if k.lower() in ("speech", "speech_result", "audio_result") else k.lower()
                raw_modalities[norm_key] = v

        return self.fuse_dict(raw_modalities)

    def fuse_dict(self, modality_results: Dict[str, ModalityResult]) -> FusedEmotionResult:
        """
        Fuse a dictionary of modality results mapping modality identifier -> ModalityResult.
        """
        iso_timestamp = datetime.now(timezone.utc).isoformat()

        active_modalities: List[str] = []
        unavailable_modalities: List[str] = []
        valid_results: Dict[str, ModalityResult] = {}

        # 1. Eligibility and Probability Validation
        for mod_name, res in modality_results.items():
            norm_name = "audio" if mod_name in ("speech", "speech_result") else mod_name

            if not res.available or res.status != "ok":
                unavailable_modalities.append(norm_name)
                logger.info(
                    "Modality '%s' excluded from fusion: available=%s, status='%s'",
                    norm_name,
                    res.available,
                    res.status,
                )
                continue

            # Validate 7-class probability vector structure
            is_valid, err_msg = validate_probability_distribution(
                res.probabilities, supported_emotions=res.supported_emotions or None
            )
            if not is_valid:
                unavailable_modalities.append(norm_name)
                logger.warning(
                    "Modality '%s' excluded from fusion due to invalid probabilities: %s",
                    norm_name,
                    err_msg,
                )
                continue

            # Modality is fully eligible for fusion
            active_modalities.append(norm_name)
            valid_results[norm_name] = res

        # 2. Case: Zero valid modalities available
        if not active_modalities:
            empty_probs = create_empty_distribution()
            conflict_info = ConflictAnalysis(
                has_conflict=False,
                conflict_score=0.0,
                divergent_pairs=[],
                summary="No valid modalities were available for conflict analysis.",
            )
            explain_info = ExplainabilityReport(
                primary_modality="none",
                concordance_score=0.0,
                modality_weights={},
                configured_weights=self.configured_weights,
                rationale="Inference blocked: No valid modality predictions were supplied to the fusion engine.",
                key_features={},
            )
            return FusedEmotionResult(
                available=False,
                status="NO_VALID_MODALITIES_AVAILABLE",
                fusion_mode="unavailable",
                emotion="none",
                confidence=0.0,
                probabilities=empty_probs,
                valence=0.0,
                arousal=0.0,
                active_modalities=[],
                unavailable_modalities=unavailable_modalities or list(modality_results.keys()),
                modalities_used=[],
                modality_results=modality_results,
                conflict=conflict_info,
                explainability=explain_info,
                timestamp=iso_timestamp,
            )

        # Extract features for explainability
        key_features: Dict[str, Any] = {}
        for mod_name, res in valid_results.items():
            if res.metadata:
                key_features[mod_name] = res.metadata

        # 3. Case: Single modality available
        if len(active_modalities) == 1:
            sole_mod = active_modalities[0]
            sole_res = valid_results[sole_mod]

            conflict_info = ConflictAnalysis(
                has_conflict=False,
                conflict_score=0.0,
                divergent_pairs=[],
                summary=f"Single modality ({sole_mod}); cross-modal conflict analysis not applicable.",
            )
            explain_info = ExplainabilityReport(
                primary_modality=sole_mod,
                concordance_score=1.0,
                modality_weights={sole_mod: 1.0},
                configured_weights={sole_mod: self.configured_weights.get(sole_mod, 1.0)},
                rationale=f"Single-modality prediction based exclusively on {sole_mod} with model confidence {sole_res.confidence:.4f}.",
                key_features=key_features,
            )
            return FusedEmotionResult(
                available=True,
                status="ok",
                fusion_mode="single_modality",
                emotion=sole_res.emotion,
                confidence=sole_res.confidence,
                probabilities=dict(sole_res.probabilities),
                valence=sole_res.valence,
                arousal=sole_res.arousal,
                active_modalities=[sole_mod],
                unavailable_modalities=unavailable_modalities,
                modalities_used=[sole_mod],
                modality_results=modality_results,
                conflict=conflict_info,
                explainability=explain_info,
                timestamp=iso_timestamp,
            )

        # 4. Case: Multi-modal fusion (>= 2 active modalities)
        effective_weights: Dict[str, float] = {}
        for mod_name, res in valid_results.items():
            base_w = self.configured_weights.get(mod_name, 1.0)
            # Confidence weighting factor: confidence in [0.0, 1.0]
            conf_factor = max(0.0001, res.confidence)
            effective_weights[mod_name] = base_w * conf_factor

        total_effective_weight = sum(effective_weights.values())
        if total_effective_weight > 0.0:
            normalized_weights = {
                mod: round(w / total_effective_weight, 4)
                for mod, w in effective_weights.items()
            }
        else:
            uniform_w = round(1.0 / len(active_modalities), 4)
            normalized_weights = {mod: uniform_w for mod in active_modalities}

        # Calculate fused 7-class probability distribution
        fused_probs = create_empty_distribution()
        for mod_name, norm_w in normalized_weights.items():
            res_probs = valid_results[mod_name].probabilities
            for emotion in CORE_EMOTIONS:
                fused_probs[emotion] += norm_w * res_probs.get(emotion, 0.0)

        # Re-normalize to guarantee exact sum = 1.0
        prob_sum = sum(fused_probs.values())
        if prob_sum > 0.0:
            for emotion in fused_probs:
                fused_probs[emotion] = round(fused_probs[emotion] / prob_sum, 4)

        # Select highest-probability emotion and confidence
        top_emotion = max(fused_probs, key=fused_probs.get)
        fused_confidence = float(fused_probs[top_emotion])

        # Compute aggregate circumplex coordinates (Valence / Arousal)
        valence = float(
            sum(fused_probs[e] * EMOTION_VALENCE_AROUSAL[e]["valence"] for e in CORE_EMOTIONS)
        )
        arousal = float(
            sum(fused_probs[e] * EMOTION_VALENCE_AROUSAL[e]["arousal"] for e in CORE_EMOTIONS)
        )

        # 5. Cross-Modal Conflict & Concordance Analysis
        concordance_score, conflict_score, divergent_pairs, conflict_summary = self._analyze_cross_modal_conflict(
            valid_results, top_emotion
        )

        # Identify primary driving modality
        primary_modality = max(normalized_weights, key=normalized_weights.get)

        weights_str = ", ".join(f"{m}: {w:.2f}" for m, w in normalized_weights.items())
        rationale = (
            f"Fused prediction '{top_emotion}' derived from {len(active_modalities)} active modalities ({weights_str}). "
            f"Primary driver: {primary_modality} (effective contribution: {normalized_weights[primary_modality]:.2f}, "
            f"model confidence: {valid_results[primary_modality].confidence:.4f}). "
            f"Cross-modal concordance score: {concordance_score:.4f}."
        )

        conflict_info = ConflictAnalysis(
            has_conflict=(conflict_score >= self.conflict_threshold) or (len(divergent_pairs) > 0),
            conflict_score=round(conflict_score, 4),
            divergent_pairs=divergent_pairs,
            summary=conflict_summary,
        )

        explain_info = ExplainabilityReport(
            primary_modality=primary_modality,
            concordance_score=round(concordance_score, 4),
            modality_weights=normalized_weights,
            configured_weights={m: self.configured_weights.get(m, 1.0) for m in active_modalities},
            rationale=rationale,
            key_features=key_features,
        )

        return FusedEmotionResult(
            available=True,
            status="ok",
            fusion_mode="multimodal",
            emotion=top_emotion,
            confidence=round(fused_confidence, 4),
            probabilities=fused_probs,
            valence=round(valence, 4),
            arousal=round(arousal, 4),
            active_modalities=active_modalities,
            unavailable_modalities=unavailable_modalities,
            modalities_used=active_modalities,
            modality_results=modality_results,
            conflict=conflict_info,
            explainability=explain_info,
            timestamp=iso_timestamp,
        )

    def _analyze_cross_modal_conflict(
        self, valid_results: Dict[str, ModalityResult], fused_top_emotion: str
    ) -> Tuple[float, float, List[Dict[str, Any]], str]:
        """
        Computes pairwise cosine similarities across active modality probability distributions.
        Returns:
            concordance_score: float in [0.0, 1.0] (1.0 = identical distributions)
            conflict_score: float in [0.0, 1.0] (1.0 - concordance_score)
            divergent_pairs: list of pairs where top predicted emotion differs
            summary: non-judgmental human-readable conflict summary
        """
        modalities = list(valid_results.keys())
        n = len(modalities)

        pairwise_similarities: List[float] = []
        divergent_pairs: List[Dict[str, Any]] = []

        for i in range(n):
            for j in range(i + 1, n):
                m1, m2 = modalities[i], modalities[j]
                r1, r2 = valid_results[m1], valid_results[m2]

                p1 = r1.probabilities
                p2 = r2.probabilities

                # Cosine similarity between probability distributions
                dot_product = sum(p1.get(e, 0.0) * p2.get(e, 0.0) for e in CORE_EMOTIONS)
                norm1 = math.sqrt(sum(p1.get(e, 0.0) ** 2 for e in CORE_EMOTIONS))
                norm2 = math.sqrt(sum(p2.get(e, 0.0) ** 2 for e in CORE_EMOTIONS))

                if norm1 > 0.0 and norm2 > 0.0:
                    sim = dot_product / (norm1 * norm2)
                else:
                    sim = 0.0

                sim = max(0.0, min(1.0, sim))
                pairwise_similarities.append(sim)

                divergence = round(1.0 - sim, 4)

                if r1.emotion != r2.emotion:
                    divergent_pairs.append({
                        "modality_1": m1,
                        "emotion_1": r1.emotion,
                        "confidence_1": r1.confidence,
                        "modality_2": m2,
                        "emotion_2": r2.emotion,
                        "confidence_2": r2.confidence,
                        "pairwise_divergence": divergence,
                    })

        if pairwise_similarities:
            concordance = sum(pairwise_similarities) / len(pairwise_similarities)
        else:
            concordance = 1.0

        conflict = 1.0 - concordance

        if divergent_pairs:
            pair_desc = ", ".join(
                f"{p['modality_1']} ({p['emotion_1']}) vs {p['modality_2']} ({p['emotion_2']})"
                for p in divergent_pairs
            )
            summary = (
                f"Prediction conflict detected between active modalities: {pair_desc}. "
                f"Overall cross-modal divergence: {conflict:.4f}."
            )
        else:
            summary = (
                f"Active modalities exhibit high prediction concordance on emotion '{fused_top_emotion}' "
                f"(concordance score: {concordance:.4f})."
            )

        return concordance, conflict, divergent_pairs, summary
