"""
Pydantic Schemas for Multi-Modal Emotion Recognition.
Provides strictly typed data models for individual modalities, fusion results,
explainability reports, and API responses.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class ModalityResult(BaseModel):
    """Prediction result from a single modality (Text, Audio, or Vision)."""
    modality: str = Field(..., description="Modality identifier: 'text', 'audio', or 'vision'")
    available: bool = Field(True, description="Whether this modality was present and successfully inferred")
    status: str = Field("ok", description="Status string: 'ok', 'VISION INFERENCE BLOCKED — TRAINED CHECKPOINT MISSING', etc.")
    emotion: str = Field(..., description="Top predicted core emotion (or 'none' if unavailable)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    probabilities: Dict[str, float] = Field(..., description="Probabilities across all 7 core emotions")
    supported_emotions: List[str] = Field(default_factory=list, description="Emotions supported by this modality's model")
    valence: float = Field(0.0, description="Estimated emotional valence (-1.0 negative to +1.0 positive)")
    arousal: float = Field(0.0, description="Estimated emotional arousal (-1.0 calm/passive to +1.0 intense)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Modality-specific details (e.g. word count, audio length, face count, layer info)")


class ConflictAnalysis(BaseModel):
    """Analysis of cross-modal emotional alignment or discordance."""
    has_conflict: bool = Field(False, description="True if significant disagreement exists between modalities")
    conflict_score: float = Field(0.0, ge=0.0, le=1.0, description="Degree of cross-modal divergence")
    divergent_pairs: List[Dict[str, Any]] = Field(default_factory=list, description="Pairs of modalities with opposing emotions")
    summary: str = Field("Modalities are concordant.", description="Human-readable conflict summary")


class ExplainabilityReport(BaseModel):
    """Explainability and attribution insights into the fusion decision."""
    primary_modality: str = Field(..., description="Modality that had the highest influence on the decision (or 'none')")
    concordance_score: float = Field(..., ge=0.0, le=1.0, description="Inter-modality agreement score (1.0 = perfect agreement, 0.0 = maximal divergence)")
    modality_weights: Dict[str, float] = Field(..., description="Normalized contribution weight of each modality")
    configured_weights: Dict[str, float] = Field(default_factory=dict, description="Configured prior weights before confidence adjustment")
    rationale: str = Field(..., description="Detailed textual rationale explaining the fused classification")
    key_features: Dict[str, Any] = Field(default_factory=dict, description="Salient features from each modality")


class FusedEmotionResult(BaseModel):
    """Final unified multi-modal emotion prediction response."""
    available: bool = Field(True, description="Whether at least one valid modality was available for prediction")
    status: str = Field("ok", description="Status string: 'ok', 'NO_VALID_MODALITIES_AVAILABLE', etc.")
    fusion_mode: str = Field("multimodal", description="'single_modality', 'multimodal', or 'unavailable'")
    emotion: str = Field(..., description="Final fused core emotion (or 'none' if unavailable)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall fused confidence score")
    probabilities: Dict[str, float] = Field(..., description="Fused probability distribution over 7 core emotions")
    valence: float = Field(0.0, description="Aggregate valence score")
    arousal: float = Field(0.0, description="Aggregate arousal score")
    active_modalities: List[str] = Field(default_factory=list, description="List of active modalities used in fusion")
    unavailable_modalities: List[str] = Field(default_factory=list, description="List of modalities that were present but unavailable")
    modalities_used: List[str] = Field(default_factory=list, description="Alias for active_modalities for backward compatibility")
    modality_results: Dict[str, ModalityResult] = Field(..., description="Individual per-modality prediction breakdowns")
    conflict: ConflictAnalysis = Field(..., description="Cross-modal conflict detection analysis")
    explainability: ExplainabilityReport = Field(..., description="Attribution and explainability report")
    timestamp: str = Field(..., description="ISO 8601 prediction timestamp")


class TextPredictionRequest(BaseModel):
    """Request payload for text emotion recognition."""
    text: str = Field(..., min_length=1, description="Raw text snippet to classify")


class HealthStatus(BaseModel):
    """System health and model readiness response."""
    status: str = Field("ok", description="Overall service status ('ok', 'degraded', 'unhealthy')")
    service: str = Field("emotion-recognition-api", description="Service identifier")
    version: str = Field("1.0.0", description="Application version")
    device: str = Field("cpu", description="Compute device in use ('cpu' or 'cuda')")
    models: Dict[str, Any] = Field(..., description="Readiness status of each modality pipeline")


class FusionEvaluateRequest(BaseModel):
    """Request payload for real-time fusion of pre-evaluated modality results."""
    results: Dict[str, ModalityResult] = Field(
        ...,
        description="Dictionary mapping modality names ('text', 'audio', 'vision') to their ModalityResult"
    )


