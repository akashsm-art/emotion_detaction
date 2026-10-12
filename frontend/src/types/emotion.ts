/**
 * TypeScript Data Models for Multi-Modal Emotion Recognition.
 * Synchronized with the FastAPI backend schemas (core/schemas.py & core/ontology.py).
 */

export type CoreEmotion =
  | 'angry'
  | 'disgust'
  | 'fear'
  | 'happy'
  | 'neutral'
  | 'sad'
  | 'surprise';

export const CORE_EMOTIONS: CoreEmotion[] = [
  'angry',
  'disgust',
  'fear',
  'happy',
  'neutral',
  'sad',
  'surprise',
];

export interface ModalityResult {
  modality: 'text' | 'audio' | 'vision';
  available: boolean;
  status: string;
  emotion: string;
  confidence: number;
  probabilities: Record<CoreEmotion, number>;
  supported_emotions: string[];
  valence: number;
  arousal: number;
  metadata: Record<string, any>;
}

export interface DivergentPair {
  modality_1: string;
  emotion_1: string;
  confidence_1?: number;
  modality_2: string;
  emotion_2: string;
  confidence_2?: number;
  pairwise_divergence: number;
}

export interface ConflictAnalysis {
  has_conflict: boolean;
  conflict_score: number;
  divergent_pairs: DivergentPair[];
  summary: string;
}

export interface ExplainabilityReport {
  primary_modality: string;
  concordance_score: number;
  modality_weights: Record<string, number>;
  configured_weights?: Record<string, number>;
  rationale: string;
  key_features: Record<string, any>;
}

export interface FusedEmotionResult {
  available: boolean;
  status: string;
  fusion_mode: 'single_modality' | 'multimodal' | 'unavailable';
  emotion: string;
  confidence: number;
  probabilities: Record<CoreEmotion, number>;
  valence: number;
  arousal: number;
  active_modalities: string[];
  unavailable_modalities: string[];
  modalities_used?: string[];
  modality_results: Record<string, ModalityResult>;
  conflict: ConflictAnalysis;
  explainability: ExplainabilityReport;
  timestamp: string;
}

export interface ModelReadinessInfo {
  operational: boolean;
  model_name?: string;
  model_architecture?: string;
  framework?: string;
  weights?: string;
  supported_classes?: string[] | number;
  status?: string;
  reason?: string;
  face_detection_available?: boolean;
}

export interface HealthStatus {
  status: 'ok' | 'degraded' | 'unhealthy';
  service: string;
  version: string;
  device: string;
  models: {
    text: ModelReadinessInfo;
    audio: ModelReadinessInfo;
    vision: ModelReadinessInfo;
  };
}
