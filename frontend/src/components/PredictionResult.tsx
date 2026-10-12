import type { FC } from 'react';
import type { FusedEmotionResult, ModalityResult } from '../types/emotion';
import { Compass, Clock, Layers } from 'lucide-react';

interface PredictionResultProps {
  result: FusedEmotionResult | ModalityResult;
}

const EMOTION_COLOR_MAP: Record<string, { bg: string; text: string; border: string }> = {
  happy: { bg: 'rgba(16, 185, 129, 0.15)', text: '#10b981', border: '#10b981' },
  sad: { bg: 'rgba(59, 130, 246, 0.15)', text: '#3b82f6', border: '#3b82f6' },
  angry: { bg: 'rgba(239, 68, 68, 0.15)', text: '#ef4444', border: '#ef4444' },
  fear: { bg: 'rgba(168, 85, 247, 0.15)', text: '#a855f7', border: '#a855f7' },
  surprise: { bg: 'rgba(245, 158, 11, 0.15)', text: '#f59e0b', border: '#f59e0b' },
  disgust: { bg: 'rgba(20, 184, 166, 0.15)', text: '#14b8a6', border: '#14b8a6' },
  neutral: { bg: 'rgba(148, 163, 184, 0.15)', text: '#94a3b8', border: '#94a3b8' },
  none: { bg: 'rgba(100, 116, 139, 0.15)', text: '#64748b', border: '#64748b' },
};

export const PredictionResult: FC<PredictionResultProps> = ({ result }) => {
  const emotion = result.emotion.toLowerCase();
  const colors = EMOTION_COLOR_MAP[emotion] || EMOTION_COLOR_MAP.neutral;
  const isFused = 'fusion_mode' in result;
  const confidencePercent = (result.confidence * 100).toFixed(2);

  return (
    <div className="result-hero-card" style={{ borderColor: colors.border }}>
      <div className="result-hero-header">
        <div className="result-mode-tag">
          <Layers size={14} />
          <span>
            {isFused
              ? result.fusion_mode === 'single_modality'
                ? 'Single Modality Prediction'
                : 'Confidence-Weighted Multi-Modal Prediction'
              : `Single Modality (${(result as ModalityResult).modality})`}
          </span>
        </div>
        {'timestamp' in result && (
          <span className="timestamp-tag">
            <Clock size={12} /> {new Date(result.timestamp).toLocaleTimeString()}
          </span>
        )}
      </div>

      <div className="result-hero-body">
        <div className="emotion-display-block">
          <span className="emotion-sublabel">Classified Emotion</span>
          <h2 className="emotion-title" style={{ color: colors.text }}>
            {emotion.toUpperCase()}
          </h2>
        </div>

        <div className="confidence-metric-block">
          <span className="metric-label">Model Confidence</span>
          <span className="confidence-number">{confidencePercent}%</span>
          <span className="disclaimer-note">Neural model certainty score (not clinical accuracy)</span>
        </div>
      </div>

      {/* Valence & Arousal Coordinates */}
      <div className="circumplex-row">
        <div className="circumplex-badge">
          <Compass size={14} className="text-indigo" />
          <span>Valence: <strong>{result.valence > 0 ? `+${result.valence.toFixed(2)}` : result.valence.toFixed(2)}</strong></span>
          <span className="circumplex-sub">{result.valence >= 0 ? '(Positive affect)' : '(Negative affect)'}</span>
        </div>
        <div className="circumplex-badge">
          <Compass size={14} className="text-violet" />
          <span>Arousal: <strong>{result.arousal > 0 ? `+${result.arousal.toFixed(2)}` : result.arousal.toFixed(2)}</strong></span>
          <span className="circumplex-sub">{result.arousal >= 0 ? '(High intensity)' : '(Calm / Low intensity)'}</span>
        </div>
      </div>
    </div>
  );
};
