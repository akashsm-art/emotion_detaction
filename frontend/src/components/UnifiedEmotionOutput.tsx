import type { FC } from 'react';
import type { FusedEmotionResult, ModalityResult } from '../types/emotion';
import { Activity, Compass, Layers, Sparkles, MessageSquare, Mic, Video } from 'lucide-react';

interface UnifiedEmotionOutputProps {
  fusedResult: FusedEmotionResult | null;
  textResult: ModalityResult | null;
  audioResult: ModalityResult | null;
  visionResult: ModalityResult | null;
  isTextActive: boolean;
  isAudioActive: boolean;
  isVisionActive: boolean;
  isFusing?: boolean;
}

const EMOTION_COLOR_MAP: Record<string, { bg: string; text: string; glow: string; border: string }> = {
  happy: {
    bg: 'rgba(16, 185, 129, 0.12)',
    text: '#10b981',
    glow: 'rgba(16, 185, 129, 0.35)',
    border: 'rgba(16, 185, 129, 0.4)',
  },
  sad: {
    bg: 'rgba(59, 130, 246, 0.12)',
    text: '#3b82f6',
    glow: 'rgba(59, 130, 246, 0.35)',
    border: 'rgba(59, 130, 246, 0.4)',
  },
  angry: {
    bg: 'rgba(239, 68, 68, 0.12)',
    text: '#ef4444',
    glow: 'rgba(239, 68, 68, 0.35)',
    border: 'rgba(239, 68, 68, 0.4)',
  },
  fear: {
    bg: 'rgba(168, 85, 247, 0.12)',
    text: '#a855f7',
    glow: 'rgba(168, 85, 247, 0.35)',
    border: 'rgba(168, 85, 247, 0.4)',
  },
  surprise: {
    bg: 'rgba(245, 158, 11, 0.12)',
    text: '#f59e0b',
    glow: 'rgba(245, 158, 11, 0.35)',
    border: 'rgba(245, 158, 11, 0.4)',
  },
  disgust: {
    bg: 'rgba(20, 184, 166, 0.12)',
    text: '#14b8a6',
    glow: 'rgba(20, 184, 166, 0.35)',
    border: 'rgba(20, 184, 166, 0.4)',
  },
  neutral: {
    bg: 'rgba(148, 163, 184, 0.12)',
    text: '#94a3b8',
    glow: 'rgba(148, 163, 184, 0.25)',
    border: 'rgba(148, 163, 184, 0.3)',
  },
};

export const UnifiedEmotionOutput: FC<UnifiedEmotionOutputProps> = ({
  fusedResult,
  textResult,
  audioResult,
  visionResult,
  isTextActive,
  isAudioActive,
  isVisionActive,
  isFusing = false,
}) => {
  if (!fusedResult || !fusedResult.available) {
    return (
      <div className="unified-hero-card unified-hero-empty">
        <div className="unified-empty-content">
          <div className="unified-empty-icon-wrap">
            <Sparkles size={36} className="text-indigo" />
          </div>
          <span className="unified-subheading">MULTI-MODAL EMOTION ENGINE</span>
          <h2 className="unified-empty-title">Awaiting Live Modality Streams</h2>
          <p className="unified-empty-desc">
            Activate Live Text, Microphone, or Camera above to begin real-time neural inference.
            Signals are fused continuously via the backend confidence-weighted late decision engine.
          </p>
          <div className="unified-ready-indicators">
            <div className={`ready-tag ${isTextActive ? 'ready-on' : 'ready-off'}`}>
              <MessageSquare size={13} /> Text {isTextActive ? 'Active' : 'Idle'}
            </div>
            <div className={`ready-tag ${isAudioActive ? 'ready-on' : 'ready-off'}`}>
              <Mic size={13} /> Audio {isAudioActive ? 'Active' : 'Idle'}
            </div>
            <div className={`ready-tag ${isVisionActive ? 'ready-on' : 'ready-off'}`}>
              <Video size={13} /> Vision {isVisionActive ? 'Active' : 'Idle'}
            </div>
          </div>
        </div>
      </div>
    );
  }

  const emotion = fusedResult.emotion.toLowerCase();
  const colors = EMOTION_COLOR_MAP[emotion] || EMOTION_COLOR_MAP.neutral;
  const confidencePercent = (fusedResult.confidence * 100).toFixed(1);

  // Affective state description
  let affectDescription = 'Neutral emotional state';
  if (fusedResult.valence > 0.2 && fusedResult.arousal > 0.2) {
    affectDescription = 'Positive, high-arousal emotional state';
  } else if (fusedResult.valence > 0.2) {
    affectDescription = 'Positive, calm emotional state';
  } else if (fusedResult.valence < -0.2 && fusedResult.arousal > 0.2) {
    affectDescription = 'Negative, high-arousal emotional state';
  } else if (fusedResult.valence < -0.2) {
    affectDescription = 'Negative, low-energy emotional state';
  }

  const activeCount = fusedResult.active_modalities?.length || 1;

  return (
    <div
      className="unified-hero-card"
      style={{
        borderColor: colors.border,
        boxShadow: `0 10px 40px -10px ${colors.glow}, inset 0 1px 0 rgba(255, 255, 255, 0.1)`,
      }}
    >
      {/* Top Header Strip */}
      <div className="unified-hero-top-strip">
        <div className="unified-mode-pill">
          <Layers size={13} />
          <span>
            {activeCount > 1
              ? `Real-Time Fusion (${activeCount} Modalities Active)`
              : `Single Modality (${fusedResult.active_modalities[0] || 'Active'})`}
          </span>
        </div>
        {isFusing && (
          <span className="live-fusing-indicator">
            <Activity size={12} className="animate-pulse" /> Updating...
          </span>
        )}
      </div>

      {/* Dominant Emotion Output */}
      <div className="unified-hero-main-display">
        <span className="unified-hero-sublabel">CURRENT EMOTIONAL STATE</span>
        <h1 className="unified-hero-emotion-title" style={{ color: colors.text }}>
          {fusedResult.emotion.toUpperCase()}
        </h1>
        <div className="unified-confidence-number" style={{ color: colors.text }}>
          {confidencePercent}%
        </div>
        <p className="unified-affect-label">{affectDescription}</p>
      </div>

      {/* Circumplex Coordinates */}
      <div className="unified-circumplex-strip">
        <div className="circumplex-badge">
          <Compass size={13} className="text-indigo" />
          <span>
            Valence:{' '}
            <strong>
              {fusedResult.valence > 0
                ? `+${fusedResult.valence.toFixed(2)}`
                : fusedResult.valence.toFixed(2)}
            </strong>
          </span>
        </div>
        <div className="circumplex-badge">
          <Compass size={13} className="text-violet" />
          <span>
            Arousal:{' '}
            <strong>
              {fusedResult.arousal > 0
                ? `+${fusedResult.arousal.toFixed(2)}`
                : fusedResult.arousal.toFixed(2)}
            </strong>
          </span>
        </div>
      </div>

      {/* Compact Modality Signals Grid */}
      <div className="unified-signals-section">
        <span className="signals-header-label">INPUT MODALITY SIGNALS</span>
        <div className="unified-signals-grid">
          {/* Text Signal */}
          <div
            className={`signal-cell ${
              isTextActive && textResult ? 'signal-cell-active' : 'signal-cell-disabled'
            }`}
          >
            <div className="signal-cell-header">
              <MessageSquare size={13} />
              <span>Text</span>
            </div>
            {isTextActive && textResult ? (
              <div className="signal-cell-body">
                <span className="signal-emotion">{textResult.emotion.toUpperCase()}</span>
                <span className="signal-confidence">
                  {(textResult.confidence * 100).toFixed(0)}%
                </span>
              </div>
            ) : (
              <span className="signal-inactive-text">
                {isTextActive ? 'Awaiting text...' : 'Inactive'}
              </span>
            )}
          </div>

          {/* Audio Signal */}
          <div
            className={`signal-cell ${
              isAudioActive && audioResult ? 'signal-cell-active' : 'signal-cell-disabled'
            }`}
          >
            <div className="signal-cell-header">
              <Mic size={13} />
              <span>Audio</span>
            </div>
            {isAudioActive && audioResult ? (
              <div className="signal-cell-body">
                <span className="signal-emotion">{audioResult.emotion.toUpperCase()}</span>
                <span className="signal-confidence">
                  {(audioResult.confidence * 100).toFixed(0)}%
                </span>
              </div>
            ) : (
              <span className="signal-inactive-text">
                {isAudioActive ? 'Listening...' : 'Inactive'}
              </span>
            )}
          </div>

          {/* Vision Signal */}
          <div
            className={`signal-cell ${
              isVisionActive && visionResult ? 'signal-cell-active' : 'signal-cell-disabled'
            }`}
          >
            <div className="signal-cell-header">
              <Video size={13} />
              <span>Vision</span>
            </div>
            {isVisionActive && visionResult ? (
              <div className="signal-cell-body">
                <span className="signal-emotion">{visionResult.emotion.toUpperCase()}</span>
                <span className="signal-confidence">
                  {(visionResult.confidence * 100).toFixed(0)}%
                </span>
              </div>
            ) : (
              <span className="signal-inactive-text">
                {isVisionActive ? 'Tracking face...' : 'Inactive'}
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Fusion Confidence Progress Bar */}
      <div className="unified-progress-section">
        <div className="progress-labels-row">
          <span className="progress-title">Fusion Confidence</span>
          <span className="progress-value">{confidencePercent}%</span>
        </div>
        <div className="confidence-progress-track">
          <div
            className="confidence-progress-fill"
            style={{
              width: `${Math.min(100, Math.max(0, fusedResult.confidence * 100))}%`,
              backgroundColor: colors.text,
              boxShadow: `0 0 12px ${colors.glow}`,
            }}
          />
        </div>
      </div>
    </div>
  );
};
