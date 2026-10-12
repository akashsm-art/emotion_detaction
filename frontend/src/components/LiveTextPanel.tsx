import type { FC } from 'react';
import type { ModalityResult } from '../types/emotion';
import { MessageSquare, Loader2, Sparkles, XCircle } from 'lucide-react';

interface LiveTextPanelProps {
  text: string;
  setText: (value: string) => void;
  isAnalyzing: boolean;
  textResult: ModalityResult | null;
  isActive: boolean;
  onToggleActive: () => void;
  onClear: () => void;
  error?: string | null;
}

const EMOTION_COLORS: Record<string, string> = {
  happy: '#10b981',
  sad: '#3b82f6',
  angry: '#ef4444',
  fear: '#a855f7',
  surprise: '#f59e0b',
  disgust: '#14b8a6',
  neutral: '#94a3b8',
};

export const LiveTextPanel: FC<LiveTextPanelProps> = ({
  text,
  setText,
  isAnalyzing,
  textResult,
  isActive,
  onToggleActive,
  onClear,
  error,
}) => {
  const emotion = textResult?.emotion?.toLowerCase() || '';
  const emotionColor = EMOTION_COLORS[emotion] || '#94a3b8';
  const confidencePercent = textResult ? (textResult.confidence * 100).toFixed(1) : null;

  return (
    <div className={`live-panel-card ${isActive ? 'panel-active' : 'panel-idle'}`}>
      <div className="live-panel-header">
        <div className="live-panel-title-wrap">
          <MessageSquare size={16} className="text-indigo" />
          <span className="live-panel-title">LIVE TEXT</span>
        </div>
        <div className="live-panel-actions">
          {isAnalyzing && (
            <span className="live-analyzing-badge">
              <Loader2 size={12} className="spin-icon" /> Analyzing...
            </span>
          )}
          <button
            type="button"
            className={`live-status-pill ${isActive ? 'status-pill-active' : 'status-pill-inactive'}`}
            onClick={onToggleActive}
            title={isActive ? 'Click to pause text stream' : 'Click to activate text stream'}
          >
            <span className="status-dot-pulse" />
            <span>{isActive ? 'Active' : 'Paused'}</span>
          </button>
        </div>
      </div>

      <div className="live-panel-body">
        <div className="text-input-relative">
          <textarea
            className="live-textarea"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Type your thoughts, dialogue, or reaction here... Continuous RoBERTa inference updates automatically."
            rows={3}
            disabled={!isActive}
          />
          {text && (
            <button
              type="button"
              className="live-text-clear-btn"
              onClick={onClear}
              title="Clear text"
            >
              <XCircle size={14} />
            </button>
          )}
        </div>

        {error && <div className="live-panel-error">{error}</div>}

        <div className="live-panel-result-bar">
          {textResult && text.trim() ? (
            <div className="live-emotion-badge-row">
              <div className="live-emotion-pill" style={{ borderColor: emotionColor }}>
                <span className="emotion-dot" style={{ backgroundColor: emotionColor }} />
                <span className="emotion-name" style={{ color: emotionColor }}>
                  {textResult.emotion.toUpperCase()}
                </span>
                <span className="confidence-pill">{confidencePercent}%</span>
              </div>
              <span className="modality-tag">RoBERTa</span>
            </div>
          ) : (
            <div className="live-emotion-awaiting">
              <Sparkles size={13} className="text-muted" />
              <span>{isActive ? 'Type to stream emotion...' : 'Text stream paused'}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
