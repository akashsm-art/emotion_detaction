import type { FC } from 'react';
import type { ModalityResult } from '../types/emotion';
import type { LiveAudioStatus } from '../hooks/useLiveAudio';
import { LiveWaveform } from './LiveWaveform';
import { Mic, MicOff, Play, Pause, Square, AlertCircle, Loader2 } from 'lucide-react';

interface LiveAudioPanelProps {
  status: LiveAudioStatus;
  error: string | null;
  audioResult: ModalityResult | null;
  volume: number;
  analyserNode: AnalyserNode | null;
  onStart: () => void;
  onPause: () => void;
  onResume: () => void;
  onStop: () => void;
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

export const LiveAudioPanel: FC<LiveAudioPanelProps> = ({
  status,
  error,
  audioResult,
  volume,
  analyserNode,
  onStart,
  onPause,
  onResume,
  onStop,
}) => {
  const isListening = status === 'recording' || status === 'analyzing';
  const isPaused = status === 'paused';
  const isRequesting = status === 'requesting';

  const emotion = audioResult?.emotion?.toLowerCase() || '';
  const emotionColor = EMOTION_COLORS[emotion] || '#94a3b8';
  const confidencePercent = audioResult ? (audioResult.confidence * 100).toFixed(1) : null;

  return (
    <div className={`live-panel-card ${isListening ? 'panel-active' : 'panel-idle'}`}>
      <div className="live-panel-header">
        <div className="live-panel-title-wrap">
          <Mic size={16} className={isListening ? 'text-violet animate-pulse' : 'text-muted'} />
          <span className="live-panel-title">LIVE AUDIO</span>
        </div>
        <div className="live-panel-actions">
          {status === 'analyzing' && (
            <span className="live-analyzing-badge">
              <Loader2 size={12} className="spin-icon" /> Inferring...
            </span>
          )}
          <span
            className={`live-status-pill ${
              isListening
                ? 'status-pill-active'
                : isPaused
                ? 'status-pill-paused'
                : 'status-pill-inactive'
            }`}
          >
            <span className="status-dot-pulse" />
            <span>
              {status === 'recording'
                ? 'Listening'
                : status === 'analyzing'
                ? 'Processing'
                : status === 'paused'
                ? 'Paused'
                : status === 'requesting'
                ? 'Connecting'
                : 'Stopped'}
            </span>
          </span>
        </div>
      </div>

      <div className="live-panel-body">
        {/* Waveform Visualization Canvas */}
        <div className="waveform-box">
          <LiveWaveform
            analyserNode={analyserNode}
            isActive={isListening}
            isPaused={isPaused}
            height={68}
          />
          {/* Subtle live volume indicator meter */}
          <div className="volume-meter-track">
            <div
              className="volume-meter-fill"
              style={{
                width: `${Math.round(volume * 100)}%`,
                backgroundColor: volume > 0.7 ? '#ef4444' : '#8b5cf6',
              }}
            />
          </div>
        </div>

        {/* Permission / Device Error Banner */}
        {error && (
          <div className="live-panel-error">
            <AlertCircle size={14} className="flex-shrink-0 text-rose" />
            <span>{error}</span>
          </div>
        )}

        {/* Audio Controls */}
        <div className="live-audio-controls-row">
          {!isListening && !isPaused ? (
            <button
              type="button"
              className="btn-mic-start"
              onClick={onStart}
              disabled={isRequesting}
            >
              {isRequesting ? (
                <>
                  <Loader2 size={14} className="spin-icon" />
                  <span>Requesting Mic...</span>
                </>
              ) : (
                <>
                  <Mic size={14} />
                  <span>Start Microphone</span>
                </>
              )}
            </button>
          ) : (
            <div className="live-audio-btn-group">
              {isListening ? (
                <button
                  type="button"
                  className="btn-mic-action btn-mic-pause"
                  onClick={onPause}
                  title="Pause microphone recording"
                >
                  <Pause size={13} />
                  <span>Pause</span>
                </button>
              ) : (
                <button
                  type="button"
                  className="btn-mic-action btn-mic-resume"
                  onClick={onResume}
                  title="Resume microphone recording"
                >
                  <Play size={13} />
                  <span>Resume</span>
                </button>
              )}

              <button
                type="button"
                className="btn-mic-action btn-mic-stop"
                onClick={onStop}
                title="Stop microphone"
              >
                <Square size={13} />
                <span>Stop</span>
              </button>
            </div>
          )}
        </div>

        {/* Live Audio Emotion Result Bar */}
        <div className="live-panel-result-bar">
          {audioResult && isListening ? (
            <div className="live-emotion-badge-row">
              <div className="live-emotion-pill" style={{ borderColor: emotionColor }}>
                <span className="emotion-dot" style={{ backgroundColor: emotionColor }} />
                <span className="emotion-name" style={{ color: emotionColor }}>
                  {audioResult.emotion.toUpperCase()}
                </span>
                <span className="confidence-pill">{confidencePercent}%</span>
              </div>
              <span className="modality-tag">2D-CNN MFCC</span>
            </div>
          ) : (
            <div className="live-emotion-awaiting">
              {isListening ? (
                <span>🎙 Listening to speech prosody...</span>
              ) : (
                <>
                  <MicOff size={13} className="text-muted" />
                  <span>Microphone offline</span>
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
