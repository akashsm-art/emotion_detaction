import type { FC } from 'react';
import type { ModalityResult } from '../types/emotion';
import type { LiveWebcamStatus } from '../hooks/useLiveWebcam';
import { Video, VideoOff, Square, AlertCircle, Loader2, ScanFace } from 'lucide-react';

interface LiveWebcamPanelProps {
  status: LiveWebcamStatus;
  error: string | null;
  visionResult: ModalityResult | null;
  videoRef: React.RefObject<HTMLVideoElement | null>;
  faceDetected: boolean;
  facesCount: number;
  cameraFps: number;
  inferenceRateHz: number;
  onStart: () => void;
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

export const LiveWebcamPanel: FC<LiveWebcamPanelProps> = ({
  status,
  error,
  visionResult,
  videoRef,
  faceDetected,
  facesCount,
  cameraFps,
  inferenceRateHz,
  onStart,
  onStop,
}) => {
  const isStreaming = status === 'streaming' || status === 'analyzing';
  const isRequesting = status === 'requesting';

  const emotion = visionResult?.emotion?.toLowerCase() || '';
  const emotionColor = EMOTION_COLORS[emotion] || '#94a3b8';
  const confidencePercent = visionResult ? (visionResult.confidence * 100).toFixed(1) : null;

  return (
    <div className={`live-panel-card ${isStreaming ? 'panel-active' : 'panel-idle'}`}>
      <div className="live-panel-header">
        <div className="live-panel-title-wrap">
          <Video size={16} className={isStreaming ? 'text-cyan animate-pulse' : 'text-muted'} />
          <span className="live-panel-title">LIVE WEBCAM</span>
        </div>
        <div className="live-panel-actions">
          {isStreaming && (
            <div className="webcam-rates-badge">
              <span title="Camera Feed Frame Rate">Cam: {cameraFps || 30} FPS</span>
              <span className="rate-divider">&bull;</span>
              <span title="Neural Inference Sampling Rate">Inf: {inferenceRateHz || 2.0} Hz</span>
            </div>
          )}
          <span
            className={`live-status-pill ${
              isStreaming ? 'status-pill-active' : 'status-pill-inactive'
            }`}
          >
            <span className="status-dot-pulse" />
            <span>{isStreaming ? 'Streaming' : isRequesting ? 'Connecting' : 'Stopped'}</span>
          </span>
        </div>
      </div>

      <div className="live-panel-body">
        {/* Video Screen & HUD Container */}
        <div className="webcam-viewport-box">
          <video
            ref={videoRef}
            className={`webcam-video-element ${isStreaming ? 'video-active' : 'video-hidden'}`}
            autoPlay
            playsInline
            muted
          />

          {!isStreaming && (
            <div className="webcam-placeholder-view">
              <VideoOff size={32} className="text-muted mb-1" />
              <span className="webcam-placeholder-text">Camera stream inactive</span>
            </div>
          )}

          {/* HUD Overlay when streaming */}
          {isStreaming && (
            <div className="webcam-hud-overlay">
              <div
                className={`hud-face-tag ${
                  faceDetected ? 'hud-face-detected' : 'hud-face-missing'
                }`}
              >
                <ScanFace size={13} />
                <span>
                  {faceDetected
                    ? `Face Detected (${facesCount})`
                    : 'No Face Detected'}
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Permission / Device Error Banner */}
        {error && (
          <div className="live-panel-error">
            <AlertCircle size={14} className="flex-shrink-0 text-rose" />
            <span>{error}</span>
          </div>
        )}

        {/* Webcam Controls */}
        <div className="live-webcam-controls-row">
          {!isStreaming ? (
            <button
              type="button"
              className="btn-webcam-start"
              onClick={onStart}
              disabled={isRequesting}
            >
              {isRequesting ? (
                <>
                  <Loader2 size={14} className="spin-icon" />
                  <span>Requesting Camera...</span>
                </>
              ) : (
                <>
                  <Video size={14} />
                  <span>Start Camera</span>
                </>
              )}
            </button>
          ) : (
            <button
              type="button"
              className="btn-webcam-stop"
              onClick={onStop}
              title="Stop camera stream"
            >
              <Square size={13} />
              <span>Stop Camera</span>
            </button>
          )}
        </div>

        {/* Live Vision Emotion Result Bar */}
        <div className="live-panel-result-bar">
          {isStreaming ? (
            faceDetected && visionResult ? (
              <div className="live-emotion-badge-row">
                <div className="live-emotion-pill" style={{ borderColor: emotionColor }}>
                  <span className="emotion-dot" style={{ backgroundColor: emotionColor }} />
                  <span className="emotion-name" style={{ color: emotionColor }}>
                    {visionResult.emotion.toUpperCase()}
                  </span>
                  <span className="confidence-pill">{confidencePercent}%</span>
                </div>
                <span className="modality-tag">AlexNet</span>
              </div>
            ) : (
              <div className="live-emotion-awaiting text-amber">
                <ScanFace size={13} />
                <span>No face detected in frame</span>
              </div>
            )
          ) : (
            <div className="live-emotion-awaiting">
              <VideoOff size={13} className="text-muted" />
              <span>Camera offline</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
