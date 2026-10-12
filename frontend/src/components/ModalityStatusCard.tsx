import type { FC } from 'react';
import type { HealthStatus } from '../types/emotion';
import { Server, AlertTriangle } from 'lucide-react';

interface ModalityStatusCardProps {
  health: HealthStatus | null;
  loading: boolean;
}

export const ModalityStatusCard: FC<ModalityStatusCardProps> = ({
  health,
  loading,
}) => {
  if (loading && !health) {
    return (
      <div className="status-overview-card loading-skeleton">
        <span className="text-sm text-slate">Querying model endpoints...</span>
      </div>
    );
  }

  const isBackendUp = !!health && (health.status === 'ok' || health.status === 'degraded');
  const textInfo = health?.models?.text;
  const audioInfo = health?.models?.audio;
  const visionInfo = health?.models?.vision;

  return (
    <div className="status-overview-card">
      <div className="status-card-header">
        <div className="flex-center gap-2">
          <Server size={16} className="text-indigo" />
          <span className="status-header-title">System & Model Status</span>
        </div>
        <span className="device-tag-sm">Device: {health?.device?.toUpperCase() || 'CPU'}</span>
      </div>

      <div className="status-grid">
        {/* Backend API */}
        <div className="status-item">
          <div className="status-indicator-dot">
            {isBackendUp ? (
              <span className="dot dot-green" />
            ) : (
              <span className="dot dot-red" />
            )}
          </div>
          <div className="status-details">
            <span className="status-name">Backend API</span>
            <span className="status-val">{isBackendUp ? 'Online (FastAPI)' : 'Offline'}</span>
          </div>
        </div>

        {/* Text Model */}
        <div className="status-item">
          <div className="status-indicator-dot">
            {textInfo?.operational ? (
              <span className="dot dot-green" />
            ) : (
              <span className="dot dot-red" />
            )}
          </div>
          <div className="status-details">
            <span className="status-name">Text Model</span>
            <span className="status-val">
              {textInfo?.operational ? 'Operational (RoBERTa)' : 'Inactive'}
            </span>
          </div>
        </div>

        {/* Audio Model */}
        <div className="status-item">
          <div className="status-indicator-dot">
            {audioInfo?.operational ? (
              <span className="dot dot-green" />
            ) : (
              <span className="dot dot-red" />
            )}
          </div>
          <div className="status-details">
            <span className="status-name">Audio Model</span>
            <span className="status-val">
              {audioInfo?.operational ? 'Operational (CNN MFCC)' : 'Inactive'}
            </span>
          </div>
        </div>

        {/* Vision Model */}
        <div className={`status-item ${visionInfo?.operational ? '' : 'status-warning'}`}>
          <div className="status-indicator-dot">
            {visionInfo?.operational ? (
              <span className="dot dot-green" />
            ) : (
              <span className="dot dot-amber" />
            )}
          </div>
          <div className="status-details">
            <span className="status-name">Vision Model</span>
            <span className={`status-val ${visionInfo?.operational ? '' : 'text-amber'}`}>
              {visionInfo?.operational ? 'Operational (AlexNet CNN)' : 'Unavailable (Weights Missing)'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

interface UnavailableModalityBannerProps {
  unavailableModalities: string[];
  activeModalities: string[];
}

export const UnavailableModalityBanner: FC<UnavailableModalityBannerProps> = ({
  unavailableModalities,
  activeModalities,
}) => {
  if (!unavailableModalities || unavailableModalities.length === 0) return null;

  return (
    <div className="unavailable-modality-card">
      <div className="flex items-start gap-3">
        <AlertTriangle size={22} className="text-amber flex-shrink-0 mt-1" />
        <div className="unavailable-content">
          <div className="flex items-center gap-2">
            <span className="unavailable-tag">MODALITY NOTICE</span>
            <span className="unavailable-names">
              {unavailableModalities.map((m) => m.toUpperCase()).join(', ')} UNAVAILABLE
            </span>
          </div>
          <p className="unavailable-explanation">
            Trained model checkpoint weights for the indicated modality were not found or failed to load.
            {activeModalities.length > 0 && (
              <span>
                {' '}Active modalities (<strong>{activeModalities.join(', ')}</strong>) were still fully processed and fused without disruption.
              </span>
            )}
          </p>
        </div>
      </div>
    </div>
  );
};
