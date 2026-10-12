import type { FC } from 'react';
import type { HealthStatus } from '../types/emotion';
import { Activity, ShieldCheck, AlertCircle } from 'lucide-react';

interface HeaderProps {
  health: HealthStatus | null;
  loadingHealth: boolean;
  onRefreshHealth: () => void;
}

export const Header: FC<HeaderProps> = ({
  health,
  loadingHealth,
  onRefreshHealth,
}) => {
  const isHealthy = health?.status === 'ok';

  return (
    <header className="header-container">
      <div className="header-brand">
        <div className="logo-badge">
          <Activity className="logo-icon" size={24} />
        </div>
        <div>
          <h1 className="header-title">Emotion Intelligence</h1>
          <p className="header-subtitle">Multi-Modal Emotion Recognition Platform</p>
        </div>
      </div>

      <div className="header-actions">
        <button
          className="health-badge-btn"
          onClick={onRefreshHealth}
          title="Click to refresh system status"
          disabled={loadingHealth}
        >
          {isHealthy ? (
            <ShieldCheck size={16} className="text-emerald" />
          ) : (
            <AlertCircle size={16} className="text-amber" />
          )}
          <span className="health-text">
            {loadingHealth
              ? 'Checking status...'
              : health
              ? `System: ${health.status.toUpperCase()}`
              : 'Backend Offline'}
          </span>
          <span className="device-tag">{health?.device ? `[${health.device.toUpperCase()}]` : ''}</span>
        </button>
      </div>
    </header>
  );
};
