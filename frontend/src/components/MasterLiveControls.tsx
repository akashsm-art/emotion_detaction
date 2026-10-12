import type { FC } from 'react';
import { Play, Square, RotateCcw } from 'lucide-react';


interface MasterLiveControlsProps {
  isLiveActive: boolean;
  activeCount: number;
  onStartAll: () => void;
  onStopAll: () => void;
  onReset: () => void;
  disabled?: boolean;
}

export const MasterLiveControls: FC<MasterLiveControlsProps> = ({
  isLiveActive,
  activeCount,
  onStartAll,
  onStopAll,
  onReset,
  disabled = false,
}) => {
  return (
    <div className="master-controls-bar">
      <div className="master-controls-left">
        <div className={`master-live-status-indicator ${isLiveActive ? 'live-on' : 'live-off'}`}>
          <span className="live-indicator-dot" />
          <span className="live-indicator-text">
            {isLiveActive
              ? `LIVE ANALYSIS ACTIVE (${activeCount} Modalities Online)`
              : 'LIVE ANALYSIS READY'}
          </span>
        </div>
      </div>

      <div className="master-controls-right">
        {!isLiveActive ? (
          <button
            type="button"
            className="btn-master-start"
            onClick={onStartAll}
            disabled={disabled}
          >
            <Play size={18} className="fill-current" />
            <span>START LIVE ANALYSIS</span>
          </button>
        ) : (
          <button
            type="button"
            className="btn-master-stop"
            onClick={onStopAll}
            disabled={disabled}
          >
            <Square size={16} />
            <span>STOP LIVE ANALYSIS</span>
          </button>
        )}

        <button
          type="button"
          className="btn-master-reset"
          onClick={onReset}
          disabled={disabled}
          title="Reset all modalities and state"
        >
          <RotateCcw size={16} />
          <span>Reset</span>
        </button>
      </div>
    </div>
  );
};
