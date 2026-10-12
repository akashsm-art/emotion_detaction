import type { FC } from 'react';
import type { FusedEmotionResult } from '../types/emotion';
import { PieChart, CheckCircle2, AlertOctagon, HelpCircle } from 'lucide-react';

interface ModalityContributionPanelProps {
  result: FusedEmotionResult;
}

export const ModalityContributionPanel: FC<ModalityContributionPanelProps> = ({
  result,
}) => {
  const { explainability, active_modalities, unavailable_modalities } = result;
  const weights = explainability.modality_weights || {};

  return (
    <div className="analytics-card">
      <div className="analytics-header">
        <div className="flex-center gap-2">
          <PieChart size={18} className="text-violet" />
          <h3 className="card-title">Modality Contribution & Attribution</h3>
        </div>
        <span className="ontology-tag">Late Decision Fusion</span>
      </div>

      <div className="contribution-summary-row">
        <div className="summary-pill active-pill">
          <CheckCircle2 size={14} />
          <span>Active: {active_modalities.join(', ') || 'None'}</span>
        </div>
        {unavailable_modalities.length > 0 && (
          <div className="summary-pill unavailable-pill">
            <AlertOctagon size={14} />
            <span>Unavailable: {unavailable_modalities.join(', ')}</span>
          </div>
        )}
      </div>

      <div className="weights-grid">
        {Object.entries(weights).map(([modality, weight]) => {
          const percent = (weight * 100).toFixed(1);
          const isPrimary = explainability.primary_modality === modality;

          return (
            <div key={modality} className={`weight-card ${isPrimary ? 'primary-weight' : ''}`}>
              <div className="weight-header">
                <span className="modality-title">{modality.toUpperCase()}</span>
                {isPrimary && <span className="primary-pill">Primary Driver</span>}
              </div>
              <div className="weight-val">{percent}%</div>
              <div className="weight-bar-bg">
                <div
                  className="weight-bar-fill"
                  style={{ width: `${Math.min(100, Math.max(0, weight * 100))}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>

      {explainability.rationale && (
        <div className="rationale-box">
          <div className="flex-center gap-1 text-slate mb-1">
            <HelpCircle size={14} />
            <span className="text-xs font-semibold uppercase tracking-wider">Fusion Rationale</span>
          </div>
          <p className="rationale-text">{explainability.rationale}</p>
        </div>
      )}
    </div>
  );
};
