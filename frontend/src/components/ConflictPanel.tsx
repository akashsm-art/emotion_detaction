import type { FC } from 'react';
import type { ConflictAnalysis } from '../types/emotion';
import { GitCompare, CheckCircle2, AlertTriangle, ArrowRight } from 'lucide-react';

interface ConflictPanelProps {
  conflict: ConflictAnalysis;
  concordanceScore?: number;
}

export const ConflictPanel: FC<ConflictPanelProps> = ({
  conflict,
  concordanceScore,
}) => {
  const { has_conflict, conflict_score, divergent_pairs, summary } = conflict;
  const concordancePercent =
    concordanceScore !== undefined
      ? (concordanceScore * 100).toFixed(1)
      : ((1 - conflict_score) * 100).toFixed(1);

  return (
    <div className={`analytics-card ${has_conflict ? 'conflict-card-active' : 'concordance-card-active'}`}>
      <div className="analytics-header">
        <div className="flex-center gap-2">
          <GitCompare size={18} className={has_conflict ? 'text-amber' : 'text-emerald'} />
          <h3 className="card-title">Cross-Modal Concordance & Agreement</h3>
        </div>
        <span className={`concordance-score-badge ${has_conflict ? 'badge-divergence' : 'badge-concordance'}`}>
          Agreement: {concordancePercent}%
        </span>
      </div>

      <div className="conflict-status-banner">
        {has_conflict ? (
          <div className="flex items-start gap-3">
            <AlertTriangle size={20} className="text-amber flex-shrink-0 mt-1" />
            <div>
              <h4 className="conflict-heading">Cross-Modal Disagreement Detected</h4>
              <p className="conflict-description">
                {summary || 'Neural outputs across active modalities diverge in their top-1 emotion classification.'}
              </p>
              <p className="disclaimer-note mt-1">
                Note: Disagreement reflects divergence between modality feature distributions (e.g. linguistic sentiment vs acoustic prosody), not human veracity or deception.
              </p>
            </div>
          </div>
        ) : (
          <div className="flex items-start gap-3">
            <CheckCircle2 size={20} className="text-emerald flex-shrink-0 mt-1" />
            <div>
              <h4 className="concordance-heading">High Cross-Modal Concordance</h4>
              <p className="concordance-description">
                {summary || 'Active modalities exhibit concordant emotional signals with aligned top probability distributions.'}
              </p>
            </div>
          </div>
        )}
      </div>

      {divergent_pairs && divergent_pairs.length > 0 && (
        <div className="divergent-pairs-list">
          <span className="divergent-title">Divergent Modality Pairs:</span>
          {divergent_pairs.map((pair, idx) => (
            <div key={idx} className="divergent-pair-item">
              <div className="modality-pair-badge">
                <span className="modality-tag">{pair.modality_1.toUpperCase()}</span>
                <span className="emotion-pill-sub">{pair.emotion_1}</span>
                {pair.confidence_1 !== undefined && (
                  <span className="pair-conf">({(pair.confidence_1 * 100).toFixed(1)}%)</span>
                )}
              </div>
              <ArrowRight size={14} className="text-slate" />
              <div className="modality-pair-badge">
                <span className="modality-tag">{pair.modality_2.toUpperCase()}</span>
                <span className="emotion-pill-sub">{pair.emotion_2}</span>
                {pair.confidence_2 !== undefined && (
                  <span className="pair-conf">({(pair.confidence_2 * 100).toFixed(1)}%)</span>
                )}
              </div>
              <div className="divergence-metric">
                <span className="metric-text">JS Divergence: {(pair.pairwise_divergence).toFixed(3)}</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
