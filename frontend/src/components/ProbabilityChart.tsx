import type { FC } from 'react';
import { CORE_EMOTIONS } from '../types/emotion';
import type { CoreEmotion } from '../types/emotion';
import { BarChart3 } from 'lucide-react';

interface ProbabilityChartProps {
  probabilities: Record<CoreEmotion, number>;
  supportedEmotions?: string[];
}

const EMOTION_BAR_COLORS: Record<CoreEmotion, string> = {
  happy: '#10b981',
  sad: '#3b82f6',
  angry: '#ef4444',
  fear: '#a855f7',
  surprise: '#f59e0b',
  disgust: '#14b8a6',
  neutral: '#94a3b8',
};

export const ProbabilityChart: FC<ProbabilityChartProps> = ({
  probabilities,
  supportedEmotions,
}) => {
  // Sort descending by probability value for clarity
  const sortedEmotions = [...CORE_EMOTIONS].sort(
    (a, b) => (probabilities[b] || 0) - (probabilities[a] || 0)
  );

  return (
    <div className="analytics-card">
      <div className="analytics-header">
        <div className="flex-center gap-2">
          <BarChart3 size={18} className="text-indigo" />
          <h3 className="card-title">7-Class Emotion Probability Distribution</h3>
        </div>
        <span className="ontology-tag">Ekman Universal Ontology</span>
      </div>

      <div className="bars-container">
        {sortedEmotions.map((emotion) => {
          const prob = probabilities[emotion] ?? 0;
          const percentage = (prob * 100).toFixed(2);
          const isUnsupported = supportedEmotions && !supportedEmotions.includes(emotion);
          const color = EMOTION_BAR_COLORS[emotion];

          return (
            <div key={emotion} className="prob-bar-row">
              <div className="prob-label-group">
                <span className="emotion-name">{emotion}</span>
                {isUnsupported && (
                  <span className="unsupported-tag">Unsupported by model</span>
                )}
              </div>

              <div className="bar-track">
                <div
                  className="bar-fill"
                  style={{
                    width: `${Math.max(0, Math.min(100, prob * 100))}%`,
                    backgroundColor: color,
                  }}
                />
              </div>

              <span className="prob-percentage">{percentage}%</span>
            </div>
          );
        })}
      </div>
    </div>
  );
};
