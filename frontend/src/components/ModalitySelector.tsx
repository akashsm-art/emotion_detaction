import React from 'react';
import { Type, Mic, Eye } from 'lucide-react';

interface ModalitySelectorProps {
  selectedModalities: Set<string>;
  onToggleModality: (modality: string) => void;
}

export const ModalitySelector: React.FC<ModalitySelectorProps> = ({
  selectedModalities,
  onToggleModality,
}) => {
  const isSelected = (m: string) => selectedModalities.has(m);

  return (
    <div className="modality-selector-card">
      <div className="modality-selector-header">
        <span className="section-label">Select Active Modalities</span>
        <span className="mode-pill">
          {selectedModalities.size > 1
            ? `Multi-Modal (${selectedModalities.size} active)`
            : selectedModalities.size === 1
            ? 'Single Modality'
            : 'No modality selected'}
        </span>
      </div>

      <div className="modality-button-group">
        <button
          type="button"
          className={`modality-btn ${isSelected('text') ? 'active' : ''}`}
          onClick={() => onToggleModality('text')}
        >
          <Type size={18} />
          <span>Text (RoBERTa)</span>
          <span className="badge-sub">Operational</span>
        </button>

        <button
          type="button"
          className={`modality-btn ${isSelected('audio') ? 'active' : ''}`}
          onClick={() => onToggleModality('audio')}
        >
          <Mic size={18} />
          <span>Audio (CNN MFCC)</span>
          <span className="badge-sub">Operational</span>
        </button>

        <button
          type="button"
          className={`modality-btn ${isSelected('vision') ? 'active' : ''}`}
          onClick={() => onToggleModality('vision')}
        >
          <Eye size={18} />
          <span>Vision (AlexNet)</span>
          <span className="badge-sub">Operational</span>
        </button>
      </div>
    </div>
  );
};
