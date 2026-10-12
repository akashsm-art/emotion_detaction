import React from 'react';
import { Loader2, BrainCircuit } from 'lucide-react';

interface LoadingStateProps {
  message?: string;
}

export const LoadingState: React.FC<LoadingStateProps> = ({
  message = 'Processing multi-modal inference...',
}) => {
  return (
    <div className="loading-container">
      <div className="loading-spinner-wrapper">
        <BrainCircuit size={36} className="loading-icon pulse" />
        <Loader2 size={44} className="loading-ring spin" />
      </div>
      <p className="loading-text">{message}</p>
      <span className="loading-subtext">Running neural feature extraction & tensor evaluation</span>
    </div>
  );
};
