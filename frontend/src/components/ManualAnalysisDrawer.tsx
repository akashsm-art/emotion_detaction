import { useState } from 'react';
import type { FC } from 'react';
import type { ModalityResult, FusedEmotionResult } from '../types/emotion';
import { ModalitySelector } from './ModalitySelector';
import { TextInput } from './TextInput';
import { AudioUploader } from './AudioUploader';
import { ImageUploader } from './ImageUploader';
import { PredictionResult } from './PredictionResult';
import { ProbabilityChart } from './ProbabilityChart';
import { ModalityContributionPanel } from './ModalityContributionPanel';
import { ConflictPanel } from './ConflictPanel';
import { UnavailableModalityBanner } from './ModalityStatusCard';
import { LoadingState } from './LoadingState';
import {
  predictText,
  predictAudio,
  predictVision,
  predictMultimodal,
  ApiError,
} from '../services/api';
import { ChevronDown, ChevronUp, Play, RotateCcw, AlertCircle, FileSpreadsheet } from 'lucide-react';

export const ManualAnalysisDrawer: FC = () => {
  const [isOpen, setIsOpen] = useState<boolean>(false);


  // Modality Selection: default Text and Audio active
  const [selectedModalities, setSelectedModalities] = useState<Set<string>>(
    new Set(['text', 'audio'])
  );

  // Inputs State
  const [textInput, setTextInput] = useState<string>(
    'I am absolutely thrilled and overjoyed with these incredible results!'
  );
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);

  // Execution & Results State
  const [loading, setLoading] = useState<boolean>(false);
  const [loadingMessage, setLoadingMessage] = useState<string>('');
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<FusedEmotionResult | ModalityResult | null>(null);

  const handleToggleModality = (modality: string) => {
    setSelectedModalities((prev) => {
      const next = new Set(prev);
      if (next.has(modality)) {
        if (next.size > 1) {
          next.delete(modality);
        }
      } else {
        next.add(modality);
      }
      return next;
    });
  };

  const handleReset = () => {
    setTextInput('');
    setAudioFile(null);
    setImageFile(null);
    setResult(null);
    setError(null);
  };

  const handleAnalyze = async () => {
    setError(null);
    setResult(null);

    const hasText = selectedModalities.has('text');
    const hasAudio = selectedModalities.has('audio');
    const hasVision = selectedModalities.has('vision');

    if (hasText && !textInput.trim()) {
      setError('Please provide text input or uncheck the Text modality.');
      return;
    }
    if (hasAudio && !audioFile) {
      setError('Please upload an audio file or uncheck the Audio modality.');
      return;
    }
    if (hasVision && !imageFile) {
      setError('Please upload an image file or uncheck the Vision modality.');
      return;
    }

    setLoading(true);

    try {
      const isMulti = selectedModalities.size > 1;

      if (isMulti) {
        setLoadingMessage('Running multimodal fusion & feature integration...');
        const res = await predictMultimodal({
          text: hasText ? textInput.trim() : undefined,
          audio: hasAudio && audioFile ? audioFile : undefined,
          image: hasVision && imageFile ? imageFile : undefined,
        });
        setResult(res);
      } else if (hasText) {
        setLoadingMessage('Analyzing text using RoBERTa sequence classification...');
        const res = await predictText(textInput);
        setResult(res);
      } else if (hasAudio) {
        setLoadingMessage('Processing speech prosody with CNN MFCC model...');
        if (!audioFile) throw new Error('Audio file missing');
        const res = await predictAudio(audioFile);
        setResult(res);
      } else if (hasVision) {
        setLoadingMessage('Classifying facial emotion using trained AlexNet model...');
        if (!imageFile) throw new Error('Image file missing');
        const res = await predictVision(imageFile);
        setResult(res);
      }
    } catch (err: any) {
      console.error('Inference error:', err);
      if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError(err.message || 'An unexpected error occurred during emotion analysis.');
      }
    } finally {
      setLoading(false);
    }
  };

  const isFusedResult = (res: any): res is FusedEmotionResult => {
    return res && 'fusion_mode' in res;
  };

  return (
    <section className="manual-drawer-section">
      <button
        type="button"
        className="manual-drawer-toggle"
        onClick={() => setIsOpen((prev) => !prev)}
      >
        <div className="manual-drawer-toggle-left">
          <FileSpreadsheet size={18} className="text-muted" />
          <div className="manual-drawer-title-group">
            <span className="manual-drawer-heading">Manual Analysis & Benchmark Input</span>
            <span className="manual-drawer-sub">
              Upload pre-recorded audio files, static images, and custom text for reproducible offline testing
            </span>
          </div>
        </div>
        <div className="manual-drawer-toggle-right">
          <span className="manual-drawer-badge">{isOpen ? 'Collapse' : 'Expand Mode'}</span>
          {isOpen ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
        </div>
      </button>

      {isOpen && (
        <div className="manual-drawer-content">
          <div className="manual-selector-bar">
            <ModalitySelector
              selectedModalities={selectedModalities}
              onToggleModality={handleToggleModality}
            />
          </div>

          <div className="dashboard-grid mt-4">
            {/* Input Column */}
            <div className="inputs-column">
              {selectedModalities.has('text') && (
                <TextInput
                  text={textInput}
                  onChange={setTextInput}
                  disabled={loading}
                />
              )}

              {selectedModalities.has('audio') && (
                <AudioUploader
                  file={audioFile}
                  onFileSelect={setAudioFile}
                  disabled={loading}
                />
              )}

              {selectedModalities.has('vision') && (
                <ImageUploader
                  file={imageFile}
                  onFileSelect={setImageFile}
                  disabled={loading}
                />
              )}

              <div className="action-bar">
                <button
                  type="button"
                  className="btn-primary"
                  onClick={handleAnalyze}
                  disabled={loading}
                >
                  <Play size={18} />
                  <span>
                    {loading
                      ? 'Processing...'
                      : selectedModalities.size > 1
                      ? 'Run Multi-Modal Fusion'
                      : 'Classify Emotion'}
                  </span>
                </button>

                <button
                  type="button"
                  className="btn-secondary"
                  onClick={handleReset}
                  disabled={loading}
                  title="Reset all manual inputs and results"
                >
                  <RotateCcw size={16} />
                  <span>Reset</span>
                </button>
              </div>

              {error && (
                <div className="error-banner">
                  <AlertCircle size={20} className="text-rose flex-shrink-0" />
                  <div className="error-text-content">
                    <strong>Analysis Notice:</strong>
                    <p>{error}</p>
                  </div>
                </div>
              )}
            </div>

            {/* Results Column */}
            <div className="results-column">
              {loading && <LoadingState message={loadingMessage} />}

              {!loading && !result && !error && (
                <div className="manual-empty-state">
                  <p>Provide test files on the left and click <strong>Run Multi-Modal Fusion</strong>.</p>
                </div>
              )}

              {!loading && result && (
                <div className="results-stack">
                  <PredictionResult result={result} />

                  {isFusedResult(result) && (
                    <UnavailableModalityBanner
                      unavailableModalities={result.unavailable_modalities}
                      activeModalities={result.active_modalities}
                    />
                  )}

                  <ProbabilityChart
                    probabilities={result.probabilities}
                    supportedEmotions={
                      'supported_emotions' in result
                        ? (result as ModalityResult).supported_emotions
                        : undefined
                    }
                  />

                  {isFusedResult(result) && result.active_modalities.length > 0 && (
                    <ModalityContributionPanel result={result} />
                  )}

                  {isFusedResult(result) && result.conflict && result.active_modalities.length > 1 && (
                    <ConflictPanel
                      conflict={result.conflict}
                      concordanceScore={result.explainability?.concordance_score}
                    />
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </section>
  );
};
