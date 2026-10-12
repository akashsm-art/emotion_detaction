import { useState, useEffect, useCallback, useMemo } from 'react';
import type { FC } from 'react';
import type { HealthStatus, ModalityResult } from '../types/emotion';
import { getHealth } from '../services/api';

import { Header } from '../components/Header';
import { ModalityStatusCard } from '../components/ModalityStatusCard';
import { MasterLiveControls } from '../components/MasterLiveControls';
import { LiveTextPanel } from '../components/LiveTextPanel';
import { LiveAudioPanel } from '../components/LiveAudioPanel';
import { LiveWebcamPanel } from '../components/LiveWebcamPanel';
import { UnifiedEmotionOutput } from '../components/UnifiedEmotionOutput';
import { ProbabilityChart } from '../components/ProbabilityChart';
import { ModalityContributionPanel } from '../components/ModalityContributionPanel';
import { ConflictPanel } from '../components/ConflictPanel';
import { ManualAnalysisDrawer } from '../components/ManualAnalysisDrawer';

import { useLiveText } from '../hooks/useLiveText';
import { useLiveAudio } from '../hooks/useLiveAudio';
import { useLiveWebcam } from '../hooks/useLiveWebcam';
import { useLiveFusionCoordinator } from '../hooks/useLiveFusionCoordinator';

export const Home: FC = () => {
  // Backend System Health
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [loadingHealth, setLoadingHealth] = useState<boolean>(true);

  // Active Modalities Set
  const [activeModalities, setActiveModalities] = useState<Set<'text' | 'audio' | 'vision'>>(
    new Set(['text'])
  );

  // Health fetch
  const fetchHealth = useCallback(async () => {
    setLoadingHealth(true);
    try {
      const data = await getHealth();
      setHealth(data);
    } catch (err: any) {
      console.warn('Backend health check error:', err);
      setHealth(null);
    } finally {
      setLoadingHealth(false);
    }
  }, []);

  useEffect(() => {
    fetchHealth();
  }, [fetchHealth]);

  // Central Live Fusion Coordinator
  const {
    fusedResult,
    isFusing,
    updateModalityResult,
    clearModalityResult,
    resetAll: resetFusion,
  } = useLiveFusionCoordinator(activeModalities);


  // Modality Result Handlers
  const handleTextResult = useCallback(
    (res: ModalityResult) => {
      updateModalityResult('text', res);
    },
    [updateModalityResult]
  );

  const handleAudioResult = useCallback(
    (res: ModalityResult) => {
      updateModalityResult('audio', res);
    },
    [updateModalityResult]
  );

  const handleVisionResult = useCallback(
    (res: ModalityResult) => {
      // If face detection found faces, send to fusion, else exclude
      if (res.metadata?.faces_detected > 0) {
        updateModalityResult('vision', res);
      } else {
        clearModalityResult('vision');
      }
    },
    [updateModalityResult, clearModalityResult]
  );

  // Live Text Hook
  const {
    text,
    setText,
    isAnalyzing: isTextAnalyzing,
    textResult,
    error: textError,
    isActive: isTextActive,
    setIsActive: setIsTextActive,
    clear: clearText,
  } = useLiveText(
    'I am thrilled with the breakthrough progress we achieved today!',
    handleTextResult
  );

  // Live Audio Hook
  const {
    status: audioStatus,
    error: audioError,
    audioResult,
    volume: audioVolume,
    analyserNode,
    start: startAudio,
    pause: pauseAudio,
    resume: resumeAudio,
    stop: stopAudio,
    resetResult: resetAudioResult,
  } = useLiveAudio(handleAudioResult);

  // Live Webcam Hook
  const {
    status: webcamStatus,
    error: webcamError,
    visionResult,
    videoRef,
    faceDetected,
    facesCount,
    cameraFps,
    inferenceRateHz,
    start: startWebcam,
    stop: stopWebcam,
    resetResult: resetVisionResult,
  } = useLiveWebcam(handleVisionResult);

  // Synchronize Text Active State with activeModalities
  const handleToggleTextActive = useCallback(() => {
    const next = !isTextActive;
    setIsTextActive(next);
    setActiveModalities((mods) => {
      const updated = new Set(mods);
      if (next) updated.add('text');
      else {
        updated.delete('text');
        clearModalityResult('text');
      }
      return updated;
    });
  }, [clearModalityResult, isTextActive, setIsTextActive]);


  // Handle Audio Start / Stop synchronization
  const handleStartAudio = useCallback(async () => {
    setActiveModalities((mods) => new Set(mods).add('audio'));
    await startAudio();
  }, [startAudio]);

  const handleStopAudio = useCallback(() => {
    stopAudio();
    clearModalityResult('audio');
    setActiveModalities((mods) => {
      const updated = new Set(mods);
      updated.delete('audio');
      return updated;
    });
  }, [stopAudio, clearModalityResult]);

  // Handle Webcam Start / Stop synchronization
  const handleStartWebcam = useCallback(async () => {
    setActiveModalities((mods) => new Set(mods).add('vision'));
    await startWebcam();
  }, [startWebcam]);

  const handleStopWebcam = useCallback(() => {
    stopWebcam();
    clearModalityResult('vision');
    setActiveModalities((mods) => {
      const updated = new Set(mods);
      updated.delete('vision');
      return updated;
    });
  }, [stopWebcam, clearModalityResult]);

  // Master START ALL Analysis
  const handleStartAll = useCallback(async () => {
    // 1. Activate Text
    setIsTextActive(true);

    // 2. Mark all modalities as target active
    setActiveModalities(new Set(['text', 'audio', 'vision']));

    // 3. Request and start audio & webcam concurrently
    // Note: Promise.allSettled guarantees a denial in one does NOT crash the other
    await Promise.allSettled([startAudio(), startWebcam()]);
  }, [setIsTextActive, startAudio, startWebcam]);

  // Master STOP ALL Analysis
  const handleStopAll = useCallback(() => {
    stopAudio();
    stopWebcam();
    setIsTextActive(false);
    clearModalityResult('audio');
    clearModalityResult('vision');
    clearModalityResult('text');
    setActiveModalities(new Set());
  }, [stopAudio, stopWebcam, setIsTextActive, clearModalityResult]);

  // Master Reset
  const handleReset = useCallback(() => {
    stopAudio();
    stopWebcam();
    clearText();
    resetAudioResult();
    resetVisionResult();
    resetFusion();
    setActiveModalities(new Set(['text']));
    setIsTextActive(true);
  }, [
    stopAudio,
    stopWebcam,
    clearText,
    resetAudioResult,
    resetVisionResult,
    resetFusion,
    setIsTextActive,
  ]);

  const isAudioListening = audioStatus === 'recording' || audioStatus === 'analyzing';
  const isWebcamStreaming = webcamStatus === 'streaming' || webcamStatus === 'analyzing';
  const isLiveActive = isTextActive || isAudioListening || isWebcamStreaming;

  const activeCount = useMemo(() => {
    let count = 0;
    if (isTextActive) count++;
    if (isAudioListening) count++;
    if (isWebcamStreaming) count++;
    return count;
  }, [isTextActive, isAudioListening, isWebcamStreaming]);

  return (
    <div className="app-layout">
      {/* Top Application Header */}
      <Header
        health={health}
        loadingHealth={loadingHealth}
        onRefreshHealth={fetchHealth}
      />

      {/* Main Emotion Intelligence Workspace */}
      <main className="main-content">
        {/* Master Control Bar */}
        <MasterLiveControls
          isLiveActive={isLiveActive}
          activeCount={activeCount}
          onStartAll={handleStartAll}
          onStopAll={handleStopAll}
          onReset={handleReset}
        />

        {/* System Model Status Overview */}
        <section className="controls-section mb-4">
          <ModalityStatusCard health={health} loading={loadingHealth} />
        </section>

        {/* 3-Column Live Modalities Grid */}
        <section className="live-inputs-row" aria-label="Live Emotion Modality Streams">
          {/* 1. Live Text */}
          <LiveTextPanel
            text={text}
            setText={setText}
            isAnalyzing={isTextAnalyzing}
            textResult={textResult}
            isActive={isTextActive}
            onToggleActive={handleToggleTextActive}
            onClear={clearText}
            error={textError}
          />

          {/* 2. Live Audio Microphone */}
          <LiveAudioPanel
            status={audioStatus}
            error={audioError}
            audioResult={audioResult}
            volume={audioVolume}
            analyserNode={analyserNode}
            onStart={handleStartAudio}
            onPause={pauseAudio}
            onResume={resumeAudio}
            onStop={handleStopAudio}
          />

          {/* 3. Live Webcam */}
          <LiveWebcamPanel
            status={webcamStatus}
            error={webcamError}
            visionResult={visionResult}
            videoRef={videoRef}
            faceDetected={faceDetected}
            facesCount={facesCount}
            cameraFps={cameraFps}
            inferenceRateHz={inferenceRateHz}
            onStart={handleStartWebcam}
            onStop={handleStopWebcam}
          />
        </section>

        {/* ONE DOMINANT UNIFIED EMOTION OUTPUT */}
        <section className="unified-output-section" aria-label="Dominant Unified Emotional State">
          <UnifiedEmotionOutput
            fusedResult={fusedResult}
            textResult={textResult}
            audioResult={audioResult}
            visionResult={visionResult}
            isTextActive={isTextActive}
            isAudioActive={isAudioListening}
            isVisionActive={isWebcamStreaming}
            isFusing={isFusing}
          />
        </section>

        {/* Emotion Analytics & Explainability (Visible when a prediction is active) */}
        {fusedResult && fusedResult.available && (
          <section className="analytics-grid" aria-label="Emotion Analytics and Explainability">
            {/* 7-Class Probability Distribution */}
            <div className="analytics-card">
              <ProbabilityChart
                probabilities={fusedResult.probabilities}
                supportedEmotions={
                  fusedResult.fusion_mode === 'single_modality' &&
                  fusedResult.active_modalities[0] &&
                  fusedResult.modality_results[fusedResult.active_modalities[0]]?.supported_emotions
                    ? fusedResult.modality_results[fusedResult.active_modalities[0]].supported_emotions
                    : undefined
                }
              />
            </div>

            {/* Explainability & Concordance */}
            <div className="analytics-card">
              {fusedResult.active_modalities.length > 0 && (
                <ModalityContributionPanel result={fusedResult} />
              )}

              {fusedResult.conflict && fusedResult.active_modalities.length > 1 && (
                <div className="mt-3">
                  <ConflictPanel
                    conflict={fusedResult.conflict}
                    concordanceScore={fusedResult.explainability?.concordance_score}
                  />
                </div>
              )}
            </div>
          </section>
        )}

        {/* Preserved Manual & Offline Analysis Mode */}
        <ManualAnalysisDrawer />
      </main>


      {/* Global Footer */}
      <footer className="footer-container">
        <p className="footer-text">
          Emotion Intelligence Platform &bull; Real-Time Multi-Modal Inference &bull;
          Confidence-Weighted Decision Fusion &bull; RoBERTa + CNN MFCC + AlexNet &bull; Ekman-7 Ontology
        </p>
      </footer>
    </div>
  );
};
