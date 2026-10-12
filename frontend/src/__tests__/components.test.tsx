// @vitest-environment jsdom
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { Header } from '../components/Header';
import { ModalitySelector } from '../components/ModalitySelector';
import { TextInput } from '../components/TextInput';
import { ImageUploader } from '../components/ImageUploader';
import { PredictionResult } from '../components/PredictionResult';
import { ProbabilityChart } from '../components/ProbabilityChart';
import { ModalityContributionPanel } from '../components/ModalityContributionPanel';
import { ConflictPanel } from '../components/ConflictPanel';
import { UnavailableModalityBanner } from '../components/ModalityStatusCard';
import { LoadingState } from '../components/LoadingState';
import { MasterLiveControls } from '../components/MasterLiveControls';
import { LiveTextPanel } from '../components/LiveTextPanel';
import { LiveAudioPanel } from '../components/LiveAudioPanel';
import { LiveWebcamPanel } from '../components/LiveWebcamPanel';
import { UnifiedEmotionOutput } from '../components/UnifiedEmotionOutput';
import type { FusedEmotionResult, HealthStatus, ModalityResult } from '../types/emotion';


describe('Frontend Component Tests', () => {
  it('Header renders product name and status pill', () => {
    const mockHealth: HealthStatus = {
      status: 'ok',
      service: 'multi-modal-emotion-api',
      version: '1.0.0',
      device: 'cpu',
      models: {
        text: { operational: true },
        audio: { operational: true },
        vision: { operational: false },
      },
    };
    render(
      <Header
        health={mockHealth}
        loadingHealth={false}
        onRefreshHealth={vi.fn()}
      />
    );
    expect(screen.getByText('Emotion Intelligence')).toBeDefined();
    expect(screen.getByText('Multi-Modal Emotion Recognition Platform')).toBeDefined();
    expect(screen.getByText('System: OK')).toBeDefined();
  });

  it('ModalitySelector allows toggling modalities', () => {
    const toggleMock = vi.fn();
    const selected = new Set(['text', 'audio']);
    render(
      <ModalitySelector
        selectedModalities={selected}
        onToggleModality={toggleMock}
      />
    );

    expect(screen.getByText('Multi-Modal (2 active)')).toBeDefined();
    const visionBtn = screen.getByText('Vision (AlexNet)');
    fireEvent.click(visionBtn);
    expect(toggleMock).toHaveBeenCalledWith('vision');
  });

  it('TextInput renders char count and triggers change', () => {
    const changeMock = vi.fn();
    render(
      <TextInput
        text="Hello emotional world"
        onChange={changeMock}
      />
    );
    expect(screen.getByText(/21 \/ 10,000/)).toBeDefined();
    const textarea = screen.getByPlaceholderText(/Enter text describing an emotional state/);
    fireEvent.change(textarea, { target: { value: 'New text' } });
    expect(changeMock).toHaveBeenCalledWith('New text');
  });

  it('ImageUploader renders trained checkpoint and pipeline info', () => {
    render(
      <ImageUploader
        file={null}
        onFileSelect={vi.fn()}
      />
    );
    expect(screen.getByText(/Vision Emotion Pipeline \(Trained AlexNet\)/)).toBeDefined();
    expect(screen.getByText(/model_image_new50.pth/)).toBeDefined();
    expect(screen.getByText(/trained weights/i)).toBeDefined();
  });

  it('PredictionResult correctly labels Model Confidence and circumplex coordinates', () => {
    const mockFused: FusedEmotionResult = {
      available: true,
      status: 'OPERATIONAL',
      fusion_mode: 'multimodal',
      emotion: 'sad',
      confidence: 0.9928,
      probabilities: {
        angry: 0.001,
        disgust: 0.0,
        fear: 0.002,
        happy: 0.0008,
        neutral: 0.0026,
        sad: 0.9928,
        surprise: 0.0,
      },
      valence: -0.6,
      arousal: -0.5,
      active_modalities: ['text', 'audio'],
      unavailable_modalities: ['vision'],
      modality_results: {} as any,
      conflict: {
        has_conflict: false,
        conflict_score: 0.05,
        divergent_pairs: [],
        summary: 'High concordance between text and audio signals.',
      },
      explainability: {
        primary_modality: 'text',
        concordance_score: 0.95,
        modality_weights: { text: 0.65, audio: 0.35 },
        rationale: 'Text model exhibited high certainty.',
        key_features: {},
      },
      timestamp: '2026-09-26T12:00:00Z',
    };

    render(<PredictionResult result={mockFused} />);
    expect(screen.getByText('SAD')).toBeDefined();
    expect(screen.getByText('99.28%')).toBeDefined();
    expect(screen.getByText('Model Confidence')).toBeDefined();
    expect(screen.getByText(/Neural model certainty score/)).toBeDefined();
  });

  it('ProbabilityChart renders Ekman classes with true backend percentages without faking unsupported ones', () => {
    const probs = {
      angry: 0.852,
      disgust: 0.0,
      fear: 0.0,
      happy: 0.048,
      neutral: 0.08,
      sad: 0.02,
      surprise: 0.0,
    };

    render(
      <ProbabilityChart
        probabilities={probs}
        supportedEmotions={['angry', 'happy', 'neutral', 'sad']}
      />
    );

    expect(screen.getByText('85.20%')).toBeDefined();
    expect(screen.getAllByText('Unsupported by model').length).toBe(3); // disgust, fear, surprise
  });

  it('ModalityContributionPanel renders active modalities and exact weights', () => {
    const mockFused: FusedEmotionResult = {
      available: true,
      status: 'OPERATIONAL',
      fusion_mode: 'multimodal',
      emotion: 'happy',
      confidence: 0.88,
      probabilities: {} as any,
      valence: 0.7,
      arousal: 0.5,
      active_modalities: ['text', 'audio'],
      unavailable_modalities: ['vision'],
      modality_results: {} as any,
      conflict: {} as any,
      explainability: {
        primary_modality: 'audio',
        concordance_score: 0.92,
        modality_weights: { text: 0.498, audio: 0.502 },
        rationale: 'Balanced acoustic and linguistic fusion.',
        key_features: {},
      },
      timestamp: '2026-09-26T12:00:00Z',
    };

    render(<ModalityContributionPanel result={mockFused} />);
    expect(screen.getByText('Active: text, audio')).toBeDefined();
    expect(screen.getByText('Unavailable: vision')).toBeDefined();
    expect(screen.getByText('49.8%')).toBeDefined();
    expect(screen.getByText('50.2%')).toBeDefined();
  });

  it('ConflictPanel renders high concordance and divergent pairs non-judgmentally', () => {
    const conflictData = {
      has_conflict: true,
      conflict_score: 0.42,
      divergent_pairs: [
        {
          modality_1: 'text',
          emotion_1: 'sad',
          confidence_1: 0.95,
          modality_2: 'audio',
          emotion_2: 'angry',
          confidence_2: 0.88,
          pairwise_divergence: 0.42,
        },
      ],
      summary: 'Linguistic sentiment indicates sadness while acoustic prosody indicates anger.',
    };

    render(<ConflictPanel conflict={conflictData} concordanceScore={0.58} />);
    expect(screen.getByText('Cross-Modal Disagreement Detected')).toBeDefined();
    expect(screen.getByText(/Agreement: 58.0%/)).toBeDefined();
    expect(screen.getByText(/JS Divergence: 0.420/)).toBeDefined();
    expect(screen.getByText(/not human veracity or deception/)).toBeDefined();
  });

  it('UnavailableModalityBanner clearly explains missing checkpoint without UI crash', () => {
    render(
      <UnavailableModalityBanner
        unavailableModalities={['vision']}
        activeModalities={['text', 'audio']}
      />
    );
    expect(screen.getByText('VISION UNAVAILABLE')).toBeDefined();
    expect(screen.getByText(/Active modalities \(.*\) were still fully processed/)).toBeDefined();
  });

  it('LoadingState displays animated spinner and phase message', () => {
    render(<LoadingState message="Running multimodal fusion..." />);
    expect(screen.getByText('Running multimodal fusion...')).toBeDefined();
  });

  it('MasterLiveControls renders Start Live Analysis button and triggers callback', () => {
    const startMock = vi.fn();
    const stopMock = vi.fn();
    const resetMock = vi.fn();

    const { rerender } = render(
      <MasterLiveControls
        isLiveActive={false}
        activeCount={0}
        onStartAll={startMock}
        onStopAll={stopMock}
        onReset={resetMock}
      />
    );

    expect(screen.getByText('LIVE ANALYSIS READY')).toBeDefined();
    const startBtn = screen.getByText('START LIVE ANALYSIS');
    fireEvent.click(startBtn);
    expect(startMock).toHaveBeenCalledTimes(1);

    // Re-render in active state
    rerender(
      <MasterLiveControls
        isLiveActive={true}
        activeCount={3}
        onStartAll={startMock}
        onStopAll={stopMock}
        onReset={resetMock}
      />
    );
    expect(screen.getByText(/LIVE ANALYSIS ACTIVE \(3 Modalities Online\)/)).toBeDefined();
    const stopBtn = screen.getByText('STOP LIVE ANALYSIS');
    fireEvent.click(stopBtn);
    expect(stopMock).toHaveBeenCalledTimes(1);
  });

  it('LiveTextPanel renders textarea, analyzing badge, and emotion badge', () => {
    const setTextMock = vi.fn();
    const toggleMock = vi.fn();
    const clearMock = vi.fn();

    const mockResult: ModalityResult = {
      modality: 'text',
      available: true,
      status: 'ok',
      emotion: 'happy',
      confidence: 0.94,
      probabilities: {
        angry: 0.01,
        disgust: 0.01,
        fear: 0.01,
        happy: 0.94,
        neutral: 0.01,
        sad: 0.01,
        surprise: 0.01,
      },
      supported_emotions: ['happy'],
      valence: 0.8,
      arousal: 0.6,
      metadata: {},
    };

    render(
      <LiveTextPanel
        text="I am so glad!"
        setText={setTextMock}
        isAnalyzing={true}
        textResult={mockResult}
        isActive={true}
        onToggleActive={toggleMock}
        onClear={clearMock}
      />
    );

    expect(screen.getByText('LIVE TEXT')).toBeDefined();
    expect(screen.getByText(/Analyzing.../)).toBeDefined();
    expect(screen.getByText('HAPPY')).toBeDefined();
    expect(screen.getByText('94.0%')).toBeDefined();
    expect(screen.getByText('RoBERTa')).toBeDefined();
  });

  it('LiveAudioPanel displays listening status and controls', () => {
    const startMock = vi.fn();
    const pauseMock = vi.fn();
    const resumeMock = vi.fn();
    const stopMock = vi.fn();

    const { rerender } = render(
      <LiveAudioPanel
        status="idle"
        error={null}
        audioResult={null}
        volume={0}
        analyserNode={null}
        onStart={startMock}
        onPause={pauseMock}
        onResume={resumeMock}
        onStop={stopMock}
      />
    );

    expect(screen.getByText('LIVE AUDIO')).toBeDefined();
    expect(screen.getByText('Start Microphone')).toBeDefined();

    rerender(
      <LiveAudioPanel
        status="recording"
        error={null}
        audioResult={null}
        volume={0.4}
        analyserNode={null}
        onStart={startMock}
        onPause={pauseMock}
        onResume={resumeMock}
        onStop={stopMock}
      />
    );
    expect(screen.getByText('Listening')).toBeDefined();
    expect(screen.getByText('Pause')).toBeDefined();
    expect(screen.getByText('Stop')).toBeDefined();
  });

  it('LiveWebcamPanel renders HUD overlay and handles no-face state honestly', () => {
    const startMock = vi.fn();
    const stopMock = vi.fn();
    const videoRef = { current: null };

    render(
      <LiveWebcamPanel
        status="streaming"
        error={null}
        visionResult={null}
        videoRef={videoRef}
        faceDetected={false}
        facesCount={0}
        cameraFps={30}
        inferenceRateHz={2.0}
        onStart={startMock}
        onStop={stopMock}
      />
    );

    expect(screen.getByText('LIVE WEBCAM')).toBeDefined();
    expect(screen.getAllByText('No Face Detected').length).toBeGreaterThan(0);
    expect(screen.getByText('Stop Camera')).toBeDefined();
  });

  it('UnifiedEmotionOutput renders dominant emotional state, signals, and progress', () => {
    const mockFused: FusedEmotionResult = {
      available: true,
      status: 'ok',
      fusion_mode: 'multimodal',
      emotion: 'happy',
      confidence: 0.824,
      probabilities: {
        angry: 0.02,
        disgust: 0.01,
        fear: 0.02,
        happy: 0.824,
        neutral: 0.05,
        sad: 0.03,
        surprise: 0.046,
      },
      valence: 0.75,
      arousal: 0.60,
      active_modalities: ['text', 'audio', 'vision'],
      unavailable_modalities: [],
      modality_results: {},
      conflict: { has_conflict: false, conflict_score: 0.1, divergent_pairs: [], summary: 'Concordant' },
      explainability: {
        primary_modality: 'text',
        concordance_score: 0.9,
        modality_weights: { text: 0.4, audio: 0.41, vision: 0.19 },
        rationale: 'Concordant',
        key_features: {},
      },
      timestamp: '2026-09-27T00:00:00Z',
    };

    const mockText: ModalityResult = {
      modality: 'text',
      available: true,
      status: 'ok',
      emotion: 'happy',
      confidence: 0.91,
      probabilities: { happy: 0.91 } as any,
      supported_emotions: ['happy'],
      valence: 0.8,
      arousal: 0.6,
      metadata: {},
    };

    const mockAudio: ModalityResult = {
      modality: 'audio',
      available: true,
      status: 'ok',
      emotion: 'happy',
      confidence: 0.78,
      probabilities: { happy: 0.78 } as any,
      supported_emotions: ['happy'],
      valence: 0.7,
      arousal: 0.5,
      metadata: {},
    };

    const mockVision: ModalityResult = {
      modality: 'vision',
      available: true,
      status: 'ok',
      emotion: 'neutral',
      confidence: 0.54,
      probabilities: { neutral: 0.54 } as any,
      supported_emotions: ['neutral'],
      valence: 0.0,
      arousal: 0.0,
      metadata: {},
    };

    render(
      <UnifiedEmotionOutput
        fusedResult={mockFused}
        textResult={mockText}
        audioResult={mockAudio}
        visionResult={mockVision}
        isTextActive={true}
        isAudioActive={true}
        isVisionActive={true}
      />
    );

    expect(screen.getByText('CURRENT EMOTIONAL STATE')).toBeDefined();
    expect(screen.getAllByText('HAPPY').length).toBeGreaterThan(0);
    expect(screen.getAllByText('82.4%').length).toBeGreaterThan(0);


    expect(screen.getByText('Positive, high-arousal emotional state')).toBeDefined();
    expect(screen.getByText('91%')).toBeDefined(); // Text confidence
    expect(screen.getByText('78%')).toBeDefined(); // Audio confidence
    expect(screen.getByText('54%')).toBeDefined(); // Vision confidence
    expect(screen.getByText('Fusion Confidence')).toBeDefined();
  });
});

