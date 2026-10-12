import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  getHealth,
  predictText,
  predictAudio,
  predictVision,
  predictMultimodal,
  evaluateFusion,
} from '../services/api';


describe('API Service Layer (services/api.ts)', () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it('predictText rejects empty or whitespace-only text with 422 ApiError', async () => {
    await expect(predictText('   ')).rejects.toThrow('Text input cannot be empty.');
  });

  it('predictText successfully calls backend endpoint and parses ModalityResult', async () => {
    const mockResponse = {
      modality: 'text',
      available: true,
      status: 'OPERATIONAL',
      emotion: 'joy',
      confidence: 0.985,
      probabilities: {
        angry: 0.001,
        disgust: 0.001,
        fear: 0.001,
        happy: 0.985,
        neutral: 0.005,
        sad: 0.005,
        surprise: 0.002,
      },
      supported_emotions: ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'],
      valence: 0.8,
      arousal: 0.6,
      metadata: {},
    };

    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => mockResponse,
    });

    const result = await predictText('I am so joyful today!');
    expect(result.emotion).toBe('joy');
    expect(result.confidence).toBe(0.985);
    expect(globalThis.fetch).toHaveBeenCalledWith(
      'http://127.0.0.1:8000/api/predict/text',
      expect.objectContaining({
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      })
    );
  });

  it('predictAudio sends FormData to /api/predict/audio', async () => {
    const mockAudioResponse = {
      modality: 'audio',
      available: true,
      status: 'OPERATIONAL',
      emotion: 'angry',
      confidence: 0.912,
      probabilities: {
        angry: 0.912,
        disgust: 0.0,
        fear: 0.0,
        happy: 0.035,
        neutral: 0.041,
        sad: 0.012,
        surprise: 0.0,
      },
      supported_emotions: ['neutral', 'angry', 'happy', 'sad'],
      valence: -0.6,
      arousal: 0.8,
      metadata: {},
    };

    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => mockAudioResponse,
    });

    const fakeFile = new File(['fake-audio-content'], 'sample.wav', { type: 'audio/wav' });
    const result = await predictAudio(fakeFile);
    expect(result.emotion).toBe('angry');
    expect(result.probabilities.disgust).toBe(0.0);
    expect(result.probabilities.fear).toBe(0.0);
    expect(result.probabilities.surprise).toBe(0.0);
  });

  it('predictVision gracefully returns structured 503 unavailable payload without uncaught crash', async () => {
    const mockUnavailableResponse = {
      modality: 'vision',
      available: false,
      status: 'VISION INFERENCE BLOCKED — TRAINED CHECKPOINT MISSING',
      emotion: 'none',
      confidence: 0.0,
      probabilities: {
        angry: 0.0,
        disgust: 0.0,
        fear: 0.0,
        happy: 0.0,
        neutral: 0.0,
        sad: 0.0,
        surprise: 0.0,
      },
      supported_emotions: [],
      valence: 0.0,
      arousal: 0.0,
      metadata: { reason: 'Checkpoint missing' },
      detail: 'Vision emotion inference unavailable because the trained checkpoint is missing.',
    };

    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 503,
      json: async () => mockUnavailableResponse,
    });

    const fakeImage = new File(['fake-img-content'], 'face.jpg', { type: 'image/jpeg' });
    const result = await predictVision(fakeImage);
    expect(result.available).toBe(false);
    expect(result.status).toContain('TRAINED CHECKPOINT MISSING');
  });

  it('predictMultimodal requires at least one modality input', async () => {
    await expect(predictMultimodal({})).rejects.toThrow(
      'Please supply at least one modality (Text, Audio, or Image).'
    );
  });

  it('getHealth returns health status object', async () => {
    const mockHealth = {
      status: 'degraded',
      service: 'multi-modal-emotion-recognition-api',
      version: '1.0.0',
      device: 'cpu',
      models: {
        text: { operational: true, model_name: 'distilroberta' },
        audio: { operational: true, model_name: 'cnn_2d' },
        vision: { operational: false, reason: 'checkpoint missing' },
      },
    };

    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => mockHealth,
    });

    const health = await getHealth();
    expect(health.status).toBe('degraded');
    expect(health.models.text.operational).toBe(true);
    expect(health.models.vision.operational).toBe(false);
  });

  it('evaluateFusion rejects empty modality results dictionary with 400 ApiError', async () => {
    await expect(evaluateFusion({})).rejects.toThrow(
      'At least one modality result must be provided for fusion.'
    );
  });

  it('evaluateFusion calls /api/predict/fuse and returns FusedEmotionResult', async () => {
    const mockFused = {
      available: true,
      status: 'ok',
      fusion_mode: 'multimodal',
      emotion: 'happy',
      confidence: 0.88,
      probabilities: {
        angry: 0.01,
        disgust: 0.01,
        fear: 0.01,
        happy: 0.88,
        neutral: 0.05,
        sad: 0.02,
        surprise: 0.02,
      },
      valence: 0.8,
      arousal: 0.6,
      active_modalities: ['text', 'audio'],
      unavailable_modalities: [],
      modality_results: {},
      conflict: { has_conflict: false, conflict_score: 0.0, divergent_pairs: [], summary: 'Concordant' },
      explainability: {
        primary_modality: 'text',
        concordance_score: 0.95,
        modality_weights: { text: 0.55, audio: 0.45 },
        rationale: 'High agreement',
        key_features: {},
      },
      timestamp: '2026-09-27T00:00:00Z',
    };

    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => mockFused,
    });

    const mockTextModality: any = {
      modality: 'text',
      available: true,
      status: 'ok',
      emotion: 'happy',
      confidence: 0.9,
      probabilities: { happy: 0.9 },
    };

    const result = await evaluateFusion({ text: mockTextModality });
    expect(result.emotion).toBe('happy');
    expect(result.fusion_mode).toBe('multimodal');
    expect(globalThis.fetch).toHaveBeenCalledWith(
      'http://127.0.0.1:8000/api/predict/fuse',
      expect.objectContaining({
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      })
    );
  });
});

