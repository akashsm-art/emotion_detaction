/**
 * Centralized API Client for FastAPI Emotion Recognition Backend.
 * Strictly communicates with backend endpoints; contains ZERO local ML inference.
 */

import type {
  HealthStatus,
  ModalityResult,
  FusedEmotionResult,
} from '../types/emotion';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

export class ApiError extends Error {
  status: number;
  data?: any;

  constructor(message: string, status: number, data?: any) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.data = data;
  }
}

/**
 * Fetch system health and model readiness.
 */
export async function getHealth(): Promise<HealthStatus> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/health`);
    if (!response.ok) {
      throw new ApiError(`Health check failed with status ${response.status}`, response.status);
    }
    return await response.json();
  } catch (error: any) {
    if (error instanceof ApiError) throw error;
    throw new ApiError(
      'Unable to connect to FastAPI backend. Ensure the server is running on ' + API_BASE_URL,
      0
    );
  }
}

/**
 * Predict emotion from natural language text.
 */
export async function predictText(text: string, signal?: AbortSignal): Promise<ModalityResult> {
  const trimmed = text.trim();
  if (!trimmed) {
    throw new ApiError('Text input cannot be empty.', 422);
  }

  const response = await fetch(`${API_BASE_URL}/api/predict/text`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: trimmed }),
    signal,
  });

  const data = await response.json();
  if (!response.ok) {
    throw new ApiError(data.detail || 'Text emotion classification failed', response.status, data);
  }

  return data;
}

/**
 * Predict emotion from audio file (WAV, MP3, OGG, FLAC).
 */
export async function predictAudio(file: File, signal?: AbortSignal): Promise<ModalityResult> {
  if (!file) {
    throw new ApiError('Please select an audio file.', 400);
  }

  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/api/predict/audio`, {
    method: 'POST',
    body: formData,
    signal,
  });

  const data = await response.json();
  if (!response.ok) {
    throw new ApiError(data.detail || 'Audio emotion classification failed', response.status, data);
  }

  return data;
}

/**
 * Predict emotion from facial image (honestly handles 503 if checkpoint missing).
 */
export async function predictVision(file: File, signal?: AbortSignal): Promise<ModalityResult> {
  if (!file) {
    throw new ApiError('Please select an image file.', 400);
  }

  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/api/predict/vision`, {
    method: 'POST',
    body: formData,
    signal,
  });

  const data = await response.json();
  if (response.status === 503) {
    // Return structured unavailable result rather than throwing raw crash
    return data;
  }

  if (!response.ok) {
    throw new ApiError(data.detail || 'Vision evaluation failed', response.status, data);
  }

  return data;
}

/**
 * Primary Multi-Modal Prediction endpoint.
 * Fuses any combination of Text, Audio, and Image.
 */
export async function predictMultimodal(
  params: {
    text?: string;
    audio?: File;
    image?: File;
  },
  signal?: AbortSignal
): Promise<FusedEmotionResult> {
  const formData = new FormData();
  let hasInput = false;

  if (params.text && params.text.trim()) {
    formData.append('text', params.text.trim());
    hasInput = true;
  }
  if (params.audio) {
    formData.append('audio', params.audio);
    hasInput = true;
  }
  if (params.image) {
    formData.append('image', params.image);
    hasInput = true;
  }

  if (!hasInput) {
    throw new ApiError('Please supply at least one modality (Text, Audio, or Image).', 400);
  }

  const response = await fetch(`${API_BASE_URL}/api/predict/multimodal`, {
    method: 'POST',
    body: formData,
    signal,
  });

  const data = await response.json();
  if (response.status === 503 && data && !data.available) {
    // All requested modalities were unavailable (e.g. only Vision requested)
    return data;
  }

  if (!response.ok) {
    throw new ApiError(data.detail || 'Multi-modal prediction failed', response.status, data);
  }

  return data;
}

/**
 * Real-Time Multimodal Fusion of pre-evaluated ModalityResult states.
 * Invokes the existing backend fusion engine without redundant re-inference.
 */
export async function evaluateFusion(
  results: Record<string, ModalityResult>,
  signal?: AbortSignal
): Promise<FusedEmotionResult> {
  const keys = Object.keys(results);
  if (keys.length === 0) {
    throw new ApiError('At least one modality result must be provided for fusion.', 400);
  }

  const response = await fetch(`${API_BASE_URL}/api/predict/fuse`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ results }),
    signal,
  });

  const data = await response.json();
  if (!response.ok) {
    throw new ApiError(data.detail || 'Real-time multi-modal fusion failed', response.status, data);
  }

  return data;
}

