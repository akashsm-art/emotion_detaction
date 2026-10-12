import type { ModalityResult, CoreEmotion } from '../types/emotion';
import { mergeAudioBuffers } from './wavEncoder';

export const MODEL_AUDIO_WINDOW_SECONDS = 2;
export const LIVE_AUDIO_INTERVAL_MS = 1500;
export const MAX_LIVE_BUFFER_SECONDS = 4;

export function appendBoundedAudio(
  chunks: Float32Array[],
  nextChunk: Float32Array,
  maxSamples: number
): Float32Array[] {
  const bounded = [...chunks, nextChunk];
  let excess = bounded.reduce((total, chunk) => total + chunk.length, 0) - maxSamples;

  while (excess > 0 && bounded.length > 0) {
    if (excess >= bounded[0].length) {
      excess -= bounded[0].length;
      bounded.shift();
    } else {
      bounded[0] = bounded[0].slice(excess);
      excess = 0;
    }
  }

  return bounded;
}

export function selectModelAudioWindow(
  chunks: Float32Array[],
  sampleRate: number,
  durationSeconds = MODEL_AUDIO_WINDOW_SECONDS
): { window: Float32Array | null; rollingChunks: Float32Array[] } {
  const merged = mergeAudioBuffers(chunks);
  const windowSamples = Math.round(sampleRate * durationSeconds);
  if (merged.length < windowSamples) {
    return { window: null, rollingChunks: chunks };
  }

  const window = merged.slice(-windowSamples);
  return { window, rollingChunks: [window] };
}

export function smoothAudioResult(
  history: ModalityResult[],
  next: ModalityResult,
  maxResults = 3
): { result: ModalityResult; history: ModalityResult[] } {
  if (!next.available || next.status !== 'ok') {
    return { result: next, history: [] };
  }

  const recent = [...history, next].slice(-maxResults);
  const emotionKeys = Object.keys(next.probabilities) as CoreEmotion[];
  const averaged = Object.fromEntries(
    emotionKeys.map((emotion) => [
      emotion,
      recent.reduce((sum, item) => sum + Math.max(0, item.probabilities[emotion] || 0), 0) /
        recent.length,
    ])
  ) as Record<CoreEmotion, number>;

  const total = Object.values(averaged).reduce((sum, value) => sum + value, 0);
  if (total > 0) {
    for (const emotion of emotionKeys) averaged[emotion] /= total;
  }

  const supported = next.supported_emotions.filter((emotion) => emotion in averaged);
  const candidates = supported.length > 0 ? supported : emotionKeys;
  const emotion = candidates.reduce((best, current) =>
    averaged[current as CoreEmotion] > averaged[best as CoreEmotion] ? current : best
  );

  return {
    result: {
      ...next,
      emotion,
      confidence: averaged[emotion as CoreEmotion],
      probabilities: averaged,
      valence: recent.reduce((sum, item) => sum + item.valence, 0) / recent.length,
      arousal: recent.reduce((sum, item) => sum + item.arousal, 0) / recent.length,
      metadata: {
        ...next.metadata,
        temporal_smoothing_windows: recent.length,
      },
    },
    history: recent,
  };
}
