import { describe, expect, it } from 'vitest';
import type { CoreEmotion, ModalityResult } from '../types/emotion';
import {
  appendBoundedAudio,
  selectModelAudioWindow,
  smoothAudioResult,
} from '../utils/audioWindow';

const emotions: CoreEmotion[] = [
  'angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise',
];

function result(happy: number, neutral: number): ModalityResult {
  const probabilities = Object.fromEntries(
    emotions.map((emotion) => [emotion, emotion === 'happy' ? happy : emotion === 'neutral' ? neutral : 0])
  ) as Record<CoreEmotion, number>;
  return {
    modality: 'audio',
    available: true,
    status: 'ok',
    emotion: happy > neutral ? 'happy' : 'neutral',
    confidence: Math.max(happy, neutral),
    probabilities,
    supported_emotions: ['neutral', 'angry', 'happy', 'sad'],
    valence: happy - neutral,
    arousal: 0.25,
    metadata: {},
  };
}

describe('live audio windowing', () => {
  it('waits for a full model window and selects exactly the newest samples', () => {
    const short = [new Float32Array([0, 1, 2, 3, 4, 5, 6])];
    expect(selectModelAudioWindow(short, 4).window).toBeNull();

    const selected = selectModelAudioWindow(
      [new Float32Array([0, 1, 2, 3, 4]), new Float32Array([5, 6, 7, 8, 9])],
      4
    );
    expect(Array.from(selected.window ?? [])).toEqual([2, 3, 4, 5, 6, 7, 8, 9]);
    expect(selected.window?.length).toBe(8);
    expect(selected.rollingChunks).toHaveLength(1);
  });

  it('bounds queued audio to avoid stale, unbounded windows during slow requests', () => {
    const bounded = appendBoundedAudio(
      [new Float32Array([1, 2]), new Float32Array([3, 4])],
      new Float32Array([5, 6, 7, 8]),
      5
    );
    expect(Array.from(bounded.flatMap((chunk) => Array.from(chunk)))).toEqual([4, 5, 6, 7, 8]);
  });
});

describe('live audio temporal smoothing', () => {
  it('averages recent probabilities without forcing a class', () => {
    const first = smoothAudioResult([], result(0.2, 0.8));
    const second = smoothAudioResult(first.history, result(0.7, 0.3));
    expect(second.result.emotion).toBe('neutral');

    const third = smoothAudioResult(second.history, result(0.8, 0.2));
    expect(third.result.emotion).toBe('happy');
    expect(third.result.probabilities.happy).toBeCloseTo(0.5667, 3);
    expect(third.result.confidence).toBe(third.result.probabilities.happy);
    expect(Object.values(third.result.probabilities).reduce((sum, p) => sum + p, 0)).toBeCloseTo(1);
    expect(third.result.metadata.temporal_smoothing_windows).toBe(3);
  });

  it('passes unavailable results through and clears stale smoothing history', () => {
    const unavailable: ModalityResult = {
      ...result(0, 0),
      available: false,
      status: 'SILENT_OR_NEAR_SILENT_AUDIO',
      emotion: 'none',
      confidence: 0,
      probabilities: Object.fromEntries(emotions.map((emotion) => [emotion, 0])) as Record<CoreEmotion, number>,
    };

    const smoothed = smoothAudioResult([result(1, 0)], unavailable);
    expect(smoothed.result).toBe(unavailable);
    expect(smoothed.history).toEqual([]);
  });
});
