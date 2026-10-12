import { useState, useRef, useCallback, useEffect } from 'react';
import type { ModalityResult } from '../types/emotion';
import { predictAudio } from '../services/api';
import { encodeWAV } from '../utils/wavEncoder';
import {
  appendBoundedAudio,
  LIVE_AUDIO_INTERVAL_MS,
  MAX_LIVE_BUFFER_SECONDS,
  MODEL_AUDIO_WINDOW_SECONDS,
  selectModelAudioWindow,
  smoothAudioResult,
} from '../utils/audioWindow';

export type LiveAudioStatus = 'idle' | 'requesting' | 'recording' | 'analyzing' | 'paused' | 'error';

export interface UseLiveAudioReturn {
  status: LiveAudioStatus;
  error: string | null;
  audioResult: ModalityResult | null;
  volume: number; // 0 to 1 for visual meter
  analyserNode: AnalyserNode | null;
  start: () => Promise<void>;
  pause: () => void;
  resume: () => void;
  stop: () => void;
  resetResult: () => void;
}

export function useLiveAudio(
  onResult?: (result: ModalityResult) => void,
  chunkIntervalMs: number = LIVE_AUDIO_INTERVAL_MS
): UseLiveAudioReturn {
  const [status, setStatus] = useState<LiveAudioStatus>('idle');
  const [error, setError] = useState<string | null>(null);
  const [audioResult, setAudioResult] = useState<ModalityResult | null>(null);
  const [volume, setVolume] = useState<number>(0);
  const [analyserNode, setAnalyserNode] = useState<AnalyserNode | null>(null);

  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const sourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const processorNodeRef = useRef<ScriptProcessorNode | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animFrameRef = useRef<number | null>(null);

  const audioChunksRef = useRef<Float32Array[]>([]);
  const recentResultsRef = useRef<ModalityResult[]>([]);
  const isPausedRef = useRef<boolean>(false);
  const isAnalyzingRef = useRef<boolean>(false);
  const abortControllerRef = useRef<AbortController | null>(null);
  const intervalTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);


  // Analyze the buffered audio chunk
  const sendChunkForInference = useCallback(async () => {
    if (isPausedRef.current || isAnalyzingRef.current || !audioContextRef.current) {
      return;
    }

    const sampleRate = audioContextRef.current.sampleRate || 16000;
    const { window, rollingChunks } = selectModelAudioWindow(
      audioChunksRef.current,
      sampleRate,
      MODEL_AUDIO_WINDOW_SECONDS
    );
    if (!window) return;
    audioChunksRef.current = rollingChunks;

    // Skip near-silence while retaining the overlapping context for the next window.
    let sumSq = 0;
    for (let i = 0; i < window.length; i++) {
      sumSq += window[i] * window[i];
    }
    const rms = Math.sqrt(sumSq / window.length);
    if (rms < 0.002) {
      // Background noise / silence floor: do not waste inference
      return;
    }

    isAnalyzingRef.current = true;
    setStatus('analyzing');

    try {
      const wavBlob = encodeWAV(window, sampleRate, 1);
      const wavFile = new File([wavBlob], 'microphone_chunk.wav', { type: 'audio/wav' });

      // Cancel any previous pending request
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      abortControllerRef.current = new AbortController();

      const result = await predictAudio(wavFile, abortControllerRef.current.signal);
      const smoothed = smoothAudioResult(recentResultsRef.current, result);
      recentResultsRef.current = smoothed.history;
      setAudioResult(smoothed.result);
      if (onResult) {
        onResult(smoothed.result);
      }
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        console.warn('Live audio inference note:', err.message);
        // Do not crash the entire mic stream on a single chunk error
      }
    } finally {
      isAnalyzingRef.current = false;
      if (!isPausedRef.current && streamRef.current) {
        setStatus('recording');
      }
    }
  }, [onResult]);

  // Volume meter animation loop
  const updateVolume = useCallback(() => {
    if (analyserRef.current && !isPausedRef.current) {
      const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
      analyserRef.current.getByteFrequencyData(dataArray);

      let sum = 0;
      for (let i = 0; i < dataArray.length; i++) {
        sum += dataArray[i];
      }
      const avg = sum / dataArray.length;
      setVolume(Math.min(1, avg / 128)); // normalized volume
    } else {
      setVolume(0);
    }

    if (streamRef.current && !isPausedRef.current) {
      animFrameRef.current = requestAnimationFrame(updateVolume);
    }
  }, []);

  const stop = useCallback(() => {
    // Stop inference interval
    if (intervalTimerRef.current) {
      clearInterval(intervalTimerRef.current);
      intervalTimerRef.current = null;
    }

    // Cancel pending inference
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }

    // Cancel animation frame
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = null;
    }

    // Disconnect Web Audio nodes
    if (processorNodeRef.current) {
      processorNodeRef.current.disconnect();
      processorNodeRef.current = null;
    }
    if (sourceNodeRef.current) {
      sourceNodeRef.current.disconnect();
      sourceNodeRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
    }

    // Stop MediaStream tracks
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    setAnalyserNode(null);
    analyserRef.current = null;
    audioChunksRef.current = [];
    recentResultsRef.current = [];
    isPausedRef.current = false;
    isAnalyzingRef.current = false;
    setVolume(0);
    setStatus('idle');
  }, []);

  const start = useCallback(async () => {
    // Stop any existing stream first
    stop();
    setError(null);
    setStatus('requesting');

    if (!navigator?.mediaDevices?.getUserMedia) {
      setError('Live microphone capture is not supported in this browser environment.');
      setStatus('error');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      streamRef.current = stream;

      // AudioContext: sampleRate ideally 16000 if supported, or browser default
      const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
      const audioCtx = new AudioCtx();
      if (audioCtx.state === 'suspended') await audioCtx.resume();
      audioContextRef.current = audioCtx;

      const sourceNode = audioCtx.createMediaStreamSource(stream);
      sourceNodeRef.current = sourceNode;

      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      analyser.smoothingTimeConstant = 0.8;
      analyserRef.current = analyser;
      setAnalyserNode(analyser);

      // ScriptProcessorNode for raw PCM chunking
      const bufferSize = 4096;
      const processor = audioCtx.createScriptProcessor(bufferSize, 1, 1);
      processorNodeRef.current = processor;

      processor.onaudioprocess = (e) => {
        if (isPausedRef.current) return;
        const inputData = e.inputBuffer.getChannelData(0);
        // Clone Float32Array
        const sampleRate = audioContextRef.current?.sampleRate || 16000;
        audioChunksRef.current = appendBoundedAudio(
          audioChunksRef.current,
          new Float32Array(inputData),
          Math.round(sampleRate * MAX_LIVE_BUFFER_SECONDS)
        );
      };

      sourceNode.connect(analyser);
      sourceNode.connect(processor);
      // Connect processor to destination to keep it alive in Chrome/Safari
      processor.connect(audioCtx.destination);

      isPausedRef.current = false;
      setStatus('recording');

      // Start volume animation
      animFrameRef.current = requestAnimationFrame(updateVolume);

      // Start periodic chunk evaluation
      intervalTimerRef.current = setInterval(() => {
        sendChunkForInference();
      }, chunkIntervalMs);
    } catch (err: any) {
      console.error('Microphone access error:', err);
      let userFriendlyMessage = 'Failed to access microphone.';

      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        userFriendlyMessage =
          'Microphone permission was denied. Please allow microphone access in your browser to enable live speech emotion recognition.';
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        userFriendlyMessage =
          'No microphone hardware was detected on your system. Please connect an audio input device.';
      } else if (err.name === 'NotReadableError') {
        userFriendlyMessage =
          'Microphone is currently in use by another application. Please close other audio programs and try again.';
      } else if (err.message) {
        userFriendlyMessage = `Microphone error: ${err.message}`;
      }

      setError(userFriendlyMessage);
      setStatus('error');
      stop();
    }
  }, [stop, sendChunkForInference, updateVolume, chunkIntervalMs]);

  const pause = useCallback(() => {
    isPausedRef.current = true;
    setStatus('paused');
    setVolume(0);
  }, []);

  const resume = useCallback(() => {
    if (streamRef.current) {
      isPausedRef.current = false;
      setStatus('recording');
      animFrameRef.current = requestAnimationFrame(updateVolume);
    }
  }, [updateVolume]);

  const resetResult = useCallback(() => {
    setAudioResult(null);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stop();
    };
  }, [stop]);

  return {
    status,
    error,
    audioResult,
    volume,
    analyserNode,
    start,
    pause,
    resume,
    stop,
    resetResult,
  };
}
