import { useState, useRef, useCallback, useEffect } from 'react';
import type { ModalityResult } from '../types/emotion';
import { predictVision } from '../services/api';

export type LiveWebcamStatus = 'idle' | 'requesting' | 'streaming' | 'analyzing' | 'error';

export interface UseLiveWebcamReturn {
  status: LiveWebcamStatus;
  error: string | null;
  visionResult: ModalityResult | null;
  videoRef: React.RefObject<HTMLVideoElement | null>;
  faceDetected: boolean;
  facesCount: number;
  cameraFps: number;
  inferenceRateHz: number;
  start: () => Promise<void>;
  stop: () => void;
  resetResult: () => void;
}

export function useLiveWebcam(
  onResult?: (result: ModalityResult) => void,
  inferenceIntervalMs: number = 500 // ~2 FPS inference rate
): UseLiveWebcamReturn {
  const [status, setStatus] = useState<LiveWebcamStatus>('idle');
  const [error, setError] = useState<string | null>(null);
  const [visionResult, setVisionResult] = useState<ModalityResult | null>(null);
  const [faceDetected, setFaceDetected] = useState<boolean>(false);
  const [facesCount, setFacesCount] = useState<number>(0);
  const [cameraFps, setCameraFps] = useState<number>(0);
  const [inferenceRateHz, setInferenceRateHz] = useState<number>(0);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const isAnalyzingRef = useRef<boolean>(false);

  const abortControllerRef = useRef<AbortController | null>(null);

  // Rate tracking
  const frameCountRef = useRef<number>(0);
  const lastFpsTimeRef = useRef<number>(performance.now());
  const inferenceCountRef = useRef<number>(0);
  const lastInferenceRateTimeRef = useRef<number>(performance.now());
  const animFrameRef = useRef<number | null>(null);

  // Video FPS calculation loop
  const trackVideoFps = useCallback(() => {
    frameCountRef.current += 1;
    const now = performance.now();
    const elapsed = now - lastFpsTimeRef.current;

    if (elapsed >= 1000) {
      setCameraFps(Math.round((frameCountRef.current * 1000) / elapsed));
      frameCountRef.current = 0;
      lastFpsTimeRef.current = now;
    }

    if (streamRef.current) {
      animFrameRef.current = requestAnimationFrame(trackVideoFps);
    }
  }, []);

  // Periodic frame inference
  const captureAndAnalyzeFrame = useCallback(async () => {
    const video = videoRef.current;
    if (!video || video.readyState < 2 || isAnalyzingRef.current) {
      return;
    }

    // Lazy create offscreen canvas
    if (!canvasRef.current) {
      canvasRef.current = document.createElement('canvas');
    }
    const canvas = canvasRef.current;
    const sourceWidth = video.videoWidth;
    const sourceHeight = video.videoHeight;
    if (!sourceWidth || !sourceHeight) return;

    const scale = Math.min(1, 640 / sourceWidth);
    const width = Math.round(sourceWidth * scale);
    const height = Math.round(sourceHeight * scale);

    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    try {
      ctx.drawImage(video, 0, 0, width, height);

      canvas.toBlob(
        async (blob) => {
          if (!blob || !streamRef.current) return;

          isAnalyzingRef.current = true;
          const frameFile = new File([blob], 'webcam_frame.jpg', { type: 'image/jpeg' });

          if (abortControllerRef.current) {
            abortControllerRef.current.abort();
          }
          abortControllerRef.current = new AbortController();

          try {
            const result = await predictVision(frameFile, abortControllerRef.current.signal);
            setVisionResult(result);

            // Track face detection state
            const detectedFaces = result.metadata?.faces_detected ?? 0;
            setFacesCount(detectedFaces);
            setFaceDetected(detectedFaces > 0);

            // Track inference rate
            inferenceCountRef.current += 1;
            const now = performance.now();
            const elapsed = now - lastInferenceRateTimeRef.current;
            if (elapsed >= 1000) {
              const hz = (inferenceCountRef.current * 1000) / elapsed;
              setInferenceRateHz(Math.round(hz * 10) / 10);
              inferenceCountRef.current = 0;
              lastInferenceRateTimeRef.current = now;
            }

            if (onResult) {
              onResult(result);
            }
          } catch (err: any) {
            if (err.name !== 'AbortError') {
              console.warn('Vision frame inference note:', err.message);
            }
          } finally {
            isAnalyzingRef.current = false;
          }
        },
        'image/jpeg',
        0.85
      );
    } catch (e) {
      isAnalyzingRef.current = false;
    }
  }, [onResult]);

  const stop = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }

    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    isAnalyzingRef.current = false;
    setCameraFps(0);
    setInferenceRateHz(0);
    setFaceDetected(false);
    setFacesCount(0);
    setStatus('idle');
  }, []);

  const start = useCallback(async () => {
    stop();
    setError(null);
    setStatus('requesting');

    if (!navigator?.mediaDevices?.getUserMedia) {
      setError('Live camera capture is not supported in this browser environment.');
      setStatus('error');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user',
        },
        audio: false,
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play().catch(() => {});
      }

      setStatus('streaming');

      // Start FPS tracking
      lastFpsTimeRef.current = performance.now();
      frameCountRef.current = 0;
      animFrameRef.current = requestAnimationFrame(trackVideoFps);

      // Start periodic frame capture
      lastInferenceRateTimeRef.current = performance.now();
      inferenceCountRef.current = 0;
      timerRef.current = setInterval(() => {
        captureAndAnalyzeFrame();
      }, inferenceIntervalMs);
    } catch (err: any) {
      console.error('Camera access error:', err);
      let userFriendlyMessage = 'Failed to access camera.';

      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        userFriendlyMessage =
          'Camera permission was denied. Please allow camera access in your browser settings to enable facial emotion recognition.';
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        userFriendlyMessage =
          'No camera device was detected on your system. Please connect a webcam.';
      } else if (err.name === 'NotReadableError') {
        userFriendlyMessage =
          'Camera is currently in use by another application. Please close other video programs and try again.';
      } else if (err.message) {
        userFriendlyMessage = `Camera error: ${err.message}`;
      }

      setError(userFriendlyMessage);
      setStatus('error');
      stop();
    }
  }, [stop, captureAndAnalyzeFrame, trackVideoFps, inferenceIntervalMs]);

  const resetResult = useCallback(() => {
    setVisionResult(null);
    setFaceDetected(false);
    setFacesCount(0);
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
    visionResult,
    videoRef,
    faceDetected,
    facesCount,
    cameraFps,
    inferenceRateHz,
    start,
    stop,
    resetResult,
  };
}
