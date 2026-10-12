import { useState, useRef, useCallback, useEffect } from 'react';
import type { ModalityResult, FusedEmotionResult } from '../types/emotion';
import { evaluateFusion } from '../services/api';

export interface UseLiveFusionCoordinatorReturn {
  fusedResult: FusedEmotionResult | null;
  isFusing: boolean;
  fusionError: string | null;
  updateModalityResult: (modality: 'text' | 'audio' | 'vision', result: ModalityResult | null) => void;
  clearModalityResult: (modality: 'text' | 'audio' | 'vision') => void;
  resetAll: () => void;
}

export function useLiveFusionCoordinator(
  activeModalities: Set<'text' | 'audio' | 'vision'>
): UseLiveFusionCoordinatorReturn {
  const [fusedResult, setFusedResult] = useState<FusedEmotionResult | null>(null);
  const [isFusing, setIsFusing] = useState<boolean>(false);
  const [fusionError, setFusionError] = useState<string | null>(null);

  const resultsRef = useRef<{
    text: ModalityResult | null;
    audio: ModalityResult | null;
    vision: ModalityResult | null;
  }>({
    text: null,
    audio: null,
    vision: null,
  });

  const abortControllerRef = useRef<AbortController | null>(null);

  const triggerFusion = useCallback(async () => {
    // Collect active modality results
    const eligibleResults: Record<string, ModalityResult> = {};

    if (activeModalities.has('text') && resultsRef.current.text?.available) {
      eligibleResults['text'] = resultsRef.current.text;
    }
    if (activeModalities.has('audio') && resultsRef.current.audio?.available) {
      eligibleResults['audio'] = resultsRef.current.audio;
    }
    if (activeModalities.has('vision') && resultsRef.current.vision?.available) {
      eligibleResults['vision'] = resultsRef.current.vision;
    }

    const eligibleKeys = Object.keys(eligibleResults);

    if (eligibleKeys.length === 0) {
      setFusedResult(null);
      return;
    }

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();

    setIsFusing(true);
    setFusionError(null);

    try {
      const fused = await evaluateFusion(eligibleResults, abortControllerRef.current.signal);
      setFusedResult(fused);
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        console.warn('Live fusion evaluation note:', err.message);
        setFusionError(err.message);
      }
    } finally {
      setIsFusing(false);
    }
  }, [activeModalities]);

  const updateModalityResult = useCallback(
    (modality: 'text' | 'audio' | 'vision', result: ModalityResult | null) => {
      resultsRef.current[modality] = result;
      triggerFusion();
    },
    [triggerFusion]
  );

  const clearModalityResult = useCallback(
    (modality: 'text' | 'audio' | 'vision') => {
      resultsRef.current[modality] = null;
      triggerFusion();
    },
    [triggerFusion]
  );

  const resetAll = useCallback(() => {
    resultsRef.current = { text: null, audio: null, vision: null };
    setFusedResult(null);
    setFusionError(null);
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
  }, []);

  // When activeModalities changes (e.g. mic or camera stopped), re-trigger fusion with dynamic exclusion
  useEffect(() => {
    triggerFusion();
  }, [triggerFusion]);

  return {
    fusedResult,
    isFusing,
    fusionError,
    updateModalityResult,
    clearModalityResult,
    resetAll,
  };
}
