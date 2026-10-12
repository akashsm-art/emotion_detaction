import { useState, useRef, useCallback, useEffect } from 'react';
import type { ModalityResult } from '../types/emotion';
import { predictText } from '../services/api';

export interface UseLiveTextReturn {
  text: string;
  setText: (value: string) => void;
  isAnalyzing: boolean;
  textResult: ModalityResult | null;
  error: string | null;
  isActive: boolean;
  setIsActive: (active: boolean) => void;
  clear: () => void;
}

export function useLiveText(
  initialText: string = 'I am thrilled with the breakthrough progress we achieved today!',
  onResult?: (result: ModalityResult) => void,
  debounceMs: number = 500
): UseLiveTextReturn {
  const [text, setText] = useState<string>(initialText);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [textResult, setTextResult] = useState<ModalityResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isActive, setIsActive] = useState<boolean>(true);

  const debounceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const lastAnalyzedTextRef = useRef<string>('');


  const runTextInference = useCallback(
    async (textToAnalyze: string) => {
      const trimmed = textToAnalyze.trim();
      if (!trimmed || !isActive) {
        setTextResult(null);
        setIsAnalyzing(false);
        return;
      }

      if (trimmed === lastAnalyzedTextRef.current && textResult) {
        return; // Already analyzed this exact text
      }

      setIsAnalyzing(true);
      setError(null);

      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      abortControllerRef.current = new AbortController();

      try {
        const result = await predictText(trimmed, abortControllerRef.current.signal);
        lastAnalyzedTextRef.current = trimmed;
        setTextResult(result);
        if (onResult) {
          onResult(result);
        }
      } catch (err: any) {
        if (err.name !== 'AbortError') {
          console.warn('Live text inference error:', err.message);
          setError(err.message);
        }
      } finally {
        setIsAnalyzing(false);
      }
    },
    [isActive, onResult, textResult]
  );

  useEffect(() => {
    if (!isActive) return;

    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }

    debounceTimerRef.current = setTimeout(() => {
      runTextInference(text);
    }, debounceMs);

    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
    };
  }, [text, isActive, debounceMs, runTextInference]);

  const clear = useCallback(() => {
    setText('');
    setTextResult(null);
    setError(null);
    lastAnalyzedTextRef.current = '';
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
  }, []);

  return {
    text,
    setText,
    isAnalyzing,
    textResult,
    error,
    isActive,
    setIsActive,
    clear,
  };
}
