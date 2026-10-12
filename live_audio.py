"""
Standalone Native Real-Time Speech Emotion Recognition Program.
Captures live microphone audio using sounddevice, runs acoustic feature extraction (39-channel MFCCs),
and executes the trained 2D CNN acoustic model (models/speech/best_model_CNN.h5).
Outputs predicted emotion, confidence, and 7-class probability distributions directly in the terminal.
"""

import sys
import time
import argparse
import logging
from datetime import datetime
from typing import Dict, Optional
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    print("\n[ERROR] 'sounddevice' library is required for live microphone capture.")
    print("Install it with: pip install sounddevice\n")
    sys.exit(1)

from core.speech.model import SpeechEmotionModel, SPEECH_SUPPORTED_EMOTIONS
from core.speech.feature_extractor import DEFAULT_SAMPLE_RATE, MEAN_SIGNAL_LENGTH
from core.ontology import CORE_EMOTIONS

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")


class LiveAudioEmotionRecognizer:
    """Manages microphone streaming, circular buffer windowing, and continuous speech CNN inference."""

    def __init__(
        self,
        weights_path: str = "models/speech/best_model_CNN.h5",
        device_index: Optional[int] = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        window_duration_sec: float = 2.0,
        step_interval_sec: float = 1.0,
        energy_threshold: float = 0.003,
        duration: Optional[float] = None,
    ):
        self.weights_path = weights_path
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.window_duration_sec = window_duration_sec
        self.step_interval_sec = step_interval_sec
        self.energy_threshold = energy_threshold
        self.duration = duration


        self.window_samples = int(self.sample_rate * self.window_duration_sec)  # 32,000 samples for 2.0s
        self.audio_buffer = np.zeros(self.window_samples, dtype=np.float32)
        self.samples_collected = 0

        print("\n" + "=" * 60)
        print("  INITIALIZING LIVE SPEECH EMOTION RECOGNITION")
        print("=" * 60)
        print(f"Loading trained Speech CNN from: {self.weights_path}")
        self.speech_model = SpeechEmotionModel(weights_path=self.weights_path)

        if not self.speech_model._is_loaded:
            print(f"[FATAL] Speech model could not be loaded: {self.speech_model.load_error}")
            sys.exit(1)

        print("[OK] Speech CNN loaded successfully.")
        print(f"Sampling Rate:      {self.sample_rate} Hz (16 kHz)")
        print(f"Window Duration:    {self.window_duration_sec:.1f} s ({self.window_samples} samples)")
        print(f"Analysis Step:      {self.step_interval_sec:.1f} s interval")
        print(f"Supported Classes:  {SPEECH_SUPPORTED_EMOTIONS}")
        print("Unsupported:        ['disgust', 'fear', 'surprise'] (strictly 0.00%)")
        print("-" * 60)

    def _audio_callback(self, indata, frames, time_info, status):
        """Streaming callback from sounddevice collecting microphone samples into circular buffer."""
        if status:
            pass  # Overflow / underflow flag
        mono_chunk = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        chunk_len = len(mono_chunk)

        # Shift buffer left and append new samples
        if chunk_len >= self.window_samples:
            self.audio_buffer[:] = mono_chunk[-self.window_samples :]
        else:
            self.audio_buffer[:-chunk_len] = self.audio_buffer[chunk_len:]
            self.audio_buffer[-chunk_len:] = mono_chunk

        self.samples_collected += chunk_len

    def run(self):
        """Continuously capture audio and execute speech CNN inference on rolling windows."""
        try:
            device_info = sd.query_devices(self.device_index, "input")
            device_name = device_info.get("name", f"Device #{self.device_index}")
        except Exception as e:
            print(f"\n[ERROR] Unable to access microphone device: {e}")
            print("Ensure a functional microphone is connected.")
            sys.exit(1)

        print(f"Microphone:         {device_name}")
        print("Status:             LISTENING")
        print("Press Ctrl+C to stop listening.")
        print("=" * 60 + "\n")

        window_index = 0

        try:
            with sd.InputStream(
                device=self.device_index,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=int(self.sample_rate * 0.1),  # 100ms blocks
                callback=self._audio_callback,
            ):
                # Wait for initial buffer to fill
                time.sleep(self.window_duration_sec)

                start_time = time.time()
                while True:
                    time.sleep(self.step_interval_sec)
                    if self.duration and (time.time() - start_time) >= self.duration:
                        print(f"\n[LIVE AUDIO] Reached requested duration ({self.duration:.1f}s). Exiting loop.")
                        break

                    window_index += 1
                    timestamp = datetime.now().strftime("%H:%M:%S")


                    # Copy latest 2.0s audio window
                    current_window = self.audio_buffer.copy()

                    # Compute RMS energy to detect silence
                    rms = float(np.sqrt(np.mean(current_window**2)))

                    if rms < self.energy_threshold:
                        print(f"[{timestamp}] Window #{window_index:03d} | (Silence / Background noise - RMS: {rms:.4f})")
                        continue

                    # Execute genuine model inference
                    res = self.speech_model.predict((current_window, self.sample_rate))

                    if not res.available:
                        print(f"[{timestamp}] Window #{window_index:03d} | Error: {res.status}")
                        continue

                    emotion = res.emotion.upper()
                    conf = res.confidence * 100.0
                    probs = res.probabilities

                    print("-" * 55)
                    print(f"[{timestamp}] Audio Window #{window_index:03d}")
                    print(f"Emotion:    {emotion}")
                    print(f"Confidence: {conf:.2f}%  (Model Certainty Score)")
                    print("-" * 55)
                    print("Probabilities:")
                    for em in ["neutral", "angry", "happy", "sad"]:
                        p = probs.get(em, 0.0) * 100.0
                        bars = "#" * int(p / 5)
                        print(f"  {em.capitalize():<10} {p:6.2f}%  [{bars:<20}]")

                    print("\nUnsupported classes (Emo-DB CNN architecture restriction):")
                    for em in ["disgust", "fear", "surprise"]:
                        p = probs.get(em, 0.0) * 100.0
                        print(f"  {em.capitalize():<10} {p:6.2f}%")

                    print("-" * 55)
                    print("# Listening for next audio window...\n")

        except KeyboardInterrupt:
            print("\n" + "=" * 60)
            print("[LIVE AUDIO] Interrupted by user (Ctrl+C).")
            print("[LIVE AUDIO] Microphone stream closed. Terminated cleanly.")
            print("=" * 60)
        except Exception as e:
            print(f"\n[ERROR] Audio streaming error: {e}")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Live Microphone Speech Emotion Recognition using 2D CNN")
    parser.add_argument("--device", type=int, default=None, help="Input microphone device index (default: system default)")
    parser.add_argument("--window", type=float, default=2.0, help="Window duration in seconds (default: 2.0)")
    parser.add_argument("--step", type=float, default=1.0, help="Inference interval step in seconds (default: 1.0)")
    parser.add_argument("--weights", type=str, default="models/speech/best_model_CNN.h5", help="Checkpoint path")
    parser.add_argument("--threshold", type=float, default=0.003, help="Silence RMS threshold (default: 0.003)")
    parser.add_argument("--duration", type=float, default=None, help="Optional duration in seconds to run before auto-exiting (default: run indefinitely until Ctrl+C)")
    args = parser.parse_args()

    recognizer = LiveAudioEmotionRecognizer(
        weights_path=args.weights,
        device_index=args.device,
        window_duration_sec=args.window,
        step_interval_sec=args.step,
        energy_threshold=args.threshold,
        duration=args.duration,
    )
    recognizer.run()



if __name__ == "__main__":
    main()
