"""
Audio / Speech Feature Extraction Pipeline.
Extracts 39-channel MFCC features matching the speech-emotion-recognition CNN input,
plus acoustic descriptors (pitch, energy, ZCR) for explainability attribution.
Zero global side effects (no os.chdir).
"""

import io
import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_RATE = 16000
MEAN_SIGNAL_LENGTH = 32000  # 2.0 seconds at 16 kHz (Empirically calibrated for Emo-DB CNN)


class AudioFeatureExtractor:
    """Extracts acoustic signals and MFCC representations for speech emotion models."""

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE, mfcc_len: int = 39):
        self.sample_rate = sample_rate
        self.mfcc_len = mfcc_len

    def load_audio(self, audio_source: Any) -> Tuple[np.ndarray, int]:
        """
        Load audio from a file path, file-like object, raw bytes, or numpy array.
        Returns:
            signal: 1D numpy array of 16-bit integer PCM or float normalized
            sr: sampling rate in Hz
        """
        if isinstance(audio_source, tuple) and len(audio_source) == 2 and isinstance(audio_source[0], np.ndarray):
            sig = audio_source[0]
            if sig.ndim > 1:
                sig = np.mean(sig, axis=1)
            if np.issubdtype(sig.dtype, np.floating):
                sig = np.clip(sig, -1.0, 1.0)
                sig = (sig * 32767).astype(np.int16)
            else:
                sig = sig.astype(np.int16)
            return sig, int(audio_source[1])

        if isinstance(audio_source, np.ndarray):
            sig = audio_source
            if sig.ndim > 1:
                sig = np.mean(sig, axis=1)
            if np.issubdtype(sig.dtype, np.floating):
                sig = np.clip(sig, -1.0, 1.0)
                sig = (sig * 32767).astype(np.int16)
            else:
                sig = sig.astype(np.int16)
            return sig, self.sample_rate

        import soundfile as sf

        try:

            if isinstance(audio_source, bytes):
                data, sr = sf.read(io.BytesIO(audio_source), dtype="int16")
            else:
                data, sr = sf.read(audio_source, dtype="int16")

            # Convert multi-channel (stereo) to mono
            if data.ndim > 1:
                data = np.mean(data, axis=1).astype(np.int16)

            return data, sr

        except Exception as e:
            logger.debug("soundfile failed (%s), attempting scipy.io.wavfile", e)

        import scipy.io.wavfile as wav

        try:
            if isinstance(audio_source, bytes):
                sr, data = wav.read(io.BytesIO(audio_source))
            else:
                sr, data = wav.read(audio_source)

            if data.ndim > 1:
                data = np.mean(data, axis=1)

            if data.dtype != np.int16:
                # Normalize float to int16 range if needed
                data = (data * 32767).astype(np.int16)

            return data, sr

        except Exception as e2:
            logger.error("Failed to decode audio: %s", e2)
            raise ValueError(f"Unable to read audio format: {e2}")

    def pad_or_crop_signal(self, signal: np.ndarray, target_length: int = MEAN_SIGNAL_LENGTH) -> np.ndarray:
        """
        Pad with zeros or centrally crop signal to exactly target_length (32,000 samples).
        Matches the exact logic in speechemotionrecognition/utilities.py.
        """
        s_len = len(signal)
        if s_len < target_length:
            pad_len = target_length - s_len
            pad_rem = pad_len % 2
            pad_len //= 2
            return np.pad(signal, (pad_len, pad_len + pad_rem), "constant", constant_values=0)
        else:
            pad_len = s_len - target_length
            pad_len //= 2
            return signal[pad_len : pad_len + target_length]

    def extract_mfcc(self, signal: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract MFCC feature matrix matching the CNN input shape (198, 39).
        Resamples to 16 kHz if sample rate differs.
        """
        import librosa
        from speechpy.feature import mfcc

        # Resample to 16 kHz if necessary
        if sr != self.sample_rate:
            float_sig = signal.astype(np.float32) / 32768.0
            resampled = librosa.resample(float_sig, orig_sr=sr, target_sr=self.sample_rate)
            signal = (resampled * 32768.0).astype(np.int16)
            sr = self.sample_rate

        # Standardize signal length to 32,000 samples
        calibrated_signal = self.pad_or_crop_signal(signal, MEAN_SIGNAL_LENGTH)

        # Extract 39-cepstral MFCC matrix: Shape (198, 39)
        mel_coefficients = mfcc(calibrated_signal, sr, num_cepstral=self.mfcc_len)
        return mel_coefficients

    def extract_acoustic_descriptors(self, signal: np.ndarray, sr: int) -> Dict[str, float]:
        """Compute acoustic prosody metrics for explainability attribution."""
        if len(signal) == 0:
            return {"duration_sec": 0.0, "rms_energy": 0.0, "zcr": 0.0}

        float_sig = signal.astype(np.float32) / 32768.0
        duration = len(signal) / float(sr)
        rms = float(np.sqrt(np.mean(float_sig**2)))

        # Zero crossing rate
        zero_crossings = np.sum(np.abs(np.diff(np.signbit(float_sig))))
        zcr = float(zero_crossings / max(1, len(float_sig)))

        descriptors = {
            "duration_sec": round(duration, 2),
            "rms_energy": round(rms, 4),
            "zcr": round(zcr, 4),
        }

        # Spectral Centroid via FFT
        try:
            fft_mag = np.abs(np.fft.rfft(float_sig))
            freqs = np.fft.rfftfreq(len(float_sig), d=1.0 / sr)
            if np.sum(fft_mag) > 0:
                centroid = float(np.sum(freqs * fft_mag) / np.sum(fft_mag))
                descriptors["spectral_centroid_hz"] = round(centroid, 1)
        except Exception:
            pass

        return descriptors
