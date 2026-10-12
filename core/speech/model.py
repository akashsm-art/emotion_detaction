"""
Speech Emotion Recognition Model.
Loads and executes the deep 2D Convolutional Acoustic Network using genuine weights
from speech-emotion-recognition-master/models/best_model_CNN.h5.
ZERO FAKE AI: Runs full neural forward pass over MFCC tensors. Unsupported emotions
(disgust, fear, surprise) are assigned 0.0 probability without artificial mass.
"""

import os
import logging
from typing import Dict, Any, Optional, List
import numpy as np
import torch
import torch.nn as nn

from core.ontology import (
    CORE_EMOTIONS,
    EMOTION_VALENCE_AROUSAL,
    normalize_to_core_ontology,
    create_empty_distribution,
)
from core.schemas import ModalityResult
from core.speech.feature_extractor import AudioFeatureExtractor

logger = logging.getLogger(__name__)

# The 4 classes of the trained Emo-DB CNN model
SPEECH_SUPPORTED_EMOTIONS = ["neutral", "angry", "happy", "sad"]
RAW_SPEECH_CLASSES = ["Neutral", "Angry", "Happy", "Sad"]
DEFAULT_SPEECH_WEIGHTS = "models/speech/best_model_CNN.h5"
MIN_AUDIO_RMS = 0.002


class _SpeechEmotionCNNModule(nn.Module):
    """
    Exact PyTorch equivalent of the Keras Sequential CNN from speech-emotion-recognition-master.
    Layers:
        Conv2D(8, 13x13) -> BatchNorm -> ReLU
        Conv2D(8, 13x13) -> BatchNorm -> ReLU -> MaxPool2D(2, 1)
        Conv2D(8, 13x13) -> BatchNorm -> ReLU
        Conv2D(8, 2x2)   -> BatchNorm -> ReLU -> MaxPool2D(2, 1)
        Flatten (matching Keras H, W, C order)
        Dense(64) -> BatchNorm -> ReLU -> Dropout
        Dense(4) -> Softmax
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(13, 13))
        self.bn1 = nn.BatchNorm2d(8, eps=0.001, momentum=0.99)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(8, 8, kernel_size=(13, 13))
        self.bn2 = nn.BatchNorm2d(8, eps=0.001, momentum=0.99)
        self.act2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1))

        self.conv3 = nn.Conv2d(8, 8, kernel_size=(13, 13))
        self.bn3 = nn.BatchNorm2d(8, eps=0.001, momentum=0.99)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(8, 8, kernel_size=(2, 2))
        self.bn4 = nn.BatchNorm2d(8, eps=0.001, momentum=0.99)
        self.act4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1))

        self.fc1 = nn.Linear(592, 64)
        self.bn5 = nn.BatchNorm1d(64, eps=0.001, momentum=0.99)
        self.act5 = nn.ReLU()

        self.fc2 = nn.Linear(64, 4)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x shape: (B, 1, 198, 39)
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.pool1(self.act2(self.bn2(self.conv2(x))))
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.pool2(self.act4(self.bn4(self.conv4(x))))

        # Transpose to (B, H, W, C) before flatten to replicate Keras memory layout
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(x.size(0), -1)

        x = self.act5(self.bn5(self.fc1(x)))
        return self.softmax(self.fc2(x))


class SpeechEmotionModel:
    """Loads trained CNN weights and executes real acoustic emotion classification."""

    def __init__(
        self,
        weights_path: str = DEFAULT_SPEECH_WEIGHTS,
        device: str = "cpu",
        sample_rate: int = 16000,
        mfcc_len: int = 39,
    ):
        self.weights_path = weights_path
        self.device = device
        self.extractor = AudioFeatureExtractor(sample_rate=sample_rate, mfcc_len=mfcc_len)
        self.model = _SpeechEmotionCNNModule().to(device)
        self.model.eval()
        self._is_loaded = False
        self.load_error: Optional[str] = None

        self._load_trained_weights()

    def _load_trained_weights(self) -> None:
        """Load trained layer weights from HDF5 into PyTorch model."""
        resolved_path = self.weights_path
        if not os.path.exists(resolved_path):
            # Try workspace fallback
            fallback_path = os.path.join(
                "speech-emotion-recognition-master", "models", "best_model_CNN.h5"
            )
            if os.path.exists(fallback_path):
                resolved_path = fallback_path
            else:
                self.load_error = f"Checkpoint file not found: {self.weights_path}"
                self._is_loaded = False
                logger.error(self.load_error)
                return

        try:
            import h5py

            with h5py.File(resolved_path, "r") as h:
                # Conv 1
                c1_k = np.array(h["conv2d_1"]["conv2d_1"]["kernel:0"]).transpose(3, 2, 0, 1)
                c1_b = np.array(h["conv2d_1"]["conv2d_1"]["bias:0"])
                self.model.conv1.weight.data = torch.from_numpy(c1_k).float().to(self.device)
                self.model.conv1.bias.data = torch.from_numpy(c1_b).float().to(self.device)

                # BN 1
                self.model.bn1.weight.data = torch.from_numpy(np.array(h["batch_normalization_1"]["batch_normalization_1"]["gamma:0"])).float().to(self.device)
                self.model.bn1.bias.data = torch.from_numpy(np.array(h["batch_normalization_1"]["batch_normalization_1"]["beta:0"])).float().to(self.device)
                self.model.bn1.running_mean.data = torch.from_numpy(np.array(h["batch_normalization_1"]["batch_normalization_1"]["moving_mean:0"])).float().to(self.device)
                self.model.bn1.running_var.data = torch.from_numpy(np.array(h["batch_normalization_1"]["batch_normalization_1"]["moving_variance:0"])).float().to(self.device)

                # Conv 2
                c2_k = np.array(h["conv2d_2"]["conv2d_2"]["kernel:0"]).transpose(3, 2, 0, 1)
                c2_b = np.array(h["conv2d_2"]["conv2d_2"]["bias:0"])
                self.model.conv2.weight.data = torch.from_numpy(c2_k).float().to(self.device)
                self.model.conv2.bias.data = torch.from_numpy(c2_b).float().to(self.device)

                # BN 2
                self.model.bn2.weight.data = torch.from_numpy(np.array(h["batch_normalization_2"]["batch_normalization_2"]["gamma:0"])).float().to(self.device)
                self.model.bn2.bias.data = torch.from_numpy(np.array(h["batch_normalization_2"]["batch_normalization_2"]["beta:0"])).float().to(self.device)
                self.model.bn2.running_mean.data = torch.from_numpy(np.array(h["batch_normalization_2"]["batch_normalization_2"]["moving_mean:0"])).float().to(self.device)
                self.model.bn2.running_var.data = torch.from_numpy(np.array(h["batch_normalization_2"]["batch_normalization_2"]["moving_variance:0"])).float().to(self.device)

                # Conv 3
                c3_k = np.array(h["conv2d_3"]["conv2d_3"]["kernel:0"]).transpose(3, 2, 0, 1)
                c3_b = np.array(h["conv2d_3"]["conv2d_3"]["bias:0"])
                self.model.conv3.weight.data = torch.from_numpy(c3_k).float().to(self.device)
                self.model.conv3.bias.data = torch.from_numpy(c3_b).float().to(self.device)

                # BN 3
                self.model.bn3.weight.data = torch.from_numpy(np.array(h["batch_normalization_3"]["batch_normalization_3"]["gamma:0"])).float().to(self.device)
                self.model.bn3.bias.data = torch.from_numpy(np.array(h["batch_normalization_3"]["batch_normalization_3"]["beta:0"])).float().to(self.device)
                self.model.bn3.running_mean.data = torch.from_numpy(np.array(h["batch_normalization_3"]["batch_normalization_3"]["moving_mean:0"])).float().to(self.device)
                self.model.bn3.running_var.data = torch.from_numpy(np.array(h["batch_normalization_3"]["batch_normalization_3"]["moving_variance:0"])).float().to(self.device)

                # Conv 4
                c4_k = np.array(h["conv2d_4"]["conv2d_4"]["kernel:0"]).transpose(3, 2, 0, 1)
                c4_b = np.array(h["conv2d_4"]["conv2d_4"]["bias:0"])
                self.model.conv4.weight.data = torch.from_numpy(c4_k).float().to(self.device)
                self.model.conv4.bias.data = torch.from_numpy(c4_b).float().to(self.device)

                # BN 4
                self.model.bn4.weight.data = torch.from_numpy(np.array(h["batch_normalization_4"]["batch_normalization_4"]["gamma:0"])).float().to(self.device)
                self.model.bn4.bias.data = torch.from_numpy(np.array(h["batch_normalization_4"]["batch_normalization_4"]["beta:0"])).float().to(self.device)
                self.model.bn4.running_mean.data = torch.from_numpy(np.array(h["batch_normalization_4"]["batch_normalization_4"]["moving_mean:0"])).float().to(self.device)
                self.model.bn4.running_var.data = torch.from_numpy(np.array(h["batch_normalization_4"]["batch_normalization_4"]["moving_variance:0"])).float().to(self.device)

                # Dense 1
                d1_k = np.array(h["dense_1"]["dense_1"]["kernel:0"]).transpose(1, 0)
                d1_b = np.array(h["dense_1"]["dense_1"]["bias:0"])
                self.model.fc1.weight.data = torch.from_numpy(d1_k).float().to(self.device)
                self.model.fc1.bias.data = torch.from_numpy(d1_b).float().to(self.device)

                # BN 5
                self.model.bn5.weight.data = torch.from_numpy(np.array(h["batch_normalization_5"]["batch_normalization_5"]["gamma:0"])).float().to(self.device)
                self.model.bn5.bias.data = torch.from_numpy(np.array(h["batch_normalization_5"]["batch_normalization_5"]["beta:0"])).float().to(self.device)
                self.model.bn5.running_mean.data = torch.from_numpy(np.array(h["batch_normalization_5"]["batch_normalization_5"]["moving_mean:0"])).float().to(self.device)
                self.model.bn5.running_var.data = torch.from_numpy(np.array(h["batch_normalization_5"]["batch_normalization_5"]["moving_variance:0"])).float().to(self.device)

                # Dense 2
                d2_k = np.array(h["dense_2"]["dense_2"]["kernel:0"]).transpose(1, 0)
                d2_b = np.array(h["dense_2"]["dense_2"]["bias:0"])
                self.model.fc2.weight.data = torch.from_numpy(d2_k).float().to(self.device)
                self.model.fc2.bias.data = torch.from_numpy(d2_b).float().to(self.device)

            self._is_loaded = True
            self.load_error = None
            logger.info("Successfully loaded trained Speech CNN weights from %s", resolved_path)

        except Exception as e:
            self.load_error = str(e)
            self._is_loaded = False
            logger.error("Failed to load Speech CNN weights from %s: %s", resolved_path, e)

    def predict(self, audio_source: Any) -> ModalityResult:
        """Classify emotion from an audio file path, file object, or raw bytes."""
        if not self._is_loaded:
            empty_probs = create_empty_distribution()
            return ModalityResult(
                modality="audio",
                available=False,
                status=f"SPEECH_INFERENCE_BLOCKED — CHECKPOINT_NOT_LOADED: {self.load_error}",
                emotion="none",
                confidence=0.0,
                probabilities=empty_probs,
                supported_emotions=SPEECH_SUPPORTED_EMOTIONS,
                valence=0.0,
                arousal=0.0,
                metadata={"error": self.load_error},
            )

        try:
            signal, sr = self.extractor.load_audio(audio_source)
        except Exception as e:
            empty_probs = create_empty_distribution()
            return ModalityResult(
                modality="audio",
                available=False,
                status=f"AUDIO_LOAD_ERROR: {e}",
                emotion="none",
                confidence=0.0,
                probabilities=empty_probs,
                supported_emotions=SPEECH_SUPPORTED_EMOTIONS,
                valence=0.0,
                arousal=0.0,
                metadata={"error": str(e)},
            )

        if len(signal) == 0:
            empty_probs = create_empty_distribution()
            return ModalityResult(
                modality="audio",
                available=False,
                status="EMPTY_AUDIO_SIGNAL",
                emotion="none",
                confidence=0.0,
                probabilities=empty_probs,
                supported_emotions=SPEECH_SUPPORTED_EMOTIONS,
                valence=0.0,
                arousal=0.0,
                metadata={"reason": "Audio signal contained 0 samples"},
            )

        try:
            # Acoustic descriptors for explainability
            descriptors = self.extractor.extract_acoustic_descriptors(signal, sr)
            signal_float = signal.astype(np.float32) / 32768.0
            rms_energy = float(np.sqrt(np.mean(signal_float**2)))
            if rms_energy < MIN_AUDIO_RMS:
                empty_probs = create_empty_distribution()
                return ModalityResult(
                    modality="audio",
                    available=False,
                    status="SILENT_OR_NEAR_SILENT_AUDIO",
                    emotion="none",
                    confidence=0.0,
                    probabilities=empty_probs,
                    supported_emotions=SPEECH_SUPPORTED_EMOTIONS,
                    valence=0.0,
                    arousal=0.0,
                    metadata={
                        **descriptors,
                        "reason": "Audio RMS is below the minimum speech level",
                        "rms_threshold": MIN_AUDIO_RMS,
                    },
                )

            # Extract 39-cepstral MFCC matrix: Shape (198, 39)
            mel_matrix = self.extractor.extract_mfcc(signal, sr)

            # Shape for CNN: (1, 1, 198, 39)
            tensor_in = torch.from_numpy(mel_matrix).float().unsqueeze(0).unsqueeze(0).to(self.device)

            with torch.no_grad():
                out_probs = self.model(tensor_in).squeeze().cpu().numpy()

            raw_probs: Dict[str, float] = {}
            for label, prob in zip(RAW_SPEECH_CLASSES, out_probs):
                raw_probs[label] = float(prob)

            # Map to 7-core ontology with STRICT 0.0 for unsupported classes (disgust, fear, surprise)
            normalized_probs = normalize_to_core_ontology(
                raw_probs, supported_emotions=SPEECH_SUPPORTED_EMOTIONS
            )

            top_emotion = max(normalized_probs, key=normalized_probs.get)
            confidence = float(normalized_probs[top_emotion])

            valence = float(
                sum(normalized_probs[e] * EMOTION_VALENCE_AROUSAL[e]["valence"] for e in CORE_EMOTIONS)
            )
            arousal = float(
                sum(normalized_probs[e] * EMOTION_VALENCE_AROUSAL[e]["arousal"] for e in CORE_EMOTIONS)
            )

            return ModalityResult(
                modality="audio",
                available=True,
                status="ok",
                emotion=top_emotion,
                confidence=round(confidence, 4),
                probabilities=normalized_probs,
                supported_emotions=SPEECH_SUPPORTED_EMOTIONS,
                valence=round(valence, 4),
                arousal=round(arousal, 4),
                metadata={
                    **descriptors,
                    "input_mfcc_shape": list(mel_matrix.shape),
                    "weights_path": self.weights_path,
                    "engine": "deep_cnn_mfcc_acoustic",
                },
            )

        except Exception as e:
            logger.error("Speech forward pass error: %s", e)
            empty_probs = create_empty_distribution()
            return ModalityResult(
                modality="audio",
                available=False,
                status=f"INFERENCE_EXECUTION_ERROR: {e}",
                emotion="none",
                confidence=0.0,
                probabilities=empty_probs,
                supported_emotions=SPEECH_SUPPORTED_EMOTIONS,
                valence=0.0,
                arousal=0.0,
                metadata={"error": str(e)},
            )
