"""
Speech Emotion Recognition Subsystem.
Extracts 39-channel MFCC acoustic features and executes deep CNN inference using
the trained weights from speech-emotion-recognition-master/models/best_model_CNN.h5.
"""

from core.speech.feature_extractor import AudioFeatureExtractor
from core.speech.model import SpeechEmotionModel

__all__ = ["AudioFeatureExtractor", "SpeechEmotionModel"]
