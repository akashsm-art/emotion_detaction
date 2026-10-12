"""
Facial Vision Emotion Recognition Subsystem.
Implements face detection, alignment, and 224x224 normalization using OpenCV.
Preserves architecture for AlexNet visual emotion classifier.
"""

from core.vision.face_detector import FaceDetector
from core.vision.model import VisionEmotionModel

__all__ = ["FaceDetector", "VisionEmotionModel"]
