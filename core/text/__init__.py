"""
Text Emotion Recognition Subsystem.
Implements deep Transformer sequence classification for natural language emotion detection.
"""

from core.text.model import TextEmotionModel
from core.text.preprocessor import TextPreprocessor

__all__ = ["TextEmotionModel", "TextPreprocessor"]
