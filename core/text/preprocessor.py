"""
Text Preprocessor for Emotion Recognition.
Provides text normalization, sanitization, and salient keyword extraction for explainability.
"""

import re
import string
from typing import List, Dict, Set

# Regex patterns
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
HTML_PATTERN = re.compile(r"<.*?>")
MULTIPLE_SPACES_PATTERN = re.compile(r"\s+")

# Emotional Salience Lexicon strictly for explainability attribution (NOT for inference)
EMOTIONAL_EXPLAINABILITY_LEXICON: Dict[str, str] = {
    # Happy
    "happy": "happy", "joy": "happy", "delighted": "happy", "ecstatic": "happy",
    "excited": "happy", "glad": "happy", "love": "happy", "wonderful": "happy",
    "awesome": "happy", "fantastic": "happy", "great": "happy", "cheerful": "happy",
    "proud": "happy", "blessed": "happy", "grateful": "happy", "optimistic": "happy",
    # Sad
    "sad": "sad", "unhappy": "sad", "depressed": "sad", "miserable": "sad",
    "heartbroken": "sad", "sorrow": "sad", "grief": "sad", "down": "sad",
    "lonely": "sad", "hopeless": "sad", "disappointed": "sad", "gloomy": "sad",
    "crying": "sad", "tears": "sad", "hurt": "sad", "regret": "sad",
    # Angry
    "angry": "angry", "furious": "angry", "enraged": "angry", "mad": "angry",
    "annoyed": "angry", "irritated": "angry", "frustrated": "angry", "hate": "angry",
    "outraged": "angry", "bitter": "angry", "hostile": "angry", "disgusted": "angry",
    # Fear
    "afraid": "fear", "scared": "fear", "terrified": "fear", "fear": "fear",
    "panic": "fear", "anxious": "fear", "nervous": "fear", "worried": "fear",
    "dread": "fear", "horrified": "fear", "frightened": "fear", "alarmed": "fear",
    # Surprise
    "surprised": "surprise", "shocked": "surprise", "astonished": "surprise",
    "amazed": "surprise", "stunned": "surprise", "unexpected": "surprise",
    "unbelievable": "surprise", "speechless": "surprise",
    # Disgust
    "disgust": "disgust", "revolted": "disgust", "nauseated": "disgust",
    "sickening": "disgust", "gross": "disgust", "repulsed": "disgust", "loathsome": "disgust",
}


class TextPreprocessor:
    """Preprocesses raw text for transformer sequence classification."""

    def __init__(self, remove_punctuation: bool = False, lowercase: bool = False):
        self.remove_punctuation = remove_punctuation
        self.lowercase = lowercase

    def clean(self, text: str) -> str:
        """Sanitize text by removing URLs, HTML tags, and normalizing spacing."""
        if not text:
            return ""

        text = URL_PATTERN.sub("", text)
        text = HTML_PATTERN.sub("", text)

        if self.lowercase:
            text = text.lower()

        if self.remove_punctuation:
            text = text.translate(str.maketrans("", "", string.punctuation))

        text = MULTIPLE_SPACES_PATTERN.sub(" ", text).strip()
        return text

    def extract_salient_keywords(self, text: str) -> List[Dict[str, str]]:
        """Identify key emotional words present in the text purely for explainability."""
        cleaned = text.lower()
        words = re.findall(r"\b\w+\b", cleaned)
        salient: List[Dict[str, str]] = []
        seen: Set[str] = set()

        for word in words:
            if word in EMOTIONAL_EXPLAINABILITY_LEXICON and word not in seen:
                seen.add(word)
                salient.append({
                    "word": word,
                    "associated_emotion": EMOTIONAL_EXPLAINABILITY_LEXICON[word],
                })

        return salient
