"""
Test script for integrated VisionEmotionModel.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from core.vision.model import VisionEmotionModel

def test():
    print("Instantiating VisionEmotionModel...")
    vm = VisionEmotionModel()
    print(f"Loaded: {vm._is_loaded}")
    print(f"Weights path: {vm.weights_path}")
    print(f"Load error: {vm.load_error}")
    assert vm._is_loaded, "VisionEmotionModel failed to load!"

    # Test prediction on a sample image
    sample_path = os.path.join(
        PROJECT_ROOT,
        "Multimodal-Emotion-Recognition-master",
        "image_data",
        "valid",
        "Actor_21",
        "01-01-01-01-01-01-21",
        "30.jpg"
    )
    result = vm.predict(sample_path)
    print("\nPrediction on Actor_21/01-01-01-01-01-01-21/30.jpg (neutral):")
    print(f"  Modality: {result.modality}")
    print(f"  Available: {result.available}")
    print(f"  Status: {result.status}")
    print(f"  Emotion: {result.emotion}")
    print(f"  Confidence: {result.confidence}")
    print(f"  Probabilities: {result.probabilities}")
    print(f"  Valence: {result.valence}, Arousal: {result.arousal}")
    print(f"  Metadata: {result.metadata}")
    assert result.available, "Vision result is not available!"
    assert result.status == "ok", f"Vision status is {result.status}"
    print("\n[PASS] VisionEmotionModel prediction successful!")

if __name__ == "__main__":
    test()
