"""
Test script verifying all 7 multimodal combinations with genuine Vision inference:
1. Text
2. Audio
3. Vision
4. Text + Audio
5. Text + Vision
6. Audio + Vision
7. Text + Audio + Vision
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from core.text.model import TextEmotionModel
from core.speech.model import SpeechEmotionModel
from core.vision.model import VisionEmotionModel
from core.fusion.engine import MultiModalFusionEngine

def main():
    print("=" * 60)
    print("TESTING ALL 7 MULTIMODAL COMBINATIONS WITH REAL INFERENCE")
    print("=" * 60)

    text_model = TextEmotionModel()
    speech_model = SpeechEmotionModel()
    vision_model = VisionEmotionModel()
    fusion = MultiModalFusionEngine()

    sample_text = "I am so happy and excited about this breakthrough!"
    sample_audio = os.path.join(PROJECT_ROOT, "Multimodal-Emotion-Recognition-master", "audio_data", "train", "Actor_01", "03-01-01-01-01-01-01.wav")
    sample_image = os.path.join(PROJECT_ROOT, "Multimodal-Emotion-Recognition-master", "image_data", "valid", "Actor_22", "01-01-03-01-01-01-22", "30.jpg")

    # Run individual predictions
    print("\nRunning individual models:")
    res_text = text_model.predict(sample_text)
    print(f"  Text:   emotion={res_text.emotion}, conf={res_text.confidence:.4f}, available={res_text.available}")
    
    res_audio = speech_model.predict(sample_audio)
    print(f"  Audio:  emotion={res_audio.emotion}, conf={res_audio.confidence:.4f}, available={res_audio.available}")

    res_vision = vision_model.predict(sample_image)
    print(f"  Vision: emotion={res_vision.emotion}, conf={res_vision.confidence:.4f}, available={res_vision.available}")
    assert res_vision.available, "Vision model must be available!"

    combinations = [
        ("1. Text only", {"text_result": res_text}),
        ("2. Audio only", {"audio_result": res_audio}),
        ("3. Vision only", {"vision_result": res_vision}),
        ("4. Text + Audio", {"text_result": res_text, "audio_result": res_audio}),
        ("5. Text + Vision", {"text_result": res_text, "vision_result": res_vision}),
        ("6. Audio + Vision", {"audio_result": res_audio, "vision_result": res_vision}),
        ("7. Text + Audio + Vision", {"text_result": res_text, "audio_result": res_audio, "vision_result": res_vision}),
    ]

    for title, kwargs in combinations:
        print(f"\n--- {title} ---")
        fused = fusion.fuse(**kwargs)
        print(f"  Available: {fused.available}")
        print(f"  Emotion: {fused.emotion}")
        print(f"  Confidence: {fused.confidence:.4f}")
        print(f"  Active Modalities: {fused.active_modalities}")
        print(f"  Modality Weights: {fused.explainability.modality_weights}")
        print(f"  Conflict Score: {fused.conflict.conflict_score:.4f} (Has Conflict={fused.conflict.has_conflict})")
        print(f"  Concordance: {fused.explainability.concordance_score:.4f} (Primary={fused.explainability.primary_modality})")
        assert fused.available, f"Fusion failed for {title}"

    print("\n" + "=" * 60)
    print("ALL 7 COMBINATIONS PASSED VERIFICATION")
    print("=" * 60)

if __name__ == "__main__":
    main()
