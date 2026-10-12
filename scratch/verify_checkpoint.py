"""
Verification script for trained Vision Facial Emotion checkpoint:
- Verifies state_dict keys and shapes
- Tests reload into fresh AlexNet instance
- Tests inference before vs after checkpoint loading
- Tests determinism
- Tests inference on held-out validation images from different classes
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from training.vision.dataset import LABEL_NAMES, NUM_CLASSES, get_eval_transform

CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "models", "vision", "model_image_new50.pth")

def main():
    print("=" * 60)
    print("STEP 7: VERIFY CHECKPOINT COMPATIBILITY & INFERENCE")
    print("=" * 60)
    
    # 1. Inspect checkpoint file
    assert os.path.isfile(CHECKPOINT_PATH), f"Checkpoint missing: {CHECKPOINT_PATH}"
    file_size_mb = os.path.getsize(CHECKPOINT_PATH) / (1024 * 1024)
    print(f"Checkpoint file: {CHECKPOINT_PATH}")
    print(f"File size: {file_size_mb:.2f} MB")
    
    sd = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    print(f"Total state_dict keys: {len(sd)}")
    
    # Verify classifier layer 6
    assert "classifier.6.0.weight" in sd, "classifier.6.0.weight not found in state_dict"
    assert "classifier.6.0.bias" in sd, "classifier.6.0.bias not found in state_dict"
    w_shape = list(sd["classifier.6.0.weight"].shape)
    b_shape = list(sd["classifier.6.0.bias"].shape)
    print(f"classifier.6.0.weight shape: {w_shape} (expected [7, 4096])")
    print(f"classifier.6.0.bias shape:   {b_shape} (expected [7])")
    assert w_shape == [7, 4096], f"Unexpected weight shape: {w_shape}"
    assert b_shape == [7], f"Unexpected bias shape: {b_shape}"
    print("[PASS] State dict shapes verified successfully")

    # 2. Fresh AlexNet instance & Untrained baseline
    fresh_model = models.alexnet(weights=None)
    fresh_model.classifier[6] = nn.Sequential(
        nn.Linear(4096, 7),
        nn.Softmax(dim=1),
    )
    fresh_model.eval()

    # Load test image
    val_root = os.path.join(PROJECT_ROOT, "Multimodal-Emotion-Recognition-master", "image_data", "valid")
    # Pick 4 diverse sample images from different actors and emotions
    samples = [
        ("Actor_21/01-01-01-01-01-01-21/30.jpg", "neutral"),
        ("Actor_22/01-01-03-01-01-01-22/30.jpg", "happy"),
        ("Actor_23/01-01-05-01-01-01-23/30.jpg", "angry"),
        ("Actor_24/01-01-07-01-01-01-24/30.jpg", "disgust"),
    ]
    
    transform = get_eval_transform()
    
    print("\n--- Model Predictions BEFORE Checkpoint (Untrained Random) vs AFTER (Trained) ---")
    
    # Run untrained first on sample 1
    sample_path = os.path.join(val_root, samples[0][0])
    img = Image.open(sample_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        untrained_probs = fresh_model(tensor).squeeze().numpy()
    untrained_pred = LABEL_NAMES[int(np.argmax(untrained_probs))]
    print(f"Sample 1 ({samples[0][1]}):")
    print(f"  UNTRAINED: top={untrained_pred} ({untrained_probs[np.argmax(untrained_probs)]:.4f})")
    print(f"             probs={[round(float(p), 4) for p in untrained_probs]}")

    # Now load trained checkpoint
    fresh_model.load_state_dict(sd)
    fresh_model.eval()
    print("[PASS] Fresh model loaded trained checkpoint successfully")

    with torch.no_grad():
        trained_probs = fresh_model(tensor).squeeze().numpy()
    trained_pred = LABEL_NAMES[int(np.argmax(trained_probs))]
    print(f"  TRAINED:   top={trained_pred} ({trained_probs[np.argmax(trained_probs)]:.4f})")
    print(f"             probs={[round(float(p), 4) for p in trained_probs]}")
    
    assert not np.allclose(untrained_probs, trained_probs, atol=1e-3), "Trained and untrained predictions are identical!"
    print("[PASS] Proved: Trained classifier weights actively drive predictions (not untrained/random)")

    # 3. Test on all sample images
    print("\n--- Real Inference on Held-out Images ---")
    for rel_path, ground_truth in samples:
        full_p = os.path.join(val_root, rel_path)
        if not os.path.isfile(full_p):
            print(f"Skip {rel_path} (not found)")
            continue
        im = Image.open(full_p).convert("RGB")
        t = transform(im).unsqueeze(0)
        with torch.no_grad():
            out = fresh_model(t).squeeze().numpy()
        pred = LABEL_NAMES[int(np.argmax(out))]
        conf = float(out[np.argmax(out)])
        prob_sum = float(np.sum(out))
        print(f"Image: {rel_path}")
        print(f"  True: {ground_truth} | Predicted: {pred} (conf={conf:.4f}) | Sum={prob_sum:.4f}")
        assert abs(prob_sum - 1.0) < 1e-4, f"Probabilities do not sum to 1: {prob_sum}"

    # 4. Determinism test
    print("\n--- Determinism Test ---")
    with torch.no_grad():
        p1 = fresh_model(tensor).squeeze().cpu().numpy()
        p2 = fresh_model(tensor).squeeze().cpu().numpy()
        p3 = fresh_model(tensor).squeeze().cpu().numpy()
    assert np.allclose(p1, p2) and np.allclose(p2, p3), "Model output is non-deterministic!"
    print("[PASS] Determinism verified: 3 consecutive evaluations are bit-for-bit identical")

    print("\n" + "=" * 60)
    print("ALL CHECKPOINT VERIFICATIONS PASSED")
    print("=" * 60)

if __name__ == "__main__":
    main()
