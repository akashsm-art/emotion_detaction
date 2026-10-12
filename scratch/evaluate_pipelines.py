"""
Evaluation Script for Vision Pipelines (Step 7 and Step 8).
Compares Direct/Original Preprocessing vs Haar Crop Preprocessing
across known labeled validation images from Actor_21 to Actor_24.
"""

import os
import sys
import cv2
import numpy as np
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from collections import defaultdict

sys.path.insert(0, os.path.abspath("."))
from core.ontology import CORE_EMOTIONS, normalize_to_core_ontology
from core.vision.face_detector import FaceDetector
from core.vision.model import VisionEmotionModel
from training.vision.dataset import RAVDESS_CODE_TO_LABEL, LABEL_NAMES

def load_alexnet():
    model = models.alexnet(weights=None)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Sequential(
        nn.Linear(num_ftrs, len(CORE_EMOTIONS)),
        nn.Softmax(dim=1),
    )
    sd = torch.load("models/vision/model_image_new50.pth", map_location="cpu")
    model.load_state_dict(sd)
    model.eval()
    return model

def original_pipeline(img_path, model):
    """Original validation transform: PIL Image -> ToTensor -> Normalize."""
    img = Image.open(img_path).convert("RGB")
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = eval_transform(img).unsqueeze(0)
    with torch.no_grad():
        out = model(tensor).squeeze().numpy()
    pred_idx = int(np.argmax(out))
    pred_emotion = CORE_EMOTIONS[pred_idx]
    confidence = float(out[pred_idx])
    return pred_emotion, confidence, out

def live_pipeline(img_path, model, detector):
    """Live pipeline: OpenCV BGR -> RGB -> Haar detect & crop -> resize 224x224 -> Normalize."""
    bgr = cv2.imread(img_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    crops, boxes = detector.detect_and_crop_faces(rgb, fallback_if_no_face=True)
    crop = crops[0] # primary crop
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = torch.from_numpy(crop).float().permute(2, 0, 1) / 255.0
    tensor = (tensor - mean) / std
    tensor = tensor.unsqueeze(0)
    
    with torch.no_grad():
        out = model(tensor).squeeze().numpy()
    pred_idx = int(np.argmax(out))
    pred_emotion = CORE_EMOTIONS[pred_idx]
    confidence = float(out[pred_idx])
    return pred_emotion, confidence, out, boxes

def run_evaluation():
    model = load_alexnet()
    detector = FaceDetector()
    vision_model = VisionEmotionModel()

    valid_root = "Multimodal-Emotion-Recognition-master/image_data/valid"
    
    # Collect balanced sample of labeled images (at least 4 images per emotion = 28+ images)
    emotion_samples = defaultdict(list)
    for actor in sorted(os.listdir(valid_root)):
        if not actor.startswith("Actor_"):
            continue
        actor_dir = os.path.join(valid_root, actor)
        for video in sorted(os.listdir(actor_dir)):
            parts = video.split("-")
            if len(parts) < 3:
                continue
            code = parts[2]
            label_idx = RAVDESS_CODE_TO_LABEL.get(code)
            if label_idx is None:
                continue
            emotion_name = LABEL_NAMES[label_idx]
            video_dir = os.path.join(actor_dir, video)
            for f in sorted(os.listdir(video_dir)):
                if f.endswith(".jpg") and os.path.getsize(os.path.join(video_dir, f)) > 0:
                    emotion_samples[emotion_name].append(os.path.join(video_dir, f))

    print("=" * 60)
    print("STEP 7 — CONTROLLED SINGLE IMAGE TEST ACROSS ALL 7 EMOTIONS")
    print("=" * 60)
    
    selected_samples = []
    for emotion in CORE_EMOTIONS:
        samples = emotion_samples[emotion]
        if samples:
            # Pick one representative frame (middle frame around 50.jpg or 70.jpg)
            mid = samples[len(samples) // 2]
            selected_samples.append((emotion, mid))

    for gt_emotion, img_path in selected_samples:
        orig_em, orig_conf, _ = original_pipeline(img_path, model)
        live_em, live_conf, _, boxes = live_pipeline(img_path, model, detector)
        core_res = vision_model.predict(img_path)
        
        print(f"\nImage: {os.path.basename(os.path.dirname(img_path))}/{os.path.basename(img_path)}")
        print(f"Ground Truth:   {gt_emotion.upper()}")
        print(f"Original Eval:  {orig_em.upper():<10} ({orig_conf*100:.1f}%) {'MATCH' if orig_em == gt_emotion else 'DIFF'}")
        print(f"Live Pipeline:  {live_em.upper():<10} ({live_conf*100:.1f}%) {'MATCH' if live_em == gt_emotion else 'DIFF'}")
        print(f"VisionModel:    {core_res.emotion.upper():<10} ({core_res.confidence*100:.1f}%) {'MATCH' if core_res.emotion == gt_emotion else 'DIFF'}")

    print("\n" + "=" * 60)
    print("STEP 8 — BENCHMARK EVALUATION (40 VALIDATION IMAGES ACROSS ACTORS 21-24)")
    print("=" * 60)
    
    test_set = []
    for emotion in CORE_EMOTIONS:
        # Take 6 samples per emotion across actors = 42 images
        samples = emotion_samples[emotion]
        step = max(1, len(samples) // 6)
        for i in range(0, min(len(samples), 6 * step), step):
            test_set.append((emotion, samples[i]))
            if len([s for s in test_set if s[0] == emotion]) >= 6:
                break

    orig_correct = 0
    live_correct = 0
    total = len(test_set)
    
    orig_conf_matrix = defaultdict(lambda: defaultdict(int))
    live_conf_matrix = defaultdict(lambda: defaultdict(int))
    
    orig_preds = []
    live_preds = []

    for gt, img_path in test_set:
        orig_em, _, _ = original_pipeline(img_path, model)
        live_em, _, _, _ = live_pipeline(img_path, model, detector)
        
        orig_preds.append((gt, orig_em))
        live_preds.append((gt, live_em))
        
        orig_conf_matrix[gt][orig_em] += 1
        live_conf_matrix[gt][live_em] += 1
        
        if orig_em == gt:
            orig_correct += 1
        if live_em == gt:
            live_correct += 1

    print(f"\nTotal Test Samples: {total}")
    print(f"Original Pipeline Accuracy: {orig_correct}/{total} ({orig_correct/total * 100:.2f}%)")
    print(f"Live Pipeline Accuracy:     {live_correct}/{total} ({live_correct/total * 100:.2f}%)")
    
    print("\nOriginal Pipeline Per-Class Accuracy:")
    for em in CORE_EMOTIONS:
        c = orig_conf_matrix[em][em]
        tot = sum(orig_conf_matrix[em].values())
        pct = (c / tot * 100) if tot > 0 else 0
        print(f"  {em:<10}: {c}/{tot} ({pct:.1f}%)")

    print("\nLive Pipeline Per-Class Accuracy:")
    for em in CORE_EMOTIONS:
        c = live_conf_matrix[em][em]
        tot = sum(live_conf_matrix[em].values())
        pct = (c / tot * 100) if tot > 0 else 0
        print(f"  {em:<10}: {c}/{tot} ({pct:.1f}%)")

    print("\nConfusion Matrix (Original Pipeline): Rows = Ground Truth, Cols = Predicted")
    header = "       " + "".join(f"{e[:5]:>7}" for e in CORE_EMOTIONS)
    print(header)
    for gt in CORE_EMOTIONS:
        row = f"{gt[:6]:<7}" + "".join(f"{orig_conf_matrix[gt][pred]:>7}" for pred in CORE_EMOTIONS)
        print(row)

    print("\nConfusion Matrix (Live Pipeline): Rows = Ground Truth, Cols = Predicted")
    print(header)
    for gt in CORE_EMOTIONS:
        row = f"{gt[:6]:<7}" + "".join(f"{live_conf_matrix[gt][pred]:>7}" for pred in CORE_EMOTIONS)
        print(row)

    print("=" * 60)

if __name__ == "__main__":
    run_evaluation()
