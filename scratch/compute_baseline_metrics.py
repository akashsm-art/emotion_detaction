"""
Compute comprehensive baseline metrics for the existing AlexNet checkpoint.
"""

import os
import sys
import json
import numpy as np
import torch
import torchvision.models as models
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

sys.path.insert(0, ".")
from core.ontology import CORE_EMOTIONS
from training.vision.dataset import RAVDESSFaceDataset, get_eval_transform, LABEL_NAMES

def main():
    model = models.alexnet(weights=None)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Sequential(
        nn.Linear(num_ftrs, 7),
        nn.Softmax(dim=1),
    )
    sd = torch.load("models/vision/model_image_new50.pth", map_location="cpu")
    model.load_state_dict(sd)
    model.eval()

    data_root = os.path.join("Multimodal-Emotion-Recognition-master", "image_data")
    valid_ds = RAVDESSFaceDataset(os.path.join(data_root, "valid"), preload=False)

    eval_tf = get_eval_transform()
    all_preds = []
    all_labels = []

    print("Evaluating 2,621 validation samples...")
    with torch.no_grad():
        for fpath, label in valid_ds.samples:
            img = Image.open(fpath).convert("RGB")
            t = eval_tf(img).unsqueeze(0)
            out = model(t).squeeze().numpy()
            pred = int(out.argmax())
            all_preds.append(pred)
            all_labels.append(label)

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    acc = accuracy_score(y_true, y_pred)
    macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    per_p = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_r = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    report = {
        "architecture": "AlexNet (torchvision pretrained backbone)",
        "checkpoint": "models/vision/model_image_new50.pth",
        "validation_samples": int(len(y_true)),
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": {},
        "confusion_matrix": cm.tolist(),
        "class_order": [LABEL_NAMES[i] for i in range(7)],
    }

    print("=" * 65)
    print("BASELINE AUDIT REPORT — ALEXNET (models/vision/model_image_new50.pth)")
    print("=" * 65)
    print(f"Overall Accuracy:    {acc * 100:.2f}%")
    print(f"Macro Precision:     {macro_p:.4f}")
    print(f"Macro Recall:        {macro_r:.4f}")
    print(f"Macro F1:            {macro_f1:.4f}")
    print(f"Weighted F1:         {weighted_f1:.4f}")
    print("-" * 65)
    header_cols = f"{'Emotion':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}"
    print(header_cols)
    print("-" * 65)
    for i in range(7):
        name = LABEL_NAMES[i]
        sup = int((y_true == i).sum())
        report["per_class"][name] = {
            "precision": round(float(per_p[i]), 4),
            "recall": round(float(per_r[i]), 4),
            "f1": round(float(per_f1[i]), 4),
            "support": sup,
        }
        print(f"{name:<10} {per_p[i]*100:6.2f}%    {per_r[i]*100:6.2f}%    {per_f1[i]*100:6.2f}%    {sup:<8}")

    print("-" * 65)
    print("Confusion Matrix (Rows=True, Cols=Pred):")
    cm_header = "       " + "".join(f"{LABEL_NAMES[i][:5]:>7}" for i in range(7))
    print(cm_header)
    for i in range(7):
        row = f"{LABEL_NAMES[i][:6]:<7}" + "".join(f"{cm[i][j]:>7}" for j in range(7))
        print(row)
    print("=" * 65)

    os.makedirs("reports", exist_ok=True)
    with open("reports/vision_baseline_metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Saved baseline metrics to reports/vision_baseline_metrics.json")

if __name__ == "__main__":
    main()
