"""
Vision Facial Emotion Recognition — Experimental Training Harness.
Supports:
- Backbones: AlexNet, ResNet-18, MobileNetV3-Small, MobileNetV3-Large
- Loss: Standard CrossEntropy vs Class-Weighted CrossEntropy
- Augmentation: Basic (RandomHorizontalFlip) vs Realistic (ColorJitter + Rotation + Flip)
- Fine-Tuning: Frozen backbone vs Staged partial unfreezing
- Metrics: Validation Accuracy, Macro F1, Macro Precision, Macro Recall, Neutral Recall, Confusion Matrix
- Model Selection: Best checkpoint saved based on Macro F1 (with Early Stopping)
"""

import os
import sys
import time
import copy
import json
import random
import logging
import argparse
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

sys.path.insert(0, os.path.abspath("."))
from training.vision.dataset import RAVDESSFaceDataset, LABEL_NAMES, NUM_CLASSES
from core.ontology import CORE_EMOTIONS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_experiments")

SEED = 42

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_transforms(aug_mode: str = "realistic") -> Tuple[transforms.Compose, transforms.Compose]:
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if aug_mode == "realistic":
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=7),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    return train_transform, eval_transform

def build_model(
    backbone_name: str,
    finetune_mode: str = "frozen",
) -> Tuple[nn.Module, list]:
    """Build model with pretrained weights and configure trainable parameters."""
    if backbone_name == "alexnet":
        model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        # Freeze conv features
        for param in model.features.parameters():
            param.requires_grad = False
        
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, NUM_CLASSES)
        
        if finetune_mode == "frozen":
            # Only train classifier[6]
            for i in [1, 4]:
                for p in model.classifier[i].parameters():
                    p.requires_grad = False
            param_groups = [{"params": model.classifier[6].parameters(), "lr": 1e-3}]
        else:
            # Unfreeze classifier 1, 4, 6
            for p in model.classifier.parameters():
                p.requires_grad = True
            param_groups = [
                {"params": model.classifier[1].parameters(), "lr": 1e-4},
                {"params": model.classifier[4].parameters(), "lr": 1e-4},
                {"params": model.classifier[6].parameters(), "lr": 1e-3},
            ]

    elif backbone_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Freeze everything initially
        for param in model.parameters():
            param.requires_grad = False
        
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
        model.fc.weight.requires_grad = True
        model.fc.bias.requires_grad = True

        if finetune_mode == "stage2_layer4":
            # Unfreeze layer4 + fc
            for param in model.layer4.parameters():
                param.requires_grad = True
            param_groups = [
                {"params": model.layer4.parameters(), "lr": 1e-4},
                {"params": model.fc.parameters(), "lr": 1e-3},
            ]
        else:
            param_groups = [{"params": model.fc.parameters(), "lr": 1e-3}]

    elif backbone_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, NUM_CLASSES)
        model.classifier[3].weight.requires_grad = True
        model.classifier[3].bias.requires_grad = True

        if finetune_mode == "stage2_layer4":
            # Unfreeze features[11] (final block) + classifier
            for param in model.features[11].parameters():
                param.requires_grad = True
            for param in model.classifier.parameters():
                param.requires_grad = True
            param_groups = [
                {"params": model.features[11].parameters(), "lr": 1e-4},
                {"params": model.classifier[0].parameters(), "lr": 2e-4},
                {"params": model.classifier[3].parameters(), "lr": 1e-3},
            ]
        else:
            param_groups = [{"params": model.classifier[3].parameters(), "lr": 1e-3}]

    elif backbone_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, NUM_CLASSES)
        model.classifier[3].weight.requires_grad = True
        model.classifier[3].bias.requires_grad = True

        if finetune_mode == "stage2_layer4":
            for param in model.features[15].parameters():
                param.requires_grad = True
            for param in model.classifier.parameters():
                param.requires_grad = True
            param_groups = [
                {"params": model.features[15].parameters(), "lr": 1e-4},
                {"params": model.classifier[0].parameters(), "lr": 2e-4},
                {"params": model.classifier[3].parameters(), "lr": 1e-3},
            ]
        else:
            param_groups = [{"params": model.classifier[3].parameters(), "lr": 1e-3}]
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())
    logger.info("Model: %s | Trainable params: %d / %d (%.2f%%)", backbone_name, trainable_count, total_count, trainable_count/total_count*100)
    return model, param_groups

def evaluate(model: nn.Module, dataloader: DataLoader, device: str = "cpu") -> Dict[str, Any]:
    """Run full evaluation and compute accuracy, macro precision, recall, and F1."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs.float())
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    acc = accuracy_score(y_true, y_pred)
    macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    per_r = recall_score(y_true, y_pred, average=None, zero_division=0)
    neutral_idx = 4 # 'neutral'
    neutral_recall = float(per_r[neutral_idx]) if len(per_r) > neutral_idx else 0.0

    cm = confusion_matrix(y_true, y_pred)

    per_class = {}
    for i in range(NUM_CLASSES):
        name = LABEL_NAMES[i]
        per_class[name] = {
            "precision": round(float(precision_score(y_true == i, y_pred == i, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true == i, y_pred == i, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true == i, y_pred == i, zero_division=0)), 4),
            "support": int((y_true == i).sum()),
        }

    return {
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "neutral_recall": round(neutral_recall, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }

def train_experiment(
    exp_name: str,
    backbone_name: str,
    loss_mode: str = "weighted_ce",
    aug_mode: str = "realistic",
    finetune_mode: str = "stage2_layer4",
    num_epochs: int = 12,
    batch_size: int = 64,
    patience: int = 4,
    device: str = "cpu",
) -> Dict[str, Any]:
    set_seed(SEED)
    logger.info("=" * 65)
    logger.info("STARTING EXPERIMENT: %s", exp_name)
    logger.info("Backbone: %s | Loss: %s | Aug: %s | Finetune: %s", backbone_name, loss_mode, aug_mode, finetune_mode)
    logger.info("=" * 65)

    data_root = os.path.join("Multimodal-Emotion-Recognition-master", "image_data")
    train_dir = os.path.join(data_root, "train")
    valid_dir = os.path.join(data_root, "valid")

    train_tf, eval_tf = get_transforms(aug_mode=aug_mode)

    train_ds = RAVDESSFaceDataset(train_dir, transform=train_tf, preload=False)
    valid_ds = RAVDESSFaceDataset(valid_dir, transform=eval_tf, preload=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Class weights calculation
    train_dist = train_ds.get_class_distribution()
    total_samples = len(train_ds)
    class_counts = [train_dist[LABEL_NAMES[i]] for i in range(NUM_CLASSES)]
    
    if loss_mode == "weighted_ce":
        weights = [total_samples / (NUM_CLASSES * max(1, c)) for c in class_counts]
        w_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
        # Moderate class weights slightly (power of 0.8) to prevent over-penalizing neutral
        w_tensor = torch.pow(w_tensor, 0.8)
        w_tensor = w_tensor / w_tensor.mean()
        criterion = nn.CrossEntropyLoss(weight=w_tensor)
        logger.info("Using weighted CrossEntropyLoss: %s", [round(float(w), 3) for w in w_tensor])
    else:
        criterion = nn.CrossEntropyLoss()
        logger.info("Using unweighted CrossEntropyLoss")

    model, param_groups = build_model(backbone_name, finetune_mode=finetune_mode)
    model = model.to(device)

    optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_macro_f1 = 0.0
    best_acc = 0.0
    best_metrics = None
    best_state_dict = None
    no_improve_epochs = 0

    checkpoint_dir = os.path.join("models", "vision")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_ckpt_path = os.path.join(checkpoint_dir, f"{exp_name}_best.pth")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        running_correct = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()

        scheduler.step()
        epoch_train_loss = running_loss / len(train_ds)
        epoch_train_acc = running_correct / len(train_ds)

        # Validation evaluation
        metrics = evaluate(model, valid_loader, device=device)
        dt = time.time() - t0

        logger.info(
            "Epoch %2d/%2d [%.1fs] | Train Loss: %.4f Acc: %.2f%% | Val Acc: %.2f%% | Val Macro F1: %.4f | Neutral Rec: %.1f%%",
            epoch, num_epochs, dt, epoch_train_loss, epoch_train_acc * 100,
            metrics["accuracy"] * 100, metrics["macro_f1"], metrics["neutral_recall"] * 100
        )

        # Primary selection metric: Macro F1
        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_acc = metrics["accuracy"]
            best_metrics = metrics
            best_state_dict = copy.deepcopy(model.state_dict())
            no_improve_epochs = 0

            # Export in standard inference wrapper format (with Softmax)
            # Save raw model weights and inference metadata
            torch.save({
                "backbone": backbone_name,
                "state_dict": best_state_dict,
                "metrics": best_metrics,
                "epoch": epoch,
                "exp_name": exp_name,
            }, best_ckpt_path)
            logger.info("  >>> NEW BEST MODEL SAVED! Macro F1: %.4f, Acc: %.2f%% (saved to %s)", best_macro_f1, best_acc * 100, best_ckpt_path)
        else:
            no_improve_epochs += 1

        if patience and no_improve_epochs >= patience:
            logger.info("Early stopping triggered at epoch %d (no Macro F1 improvement for %d epochs)", epoch, patience)
            break

    logger.info("=" * 65)
    logger.info("EXPERIMENT %s FINISHED | Best Macro F1: %.4f | Best Acc: %.2f%%", exp_name, best_macro_f1, best_acc * 100)
    logger.info("=" * 65)

    return {
        "exp_name": exp_name,
        "backbone": backbone_name,
        "finetune": finetune_mode,
        "loss": loss_mode,
        "aug": aug_mode,
        "best_macro_f1": best_macro_f1,
        "best_accuracy": best_acc,
        "neutral_recall": best_metrics["neutral_recall"] if best_metrics else 0.0,
        "metrics": best_metrics,
        "checkpoint_path": best_ckpt_path,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="exp_test", help="Experiment name")
    parser.add_argument("--backbone", type=str, default="mobilenet_v3_small", choices=["alexnet", "resnet18", "mobilenet_v3_small", "mobilenet_v3_large"])
    parser.add_argument("--loss", type=str, default="weighted_ce", choices=["ce", "weighted_ce"])
    parser.add_argument("--aug", type=str, default="realistic", choices=["basic", "realistic"])
    parser.add_argument("--finetune", type=str, default="stage2_layer4", choices=["frozen", "stage2_layer4"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(8)
    res = train_experiment(
        exp_name=args.name,
        backbone_name=args.backbone,
        loss_mode=args.loss,
        aug_mode=args.aug,
        finetune_mode=args.finetune,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
    )
    print("\nFINAL RESULT:", json.dumps({k: v for k, v in res.items() if k != "metrics"}, indent=2))
