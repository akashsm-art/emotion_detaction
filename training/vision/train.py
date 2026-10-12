"""Train the existing AlexNet face-emotion model and export its inference format."""

import argparse
import os
import sys
import time
import copy
import json
import logging
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision.models as models

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from training.vision.dataset import (
    RAVDESSFaceDataset,
    get_train_transform,
    get_eval_transform,
    NUM_CLASSES,
    LABEL_NAMES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Reproducibility ────────────────────────────────────────────────────────
SEED = 42

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─── Model Construction ─────────────────────────────────────────────────────

def build_training_model(
    num_classes: int = NUM_CLASSES,
    initial_checkpoint: str = None,
    freeze_features: bool = True,
    fine_tune_depth: str = "conv5",
):
    """
    Build the same AlexNet and optionally warm-start it from an inference checkpoint.
    Only the final convolution block and selected classifier tensors are fine-tuned.

    For TRAINING we use raw logits → CrossEntropyLoss.
    After training, we wrap with Softmax for inference compatibility.
    """
    use_pretrained = not (initial_checkpoint and os.path.isfile(initial_checkpoint))
    model = models.alexnet(
        weights=models.AlexNet_Weights.DEFAULT if use_pretrained else None
    )

    if freeze_features:
        for param in model.parameters():
            param.requires_grad = False

    # Replace final layer with 7-class output (NO Softmax for training)
    num_ftrs = model.classifier[6].in_features  # 4096
    model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    if not use_pretrained:
        state_dict = torch.load(initial_checkpoint, map_location="cpu", weights_only=True)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        state_dict = dict(state_dict)
        state_dict["classifier.6.weight"] = state_dict.pop("classifier.6.0.weight")
        state_dict["classifier.6.bias"] = state_dict.pop("classifier.6.0.bias")
        model.load_state_dict(state_dict)
        logger.info("Warm-started AlexNet from %s", initial_checkpoint)

    if fine_tune_depth not in {"classifier_only", "conv5", "conv4_conv5"}:
        raise ValueError(f"Unsupported fine-tuning depth: {fine_tune_depth}")

    # Keep the current recipe at conv5; broader depth is opt-in for ablations.
    # classifier.1.bias, classifier.4.weight, classifier.4.bias,
    # classifier.6.weight, classifier.6.bias
    trainable_names = []
    for name, param in model.named_parameters():
        trainable = (
            name == "classifier.1.bias"
            or name.startswith("classifier.4.")
            or name.startswith("classifier.6.")
            or (fine_tune_depth in {"conv5", "conv4_conv5"} and name.startswith("features.10."))
            or (fine_tune_depth == "conv4_conv5" and name.startswith("features.8."))
        )
        param.requires_grad = trainable
        if trainable:
            trainable_names.append(name)

    logger.info("Trainable parameters: %s", trainable_names)
    return model, trainable_names


def macro_f1_score(predictions, labels) -> float:
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    scores = []
    for class_index in range(NUM_CLASSES):
        true_positive = np.sum((predictions == class_index) & (labels == class_index))
        false_positive = np.sum((predictions == class_index) & (labels != class_index))
        false_negative = np.sum((predictions != class_index) & (labels == class_index))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores))


def per_class_f1(predictions, labels) -> dict:
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    scores = {}
    for class_index in range(NUM_CLASSES):
        true_positive = np.sum((predictions == class_index) & (labels == class_index))
        false_positive = np.sum((predictions == class_index) & (labels != class_index))
        false_negative = np.sum((predictions != class_index) & (labels == class_index))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        scores[LABEL_NAMES[class_index]] = round(
            float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
            4,
        )
    return scores


def convert_to_inference_model(state_dict: dict, num_classes: int = NUM_CLASSES):
    """
    Load state_dict into AlexNet with nn.Sequential(Linear, Softmax) at classifier[6]
    to match the expected inference architecture in core/vision/model.py.
    """
    model = models.alexnet(weights=None)
    num_ftrs = model.classifier[6].in_features

    # Build inference architecture with Softmax
    model.classifier[6] = nn.Sequential(
        nn.Linear(num_ftrs, num_classes),
        nn.Softmax(dim=1),
    )

    # Map training state_dict keys to inference keys
    # Training: classifier.6.weight → Inference: classifier.6.0.weight
    inference_sd = {}
    for k, v in state_dict.items():
        if k == "classifier.6.weight":
            inference_sd["classifier.6.0.weight"] = v
        elif k == "classifier.6.bias":
            inference_sd["classifier.6.0.bias"] = v
        else:
            inference_sd[k] = v

    model.load_state_dict(inference_sd)
    return model


# ─── Training Loop ───────────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    dataloaders: dict,
    dataset_sizes: dict,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    num_epochs: int = 50,
    device: str = "cpu",
    checkpoint_dir: str = "models/vision",
    patience: int = 15,
):
    """
    Select the best weights by held-out validation macro-F1 with early stopping.
    """
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_macro_f1 = -1.0
    epochs_without_improvement = 0

    history = {
        "train_loss": [], "train_acc": [],
        "valid_loss": [], "valid_acc": [],
        "valid_macro_f1": [],
        "valid_per_class_f1": [],
        "best_epoch": None,
    }

    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        logger.info("Epoch %d/%d", epoch, num_epochs)
        logger.info("-" * 40)

        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            phase_predictions = []
            phase_labels = []

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs.float())
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                if phase == "valid":
                    phase_predictions.extend(preds.detach().cpu().tolist())
                    phase_labels.extend(labels.detach().cpu().tolist())

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            epoch_macro_f1 = macro_f1_score(phase_predictions, phase_labels) if phase == "valid" else None

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc)
            if phase == "valid":
                history["valid_macro_f1"].append(epoch_macro_f1)
                history["valid_per_class_f1"].append(
                    per_class_f1(phase_predictions, phase_labels)
                )

            if phase == "valid":
                logger.info(
                    "%s Loss: %.4f  Acc: %.4f  Macro-F1: %.4f",
                    phase.upper(), epoch_loss, epoch_acc, epoch_macro_f1,
                )
            else:
                logger.info("%s Loss: %.4f  Acc: %.4f", phase.upper(), epoch_loss, epoch_acc)

            if phase == "valid":
                if epoch_macro_f1 > best_macro_f1:
                    best_acc = epoch_acc
                    best_macro_f1 = epoch_macro_f1
                    history["best_epoch"] = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_without_improvement = 0
                    logger.info(
                        "*** New best validation macro-F1: %.4f (accuracy %.4f) ***",
                        best_macro_f1,
                        best_acc,
                    )
                else:
                    epochs_without_improvement += 1

        # Save periodic checkpoints (matching original: every 10 epochs)
        if epoch % 10 == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"model_image_new{epoch}.pth")
            torch.save(best_model_wts, ckpt_path)
            logger.info("Periodic checkpoint saved: %s", ckpt_path)

        # Early stopping
        if patience and epochs_without_improvement >= patience:
            logger.info(
                "Early stopping at epoch %d (no improvement for %d epochs)",
                epoch, patience,
            )
            break

        logger.info("")

    elapsed = time.time() - since
    logger.info("Training complete in %dm %ds", elapsed // 60, elapsed % 60)
    logger.info("Best validation accuracy: %.4f", best_acc)

    # Load best weights
    model.load_state_dict(best_model_wts)
    return model, history, best_acc


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_model(model: nn.Module, dataloader, device: str = "cpu"):
    """
    Evaluate model on a dataset and return per-class metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs.float())
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    total = len(all_labels)
    correct = (all_preds == all_labels).sum()
    accuracy = correct / total

    # Per-class metrics
    per_class = {}
    for cls_idx in range(NUM_CLASSES):
        cls_name = LABEL_NAMES[cls_idx]
        tp = ((all_preds == cls_idx) & (all_labels == cls_idx)).sum()
        fp = ((all_preds == cls_idx) & (all_labels != cls_idx)).sum()
        fn = ((all_preds != cls_idx) & (all_labels == cls_idx)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[cls_name] = {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "support": int((all_labels == cls_idx).sum()),
        }

    # Confusion matrix
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for true, pred in zip(all_labels, all_preds):
        confusion[true][pred] += 1

    return {
        "accuracy": round(float(accuracy), 4),
        "macro_f1": round(float(np.mean([metrics["f1"] for metrics in per_class.values()])), 4),
        "total_samples": total,
        "correct": int(correct),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "class_names": [LABEL_NAMES[i] for i in range(NUM_CLASSES)],
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main(
    checkpoint_dir: str = None,
    initial_checkpoint: str = None,
    class_weight_strength: float = 0.5,
    frame_strategy: str = "all",
    fine_tune_depth: str = "conv5",
    label_smoothing: float = 0.0,
    experiment_name: str = "current_recipe",
):
    set_seed(SEED)
    torch.set_num_threads(8)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    # Dataset paths
    data_root = os.path.join(
        PROJECT_ROOT,
        "Multimodal-Emotion-Recognition-master",
        "image_data",
    )
    train_dir = os.path.join(data_root, "train")
    valid_dir = os.path.join(data_root, "valid")

    # Build datasets
    logger.info("Loading training dataset...")
    frames_per_clip = {"all": None, "representative_3": 3}.get(frame_strategy)
    if frame_strategy not in {"all", "representative_3"}:
        raise ValueError(f"Unsupported frame strategy: {frame_strategy}")
    train_dataset = RAVDESSFaceDataset(
        train_dir,
        transform=get_train_transform(),
        frames_per_clip=frames_per_clip,
    )
    logger.info("Training distribution: %s", train_dataset.get_class_distribution())

    logger.info("Loading validation dataset...")
    valid_dataset = RAVDESSFaceDataset(valid_dir, transform=get_eval_transform())
    logger.info("Validation distribution: %s", valid_dataset.get_class_distribution())

    # DataLoaders — batch_size=64 for faster CPU training
    # (Original notebook used batch_size=4 on GPU; we scale up for CPU throughput)
    BATCH_SIZE = 64
    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
        "valid": DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
    }
    dataset_sizes = {
        "train": len(train_dataset),
        "valid": len(valid_dataset),
    }

    logger.info("Dataset sizes: train=%d, valid=%d", dataset_sizes["train"], dataset_sizes["valid"])

    # Warm-start the existing checkpoint and adapt only the final convolution block.
    initial_checkpoint = initial_checkpoint or os.path.join(
        PROJECT_ROOT, "models", "vision", "model_image_new50.pth"
    )
    if not os.path.isabs(initial_checkpoint):
        initial_checkpoint = os.path.join(PROJECT_ROOT, initial_checkpoint)
    model, trainable_names = build_training_model(
        num_classes=NUM_CLASSES,
        initial_checkpoint=initial_checkpoint,
        freeze_features=True,
        fine_tune_depth=fine_tune_depth,
    )
    model = model.to(device)

    if class_weight_strength < 0:
        raise ValueError("class_weight_strength must be non-negative")

    # A zero strength is an unweighted-loss ablation; 0.5 is the current mild setting.
    class_counts = train_dataset.get_full_class_distribution()
    weights = torch.tensor(
        [
            (max(class_counts.values()) / class_counts[LABEL_NAMES[i]]) ** class_weight_strength
            for i in range(NUM_CLASSES)
        ],
        dtype=torch.float32,
        device=device,
    )
    weights = weights / weights.mean()
    logger.info(
        "Class-weight strength %.3f; normalized weights: %s",
        class_weight_strength,
        [round(float(weight), 3) for weight in weights],
    )
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1)")
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)

    named_parameters = dict(model.named_parameters())
    optimizer_groups = []
    feature_prefixes = ("features.10.",) if fine_tune_depth == "conv5" else (
        ("features.8.", "features.10.") if fine_tune_depth == "conv4_conv5" else ()
    )
    feature_params = [
        p for n, p in named_parameters.items()
        if any(n.startswith(prefix) for prefix in feature_prefixes)
    ]
    if feature_params:
        optimizer_groups.append({"params": feature_params, "lr": 2e-5})
    optimizer_groups.extend([
        {"params": [p for n, p in named_parameters.items() if n.startswith("classifier.1.") or n.startswith("classifier.4.")], "lr": 1e-4},
        {"params": [p for n, p in named_parameters.items() if n.startswith("classifier.6.")], "lr": 5e-4},
    ])
    optimizer = optim.AdamW(optimizer_groups, weight_decay=1e-4)
    MAX_EPOCHS = 20
    PATIENCE = 5
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)

    logger.info("Starting training...")
    checkpoint_dir = os.path.abspath(
        checkpoint_dir or os.path.join(PROJECT_ROOT, "models", "vision")
    )
    model, history, best_acc = train_model(
        model=model,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=MAX_EPOCHS,
        device=device,
        checkpoint_dir=checkpoint_dir,
        patience=PATIENCE,
    )

    # ── Save best checkpoint in INFERENCE format ──
    # Convert training model (Linear) → inference model (Sequential(Linear, Softmax))
    logger.info("Converting to inference model format...")
    training_state_dict = model.state_dict()
    inference_model = convert_to_inference_model(training_state_dict, NUM_CLASSES)
    inference_model.eval()

    final_path = os.path.join(checkpoint_dir, "model_image_new50.pth")
    torch.save(inference_model.state_dict(), final_path)
    logger.info("Final inference checkpoint saved: %s", final_path)
    logger.info("File size: %.2f MB", os.path.getsize(final_path) / (1024 * 1024))

    # ── Verify checkpoint architecture ──
    sd = torch.load(final_path, map_location="cpu", weights_only=True)
    assert "classifier.6.0.weight" in sd, "Missing classifier.6.0.weight"
    assert sd["classifier.6.0.weight"].shape == torch.Size([7, 4096]), \
        f"Wrong weight shape: {sd['classifier.6.0.weight'].shape}"
    assert sd["classifier.6.0.bias"].shape == torch.Size([7]), \
        f"Wrong bias shape: {sd['classifier.6.0.bias'].shape}"
    logger.info("✓ Checkpoint architecture verified: classifier.6.0.weight=[7,4096], bias=[7]")

    # ── Evaluate on validation set ──
    logger.info("Running final evaluation on validation set...")
    # Reload with eval transform (no augmentation)
    eval_dataset = RAVDESSFaceDataset(valid_dir, transform=get_eval_transform())
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Use the inference model for evaluation
    inference_model = inference_model.to(device)
    eval_results = evaluate_model(inference_model, eval_loader, device)

    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS (Validation Set)")
    logger.info("=" * 60)
    logger.info("Overall accuracy: %.4f (%d/%d)", eval_results["accuracy"], eval_results["correct"], eval_results["total_samples"])
    logger.info("")
    logger.info("%-12s  %8s  %8s  %8s  %8s", "CLASS", "PREC", "RECALL", "F1", "SUPPORT")
    logger.info("-" * 52)
    for cls_name in [LABEL_NAMES[i] for i in range(NUM_CLASSES)]:
        m = eval_results["per_class"][cls_name]
        logger.info("%-12s  %8.4f  %8.4f  %8.4f  %8d", cls_name, m["precision"], m["recall"], m["f1"], m["support"])

    logger.info("")
    logger.info("Confusion Matrix:")
    header = "          " + "  ".join(f"{LABEL_NAMES[i][:4]:>5s}" for i in range(NUM_CLASSES))
    logger.info(header)
    for i in range(NUM_CLASSES):
        row = f"{LABEL_NAMES[i][:4]:>8s}  " + "  ".join(f"{v:5d}" for v in eval_results["confusion_matrix"][i])
        logger.info(row)

    # ── Save evaluation results ──
    results_path = os.path.join(checkpoint_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "best_validation_accuracy_during_training": round(best_acc, 4),
            "best_validation_macro_f1_during_training": round(max(history["valid_macro_f1"]), 4),
            "best_epoch": history["best_epoch"],
            "final_evaluation": eval_results,
            "training_history": {
                "train_loss": history["train_loss"],
                "train_acc": history["train_acc"],
                "valid_loss": history["valid_loss"],
                "valid_acc": history["valid_acc"],
                "valid_macro_f1": history["valid_macro_f1"],
                "valid_per_class_f1": history["valid_per_class_f1"],
            },
            "config": {
                "architecture": "AlexNet (torchvision, warm-started from existing face checkpoint)",
                "num_classes": NUM_CLASSES,
                "optimizer": f"AdamW({fine_tune_depth}, classifier.1/4=1e-4, classifier.6=5e-4, weight_decay=1e-4)",
                "scheduler": f"CosineAnnealingLR(T_max={MAX_EPOCHS}, eta_min=1e-6)",
                "batch_size": BATCH_SIZE,
                "max_epochs": MAX_EPOCHS,
                "epochs_trained": len(history["train_loss"]),
                "early_stopping_patience": PATIENCE,
                "seed": SEED,
                "trainable_layers": [
                    *(list(feature_prefixes)),
                    "classifier.1.bias", "classifier.4", "classifier.6",
                ],
                "trainable_params": trainable_names,
                "loss": "CrossEntropyLoss with inverse-frequency class weights",
                "class_weight_strength": class_weight_strength,
                "label_smoothing": label_smoothing,
                "experiment_name": experiment_name,
                "frame_strategy": frame_strategy,
                "frames_per_clip": frames_per_clip,
                "fine_tune_depth": fine_tune_depth,
                "selection_metric": "validation macro-F1",
                "train_augmentation": "horizontal flip, affine +/-5deg / 4% translation / 0.92-1.08 scale, mild color jitter",
                "train_samples": dataset_sizes["train"],
                "valid_samples": dataset_sizes["valid"],
            },
        }, f, indent=2)

    logger.info("Evaluation results saved: %s", results_path)

    # ── Determinism test ──
    logger.info("Running determinism test (3 forward passes on same image)...")
    inference_model.eval()
    sample_img, sample_label = eval_dataset[0]
    sample_tensor = sample_img.unsqueeze(0).to(device)

    results_list = []
    for run in range(3):
        with torch.no_grad():
            out = inference_model(sample_tensor)
        probs = out.squeeze().cpu().numpy()
        pred_class = int(np.argmax(probs))
        results_list.append((pred_class, probs.tolist()))
        logger.info("  Run %d: predicted=%s (%.4f), probs=%s",
                     run + 1, LABEL_NAMES[pred_class], probs[pred_class],
                     [round(p, 4) for p in probs])

    # Verify determinism
    assert results_list[0][0] == results_list[1][0] == results_list[2][0], \
        "DETERMINISM FAILED: Predictions differ across runs!"
    assert np.allclose(results_list[0][1], results_list[1][1], atol=1e-6), \
        "DETERMINISM FAILED: Probabilities differ across runs!"
    logger.info("✓ Determinism verified: 3 identical forward passes")

    logger.info("")
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("Checkpoint: %s", final_path)
    logger.info("Best validation accuracy: %.4f", best_acc)
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-dir",
        default=os.path.join(PROJECT_ROOT, "models", "vision"),
        help="Directory for checkpoints and evaluation results",
    )
    parser.add_argument(
        "--initial-checkpoint",
        default=None,
        help="Inference-format checkpoint to warm-start from",
    )
    parser.add_argument(
        "--class-weight-strength",
        type=float,
        default=0.5,
        help="Inverse-frequency exponent; 0 disables class weighting",
    )
    parser.add_argument("--frame-strategy", choices=["all", "representative_3"], default="all")
    parser.add_argument(
        "--fine-tune-depth",
        choices=["classifier_only", "conv5", "conv4_conv5"],
        default="conv5",
    )
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--experiment-name", default="current_recipe")
    args = parser.parse_args()
    main(
        args.checkpoint_dir,
        args.initial_checkpoint,
        args.class_weight_strength,
        args.frame_strategy,
        args.fine_tune_depth,
        args.label_smoothing,
        args.experiment_name,
    )
