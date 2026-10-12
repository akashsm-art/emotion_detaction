"""
RAVDESS Facial Emotion Dataset Loader.

Faithfully reproduces the original visual_network.ipynb data pipeline:
- Reads pre-extracted face crops from image_data/{train,valid}/Actor_XX/RAVDESS_ID/
- Derives emotion labels from RAVDESS filename convention (3rd dash-segment)
- Skips corrupt 0-byte images
- Maps RAVDESS codes to 7-class Ekman emotion indices matching core ontology:
    0=angry, 1=disgust, 2=fear, 3=happy, 4=neutral, 5=sad, 6=surprise

Speed optimization: preloads all images into memory as tensors for fast CPU training.
"""

import os
import logging
from typing import Tuple, List, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

# RAVDESS emotion code → 7-class index
# Matches the original visual_network.ipynb mapping EXACTLY,
# but using core ontology ordering (fear/surprise not fearful/surprised)
RAVDESS_CODE_TO_LABEL = {
    "01": 4,  # neutral
    "02": 4,  # calm → neutral (original author's choice)
    "03": 3,  # happy
    "04": 5,  # sad
    "05": 0,  # angry
    "06": 2,  # fearful → fear
    "07": 1,  # disgust
    "08": 6,  # surprised → surprise
}

LABEL_NAMES = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise",
}

NUM_CLASSES = 7


class RAVDESSFaceDataset(Dataset):
    """
    PyTorch Dataset for pre-extracted RAVDESS face crops.

    Directory structure:
        root/Actor_XX/RAVDESS_VIDEO_ID/{10,20,30,...}.jpg

    Labels are derived from the RAVDESS video ID filename, not from any CSV.
    The third segment (split on '-') is the emotion code.

    If preload=True, all images are read into RAM as PIL-ready tensors
    on initialization for faster CPU training throughput.
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[transforms.Compose] = None,
        skip_corrupt: bool = True,
        preload: bool = True,
        frames_per_clip: Optional[int] = None,
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.skip_corrupt = skip_corrupt
        self.preload = preload
        if frames_per_clip is not None and frames_per_clip < 1:
            raise ValueError("frames_per_clip must be positive")
        self.frames_per_clip = frames_per_clip

        self.samples: List[Tuple[str, int]] = []
        self.full_class_counts = {}
        self._images: List[Optional[Image.Image]] = []  # preloaded PIL images
        self._scan_directory()

        if self.preload:
            self._preload_images()

        logger.info(
            "RAVDESSFaceDataset loaded: %d samples from %s (preload=%s)",
            len(self.samples),
            root_dir,
            preload,
        )

    def _scan_directory(self):
        """Walk the directory tree and collect (image_path, label) pairs."""
        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(f"Dataset root not found: {self.root_dir}")

        for actor_name in sorted(os.listdir(self.root_dir)):
            actor_dir = os.path.join(self.root_dir, actor_name)
            if not os.path.isdir(actor_dir):
                continue

            for video_name in sorted(os.listdir(actor_dir)):
                video_dir = os.path.join(actor_dir, video_name)
                if not os.path.isdir(video_dir):
                    continue

                # Extract emotion code from RAVDESS filename
                parts = video_name.split("-")
                if len(parts) < 3:
                    continue
                emotion_code = parts[2]
                label = RAVDESS_CODE_TO_LABEL.get(emotion_code)
                if label is None:
                    continue

                clip_samples = []
                for frame_file in sorted(os.listdir(video_dir)):
                    if not frame_file.lower().endswith(".jpg"):
                        continue
                    fpath = os.path.join(video_dir, frame_file)

                    # Skip corrupt 0-byte images
                    if self.skip_corrupt and os.path.getsize(fpath) == 0:
                        continue

                    clip_samples.append((fpath, label))

                self.full_class_counts[label] = self.full_class_counts.get(label, 0) + len(clip_samples)
                if self.frames_per_clip is not None and len(clip_samples) > self.frames_per_clip:
                    numeric_samples = sorted(
                        clip_samples,
                        key=lambda sample: int(os.path.splitext(os.path.basename(sample[0]))[0]),
                    )
                    count = len(numeric_samples)
                    positions = [
                        round((i + 1) * (count - 1) / (self.frames_per_clip + 1))
                        for i in range(self.frames_per_clip)
                    ]
                    clip_samples = [numeric_samples[position] for position in positions]
                self.samples.extend(clip_samples)

    def _preload_images(self):
        """Load all images into memory as PIL Images for fast access."""
        logger.info("Preloading %d images into memory...", len(self.samples))
        self._images = []
        for i, (fpath, _) in enumerate(self.samples):
            img = Image.open(fpath).convert("RGB")
            self._images.append(img)
            if (i + 1) % 2000 == 0:
                logger.info("  Preloaded %d/%d images", i + 1, len(self.samples))
        logger.info("  Preload complete: %d images in memory", len(self._images))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        _, label = self.samples[idx]

        if self.preload:
            image = self._images[idx].copy()  # copy to prevent transform side-effects
        else:
            img_path = self.samples[idx][0]
            image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def get_class_distribution(self) -> dict:
        """Return {label_name: count} for dataset analysis."""
        from collections import Counter
        counts = Counter(label for _, label in self.samples)
        return {LABEL_NAMES[k]: v for k, v in sorted(counts.items())}

    def get_full_class_distribution(self) -> dict:
        """Return class counts before optional per-clip frame sampling."""
        return {LABEL_NAMES[k]: v for k, v in sorted(self.full_class_counts.items())}


def get_train_transform() -> transforms.Compose:
    """
    Mild augmentation for lighting, framing, and small head-pose variation.
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(
            degrees=5,
            translate=(0.04, 0.04),
            scale=(0.92, 1.08),
        ),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_eval_transform() -> transforms.Compose:
    """
    Evaluation transform — no random augmentation, only normalize.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
