"""
Facial Detection and Image Preprocessing Subsystem.
Detects faces using OpenCV Haar Cascades, extracts and aligns facial bounding boxes,
and normalizes crops to 224x224 RGB tensors for deep emotion network inference.
"""

import io
import logging
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

TARGET_IMAGE_SIZE = (224, 224)


class FaceDetector:
    """Detects and crops faces from static images or video frames."""

    def __init__(self, target_size: Tuple[int, int] = TARGET_IMAGE_SIZE):
        self.target_size = target_size
        self._cascade = None
        self._alt_cascade = None
        self.cascade_path: Optional[str] = None
        self.detector_name = "OpenCV Haar Cascade"
        self._init_cascade()

    def _init_cascade(self) -> None:
        """Initialize OpenCV Haar cascade classifiers dynamically from installed cv2 data."""
        try:
            import os
            import cv2

            if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
                primary_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
                if os.path.isfile(primary_path):
                    self._cascade = cv2.CascadeClassifier(primary_path)
                    if self._cascade.empty():
                        logger.warning("Primary face cascade file loaded but reports empty: %s", primary_path)
                        self._cascade = None
                    else:
                        self.cascade_path = primary_path

                alt_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_alt2.xml")
                if os.path.isfile(alt_path):
                    self._alt_cascade = cv2.CascadeClassifier(alt_path)
                    if self._alt_cascade.empty():
                        self._alt_cascade = None
            else:
                logger.warning("cv2 does not provide CascadeClassifier or cv2.data.haarcascades")
                self._cascade = None
        except Exception as e:
            logger.error("Failed to initialize OpenCV CascadeClassifier: %s", e)
            self._cascade = None

    @property
    def is_loaded(self) -> bool:
        """Returns True if at least one face detector cascade is ready."""
        return self._cascade is not None and not self._cascade.empty()

    def load_image(self, image_source: Any) -> np.ndarray:
        """Loads an image from file path, raw bytes, or PIL Image into RGB numpy array."""
        from PIL import Image

        if isinstance(image_source, bytes):
            image = Image.open(io.BytesIO(image_source)).convert("RGB")
            return np.array(image)
        elif isinstance(image_source, str):
            image = Image.open(image_source).convert("RGB")
            return np.array(image)
        elif isinstance(image_source, Image.Image):
            return np.array(image_source.convert("RGB"))
        elif isinstance(image_source, np.ndarray):
            if image_source.ndim == 2:
                return np.stack([image_source] * 3, axis=-1)
            elif image_source.shape[-1] == 4:
                return image_source[:, :, :3]
            return image_source
        else:
            raise ValueError(f"Unsupported image input type: {type(image_source)}")

    def detect_and_crop_faces(
        self, image: np.ndarray, fallback_if_no_face: bool = True
    ) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        """
        Detects faces in an RGB image and returns normalized 224x224 crops.
        Uses a robust multi-pass detection strategy:
          1. Frontal default cascade (scale=1.1, minNeighbors=4)
          2. Frontal default with histogram equalization (for lighting/shadow compensation)
          3. Frontal alt2 cascade (for head pose/slight angle variations)
        
        Args:
            image: RGB numpy uint8 array.
            fallback_if_no_face: If True, returns a center crop with is_face=False when no face detected.
                                 If False, returns empty crops and boxes (preferred for live webcam).
        Returns:
            crops: List of 224x224x3 RGB numpy uint8 arrays.
            boxes: List of bounding box dicts {'x', 'y', 'w', 'h', 'is_face'}.
        """
        from PIL import Image

        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 4:
            image = image[:, :, :3]

        h, w = image.shape[:2]
        boxes: List[Dict[str, Any]] = []
        crops: List[np.ndarray] = []

        if self._cascade is not None:
            try:
                import cv2

                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

                # Pass 1: Standard frontal cascade
                detected = self._cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
                )

                # Pass 2: If no faces, equalize histogram to counter harsh/low lighting
                if len(detected) == 0:
                    gray_eq = cv2.equalizeHist(gray)
                    detected = self._cascade.detectMultiScale(
                        gray_eq, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
                    )

                # Pass 3: If still no faces, try alt2 cascade (tolerant to tilt)
                if len(detected) == 0 and self._alt_cascade is not None:
                    detected = self._alt_cascade.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
                    )
                    if len(detected) == 0:
                        gray_eq = cv2.equalizeHist(gray)
                        detected = self._alt_cascade.detectMultiScale(
                            gray_eq, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30)
                        )

                # Sort detected faces by area descending (largest face first)
                if len(detected) > 0:
                    sorted_detected = sorted(detected, key=lambda b: b[2] * b[3], reverse=True)
                    for x, y, bw, bh in sorted_detected:
                        boxes.append({"x": int(x), "y": int(y), "w": int(bw), "h": int(bh), "is_face": True})
                        # Square crop centered on detected face (1.30x face dimension)
                        side = int(max(bw, bh) * 1.30)
                        center_x = x + bw // 2
                        center_y = y + bh // 2
                        x1 = center_x - side // 2
                        y1 = center_y - side // 2
                        x2 = x1 + side
                        y2 = y1 + side

                        # Shift to keep within frame bounds while preserving square dimensions
                        if x1 < 0:
                            x2 -= x1
                            x1 = 0
                        if y1 < 0:
                            y2 -= y1
                            y1 = 0
                        if x2 > w:
                            x1 = max(0, x1 - (x2 - w))
                            x2 = w
                        if y2 > h:
                            y1 = max(0, y1 - (y2 - h))
                            y2 = h

                        side_clamped = min(x2 - x1, y2 - y1)
                        x2 = x1 + side_clamped
                        y2 = y1 + side_clamped

                        face_crop = image[y1:y2, x1:x2]
                        pil_crop = Image.fromarray(face_crop).resize(
                            self.target_size, resample=Image.Resampling.BICUBIC
                        )
                        crops.append(np.array(pil_crop))
            except Exception as e:
                logger.error("Haar face detection error: %s", e)

        # Fallback: If no face detected via cascade and fallback requested
        if not crops and fallback_if_no_face:
            min_dim = min(h, w)
            start_y = (h - min_dim) // 2
            start_x = (w - min_dim) // 2
            center_crop = image[start_y : start_y + min_dim, start_x : start_x + min_dim]
            pil_crop = Image.fromarray(center_crop).resize(
                self.target_size, resample=Image.Resampling.BICUBIC
            )
            crops.append(np.array(pil_crop))
            boxes.append({"x": int(start_x), "y": int(start_y), "w": int(min_dim), "h": int(min_dim), "is_face": False, "fallback_crop": True})

        return crops, boxes
