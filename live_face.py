"""
Standalone Native Real-Time Facial Emotion Recognition Program.
Uses OpenCV VideoCapture, robust multi-pass Haar face detection, and trained AlexNet (models/vision/model_image_new50.pth).
Displays face bounding boxes, predicted emotion, confidence, and 7-class probabilities directly on native OpenCV window.
"""

import sys
import time
import argparse
import logging
from typing import Dict, Optional, Tuple, List, Any

import cv2
import numpy as np
import torch
import torchvision.models as models
import torch.nn as nn
from PIL import Image

from core.ontology import (
    CORE_EMOTIONS,
    normalize_to_core_ontology,
)
from core.vision.face_detector import FaceDetector

# Configure terminal logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("live_face")

# Color palette (BGR for OpenCV)
EMOTION_BGR_MAP = {
    "happy": (46, 204, 113),     # Green
    "sad": (231, 76, 60),       # Blue/Red tint
    "angry": (36, 36, 235),     # Crimson Red
    "fear": (211, 84, 155),     # Purple
    "surprise": (26, 188, 252), # Amber/Yellow
    "disgust": (166, 184, 20),  # Teal/Olive
    "neutral": (180, 180, 180), # Light Slate
    "none": (120, 120, 120),    # Dim Gray
}


class LiveFaceEmotionRecognizer:
    """Manages model loading, face detection, and decoupled live inference loop."""

    def __init__(
        self,
        weights_path: str = "models/vision/model_image_new50.pth",
        camera_id: int = 0,
        target_inference_fps: float = 3.5,
        device: str = "cpu",
        duration: Optional[float] = None,
        debug: bool = False,
    ):
        self.weights_path = weights_path
        self.camera_id = camera_id
        self.target_inference_fps = target_inference_fps
        self.inference_interval = 1.0 / max(0.5, target_inference_fps)
        self.device = torch.device(device)
        self.duration = duration
        self.debug = debug

        logger.info("Initializing FaceDetector subsystem...")
        self.detector = FaceDetector()
        if not self.detector.is_loaded:
            logger.error("FACE DETECTOR FAILED TO LOAD: Haar cascade file was empty or missing.")
        else:
            logger.info("Face detector loaded: %s (%s)", self.detector.detector_name, self.detector.cascade_path)

        logger.info("Loading AlexNet emotion model from %s...", self.weights_path)
        self.model = self._load_model()
        self.model.eval()

        # ImageNet normalization tensors
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)

        # Runtime inference state
        self.last_inference_time = 0.0
        self.latest_emotion: str = "none"
        self.latest_confidence: float = 0.0
        self.latest_probabilities: Optional[Dict[str, float]] = None
        self.latest_face_detected: bool = False
        self.latest_face_box: Optional[Dict[str, Any]] = None
        self.latest_crop_dim: Optional[Tuple[int, int]] = None
        self.faces_count: int = 0

        # Rate tracking
        self.camera_fps: float = 0.0
        self.actual_inference_rate: float = 0.0
        self._frame_count = 0
        self._last_fps_calc = time.time()
        self._inference_count = 0
        self._last_inf_calc = time.time()

    def _load_model(self) -> nn.Module:
        """Instantiate and load trained 7-class AlexNet checkpoint."""
        model = models.alexnet(weights=None)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Linear(num_ftrs, len(CORE_EMOTIONS)),
            nn.Softmax(dim=1),
        )

        state_dict = torch.load(self.weights_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        logger.info("AlexNet weights successfully loaded. 7 emotion classes ready.")
        return model

    def run_inference_on_frame(self, rgb_frame: np.ndarray, bgr_frame: Optional[np.ndarray] = None) -> None:
        """Run Haar face detection and AlexNet forward pass on a single video frame."""
        # Use fallback_if_no_face=False so no crops or false boxes are returned when face is absent
        crops, boxes = self.detector.detect_and_crop_faces(rgb_frame, fallback_if_no_face=False)
        face_boxes = [b for b in boxes if b.get("is_face", False)]
        self.faces_count = len(face_boxes)

        if not face_boxes or not crops:
            # Deterministic policy: No face detected -> strictly DO NOT execute AlexNet or invent emotions
            self.latest_face_detected = False
            self.latest_face_box = None
            self.latest_crop_dim = None
            self.latest_emotion = "none"
            self.latest_confidence = 0.0
            self.latest_probabilities = None
            return

        # Policy for multiple faces: choose primary face with largest bounding box area
        primary_box = max(face_boxes, key=lambda b: b["w"] * b["h"])
        self.latest_face_box = primary_box
        self.latest_crop_dim = (primary_box["w"], primary_box["h"])
        self.latest_face_detected = True

        # Preprocessing: crop 224x224 RGB, normalize, forward pass
        primary_crop = crops[0]

        # Save debug artifacts if --debug is enabled
        if self.debug:
            try:
                import os
                os.makedirs("debug", exist_ok=True)
                if bgr_frame is not None:
                    cv2.imwrite("debug/live_face_frame.jpg", bgr_frame)
                cv2.imwrite("debug/live_face_crop.jpg", cv2.cvtColor(primary_crop, cv2.COLOR_RGB2BGR))
                logger.info(
                    "[DEBUG] Detected box: x=%d y=%d w=%d h=%d | Crop: %dx%d | Model input: 224x224x3",
                    primary_box["x"], primary_box["y"], primary_box["w"], primary_box["h"],
                    primary_crop.shape[1], primary_crop.shape[0]
                )
            except Exception as e:
                logger.debug("Failed saving debug crops: %s", e)

        tensor = torch.from_numpy(primary_crop).float().permute(2, 0, 1) / 255.0
        tensor = tensor.to(self.device)
        tensor = (tensor - self.mean) / self.std
        tensor = tensor.unsqueeze(0)

        with torch.no_grad():
            out = self.model(tensor)
            probs = out.squeeze().cpu().numpy()

        raw_probs = {CORE_EMOTIONS[i]: float(probs[i]) for i in range(len(CORE_EMOTIONS))}
        normalized_probs = normalize_to_core_ontology(raw_probs, supported_emotions=CORE_EMOTIONS)
        top_emotion = max(normalized_probs, key=normalized_probs.get)
        confidence = float(normalized_probs[top_emotion])

        self.latest_emotion = top_emotion
        self.latest_confidence = confidence
        self.latest_probabilities = normalized_probs

        # Track inference rate
        self._inference_count += 1
        now = time.time()
        elapsed = now - self._last_inf_calc
        if elapsed >= 1.0:
            self.actual_inference_rate = round(self._inference_count / elapsed, 1)
            self._inference_count = 0
            self._last_inf_calc = now

    def draw_hud(self, frame: np.ndarray) -> np.ndarray:
        """Render HUD overlay with bounding box, rates, and 7-class distribution."""
        h, w = frame.shape[:2]

        # 1. Draw Face Bounding Box (Only when face is detected)
        color = EMOTION_BGR_MAP.get(self.latest_emotion, (180, 180, 180))

        if self.latest_face_detected and self.latest_face_box:
            bx = self.latest_face_box["x"]
            by = self.latest_face_box["y"]
            bw = self.latest_face_box["w"]
            bh = self.latest_face_box["h"]

            # Main bounding box
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color, 2)

            # Corner brackets for HUD aesthetic
            corner_len = int(min(bw, bh) * 0.2)
            # Top-left
            cv2.line(frame, (bx, by), (bx + corner_len, by), color, 4)
            cv2.line(frame, (bx, by), (bx, by + corner_len), color, 4)
            # Top-right
            cv2.line(frame, (bx + bw, by), (bx + bw - corner_len, by), color, 4)
            cv2.line(frame, (bx + bw, by), (bx + bw, by + corner_len), color, 4)
            # Bottom-left
            cv2.line(frame, (bx, by + bh), (bx + corner_len, by + bh), color, 4)
            cv2.line(frame, (bx, by + bh), (bx, by + bh - corner_len), color, 4)
            # Bottom-right
            cv2.line(frame, (bx + bw, by + bh), (bx + bw - corner_len, by + bh), color, 4)
            cv2.line(frame, (bx + bw, by + bh), (bx + bw, by + bh - corner_len), color, 4)

            # Face label tag above box
            tag_text = f"{self.latest_emotion.upper()} {self.latest_confidence * 100:.1f}%"
            (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
            cv2.rectangle(frame, (bx, by - th - 10), (bx + tw + 10, by), color, -1)
            cv2.putText(
                frame,
                tag_text,
                (bx + 5, by - 6),
                cv2.FONT_HERSHEY_DUPLEX,
                0.55,
                (10, 15, 20),
                1,
                cv2.LINE_AA,
            )

        # 2. Draw Debug Overlay (Top-Left) if --debug is enabled
        if self.debug:
            dbg_w = 320
            dbg_h = 160
            dbg_x = 10
            dbg_y = 10
            sub_dbg = frame[dbg_y : dbg_y + dbg_h, dbg_x : dbg_x + dbg_w]
            dbg_overlay = sub_dbg.copy()
            cv2.rectangle(dbg_overlay, (0, 0), (dbg_w, dbg_h), (12, 16, 24), -1)
            cv2.addWeighted(dbg_overlay, 0.85, sub_dbg, 0.15, 0, sub_dbg)
            cv2.rectangle(frame, (dbg_x, dbg_y), (dbg_x + dbg_w, dbg_y + dbg_h), (0, 200, 255), 1)

            cv2.putText(frame, "DIAGNOSTIC MONITOR", (dbg_x + 10, dbg_y + 20), cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Camera: {int(self.camera_fps)} FPS", (dbg_x + 10, dbg_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Detector: {self.detector.detector_name}", (dbg_x + 10, dbg_y + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Detector Loaded: {'YES' if self.detector.is_loaded else 'NO'}", (dbg_x + 10, dbg_y + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 255, 100) if self.detector.is_loaded else (50, 50, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Faces Detected: {self.faces_count}", (dbg_x + 10, dbg_y + 88), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
            
            box_str = f"x={self.latest_face_box['x']} y={self.latest_face_box['y']} w={self.latest_face_box['w']} h={self.latest_face_box['h']}" if self.latest_face_box else "None"
            cv2.putText(frame, f"Primary Face: {box_str}", (dbg_x + 10, dbg_y + 104), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1, cv2.LINE_AA)
            
            crop_str = f"224x224 (Source: {self.latest_crop_dim[0]}x{self.latest_crop_dim[1]})" if self.latest_crop_dim else "None"
            cv2.putText(frame, f"Crop: {crop_str}", (dbg_x + 10, dbg_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Model: AlexNet | Checkpoint: LOADED", (dbg_x + 10, dbg_y + 136), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (100, 255, 100), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Inference: {self.actual_inference_rate:.1f}/sec", (dbg_x + 10, dbg_y + 152), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (100, 210, 255), 1, cv2.LINE_AA)

        # 3. Draw Translucent HUD Card on Right Side
        hud_w = 260
        hud_h = h - 20
        hud_x = w - hud_w - 10
        hud_y = 10

        sub_img = frame[hud_y : hud_y + hud_h, hud_x : hud_x + hud_w]
        overlay = sub_img.copy()
        cv2.rectangle(overlay, (0, 0), (hud_w, hud_h), (15, 20, 30), -1)
        # Blend overlay (alpha = 0.82)
        cv2.addWeighted(overlay, 0.82, sub_img, 0.18, 0, sub_img)

        # HUD Border
        cv2.rectangle(frame, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (60, 70, 90), 1)

        # Title
        cv2.putText(
            frame,
            "EMOTION RECOGNITION",
            (hud_x + 12, hud_y + 24),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Rates & model architecture
        rates_text = f"Cam: {int(self.camera_fps)} FPS | Inf: {self.actual_inference_rate:.1f}/s"
        cv2.putText(
            frame,
            rates_text,
            (hud_x + 12, hud_y + 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (100, 210, 255),
            1,
            cv2.LINE_AA,
        )

        # Separator line
        cv2.line(frame, (hud_x + 10, hud_y + 54), (hud_x + hud_w - 10, hud_y + 54), (50, 60, 80), 1)

        # Detection Status
        if self.latest_face_detected:
            status_text = f"FACE DETECTED ({self.faces_count})"
            status_color = (46, 204, 113)
        else:
            status_text = "NO FACE DETECTED"
            status_color = (0, 165, 255)

        cv2.putText(
            frame,
            status_text,
            (hud_x + 12, hud_y + 74),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            status_color,
            1,
            cv2.LINE_AA,
        )

        # Dominant Emotion & Confidence Display
        if self.latest_face_detected:
            cv2.putText(
                frame,
                "CLASSIFIED EMOTION:",
                (hud_x + 12, hud_y + 98),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (160, 170, 185),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                self.latest_emotion.upper(),
                (hud_x + 12, hud_y + 130),
                cv2.FONT_HERSHEY_DUPLEX,
                0.95,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Confidence: {self.latest_confidence * 100:.2f}%",
                (hud_x + 12, hud_y + 152),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (240, 240, 245),
                1,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                frame,
                "Awaiting Face...",
                (hud_x + 12, hud_y + 115),
                cv2.FONT_HERSHEY_DUPLEX,
                0.65,
                (140, 150, 165),
                1,
                cv2.LINE_AA,
            )

        # Separator line before probability list
        cv2.line(frame, (hud_x + 10, hud_y + 168), (hud_x + hud_w - 10, hud_y + 168), (50, 60, 80), 1)

        # 7-Class Probability Distribution List
        prob_header = "7-CLASS PROBABILITIES:" if self.latest_face_detected else "7-CLASS PROBABILITIES (Awaiting):"
        cv2.putText(
            frame,
            prob_header,
            (hud_x + 12, hud_y + 186),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (180, 190, 205) if self.latest_face_detected else (110, 115, 125),
            1,
            cv2.LINE_AA,
        )

        y_offset = hud_y + 208
        max_bar_w = 90

        for em in CORE_EMOTIONS:
            if self.latest_face_detected and self.latest_probabilities is not None:
                p = self.latest_probabilities.get(em, 0.0)
                is_top = (em == self.latest_emotion)
                label_col = EMOTION_BGR_MAP[em] if is_top else (170, 175, 185)

                # Emotion name
                cv2.putText(
                    frame,
                    f"{em:<8}",
                    (hud_x + 12, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    label_col,
                    1,
                    cv2.LINE_AA,
                )

                # Percentage
                pct_str = f"{p * 100:4.1f}%"
                cv2.putText(
                    frame,
                    pct_str,
                    (hud_x + 85, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (240, 240, 240) if is_top else (150, 155, 165),
                    1,
                    cv2.LINE_AA,
                )

                # Progress bar track
                bar_x = hud_x + 140
                bar_y = y_offset - 9
                bar_h = 7
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + max_bar_w, bar_y + bar_h), (40, 48, 62), -1)

                # Progress bar fill
                fill_w = int(max_bar_w * p)
                if fill_w > 0:
                    cv2.rectangle(
                        frame,
                        (bar_x, bar_y),
                        (bar_x + fill_w, bar_y + bar_h),
                        EMOTION_BGR_MAP[em] if is_top else (99, 102, 241),
                        -1,
                    )
            else:
                # Strictly keep probabilities unavailable when no face is present (Do not render fake 0.0%)
                cv2.putText(
                    frame,
                    f"{em:<8}",
                    (hud_x + 12, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (90, 95, 105),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    "--.-%",
                    (hud_x + 85, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (80, 85, 95),
                    1,
                    cv2.LINE_AA,
                )
                bar_x = hud_x + 140
                bar_y = y_offset - 9
                bar_h = 7
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + max_bar_w, bar_y + bar_h), (30, 35, 45), -1)

            y_offset += 24

        # Footer commands
        cv2.line(frame, (hud_x + 10, hud_h - 26), (hud_x + hud_w - 10, hud_h - 26), (50, 60, 80), 1)
        cv2.putText(
            frame,
            "[Q/ESC] Quit  |  [R] Reset",
            (hud_x + 28, hud_h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (140, 150, 165),
            1,
            cv2.LINE_AA,
        )

        return frame

    def run(self) -> None:
        """Main execution loop for real-time webcam capture and display."""
        logger.info("Opening webcam #%d...", self.camera_id)
        # On Windows, try DirectShow first for robust frame grabbing
        cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        if not cap.isOpened():
            logger.info("DirectShow unavailable for webcam #%d, attempting standard capture...", self.camera_id)
            cap = cv2.VideoCapture(self.camera_id)

        if not cap.isOpened():
            logger.error("Failed to open webcam index %d. Check connection and permissions.", self.camera_id)
            sys.exit(1)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        window_name = "Live Facial Emotion Recognition (AlexNet)"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        logger.info("=" * 60)
        logger.info("LIVE WEBCAM EMOTION RECOGNITION ACTIVE")
        logger.info("Press 'Q' or 'ESC' in the webcam window to exit.")
        logger.info("Press 'R' to reset prediction state.")
        if self.debug:
            logger.info("Debug mode: ENABLED (HUD diagnostic card active)")
        logger.info("=" * 60)

        start_time = time.time()
        try:
            while True:
                if self.duration and (time.time() - start_time) >= self.duration:
                    logger.info("Reached requested duration (%.1fs). Exiting camera loop.", self.duration)
                    break

                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.02)
                    continue

                # Flip horizontally for natural mirror feel
                frame = cv2.flip(frame, 1)

                # Camera FPS tracking
                self._frame_count += 1
                now = time.time()
                elapsed = now - self._last_fps_calc
                if elapsed >= 1.0:
                    self.camera_fps = self._frame_count / elapsed
                    self._frame_count = 0
                    self._last_fps_calc = now

                # Controlled inference rate: only evaluate AlexNet every inference_interval seconds
                if (now - self.last_inference_time) >= self.inference_interval:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.run_inference_on_frame(rgb_frame, bgr_frame=frame)
                    self.last_inference_time = now

                # Draw bounding box, debug card (if enabled), and HUD metrics directly on frame
                rendered_frame = self.draw_hud(frame)

                cv2.imshow(window_name, rendered_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == ord("Q") or key == 27:  # 'q' or ESC
                    logger.info("Exit command received ('%s'). Closing camera...", chr(key) if key != 27 else "ESC")
                    break
                elif key == ord("r") or key == ord("R"):
                    logger.info("Resetting prediction state.")
                    self.latest_emotion = "none"
                    self.latest_confidence = 0.0
                    self.latest_probabilities = None
                    self.latest_face_detected = False
                    self.latest_face_box = None
                    self.latest_crop_dim = None

        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Webcam released. OpenCV windows closed. Clean termination complete.")


def main():
    parser = argparse.ArgumentParser(description="Live Facial Emotion Recognition using OpenCV and AlexNet")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--fps", type=float, default=3.5, help="Target inference rate in Hz (default: 3.5)")
    parser.add_argument("--weights", type=str, default="models/vision/model_image_new50.pth", help="Checkpoint path")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device: 'cpu' or 'cuda'")
    parser.add_argument("--duration", type=float, default=None, help="Optional duration in seconds to run before auto-exiting")
    parser.add_argument("--debug", action="store_true", help="Enable diagnostic overlay and verbose logging")
    args = parser.parse_args()

    recognizer = LiveFaceEmotionRecognizer(
        weights_path=args.weights,
        camera_id=args.camera,
        target_inference_fps=args.fps,
        device=args.device,
        duration=args.duration,
        debug=args.debug,
    )
    recognizer.run()


if __name__ == "__main__":
    main()
