"""
Face Detector Diagnostic Script
Evaluates OpenCV Haar Cascade detector on camera and test assets.
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))
import cv2
from core.vision.face_detector import FaceDetector
from core.vision.model import VisionEmotionModel

def run_diagnostic():
    detector = FaceDetector()
    model = VisionEmotionModel()

    print("==================================================")
    print("FACE DETECTOR DIAGNOSTIC")
    print("==================================================")

    # 1. Camera check
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    
    camera_status = "PASS" if cap.isOpened() else "FAIL"
    frame_w, frame_h = 640, 480
    frame = None
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        ret, frame = cap.read()
        if ret and frame is not None:
            frame_h, frame_w = frame.shape[:2]
        cap.release()

    print(f"Camera:\n{camera_status}\n")
    print(f"Frame:\n{frame_w}x{frame_h}\n")
    print(f"Detector: {detector.detector_name}\n")
    print(f"Cascade/Model: {detector.cascade_path}\n")
    print(f"Detector loaded:\n{'YES' if detector.is_loaded else 'NO'}\n")

    # 2. Test detection on actual test face image
    test_face_path = "Multimodal-Emotion-Recognition-master/image_data/valid/Actor_24/01-01-08-01-01-01-24/70.jpg"
    test_img = cv2.imread(test_face_path)
    test_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    crops, boxes = detector.detect_and_crop_faces(test_rgb, fallback_if_no_face=False)
    face_boxes = [b for b in boxes if b.get("is_face", False)]

    print(f"Faces detected: {len(face_boxes)}\n")
    print(f"Bounding boxes: {face_boxes}\n")

    if crops:
        crop = crops[0]
        print("FACE DETECTED")
        print(f"Bounding box = x={face_boxes[0]['x']}, y={face_boxes[0]['y']}, w={face_boxes[0]['w']}, h={face_boxes[0]['h']}")
        print(f"Crop size = {crop.shape}")
        print("Final model input = 224x224x3\n")
        
        # Run inference on verified model
        res = model.predict(test_face_path)
        print("MODEL TEST:")
        print(f"Model Checkpoint: {model.weights_path}")
        print(f"Checkpoint Loaded: {model._is_loaded}")
        print(f"Detected Emotion: {res.emotion.upper()}")
        print(f"Confidence: {res.confidence * 100:.2f}%\n")
        print("Probabilities:")
        for em, p in res.probabilities.items():
            print(f"  {em:<10} {p * 100:.2f}%")

    print("==================================================")

if __name__ == "__main__":
    run_diagnostic()
