"""
API Integration Test Suite for FastAPI Backend.
Validates Health, Text, Audio, Vision, and Multi-Modal endpoints.
Tests Zero Fake AI enforcement, file safety, and late fusion API behavior.
"""

import os
import io
import math
import pytest
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from api.main import app
from core.ontology import CORE_EMOTIONS

client = TestClient(app)

AUDIO_SAMPLE_PATH = "speech-emotion-recognition-master/dataset/Sad/09b03Ta.wav"
IMAGE_SAMPLE_PATH = "Multimodal-Emotion-Recognition-master/image_data/train/Actor_01/01-01-01-01-01-01-01/10.jpg"


# ---------------------------------------------------------------------------
# Test 1: Health Endpoint
# ---------------------------------------------------------------------------
def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] in ("ok", "degraded")
    assert data["service"] == "emotion-recognition-api"
    assert "models" in data

    # Verify model readiness statuses
    models = data["models"]
    assert models["text"]["operational"] is True
    assert models["audio"]["operational"] is True
    assert models["vision"]["operational"] is True
    assert models["vision"]["status"] == "ok"


# ---------------------------------------------------------------------------
# Test 2: Text Prediction Success
# ---------------------------------------------------------------------------
def test_text_prediction_success():
    payload = {"text": "I am absolutely thrilled and overjoyed with these incredible results!"}
    response = client.post("/api/predict/text", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["available"] is True
    assert data["status"] == "ok"
    assert data["modality"] == "text"
    assert data["emotion"] == "happy"
    assert data["confidence"] > 0.90

    # Validate probability distribution
    probs = data["probabilities"]
    assert all(e in probs for e in CORE_EMOTIONS)
    assert math.isclose(sum(probs.values()), 1.0, abs_tol=0.01)


# ---------------------------------------------------------------------------
# Test 3: Text Empty / Whitespace Validation
# ---------------------------------------------------------------------------
def test_text_prediction_empty():
    # Whitespace only
    response = client.post("/api/predict/text", json={"text": "    "})
    assert response.status_code == 422

    # Empty string
    response2 = client.post("/api/predict/text", json={"text": ""})
    assert response2.status_code == 422


# ---------------------------------------------------------------------------
# Test 4: Audio Prediction Success
# ---------------------------------------------------------------------------
def test_audio_prediction_success():
    assert os.path.exists(AUDIO_SAMPLE_PATH), f"Audio test file missing: {AUDIO_SAMPLE_PATH}"

    with open(AUDIO_SAMPLE_PATH, "rb") as f:
        response = client.post(
            "/api/predict/audio",
            files={"file": ("09b03Ta.wav", f, "audio/wav")},
        )

    assert response.status_code == 200
    data = response.json()

    assert data["available"] is True
    assert data["status"] == "ok"
    assert data["modality"] == "audio"
    assert data["emotion"] == "sad"
    assert data["confidence"] > 0.95

    # Verify probability distribution
    probs = data["probabilities"]
    assert all(e in probs for e in CORE_EMOTIONS)
    assert math.isclose(sum(probs.values()), 1.0, abs_tol=0.01)

    # Unsupported classes must be strictly 0.0
    assert probs["disgust"] == 0.0
    assert probs["fear"] == 0.0
    assert probs["surprise"] == 0.0


def test_audio_silence_is_not_classified():
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, np.zeros(32000, dtype=np.int16), 16000, format="WAV", subtype="PCM_16")

    response = client.post(
        "/api/predict/audio",
        files={"file": ("silence.wav", audio_buffer.getvalue(), "audio/wav")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["available"] is False
    assert data["status"] == "SILENT_OR_NEAR_SILENT_AUDIO"
    assert data["emotion"] == "none"
    assert data["confidence"] == 0.0
    assert all(probability == 0.0 for probability in data["probabilities"].values())


# ---------------------------------------------------------------------------
# Test 5: Vision Prediction Success (Genuine Trained Checkpoint)
# ---------------------------------------------------------------------------
def test_vision_prediction_success():
    assert os.path.exists(IMAGE_SAMPLE_PATH), f"Image test file missing: {IMAGE_SAMPLE_PATH}"

    with open(IMAGE_SAMPLE_PATH, "rb") as f:
        response = client.post(
            "/api/predict/vision",
            files={"file": ("10.jpg", f, "image/jpeg")},
        )

    # Must return 200 OK with genuine emotion prediction
    assert response.status_code == 200
    data = response.json()

    assert data["available"] is True
    assert data["status"] == "ok"
    assert data["emotion"] in CORE_EMOTIONS
    assert data["confidence"] > 0.0
    assert math.isclose(sum(data["probabilities"].values()), 1.0, abs_tol=0.01)


# ---------------------------------------------------------------------------
# Test 6: Multimodal Text + Speech
# ---------------------------------------------------------------------------
def test_multimodal_text_and_speech():
    assert os.path.exists(AUDIO_SAMPLE_PATH)

    with open(AUDIO_SAMPLE_PATH, "rb") as f:
        response = client.post(
            "/api/predict/multimodal",
            data={"text": "I feel so devastated, lonely, and hopeless after losing everything."},
            files={"audio": ("09b03Ta.wav", f, "audio/wav")},
        )

    assert response.status_code == 200
    data = response.json()

    assert data["available"] is True
    assert data["status"] == "ok"
    assert data["fusion_mode"] == "multimodal"
    assert set(data["active_modalities"]) == {"text", "audio"}
    assert data["emotion"] == "sad"
    assert data["confidence"] > 0.95
    assert math.isclose(sum(data["probabilities"].values()), 1.0, abs_tol=0.01)


# ---------------------------------------------------------------------------
# Test 7: Multimodal Text + Speech + Vision (All Three Modalities Active)
# ---------------------------------------------------------------------------
def test_multimodal_text_speech_and_vision():
    assert os.path.exists(AUDIO_SAMPLE_PATH)
    assert os.path.exists(IMAGE_SAMPLE_PATH)

    text_input = "I feel so devastated, lonely, and hopeless after losing everything."

    # Send all three: text + audio + image
    with open(AUDIO_SAMPLE_PATH, "rb") as f_audio, open(IMAGE_SAMPLE_PATH, "rb") as f_image:
        tri_response = client.post(
            "/api/predict/multimodal",
            data={"text": text_input},
            files={
                "audio": ("09b03Ta.wav", f_audio, "audio/wav"),
                "image": ("10.jpg", f_image, "image/jpeg"),
            },
        )

    assert tri_response.status_code == 200
    tri_data = tri_response.json()

    assert tri_data["available"] is True
    assert set(tri_data["active_modalities"]) == {"text", "audio", "vision"}
    assert len(tri_data["unavailable_modalities"]) == 0
    assert tri_data["emotion"] in CORE_EMOTIONS
    assert tri_data["confidence"] > 0.0
    assert math.isclose(sum(tri_data["probabilities"].values()), 1.0, abs_tol=0.01)


# ---------------------------------------------------------------------------
# Test 8: Multimodal No Modality Supplied
# ---------------------------------------------------------------------------
def test_multimodal_no_modality():
    response = client.post("/api/predict/multimodal", data={})
    assert response.status_code == 400
    assert "At least one modality" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Test 9: Unsupported File Format Handled Cleanly
# ---------------------------------------------------------------------------
def test_unsupported_upload_format():
    # Sending a text file to audio endpoint
    fake_content = b"This is not audio content."
    response = client.post(
        "/api/predict/audio",
        files={"file": ("notes.txt", fake_content, "text/plain")},
    )
    assert response.status_code == 415
    assert "Unsupported audio format" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Test 10: Real-Time Fusion Endpoint (/api/predict/fuse)
# ---------------------------------------------------------------------------
def test_predict_fuse_endpoint():
    # Construct mock ModalityResult objects matching schemas
    payload = {
        "results": {
            "text": {
                "modality": "text",
                "available": True,
                "status": "ok",
                "emotion": "happy",
                "confidence": 0.92,
                "probabilities": {
                    "angry": 0.01,
                    "disgust": 0.01,
                    "fear": 0.01,
                    "happy": 0.92,
                    "neutral": 0.03,
                    "sad": 0.01,
                    "surprise": 0.01,
                },
                "supported_emotions": list(CORE_EMOTIONS),
                "valence": 0.8,
                "arousal": 0.6,
                "metadata": {},
            },
            "audio": {
                "modality": "audio",
                "available": True,
                "status": "ok",
                "emotion": "happy",
                "confidence": 0.80,
                "probabilities": {
                    "angry": 0.05,
                    "disgust": 0.0,
                    "fear": 0.0,
                    "happy": 0.80,
                    "neutral": 0.10,
                    "sad": 0.05,
                    "surprise": 0.0,
                },
                "supported_emotions": ["angry", "happy", "neutral", "sad"],
                "valence": 0.7,
                "arousal": 0.5,
                "metadata": {},
            },
        }
    }
    response = client.post("/api/predict/fuse", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["available"] is True
    assert data["emotion"] == "happy"
    assert data["fusion_mode"] == "multimodal"
    assert set(data["active_modalities"]) == {"text", "audio"}
    assert "explainability" in data
    assert "conflict" in data
    assert data["confidence"] > 0.80
