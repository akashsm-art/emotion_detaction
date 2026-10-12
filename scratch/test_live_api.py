"""
Test live running FastAPI server endpoints:
1. GET /api/health
2. POST /api/predict/text
3. POST /api/predict/audio
4. POST /api/predict/vision
5. POST /api/predict/multimodal (Text + Audio + Vision)
"""

import os
import sys
import json
import requests

BASE_URL = "http://127.0.0.1:8000"

def main():
    print("=" * 60)
    print("TESTING LIVE FASTAPI SERVER (http://127.0.0.1:8000)")
    print("=" * 60)

    # 1. Health
    r = requests.get(f"{BASE_URL}/api/health")
    print("\n1. GET /api/health:")
    print(f"   Status: {r.status_code}")
    print(f"   Payload: {json.dumps(r.json(), indent=2)}")
    assert r.status_code == 200
    assert r.json()["models"]["vision"]["operational"] is True

    # 2. Text
    r = requests.post(f"{BASE_URL}/api/predict/text", json={"text": "I am so happy and overjoyed today!"})
    print("\n2. POST /api/predict/text:")
    print(f"   Status: {r.status_code}")
    print(f"   Emotion: {r.json()['emotion']}, Conf: {r.json()['confidence']}")
    assert r.status_code == 200

    # 3. Audio
    audio_path = "Multimodal-Emotion-Recognition-master/audio_data/train/Actor_01/03-01-01-01-01-01-01.wav"
    with open(audio_path, "rb") as f:
        r = requests.post(f"{BASE_URL}/api/predict/audio", files={"file": ("test.wav", f, "audio/wav")})
    print("\n3. POST /api/predict/audio:")
    print(f"   Status: {r.status_code}")
    print(f"   Emotion: {r.json()['emotion']}, Conf: {r.json()['confidence']}")
    assert r.status_code == 200

    # 4. Vision
    img_path = "Multimodal-Emotion-Recognition-master/image_data/valid/Actor_21/01-01-01-01-01-01-21/30.jpg"
    with open(img_path, "rb") as f:
        r = requests.post(f"{BASE_URL}/api/predict/vision", files={"file": ("test.jpg", f, "image/jpeg")})
    print("\n4. POST /api/predict/vision:")
    print(f"   Status: {r.status_code}")
    print(f"   Emotion: {r.json()['emotion']}, Conf: {r.json()['confidence']}")
    print(f"   Probabilities: {r.json()['probabilities']}")
    assert r.status_code == 200
    assert r.json()["available"] is True
    assert r.json()["status"] == "ok"

    # 5. Multimodal (All 3 modalities: Text + Audio + Vision)
    with open(audio_path, "rb") as fa, open(img_path, "rb") as fi:
        r = requests.post(
            f"{BASE_URL}/api/predict/multimodal",
            data={"text": "I feel absolutely great and satisfied with everything."},
            files={
                "audio": ("audio.wav", fa, "audio/wav"),
                "image": ("image.jpg", fi, "image/jpeg"),
            }
        )
    print("\n5. POST /api/predict/multimodal (Text + Audio + Vision):")
    print(f"   Status: {r.status_code}")
    res = r.json()
    print(f"   Emotion: {res['emotion']}, Conf: {res['confidence']}")
    print(f"   Active modalities: {res['active_modalities']}")
    print(f"   Modality weights: {res['explainability']['modality_weights']}")
    print(f"   Concordance: {res['explainability']['concordance_score']}")
    assert r.status_code == 200
    assert set(res["active_modalities"]) == {"text", "audio", "vision"}
    assert len(res["unavailable_modalities"]) == 0

    print("\n" + "=" * 60)
    print("ALL LIVE ENDPOINTS RETURNED 200 OK WITH GENUINE INFERENCE")
    print("=" * 60)

if __name__ == "__main__":
    main()
