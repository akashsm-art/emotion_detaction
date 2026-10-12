"""
Live verification script for running FastAPI backend server on http://127.0.0.1:8000.
"""

import httpx

base_url = "http://127.0.0.1:8000"

print("--- Testing Live Endpoints ---")
with httpx.Client(base_url=base_url, timeout=30.0) as client:
    # 1. Health
    r_health = client.get("/api/health")
    print(f"GET /api/health           -> Status: {r_health.status_code}, Body: {r_health.json()}")

    # 2. Docs
    r_docs = client.get("/api/docs")
    print(f"GET /api/docs             -> Status: {r_docs.status_code}")

    # 3. ReDoc
    r_redoc = client.get("/api/redoc")
    print(f"GET /api/redoc            -> Status: {r_redoc.status_code}")

    # 4. Text
    r_text = client.post("/api/predict/text", json={"text": "I am truly delighted and joyful with these findings!"})
    t_data = r_text.json()
    print(f"POST /api/predict/text    -> Status: {r_text.status_code}, Emotion: {t_data.get('emotion')}, Conf: {t_data.get('confidence')}")

    # 5. Audio
    with open("speech-emotion-recognition-master/dataset/Sad/09b03Ta.wav", "rb") as f_audio:
        r_audio = client.post("/api/predict/audio", files={"file": ("09b03Ta.wav", f_audio, "audio/wav")})
    a_data = r_audio.json()
    print(f"POST /api/predict/audio   -> Status: {r_audio.status_code}, Emotion: {a_data.get('emotion')}, Conf: {a_data.get('confidence')}")

    # 6. Vision (expecting 503)
    with open("Multimodal-Emotion-Recognition-master/image_data/train/Actor_01/01-01-01-01-01-01-01/10.jpg", "rb") as f_img:
        r_vision = client.post("/api/predict/vision", files={"file": ("10.jpg", f_img, "image/jpeg")})
    v_data = r_vision.json()
    print(f"POST /api/predict/vision  -> Status: {r_vision.status_code}, Available: {v_data.get('available')}, Status Msg: {v_data.get('status')}")

    # 7. Multimodal (Text + Audio + Image)
    with open("speech-emotion-recognition-master/dataset/Sad/09b03Ta.wav", "rb") as f_audio, \
         open("Multimodal-Emotion-Recognition-master/image_data/train/Actor_01/01-01-01-01-01-01-01/10.jpg", "rb") as f_img:
        r_multi = client.post(
            "/api/predict/multimodal",
            data={"text": "I feel so devastated, lonely, and hopeless after losing everything."},
            files={
                "audio": ("09b03Ta.wav", f_audio, "audio/wav"),
                "image": ("10.jpg", f_img, "image/jpeg"),
            }
        )
    m_data = r_multi.json()
    print(f"POST /api/predict/multi   -> Status: {r_multi.status_code}, Emotion: {m_data.get('emotion')}, Active: {m_data.get('active_modalities')}, Unavailable: {m_data.get('unavailable_modalities')}")
