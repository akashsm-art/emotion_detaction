# Multi-Modal Emotion Recognition System

A production-grade, local-only Multi-Modal Emotion Recognition platform integrating **Text**, **Speech/Audio**, and **Facial Vision** emotion recognition with a principled **Confidence-Aware Multi-Modal Fusion Engine**, a **FastAPI backend**, and a **React 19 + TypeScript + Vite frontend**.

---

## 1. System Overview & Architecture

The system accepts natural language text, acoustic audio recordings, and facial images in any combination, processes them through deep neural networks, unifies classifications onto the canonical Ekman 7-emotion schema, and executes late decision fusion with cross-modal concordance analysis.

```text
                               ┌─────────────────────────┐
                               │       Input Media       │
                               │  Text / Audio / Vision  │
                               └────────────┬────────────┘
                     ┌──────────────────────┼──────────────────────┐
                     ▼                      ▼                      ▼
             ┌───────────────┐      ┌───────────────┐      ┌───────────────┐
             │  Text Engine  │      │ Audio Engine  │      │ Vision Engine │
             │   (RoBERTa)   │      │    (2D CNN)   │      │   (AlexNet)   │
             └───────┬───────┘      └───────┬───────┘      └───────┬───────┘
                     ▼                      ▼                      ▼
               ModalityResult         ModalityResult         ModalityResult
             (Probs + Conf)         (Probs + Conf)         (Probs + Conf)
                     └──────────────────────┬──────────────────────┘
                                            ▼
                           ┌─────────────────────────────────┐
                           │ Confidence-Aware Decision Fusion│
                           │ - Modality Availability Gate    │
                           │ - Confidence Prior Weighting    │
                           │ - JS Divergence Conflict Check  │
                           │ - Concordance & Rationale Gen   │
                           └────────────────┬────────────────┘
                                            ▼
                           ┌─────────────────────────────────┐
                           │       FusedEmotionResult        │
                           │ - Dominant Emotion Prediction   │
                           │ - Overall Model Confidence      │
                           │ - Unified Probability Vector    │
                           │ - Circumplex (Valence / Arousal)│
                           │ - Modality Attribution Weights  │
                           │ - Explainability Report         │
                           └─────────────────────────────────┘
```

---

## 2. Modality Models & Operational Reality

This system enforces a strict **Zero Fake AI Policy**: every prediction originates from genuine model forward passes. No heuristics, random guesses, or simulated probabilities are generated.

| Modality         | Model Architecture                                                      | Trained Weights Checkpoint                              | Status                | Supported Classes                                                                                     |
| ---------------- | ----------------------------------------------------------------------- | ------------------------------------------------------- | --------------------- | ----------------------------------------------------------------------------------------------------- |
| **Text**   | RoBERTa Transformer (`j-hartmann/emotion-english-distilroberta-base`) | HuggingFace Hub cached (`~300 MB`)                    | **Operational** | `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise` (All 7 Ekman classes) |
| **Speech** | 4-layer 2D CNN acoustic model on 39-channel MFCCs                       | `models/speech/best_model_CNN.h5` (`296 KB`)        | **Operational** | `neutral`, `angry`, `happy`, `sad` *(disgust, fear, surprise assigned 0.0 with zero mass)*  |
| **Vision** | AlexNet 7-Class Deep Classifier with Haar Cascade Detection             | `models/vision/model_image_new50.pth` (`217.57 MB`) | **Operational** | `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise` (All 7 Ekman classes) |

---

## 3. Vision Training Methodology & Measured Evaluation

The original `Multimodal-Emotion-Recognition-master` repository contained the PyTorch AlexNet architecture and RAVDESS video frames, but did not include the author's trained `.pth` checkpoint. To legitimately provide Vision classification without fabricating outputs, an isolated training pipeline was built in `training/vision/` to reproduce the model on the available dataset.

### A. Original Author's Configuration vs. Our Optimized Reproduction

| Parameter                            | Original Author's Notebook                                    | Our Optimized Training Run                                        |
| ------------------------------------ | ------------------------------------------------------------- | ----------------------------------------------------------------- |
| **Backbone**                   | `torchvision.models.alexnet(pretrained=True)`               | `torchvision.models.alexnet(weights=AlexNet_Weights.DEFAULT)`   |
| **Feature Extractor**          | Frozen conv layers                                            | Frozen conv layers                                                |
| **Trainable Layers**           | `classifier.1.bias`, `classifier.4.*`, `classifier.6.*` | `classifier.1.bias`, `classifier.4.*`, `classifier.6.*`     |
| **Training Loss**              | `CrossEntropyLoss` (applied over softmax)                   | `CrossEntropyLoss` on raw logits during training                |
| **Export Format**              | `classifier.6 = nn.Sequential(Linear, Softmax)`             | `classifier.6 = nn.Sequential(Linear(4096, 7), Softmax(dim=1))` |
| **Batch Size**                 | 4 (GPU)                                                       | 64 (CPU, with RAM tensor preloading)                              |
| **Learning Rate**              | 0.001                                                         | 0.016 (linearly scaled:$0.001 \times 16$)                       |
| **Optimizer**                  | `SGD(lr=0.001, momentum=0.9)`                               | `SGD(lr=0.016, momentum=0.9)`                                   |
| **Scheduler**                  | `StepLR(step_size=7, gamma=0.1)`                            | `StepLR(step_size=7, gamma=0.1)`                                |
| **Max Epochs**                 | 50                                                            | 50 (Early stopped at Epoch 41, patience=15)                       |
| **Original Reported Accuracy** | **~38.89%**                                             | *Not applicable (author's reported figure)*                     |

### B. Dataset Split & Zero Data Leakage

* **Total Clean Samples:** 15,268 face crops (0-byte corrupt files filtered)
* **Training Set:** 12,647 images strictly from `Actor_01` through `Actor_20` (20 actors)
* **Validation / Test Set:** 2,621 images strictly from `Actor_21` through `Actor_24` (4 actors)
* **Actor Leakage:** **0%** (the validation actors are completely held out and unseen during training)

### C. Actual Measured Evaluation (Held-Out Validation Set)

* **Best Measured Validation Accuracy:** **37.39%** (980 / 2,621 correct on unseen actors)
* **Macro Precision:** 0.4356
* **Macro Recall:** 0.3317
* **Macro F1-Score:** 0.3259
* **Baseline Context:** Random guessing across 7 classes is **14.28%**; the original author reported **~38.89%**.

#### Per-Class Performance Breakdown

| Class              | Precision | Recall | F1-Score | Validation Support |
| ------------------ | :-------: | :----: | :------: | :----------------: |
| **angry**    |  0.5545  | 0.1685 |  0.2585  |        362        |
| **disgust**  |  0.5146  | 0.2896 |  0.3706  |        366        |
| **fear**     |  0.4711  | 0.1686 |  0.2484  |        338        |
| **happy**    |  0.6150  | 0.4187 |  0.4982  |        332        |
| **neutral**  |  0.3236  | 0.9381 |  0.4812  |        533        |
| **sad**      |  0.2731  | 0.1752 |  0.2135  |        371        |
| **surprise** |  0.2971  | 0.1630 |  0.2105  |        319        |

#### Confusion Matrix

```text
               angr   disg   fear   happ   neut    sad   surp
angry            61     24      9     20    175     46     27
disgust          15    106     15     12    121     93      4
fear              8     14     57      4    172     25     58
happy             0      3      0    139    178      0     12
neutral           1      0      0     18    500      1     13
sad               5     47      0     18    227     65      9
surprise         20     12     40     15    172      8     52
```

---

## 4. Multi-Modal Decision Fusion Engine

The fusion layer (`core/fusion/engine.py`) implements confidence-aware weighted late decision fusion:

1. **Modality Availability Gate:** If a modality input is missing or returns an error, it is excluded and receives 0 weight.
2. **Confidence-Adjusted Weighting:** Each active modality $m$ has a configured prior weight $W_m$ and neural confidence $C_m$. Effective weight:
   $$
   w_m = \frac{W_m \cdot C_m}{\sum_{k} W_k \cdot C_k}
   $$
3. **Probability Vector Fusion:**
   $$
   P_{\text{fused}}(e) = \sum_{m} w_m \cdot P_m(e)
   $$
4. **Cross-Modal Divergence & Conflict:** Pairwise Jensen-Shannon divergence across active probability vectors detects emotional discordance.
5. **Explainability & Attribution:** Reports the dominant primary driver, concordance score (0.0 to 1.0), and a structured textual rationale.

All 7 combinations are supported and verified:

1. `Text only`
2. `Audio only`
3. `Vision only`
4. `Text + Audio`
5. `Text + Vision`
6. `Audio + Vision`
7. `Text + Audio + Vision`

---

## 5. Unified 7-Class Emotion Ontology

Predictions across all modalities align with the canonical Ekman 7-emotion schema:

```python
CORE_EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
```

Russell's Circumplex Model coordinates:

* `happy`: Valence +0.85, Arousal +0.60
* `surprise`: Valence +0.25, Arousal +0.85
* `neutral`: Valence 0.00, Arousal 0.00
* `disgust`: Valence -0.60, Arousal +0.35
* `angry`: Valence -0.65, Arousal +0.80
* `fear`: Valence -0.70, Arousal +0.75
* `sad`: Valence -0.80, Arousal -0.45

---

## 6. Project Structure

```text
erm/
├── api/                                    # FastAPI Backend
│   ├── routes/                             # Modular route handlers
│   │   ├── health.py                       # System health & readiness
│   │   ├── text.py                         # Text emotion classification
│   │   ├── audio.py                        # Speech emotion classification
│   │   ├── vision.py                       # Facial emotion classification
│   │   └── multimodal.py                   # Multi-modal fusion endpoint
│   ├── dependencies.py                     # Singleton holders & path sanitization
│   └── main.py                             # FastAPI application entrypoint
├── core/                                   # Core ML & Fusion Logic
│   ├── ontology.py                         # Ekman 7-class schema & circumplex map
│   ├── schemas.py                          # Pydantic data schemas
│   ├── text/                               # Text preprocessing & RoBERTa wrapper
│   ├── speech/                             # Acoustic CNN & MFCC feature extractor
│   ├── vision/                             # Face detector & AlexNet wrapper
│   └── fusion/                             # Confidence-aware late fusion engine
├── frontend/                               # React 19 + TypeScript + Vite Web UI
│   ├── src/
│   │   ├── components/                     # Glassmorphic UI components
│   │   ├── pages/Home.tsx                  # Main interactive dashboard
│   │   ├── services/api.ts                 # Backend HTTP client
│   │   └── types/emotion.ts                # TypeScript type definitions
│   └── package.json
├── models/                                 # Trained Model Checkpoints
│   ├── speech/best_model_CNN.h5            # Emo-DB CNN speech checkpoint (296 KB)
│   └── vision/model_image_new50.pth        # RAVDESS AlexNet vision checkpoint (217.57 MB)
├── training/                               # Training Pipelines (Isolated)
│   └── vision/                             # RAVDESS facial image training pipeline
├── tests/                                  # Backend Test Suite
│   ├── test_api.py                         # API integration & security tests
│   └── test_fusion.py                      # Multi-modal fusion unit tests
├── requirements.txt                        # Python dependencies
├── .env.example                            # Configuration template
└── README.md                               # Master documentation
```

---

## 7. Installation & Local Setup

### Prerequisites

* Windows 10/11
* Python 3.10 to 3.14 (Verified on Python 3.14.0 64-bit AMD64)
* Node.js >= 18 (npm >= 9)

### 1. Python Environment Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Frontend Setup

Install frontend dependencies:

```bash
cd frontend
npm install
cd ..
```

---

## 8. Running the Application Locally

The project runs completely locally on Windows with zero cloud requirements:

### Terminal 1: Backend API Server

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

* **API Base:** `http://127.0.0.1:8000`
* **Interactive Docs (Swagger):** `http://127.0.0.1:8000/api/docs`
* **Alternative Docs (ReDoc):** `http://127.0.0.1:8000/api/redoc`

### Terminal 2: Frontend Web UI

```bash
cd frontend
npm run dev
```

* **Frontend Web UI:** `http://127.0.0.1:5173`

---

## 9. API Reference

### `GET /api/health`

Returns service readiness and operational status across all modalities.

```json
{
  "status": "ok",
  "service": "emotion-recognition-api",
  "version": "1.0.0",
  "device": "cpu",
  "models": {
    "text": { "operational": true, "model_name": "j-hartmann/emotion-english-distilroberta-base" },
    "audio": { "operational": true, "weights": "best_model_CNN.h5" },
    "vision": { "operational": true, "status": "ok", "face_detection_available": true }
  }
}
```

### `POST /api/predict/text`

* Content-Type: `application/json`
* Body: `{"text": "I am so happy and overjoyed today!"}`

### `POST /api/predict/audio`

* Content-Type: `multipart/form-data`
* Body: `file: <audio_file.wav>` (Max 25 MB; WAV, MP3, OGG, FLAC)

### `POST /api/predict/vision`

* Content-Type: `multipart/form-data`
* Body: `file: <image_file.jpg>` (Max 10 MB; JPG, PNG, WebP, BMP)

### `POST /api/predict/multimodal`

* Content-Type: `multipart/form-data`
* Form fields:
  * `text` *(optional string)*
  * `audio` *(optional audio file)*
  * `image` *(optional image file)*

---

## 10. Automated Testing

### Backend Test Suite (PyTest)

```bash
pytest -v
```

* **21 / 21 tests passing** covering health status, single-modality endpoints, late fusion math, probability normalization, edge cases, and file safety.

### Frontend Test Suite (Vitest)

```bash
cd frontend
npm test -- --run
```

* **16 / 16 tests passing** covering UI components, API services, chart rendering, and operational badges.

### Frontend Production Build

```bash
cd frontend
npm run build
```

* Type checks with `tsc -b` and builds optimized bundle via Vite with 0 errors.

---

## 11. Security & Local Privacy

* **100% Local Inference:** No audio, image, or text data is transmitted to external cloud APIs. All model weights execute locally on `127.0.0.1`.
* **Zero Telemetry:** No tracking scripts, analytics, or remote telemetry.
* **Temporary File Sanitization:** All multipart uploads are processed in system temp folders and strictly unlinked in `finally` blocks.
* **Path Obfuscation:** The API sanitizes response metadata so internal operating system filesystem paths are never leaked to clients.

---

## 12. Known Limitations & Non-Claims

1. **Facial Emotion Accuracy:** The trained AlexNet classifier achieves **37.39%** accuracy on unseen actors in the RAVDESS dataset. While substantially higher than the 14.28% random baseline, facial emotion recognition across unfamiliar faces is inherently challenging. This figure is our honest measured result, whereas the original author reported **~38.89%**.
2. **Speech Model Coverage:** The Emo-DB CNN model predicts 4 emotions (`angry`, `happy`, `neutral`, `sad`). It does not predict `disgust`, `fear`, or `surprise`; these classes receive strict `0.0` probability mass without fabrication.
3. **Not for Clinical Diagnosis:** This system is an academic engineering integration for affective computing research. Model confidence reflects neural certainty, **not** clinical diagnosis, psychological truth, or deception detection.




python live_text.py

python live_audio.py


python live_face.py
