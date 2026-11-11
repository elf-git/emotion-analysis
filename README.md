# Explainable Multimodal Emotion Recognition and Communication Analysis

An M.Sc. AI & ML project for explainable multimodal emotion recognition using the MELD dataset with deep learning, uncertainty quantification, dynamic modality gating, and interactive web visualization.

---

## ⚠️ Current Status: What Actually Runs

The app runs **public pretrained models, not models trained by this project**:

| Part | Model | Limits |
|---|---|---|
| Text | `j-hartmann/emotion-english-distilroberta-base` | English only, 7 emotions |
| Face | `trpakov/vit-face-expression` + OpenCV face detection | 7 emotions, largest face only, not linked to the speaker |
| Voice | `superb/wav2vec2-base-superb-er` | 4 emotions only, unreliable on film audio (labels white noise as joy), so it is used only where speech is recognized and has a low weight |
| Speech | Faster-Whisper `small` | Malayalam and Tamil quality is weak |
| Fusion | Fixed weighted late fusion (0.40 text, 0.15 voice, 0.45 face) | Weights set by hand, not fitted on labelled data |

- Missing evidence is reported as `unknown`. It is never guessed or treated as neutral.
- Hindi, Malayalam and Tamil text is scored with a keyword baseline and through English translation. IndicTrans2 and a native-script emotion model are not installed.
- **The accuracy figures in this README and in the Evaluation tab are unverified.** No trained weights are in this repository, and the only saved measurements are 1 to 2 epoch smoke tests (test accuracy 31.0% text, 16.5% audio, 26.5% video). The system has not been evaluated on MELD.
- Parts of the sections below describe the planned design, including DistilBERT, ResNet-18, IndicTrans2, openSMILE and gated fusion. They are not what the app currently runs.
- Models download to `.model_cache/` on first use, about 1.5 GB.

---

## 🚀 Phase 5: Web Application & Production Inference Platform

The system features a **FastAPI** backend for low-latency multimodal neural inference and a modern **Next.js** frontend with glassmorphic UI aesthetics, dynamic charts, prediction timelines, uncertainty metrics, and visual Grad-CAM overlays.

### Key Capabilities
- 🎯 **7-Class Multimodal Emotion Classification**: `neutral`, `surprise`, `fear`, `sadness`, `joy`, `disgust`, `anger`.
- 🔤 **Text Modality**: Transformer utterance representation with word-level attribution scoring.
- 🎵 **Audio Modality**: Mel-spectrogram & pitch/prosodic extraction with audio energy dynamics.
- 🎬 **Video Modality**: ResNet frame feature aggregation and **Visual Grad-CAM** spatial attention heatmaps.
- 🔗 **Multimodal Fusion & Gating**: Dynamic attention gate yielding explicit modality importance weights ($\alpha_T, \alpha_A, \alpha_V$).
- 📊 **Uncertainty Quantification**:
  - Normalized Shannon Entropy: $H_{norm} = -\sum p_i \log_7(p_i) \in [0, 1]$
  - Top-2 Margin: Separation between highest and second-highest predictions
  - 95% Confidence Interval error bounds
- 💬 **Dialogue Timeline Engine**: Sequential tracking of multi-turn conversational dynamics, sentiment shifts, and turn-by-turn emotion trajectories.
- 🛡️ **Safe Media Handling & Limits**:
  - Max Audio File Size: **15 MB** (`.wav`, `.mp3`, `.ogg`, `.flac`)
  - Max Video File Size: **30 MB** (`.mp4`, `.avi`, `.mov`, `.mkv`)
  - Temporary files cleaned up via secure context handlers.

---

## 📁 Repository Structure

```
emotion-analysis/
├── backend/                        # FastAPI Neural Inference Server
│   ├── main.py                     # API routing, CORS, and endpoint definitions
│   ├── services/
│   │   ├── inference_service.py    # PyTorch inference, gating, uncertainty calculation
│   │   └── media_processor.py      # Audio prosody extraction & Video Grad-CAM heatmaps
│   └── tests/
│       └── test_api.py             # Pytest suite with 8 end-to-end integration tests
├── frontend/                       # Next.js 16 Web Application (App Router)
│   ├── app/
│   │   ├── layout.js               # Root layout, Google Fonts (Plus Jakarta Sans)
│   │   ├── page.js                 # Interactive 3-tab dashboard (Predictor, Timeline, Benchmark)
│   │   └── globals.css             # Vanilla CSS glassmorphic design system
│   └── next.config.mjs             # API proxy rewrites to FastAPI backend
├── ml/
│   ├── models/
│   │   ├── fusion_model.py         # Multimodal gated cross-attention fusion architecture
│   │   ├── text_model.py           # Text classification branch (DistilBERT)
│   │   ├── audio_model.py          # Audio classification branch
│   │   ├── video_model.py          # Video classification branch
│   │   └── late_fusion.py          # Late fusion baseline ensembles
│   ├── explainability/
│   │   ├── text_attribution.py     # Token-level integrated gradients & attention attribution
│   │   └── grad_cam.py             # Visual Grad-CAM spatial heatmap generator
│   └── evaluation/
│       └── evaluate_ablation.py    # Modality ablation evaluation benchmark
├── checkpoints/                    # Local PyTorch model weights
├── exports/                        # Official evaluation benchmarks and trained checkpoints
├── notebooks/                      # Kaggle 2x T4 distributed training notebooks
├── run_app.py                      # One-click dual server launcher
├── docker-compose.yml              # Production container orchestration
├── Dockerfile.backend              # Backend Docker image specification
├── Dockerfile.frontend             # Frontend Docker image specification
├── requirements.txt                # Python dependencies
└── README.md
```

---

## ⚡ Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+ and npm
- (Optional) CUDA-compatible GPU

### 1. Install Dependencies

**Python Backend Dependencies:**
```bash
pip install -r requirements.txt
```

**Frontend Dependencies:**
```bash
cd frontend
npm install
cd ..
```

---

### 2. Run the Full Application (One-Click)

Launch both the **FastAPI backend** (port 8000) and **Next.js frontend** (port 3000) concurrently:
```bash
python run_app.py
```

Once running:
- **Web Interface**: [http://localhost:3000](http://localhost:3000)
- **FastAPI Interactive Swagger Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **API Health Check**: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

---

### 3. Alternative: Run Services Manually

**Terminal 1 (Backend):**
```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 2 (Frontend):**
```bash
cd frontend
npm run dev
```

---

### 4. Run Automated Tests

The backend test suite covers health checks, single/multimodal inputs, file uploads, dialogue timelines, ablation benchmarks, and file validation errors:
```bash
python -m pytest backend/tests/test_api.py -v
```

All 8 tests pass with zero failures:
```
backend/tests/test_api.py::test_health_endpoint PASSED
backend/tests/test_api.py::test_predict_text_only PASSED
backend/tests/test_api.py::test_predict_empty_input_fails PASSED
backend/tests/test_api.py::test_predict_with_audio_upload PASSED
backend/tests/test_api.py::test_predict_with_video_upload PASSED
backend/tests/test_api.py::test_dialogue_timeline PASSED
backend/tests/test_api.py::test_benchmark_endpoint PASSED
backend/tests/test_api.py::test_sample_dialogues_endpoint PASSED
```

---

## 🐳 Docker Deployment

To build and run the entire stack using Docker Compose:

```bash
docker-compose up --build
```

- Backend runs in container `meld_emotion_backend` on `http://localhost:8000`.
- Frontend runs in container `meld_emotion_frontend` on `http://localhost:3000`.

---

## 📡 API Reference

### 1. `POST /api/predict/multimodal`
Predicts emotion from any combination of Text, Audio, and Video files.

**Request (Multipart Form-Data):**
- `text` (optional string): Utterance transcription.
- `audio` (optional file): Audio recording (`.wav`, `.mp3`). Max 15MB.
- `video` (optional file): Video recording (`.mp4`, `.avi`). Max 30MB.

**Response Structure:**
```json
{
  "predicted_emotion": "joy",
  "confidence": 0.8842,
  "confidence_interval": [0.8342, 0.9342],
  "class_probabilities": {
    "neutral": 0.031,
    "surprise": 0.042,
    "fear": 0.005,
    "sadness": 0.012,
    "joy": 0.8842,
    "disgust": 0.008,
    "anger": 0.0178
  },
  "modality_contributions": {
    "text": 0.521,
    "audio": 0.284,
    "video": 0.195
  },
  "uncertainty": {
    "shannon_entropy": 0.183,
    "top2_margin": 0.8422,
    "is_confident": true
  },
  "explainability": {
    "text_attributions": [
      {"token": "That", "importance": 0.12},
      {"token": "is", "importance": 0.08},
      {"token": "amazing", "importance": 0.65}
    ],
    "video_gradcam_base64": "data:image/jpeg;base64,...",
    "audio_prosody": {
      "mean_pitch_hz": 194.2,
      "energy_rms": 0.082
    }
  }
}
```

### 2. `POST /api/predict/dialogue`
Evaluates an entire conversational exchange across speakers and returns the emotional trajectory over time.

### 3. `GET /api/benchmark`
Returns official empirical test split ablation comparisons across all modality permutations ($T$, $A$, $V$, $T+A$, $T+V$, $A+V$, $T+A+V$).

---

## 🎬 Movie Scene Discovery (SerpApi) & Full-Clip Emotion Analysis

The system integrates **SerpApi YouTube Search** and **Full-Clip Multimodal Video Analysis** for cinematic dialogue emotion recognition:

### 1. SerpApi Movie Scene Discovery
- Search cinematic scenes via:
  - **Movie Name** (e.g. *Kumbalangi Nights*, *3 Idiots*, *Super Deluxe*, *The Dark Knight*)
  - **Scene Description** (e.g. *dinner confrontation*, *rooftop argument*, *confession*)
  - **Character Name** (optional, e.g. *Shammi*, *Rancho*, *Shilpa*)
  - **Language** (`en`, `hi`, `ml`, `ta`, `all`)
  - **Emotion Tag** (`joy`, `anger`, `sadness`, `surprise`, `fear`, `disgust`, `neutral`, `all`)
- Returns structured metadata: `video_id`, `title`, `thumbnail_url`, `channel_name`, `duration`, `video_url`, `embed_url`, and `search_metadata`.
- Embedded official **YouTube IFrame Player API** for responsive in-app scene preview.
- **YouTube URL Direct Analysis**: Users can analyze YouTube movie scenes directly by entering any valid YouTube URL or 11-character video ID, powered by `yt-dlp` stream extraction (`POST /api/scenes/analyze-youtube`), accompanied by local streaming (`GET /api/scenes/video-stream/{video_id}`) and a real-time progress percentage bar.
- **Quota & Offline Resiliency**: Seamless fallback to curated multilingual movie scene archives if `SERP_API_KEY` is not provided, if offline, or if an API quota is reached.
- **Dual Ingestion Options**: Analyze clips either directly via YouTube links or by dragging and dropping authorized local video files (`.mp4`, `.webm`, `.mov`).

### 2. Full-Clip Video Emotion Analysis Pipeline
- Accepts authorized uploaded video files (`.mp4`, `.webm`, `.avi`, `.mov`).
- **Configurable Frame & Segment Parameters**:
  - `frame_sampling_interval_sec`: Sampling frequency (0.5s - 2.0s, default 1.0s).
  - `segment_interval_sec`: Temporal window duration (2.0s - 5.0s, default 3.0s).
  - `language_override`: Auto-detect or specify spoken language (`en`, `hi`, `ml`, `ta`).
  - `include_translation`: Toggle automatic English translation for Indic speech transcripts.
- **Pipeline Stages**:
  1. Video validation and format inspection.
  2. Audio track extraction with FFmpeg.
  3. Speech transcription with multilingual Faster-Whisper, generating timestamped segments.
  4. Frame extraction at sampled timestamps.
  5. Segment linguistic emotion analysis (Approach A Indic direct + Approach B translated MELD DistilBERT).
  6. Segment acoustic prosody extraction (fundamental pitch $F_0$, RMS energy).
  7. Segment visual emotion classification with **ResNet-18 Grad-CAM** saliency heatmaps.
  8. Cross-modal dynamic gating fusion per segment ($\alpha_T, \alpha_A, \alpha_V$).
  9. Complete scene-level chronological aggregation:
     - Dominant predicted emotion & average valence/arousal
     - Emotion transition flow (e.g. `neutral ➔ surprise ➔ anger`)
     - 7-class scene emotion distribution percentages
     - Interactive player synchronized timeline with seek scrubbing and deep inspection.

### 3. Movie Scene API Endpoints:
- `GET /api/scenes/search`: Search YouTube movie scenes via SerpApi with language and emotion filtering.
- `GET /api/scenes/featured`: Returns curated high-emotional-intensity scenes categorized by language.
- `POST /api/scenes/analyze-clip`: Full-clip multimodal emotion analysis on uploaded video files.

### 4. Environment Variables (`.env`)
Create a `.env` file in the project root:
```env
# Optional: SerpApi Key for live YouTube movie scene discovery
# If omitted or quota exceeded, the system automatically falls back to curated mock scenes
SERP_API_KEY=your_serpapi_key_here
```

---

## 🌐 Multilingual Emotion Recognition & Full-Clip Scene Analysis

The application supports **English (`en`)**, **Hindi (`hi`)**, **Malayalam (`ml`)**, and **Tamil (`ta`)** with modular, culturally calibrated pipelines:

### 1. Language Detection & Manual Override
- Automatic Unicode script block boundary identification (Devanagari, Malayalam, Tamil, Latin) and character distributions.
- Returns language code, language name, script name, and detection confidence.
- Users can override auto-detection by passing `override_language` (`en`, `hi`, `ml`, `ta`).

### 2. Dual-Approach Text Emotion Analysis
- **Approach A (Direct Multilingual Model)**: Evaluates native Indic script directly using multilingual representations, extracting token attributions in native script.
- **Approach B (Translation-Based Model)**: Translates Hindi, Malayalam, or Tamil into English via IndicTrans2, then evaluates with the MELD-trained DistilBERT model.
- **Comparison Engine**: Computes agreement, probability divergence, consensus emotion, and flags cases where translation smooths away culturally specific emotional expressions.

### 3. Multilingual Speech Processing (Whisper)
- Transcribes conversational audio in English, Hindi, Malayalam, and Tamil with automated speech language detection.
- Preserves native script transcripts, segment timestamps (`start`, `end`), and optional English translation.

### 4. Multilingual Full-Clip Scene Analysis
- **End-to-End Video Processing**:
  1. Extracts audio track via FFmpeg.
  2. Transcribes speech and segments timestamps with Whisper.
  3. Samples facial frames per segment and overlays visual **Grad-CAM** saliency heatmaps.
  4. Extracts acoustic prosody (pitch, RMS energy, syllable rate) per segment.
  5. Evaluates text via both Approach A (native) and Approach B (translated).
  6. Fuses modalities dynamically per turn and aggregates the complete chronological scene timeline with emotion transitions and clip-level distribution.

### 5. Multilingual Evaluation Suite
Run the reproducible multilingual evaluation across all 4 languages:
```bash
python ml/evaluation/evaluate_multilingual.py
```
Outputs:
- Markdown Benchmark Report: `exports/results_multilingual_benchmark.md`
- Structured JSON Results: `exports/multilingual_benchmark_results.json`

### Multilingual API Endpoints:
- `GET /api/multilingual/languages`: Metadata for English, Hindi, Malayalam, and Tamil.
- `POST /api/multilingual/detect-language`: Automatic script & language detection.
- `POST /api/multilingual/predict/text`: Dual-approach text emotion prediction.
- `POST /api/multilingual/transcribe`: Speech-to-text with timestamps and translation.
- `POST /api/multilingual/analyze/clip`: Comprehensive full-clip video scene timeline analysis.
- `GET /api/multilingual/benchmark`: Empirical evaluation results comparing Approach A and Approach B.


---

## 🔬 Dataset & Benchmark Summary

Trained and evaluated on the official **MELD** (Multimodal EmotionLines Dataset) benchmark:
- **Classes**: 7 emotions (`neutral`, `surprise`, `fear`, `sadness`, `joy`, `disgust`, `anger`)
- **Official Splits**: 9,989 Train / 1,109 Dev / 2,610 Test utterances
- **Fusion Modality Ablations**:
  - Text-Only Baseline: 61.4% Weighted F1
  - Late Fusion ($T+A+V$): 62.8% Weighted F1
  - Gated Multimodal Fusion ($T+A+V$ with Dynamic Cross-Attention): **65.1% Weighted F1**

---

## 📜 License
Academic research use only. MELD dataset licensed under GPL-3.0.
