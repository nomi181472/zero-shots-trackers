# Zero-Shots Trackers

A unified monorepo of open-vocabulary and zero-shot visual object tracking web applications with modern Next.js frontends and high-performance FastAPI backends.

Each project is fully standalone with its own environment, dependencies, tests, and web UI, sharing a common streaming architecture while implementing fundamentally different identity association paradigms.

---

## Architecture Comparison

| Project | Tracking Paradigm | Detection Method | Identity / Association Model | Target Use Case |
| :--- | :--- | :--- | :--- | :--- |
| [`prompt-track/`](./prompt-track) | **Algorithmic Multi-Object Tracking** | GroundingDINO (Open-Vocabulary Text Prompt) | **Kalman Filter + IoU + Hungarian** (ByteTrack, SORT, Greedy IoU) | Fast, CPU-friendly multi-object tracking based on motion continuity. |
| [`prompt-track-rag/`](./prompt-track-rag) | **Retrieval-Augmented Multi-Object Tracking** | GroundingDINO (Open-Vocabulary Text Prompt) | **RAG Vector Memory** (CLIP / Histogram embeddings + Score/LLM Generator) | Complex multi-object scenes with re-identification after occlusions and camera cuts. |
| [`prompt-track-ostrack/`](./prompt-track-ostrack) | **Detector-Free Visual Single-Object Tracking** | Interactive Frame 1 Canvas Selection (Box / Crop) | **Appearance Correlation** (OSTrack search window, STARK dual-template, Mock NCC) | Real-time visual tracking of arbitrary targets without predefined detector classes or text prompts. |

---

## System Architecture

All three applications share a high-performance, non-blocking asynchronous pipeline:

```
   ┌────────────────────────────────────────────────────────┐
   │             Next.js 15 Frontend (:3000)                │
   │  - Video Drag & Drop / RTSP URL / Frame 1 Canvas Box   │
   │  - Live Controls: confidence, prompt, tracker switch   │
   │  - Live Statistics: FPS, frame index, active track IDs │
   └───────────────────────────┬────────────────────────────┘
                               │  Reverse proxy /api/*
                               ▼
   ┌────────────────────────────────────────────────────────┐
   │                 FastAPI Backend (:8000)                │
   │  POST /api/track ─────────► Spawns background worker   │
   │  GET  /api/stream/{id} ───► MJPEG multipart stream     │
   │  POST /api/control/{id} ──► Real-time runtime updates  │
   │  GET  /api/status/{id} ───► Live session telemetry     │
   └───────────────────────────┬────────────────────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
   ┌───────────────────────┐             ┌───────────────────────┐
   │  Detection / Initial  │             │  Tracking & Re-ID     │
   │  GroundingDINO (text) │             │  - ByteTrack / SORT   │
   │  OR Frame 1 Box Crop  │             │  - RAG Memory (CLIP)  │
   └───────────┬───────────┘             │  - OSTrack / STARK    │
               │                         └───────────┬───────────┘
               └──────────────────┬──────────────────┘
                                  ▼
                     ┌─────────────────────────┐
                     │ Annotator & Visualizer  │
                     │ Bounding boxes, trails, │
                     │ IDs, and status pills   │
                     └────────────┬────────────┘
                                  ▼
               MJPEG Output (24+ FPS directly to <img>)
```

---

## Repository Structure

```
zero-shots-trackers/
├── .gitignore                     # Monorepo-wide git exclusion rules
├── README.md                      # Project documentation and guide
│
├── prompt-track/                  # GroundingDINO + Classical Motion Tracking
│   ├── backend/                   # FastAPI backend service
│   ├── frontend/                  # Next.js web application
│   ├── zero_shot_tracking/        # Detection & ByteTrack/SORT tracking engine
│   ├── configs/                   # Configuration files
│   ├── scripts/                   # Weight download and video helpers
│   ├── tests/                     # Test suite
│   ├── setup.sh                   # Environment setup script
│   └── start.sh                   # One-command startup script
│
├── prompt-track-rag/              # GroundingDINO + RAG Vector Re-ID Tracking
│   ├── backend/                   # FastAPI backend service
│   ├── frontend/                  # Next.js web application
│   ├── zero_shot_tracking/        # RAG pipeline, CLIP embedder & vector memory
│   ├── configs/                   # Configuration files
│   ├── scripts/                   # Weight download and video helpers
│   ├── tests/                     # Test suite
│   ├── setup.sh                   # Environment setup script
│   └── start.sh                   # One-command startup script
│
└── prompt-track-ostrack/          # Interactive Frame 1 Box + OSTrack/STARK
    ├── backend/                   # FastAPI backend service
    ├── frontend/                  # Next.js canvas-based interactive web UI
    ├── zero_shot_tracking/        # OSTrack, STARK, and NCC template matchers
    ├── tests/                     # Test suite
    ├── setup.sh                   # Environment setup script
    └── start.sh                   # One-command startup script
```

---

## Quick Start

### Prerequisites
- **Python**: 3.10 – 3.12 (Python 3.13+ is not yet supported by official GroundingDINO CUDA extensions).
- **Node.js**: v18.0.0 or higher, with `npm`.
- **FFmpeg**: Required for video reading and stream encoding.
- **CUDA GPU** (Recommended): Highly recommended for real-time neural detection (GroundingDINO) and neural embeddings (CLIP). CPU fallback (`ZST_DETECTOR=mock`) is fully supported out of the box.

---

### 1. Classical Algorithmic Tracking (`prompt-track`)

Tracks multiple objects using text queries with GroundingDINO and associates identities via ByteTrack Kalman filtering:

```bash
cd prompt-track

# Run with mock detector (no GPU/weights required)
./start.sh

# Or setup full neural model with CUDA:
CUDA=true ./setup.sh
ZST_DETECTOR=gdino ./start.sh
```

- Web UI: [http://localhost:3000](http://localhost:3000)
- Backend API: [http://localhost:8000](http://localhost:8000)

---

### 2. Retrieval-Augmented Tracking (`prompt-track-rag`)

Maintains an appearance vector memory store for every track, retrieving nearest visual embeddings using CLIP or color histograms, and assigning IDs via score fusion or an LLM:

```bash
cd prompt-track-rag

# Run with mock detector and color histogram embedder (lightweight)
./start.sh

# Or run with full neural pipeline (GroundingDINO + CLIP):
CUDA=true ./setup.sh
ZST_DETECTOR=gdino ZST_EMBEDDER=clip ./start.sh
```

- Web UI: [http://localhost:3000](http://localhost:3000)
- Backend API: [http://localhost:8000](http://localhost:8000)

---

### 3. Detector-Free Appearance Tracking (`prompt-track-ostrack`)

No object detector or text prompt required. Upload a video or load demo, draw a bounding box over any arbitrary object on **Frame 1**, and track it using OSTrack or STARK appearance models:

```bash
cd prompt-track-ostrack

# Start both backend and frontend
./start.sh
```

- Web UI: [http://localhost:3000](http://localhost:3000)
- Backend API: [http://localhost:8000](http://localhost:8000)

---

## Running Multiple Trackers Simultaneously

To run multiple trackers side-by-side on the same machine without port conflicts, override the backend and frontend ports:

```bash
# Terminal 1: prompt-track on :8000 / :3000
cd prompt-track
./start.sh

# Terminal 2: prompt-track-rag on :8010 / :3010
cd prompt-track-rag
BACKEND_ORIGIN=http://127.0.0.1:8010 uvicorn backend.main:app --port 8010 &
cd frontend && BACKEND_ORIGIN=http://127.0.0.1:8010 npm run dev -- -p 3010

# Terminal 3: prompt-track-ostrack on :8020 / :3020
cd prompt-track-ostrack
PORT_BACKEND=8020 PORT_FRONTEND=3020 ./start.sh
```

---

## Configuration & Environment Variables

| Variable | Values | Projects | Description |
| :--- | :--- | :--- | :--- |
| `ZST_DETECTOR` | `gdino` \| `mock` | `prompt-track`, `prompt-track-rag` | Chooses between real GroundingDINO detector or zero-dependency synthetic mock. |
| `ZST_DEVICE` | `cuda` \| `cpu` | All | Target compute device for PyTorch inference. |
| `ZST_EMBEDDER` | `clip` \| `histogram` \| `auto` | `prompt-track-rag` | Appearance feature extractor for RAG memory (CLIP ViT or RGB histogram). |
| `ZST_RAG_GENERATOR` | `score` \| `llm` \| `auto` | `prompt-track-rag` | Strategy used to generate track ID matches (cosine+IoU score vs. LLM prompt). |
| `ZST_LLM_BASE_URL` | URL string | `prompt-track-rag` | OpenAI-compatible endpoint (e.g., `http://localhost:11434/v1` for local Ollama). |
| `ZST_LLM_MODEL` | model identifier | `prompt-track-rag` | Model name for LLM-based re-ID generation (e.g., `llama3.2`). |
| `ZST_TRACKER` | `ostrack` \| `stark` \| `mock` | `prompt-track-ostrack` | Default visual tracking algorithm. |
| `BACKEND_ORIGIN` | URL string | All Frontends | Target backend API URL for Next.js API route rewrites. |

---

## API Reference Overview

All backend services expose a consistent REST and streaming API:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Service status, active sessions, and loaded model capabilities. |
| `POST` | `/api/first-frame` | Extract the first frame of an uploaded video for UI target selection. |
| `POST` | `/api/track` | Initialize a tracking session from an uploaded video file. |
| `POST` | `/api/rtsp` | Initialize tracking on a live RTSP/RTMP/HTTP video stream. |
| `GET` | `/api/stream/{session_id}` | Live annotated MJPEG video stream (consumable directly via `<img>`). |
| `POST` | `/api/control/{session_id}` | Dynamically change runtime parameters (confidence, prompt, tracker, loop). |
| `GET` | `/api/status/{session_id}` | Real-time session metrics (FPS, current frame, active tracks, bounding boxes). |
| `POST` | `/api/stop/{session_id}` | Gracefully terminate an active streaming session and release resources. |

---

## Running Automated Tests

Each subproject contains automated tests with mock backends requiring no GPU:

```bash
# Test classical tracking suite
cd prompt-track
pytest tests/ -v

# Test RAG tracking suite
cd ../prompt-track-rag
ZST_DETECTOR=mock ZST_EMBEDDER=histogram pytest tests/ -v

# Test OSTrack / STARK visual tracking suite
cd ../prompt-track-ostrack
PYTHONPATH=. pytest tests/ -v
```

---

## Citations & Acknowledgments

- **GroundingDINO**: Liu et al., *Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection*, ECCV 2024.
- **ByteTrack**: Zhang et al., *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*, ECCV 2022.
- **SORT**: Bewley et al., *Simple Online and Realtime Tracking*, ICIP 2016.
- **CLIP**: Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*, ICML 2021.
- **OSTrack**: Ye et al., *Joint Feature Learning and Relation Modeling for Tracking: A One-Stream Framework*, ECCV 2022.
- **STARK**: Yan et al., *Learning Spatio-Temporal Transformer for Visual Tracking*, ICCV 2021.