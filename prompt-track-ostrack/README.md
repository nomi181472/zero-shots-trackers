# Detector-Free Visual Tracking — OSTrack & STARK

Open-vocabulary single-object visual tracking with an **interactive live web interface**: upload a video or click "Load Demo Video", select an object on **Frame 1** (by drawing a bounding box or uploading an object crop), and watch the tracker follow its visual appearance across subsequent frames without needing a generic object detector like GroundingDINO.

```
                  ┌──────────────────────────────────────────────────────────┐
                  │                 Frame 1 (User Selection)                 │
                  │   User draws bounding box OR provides crop of target     │
                  └────────────────────────────┬─────────────────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │  Initialize Appearance Model     │
                              │  OSTrack (Search Window + Hann)  │
                              │  STARK   (Adaptive Dual-Template)│
                              │  Mock    (Multi-Scale Correlation│
                              └────────────────┬─────────────────┘
                                               │
                                 Frame 2..N ───┴───▶ Update Target Appearance
                                                     & Predict (x1, y1, x2, y2)
                                                     without generic object detection
```

## Trackers

| Tracker | Strategy | Description |
| :--- | :--- | :--- |
| **`ostrack`** | One-Stream Search Window Correlation | Extracts a spatial search region centered on the prior target position, applying a 2D cosine/Hanning spatial prior to match the target appearance. |
| **`stark`** | Spatio-Temporal Adaptive Reliability | Maintains dual appearance templates: the ground-truth template from frame 1 and a dynamic online template updated when tracking confidence is high to adapt to lighting/viewpoint changes. |
| **`mock`** | Multi-Scale Template Matching | Fast OpenCV normalized cross-correlation (`cv2.matchTemplate`) baseline requiring zero heavyweight weights or neural dependencies. |

## Quick Start

Run both the FastAPI backend (`:8000`) and the Next.js frontend (`:3000`) with a single command:

```bash
cd prompt-track-ostrack
./start.sh
```

Open **http://localhost:3000**:
1. Click **⚡ Load Demo Video** (or drag & drop any `.mp4` file).
2. On **Frame 1**, click and drag a box around the object you want to track (or click "Center Box").
3. Choose your tracking algorithm (**OSTrack**, **STARK**, or **Mock**).
4. Click **▶ Start Tracking** to stream the annotated video in real-time with motion trails and live bounding box coordinates.

## Architecture

```
   Browser (Next.js :3000)
      │   <img src="/api/stream/{id}">          Frame 1 Canvas Box Selection
      ▼                                            POST /api/track
   Next rewrite ──/api/*──►  FastAPI (:8000)
                              │
                ┌─────────────┴──────────────┐
         POST /api/first-frame        GET /api/stream/{id}
         POST /api/track              (multipart/x-mixed-replace MJPEG)
                         ▼
                ┌─────────────────┐
                │  StreamSession  │  background worker thread
                └─────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │ tracker   OSTrack / STARK / Mock   │  frame 1 template -> follow target
        │ visualizer draw box / trail / pill │
        └────────────────────────────────────┘
```

## API Reference

| Method | Endpoint | Purpose |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Service status, loaded tracker capabilities, and active sessions |
| `GET` | `/api/trackers` | List of available tracking backends (`ostrack`, `stark`, `mock`) |
| `POST` | `/api/first-frame` | Extracts Frame 1 from a video for interactive UI box selection |
| `GET` | `/api/demo-video` | Generates on-the-fly synthetic test video with moving target |
| `POST` | `/api/track` | Starts a tracking session with video + initial box / crop |
| `POST` | `/api/rtsp` | Starts tracking a live RTSP / RTMP / HTTP camera stream |
| `GET` | `/api/stream/{id}` | Live MJPEG stream rendered directly inside `<img>` tags |
| `POST` | `/api/control/{id}`| Live parameter adjustment (switch tracker, toggle trails, loop) |
| `GET` | `/api/status/{id}` | Real-time tracking statistics (frames, coordinates, FPS) |
| `POST` | `/api/stop/{id}` | Stops a tracking session |

## Running Tests

To run the automated test suite:

```bash
PYTHONPATH=. pytest tests/ -v
```
