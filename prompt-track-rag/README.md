# Zero-Shot Tracking example — RAG tracker
## GroundingDINO + RAG (retrieval-augmented) tracker + live MJPEG web app

Open-vocabulary multi-object tracking with a **live web preview**: upload a
video, type any classes, and watch the annotated stream appear in a plain
`<img src=...>` tag. While it runs you can swap the tracker
(RAG / ByteTrack / SORT / IoU) and retune thresholds — no restart, no training.

This project is the **RAG-based** sibling of the algorithm-based tracker. The
detector stays GroundingDINO (same prompt-based detection), but the tracker no
longer uses an algebraic motion model (Kalman + IoU + Hungarian). Instead the
detection → association step is a retrieval-augmented decision:

```
     detect (GroundingDINO)          retrieve (vector store)        generate
  frame ──▶ boxes for "person, car" ──▶ top-k similar track     ──▶ track id per box
                          │                memories by cosine          (score head,
                    embed crop              (CLIP / histogram)         or optional LLM)
                    with CLIP ────────────────────────▶ MemoryStore
```

```
   browser (Next.js :3000)
      │   <img src="/api/stream/{id}">          upload + live controls
      ▼                                            (POST→/api/control/{id})
   Next rewrite ──/api/*──►  FastAPI (:8000)
                              │
                ┌─────────────┴──────────────┐
        POST /api/track ─┐            GET /api/stream/{id}
                         ▼            (multipart/x-mixed-replace MJPEG)
                ┌─────────────────┐
                │  StreamSession  │  worker thread per upload
                │  (daemon)       │
                └─────────────────┘
                         │  loop over frames
                         ▼
        ┌────────────────────────────────────┐
        │ detector  GroundingDINO / mock      │  text prompt → boxes
        │ embedder  CLIP / histogram         │  crops → appearance vectors
        │ tracker   RAG (default)            │  retrieve → generate → assign
        │ tracker   ByteTrack, SORT, IoU     │  (optional baselines)
        │ annotator draw boxes/ids/trails     │
        └────────────────────────────────────┘
                         ▼
                latest annotated frame (jpeg)
                → served at up to 24 fps
```

## The RAG tracker

`zero_shot_tracking/rag.py` implements all three parts:

| stage | component | what it does |
|-------|-----------|--------------|
| **R**etrieval | `TrackEmbedder` (`embedder.py`) | detection crops → L2-normalized vectors. `CLIPEmbedder` (transformers, `openai/clip-vit-base-patch32`) is the real encoder; `HistogramEmbedder` (HSV grid + shape) is the torch-free fallback / test stand-in. |
| **A**ugmented | `TrackMemory` + `MemoryStore` | each identity is a *memory bank*: a rolling window of appearance embeddings, a running centroid, predicted box, EMA velocity, class, hit/lost bookkeeping. This is the in-memory vector database (embed → index → retrieve). |
| **G**eneration | `ScoreGenerator` / `LLMGenerator` | consumes the retrieved context (appearance cosine + positional IoU with the predicted box, fused as `w_appearance`/`w_position`) and *generates the association* — one-to-one with a score gate, or, for `LLMGenerator`, a natural-language prompt answered by an OpenAI-compatible chat model (Ollama by default). |

Trackers available (swappable **while streaming**):

| tracker | model | notes |
|---------|-------|-------|
| `rag` | CLIP/histogram retrieval + score/LLM generation | **default** — no Kalman filter |
| `bytetrack` | Kalman + two-stage IoU (ECCV 2022) | rescues low-score boxes |
| `sort` | Kalman + Hungarian IoU (CVPR 2016) | SORT classic |
| `iou` | greedy IoU, no motion model | lightest baseline |

Retrieval knobs (live-tunable from the UI):
`w_appearance` / `w_position` (fusion weights), `memory_slots` (appearance
slots kept per track), `top_k` (retrieval candidates per detection),
`match_thresh` (minimum fused score to associate), `track_buffer` (frames a
memory survives while lost).

## Layout

```
├── run_tracking.py              # CLI (single video, no server)
├── configs/config.yaml          # CLI defaults
├── zero_shot_tracking/
│   ├── detector.py              # GroundingDINO wrapper (lazy deps)
│   ├── detectors.py             # detector factory + mock blob detector
│   ├── embedder.py              # RAG encoder: CLIP + histogram fallback
│   ├── rag.py                   # RAG tracker: memory store + generators
│   ├── trackers.py              # SORT + IoU baseline (numpy)
│   ├── registry.py              # create_tracker("rag|bytetrack|sort|iou")
│   ├── tracker.py               # ByteTrack port (numpy Kalman)
│   ├── visualizer.py            # boxes, IDs, trails, legend
│   └── pipeline.py              # CLI pipeline
├── backend/                     # FastAPI streaming server
│   ├── main.py                  # routes: track/stream/control/status/stop
│   ├── stream.py                # sessions + MJPEG generator (async)
│   ├── worker.py                # read→detect→embed→retrieve→annotate thread
│   └── params.py                # RunParams (live-tunable settings)
├── frontend/                    # Next.js (App Router, :3000)
│   ├── app/page.tsx             # upload, <img> stream, runtime tracker switch
│   └── next.config.mjs          # /api/* rewrite → http://127.0.0.1:8000
├── scripts/download_weights.py  # GroundingDINO Swin-T checkpoint
├── tests/test_rag.py            # RAG retrieval/generation unit tests
├── tests/                       # API, and live-TCP integration tests
└── setup.sh                     # venv + torch + GroundingDINO + CLIP
```

## Zero-shot

GroundingDINO is trained once on large grounding data. At inference you swap
the category list freely — `"person, car"`, `"poodle, kite"` — with zero
training. The RAG tracker re-identifies each detected box across frames by
retrieving its most similar appearance memories, so identity does not depend on
a hand-tuned motion model.

## Run it (full web app)

Two processes. The real GroundingDINO detector works on Python 3.14 via
`zero_shot_tracking/gdino_compat.py` (reattaches the BERT mask helpers that
transformers ≥4.45 removed) and needs no C extensions: with CPU-only torch,
`MultiScaleDeformableAttention` falls back to the pure-PyTorch kernel, so the
CUDA ops never need compiling. The mock detector needs no torch/weights at all.

The CLIP retrieval encoder (`ZST_EMBEDDER=clip`) downloads its weights on first
use from HuggingFace; if that fails (offline/no network) the app automatically
falls back to the torch-free histogram embedder, so the full stack keeps
running — the `/api/health` `embedder` field reports which one is active.

### 1. Backend — `:8000`

```bash
# real detector (GPU recommended)
./setup.sh                                   # venv + torch + GroundingDINO + weights + CLIP
source .venv/bin/activate
ZST_DETECTOR=groundingdino uvicorn backend.main:app --host 0.0.0.0 --port 8000

# OR mock detector (no torch/weights, for local dev and quick tests)
ZST_DETECTOR=mock ZST_EMBEDDER=histogram \
  python3 -m uvicorn backend.main:app --port 8000
```

Real-model notes:
- Weights + config live in `checkpoints/`
  (`groundingdino_swint_ogc.pth`, `GroundingDINO_SwinT_OGC.cfg.py`); set
  `ZST_CONFIG` / `ZST_WEIGHTS` to override.
- Without CUDA the model runs on CPU at ~4.5 s/frame for an 800 px frame
  (`ZST_DEVICE=cpu`). Install the CUDA-build versions of torch/torchvision plus
  a CUDA toolkit and build GroundingDINO's ops (`python setup.py build_ext
  --inplace` in the repo) for ~30–50× speedup; it will then use the GPU and
  report `device: cuda` in `/api/health`.
- The UI's model badge confirms the real model (vs `mock`) is loaded.

Env knobs:

| var | default | purpose |
|-----|---------|---------|
| `ZST_DETECTOR` | `auto` | `groundingdino` / `gdino` / `mock` / `blob` |
| `ZST_EMBEDDER` | `auto` | `clip` / `histogram` (`auto` = CLIP, fallback histogram) |
| `ZST_CLIP_MODEL` | `openai/clip-vit-base-patch32` | CLIP encoder to load |
| `ZST_RAG_GENERATOR` | `score` | `score` / `llm` / `auto` (auto = LLM only if `ZST_LLM_BASE_URL` set) |
| `ZST_LLM_BASE_URL` | `http://127.0.0.1:11434/v1` | OpenAI-compatible endpoint (Ollama) for `llm` |
| `ZST_LLM_MODEL` | `llama3.2` | chat model name |
| `ZST_DEVICE` | `cuda` | detector + embedder device |

### Quick start (one‑command)

Run everything with a single script (starts the backend and the frontend, proxying
/api/* to the backend):

```bash
cd prompt-track-rag
chmod +x start.sh
./start.sh
```

The script launches the backend (`ZST_DETECTOR=mock ZST_EMBEDDER=histogram` +
uvicorn) and the frontend (Next.js) on the conventional ports (:8000 / :3000) and
keeps them running. Press Ctrl‑C to stop both.

### 2. Frontend — `:3000`

```bash
cd frontend
npm install          # Next.js 15; needs Node 18.18+ (Node 24 works)
npm run dev          # http://localhost:3000  (proxies /api/* to :8000)
```

> If another service already owns the default ports (8000/3000), run the pair
> on custom ports and tell Next where the backend lives:
>
> ```bash
> ZST_DETECTOR=mock python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8010
> BACKEND_ORIGIN=http://127.0.0.1:8010 npm run dev -- -p 3100   # -> http://localhost:3100
> ```

Open http://localhost:3000, drop a video in, type `person, car`, press
**start tracking** (tracker = **RAG** by default). Switch the tracker dropdown
or drag the RAG appearance/position weights while the video runs — the stream
keeps playing and the identity model changes live.

### 3. RTSP / IP camera

Click the **rtsp / ip cam** tab in the UI and enter an RTSP/RTMP/MJPEG
URL. The server reconnects automatically if the feed drops; stop it from
the UI or call `POST /api/stop/{id}`. Example:

```
rtsp://user:password@192.168.1.100:554/stream1
rtmp://host/live/key
http://192.168.1.100/mjpg/video.mjpg
```

> The stream must be MJPEG-over-HTTP (`multipart/x-mixed-replace`), not raw
> MPEG-TS: an `<img src=...>` tag can only render progressive JPEG frames, and
> that's exactly the format the backend pushes.

## Optional: LLM-generation head (true generation step)

By default the trajectory "generation" is the built-in `ScoreGenerator` (fast,
offline). To make the association decision with a real language model over the
retrieved memories:

```bash
# run Ollama once, then start the backend with the LLM head active
ZST_RAG_GENERATOR=llm ZST_DETECTOR=groundingdino \
  uvicorn backend.main:app --port 8000
```

Each detection is rendered with its retrieved candidate tracks (id, class,
appearance similarity, last center, age, lost) and the model answers
`{"track_id": N}` or `{"track_id": null}`. Expect slower frame rates; this is
for experiments and debugging.

## API

| method | path | purpose |
|--------|------|---------|
| POST | `/api/track` | multipart upload + prompt + tracker → `{id, stream_url}` |
| POST | `/api/rtsp` | live URL + prompt + tracker → `{id, stream_url}` (see `GET /api/status` for `is_live`) |
| GET | `/api/stream/{id}` | infinite MJPEG; point `<img src=...>` here |
| POST | `/api/control/{id}` | JSON partial update (prompt, tracker, thresholds…) |
| GET | `/api/status/{id}` | frame count, active tracks, `det_count`, `model_state`, params, errors |
| POST | `/api/stop/{id}` | stop a streaming session (live or upload) |
| GET | `/api/demo-video` | synthetic red-blob clip for the mock detector |
| GET | `/api/trackers` | available tracker names |
| GET | `/api/health` | liveness + `detector` + `embedder` info (loaded / kind / device / mock) |

Runtime fields (also the "live" panel in the UI): `text_prompt`,
`box_threshold`, `text_threshold`, `detect_interval`, `tracker_name`,
`track_thresh`, `match_thresh`, `track_buffer`, `w_appearance`, `w_position`,
`memory_slots`, `top_k`, `show_trails`, `loop` (uploads only),
`reconnect_limit` (RTSP only; `0` = retry forever).

## Am I seeing detections?

The UI's **model** badge reports whether the detector loaded (`running ✓` /
`NOT loaded ✗` from `GET /api/health`), and the stream column shows
**boxes / frame** and **total detections** (from `GET /api/status/{id}`).

Heads-up for local runs: the mock detector only finds **red** objects. If a
video has no red content you'll see zero boxes — that's expected, not a bug
in the pipeline. Grab a clip that contains red, or download the bundled
synthetic clip:

```bash
python scripts/make_demo_video.py -o runtime/demo-blobs.mp4
# or just:  curl -O http://localhost:3000/api/demo-video
```

Upload that and you'll immediately see moving bounding boxes, tracks, and
IDs. Use the real GroundingDINO (`ZST_DETECTOR=groundingdino`) for arbitrary
classes.

## CLI (no server)

```bash
python run_tracking.py --video clips/street.mp4 --text "person, car" -o output/demo.mp4
python run_tracking.py --video clips/street.mp4 --text person --tracker rag \
    --w-appearance 0.8 --w-position 0.2
```

## Tests

```bash
# use the torch-free retrievers/detector so nothing heavy is needed:
ZST_DETECTOR=mock ZST_EMBEDDER=histogram pytest tests/ -v
```

- `test_rag.py` — RAG retrieval/generation: histogram embedder, memory
  store embed→retrieve→retire, identity stability (static/moving/occlusion),
  two-object separation by appearance, positional-only degradation, one-to-one
  assignment gate.
- `test_tracker.py` / `test_trackers.py` — ByteTrack/SORT/IoU math, identity,
  registry (numpy only).
- `test_server.py` — FastAPI routes via TestClient (mock detector): uploads,
  RTSP session lifecycle/validation, detector+embedder status in
  `/api/health`, detection counters, demo-video endpoint.
- `test_live_server.py` — boots a real uvicorn instance and consumes the
  MJPEG stream over TCP exactly like a browser `<img>`, verifies JPEG frames
  decode, switches trackers mid-stream, and exercises an RTSP session.

## Credits

- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) — Liu et al.,
  *Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object
  Detection* (ECCV 2024).
- CLIP — Radford et al., *Learning Transferable Visual Models From Natural
  Language Supervision* (ICML 2021), via 🤗 Transformers.
- [ByteTrack](https://github.com/ifzhang/ByteTrack) — Zhang et al., *ByteTrack:
  Multi-Object Tracking by Associating Every Detection Box* (ECCV 2022).
- SORT — Bewley et al., *Simple Online and Realtime Tracking* (ICIP 2016).
- Weights/config are fetched from the official GroundingDINO releases.