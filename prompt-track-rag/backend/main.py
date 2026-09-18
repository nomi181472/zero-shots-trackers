"""FastAPI app: upload -> track -> stream MJPEG, with live runtime controls.

Run from the project root:

    ZST_DETECTOR=mock uvicorn backend.main:app --host 0.0.0.0 --port 8000

Endpoints
    POST /api/track          upload video + prompt + tracker -> {id, stream_url}
    GET  /api/stream/{id}    MJPEG stream (render with `<img src=...>`)
    POST /api/control/{id}   retune prompt/tracker/thresholds live
    GET  /api/status/{id}    frame count, active tracks, params, errors
    POST /api/stop/{id}      stop a session
    GET  /api/trackers       list of runtime-switchable trackers
"""

from __future__ import annotations

import uuid
from pathlib import Path

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from zero_shot_tracking.demo_video import make_demo_video
from zero_shot_tracking.embedder import embedder_info
from zero_shot_tracking.registry import AVAILABLE_TRACKERS

from .params import RunParams
from .stream import StreamManager
from .worker import get_detector_info

ROOT = Path(__file__).resolve().parent.parent
UPLOADS = ROOT / "runtime" / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)

ALLOWED_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
ALLOWED_STREAM_SCHEMES = {"rtsp", "rtmp", "http", "https"}

app = FastAPI(title="Zero-Shot Tracking API (RAG tracker)", version="0.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Next.js dev proxy makes this optional anyway
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = StreamManager()


class RtspRequest(BaseModel):
    """JSON body for live sources (RTSP / RTMP / HTTP MJPEG)."""

    url: str
    text_prompt: str = "person"
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    detect_interval: int = 1
    tracker_name: str = "rag"
    track_thresh: float = 0.5
    match_thresh: float = 0.8
    track_buffer: int = 30
    min_box_area: int = 10
    w_appearance: float = 0.65
    w_position: float = 0.35
    memory_slots: int = 16
    top_k: int = 8
    show_trails: bool = True
    reconnect_limit: int = 0  # 0 = keep retrying forever


def _rtsp_params(req: RtspRequest) -> RunParams:
    return RunParams(
        text_prompt=req.text_prompt,
        box_threshold=req.box_threshold,
        text_threshold=req.text_threshold,
        detect_interval=req.detect_interval,
        tracker_name=req.tracker_name,
        track_thresh=req.track_thresh,
        match_thresh=req.match_thresh,
        track_buffer=req.track_buffer,
        min_box_area=req.min_box_area,
        w_appearance=req.w_appearance,
        w_position=req.w_position,
        memory_slots=req.memory_slots,
        top_k=req.top_k,
        show_trails=req.show_trails,
        loop=False,  # a live source never needs looping
        reconnect_limit=req.reconnect_limit,
    )


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "trackers": AVAILABLE_TRACKERS,
        "sessions": manager.count(),
        "detector": get_detector_info(),
        "embedder": embedder_info(),
    }


@app.get("/api/trackers")
def list_trackers() -> dict:
    return {"trackers": AVAILABLE_TRACKERS}


@app.post("/api/track")
async def create_session(
    file: UploadFile = File(...),
    text_prompt: str = Form("person"),
    tracker_name: str = Form("rag"),
    box_threshold: float = Form(0.35),
    text_threshold: float = Form(0.25),
    detect_interval: int = Form(1),
    track_thresh: float = Form(0.5),
    match_thresh: float = Form(0.8),
    track_buffer: int = Form(30),
    min_box_area: int = Form(10),
    w_appearance: float = Form(0.65),
    w_position: float = Form(0.35),
    memory_slots: int = Form(16),
    top_k: int = Form(8),
    loop: bool = Form(False),
    show_trails: bool = Form(True),
) -> dict:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(
            400, f"unsupported file type {ext or '(none)'}; use mp4/avi/mov/mkv/webm"
        )
    if tracker_name not in AVAILABLE_TRACKERS:
        raise HTTPException(
            400, f"unknown tracker {tracker_name!r}; options: {AVAILABLE_TRACKERS}"
        )

    content = await file.read()
    if not content:
        raise HTTPException(400, "empty file uploaded")

    dest = UPLOADS / f"{uuid.uuid4().hex}{ext}"
    dest.write_bytes(content)

    probe = cv2.VideoCapture(str(dest))
    width, height = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        probe.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )
    frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = probe.get(cv2.CAP_PROP_FPS) or 30.0
    probe.release()
    if width <= 0 or height <= 0:
        raise HTTPException(400, "could not read video (corrupt or unsupported codec)")

    params = RunParams(
        text_prompt=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        detect_interval=detect_interval,
        tracker_name=tracker_name,
        track_thresh=track_thresh,
        match_thresh=match_thresh,
        track_buffer=track_buffer,
        min_box_area=min_box_area,
        w_appearance=w_appearance,
        w_position=w_position,
        memory_slots=memory_slots,
        top_k=top_k,
        frame_rate=int(fps),
        loop=loop,
        show_trails=show_trails,
    )

    session = manager.create(dest, params)
    return {
        "id": session.id,
        "source_type": "upload",
        "stream_url": f"/api/stream/{session.id}",
        "width": width,
        "height": height,
        "frame_count": max(frames, 0),
        "fps": round(fps, 2),
        "tracker": tracker_name,
        "status_url": f"/api/status/{session.id}",
    }


@app.post("/api/rtsp")
def create_rtsp_stream(req: RtspRequest) -> dict:
    """Start tracking a live URL (camera / RTSP / RTMP / HTTP-MJPEG feed).

    Unlike an upload, a live session runs until ``/api/stop/{id}`` — the
    worker transparently reconnects if the feed drops (see ``reconnect_limit``).
    """
    scheme = (req.url or "").split(":", 1)[0].lower()
    if scheme not in ALLOWED_STREAM_SCHEMES:
        raise HTTPException(
            400,
            f"unsupported scheme {scheme or '(empty)'!r}; use rtsp://, rtmp://, "
            "or an http(s) MJPEG url",
        )
    if req.tracker_name not in AVAILABLE_TRACKERS:
        raise HTTPException(
            400, f"unknown tracker {req.tracker_name!r}; options: {AVAILABLE_TRACKERS}"
        )

    session = manager.create(req.url, _rtsp_params(req), source_type="rtsp", is_live=True)
    return {
        "id": session.id,
        "source_type": "rtsp",
        "stream_url": f"/api/stream/{session.id}",
        "status_url": f"/api/status/{session.id}",
        "fps": None,  # probed by the worker once the feed is open
    }


DEMO_VIDEO = ROOT / "runtime" / "demo-blobs.mp4"


@app.get("/api/demo-video")
def demo_video():
    """Download a synthetic clip with red blobs for the mock detector."""
    if not DEMO_VIDEO.exists():
        make_demo_video(DEMO_VIDEO)
    return FileResponse(DEMO_VIDEO, media_type="video/mp4", filename="demo-blobs.mp4")


def _get_session(session_id: str):
    session = manager.get(session_id)
    if session is None:
        raise HTTPException(404, "session not found")
    return session


@app.get("/api/stream/{session_id}")
async def stream_video(session_id: str):
    _get_session(session_id)
    return StreamingResponse(
        manager.mjpeg(session_id, fps=24),
        media_type=f"multipart/x-mixed-replace; boundary={StreamManager.BOUNDARY.decode()}",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/control/{session_id}")
def control(session_id: str, body: dict) -> dict:
    session = _get_session(session_id)
    session.update_params(**body)
    return session.status()


@app.get("/api/status/{session_id}")
def status(session_id: str) -> dict:
    return _get_session(session_id).status()


@app.post("/api/stop/{session_id}")
def stop(session_id: str) -> dict:
    session = _get_session(session_id)
    session.stop()
    return {"stopped": True, "id": session_id}