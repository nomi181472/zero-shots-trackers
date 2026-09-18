"""FastAPI application for Detector-Free Visual Tracking (OSTrack / STARK / Mock).

Run from the project root:
    python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import base64
import json
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from zero_shot_tracking.demo_video import make_demo_video
from zero_shot_tracking.registry import AVAILABLE_TRACKERS

from .params import RunParams
from .stream import StreamManager
from .worker import get_tracker_info

ROOT = Path(__file__).resolve().parent.parent
UPLOADS = ROOT / "runtime" / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)

ALLOWED_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
ALLOWED_STREAM_SCHEMES = {"rtsp", "rtmp", "http", "https"}

app = FastAPI(
    title="Detector-Free Visual Tracking API (OSTrack / STARK)",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = StreamManager()


class RtspRequest(BaseModel):
    url: str
    text_prompt: str = "visual target"
    initial_box: Optional[str] = None  # JSON string "[x1,y1,x2,y2]"
    tracker_name: str = "ostrack"
    show_trails: bool = True
    reconnect_limit: int = 0


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "trackers": list(AVAILABLE_TRACKERS),
        "sessions": manager.count(),
        "info": get_tracker_info(),
    }


@app.get("/api/trackers")
def list_trackers() -> dict:
    return {"trackers": list(AVAILABLE_TRACKERS)}


@app.post("/api/first-frame")
async def get_first_frame(file: UploadFile = File(...)) -> dict:
    """Extract frame 1 from an uploaded video to allow interactive bounding-box drawing."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"unsupported video format {ext}")

    content = await file.read()
    temp_path = UPLOADS / f"preview_{uuid.uuid4().hex[:8]}{ext}"
    temp_path.write_bytes(content)

    cap = cv2.VideoCapture(str(temp_path))
    ret, frame = cap.read()
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = round(float(cap.get(cv2.CAP_PROP_FPS) or 30.0), 1)
    cap.release()

    if not ret or frame is None:
        if temp_path.exists():
            temp_path.unlink()
        raise HTTPException(400, "could not read first frame from video")

    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    b64_frame = base64.b64encode(buf).decode("utf-8") if ok else ""

    return {
        "ok": True,
        "video_path": str(temp_path),
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "fps": fps,
        "first_frame_b64": f"data:image/jpeg;base64,{b64_frame}",
    }


@app.post("/api/track")
async def create_session(
    file: Optional[UploadFile] = File(None),
    video_path: Optional[str] = Form(None),
    crop_file: Optional[UploadFile] = File(None),
    text_prompt: str = Form("visual target"),
    tracker_name: str = Form("ostrack"),
    initial_box: Optional[str] = Form(None),  # JSON string "[x1,y1,x2,y2]"
    initial_image_b64: Optional[str] = Form(None),
    loop: bool = Form(False),
    show_trails: bool = Form(True),
) -> dict:
    """Start tracking an uploaded video or demo clip."""
    if tracker_name not in AVAILABLE_TRACKERS:
        tracker_name = "ostrack"

    dest: Path
    if file is not None and file.filename:
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXT:
            raise HTTPException(400, f"unsupported format {ext}")
        content = await file.read()
        dest = UPLOADS / f"{uuid.uuid4().hex}{ext}"
        dest.write_bytes(content)
    elif video_path and Path(video_path).exists():
        dest = Path(video_path)
    else:
        # Fall back to demo video
        dest = ROOT / "runtime" / "demo-target.mp4"
        if not dest.exists():
            make_demo_video(dest)

    probe = cv2.VideoCapture(str(dest))
    width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360
    frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = probe.get(cv2.CAP_PROP_FPS) or 30.0
    probe.release()

    # Parse initial box if provided
    init_box = None
    if initial_box:
        try:
            parsed = json.loads(initial_box)
            if len(parsed) == 4:
                init_box = tuple(float(v) for v in parsed)
        except Exception:
            init_box = None

    # Parse initial crop image if provided
    init_crop_np = None
    if crop_file is not None and crop_file.filename:
        crop_bytes = await crop_file.read()
        nparr = np.frombuffer(crop_bytes, dtype=np.uint8)
        init_crop_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif initial_image_b64:
        try:
            raw_b64 = initial_image_b64.split(",")[-1]
            decoded = base64.b64decode(raw_b64)
            nparr = np.frombuffer(decoded, dtype=np.uint8)
            init_crop_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception:
            init_crop_np = None

    # If no initial box or crop given, choose default center box
    if init_box is None and init_crop_np is None:
        bw, bh = float(width) * 0.25, float(height) * 0.25
        cx, cy = float(width) / 2.0, float(height) / 2.0
        init_box = (cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0)

    params = RunParams(
        text_prompt=text_prompt,
        initial_box=init_box,
        initial_image=init_crop_np,
        tracker_name=tracker_name,
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
    """Start tracking a live RTSP / RTMP / HTTP camera feed."""
    scheme = (req.url or "").split(":", 1)[0].lower()
    if scheme not in ALLOWED_STREAM_SCHEMES:
        raise HTTPException(
            400,
            f"unsupported scheme {scheme or '(empty)'!r}; use rtsp://, rtmp://, or http(s) MJPEG url",
        )

    init_box = None
    if req.initial_box:
        try:
            parsed = json.loads(req.initial_box)
            if len(parsed) == 4:
                init_box = tuple(float(v) for v in parsed)
        except Exception:
            init_box = None

    params = RunParams(
        text_prompt=req.text_prompt,
        initial_box=init_box,
        tracker_name=req.tracker_name if req.tracker_name in AVAILABLE_TRACKERS else "ostrack",
        show_trails=req.show_trails,
        loop=False,
        reconnect_limit=req.reconnect_limit,
    )

    session = manager.create(req.url, params, source_type="rtsp", is_live=True)
    return {
        "id": session.id,
        "source_type": "rtsp",
        "stream_url": f"/api/stream/{session.id}",
        "status_url": f"/api/status/{session.id}",
    }


DEMO_VIDEO = ROOT / "runtime" / "demo-target.mp4"


@app.get("/api/demo-video")
def demo_video():
    """Generate and return a synthetic demo video with a moving target object."""
    if not DEMO_VIDEO.exists():
        make_demo_video(DEMO_VIDEO)
    return FileResponse(DEMO_VIDEO, media_type="video/mp4", filename="demo-target.mp4")


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