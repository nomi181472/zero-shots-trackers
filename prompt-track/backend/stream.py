"""Streaming session management.

Each upload becomes a :class:`StreamSession` with its own background worker
thread (detect + track + annotate). The frontend simply points an ``<img>`` at
the session's MJPEG endpoint, and can retune the pipeline live via
``/api/control/{id}`` while the worker reads the latest params every frame.
"""

from __future__ import annotations

import asyncio
import copy
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2

from .params import RunParams


class StreamSession:
    def __init__(
        self,
        source: str | Path,
        params: RunParams,
        source_type: str = "upload",  # "upload" | "rtsp"
        is_live: bool = False,
    ) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.source_type = source_type
        self.is_live = bool(is_live)  # live source: never reaches EOF
        self.source = str(source) if self.is_live else Path(source)
        self.source_uri = str(self.source)

        self._params = copy.deepcopy(params)
        self._params_lock = threading.Lock()

        self._frame: Optional[bytes] = None      # jpeg bytes of latest frame
        self._frame_lock = threading.Lock()
        self._progress_lock = threading.Lock()

        self.frame_count = 0
        self.active_tracks = 0
        self.det_count = 0           # cumulative detected boxes this session
        self.last_box_count = 0      # boxes found on the most recent detect
        self.model_state = "starting"  # starting | running | reconnecting | failed | stopped
        self.warn: Optional[str] = None
        self.finished = False
        self.error: Optional[str] = None
        self.stop_requested = False

        self.created = time.time()
        self.last_frame_ts = 0.0

        self._worker = threading.Thread(
            target=self._run, name=f"zst-{self.id}", daemon=True
        )

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> "StreamSession":
        self._worker.start()
        return self

    def _run(self) -> None:
        from .worker import run_session

        run_session(self)

    def stop(self) -> None:
        self.stop_requested = True

    def finish(self, error: str | None = None) -> None:
        with self._progress_lock:
            self.error = error
            self.finished = True
            self.model_state = "failed" if error else "stopped"

    def set_detection_stats(self, box_count: int) -> None:
        """Record detection output (boxes on the most recent frame + total)."""
        with self._progress_lock:
            self.last_box_count = int(box_count)
            self.det_count += int(box_count)

    def set_model_state(self, state: str, warn: str | None = None) -> None:
        with self._progress_lock:
            self.model_state = state
            if warn is not None:
                self.warn = warn
            elif state == "running":
                self.warn = None

    # -- params --------------------------------------------------------------
    def params(self) -> RunParams:
        with self._params_lock:
            return copy.deepcopy(self._params)

    def update_params(self, **kwargs) -> RunParams:
        with self._params_lock:
            self._params.update(**kwargs)
            return copy.deepcopy(self._params)

    # -- frame / progress -----------------------------------------------------
    def set_frame(self, frame_bgr, frame_count: int, track_count: int) -> None:
        ok, buf = cv2.imencode(
            ".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 82]
        )
        if not ok:
            return
        with self._frame_lock:
            self._frame = buf.tobytes()
        with self._progress_lock:
            self.frame_count = frame_count
            self.active_tracks = track_count
        self.last_frame_ts = time.time()

    def get_frame(self) -> Optional[bytes]:
        with self._frame_lock:
            return self._frame

    def status(self) -> dict:
        with self._progress_lock:
            frame_count, active_tracks, finished, error = (
                self.frame_count,
                self.active_tracks,
                self.finished,
                self.error,
            )
            det_count, last_box_count, model_state, warn = (
                self.det_count,
                self.last_box_count,
                self.model_state,
                self.warn,
            )
        return {
            "id": self.id,
            "source_type": self.source_type,
            "is_live": self.is_live,
            "finished": finished,
            "error": error,
            "warn": warn,
            "model_state": model_state,
            "frame_count": frame_count,
            "active_tracks": active_tracks,
            "det_count": det_count,
            "last_box_count": last_box_count,
            "elapsed_s": round(time.time() - self.created, 2),
            "params": self.params().to_dict(),
            "stream_url": f"/api/stream/{self.id}",
        }


class StreamManager:
    def __init__(self) -> None:
        self._sessions: dict[str, StreamSession] = {}
        self._lock = threading.Lock()

    def create(
        self,
        source: str | Path,
        params: RunParams,
        source_type: str = "upload",
        is_live: bool = False,
    ) -> StreamSession:
        session = StreamSession(source, params, source_type=source_type, is_live=is_live)
        with self._lock:
            self._sessions[session.id] = session
        session.start()
        return session

    def get(self, session_id: str) -> Optional[StreamSession]:
        with self._lock:
            return self._sessions.get(session_id)

    def remove(self, session_id: str) -> Optional[StreamSession]:
        session = self.get(session_id)
        if session:
            session.stop()
            with self._lock:
                self._sessions.pop(session_id, None)
        return session

    def count(self) -> int:
        with self._lock:
            return len(self._sessions)

    # -- MJPEG --------------------------------------------------------------
    BOUNDARY = b"frame"

    async def mjpeg(self, session_id: str, fps: float = 24.0):
        """Yield ``multipart/x-mixed-replace`` frames (what `<img>` renders).

        Serves the *latest* annotated frame (never queues, so it never lags).
        """
        session = self.get(session_id)
        if session is None:
            return

        delay = 1.0 / float(fps) if fps and fps > 0 else 0.0
        while not session.stop_requested:
            data = session.get_frame()
            if data is None:
                await asyncio.sleep(0.02)
                continue
            yield (
                b"--" + self.BOUNDARY + b"\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n"
                + data + b"\r\n"
            )
            if delay:
                await asyncio.sleep(delay)