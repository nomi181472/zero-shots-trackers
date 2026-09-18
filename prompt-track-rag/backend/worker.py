"""Background worker: read video -> detect -> track -> annotate -> stream.

Supports uploaded files and live sources (RTSP / RTMP / HTTP MJPEG). Live
sources never "finish" — if the stream drops the worker reconnects with
exponential backoff and keeps the session alive until it is stopped.
"""

from __future__ import annotations

import os
import threading
import time

import cv2
import numpy as np

from zero_shot_tracking import detectors as detector_factory
from zero_shot_tracking.detector import GroundingDINODetector
from zero_shot_tracking.registry import create_tracker
from zero_shot_tracking.visualizer import draw_frame_counter, draw_legend, draw_tracks


_detector = None
_detector_info: dict | None = None
_detector_lock = threading.Lock()


def get_detector():
    """Lazily build the shared detector once per process.

    ``ZST_DETECTOR=mock`` uses the torch-free blob detector (good for local
    dev); ``ZST_DETECTOR=groundingdino`` (default) loads the real model.
    """
    global _detector
    with _detector_lock:
        if _detector is not None:
            return _detector
        kind = os.environ.get("ZST_DETECTOR", "auto")
        _detector = detector_factory.create_detector(
            kind,
            device=os.environ.get("ZST_DEVICE", "cuda"),
            config_path=os.environ.get("ZST_CONFIG"),
            weights_path=os.environ.get("ZST_WEIGHTS"),
        )
        return _detector


def get_detector_info() -> dict:
    """Describe the loaded detector for ``/api/health`` (never raises)."""
    global _detector_info
    if _detector_info is None:
        info = {"kind": os.environ.get("ZST_DETECTOR", "auto"), "loaded": False}
        try:
            det = get_detector()
            info["loaded"] = True
            info["name"] = type(det).__name__
            info["mock"] = info["name"] == "MockBlobDetector"
            info["device"] = getattr(det, "device", "cpu")
        except Exception as exc:  # missing weights, no torch, etc.
            info["error"] = str(exc)
        _detector_info = info
    return _detector_info


def _tracker_key(p) -> tuple:
    return (
        p.tracker_name,
        p.track_thresh,
        p.match_thresh,
        p.track_buffer,
        p.frame_rate,
        p.min_box_area,
        p.w_appearance,
        p.w_position,
        p.memory_slots,
        p.top_k,
    )


def _open_capture(session):
    """Open the source; live URLs prefer TCP transport via ffmpeg."""
    if session.is_live:
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        cap = cv2.VideoCapture(session.source_uri, cv2.CAP_FFMPEG)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap
    return cv2.VideoCapture(session.source_uri)


def _read_fps(cap, fallback: float) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    return float(fps) if fps and fps > 1 else float(fallback)


def run_session(session) -> None:
    """Main loop for one session (runs in a daemon thread)."""
    try:
        detector = get_detector()
        session.set_model_state("running")
    except Exception as exc:  # e.g. missing weights, no torch
        session.finish(error=f"detector init failed: {exc}")
        return

    tracker = None
    tracker_key = None
    labels: list[str] = []

    cap = _open_capture(session)
    if not cap.isOpened():
        if not session.is_live:
            session.finish(error="cannot open uploaded video")
            return
        session.set_model_state("reconnecting", "cannot connect to source — retrying…")

    if cap.isOpened():
        session.update_params(frame_rate=int(_read_fps(cap, session.params().frame_rate)))

    reconnect_attempts = 0
    frame_idx = 0
    try:
        while not session.stop_requested:
            p = session.params()
            cap_fps = _read_fps(cap, p.frame_rate) if cap.isOpened() else float(p.frame_rate)
            if cap_fps != p.frame_rate:
                p.frame_rate = int(cap_fps)

            ret, frame = cap.read()
            if not ret:
                if not session.is_live:
                    if p.loop and frame_idx > 0:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        frame_idx = 0
                        continue
                    break  # end of file

                # ---- live source dropped: reconnect with capped backoff ----
                cap.release()
                reconnect_attempts += 1
                limit = p.reconnect_limit
                if limit and reconnect_attempts >= limit:
                    session.finish(
                        error=f"source lost after {reconnect_attempts} reconnect attempts"
                    )
                    return
                session.set_model_state(
                    "reconnecting",
                    f"stream dropped (attempt {reconnect_attempts}/{limit or '∞'}) "
                    "— reconnecting…",
                )
                backoff = min(0.5 * 2 ** (reconnect_attempts - 1), 5.0)
                deadline = time.time() + backoff
                while time.time() < deadline and not session.stop_requested:
                    time.sleep(0.1)
                if session.stop_requested:
                    break  # stop() during backoff -> clean finish
                cap = _open_capture(session)
                if cap.isOpened():
                    reconnect_attempts = 0
                    session.update_params(
                        frame_rate=int(_read_fps(cap, p.frame_rate))
                    )
                    session.set_model_state("running")
                continue

            frame_idx += 1

            # rebuild tracker only when its config changes (live swap)
            key = _tracker_key(p)
            if key != tracker_key:
                tracker = create_tracker(
                    p.tracker_name,
                    frame_rate=p.frame_rate,
                    track_thresh=p.track_thresh,
                    match_thresh=p.match_thresh,
                    track_buffer=p.track_buffer,
                    min_box_area=p.min_box_area,
                    w_appearance=p.w_appearance,
                    w_position=p.w_position,
                    memory_slots=p.memory_slots,
                    top_k=p.top_k,
                )
                tracker_key = key

            # detect on the configured cadence
            if (frame_idx - 1) % max(1, p.detect_interval) == 0:
                dets, labels, _ = detector.detect(
                    frame, p.text_prompt, p.box_threshold, p.text_threshold
                )
                if len(dets):
                    dets = GroundingDINODetector.nms(dets)
                session.set_detection_stats(len(dets))
            else:
                dets = np.empty((0, 6), dtype=np.float32)

            # the RAG tracker embeds detection crops from the current frame;
            # the algebraic trackers only need the boxes.
            if getattr(tracker, "needs_frame", False):
                tracks = tracker.update(dets, frame, p.text_prompt)
            else:
                tracks = tracker.update(dets)

            annotated = draw_tracks(frame, tracks, labels, {}, p.show_trails)
            annotated = draw_legend(annotated, labels, p.text_prompt)
            annotated = draw_frame_counter(annotated, frame_idx)

            session.set_frame(annotated, frame_idx, len(tracks))
        session.finish()
    except Exception as exc:
        session.finish(error=f"processing failed: {exc}")
    finally:
        cap.release()