"""Background worker: read video -> track -> annotate -> stream.

Detector-free version: no generic object detector required. The tracker is
initialized on frame 1 from the user-supplied initial box or object crop,
and then follows the object appearance across all subsequent frames.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Optional

import cv2
import numpy as np

from zero_shot_tracking.registry import AVAILABLE_TRACKERS, create_tracker
from zero_shot_tracking.visualizer import draw_frame_counter, draw_legend, draw_tracks


def get_tracker_info() -> dict:
    """Return runtime detector-free tracking capability info for /api/health."""
    return {
        "kind": "detector-free",
        "loaded": True,
        "name": "OSTrack / STARK / TemplateMatching",
        "available_trackers": list(AVAILABLE_TRACKERS),
    }


class _TrackObj:
    """Track representation for visualizer."""
    __slots__ = ("tlbr", "track_id", "class_id", "score", "trail")

    def __init__(self, tlbr, track_id: int, score: float, trail: list):
        self.tlbr = tlbr
        self.track_id = track_id
        self.class_id = 0
        self.score = score
        self.trail = trail


def run_session(session) -> None:
    """Main streaming loop for one session."""
    p = session.params()
    try:
        tracker = create_tracker(p.tracker_name)
    except Exception as exc:
        session.finish(error=f"tracker creation failed: {exc}")
        return

    current_tracker_name = p.tracker_name
    labels = [p.text_prompt or "target"]
    frame_idx = 0

    cap = cv2.VideoCapture(session.source_uri)
    if not cap.isOpened():
        if not session.is_live:
            session.finish(error="cannot open uploaded video file")
            return
        session.set_model_state("reconnecting", "cannot connect to source — retrying…")

    if cap.isOpened():
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        session.update_params(frame_rate=fps)

    reconnect_attempts = 0
    tracker_initialized = False

    try:
        session.set_model_state("running")
        while not session.stop_requested:
            p = session.params()

            # Dynamic live tracker switching if user changed tracker in UI
            if p.tracker_name != current_tracker_name:
                try:
                    new_tracker = create_tracker(p.tracker_name)
                    # Re-initialize new tracker using current state/box if available
                    if tracker_initialized and tracker.last_box is not None and 'last_frame' in locals():
                        new_tracker.init(last_frame, box=tracker.last_box)
                    tracker = new_tracker
                    current_tracker_name = p.tracker_name
                except Exception as e:
                    session.set_model_state("running", f"could not switch to {p.tracker_name}: {e}")

            ret, frame = cap.read()
            if not ret:
                if not session.is_live:
                    if p.loop and frame_idx > 0:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        frame_idx = 0
                        continue
                    break  # End of video

                # Live stream disconnected: reconnect
                cap.release()
                reconnect_attempts += 1
                limit = p.reconnect_limit
                if limit and reconnect_attempts >= limit:
                    session.finish(
                        error=f"stream lost after {reconnect_attempts} reconnect attempts"
                    )
                    return

                session.set_model_state(
                    "reconnecting",
                    f"stream dropped (attempt {reconnect_attempts}/{limit or '∞'}) — reconnecting…",
                )
                backoff = min(0.5 * (2 ** (reconnect_attempts - 1)), 5.0)
                deadline = time.time() + backoff
                while time.time() < deadline and not session.stop_requested:
                    time.sleep(0.1)
                if session.stop_requested:
                    break
                cap = cv2.VideoCapture(session.source_uri)
                if cap.isOpened():
                    reconnect_attempts = 0
                    session.set_model_state("running")
                continue

            frame_idx += 1
            last_frame = frame
            h, w = frame.shape[:2]

            # -------- Frame 1: Initialize Tracker --------
            if not tracker_initialized:
                init_box = p.initial_box
                init_image = p.initial_image

                if init_box is None and init_image is None:
                    # Default center box if user didn't specify one
                    cw, ch = float(w) * 0.25, float(h) * 0.25
                    cx, cy = float(w) / 2.0, float(h) / 2.0
                    init_box = (cx - cw / 2.0, cy - ch / 2.0, cx + cw / 2.0, cy + ch / 2.0)

                try:
                    tracker.init(frame, box=init_box, crop=init_image)
                    tracker_initialized = True
                except Exception as e:
                    session.finish(error=f"tracker init on frame 1 failed: {e}")
                    return

                tracked_box = tracker.last_box or init_box
                score = getattr(tracker, "last_score", 1.0)
            else:
                # Subsequent frames
                tracked_box = tracker.update(frame)
                score = getattr(tracker, "last_score", 1.0)

            # Build visualization
            trail = getattr(tracker, "trail", [])
            track_obj = _TrackObj(
                tlbr=tracked_box,
                track_id=1,
                score=score,
                trail=trail,
            )

            annotated = draw_tracks(
                frame,
                [track_obj],
                labels=[p.text_prompt or "target"],
                show_trails=p.show_trails,
            )
            annotated = draw_legend(
                annotated,
                labels=labels,
                text_prompt=p.text_prompt,
                tracker_name=current_tracker_name,
            )
            annotated = draw_frame_counter(annotated, frame_idx)

            session.set_frame(annotated, frame_idx, track_count=1, current_box=tracked_box)

        session.finish()
    except Exception as exc:
        session.finish(error=f"processing failed: {exc}")
    finally:
        cap.release()