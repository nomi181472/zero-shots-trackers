"""Runtime-switchable tracker registry.

The pipeline never imports a tracker directly; it asks the registry. That is
how the web UI is able to swap ByteTrack -> SORT -> IoU-baseline live while a
video keeps streaming.
"""

from __future__ import annotations

from .tracker import BYTETracker
from .trackers import IOUTracker, SORTTracker

AVAILABLE_TRACKERS = ("bytetrack", "sort", "iou")

_TABLE = {
    "bytetrack": BYTETracker,
    "sort": SORTTracker,
    "iou": IOUTracker,
}


def create_tracker(
    name: str,
    frame_rate: int = 30,
    track_thresh: float = 0.5,
    match_thresh: float = 0.8,
    track_buffer: int = 30,
    min_box_area: int = 10,
    **extra,
):
    """Instantiate a tracker by name, translating the common params.

    ``match_thresh`` is an IoU *distance* gate for ByteTrack but an IoU *score*
    gate for the other two, so we invert it there (1 - distance).
    """
    name = (name or "bytetrack").lower().strip()
    if name not in _TABLE:
        raise ValueError(
            f"unknown tracker {name!r}; choose from {AVAILABLE_TRACKERS}"
        )

    if name == "bytetrack":
        return BYTETracker(
            frame_rate=frame_rate,
            track_thresh=track_thresh,
            match_thresh=match_thresh,
            track_buffer=track_buffer,
            min_box_area=min_box_area,
            fuse_score=extra.get("fuse_score", True),
        )

    if name == "sort":
        return SORTTracker(
            min_iou=max(0.05, 1.0 - match_thresh),
            max_age=track_buffer,
            min_hits=extra.get("min_hits", 1),
            track_thresh=extra.get("sort_min_score", track_thresh),
            min_box_area=min_box_area,
        )

    # name == "iou"
    return IOUTracker(
        match_thresh=max(0.05, 1.0 - match_thresh),
        track_buffer=max(1, track_buffer // 4),
        min_box_area=min_box_area,
        min_hits=extra.get("min_hits", 1),
    )