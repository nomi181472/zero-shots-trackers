"""Runtime-switchable tracker registry.

The pipeline never imports a tracker directly; it asks the registry. That is
how the web UI is able to swap ByteTrack -> SORT -> IoU-baseline live while a
video keeps streaming, and now also OSTrack / STARK / mock for detector-free tracking.
"""

from __future__ import annotations

from .ostracker import OSTracker

AVAILABLE_TRACKERS = ("ostrack", "stark", "mock")

_TABLE = {
    "ostrack": OSTracker,
    "stark": OSTracker,  # alias; real STARK integration added when package is available
    "mock": OSTracker,  # mock backend (template-matching fallback)
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

    Supported trackers:
    - ``ostrack`` / ``stark`` — detector-free single-object trackers
      (template/init + update, no detector required).
    - ``mock`` — template-matching fallback (no extra packages needed).
    """
    name = (name or "ostrack").lower().strip()
    if name not in AVAILABLE_TRACKERS:
        raise ValueError(
            f"unknown tracker {name!r}; choose from {AVAILABLE_TRACKERS}"
        )

    # All supported trackers share the OSTracker interface
    return OSTracker(tracker_type=name)