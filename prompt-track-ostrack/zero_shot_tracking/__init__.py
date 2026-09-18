"""Detector-free single-object tracking with OSTrack / STARK.

Detect is not required — the user provides an object image / bounding box in
frame 1, and the tracker follows its appearance across subsequent frames.

Example usage:

    from zero_shot_tracking import OSTracker

    tracker = OSTracker()                        # or "stark"
    # On first frame (frame_idx == 1):
    tracker.init(frame_bgr, crop_bgr)            # or tracker.init(frame_bgr, box=[x1,y1,x2,y2])
    # On every subsequent frame:
    box = tracker.update(frame_bgr)              # -> [x1, y1, x2, y2]
"""
from .ostracker import OSTracker
from .registry import AVAILABLE_TRACKERS, create_tracker

__all__ = ["OSTracker", "create_tracker", "AVAILABLE_TRACKERS"]
__version__ = "0.1.0"