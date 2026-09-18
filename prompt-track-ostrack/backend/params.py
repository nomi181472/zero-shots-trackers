"""Runtime-tunable pipeline parameters (per streaming session).

Detector-free tracking: no GroundingDINO is needed. The user supplies
an initial bounding box (or object crop) on frame 1, and the OSTrack/STARK/
mock tracker follows the object appearance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional, Tuple


@dataclass
class RunParams:
    # -- detector-free tracking ------------------------------------------------
    #: Initial bounding box in frame 1 ``(x1, y1, x2, y2)`` in pixel coords.
    initial_box: Optional[Tuple[float, float, float, float]] = None
    #: Initial object crop (BGR numpy array) from frame 1.
    initial_image: Optional[Any] = None
    #: How often (in frames) to re-update the appearance template.
    template_update_interval: int = 5
    # -------------------------------------------------------------------------

    # -- prompt / metadata -----------------------------------------------------
    text_prompt: str = "visual target"
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    detect_interval: int = 1

    # -- tracker ---------------------------------------------------------------
    tracker_name: str = "ostrack"
    track_thresh: float = 0.5
    match_thresh: float = 0.8
    track_buffer: int = 30
    min_box_area: int = 10

    # -- video/io --------------------------------------------------------------
    frame_rate: int = 30
    loop: bool = False
    show_trails: bool = True
    reconnect_limit: int = 0  # RTSP only: 0 = retry forever

    def update(self, **kwargs) -> "RunParams":
        """Merge partial updates of known fields only, then clamp values."""
        allowed = set(self.__dataclass_fields__)
        for key, value in kwargs.items():
            if key in allowed and value is not None:
                setattr(self, key, value)

        # Normalize initial_box if provided
        if self.initial_box is not None and len(self.initial_box) == 4:
            x1, y1, x2, y2 = [float(v) for v in self.initial_box]
            if x2 > x1 and y2 > y1:
                self.initial_box = (max(0.0, x1), max(0.0, y1), max(1.0, x2), max(1.0, y2))

        if isinstance(self.text_prompt, str):
            self.text_prompt = self.text_prompt.strip() or "visual target"
        self.tracker_name = str(self.tracker_name).lower().strip()
        self.template_update_interval = int(max(0, self.template_update_interval))
        self.frame_rate = int(max(1, self.frame_rate))
        self.loop = bool(self.loop)
        self.show_trails = bool(self.show_trails)
        self.reconnect_limit = int(max(0, self.reconnect_limit))
        return self

    def to_dict(self) -> dict:
        d = asdict(self)
        # Avoid serializing raw numpy arrays into JSON
        if d.get("initial_image") is not None:
            d["initial_image"] = True
        return d