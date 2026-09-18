"""Runtime-tunable pipeline parameters (per streaming session)."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class RunParams:
    # detection
    text_prompt: str = "person"
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    detect_interval: int = 1

    # tracker
    tracker_name: str = "rag"
    track_thresh: float = 0.5
    match_thresh: float = 0.8
    track_buffer: int = 30
    min_box_area: int = 10

    # RAG tracker (retrieval + generation knobs)
    w_appearance: float = 0.65  # weight of CLIP/appearance similarity
    w_position: float = 0.35    # weight of positional (IoU with predicted box)
    memory_slots: int = 16      # appearance embeddings kept per track memory
    top_k: int = 8              # retrieval candidates considered per detection

    # video/io
    frame_rate: int = 30
    loop: bool = False
    show_trails: bool = True
    reconnect_limit: int = 0  # RTSP only: 0 = retry forever

    def update(self, **kwargs) -> "RunParams":
        """Merge partial updates of known-fields only, then clamp values."""
        allowed = set(self.__dataclass_fields__)
        for key, value in kwargs.items():
            if key in allowed and value is not None:
                setattr(self, key, value)
        if isinstance(self.text_prompt, str):
            self.text_prompt = self.text_prompt.strip() or self.text_prompt
        self.box_threshold = float(min(max(self.box_threshold, 0.01), 0.99))
        self.text_threshold = float(min(max(self.text_threshold, 0.01), 0.99))
        self.detect_interval = int(max(1, self.detect_interval))
        self.track_thresh = float(min(max(self.track_thresh, 0.01), 0.99))
        self.match_thresh = float(min(max(self.match_thresh, 0.05), 0.95))
        self.track_buffer = int(max(1, self.track_buffer))
        self.min_box_area = int(max(0, self.min_box_area))
        self.w_appearance = float(min(max(self.w_appearance, 0.0), 1.0))
        self.w_position = float(min(max(self.w_position, 0.0), 1.0))
        self.memory_slots = int(max(1, self.memory_slots))
        self.top_k = int(max(1, self.top_k))
        self.frame_rate = int(max(1, self.frame_rate))
        self.loop = bool(self.loop)
        self.show_trails = bool(self.show_trails)
        self.reconnect_limit = int(max(0, self.reconnect_limit))
        return self

    def to_dict(self) -> dict:
        return asdict(self)