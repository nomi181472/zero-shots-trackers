"""Runtime-switchable tracker registry.

The pipeline never imports a tracker directly; it asks the registry. That is
how the web UI is able to swap ByteTrack -> SORT -> IoU -> RAG live while a
video keeps streaming.

``rag`` is the Retrieval-Augmented-Generation tracker (:mod:`.rag`) — it
associates detections by embedding crops (CLIP / histogram) and retrieving the
most similar track memories from a vector store instead of using an algebraic
motion model.
"""

from __future__ import annotations

from .rag import RAGTracker
from .tracker import BYTETracker
from .trackers import IOUTracker, SORTTracker

AVAILABLE_TRACKERS = ("rag", "bytetrack", "sort", "iou")

_TABLE = {
    "rag": RAGTracker,
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
    gate for the other three, so we invert it there (1 - distance). RAG takes
    the common params verbatim and adds retrieval/generation knobs via
    ``extra``.
    """
    name = (name or "rag").lower().strip()
    if name not in _TABLE:
        raise ValueError(
            f"unknown tracker {name!r}; choose from {AVAILABLE_TRACKERS}"
        )

    if name == "rag":
        return RAGTracker(
            frame_rate=frame_rate,
            track_thresh=track_thresh,
            match_thresh=match_thresh,
            track_buffer=track_buffer,
            min_box_area=min_box_area,
            w_appearance=extra.get("w_appearance", 0.65),
            w_position=extra.get("w_position", 0.35),
            memory_slots=extra.get("memory_slots", 16),
            top_k=extra.get("top_k", 8),
            min_hits=extra.get("min_hits", 1),
            embedder=extra.get("embedder"),
            generator=extra.get("generator"),
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