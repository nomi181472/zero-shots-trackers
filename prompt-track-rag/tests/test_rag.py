"""Unit tests for the RAG tracker (retrieval + generation, no Kalman/IoU-Hungarian).

Uses the torch-free :class:`HistogramEmbedder` and synthetic frames so the
full retrieval loop runs without CLIP weights or an LLM endpoint.
"""

from __future__ import annotations

import os

os.environ.setdefault("ZST_EMBEDDER", "histogram")

import numpy as np
import pytest

import cv2

from zero_shot_tracking.embedder import HistogramEmbedder
from zero_shot_tracking.rag import MemoryStore, RAGTracker, ScoreGenerator, TrackMemory
from zero_shot_tracking.registry import create_tracker


def make_frame(boxes_colors, h=120, w=160):
    """Draw colored tlbr boxes on a black frame. colors are BGR tuples."""
    frame = np.zeros((h, w, 3), np.uint8)
    for (x1, y1, x2, y2), color in boxes_colors:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)
    return frame


def make_dets(x1, y1, x2, y2, scores, cls=0):
    return np.asarray(
        [[a, b, c, d, s, cls] for a, b, c, d, s in zip(x1, y1, x2, y2, scores)],
        dtype=np.float32,
    )


def make_rag(**kwargs):
    kwargs.setdefault("track_thresh", 0.5)
    kwargs.setdefault("match_thresh", 0.8)
    kwargs.setdefault("track_buffer", 30)
    return create_tracker("rag", **kwargs)


def test_histogram_embedder_dims_and_norm():
    emb = HistogramEmbedder()
    frame = make_frame([((10, 10, 50, 50), (0, 0, 255))])
    vecs = emb.embed_crops(frame, np.asarray([[10, 10, 50, 50]]))
    assert vecs.shape == (1, HistogramEmbedder.embedding_dim)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)


def test_memory_store_create_retrieve_retire():
    store = MemoryStore(memory_slots=4)
    emb = HistogramEmbedder()
    frame = make_frame([((10, 10, 50, 50), (0, 0, 255))])
    query = emb.embed_crops(frame, np.asarray([[10, 10, 50, 50]]))[0]
    t = store.create(np.asarray([10, 10, 50, 50]), 0, 0.9, 1, query)
    assert t.hits == 1
    assert store.retrieve(query, top_k=5)[0][0] > 0.9
    store.retire(5)
    assert len(store.tracks) == 1
    t.lost = 6
    store.retire(5)
    assert len(store.tracks) == 0


def test_rag_static_object_keeps_single_id():
    tracker = make_rag()
    ids = []
    for _ in range(20):
        frame = make_frame([((10, 10, 50, 50), (0, 0, 255))])
        tracks = tracker.update(
            make_dets([10], [10], [50], [50], [0.9]), frame=frame
        )
        assert len(tracks) == 1
        ids.append(tracks[0].track_id)
    assert len(set(ids)) == 1


def test_rag_moving_object_keeps_id():
    tracker = make_rag()
    ids = []
    for t in range(15):
        x = 10 + t * 5
        frame = make_frame([((x, 10, x + 40, 50), (0, 0, 255))])
        tracks = tracker.update(
            make_dets([x], [10], [x + 40], [50], [0.9]), frame=frame
        )
        assert len(tracks) == 1
        ids.append(tracks[0].track_id)
    assert len(set(ids)) == 1  # same red object, stable identity


def test_rag_two_objects_two_ids():
    """Distinct colors must produce two independent memories, not one merged."""
    tracker = make_rag()
    red, blue = (0, 0, 255), (255, 0, 0)
    ids = set()
    for t in range(8):
        rx = 10 + t * 2
        frame = make_frame([
            ((rx, 10, rx + 30, 40), red),
            ((90, 60, 130, 100), blue),
        ])
        tracks = tracker.update(
            make_dets([rx, 90], [10, 60], [rx + 30, 130], [40, 100], [0.9, 0.95]),
            frame=frame,
        )
        ids.update(tr.track_id for tr in tracks)
    assert len(ids) == 2


def test_rag_occlusion_reassociates_same_id():
    """After a full occlusion, a reappearing object retrieves its old memory."""
    tracker = make_rag()
    first_ids = []
    for _ in range(10):
        frame = make_frame([((10, 10, 50, 50), (0, 0, 255))])
        tracks = tracker.update(
            make_dets([10], [10], [50], [50], [0.9]), frame=frame
        )
        first_ids.append(tracks[0].track_id)
    for _ in range(5):  # full occlusion
        tracker.update(np.empty((0, 6), dtype=np.float32), frame=make_frame([]))
    tracks = tracker.update(
        make_dets([12], [10], [52], [50], [0.9]),
        frame=make_frame([((12, 10, 52, 50), (0, 0, 255))]),
    )
    assert tracks[0].track_id == first_ids[-1]


def test_rag_no_frame_positional_only():
    """Without a frame the tracker still associates via positional retrieval."""
    tracker = make_rag(match_thresh=0.5)  # degraded mode: IoU is the whole score
    ids = []
    for t in range(12):
        x = 10 + t * 5
        tracks = tracker.update(
            make_dets([x], [10], [x + 40], [50], [0.9])
        )
        ids.append(tracks[0].track_id)
    assert len(set(ids)) == 1


def test_rag_expires_lost_memories():
    tracker = make_rag(track_buffer=3)
    tracker.update(make_dets([10], [10], [50], [50], [0.9]),
                   frame=make_frame([((10, 10, 50, 50), (0, 0, 255))]))
    for _ in range(6):
        tracker.update(np.empty((0, 6), dtype=np.float32), frame=make_frame([]))
    assert len(tracker.store.tracks) == 0  # memory expired


def test_rag_gate_rejects_weak_match():
    """A detection that matches nothing above the gate spawns a new memory."""
    tracker = make_rag(match_thresh=0.95, w_position=0.0, w_appearance=1.0)
    tracker.update(make_dets([10], [10], [50], [50], [0.9]),
                   frame=make_frame([((10, 10, 50, 50), (0, 0, 255))]))
    # a far away, differently-colored box has appearance ~0 -> new memory
    tracker.update(
        make_dets([100], [100], [140], [140], [0.9]),
        frame=make_frame([((100, 100, 140, 140), (0, 255, 0))]),
    )
    assert len(tracker.store.tracks) == 2


def test_score_generator_one_to_one():
    """The default head never assigns two detections to one memory."""
    dets = make_dets([0, 0], [0, 0], [10, 10], [10, 10], [0.9, 0.9])
    tracks = [
        TrackMemory(1, np.asarray([0, 0, 10, 10]), 0, 0.9, 1, None, 4),
        TrackMemory(2, np.asarray([100, 100, 110, 110]), 0, 0.9, 1, None, 4),
    ]
    combined = np.asarray([[0.9, 0.9], [0.8, 0.9]])  # both dets like track 1 best
    gen = ScoreGenerator()
    res = gen.assign(combined, dets, tracks, match_thresh=0.8)
    assert len(res.matches) == 2
    assert len({t.track_id for _, t in res.matches}) == 2  # one-to-one