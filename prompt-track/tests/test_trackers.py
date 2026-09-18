"""Tests for the runtime-switchable tracker registry + SORT/IoU baselines."""

from __future__ import annotations

import numpy as np
import pytest

from zero_shot_tracking.registry import AVAILABLE_TRACKERS, create_tracker
from zero_shot_tracking.trackers import SORTTracker


def make_dets(x1, y1, x2, y2, scores, cls=0):
    return np.asarray(
        [[a, b, c, d, s, cls]
         for a, b, c, d, s in zip(x1, y1, x2, y2, scores)],
        dtype=np.float32,
    )


@pytest.mark.parametrize("name", ["bytetrack", "sort", "iou"])
def test_registry_creates_all_trackers(name):
    trk = create_tracker(name)
    assert callable(trk.update)


def test_registry_rejects_unknown():
    with pytest.raises(ValueError):
        create_tracker("nope")


def test_sort_static_track_keeps_id():
    trk = SORTTracker()
    ids = set()
    for _ in range(10):
        tracks = trk.update(make_dets([10], [10], [50], [50], [0.9]))
        ids.update(t.track_id for t in tracks)
    assert ids, "sort must emit some track"
    assert len(ids) == 1


def test_sort_moving_track_keeps_id():
    trk = SORTTracker()
    ids = set()
    for t in range(15):
        x = 10 + t * 5
        tracks = trk.update(make_dets([x], [10], [x + 40], [50], [0.9]))
        ids.update(tr.track_id for tr in tracks)
    assert len(ids) == 1


def test_sort_swallows_empty_frames():
    trk = SORTTracker()
    for _ in range(5):
        trk.update(np.empty((0, 6), dtype=np.float32))
    trk.update(make_dets([10], [10], [50], [50], [0.9]))
    assert len(trk.update(make_dets([12], [10], [52], [50], [0.9]))) == 1


def test_iou_tracker_static_id():
    trk = create_tracker("iou", match_thresh=0.8, track_buffer=6)
    ids = set()
    for _ in range(10):
        tracks = trk.update(make_dets([10], [10], [50], [50], [0.9]))
        ids.update(t.track_id for t in tracks)
    assert len(ids) == 1


def test_iou_tracker_radius():
    """IoU baseline drifts: beyond match_thresh the box loses its id."""
    trk = create_tracker("iou", match_thresh=0.5, track_buffer=2)
    first = trk.update(make_dets([10], [10], [50], [50], [0.9]))
    assert first
    ids = {first[0].track_id}
    # box jumps far away -> IoU below threshold -> new id, old expires
    far = trk.update(make_dets([200], [200], [240], [240], [0.9]))
    ids.update(t.track_id for t in far)
    assert len(ids) == 2


def test_registry_maps_match_thresh_semantics():
    """ByteTrack match_thresh is a distance; SORT/IoU treat it as a score."""
    sort_tight = create_tracker("sort", match_thresh=0.98)  # very strict
    assert sort_tight.min_iou < 0.1
    sort_loose = create_tracker("sort", match_thresh=0.1)
    assert sort_loose.min_iou > 0.85
    assert AVAILABLE_TRACKERS == ("bytetrack", "sort", "iou")