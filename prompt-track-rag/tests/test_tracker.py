"""Unit tests for the ByteTrack port and detector helpers.

ByteTrack needs only numpy, so this runs before the heavy deps are installed
(torch / GroundingDINO are imported lazily and NOT required here).
"""

from __future__ import annotations

import numpy as np
import pytest

from zero_shot_tracking.detector import GroundingDINODetector
from zero_shot_tracking.tracker import BYTETracker, STrack, _iou_batch


def make_dets(x1, y1, x2, y2, scores, cls=0):
    return np.asarray(
        [[a, b, c, d, s, cls] for a, b, c, d, s in zip(x1, y1, x2, y2, scores)],
        dtype=np.float32,
    )


def test_iou_batch():
    a = np.asarray([[0, 0, 10, 10], [0, 0, 20, 20]], dtype=np.float64)
    b = np.asarray([[0, 0, 10, 10], [10, 10, 20, 20]], dtype=np.float64)
    iou = _iou_batch(a, b)
    assert iou[0, 0] == pytest.approx(1.0)
    assert iou[0, 1] == pytest.approx(0.0)
    assert iou[1, 0] == pytest.approx(0.25)


def test_static_object_keeps_single_id():
    tracker = BYTETracker(track_thresh=0.5, match_thresh=0.8, track_buffer=30)
    ids = []
    for _ in range(20):
        tracks = tracker.update(make_dets([10], [10], [50], [50], [0.9]))
        assert len(tracks) == 1
        ids.append(tracks[0].track_id)
    assert len(set(ids)) == 1  # same object, stable identity


def test_moving_object_tracked():
    tracker = BYTETracker(track_thresh=0.5, match_thresh=0.8, track_buffer=30)
    ids = []
    for t in range(15):
        x = 10 + t * 5
        tracks = tracker.update(
            make_dets([x], [10], [x + 40], [50], [0.9]))
        ids.append(tracks[0].track_id)
    assert len(set(ids)) == 1
    assert ids[-1] == ids[0]


def test_two_objects_two_ids():
    tracker = BYTETracker(track_thresh=0.5, match_thresh=0.8, track_buffer=30)
    from collections import Counter
    all_ids = Counter()
    frames_a = [(10, 10, 40, 40), (14, 10, 44, 40), (18, 10, 48, 40)]
    frames_b = [(60 + i * 0, 60, 90, 90) for i in range(3)]
    for i, (a, b) in enumerate(zip(frames_a, frames_b)):
        tracks = tracker.update(make_dets(
            [a[0], b[0]], [a[1], b[1]], [a[2], b[2]], [a[3], b[3]],
            [0.9, 0.95]))
        for tr in tracks:
            all_ids[tr.track_id] += 1
    assert len(all_ids) == 2  # two distinct identities emerge


def test_identity_switch_needs_occlusion():
    """Performance guardrail: after a full occlusion, the same box may get a
    new id only if it was lost; a well-behaved tracker keeps the id here."""
    tracker = BYTETracker(track_thresh=0.5, match_thresh=0.8, track_buffer=30)
    first_ids = []
    for _ in range(10):
        tracks = tracker.update(
            make_dets([10], [10], [50], [50], [0.9]))
        first_ids.append(tracks[0].track_id)
    # full occlusion for 5 frames (no detections)
    for _ in range(5):
        tracker.update(np.empty((0, 6), dtype=np.float32))
    # object reappears (mot-wise this is hard; ByteTrack will reassociate
    # because track_buffer=30 keeps the lost track alive)
    tracks = tracker.update(make_dets([12], [10], [52], [50], [0.9]))
    assert tracks[0].track_id == first_ids[-1]


def test_low_score_detections_are_used():
    """ByteTrack's key trick: a low-score box can keep a track alive."""
    tracker = BYTETracker(track_thresh=0.5, match_thresh=0.8, track_buffer=30)
    # born with a high-score box
    tracker.update(make_dets([10], [10], [50], [50], [0.9]))
    # next frame the box is low-score: still matched (second association)
    tracks = tracker.update(make_dets([12], [10], [52], [50], [0.2]))
    assert len(tracks) == 1
    assert tracks[0].score == pytest.approx(0.2)


def test_nms():
    dets = make_dets([10, 12], [10, 12], [200, 202], [200, 202], [0.9, 0.8])
    kept = GroundingDINODetector.nms(dets, iou_threshold=0.5)
    assert len(kept) == 1
    assert kept[0, 4] == pytest.approx(0.9)


def test_match_class():
    terms = "person, bicycle"
    assert GroundingDINODetector._match_class("a person riding", terms) == 0
    assert GroundingDINODetector._match_class("bicycle", terms) == 1


def test_prompt_split():
    assert GroundingDINODetector._split_prompt("Person, Car") == ["person", "car"]
    assert GroundingDINODetector._split_prompt("person") == ["person"]