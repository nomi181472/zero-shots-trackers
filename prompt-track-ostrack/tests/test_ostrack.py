"""Unit tests for the detector-free OSTracker / mock tracker.

These tests run without GroundingDINO, CLIP, or OSTrack/STARK packages —
they only need numpy and opencv, which are test dependencies of the
existing ``prompt-track`` / ``prompt-track-rag`` projects.
"""

from __future__ import annotations

import os
import cv2
import numpy as np

from zero_shot_tracking.ostracker import OSTracker


def test_mock_init_with_box():
    """Init with a box; update should return the box (identity fallback)."""
    tracker = OSTracker(tracker_type="mock")
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    box = (400, 300, 600, 500)  # [x1, y1, x2, y2]
    tracker.init(frame, box)
    # update on a different frame — should return the same box
    box2 = tracker.update(frame)
    assert box2 == box, f"expected {box}, got {box2}"


def test_mock_init_with_crop():
    """Init with a crop (colored rectangle); template matching should work."""
    tracker = OSTracker(tracker_type="mock")
    # Create a frame with a white rectangle in the center
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (200, 150), (440, 330), (255, 255, 255), -1)
    # Crop the white rectangle
    crop = frame[150:330, 200:440]
    tracker.init(frame, crop=crop)
    # Update on a frame that contains the same rectangle
    frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame2, (200, 150), (440, 330), (255, 255, 255), -1)
    box2 = tracker.update(frame2)
    # Box should be within the rectangle bounds
    x1, y1, x2, y2 = box2
    assert 200 <= x1 < x2 <= 440, f"x1={x1}, x2={x2} out of expected range"
    assert 150 <= y1 < y2 <= 330, f"y1={y1}, y2={y2} out of expected range"


def test_mock_init_requires_box_or_crop():
    """Init without box or crop should raise."""
    tracker = OSTracker(tracker_type="mock")
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    try:
        tracker.init(frame)  # type: ignore[arg-type]
        assert False, "expected ValueError"
    except ValueError:
        pass  # expected


def test_ostracker_draw():
    """draw() should add a green rectangle to the frame."""
    tracker = OSTracker(tracker_type="mock")
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    box = (10, 10, 50, 50)
    tracker.init(frame, box)
    tracker.update(frame)  # warm up
    out = tracker.draw(frame)
    # draw() modifies in-place; check that green color was applied
    # the rectangle at (10,10)-(50,50) should have green channel > 0
    assert out[20, 20, 1] > 0, "green channel should be > 0 inside the rectangle"


def test_tracker_registry():
    """Registry should accept ostrack/stark/mock names."""
    from zero_shot_tracking.registry import create_tracker, AVAILABLE_TRACKERS
    assert "ostrack" in AVAILABLE_TRACKERS
    assert "stark" in AVAILABLE_TRACKERS
    assert "mock" in AVAILABLE_TRACKERS

    for name in ("ostrack", "stark", "mock"):
        t = create_tracker(name)
        assert isinstance(t, OSTracker), f"create_tracker({name!r}) should return OSTracker, got {type(t)}"


def test_template_matching_different_size():
    """Template matching should handle ROI larger than template."""
    tracker = OSTracker(tracker_type="mock")
    # Small template
    template = np.zeros((50, 50, 3), dtype=np.uint8)
    cv2.rectangle(template, (10, 10), (40, 40), (255, 255, 255), -1)
    # Larger frame with the same pattern in different location
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    cv2.rectangle(frame, (80, 60), (130, 100), (255, 255, 255), -1)
    tracker.init(frame, crop=template)
    box = tracker.update(frame)
    x1, y1, x2, y2 = box
    # The matched box should be near the rectangle location
    assert 70 <= x1 <= 150, f"x1={x1} unexpected (expected ~80)"
    assert 50 <= y1 <= 120, f"y1={y1} unexpected (expected ~60)"


def test_ostrack_tracking_motion():
    """OSTrack tracker should accurately follow an object moving across frames."""
    tracker = OSTracker(tracker_type="ostrack")
    # Frame 1: target at (100, 100, 150, 150)
    f1 = np.full((300, 400, 3), 200, dtype=np.uint8)
    cv2.circle(f1, (125, 125), 20, (0, 0, 255), -1)
    tracker.init(f1, box=(105, 105, 145, 145))

    # Frame 2: target moved to center (145, 135)
    f2 = np.full((300, 400, 3), 200, dtype=np.uint8)
    cv2.circle(f2, (145, 135), 20, (0, 0, 255), -1)
    b2 = tracker.update(f2)
    cx2 = (b2[0] + b2[2]) / 2.0
    cy2 = (b2[1] + b2[3]) / 2.0
    assert abs(cx2 - 145) < 8.0, f"expected center ~145, got {cx2}"
    assert abs(cy2 - 135) < 8.0, f"expected center ~135, got {cy2}"


def test_stark_adaptive_tracking():
    """STARK tracker should follow moving target and update dynamic template."""
    tracker = OSTracker(tracker_type="stark")
    f1 = np.full((300, 400, 3), 220, dtype=np.uint8)
    cv2.rectangle(f1, (80, 80), (130, 130), (50, 100, 240), -1)
    tracker.init(f1, box=(80, 80, 130, 130))

    for step in range(1, 6):
        frame = np.full((300, 400, 3), 220, dtype=np.uint8)
        new_x = 80 + step * 8
        new_y = 80 + step * 4
        cv2.rectangle(frame, (new_x, new_y), (new_x + 50, new_y + 50), (50, 100, 240), -1)
        box = tracker.update(frame)
        assert abs(box[0] - new_x) < 8.0
        assert abs(box[1] - new_y) < 8.0


def test_demo_video_generation(tmp_path):
    """make_demo_video should create a readable MP4 file."""
    from zero_shot_tracking.demo_video import make_demo_video
    target_path = tmp_path / "test-demo.mp4"
    out = make_demo_video(target_path, frames=10, fps=10, width=320, height=240)
    assert os.path.exists(out)
    cap = cv2.VideoCapture(out)
    assert cap.isOpened()
    ret, frame = cap.read()
    assert ret
    assert frame.shape == (240, 320, 3)
    cap.release()


def test_api_health_and_trackers():
    """Verify health and trackers endpoints return valid JSON."""
    from fastapi.testclient import TestClient
    from backend.main import app

    client = TestClient(app)
    h_res = client.get("/api/health")
    assert h_res.status_code == 200
    data = h_res.json()
    assert data["ok"] is True
    assert "ostrack" in data["trackers"]
    assert "stark" in data["trackers"]

    t_res = client.get("/api/trackers")
    assert t_res.status_code == 200
    assert "trackers" in t_res.json()


def test_api_track_demo():
    """Verify session creation and status endpoints."""
    from fastapi.testclient import TestClient
    from backend.main import app

    client = TestClient(app)
    res = client.post("/api/track", data={
        "tracker_name": "ostrack",
        "initial_box": "[200, 100, 350, 250]",
        "show_trails": "true",
    })
    assert res.status_code == 200
    info = res.json()
    assert "id" in info
    assert "stream_url" in info

    session_id = info["id"]
    status_res = client.get(f"/api/status/{session_id}")
    assert status_res.status_code == 200

    stop_res = client.post(f"/api/stop/{session_id}")
    assert stop_res.status_code == 200