"""End-to-end API tests for the FastAPI streaming backend.

Uses the torch-free mock detector (ZST_DETECTOR=mock) so the whole loop runs
with no model weights: upload video -> track -> stream MJPEG -> live controls.
"""

from __future__ import annotations

import os

os.environ.setdefault("ZST_DETECTOR", "mock")
os.environ.setdefault("ZST_EMBEDDER", "histogram")  # torch-free retrieval encoder

import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from zero_shot_tracking.demo_video import make_demo_video


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def sample_video(tmp_path_factory):
    """30-frame 160x120 clip: a red square drifting right (mock detects red)."""
    path = tmp_path_factory.mktemp("vids") / "sample.mp4"
    h, w, fps = 120, 160, 15
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for t in range(30):
        frame = np.zeros((h, w, 3), np.uint8)
        x = 10 + t * 4
        cv2.rectangle(frame, (x, 50), (x + 20, 70), (0, 0, 255), -1)  # BGR red
        vw.write(frame)
    vw.release()
    return path, fps


def _wait_done(client, sid, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.get(f"/api/status/{sid}").json()
        if status["finished"] or status["error"]:
            return status
        time.sleep(0.05)
    return client.get(f"/api/status/{sid}").json()


def test_health_and_trackers(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["trackers"] == ["rag", "bytetrack", "sort", "iou"]
    assert client.get("/api/trackers").json()["trackers"] == ["rag", "bytetrack", "sort", "iou"]
    # the detector (mock in tests) must report itself as loaded & running
    assert body["detector"]["loaded"] is True
    assert body["detector"]["mock"] is True
    # the retrieval embedder must be reported too
    assert body["embedder"]["loaded"] is True
    assert body["embedder"]["name"] == "histogram"


def test_upload_track_and_mjpeg_metadata(client, sample_video):
    """Upload creates a session; streaming bytes are verified over real TCP
    in ``tests/test_live_server.py`` (TestClient cannot consume an infinitely
    running multipart stream)."""
    video_path, fps = sample_video
    with open(video_path, "rb") as f:
        r = client.post(
            "/api/track",
            files={"file": ("sample.mp4", f, "video/mp4")},
            data={
                "text_prompt": "red blob",
                "tracker_name": "bytetrack",
                "box_threshold": "0.3",
                "track_buffer": "30",
            },
        )
    assert r.status_code == 200, r.text
    body = r.json()
    sid = body["id"]
    assert body["frame_count"] == 30
    assert body["tracker"] == "bytetrack"
    assert body["stream_url"] == f"/api/stream/{sid}"

    status = _wait_done(client, sid)
    assert status["error"] is None, status["error"]
    assert status["frame_count"] > 0
    assert status["finished"] is True
    # the mock detector must have found the red square -> bboxes happened
    assert status["det_count"] > 0
    assert status["last_box_count"] > 0


def test_demo_video_endpoint(client):
    r = client.get("/api/demo-video")
    assert r.status_code == 200
    assert r.headers["content-type"] == "video/mp4"
    assert len(r.content) > 1000


def test_runtime_tracker_switch(client, sample_video):
    video_path, fps = sample_video
    with open(video_path, "rb") as f:
        r = client.post(
            "/api/track",
            files={"file": ("sample.mp4", f, "video/mp4")},
            data={"text_prompt": "blob", "tracker_name": "bytetrack"},
        )
    sid = r.json()["id"]

    # switch tracker + prompt live while the worker is still streaming
    ctrl = client.post(
        f"/api/control/{sid}",
        json={"tracker_name": "sort", "text_prompt": "a red square",
              "box_threshold": 0.4},
    )
    assert ctrl.status_code == 200
    status = ctrl.json()
    assert status["params"]["tracker_name"] == "sort"
    assert status["params"]["text_prompt"] == "a red square"
    assert status["params"]["box_threshold"] == 0.4

    final = _wait_done(client, sid)
    assert final["error"] is None, final["error"]
    assert final["finished"] is True


def test_runtime_loop_with_reopen(client, sample_video):
    video_path, fps = sample_video
    with open(video_path, "rb") as f:
        r = client.post(
            "/api/track",
            files={"file": ("sample.mp4", f, "video/mp4")},
            data={"text_prompt": "blob", "loop": "true"},
        )
    sid = r.json()["id"]

    client.post(f"/api/control/{sid}", json={"loop": False})
    time.sleep(0.5)
    status = _wait_done(client, sid)
    assert status["error"] is None
    assert status["finished"] is True


def test_stop_signal(client, sample_video):
    video_path, fps = sample_video
    with open(video_path, "rb") as f:
        r = client.post(
            "/api/track",
            files={"file": ("sample.mp4", f, "video/mp4")},
            data={"text_prompt": "blob"},
        )
    sid = r.json()["id"]
    ok = client.post(f"/api/stop/{sid}")
    assert ok.status_code == 200
    time.sleep(0.3)
    assert client.get(f"/api/status/{sid}").json()["finished"] is True


def test_upload_validation(client, sample_video):
    video_path, _ = sample_video
    with open(video_path, "rb") as f:
        r = client.post(
            "/api/track",
            files={"file": ("sample.mp4", f, "video/mp4")},
            data={"tracker_name": "hypernet"},
        )
    assert r.status_code == 400
    assert "unknown tracker" in r.text


def test_rtsp_session_lifecycle(client):
    """A live (RTSP) session never finishes on its own, survives a dead
    source in 'reconnecting' state, accepts live controls, and stops."""
    r = client.post(
        "/api/rtsp",
        json={"url": "rtsp://127.0.0.1:9/nope", "text_prompt": "red",
              "tracker_name": "bytetrack"},
    )
    assert r.status_code == 200, r.text
    info = r.json()
    sid = info["id"]
    assert info["source_type"] == "rtsp"

    const = client.get(f"/api/status/{sid}")
    assert const.status_code == 200
    st = const.json()
    assert st["is_live"] is True
    assert st["source_type"] == "rtsp"
    assert st["finished"] is False

    # live controls work before/while the feed is unreachable
    ctrl = client.post(f"/api/control/{sid}", json={"tracker_name": "sort"})
    assert ctrl.status_code == 200
    assert ctrl.json()["params"]["tracker_name"] == "sort"

    # for a dead source the worker should have entered the reconnect state
    wait_until(lambda: client.get(f"/api/status/{sid}").json()["model_state"],
               lambda s: s == "reconnecting", timeout=15)
    st = client.get(f"/api/status/{sid}").json()
    assert st["finished"] is False
    assert "reconnect" in (st["warn"] or "")

    # stopping a live session finishes it cleanly
    stop = client.post(f"/api/stop/{sid}")
    assert stop.status_code == 200
    wait_until(lambda: client.get(f"/api/status/{sid}").json()["finished"],
               lambda v: v is True, timeout=15)


def test_rtsp_validation(client):
    bad_scheme = client.post("/api/rtsp", json={"url": "ftp://host/x"})
    assert bad_scheme.status_code == 400
    assert "unsupported scheme" in bad_scheme.text

    bad_tracker = client.post(
        "/api/rtsp", json={"url": "rtsp://ok", "tracker_name": "hypernet"}
    )
    assert bad_tracker.status_code == 400

    missing_url = client.post("/api/rtsp", json={"tracker_name": "iou"})
    assert missing_url.status_code == 422


def test_upload_finds_boxes_with_mock(client, tmp_path):
    """End-to-end: the exported demo clip must yield visible bboxes in mock
    mode (this is exactly the 'can't see bbox' local scenario)."""
    clip = make_demo_video(str(tmp_path / "demo.mp4"), frames=30, fps=12)
    with open(clip, "rb") as f:
        r = client.post(
            "/api/track",
            files={"file": ("demo.mp4", f, "video/mp4")},
            data={"text_prompt": "blob", "box_threshold": "0.3"},
        )
    assert r.status_code == 200, r.text
    sid = r.json()["id"]
    status = _wait_done(client, sid)
    assert status["error"] is None, status["error"]
    assert status["det_count"] > 0
    assert status["last_box_count"] > 0


def wait_until(fn, predicate, timeout=10):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = fn()
        if predicate(last):
            return last
        time.sleep(0.05)
    assert predicate(last), f"timed out, last value: {last!r}"