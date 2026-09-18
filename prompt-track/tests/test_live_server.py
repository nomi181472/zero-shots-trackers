"""Live integration tests against a real uvicorn server over TCP.

TestClient cannot read an infinite (never-ending) MJPEG stream — the
multipart body never signals `more_body=False`, so the streaming response is
only exercised truthfully over a real socket. These tests boot a thread-local
uvicorn instance and talk to it with httpx, exactly like the browser's `<img>`.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from pathlib import Path

os.environ.setdefault("ZST_DETECTOR", "mock")

import cv2
import httpx
import numpy as np
import pytest
from uvicorn import Config, Server

ROOT = Path(__file__).resolve().parent.parent


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server_url():
    port = _free_port()
    config = Config(
        "backend.main:app",
        host="127.0.0.1",
        port=port,
        log_level="warning",
        lifespan="off",
    )
    server = Server(config)
    thread = threading.Thread(target=server.run, daemon=True, name="uvicorn-test")
    thread.start()

    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            if httpx.get(f"{base}/api/health", timeout=1).json()["ok"]:
                break
        except Exception:
            time.sleep(0.1)
    else:
        raise RuntimeError("live server did not start")

    yield base
    server.should_exit = True
    thread.join(timeout=10)


@pytest.fixture(scope="module")
def video_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("vids") / "live.mp4"
    h, w, fps = 320, 480, 15
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for t in range(40):
        frame = np.zeros((h, w, 3), np.uint8)
        x = 30 + t * 10
        cv2.rectangle(frame, (x, 120), (x + 60, 180), (0, 0, 255), -1)
        vw.write(frame)
    vw.release()
    return path


def _upload(server_url, video_path, **fields):
    data = {"text_prompt": "red", "tracker_name": "bytetrack", **fields}
    with open(video_path, "rb") as vf:
        res = httpx.post(
            f"{server_url}/api/track",
            files={"file": ("live.mp4", vf, "video/mp4")},
            data=data,
            timeout=30,
        )
    assert res.status_code == 200, res.text
    return res.json()


def test_stream_over_tcp_is_mjpeg(server_url, video_path):
    info = _upload(server_url, video_path)
    sid = info["id"]

    seen = 0
    blobs = []
    with httpx.stream("GET", f"{server_url}/api/stream/{sid}", timeout=10) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith(
            "multipart/x-mixed-replace"
        )
        buf = b""
        for piece in resp.iter_bytes():
            buf += piece
            seen = buf.count(b"--frame")
            if seen >= 5:
                break
        # parse out the jpeg payloads for decode checks
        for part in buf.split(b"--frame")[1:]:
            if b"\r\n\r\n" in part:
                blob = part.split(b"\r\n\r\n", 1)[1]
                blob = blob.split(b"\r\n--frame", 1)[0]
                if blob:
                    blobs.append(blob)

    assert seen >= 5
    assert b"Content-Type: image/jpeg" in buf
    img = cv2.imdecode(np.frombuffer(blobs[0], np.uint8), cv2.IMREAD_COLOR)
    assert img is not None and img.shape == (320, 480, 3)


def test_runtime_switch_while_streaming(server_url, video_path):
    info = _upload(server_url, video_path)
    sid = info["id"]

    # drain a few frames first
    with httpx.stream("GET", f"{server_url}/api/stream/{sid}", timeout=10) as resp:
        buf = b""
        for piece in resp.iter_bytes():
            buf += piece
            if buf.count(b"--frame") >= 3:
                break

    ctrl = httpx.post(
        f"{server_url}/api/control/{sid}",
        json={"tracker_name": "sort", "text_prompt": "red square",
              "box_threshold": 0.2},
        timeout=10,
    )
    assert ctrl.status_code == 200
    params = ctrl.json()["params"]
    assert params["tracker_name"] == "sort"
    assert params["text_prompt"] == "red square"
    assert params["box_threshold"] == 0.2


def test_status_and_finish(server_url, video_path):
    info = _upload(server_url, video_path)
    sid = info["id"]
    deadline = time.time() + 30
    status = None
    while time.time() < deadline:
        status = httpx.get(f"{server_url}/api/status/{sid}", timeout=5).json()
        if status["finished"] or status["error"]:
            break
        time.sleep(0.05)
    assert status is not None
    assert status["error"] is None, status["error"]
    assert status["finished"] is True
    assert status["frame_count"] > 0
    assert status["active_tracks"] >= 1


def test_validation_errors(server_url, video_path):
    with open(video_path, "rb") as vf:
        res = httpx.post(
            f"{server_url}/api/track",
            files={"file": ("live.mp4", vf, "video/mp4")},
            data={"tracker_name": "hypernet"},
            timeout=10,
        )
    assert res.status_code == 400
    assert "unknown tracker" in res.text


def test_rtsp_live_session_over_tcp(server_url):
    """A live RTSP session over a real socket: stays alive on a dead source,
    supports runtime controls, and stops cleanly when asked."""
    res = httpx.post(
        f"{server_url}/api/rtsp",
        json={
            "url": "rtsp://127.0.0.1:9/nope",
            "tracker_name": "iou",
            "text_prompt": "red",
        },
        timeout=10,
    )
    assert res.status_code == 200, res.text
    sid = res.json()["id"]

    status = httpx.get(f"{server_url}/api/status/{sid}", timeout=5).json()
    assert status["is_live"] is True
    assert status["source_type"] == "rtsp"
    assert status["finished"] is False
    assert status["params"]["tracker_name"] == "iou"

    ctrl = httpx.post(
        f"{server_url}/api/control/{sid}",
        json={"tracker_name": "sort", "detect_interval": 3},
        timeout=10,
    )
    assert ctrl.status_code == 200
    assert ctrl.json()["params"]["tracker_name"] == "sort"
    assert ctrl.json()["params"]["detect_interval"] == 3

    assert httpx.post(f"{server_url}/api/stop/{sid}", timeout=10).status_code == 200
    deadline = time.time() + 15
    while time.time() < deadline:
        status = httpx.get(f"{server_url}/api/status/{sid}", timeout=5).json()
        if status["finished"]:
            break
        time.sleep(0.05)
    assert status["finished"] is True


def test_demo_video_over_tcp(server_url):
    res = httpx.get(f"{server_url}/api/demo-video", timeout=30)
    assert res.status_code == 200
    assert res.headers["content-type"] == "video/mp4"
    assert len(res.content) > 1000