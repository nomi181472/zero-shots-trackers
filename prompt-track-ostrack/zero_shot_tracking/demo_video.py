"""Synthetic demo video generator with an identifiable moving visual target.

Generates an MP4 video with a distinct colored textured target moving across
a background, perfect for testing detector-free visual tracking (OSTrack, STARK,
and template matching).
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np


def make_demo_video(
    path: str | Path,
    frames: int = 150,
    fps: int = 24,
    width: int = 640,
    height: int = 360,
    seed: int = 42,
) -> str:
    """Render and save an MP4 of an identifiable object with rich texture moving smoothly."""
    path = os.fspath(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    vw = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*"mp4v"), int(fps), (int(width), int(height))
    )

    # Initial target parameters (starts near center of frame 1: box around [270, 130, 370, 230])
    tx, ty = float(width) / 2.0, float(height) / 2.0
    vx, vy = 3.2, 1.8
    obj_w, obj_h = 70, 70

    # Distractor shapes
    distractors = [
        {"x": 120.0, "y": 80.0, "vx": -1.5, "vy": 1.2, "radius": 24, "color": (50, 180, 50)},
        {"x": 520.0, "y": 260.0, "vx": 1.2, "vy": -2.0, "radius": 30, "color": (220, 100, 40)},
    ]

    for f_idx in range(int(frames)):
        # Dynamic subtle background
        frame = np.full((height, width, 3), 240, dtype=np.uint8)
        # Grid lines in background
        for gx in range(0, width, 80):
            cv2.line(frame, (gx, 0), (gx, height), (225, 225, 225), 1)
        for gy in range(0, height, 60):
            cv2.line(frame, (0, gy), (width, gy), (225, 225, 225), 1)

        # Update distractors
        for d in distractors:
            d["x"] += d["vx"]
            d["y"] += d["vy"]
            if d["x"] < d["radius"] or d["x"] > width - d["radius"]:
                d["vx"] = -d["vx"]
            if d["y"] < d["radius"] or d["y"] > height - d["radius"]:
                d["vy"] = -d["vy"]
            cv2.circle(frame, (int(d["x"]), int(d["y"])), int(d["radius"]), d["color"], -1)

        # Update main target position
        tx += vx
        ty += vy
        hw, hh = obj_w / 2.0, obj_h / 2.0
        if tx - hw < 10 or tx + hw > width - 10:
            vx = -vx
        if ty - hh < 10 or ty + hh > height - 10:
            vy = -vy

        # Draw primary target (vibrant blue/orange with internal circle and crosshair)
        x1, y1 = int(tx - hw), int(ty - hh)
        x2, y2 = int(tx + hw), int(ty + hh)
        # Target outer rounded card
        cv2.rectangle(frame, (x1, y1), (x2, y2), (240, 60, 40), -1)
        cv2.rectangle(frame, (x1 + 6, y1 + 6), (x2 - 6, y2 - 6), (255, 255, 255), -1)
        cv2.circle(frame, (int(tx), int(ty)), 16, (40, 140, 255), -1)
        cv2.circle(frame, (int(tx), int(ty)), 6, (255, 255, 255), -1)
        cv2.putText(
            frame, "TARGET", (x1 + 8, y2 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (40, 40, 40), 1, cv2.LINE_AA
        )

        vw.write(frame)

    vw.release()
    return path
