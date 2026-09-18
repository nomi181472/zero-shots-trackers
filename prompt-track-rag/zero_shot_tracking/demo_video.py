"""Synthetic clip with moving red blobs — something the local *mock*
detector can actually find boxes in (it looks for red, hue 0-15)."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np


def make_demo_video(
    path: str | Path,
    frames: int = 150,
    fps: int = 24,
    width: int = 720,
    height: int = 404,
    blobs: int = 5,
    seed: int = 7,
) -> str:
    """Render and save an MP4 of drifting red circles; returns the path."""
    path = os.fspath(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    vw = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*"mp4v"), int(fps), (int(width), int(height))
    )
    rng = np.random.default_rng(seed)
    items = []
    for _ in range(int(blobs)):
        items.append(
            [
                rng.uniform(0.05, 0.9) * width,
                rng.uniform(0.15, 0.7) * height,
                rng.uniform(-2.5, 2.5),
                rng.uniform(-1.5, 1.5),
                rng.uniform(14, 28),
            ]
        )

    margin = 24.0
    for _ in range(int(frames)):
        frame = np.full((height, width, 3), 246, np.uint8)
        for item in items:
            x, y, dx, dy, r = item
            x += dx
            y += dy
            if x < margin:
                x, dx = margin, abs(dx)
            if x > width - margin:
                x, dx = width - margin, -abs(dx)
            if y < margin:
                y, dy = margin, abs(dy)
            if y > height - margin:
                y, dy = height - margin, -abs(dy)
            item[:4] = x, y, dx, dy
            cv2.circle(frame, (int(x), int(y)), int(r), (0, 0, 255), -1)
            cv2.circle(frame, (int(x), int(y)), max(int(r), 14), (180, 180, 180), 1)
        cv2.circle(frame, (2, 2), 2, (0, 0, 0), -1)  # tiny marker to avoid pure frames
        vw.write(frame)
    vw.release()
    return path


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        description="Generate a red-blob demo clip for the mock detector."
    )
    p.add_argument("-o", "--output", default="runtime/demo-blobs.mp4")
    p.add_argument("--frames", type=int, default=150)
    p.add_argument("--fps", type=int, default=24)
    a = p.parse_args(argv)
    out = make_demo_video(a.output, frames=a.frames, fps=a.fps)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())