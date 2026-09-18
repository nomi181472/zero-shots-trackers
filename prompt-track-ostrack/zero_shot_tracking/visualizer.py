"""Drawing helpers for detector-free tracker output."""

from __future__ import annotations

import cv2
import numpy as np


def _color(track_id: int) -> tuple[int, int, int]:
    """Distinct, vibrant BGR color for the target track."""
    hue = (track_id * 47 + 60) % 180
    hsv = np.uint8([[[hue, 220, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_tracks(
    frame: np.ndarray,
    tracks: list,
    labels: list[str] | None = None,
    class_map: dict[str, str] | None = None,
    show_trails: bool = True,
) -> np.ndarray:
    """Draw every tracked target: box, id/label, score, and motion trail."""
    annotated = frame.copy()
    labels = labels or []

    for tr in tracks:
        if not hasattr(tr, "tlbr"):
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in tr.tlbr]
        track_id = getattr(tr, "track_id", 1)
        score = getattr(tr, "score", 1.0)
        trail = getattr(tr, "trail", [])
        color = _color(track_id)

        # motion trail
        if show_trails and len(trail) > 1:
            pts = np.asarray(trail, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(
                annotated, [pts], False, color, thickness=2, lineType=cv2.LINE_AA
            )

        # bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # label tag
        class_name = labels[0] if labels else "target"
        tag = f"#{track_id} {class_name} ({score:.2f})"
        (tw, th), baseline = cv2.getTextSize(
            tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )

        y0 = max(y1 - th - baseline - 4, 0)
        fill = tuple(int(c * 0.6) for c in color)
        cv2.rectangle(annotated, (x1, y0), (x1 + tw + 8, y1), fill, -1)
        cv2.putText(
            annotated,
            tag,
            (x1 + 4, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return annotated


def draw_legend(
    frame: np.ndarray,
    labels: list[str] | None = None,
    text_prompt: str = "visual target",
    tracker_name: str = "ostrack",
) -> np.ndarray:
    """Overlay the tracker name and prompt in the top-left corner."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (420, 50), (0, 0, 0), -1)
    annotated = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    cv2.putText(
        annotated,
        f"Tracker: {tracker_name.upper()}",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (100, 255, 180),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"Mode: Detector-Free ({text_prompt})",
        (12, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 255),
        1,
        cv2.LINE_AA,
    )
    return annotated


def draw_frame_counter(
    frame: np.ndarray, frame_id: int, total: int | None = None
) -> np.ndarray:
    """Draw frame index counter in the bottom-left corner."""
    tag = f"frame {frame_id}" if total is None else f"frame {frame_id}/{total}"
    cv2.putText(
        frame,
        tag,
        (10, frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return frame
