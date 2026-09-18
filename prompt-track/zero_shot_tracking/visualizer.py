"""Drawing helpers for tracker output."""

from __future__ import annotations

import cv2
import numpy as np

from .tracker import Track


def _color(track_id: int) -> tuple[int, int, int]:
    """Stable, distinct BGR color per track id."""
    hue = (track_id * 47) % 180
    hsv = np.uint8([[[hue, 200, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_tracks(
    frame: np.ndarray,
    tracks: list[Track],
    labels: list[str],
    class_map: dict[str, str] | None = None,
    show_trails: bool = True,
) -> np.ndarray:
    """Draw every track: box, id, class, score, and motion trail."""
    annotated = frame.copy()
    class_map = class_map or {}

    for tr in tracks:
        x1, y1, x2, y2 = [int(v) for v in tr.tlbr]
        color = _color(tr.track_id)

        # motion trail from the last positions of the Kalman state
        if show_trails and len(tr.trail) > 1:
            pts = np.asarray(tr.trail, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(annotated, [pts], False, color, thickness=2,
                          lineType=cv2.LINE_AA)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        class_name = class_map.get(labels[tr.class_id], labels[tr.class_id]) \
            if labels else str(tr.class_id)
        tag = f"#{tr.track_id} {class_name} {tr.score:.2f}"
        (tw, th), baseline = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX,
                                             0.5, 1)

        # label pill
        y0 = max(y1 - th - baseline - 4, 0)
        fill = tuple(int(c * 0.6) for c in color)
        cv2.rectangle(annotated, (x1, y0), (x1 + tw + 6, y1), fill, -1)
        cv2.putText(annotated, tag, (x1 + 3, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)

    return annotated


def draw_legend(
    frame: np.ndarray, labels: list[str], text_prompt: str
) -> np.ndarray:
    """Overlay the prompt and active classes in the top-left corner."""
    h, _ = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (600, 30 + 18 * max(len(labels), 1)),
                  (0, 0, 0), -1)
    annotated = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

    cv2.putText(annotated, f"prompt: {text_prompt}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1, cv2.LINE_AA)
    for i, name in enumerate(labels):
        cv2.putText(annotated, f"  [{i}] {name}", (10, 22 + 18 * (i + 1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 255, 150), 1,
                    cv2.LINE_AA)
    return annotated


def draw_frame_counter(frame: np.ndarray, frame_id: int,
                       total: int | None = None) -> np.ndarray:
    tag = f"frame {frame_id}" if total is None else f"frame {frame_id}/{total}"
    cv2.putText(frame, tag, (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return frame