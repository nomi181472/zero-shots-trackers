"""Detector factory.

The pipeline talks to a detector through one interface::

    detect(frame_bgr, text_prompt[, box_threshold, text_threshold])
        -> (dets (N,6), labels (list[str]), height)

``groundingdino`` is the real open-vocabulary detector; ``mock`` is a small
color-blob detector used to run and test the full stack locally (no torch /
weights needed) — it stands in for GroundingDINO during development.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from .detector import GroundingDINODetector


class MockBlobDetector:
    """Detect a single hue range; confidence = pure-pixel coverage of the box."""

    device = "cpu"

    def __init__(
        self,
        hue_low: int = 0,
        hue_high: int = 15,
        sat_min: int = 60,
        val_min: int = 60,
        min_area: int = 60,
        box_threshold: float = 0.3,
        text_threshold: float = 0.1,
    ) -> None:
        self._hue_range = (int(hue_low), int(hue_high))
        self._sat_min = int(sat_min)
        self._val_min = int(val_min)
        self.min_area = int(min_area)
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    def detect(
        self,
        frame_bgr: np.ndarray,
        text_prompt: str = "blob",
        box_threshold: float | None = None,
        text_threshold: float | None = None,
    ) -> tuple[np.ndarray, list[str], int]:
        box_tr = box_threshold if box_threshold is not None else self.box_threshold
        h, w = frame_bgr.shape[:2]

        lower = np.uint8([self._hue_range[0], self._sat_min, self._val_min])
        upper = np.uint8([self._hue_range[1], 255, 255])

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        dets = []
        for contour in contours:
            x, y, bbox_w, bbox_h = cv2.boundingRect(contour)
            area = bbox_w * bbox_h
            if area < self.min_area:
                continue
            box_mask = mask[y : y + bbox_h, x : x + bbox_w]
            coverage = float((box_mask > 0).sum()) / max(area, 1)
            score = float(np.clip(coverage, 0.0, 1.0))
            if score < box_tr:
                continue
            dets.append(
                [x, y, x + bbox_w, y + bbox_h, score, 0]
            )

        labels = ["blob"]
        if not dets:
            return np.empty((0, 6), dtype=np.float32), labels, h
        return np.asarray(dets, dtype=np.float32), labels, h

    @staticmethod
    def nms(dets: np.ndarray, iou_threshold: float = 0.5) -> np.ndarray:
        return GroundingDINODetector.nms(dets, iou_threshold)


def create_detector(kind: str = "auto", **kwargs):
    """Return a detector instance.

    kind:
      * "auto"       -> groundingsdino if weights exist, else mock, else error
      * "groundingdino" / "gdino"
      * "mock" / "blob"
    """
    kind = (kind or "auto").lower().strip()

    if kind in ("mock", "blob"):
        mock_kw = {k: v for k, v in kwargs.items()
                   if k not in ("device", "config_path", "weights_path")}
        return MockBlobDetector(**mock_kw)

    if kind in ("groundingdino", "gdino", "auto"):
        config_path = kwargs.get("config_path") or os.environ.get(
            "ZST_CONFIG", "checkpoints/GroundingDINO_SwinT_OGC.cfg.py"
        )
        weights_path = kwargs.get("weights_path") or os.environ.get(
            "ZST_WEIGHTS", "checkpoints/groundingdino_swint_ogc.pth"
        )
        if not os.path.exists(weights_path):
            if kind == "auto":
                mock_kw = {k: v for k, v in kwargs.items()
                           if k not in ("device", "config_path", "weights_path")}
                return MockBlobDetector(**mock_kw)
            raise FileNotFoundError(
                f"model weights missing: {weights_path} (run "
                "scripts/download_weights.py, or set ZST_DETECTOR=mock)"
            )
        return GroundingDINODetector(
            config_path=config_path,
            weights_path=weights_path,
            device=kwargs.get("device") or os.environ.get("ZST_DEVICE", "cuda"),
            box_threshold=kwargs.get("box_threshold", 0.35),
            text_threshold=kwargs.get("text_threshold", 0.25),
        )

    raise ValueError(f"unknown detector {kind!r}")