"""Zero-shot detector built on GroundingDINO (IDEA-Research)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class GroundingDINODetector:
    """Detect arbitrary categories from a free-text prompt.

    Example prompt for two classes: ``"person, car"``.

    Detections are returned in pixel coordinates: ``[x1, y1, x2, y2, score,
    class_id]`` with ``class_id`` indexing the split prompt terms. This layout
    is exactly what ByteTrack consumes.

    Heavy dependencies (torch, cv2, PIL, groundingdino) are imported lazily so
    the helper/geometry methods stay importable without them — handy for unit
    testing and for checking prompts before the model is even downloaded.
    """

    def __init__(
        self,
        config_path: str,
        weights_path: str,
        device: str = "cuda",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> None:
        import torch

        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        # GroundingDINO requires transformers APIs removed in 4.45+; reattach them.
        from . import gdino_compat

        gdino_compat.apply()

        from groundingdino.util.inference import load_model as _load_model

        self.model = _load_model(config_path, weights_path, device=self.device)
        self.model.eval()

    def detect(
        self,
        frame_bgr: np.ndarray,
        text_prompt: str,
        box_threshold: float | None = None,
        text_threshold: float | None = None,
    ) -> tuple[np.ndarray, list[str], int]:
        """Run open-vocabulary detection on one BGR frame.

        Returns ``(dets, labels, height)`` where ``dets`` is an ``(N, 6)``
        array of ``[x1, y1, x2, y2, score, class_id]`` and ``labels`` maps
        ``class_id -> phrase``.
        """
        import cv2
        import torch
        from PIL import Image

        import groundingdino.datasets.transforms as T

        box_tr = self.box_threshold if box_threshold is None else box_threshold
        text_tr = self.text_threshold if text_threshold is None else text_threshold

        h, w = frame_bgr.shape[:2]

        # GroundingDINO expects a (H, W) torch tensor normalized the same way
        # as its training pipeline (RandomResize -> ToTensor -> Normalize).
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_tensor, _ = transform(image_pil, None)

        dets = np.empty((0, 6), dtype=np.float32)
        labels: list[str] = []

        # GroundingDINO expects the official "class . class ." sentence format
        # (literal "." tokens become the segment separators that shape its
        # chunked text self-attention masks) — a plain "a, b" string makes its
        # mask builder produce an empty segment list and crash.
        caption = " . ".join(self._split_prompt(text_prompt)) + " ."

        from groundingdino.util.inference import predict

        boxes_cxcywh, scores, phrases = predict(
            self.model,
            image_tensor,
            caption,
            box_tr,
            text_tr,
            device=self.device,
        )
        if len(scores) > 0:
            cx = boxes_cxcywh[:, 0].numpy() * w
            cy = boxes_cxcywh[:, 1].numpy() * h
            bw = boxes_cxcywh[:, 2].numpy() * w
            bh = boxes_cxcywh[:, 3].numpy() * h
            x1 = (cx - bw / 2).clip(0, w - 1)
            y1 = (cy - bh / 2).clip(0, h - 1)
            x2 = (cx + bw / 2).clip(1, w)
            y2 = (cy + bh / 2).clip(1, h)
            class_ids = [self._match_class(p, text_prompt) for p in phrases]
            dets = np.stack(
                [x1, y1, x2, y2, scores.numpy(),
                 np.asarray(class_ids, dtype=np.float32)],
                axis=1,
            )
            labels = self._split_prompt(text_prompt)

        return dets.astype(np.float32), labels, h

    # ------------------------------------------------------------------
    # numpy-only helpers (usable without torch / groundingdino installed)
    # ------------------------------------------------------------------
    @staticmethod
    def _split_prompt(text_prompt: str) -> list[str]:
        """Split ``"person, car"`` into ``["person", "car"]``."""
        parts = [p.strip().lower() for p in text_prompt.split(",") if p.strip()]
        return parts or ["object"]

    @staticmethod
    def _match_class(phrase: str, text_prompt: str) -> int:
        """Return the index of the prompt term best-served by ``phrase``.

        Examples:
            phrase="a person riding"      prompt="person, bicycle"  -> 0
            phrase="car"                  prompt="person, car"      -> 1
            phrase="some people"          prompt="person"           -> 0
        """
        terms = GroundingDINODetector._split_prompt(text_prompt)
        phrase_l = phrase.lower()
        scores = [max(int(t in phrase_l), int(phrase_l in t)) for t in terms]
        best = max(range(len(terms)), key=lambda i: (scores[i], -i))
        return best if scores[best] else 0

    @staticmethod
    def nms(dets: np.ndarray, iou_threshold: float = 0.5) -> np.ndarray:
        """Greedy NMS on ``[x1, y1, x2, y2, score, class_id]`` detections."""
        if len(dets) == 0:
            return dets

        x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = np.argsort(-dets[:, 4])

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            iou = inter / np.maximum(areas[i] + areas[order[1:]] - inter, 1e-9)
            order = order[1:][iou <= iou_threshold]

        return dets[np.asarray(keep)]