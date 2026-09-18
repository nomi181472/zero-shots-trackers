"""Appearance embedders — the retrieval encoder ("R") of the RAG tracker.

The RAG tracker does not associate detections with an algebraic motion model
(Kalman + IoU + Hungarian). Instead every detection crop is turned into an
embedding, and the tracker *retrieves* the most relevant memories from its
per-track store by cosine similarity. Two embedders are provided:

  * :class:`CLIPEmbedder`      — real CLIP visual features (transformers +
    torch, ``openai/clip-vit-base-patch32`` by default).
  * :class:`HistogramEmbedder` — torch-free HSV-grid + shape descriptor used
    as the offline fallback and by the unit tests (the trait of the existing
    ``MockBlobDetector``).

:func:`create_embedder` picks one via ``ZST_EMBEDDER``:

```
  auto      -> CLIP if the model loads, otherwise histogram
  clip      -> CLIP (raises if unavailable)
  histogram -> handcrafted descriptor (never needs weights/network)
```

Embedders expose two methods:

    embed_crops(frame_bgr, boxes)      -> (N, D) L2-normalized rows
    embed_text(text_prompt)            -> (K, D) L2-normalized rows, one per class
"""

from __future__ import annotations

import os
import threading
from typing import List, Optional

import cv2
import numpy as np

from .detector import GroundingDINODetector


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------
class TrackEmbedder:
    """Interface every embedder implements."""

    name = "base"
    embedding_dim = 0

    def embed_crops(
        self, frame_bgr: np.ndarray, boxes: np.ndarray
    ) -> np.ndarray:
        """L2-normalized appearance vectors, one row per tlbr box."""
        raise NotImplementedError

    def embed_text(self, text_prompt: str) -> Optional[np.ndarray]:
        """L2-normalized vectors, one per prompt class (may return None)."""
        return None

    @staticmethod
    def normalize(mat: np.ndarray) -> np.ndarray:
        mat = np.asarray(mat, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / np.maximum(norms, 1e-8)


# ---------------------------------------------------------------------------
# CLIP embedder (real RAG encoder)
# ---------------------------------------------------------------------------
class CLIPEmbedder(TrackEmbedder):
    """CLIP ViT-B/32 visual features via ``transformers``.

    Text prompts are embedded with the same model so retrieval — and the
    optional LLM generator — can also reason about classes. Heavy imports are
    lazy so the module stays importable in a minimal environment.
    """

    name = "clip"

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
    ) -> None:
        import torch  # lazily imported: heavy

        if device is None:
            device = os.environ.get("ZST_DEVICE", "cuda")
        self.device = device if (torch.cuda.is_available() or device == "cpu") else "cpu"
        self.model_name = model_name
        self.embedding_dim = 512  # clip-vit-base-patch32 pooled features

        from transformers import CLIPModel, CLIPProcessor

        kwargs = dict(local_files_only=(os.environ.get("ZST_CLIP_OFFLINE") == "1"))
        self.model = CLIPModel.from_pretrained(model_name, **kwargs).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name, **kwargs)
        self.model.eval()

    def embed_crops(
        self, frame_bgr: np.ndarray, boxes: np.ndarray
    ) -> np.ndarray:
        import torch  # lazily imported

        crops = []
        h, w = frame_bgr.shape[:2]
        for x1, y1, x2, y2 in np.asarray(boxes, dtype=np.float64):
            x1i, y1i = int(max(0, min(x1, w - 1))), int(max(0, min(y1, h - 1)))
            x2i, y2i = int(max(x1i + 1, min(x2, w))), int(max(y1i + 1, min(y2, h)))
            crop = frame_bgr[y1i:y2i, x1i:x2i]
            if crop.size == 0:
                crop = frame_bgr
            crops.append(crop)
        if not crops:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        inputs = self.processor(images=crops, return_tensors="pt").to(self.device)
        with torch.no_grad():
            feats = self.model.get_image_features(**inputs)
        return self.normalize(feats.float().cpu().numpy())

    def embed_text(self, text_prompt: str) -> Optional[np.ndarray]:
        import torch  # lazily imported

        texts = GroundingDINODetector._split_prompt(text_prompt)
        if not texts:
            return None
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(
            self.device
        )
        with torch.no_grad():
            feats = self.model.get_text_features(**inputs)
        return self.normalize(feats.float().cpu().numpy())


# ---------------------------------------------------------------------------
# Handcrafted embedder (torch-free fallback + tests)
# ---------------------------------------------------------------------------
class HistogramEmbedder(TrackEmbedder):
    """HSV grid histogram + shape descriptor, no dependencies beyond numpy/cv2.

    Each crop is divided into a 2x2 grid; per cell we compute an 8-bin hue
    histogram plus mean saturation/value. Two global shape features (aspect
    ratio, log-area) are appended. The whole descriptor is L2-normalized.
    """

    name = "histogram"
    embedding_dim = 2 * 2 * (8 + 2) + 2  # grid *= (hue + sat + val), + shape

    def embed_crops(
        self, frame_bgr: np.ndarray, boxes: np.ndarray,
    ) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        vecs = []
        for x1, y1, x2, y2 in np.asarray(boxes, dtype=np.float64):
            x1i, y1i = int(max(0, min(x1, w - 1))), int(max(0, min(y1, h - 1)))
            x2i, y2i = int(max(x1i + 1, min(x2, w))), int(max(y1i + 1, min(y2, h)))
            crop_w, crop_h = x2i - x1i, y2i - y1i
            crop = hsv[y1i:y2i, x1i:x2i]

            vec = []
            for gy in (0, 1):
                for gx in (0, 1):
                    cy0 = int(gy * crop_h / 2)
                    cx0 = int(gx * crop_w / 2)
                    cy1 = int((gy + 1) * crop_h / 2)
                    cx1 = int((gx + 1) * crop_w / 2)
                    cell = crop[cy0:cy1, cx0:cx1]
                    if cell.size == 0:
                        vec.extend([0.0] * (8 + 2))
                        continue
                    hu = cell[:, :, 0].ravel()
                    sat = cell[:, :, 1].ravel()
                    val = cell[:, :, 2].ravel()
                    hist, _ = np.histogram(
                        hu, bins=8, range=(0, 180), density=False
                    )
                    hist = hist / max(hist.sum(), 1.0)
                    vec.extend(hist.tolist())
                    vec.append(float(sat.mean() / 255.0))
                    vec.append(float(val.mean() / 255.0))

            area = max(crop_w * crop_h, 1.0)
            vec.append(float(min(crop_w / max(crop_h, 1e-6), 5.0) / 5.0))
            vec.append(float(min(np.log1p(area) / 12.0, 1.0)))
            vecs.append(vec)

        if not vecs:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        return self.normalize(np.asarray(vecs, dtype=np.float32))


# ---------------------------------------------------------------------------
# Factory (process-wide cached)
# ---------------------------------------------------------------------------
_embedder = None
_embedder_lock = threading.Lock()


def create_embedder(kind: str = "auto", **kwargs) -> TrackEmbedder:
    """Return a process-wide shared appearance embedder.

    ``auto`` tries CLIP first (live model download on first use) and falls
    back to the torch-free histogram descriptor if anything fails, so the full
    stack keeps running in offline/mock mode just like the mock detector.
    """
    global _embedder
    with _embedder_lock:
        if _embedder is not None:
            return _embedder

        kind = (kind or os.environ.get("ZST_EMBEDDER", "auto")).lower().strip()

        def _clip() -> CLIPEmbedder:
            return CLIPEmbedder(
                model_name=kwargs.get("model_name")
                or os.environ.get("ZST_CLIP_MODEL", "openai/clip-vit-base-patch32"),
                device=kwargs.get("device") or os.environ.get("ZST_DEVICE", "cuda"),
            )

        if kind in ("clip", "clip-vit", "chinese-clip"):
            return _clip()
        if kind in ("histogram", "handcrafted", "offline", "mock", "blob"):
            _embedder = HistogramEmbedder()
            return _embedder
        if kind == "auto":
            try:
                _embedder = _clip()
            except Exception:
                _embedder = HistogramEmbedder()
            return _embedder

        raise ValueError(
            f"unknown embedder {kind!r}; use auto|clip|histogram"
        )


def embedder_info() -> dict:
    """Best-effort description of the shared embedder for ``/api/health``."""
    try:
        emb = create_embedder(os.environ.get("ZST_EMBEDDER", "auto"))
        return {
            "name": emb.name,
            "loaded": True,
            "embedding_dim": emb.embedding_dim,
        }
    except Exception as exc:
        return {"loaded": False, "error": str(exc)}