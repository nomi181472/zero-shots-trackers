"""OSTrack / STARK / Template-Matching for Detector-Free Visual Tracking.

Provides a unified :class:`OSTracker` interface that supports:
1. ``"ostrack"`` — One-stream transformer tracker (or spatial-window correlation).
2. ``"stark"`` — Spatio-temporal adaptive tracking with dual-template updates.
3. ``"mock"`` — Normalized cross-correlation multi-scale template matching.

Usage:
    tracker = OSTracker(tracker_type="ostrack")
    tracker.init(frame_bgr, box=[x1, y1, x2, y2])  # or crop=crop_bgr
    box = tracker.update(frame_bgr)               # returns (x1, y1, x2, y2)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import ostrack  # type: ignore
    _HAS_OSTRACK = True
except Exception:
    _HAS_OSTRACK = False

try:
    import stark_tracking  # type: ignore
    _HAS_STARK = True
except Exception:
    _HAS_STARK = False


class OSTracker:
    """Detector-free single-object visual tracker.

    Parameters
    ----------
    tracker_type : {"ostrack", "stark", "mock"}
        Visual tracking algorithm backend.
    """

    def __init__(self, tracker_type: str = "ostrack") -> None:
        self.tracker_type = (tracker_type or "ostrack").lower().strip()
        self._template: Optional[np.ndarray] = None          # Initial frame 1 template
        self._dynamic_template: Optional[np.ndarray] = None  # STARK online template
        self._init_box: Optional[Tuple[float, float, float, float]] = None
        self._last_box: Optional[Tuple[float, float, float, float]] = None
        self._last_score: float = 1.0
        self._frame_count: int = 0
        self._trail: list[Tuple[int, int]] = []
        self._search_sz = 64
        self._update_interval = 5

        # Backend selection
        if self.tracker_type == "ostrack" and _HAS_OSTRACK:
            self._backend_name = "OSTrack (Native PyTorch)"
            self._init_fn = self._ostrack_native_init
            self._update_fn = self._ostrack_native_update
        elif self.tracker_type == "stark" and _HAS_STARK:
            self._backend_name = "STARK (Native PyTorch)"
            self._init_fn = self._stark_native_init
            self._update_fn = self._stark_native_update
        elif self.tracker_type == "stark":
            self._backend_name = "STARK (Adaptive Dual-Template)"
            self._init_fn = self._stark_adaptive_init
            self._update_fn = self._stark_adaptive_update
        elif self.tracker_type == "ostrack":
            self._backend_name = "OSTrack (Spatial Search Correlation)"
            self._init_fn = self._ostrack_search_init
            self._update_fn = self._ostrack_search_update
        else:
            self.tracker_type = "mock"
            self._backend_name = "Mock (Template Matching)"
            self._init_fn = self._mock_init
            self._update_fn = self._mock_update

    @property
    def last_box(self) -> Optional[Tuple[float, float, float, float]]:
        return self._last_box

    @property
    def last_score(self) -> float:
        return self._last_score

    @property
    def trail(self) -> list[Tuple[int, int]]:
        return self._trail

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init(
        self,
        frame_bgr: np.ndarray,
        box: Optional[Tuple[float, float, float, float]] = None,
        crop: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize the tracker on frame 1 with either a box or object crop."""
        h, w = frame_bgr.shape[:2]

        if crop is not None and crop.size > 0:
            crop_h, crop_w = crop.shape[:2]
            if box is not None:
                self._init_box = tuple(float(v) for v in box)
                self._template = crop.copy()
            else:
                # Find crop in frame 1
                if crop_h <= h and crop_w <= w:
                    # Choose matching method based on variance
                    if np.std(crop) < 1.0:
                        res = cv2.matchTemplate(frame_bgr, crop, cv2.TM_SQDIFF)
                        min_val, _, min_loc, _ = cv2.minMaxLoc(res)
                        x1, y1 = float(min_loc[0]), float(min_loc[1])
                    else:
                        res = cv2.matchTemplate(frame_bgr, crop, cv2.TM_CCOEFF_NORMED)
                        _, _, _, max_loc = cv2.minMaxLoc(res)
                        x1, y1 = float(max_loc[0]), float(max_loc[1])
                    x2, y2 = x1 + float(crop_w), y1 + float(crop_h)
                    self._init_box = (x1, y1, x2, y2)
                    self._template = crop.copy()
                else:
                    cx, cy = w / 2.0, h / 2.0
                    ow, oh = min(crop_w, w * 0.5), min(crop_h, h * 0.5)
                    self._init_box = (cx - ow / 2, cy - oh / 2, cx + ow / 2, cy + oh / 2)
                    self._template = cv2.resize(crop, (int(ow), int(oh)))
        elif box is not None:
            x1, y1, x2, y2 = [float(v) for v in box]
            x1 = max(0.0, min(x1, float(w - 2)))
            y1 = max(0.0, min(y1, float(h - 2)))
            x2 = max(x1 + 1.0, min(x2, float(w)))
            y2 = max(y1 + 1.0, min(y2, float(h)))
            self._init_box = (x1, y1, x2, y2)

            ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
            crop_patch = frame_bgr[iy1:iy2, ix1:ix2]
            if crop_patch.size > 0:
                self._template = crop_patch.copy()
            else:
                self._template = np.zeros((32, 32, 3), dtype=np.uint8)
        else:
            raise ValueError("Either ``box`` or ``crop`` must be provided on init().")

        self._last_box = self._init_box
        self._last_score = 1.0
        self._dynamic_template = self._template.copy() if self._template is not None else None
        self._frame_count = 1

        cx = int((self._init_box[0] + self._init_box[2]) / 2.0)
        cy = int((self._init_box[1] + self._init_box[3]) / 2.0)
        self._trail = [(cx, cy)]

        self._init_fn(frame_bgr, self._init_box, self._template)

    # ------------------------------------------------------------------
    # Frame Update
    # ------------------------------------------------------------------

    def update(self, frame_bgr: np.ndarray) -> Tuple[float, float, float, float]:
        """Track the target in the current frame and return (x1, y1, x2, y2)."""
        if self._last_box is None:
            return (0.0, 0.0, 0.0, 0.0)

        # Blank / uninformative frame fallback (identity)
        if np.std(frame_bgr) < 1.0:
            return self._last_box

        self._frame_count += 1
        box, score = self._update_fn(frame_bgr)

        self._last_box = box
        self._last_score = score
        cx = int((box[0] + box[2]) / 2.0)
        cy = int((box[1] + box[3]) / 2.0)
        self._trail.append((cx, cy))
        if len(self._trail) > 60:
            self._trail.pop(0)

        return box

    # ------------------------------------------------------------------
    # Matching Helper
    # ------------------------------------------------------------------

    @staticmethod
    def _match_patch(search_roi: np.ndarray, template: np.ndarray, apply_spatial_prior: bool = False) -> Tuple[int, int, float]:
        """Find best match of template in search ROI, returning (dx, dy, score)."""
        th, tw = template.shape[:2]
        sh, sw = search_roi.shape[:2]
        if th > sh or tw > sw:
            return 0, 0, 0.0

        is_flat = float(np.std(template, axis=(0, 1)).max()) < 1.0
        if is_flat:
            res = cv2.matchTemplate(search_roi, template, cv2.TM_SQDIFF_NORMED)
            min_val, _, min_loc, _ = cv2.minMaxLoc(res)
            score = float(max(0.0, 1.0 - min_val))
            return min_loc[0], min_loc[1], score

        res = cv2.matchTemplate(search_roi, template, cv2.TM_CCOEFF_NORMED)
        if apply_spatial_prior and res.shape[0] > 2 and res.shape[1] > 2:
            rh, rw = res.shape[:2]
            hann_y = np.hanning(rh)
            hann_x = np.hanning(rw)
            hann_mask = np.outer(hann_y, hann_x).astype(np.float32)
            res = res * (0.7 + 0.3 * hann_mask)

        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if np.isnan(max_val) or max_val < 0.2:
            return -1, -1, 0.0
        return max_loc[0], max_loc[1], float(max_val)

    # ------------------------------------------------------------------
    # Backend: OSTrack (Spatial Window Correlation)
    # ------------------------------------------------------------------

    def _ostrack_search_init(self, frame_bgr, init_box, template):
        pass

    def _ostrack_search_update(self, frame_bgr) -> Tuple[Tuple[float, float, float, float], float]:
        """One-Stream style search region extraction with spatial cosine window weighting."""
        h, w = frame_bgr.shape[:2]
        prev_x1, prev_y1, prev_x2, prev_y2 = self._last_box
        prev_w = max(4.0, prev_x2 - prev_x1)
        prev_h = max(4.0, prev_y2 - prev_y1)
        prev_cx = (prev_x1 + prev_x2) / 2.0
        prev_cy = (prev_y1 + prev_y2) / 2.0

        search_factor = 3.5
        search_w = prev_w * search_factor
        search_h = prev_h * search_factor

        sx1 = max(0, int(prev_cx - search_w / 2.0))
        sy1 = max(0, int(prev_cy - search_h / 2.0))
        sx2 = min(w, int(prev_cx + search_w / 2.0))
        sy2 = min(h, int(prev_cy + search_h / 2.0))

        if sx2 - sx1 < 10 or sy2 - sy1 < 10 or self._template is None:
            return self._last_box, 0.5

        search_roi = frame_bgr[sy1:sy2, sx1:sx2]
        tmpl = self._template
        th, tw = tmpl.shape[:2]

        if th > search_roi.shape[0] or tw > search_roi.shape[1]:
            search_roi = frame_bgr
            sx1, sy1 = 0, 0

        dx, dy, score = self._match_patch(search_roi, tmpl, apply_spatial_prior=True)
        if dx < 0 or score < 0.2:
            return self._last_box, 0.2

        new_x1 = float(sx1 + dx)
        new_y1 = float(sy1 + dy)
        new_x2 = new_x1 + float(tw)
        new_y2 = new_y1 + float(th)

        new_x1 = max(0.0, min(new_x1, float(w - 2)))
        new_y1 = max(0.0, min(new_y1, float(h - 2)))
        new_x2 = max(new_x1 + 2.0, min(new_x2, float(w)))
        new_y2 = max(new_y1 + 2.0, min(new_y2, float(h)))

        return (new_x1, new_y1, new_x2, new_y2), score

    # ------------------------------------------------------------------
    # Backend: STARK (Spatio-Temporal Adaptive Reliability Tracking)
    # ------------------------------------------------------------------

    def _stark_adaptive_init(self, frame_bgr, init_box, template):
        self._dynamic_template = template.copy() if template is not None else None

    def _stark_adaptive_update(self, frame_bgr) -> Tuple[Tuple[float, float, float, float], float]:
        """STARK-style dual-template tracking with confidence-gated online adaptation."""
        h, w = frame_bgr.shape[:2]
        prev_x1, prev_y1, prev_x2, prev_y2 = self._last_box
        prev_w = max(4.0, prev_x2 - prev_x1)
        prev_h = max(4.0, prev_y2 - prev_y1)
        prev_cx = (prev_x1 + prev_x2) / 2.0
        prev_cy = (prev_y1 + prev_y2) / 2.0

        search_factor = 3.2
        sx1 = max(0, int(prev_cx - prev_w * search_factor / 2.0))
        sy1 = max(0, int(prev_cy - prev_h * search_factor / 2.0))
        sx2 = min(w, int(prev_cx + prev_w * search_factor / 2.0))
        sy2 = min(h, int(prev_cy + prev_h * search_factor / 2.0))

        if sx2 - sx1 < 10 or sy2 - sy1 < 10 or self._template is None:
            return self._last_box, 0.5

        search_roi = frame_bgr[sy1:sy2, sx1:sx2]
        init_tmpl = self._template
        dyn_tmpl = self._dynamic_template if self._dynamic_template is not None else init_tmpl
        th, tw = init_tmpl.shape[:2]

        if th > search_roi.shape[0] or tw > search_roi.shape[1]:
            search_roi = frame_bgr
            sx1, sy1 = 0, 0

        dx, dy, score = self._match_patch(search_roi, init_tmpl)
        # If dynamic template differs and is valid, blend decision
        if dyn_tmpl.shape == init_tmpl.shape and self._dynamic_template is not None:
            dx_dyn, dy_dyn, score_dyn = self._match_patch(search_roi, dyn_tmpl)
            if score_dyn > score:
                dx, dy, score = dx_dyn, dy_dyn, score_dyn

        if dx < 0 or score < 0.2:
            return self._last_box, 0.2

        new_x1 = float(sx1 + dx)
        new_y1 = float(sy1 + dy)
        new_x2 = new_x1 + float(tw)
        new_y2 = new_y1 + float(th)

        new_x1 = max(0.0, min(new_x1, float(w - 2)))
        new_y1 = max(0.0, min(new_y1, float(h - 2)))
        new_x2 = max(new_x1 + 2.0, min(new_x2, float(w)))
        new_y2 = max(new_y1 + 2.0, min(new_y2, float(h)))

        # STARK online template adaptation
        if score > 0.65 and (self._frame_count % self._update_interval == 0):
            patch = frame_bgr[int(new_y1):int(new_y2), int(new_x1):int(new_x2)]
            if patch.shape[:2] == init_tmpl.shape[:2] and self._dynamic_template is not None:
                self._dynamic_template = cv2.addWeighted(
                    self._dynamic_template, 0.75, patch, 0.25, 0
                )

        return (new_x1, new_y1, new_x2, new_y2), score

    # ------------------------------------------------------------------
    # Backend: Mock (Template Matching)
    # ------------------------------------------------------------------

    def _mock_init(self, frame_bgr, init_box, template):
        pass

    def _mock_update(self, frame_bgr) -> Tuple[Tuple[float, float, float, float], float]:
        """Template matching fallback."""
        if self._template is None or self._template.size == 0:
            return self._last_box, 0.0

        h, w = frame_bgr.shape[:2]
        prev_x1, prev_y1, prev_x2, prev_y2 = self._last_box
        prev_w = max(4.0, prev_x2 - prev_x1)
        prev_h = max(4.0, prev_y2 - prev_y1)
        prev_cx = (prev_x1 + prev_x2) / 2.0
        prev_cy = (prev_y1 + prev_y2) / 2.0

        margin = 3.5
        sx1 = max(0, int(prev_cx - prev_w * margin / 2.0))
        sy1 = max(0, int(prev_cy - prev_h * margin / 2.0))
        sx2 = min(w, int(prev_cx + prev_w * margin / 2.0))
        sy2 = min(h, int(prev_cy + prev_h * margin / 2.0))

        search_roi = frame_bgr[sy1:sy2, sx1:sx2]
        tmpl = self._template
        th, tw = tmpl.shape[:2]

        if th > search_roi.shape[0] or tw > search_roi.shape[1]:
            search_roi = frame_bgr
            sx1, sy1 = 0, 0

        dx, dy, score = self._match_patch(search_roi, tmpl)
        if dx < 0 or score < 0.2:
            return self._last_box, 0.2

        new_x1 = float(sx1 + dx)
        new_y1 = float(sy1 + dy)
        new_x2 = new_x1 + float(tw)
        new_y2 = new_y1 + float(th)

        new_x1 = max(0.0, min(new_x1, float(w - 2)))
        new_y1 = max(0.0, min(new_y1, float(h - 2)))
        new_x2 = max(new_x1 + 2.0, min(new_x2, float(w)))
        new_y2 = max(new_y1 + 2.0, min(new_y2, float(h)))

        return (new_x1, new_y1, new_x2, new_y2), score


    # ------------------------------------------------------------------
    # Native PyTorch hooks
    # ------------------------------------------------------------------

    def _ostrack_native_init(self, frame_bgr, init_box, template):
        import ostrack  # type: ignore
        self._native_tracker = ostrack.OSTracker()
        self._native_tracker.initialize(template if template is not None else frame_bgr, init_box)

    def _ostrack_native_update(self, frame_bgr) -> Tuple[Tuple[float, float, float, float], float]:
        res = self._native_tracker.update(frame_bgr)
        box = getattr(res, "box", getattr(res, "bbox", self._last_box))
        return (float(box[0]), float(box[1]), float(box[2]), float(box[3])), 0.95

    def _stark_native_init(self, frame_bgr, init_box, template):
        import stark_tracking  # type: ignore
        self._native_tracker = stark_tracking.build_tracker()
        self._native_tracker.initialize(frame_bgr, init_box)

    def _stark_native_update(self, frame_bgr) -> Tuple[Tuple[float, float, float, float], float]:
        res = self._native_tracker.update(frame_bgr)
        box = getattr(res, "box", getattr(res, "bbox", self._last_box))
        return (float(box[0]), float(box[1]), float(box[2]), float(box[3])), 0.95

    # ------------------------------------------------------------------
    # Helper: draw()
    # ------------------------------------------------------------------

    def draw(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Draw the current tracked box and motion trail on the frame."""
        if self._last_box is None:
            return frame_bgr
        x1, y1, x2, y2 = [int(round(v)) for v in self._last_box]
        if len(self._trail) > 1:
            pts = np.asarray(self._trail, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame_bgr, [pts], False, (0, 255, 120), 2, cv2.LINE_AA)

        overlay = frame_bgr.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.25, frame_bgr, 0.75, 0, frame_bgr)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        tag = f"#{self.tracker_type.upper()} ({self._last_score:.2f})"
        cv2.putText(
            frame_bgr, tag, (x1, max(y1 - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
        )
        return frame_bgr