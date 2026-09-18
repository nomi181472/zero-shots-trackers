"""Alternative trackers: SORT (Bewley et al. 2016) and a naive IoU baseline.

Both expose the same ``update(dets)`` interface as :class:`BYTETracker`, where
``dets`` is an ``(N, 6)`` array of ``[x1, y1, x2, y2, score, class_id]`` and
``update`` returns a list of :class:`Track` namedtuples — so the pipeline can
swap tracker implementations at runtime.
"""

from __future__ import annotations

import numpy as np

from .tracker import Track, _iou_batch


# ---------------------------------------------------------------------------
# minimal algebraic Kalman filter (subset of filterpy's KalmanFilter)
# ---------------------------------------------------------------------------
class _KF:
    def __init__(self, dim_x: int, dim_z: int) -> None:
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.x = np.zeros((dim_x, 1))
        self.P = np.eye(dim_x)
        self.F = np.eye(dim_x)
        self.H = np.zeros((dim_z, dim_x))
        self.R = np.eye(dim_z)
        self.Q = np.eye(dim_x)

    def predict(self) -> None:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray) -> None:
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.dim_x) - K @ self.H) @ self.P


# ---------------------------------------------------------------------------
# SORT -- Simple Online and Realtime Tracking (Bewley et al., 2016)
# ---------------------------------------------------------------------------
def _bbox_to_z(tlbr: np.ndarray) -> np.ndarray:
    """tlbr -> 4x1 measurement ``[cx, cy, area, aspect_ratio]``."""
    tlbr = np.asarray(tlbr, dtype=np.float64)
    w, h = tlbr[2] - tlbr[0], tlbr[3] - tlbr[1]
    x = tlbr[0] + w / 2.0
    y = tlbr[1] + h / 2.0
    s = max(w * h, 1.0)
    r = w / h if h > 1e-6 else w
    return np.array([[x], [y], [s], [r]])


def _x_to_bbox(x: np.ndarray) -> np.ndarray:
    """state vector -> tlbr box."""
    s, r = float(x[2, 0]), float(x[3, 0])
    w = np.sqrt(np.clip(s * r, 1e-6, 1e12))
    h = s / w if w > 0 else 0.0
    x1, y1 = float(x[0, 0]) - w / 2.0, float(x[1, 0]) - h / 2.0
    return np.asarray([x1, y1, x1 + w, y1 + h], dtype=np.float64)


class KalmanBoxTracker:
    _count = 0

    def __init__(self, tlbr: np.ndarray, score: float, class_id: int) -> None:
        self.kf = _KF(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ]
        )
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ]
        )
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = _bbox_to_z(tlbr)
        self.time_since_update = 0
        self.id = KalmanBoxTracker._count
        KalmanBoxTracker._count += 1
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.score = score
        self.class_id = int(class_id)

    def update(self, tlbr: np.ndarray, score: float, class_id: int) -> None:
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.score = score
        self.class_id = int(class_id)
        self.kf.update(_bbox_to_z(tlbr))

    def predict(self) -> np.ndarray:
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] = 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.get_state()

    def get_state(self) -> np.ndarray:
        return _x_to_bbox(self.kf.x)


def _associate_detections_to_trackers(
    detections: np.ndarray, trackers: np.ndarray, iou_threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SORT's Hungarian association on IoU (cost = -IoU)."""
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0,), dtype=int),
        )

    iou_matrix = _iou_batch(detections, trackers)
    if iou_matrix.shape[0] > 0:
        try:
            from scipy.optimize import linear_sum_assignment

            ri, ci = linear_sum_assignment(-iou_matrix)
            matched = np.stack([ri, ci], axis=1)
        except ImportError:  # greedy fallback (scipy is a hard dep, still)
            matched = np.empty((0, 2), dtype=int)
            cost = iou_matrix.copy()
            while True:
                rr, cc = np.unravel_index(np.argmax(cost), cost.shape)
                if cost[rr, cc] < iou_threshold:
                    break
                matched = np.append(matched, [[rr, cc]], axis=0)
                cost[rr, :] = -1.0
                cost[:, cc] = -1.0
    else:
        matched = np.empty((0, 2), dtype=int)

    matched = matched[iou_matrix[matched[:, 0], matched[:, 1]] >= iou_threshold]

    unmatched_det = np.asarray(
        [d for d in range(len(detections)) if d not in matched[:, 0]], dtype=int
    )
    unmatched_trk = np.asarray(
        [t for t in range(len(trackers)) if t not in matched[:, 1]], dtype=int
    )
    return matched, unmatched_det, unmatched_trk


class SORTTracker:
    """SORT: Kalman (7-dim) + Hungarian IoU association + age-based cleanup."""

    def __init__(
        self,
        min_iou: float = 0.3,
        max_age: int = 30,
        min_hits: int = 1,
        track_thresh: float = 0.0,
        min_box_area: int = 10,
    ) -> None:
        self.min_iou = min_iou
        self.max_age = max_age
        self.min_hits = min_hits
        self.track_thresh = track_thresh
        self.min_box_area = min_box_area
        self.trackers: list[KalmanBoxTracker] = []
        self.frame_id = 0

    def update(self, dets: np.ndarray) -> list[Track]:
        self.frame_id += 1

        if len(dets):
            areas = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
            dets = dets[areas >= self.min_box_area]

        # predict every track toward the current time step
        trks = np.zeros((len(self.trackers), 4))
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks[t, :] = pos
            if np.any(np.isnan(pos)):
                to_del.append(t)
        for t in reversed(to_del):
            self.trackers.pop(t)

        if len(dets):
            boxes, scores, cls = dets[:, :4], dets[:, 4], dets[:, 5]
            matched, unmatched_det, unmatched_trks = _associate_detections_to_trackers(
                boxes, trks[~np.isin(np.arange(len(trks)), to_del)] if to_del else trks,
                self.min_iou,
            )
            for m in matched:
                self.trackers[m[1]].update(boxes[m[0]], scores[m[0]], cls[m[0]])
            for i in unmatched_det:
                if scores[i] >= self.track_thresh:
                    self.trackers.append(
                        KalmanBoxTracker(boxes[i], scores[i], cls[i])
                    )

        # remove tracks that have been unseen too long
        for trk in list(self.trackers):
            if trk.time_since_update > self.max_age:
                self.trackers.remove(trk)

        result = []
        for trk in self.trackers:
            if trk.hit_streak >= self.min_hits and trk.time_since_update == 0:
                x1, y1, x2, y2 = trk.get_state()
                result.append(
                    Track(
                        tlbr=(float(x1), float(y1), float(x2), float(y2)),
                        track_id=trk.id,
                        score=float(trk.score),
                        class_id=trk.class_id,
                        frame_id=self.frame_id,
                        trail=[],
                    )
                )
        return result


# ---------------------------------------------------------------------------
# IoU-only baseline (no motion model, greedy matching)
# ---------------------------------------------------------------------------
class _IouTrack:
    def __init__(self, track_id: int, box: np.ndarray, score: float,
                 class_id: int) -> None:
        self.id = track_id
        self.box = np.asarray(box, dtype=np.float64).copy()
        self.score = score
        self.class_id = int(class_id)
        self.hits = 1
        self.lost = 0


class IOUTracker:
    """Simplest baseline: greedy IoU matching, no Kalman prediction."""

    def __init__(
        self,
        match_thresh: float = 0.3,
        track_buffer: int = 5,
        min_box_area: int = 10,
        min_hits: int = 1,
    ) -> None:
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.min_box_area = min_box_area
        self.min_hits = min_hits
        self._tracks: list[_IouTrack] = []
        self._next_id = 0
        self.frame_id = 0

    def update(self, dets: np.ndarray) -> list[Track]:
        self.frame_id += 1

        if len(dets):
            areas = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
            dets = dets[areas >= self.min_box_area]

        used_trk: set[int] = set()
        used_det: set[int] = set()

        if len(dets):
            for d, det in enumerate(dets):
                box, score, cls = det[:4], det[4], det[5]
                best_iou, best_t = -1.0, None
                for t, trk in enumerate(self._tracks):
                    if t in used_trk:
                        continue
                    iou = _iou_batch(box[None, :], trk.box[None, :])[0, 0]
                    if iou > best_iou:
                        best_iou, best_t = iou, t
                if best_t is not None and best_iou >= self.match_thresh:
                    trk = self._tracks[best_t]
                    trk.box = box.astype(np.float64)
                    trk.score = float(score)
                    trk.class_id = int(cls)
                    trk.hits += 1
                    trk.lost = 0
                    used_trk.add(best_t)
                    used_det.add(d)
            for d, det in enumerate(dets):
                if d in used_det:
                    continue
                self._tracks.append(
                    _IouTrack(self._next_id, det[:4], det[4], det[5])
                )
                self._next_id += 1

        for trk in self._tracks:
            trk.lost += 1
        self._tracks = [
            trk for trk in self._tracks if trk.lost <= self.track_buffer
        ]

        return [
            Track(
                tlbr=tuple(float(v) for v in trk.box),
                track_id=trk.id,
                score=trk.score,
                class_id=trk.class_id,
                frame_id=self.frame_id,
                trail=[],
            )
            for trk in self._tracks
            if trk.hits >= self.min_hits
        ]