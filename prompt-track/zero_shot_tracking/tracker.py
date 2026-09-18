"""ByteTrack multi-object tracker (ECCV 2022), numpy re-implementation.

Y. Zhang, P. Sun, Y. Jiang, D. Yu, F. Weng, Z. Yuan, P. Luo, W. Liu, X. Wang.
"ByteTrack: Multi-Object Tracking by Associating Every Detection Box."

The original implementation associates *every* detection, including low-score
ones, which recovers the boxes that background-cancellation thresholds throw
away. This port keeps the two-stage association:
  1. high-score detections -> tracked + lost tracks (greedy IoU matching)
  2. low-score  detections -> the remaining *tracked* tracks
plus activation of brand-new tracks and re-activation of re-found lost tracks.
"""

from __future__ import annotations

from collections import namedtuple
from typing import List, Optional, Sequence

import numpy as np

try:  # lap is optional; falls back to scipy or greedy
    import lap  # type: ignore
    _HAS_LAP = True
except ImportError:  # pragma: no cover
    _HAS_LAP = False

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False

Track = namedtuple("Track", "tlbr track_id score class_id frame_id trail")


# ---------------------------------------------------------------------------
# Track states
# ---------------------------------------------------------------------------
class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


# ---------------------------------------------------------------------------
# Kalman filter on (cx, cy, aspect-ratio, height) + velocities
# ---------------------------------------------------------------------------
class KalmanFilter:
    """8-dimensional constant-velocity Kalman filter (SORT-style math)."""

    def __init__(self, std_weight_position: float = 1.0 / 20,
                 std_weight_velocity: float = 1.0 / 160) -> None:
        self._std_weight_position = std_weight_position
        self._std_weight_velocity = std_weight_velocity
        self.ndim, self.dt = 4, 1.0

        # constant-velocity motion model
        self._motion_mat = np.eye(2 * self.ndim, 2 * self.ndim)
        for i in range(self.ndim):
            self._motion_mat[i, self.ndim + i] = self.dt
        # measurement model
        self._update_mat = np.eye(self.ndim, 2 * self.ndim)

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Create a track from a measurement ``[cx, cy, ratio, h]``."""
        mean_pos = np.asarray(measurement, dtype=np.float64)
        mean_vel = np.zeros_like(mean_pos)
        mean = np.concatenate([mean_pos, mean_vel])
        std = [
            2 * self._std_weight_position * mean[3],
            2 * self._std_weight_position * mean[3],
            1e-2,
            2 * self._std_weight_position * mean[3],
            10 * self._std_weight_position * mean[3],
            10 * self._std_weight_position * mean[3],
            1e-5,
            10 * self._std_weight_position * mean[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray,
                covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run the (constant-velocity) prediction step for one track."""
        mean, covariance = self.multi_predict(mean[None], covariance[None])
        return mean[0], covariance[0]

    def multi_predict(
        self, means: np.ndarray, covariances: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict a batch of states in one shot."""
        means_out = (self._motion_mat @ means[..., None])[..., 0]
        covariances_out = self._motion_mat @ covariances @ self._motion_mat.T
        for k, mean in enumerate(means):
            scale = mean[3]
            std = [
                self._std_weight_position * scale,
                self._std_weight_position * scale,
                1e-2,
                self._std_weight_position * scale,
                self._std_weight_velocity * scale,
                self._std_weight_velocity * scale,
                1e-5,
                self._std_weight_velocity * scale,
            ]
            covariances_out[k] += np.diag(np.square(std))
        return means_out, covariances_out

    def update(self, mean: np.ndarray, covariance: np.ndarray,
               measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Correct the state with a new measurement (standard Kalman)."""
        mean_pred = self._update_mat @ mean
        scale = mean[3]
        std = [self._std_weight_position * scale] * self.ndim
        innovation_cov = np.diag(np.square(std))
        projected_cov = (
            self._update_mat @ covariance @ self._update_mat.T + innovation_cov
        )
        kalman_gain = (
            covariance @ self._update_mat.T @ np.linalg.inv(projected_cov)
        )
        new_mean = mean + kalman_gain @ (measurement - mean_pred)
        new_cov = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_cov

    def gating_distance(self, mean: np.ndarray, covariance: np.ndarray,
                        measurements: np.ndarray,
                        only_position: bool = False) -> np.ndarray:
        """Mahalanobis distance of ``measurements`` to the predicted state."""
        mean_pred = self._update_mat @ mean
        scale = mean[3]
        std = [self._std_weight_position * scale] * self.ndim
        innovation_cov = np.diag(np.square(std))
        projected_cov = (
            self._update_mat @ covariance @ self._update_mat.T + innovation_cov
        )

        if only_position:
            mean_pred, projected_cov = mean_pred[:2], projected_cov[:2, :2]
            measurements = measurements[:, :2]

        diff = measurements - mean_pred
        chol = np.linalg.cholesky(projected_cov)
        z = np.linalg.solve(chol, diff.T)
        return (np.einsum("ij,ij->j", z, z))


# ---------------------------------------------------------------------------
# Single track
# ---------------------------------------------------------------------------
class STrack:
    """A tracklet with a shared Kalman filter across all instances."""

    _shared_kalman = None
    _next_id = 0

    def __init__(self, tlwh: Sequence[float], score: float,
                 class_id: int, trail_len: int = 30) -> None:
        self._tlwh = np.asarray(tlwh, dtype=np.float64)  # x, y, w, h
        self.kalman_filter = self._shared_kalman
        self.mean: Optional[np.ndarray] = None
        self.covariance: Optional[np.ndarray] = None

        self.is_activated = False
        self.state = TrackState.New
        self.track_id = None
        self.frame_id = 0
        self.tracklet_len = 0
        self.end_frame = 0

        self.score = score
        self.class_id = class_id
        self._trail = None  # list of recent (cx, cy); built lazily

    # -- id management ------------------------------------------------------
    @classmethod
    def next_id(cls) -> int:
        cls._next_id += 1
        return cls._next_id

    # -- coordinate helpers --------------------------------------------------
    @property
    def tlwh(self) -> np.ndarray:
        """``[x, y, w, h]``: detection box until activated, then Kalman state."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]  # width = aspect_ratio * height
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self) -> np.ndarray:
        """``[x1, y1, x2, y2]`` with the same fallback as :attr:`tlwh`."""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def trail(self) -> list:
        return self._trail or []

    def _push_center(self) -> None:
        if self.mean is None:
            return
        if self._trail is None:
            self._trail = []
        self._trail.append((self.mean[0], self.mean[1]))

    @classmethod
    def tlwh_to_xyah(cls, tlwh: Sequence[float]) -> np.ndarray:
        """``[x, y, w, h]`` -> ``[cx, cy, aspect_ratio, height]``."""
        ret = np.asarray(tlwh, dtype=np.float64).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= max(ret[3], 1e-6)
        return ret

    @staticmethod
    def tlbr_to_tlwh(tlbr: Sequence[float]) -> np.ndarray:
        ret = np.asarray(tlbr, dtype=np.float64).copy()
        ret[2:] -= ret[:2]
        return ret

    def is_high_conf(self) -> bool:
        """A track is high-confidence once it has lived >= 2 frames."""
        return self.tracklet_len >= 2

    # -- lifecycle -----------------------------------------------------------
    def activate(self, kalman_filter: "KalmanFilter", frame_id: int) -> None:
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(
            self.tlwh_to_xyah(self._tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.is_activated = True

    def re_activate(self, new_track: "STrack", frame_id: int,
                    new_id: bool = False) -> None:
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track._tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.score = new_track.score
        self.class_id = new_track.class_id
        self._trail = None
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track: "STrack", frame_id: int) -> None:
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track._tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        self.class_id = new_track.class_id
        self._push_center()

    def predict(self) -> None:
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0  # freeze height-velocity when lost
        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance
        )

    @classmethod
    def multi_predict(cls, stracks: Sequence["STrack"]) -> None:
        if not stracks:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = cls._shared_kalman.multi_predict(
            multi_mean, multi_covariance
        )
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    def mark_lost(self) -> None:
        self.state = TrackState.Lost
        self.end_frame = self.frame_id

    def mark_removed(self) -> None:
        self.state = TrackState.Removed

    # -- report ---------------------------------------------------------------
    def output(self) -> Track:
        x1, y1, x2, y2 = self.tlbr
        return Track(
            tlbr=(float(x1), float(y1), float(x2), float(y2)),
            track_id=self.track_id,
            score=float(self.score),
            class_id=int(self.class_id),
            frame_id=self.frame_id,
            trail=list(self.trail),
        )


STrack._shared_kalman = KalmanFilter()


# ---------------------------------------------------------------------------
# Association helpers
# ---------------------------------------------------------------------------
def _iou_batch(tlbr_a: np.ndarray, tlbr_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two (N,4) and (M,4) arrays of tlbr boxes."""
    if len(tlbr_a) == 0 or len(tlbr_b) == 0:
        return np.zeros((len(tlbr_a), len(tlbr_b)), dtype=np.float64)
    area_a = (tlbr_a[:, 2] - tlbr_a[:, 0]) * (tlbr_a[:, 3] - tlbr_a[:, 1])
    area_b = (tlbr_b[:, 2] - tlbr_b[:, 0]) * (tlbr_b[:, 3] - tlbr_b[:, 1])

    x1 = np.maximum(tlbr_a[:, None, 0], tlbr_b[None, :, 0])
    y1 = np.maximum(tlbr_a[:, None, 1], tlbr_b[None, :, 1])
    x2 = np.minimum(tlbr_a[:, None, 2], tlbr_b[None, :, 2])
    y2 = np.minimum(tlbr_a[:, None, 3], tlbr_b[None, :, 3])

    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def iou_distance(tracks: Sequence[STrack],
                 detections: Sequence[STrack]) -> np.ndarray:
    """IoU distance matrix (1 - IoU) between tracks and detections."""
    track_boxes = np.asarray([t.tlbr for t in tracks], dtype=np.float64)
    det_boxes = np.asarray([d.tlbr for d in detections], dtype=np.float64)
    return 1.0 - _iou_batch(track_boxes, det_boxes)


def fuse_score(distances: np.ndarray,
               detections: Sequence[STrack]) -> np.ndarray:
    """Weight IoU similarity by detection score (ByteTrack eq. improvement)."""
    if len(detections) == 0:
        return distances
    iou_sim = 1.0 - distances
    scores = np.asarray([d.score for d in detections], dtype=np.float64)[None, :]
    fused = 1.0 - iou_sim * scores
    return fused


def linear_assignment(cost_matrix: np.ndarray,
                      thresh: float) -> tuple[list, list, list]:
    """Hungarian matching, then greedy fills for anything below ``thresh``."""
    if np.any(np.isnan(cost_matrix)) or np.any(cost_matrix > 1e5):
        cost_matrix = np.nan_to_num(cost_matrix, nan=1e5, posinf=1e5, neginf=1e5)

    matches, unmatched_a, unmatched_b = [], [], []

    rows, cols = cost_matrix.shape
    if min(rows, cols) == 0:
        return matches, list(range(rows)), list(range(cols))

    if _HAS_LAP:
        # lapjv: x[i] = col assigned to row i (-1 if unmatched)
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
        for i in range(rows):
            j = int(x[i])
            if j >= 0:
                matches.append((i, j))
        unmatched_a = [i for i in range(rows) if x[i] < 0]
        unmatched_b = [j for j in range(cols) if y[j] < 0]
        return matches, unmatched_a, unmatched_b

    if _HAS_SCIPY:
        # linear_sum_assignment returns len(min(rows, cols)) pairs, so
        # iterate the pairs directly (indexing col_idx[i] for all rows
        # goes out of bounds when rows > cols).
        row_idx, col_idx = linear_sum_assignment(cost_matrix, maximize=False)
        matched_i = set(row_idx.tolist())
        matched_j = set(col_idx.tolist())
        for i, j in zip(row_idx.tolist(), col_idx.tolist()):
            if cost_matrix[i, j] <= thresh:
                matches.append((i, j))
        unmatched_a = [i for i in range(rows) if i not in matched_i]
        unmatched_b = [j for j in range(cols) if j not in matched_j]
        return matches, unmatched_a, unmatched_b

    # greedy fallback
    cost = cost_matrix.copy()
    while cost.size:
        i, j = np.unravel_index(np.argmin(cost), cost.shape)
        if cost[i, j] > thresh:
            break
        matches.append((i, j))
        cost[i, :] = np.inf
        cost[:, j] = np.inf
    matched_i = {i for i, _ in matches}
    matched_j = {j for _, j in matches}
    unmatched_a = [i for i in range(rows) if i not in matched_i]
    unmatched_b = [j for j in range(cols) if j not in matched_j]
    return matches, unmatched_a, unmatched_b


def joint_stracks(tlista: List[STrack],
                  tlistb: List[STrack]) -> List[STrack]:
    result = list(tlista)
    seen = {id(s) for s in result}
    for s in tlistb:
        if id(s) not in seen:
            result.append(s)
    return result


def sub_stracks(tlista: List[STrack],
                tlistb: List[STrack]) -> List[STrack]:
    stracks = {id(s) for s in tlistb}
    return [s for s in tlista if id(s) not in stracks]


def remove_duplicate_stracks(
    tlista: List[STrack], tlistb: List[STrack]
) -> tuple[List[STrack], List[STrack]]:
    """Drop tracks that overlap ~fully; keep the longer-lived one."""
    all_tracks = []
    seen = set()
    for st in tlista + tlistb:
        if id(st) not in seen:
            all_tracks.append(st)
            seen.add(id(st))

    rem1, rem2 = list(tlista), list(tlistb)
    for i, sa in enumerate(tlista):
        for sb in tlistb:
            if sa.class_id != sb.class_id:
                continue
            iou = _iou_batch(sa.tlbr[None, :], sb.tlbr[None, :])[0, 0]
            if iou > 0.15:
                if sa.tracklet_len > sb.tracklet_len:
                    rem2.remove(sb)
                else:
                    rem1.remove(sa)
    return rem1, rem2


# ---------------------------------------------------------------------------
# The tracker itself
# ---------------------------------------------------------------------------
class BYTETracker:
    """Associate detections across frames; call ``update`` every frame."""

    def __init__(
        self,
        frame_rate: int = 30,
        track_thresh: float = 0.5,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
        min_box_area: int = 10,
        fuse_score: bool = True,
        low_score_thresh: float = 0.1,
    ) -> None:
        self.det_thresh = track_thresh
        self.match_thresh = match_thresh
        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size
        self.min_box_area = min_box_area
        self.fuse_score = fuse_score
        self.low_score_thresh = low_score_thresh

        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []
        self.frame_id = 0

    def update(self, dets: np.ndarray) -> List[Track]:
        """Feed one frame's detections ``[x1, y1, x2, y2, score, class_id]``.

        Returns a list of :class:`Track` (track_id, tlbr, score, class_id)
        sorted by track_id.
        """
        self.frame_id += 1
        activated: List[STrack] = []
        refind: List[STrack] = []
        lossed: List[STrack] = []
        removed: List[STrack] = []

        scores = dets[:, 4]
        bboxes = dets[:, :4]
        classes = dets[:, 5]

        # --- filter tiny boxes ------------------------------------------------
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        big = areas > self.min_box_area
        bboxes, scores, classes = bboxes[big], scores[big], classes[big]

        # --- split into high- and low-score detections ------------------------
        inds_high = scores >= self.det_thresh
        inds_second = (scores >= self.low_score_thresh) & (scores < self.det_thresh)

        detections_high = [
            STrack(self._tlbr_to_tlwh(b), s, c)
            for b, s, c in zip(bboxes[inds_high], scores[inds_high],
                               classes[inds_high])
        ]
        detections_low = [
            STrack(self._tlbr_to_tlwh(b), s, c)
            for b, s, c in zip(bboxes[inds_second], scores[inds_second],
                               classes[inds_second])
        ]

        unconfirmed = [t for t in self.tracked_stracks if not t.is_activated]
        tracked = [t for t in self.tracked_stracks if t.is_activated]

        # ---- Step 1: predict + associate high-score detections --------------
        strack_pool = joint_stracks(tracked, self.lost_stracks)
        STrack.multi_predict(strack_pool)

        dists = iou_distance(strack_pool, detections_high)
        if self.fuse_score:
            dists = fuse_score(dists, detections_high)
        matches, u_track, u_det = linear_assignment(dists, self.match_thresh)

        for itracked, idet in matches:
            track, det = strack_pool[itracked], detections_high[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(det, self.frame_id)
                refind.append(track)

        remaining_high = [detections_high[i] for i in u_det]

        # ---- Step 2: low-score detections revive the tiny boxes --------------
        r_tracked = [
            strack_pool[i] for i in u_track
            if strack_pool[i].state == TrackState.Tracked
        ]
        dists = iou_distance(r_tracked, detections_low)
        matches, u_track2, u_det2 = linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track, det = r_tracked[itracked], detections_low[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(det, self.frame_id)
                refind.append(track)
        for it in u_track2:
            track = r_tracked[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lossed.append(track)
        remaining_low = [detections_low[i] for i in u_det2]

        # ---- Step 3: unconfirmed tracks fight over remaining high dets --------
        dists = iou_distance(unconfirmed, remaining_high)
        if self.fuse_score:
            dists = fuse_score(dists, remaining_high)
        matches, u_unconfirmed, u_det3 = linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            track = unconfirmed[itracked]
            track.update(remaining_high[idet], self.frame_id)
            activated.append(track)
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed.append(track)

        # ---- Step 4: new tracks from leftover high-score detections ----------
        leftover = [remaining_high[i] for i in u_det3]
        for det in leftover:
            det.activate(STrack._shared_kalman, self.frame_id)
            activated.append(det)

        # ---- Step 5: expire tracks lost for too long --------------------------
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed.append(track)

        # ---- Bookkeeping -------------------------------------------------------
        self.tracked_stracks = [
            t for t in self.tracked_stracks if t.state == TrackState.Tracked
        ]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lossed)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )

        out = sorted(
            [t for t in self.tracked_stracks if t.is_activated],
            key=lambda t: t.track_id,
        )
        return [t.output() for t in out]

    @staticmethod
    def _tlbr_to_tlwh(tlbr: np.ndarray) -> np.ndarray:
        ret = np.asarray(tlbr, dtype=np.float64).copy()
        ret[2:] -= ret[:2]
        return ret