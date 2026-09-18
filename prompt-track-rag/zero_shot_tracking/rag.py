"""Retrieval-Augmented-Generation (RAG) multi-object tracker.

Unlike the algebraic trackers (ByteTrack / SORT / IoU — Kalman + IoU +
Hungarian), the RAG tracker frames association as a retrieval-augmented
decision problem:

  R  *Retrieval*   — every detection crop is embedded (CLIP, or the histogram
                     fallback, see :mod:`embedder`). Each track keeps a small
                     memory bank of past appearances. A new detection
                     *retrieves* its top-k most similar track memories by
                     cosine similarity, fused with a positional prior (IoU with
                     each track's predicted box).

  A  *Augmented*   — the retrieved memories (appearance vectors + kinematics +
                     class) are the augmented context the decision head reads,
                     replacing a hand-crafted cost matrix of raw IoU.

  G  *Generation*  — an assignment head generates the association decision from
                     the augmented context. The default head is
                     :class:`ScoreGenerator` (deterministic, offline). An
                     optional :class:`LLMGenerator` passes the same context,
                     rendered as natural language, to an OpenAI-compatible chat
                     model (Ollama by default) — see ``ZST_RAG_GENERATOR``.

Track memories live in a :class:`MemoryStore` — the vector database of this
design (embed -> index -> retrieve -> decide -> update).

:class:`RAGTracker` exposes the same ``update(dets)`` interface as the other
trackers (plus optional ``frame``/``text_prompt`` so it can embed actual
detection crops), and returns the same :class:`Track` namedtuples to the
visualizer, so the rest of the pipeline is unchanged.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from collections import deque
from typing import List, Optional, Sequence

import numpy as np

from .embedder import TrackEmbedder
from .tracker import Track


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _iou(bbox_a: np.ndarray, bbox_b: np.ndarray) -> np.ndarray:
    """Vectorized pairwise IoU between (N,4) and (M,4) tlbr arrays."""
    bbox_a = np.asarray(bbox_a, dtype=np.float64)
    bbox_b = np.asarray(bbox_b, dtype=np.float64)
    if len(bbox_a) == 0 or len(bbox_b) == 0:
        return np.zeros((len(bbox_a), len(bbox_b)), dtype=np.float64)
    area_a = (bbox_a[:, 2] - bbox_a[:, 0]) * (bbox_a[:, 3] - bbox_a[:, 1])
    area_b = (bbox_b[:, 2] - bbox_b[:, 0]) * (bbox_b[:, 3] - bbox_b[:, 1])

    x1 = np.maximum(bbox_a[:, None, 0], bbox_b[None, :, 0])
    y1 = np.maximum(bbox_a[:, None, 1], bbox_b[None, :, 1])
    x2 = np.minimum(bbox_a[:, None, 2], bbox_b[None, :, 2])
    y2 = np.minimum(bbox_a[:, None, 3], bbox_b[None, :, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


# ---------------------------------------------------------------------------
# Track memory
# ---------------------------------------------------------------------------
class TrackMemory:
    """A single identity's memory bank: appearance slots + kinematics."""

    def __init__(
        self,
        track_id: int,
        box: np.ndarray,
        class_id: int,
        score: float,
        frame_id: int,
        embedding: Optional[np.ndarray],
        memory_slots: int,
    ) -> None:
        self.track_id = track_id
        self.class_id = int(class_id)
        self.memory_slots = int(max(1, memory_slots))
        self.embeddings: deque = deque(maxlen=self.memory_slots)
        self.centroid: Optional[np.ndarray] = None

        self.last_box = np.asarray(box, dtype=np.float64).copy()
        self.predicted_box = np.asarray(box, dtype=np.float64).copy()
        self.velocity = np.zeros(2, dtype=np.float64)  # (dx, dy) per frame, EMA
        self._last_center = self._center(box)

        self.age = 0
        self.hits = 1
        self.lost = 0
        self.score = float(score)
        self.last_frame_id = frame_id
        self.trail: List[tuple] = [tuple(self._last_center)]

        if embedding is not None:
            self.push(embedding)

    # -- geometry -----------------------------------------------------------
    @staticmethod
    def _center(box: Sequence[float]) -> np.ndarray:
        box = np.asarray(box, dtype=np.float64)
        return np.asarray(
            [(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0], dtype=np.float64
        )

    def predict(self) -> None:
        """Constant-velocity extrapolation of the box (no Kalman filter)."""
        self.age += 1
        dx, dy = self.velocity
        self.predicted_box = self.last_box + np.asarray([dx, dy, dx, dy])
        self.predicted_box[:2] = np.clip(self.predicted_box[:2], 0, None)
        w = max(self.last_box[2] - self.last_box[0], 1.0)
        h = max(self.last_box[3] - self.last_box[1], 1.0)
        self.predicted_box[2] = max(
            self.predicted_box[2], self.predicted_box[0] + w * 0.2
        )
        self.predicted_box[3] = max(
            self.predicted_box[3], self.predicted_box[1] + h * 0.2
        )

    def push(self, embedding: np.ndarray) -> None:
        """Store a new appearance; keep the centroid as the running mean."""
        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        self.embeddings.append(emb)
        if self.centroid is None:
            self.centroid = emb.copy()
        else:
            n = len(self.embeddings)
            self.centroid = (self.centroid * (n - 1) + emb) / max(n, 1)

    def update(
        self,
        box: np.ndarray,
        score: float,
        frame_id: int,
        embedding: Optional[np.ndarray] = None,
    ) -> None:
        box = np.asarray(box, dtype=np.float64).copy()
        center = self._center(box)
        alpha = 0.2  # EMA for the velocity estimate
        if self.hits > 1:
            new_vel = center - self._last_center
            self.velocity = alpha * new_vel + (1 - alpha) * self.velocity
        self._last_center = center

        self.last_frame_id = frame_id
        self.last_box = box
        self.predicted_box = box.copy()
        self.score = float(score)
        self.lost = 0
        self.hits += 1
        self.trail.append(tuple(center))
        self.trail = self.trail[-self.memory_slots:]
        if embedding is not None:
            self.push(embedding)

    def best_similarity(self, query: Optional[np.ndarray]) -> float:
        """Max cosine similarity against the stored memory slots."""
        if query is None or self.centroid is None:
            return 0.5
        slots = np.asarray(list(self.embeddings), dtype=np.float32)
        if slots.size == 0:
            return 0.5
        sims = np.asarray(query, dtype=np.float32) @ slots.T
        return float(sims.max()) if sims.size else 0.5

    def to_context(self) -> dict:
        """Serialize the memory for the generation head."""
        return {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "last_box": [float(v) for v in self.last_box],
            "predicted_box": [float(v) for v in self.predicted_box],
            "velocity": [float(v) for v in self.velocity],
            "age": self.age,
            "hits": self.hits,
            "lost": self.lost,
            "last_seen_frame": self.last_frame_id,
        }

    def output(self, frame_id: int) -> Track:
        return Track(
            tlbr=tuple(float(v) for v in self.last_box),
            track_id=self.track_id,
            score=self.score,
            class_id=self.class_id,
            frame_id=frame_id,
            trail=list(self.trail),
        )


class MemoryStore:
    """Vector database of track memories (embed -> index -> retrieve)."""

    def __init__(self, memory_slots: int = 16) -> None:
        self._tracks: List[TrackMemory] = []
        self._next_id = 0
        self.memory_slots = int(max(1, memory_slots))

    @property
    def tracks(self) -> List[TrackMemory]:
        return list(self._tracks)

    def active(self) -> List[TrackMemory]:
        return [t for t in self._tracks if t.lost == 0]

    def create(
        self,
        box: np.ndarray,
        class_id: int,
        score: float,
        frame_id: int,
        embedding: Optional[np.ndarray],
    ) -> TrackMemory:
        self._next_id += 1
        track = TrackMemory(
            self._next_id, box, int(class_id), score, frame_id, embedding,
            self.memory_slots,
        )
        self._tracks.append(track)
        return track

    def retrieve(
        self,
        query: Optional[np.ndarray],
        top_k: int = 8,
        exclude: Optional[set] = None,
        class_ids: Optional[set] = None,
    ) -> List[tuple]:
        """Return ``[(similarity, TrackMemory)]`` for the top-k memories."""
        scored = []
        for t in self._tracks:
            if exclude is not None and t.track_id in exclude:
                continue
            if class_ids is not None and t.class_id not in class_ids:
                continue
            scored.append((t.best_similarity(query), t))
        scored.sort(key=lambda pair: -pair[0])
        return scored[: max(0, int(top_k))]

    def retire(self, track_buffer: int) -> None:
        """Drop memories lost for longer than ``track_buffer`` frames."""
        self._tracks = [t for t in self._tracks if t.lost <= int(track_buffer)]


# ---------------------------------------------------------------------------
# Assignment heads ("G" in RAG)
# ---------------------------------------------------------------------------
class AssignResult:
    __slots__ = ("matches", "unmatched_dets")

    def __init__(self, matches: List[tuple], unmatched_dets: List[int]) -> None:
        self.matches = matches                # list[(det_idx, TrackMemory)]
        self.unmatched_dets = unmatched_dets  # list[int]


class ScoreGenerator:
    """Default head: generate the assignment from the retrieved scores.

    Retrieval already produced fused appearance + positional scores for every
    (detection, memory) pair; this head performs the one-to-one assignment with
    a score gate, always taking the globally strongest evidence first.
    """

    name = "score"

    def assign(
        self,
        combined: np.ndarray,
        dets: np.ndarray,
        tracks: List[TrackMemory],
        match_thresh: float,
    ) -> AssignResult:
        n, m = combined.shape
        if n == 0 or m == 0:
            return AssignResult([], list(range(n)))

        matches = []
        used_det: set = set()
        used_trk: set = set()
        for di, ti in zip(*np.unravel_index(
            np.argsort(-combined.ravel()), combined.shape
        )):
            di, ti = int(di), int(ti)
            if di in used_det or ti in used_trk:
                continue
            if combined[di, ti] < match_thresh:
                break
            matches.append((di, tracks[ti]))
            used_det.add(di)
            used_trk.add(ti)
        unmatched = [i for i in range(n) if i not in used_det]
        return AssignResult(matches, unmatched)


class LLMGenerator:
    """Retrieval-augmented *generation* via an OpenAI-compatible chat model.

    The retrieved memories are rendered as a natural-language context and the
    model generates the association decision for each detection. Slower than
    :class:`ScoreGenerator` by design — useful for experiments/debugging.

    Environment:
        ZST_LLM_BASE_URL  default ``http://127.0.0.1:11434/v1`` (Ollama)
        ZST_LLM_MODEL     default ``llama3.2``
        ZST_LLM_API_KEY   optional, for hosted endpoints
    """

    name = "llm"

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.base_url = (
            base_url or os.environ.get(
                "ZST_LLM_BASE_URL", "http://127.0.0.1:11434/v1"
            )
        ).rstrip("/")
        self.model = model or os.environ.get("ZST_LLM_MODEL", "llama3.2")
        self.api_key = api_key or os.environ.get("ZST_LLM_API_KEY", "")

    def assign(
        self,
        combined: np.ndarray,
        dets: np.ndarray,
        tracks: List[TrackMemory],
        embeddings: Optional[np.ndarray],
        match_thresh: float,
        top_k: int = 5,
    ) -> AssignResult:
        n, m = combined.shape
        if n == 0:
            return AssignResult([], [])

        matches = []
        used_trk: set = set()
        unmatched = []
        for det_i in range(n):
            order = np.argsort(-combined[det_i])[:top_k]
            cand_pairs = [
                (int(ti), tracks[int(ti)])
                for ti in order
                if combined[det_i, int(ti)] >= match_thresh
                and tracks[int(ti)].track_id not in used_trk
            ]
            if not cand_pairs:
                unmatched.append(det_i)
                continue

            query = (
                embeddings[det_i]
                if embeddings is not None and embeddings.shape[0] == len(dets)
                else None
            )
            candidate_scores = {
                t.track_id: t.best_similarity(query) for _, t in cand_pairs
            }
            prompt = self._build_prompt(dets[det_i], cand_pairs, candidate_scores)
            track = None
            try:
                chosen_id = self._chat(prompt)
                for _, t in cand_pairs:
                    if t.track_id == chosen_id:
                        track = t
                        break
            except Exception:
                track = None
            if track is None:  # best-effort fallback to the score head
                track = max(
                    cand_pairs, key=lambda pair: -combined[det_i, pair[0]]
                )[1]
            matches.append((det_i, track))
            used_trk.add(track.track_id)

        return AssignResult(matches, unmatched)

    @staticmethod
    def _build_prompt(det, candidates, candidate_scores: dict) -> str:
        x1, y1, x2, y2 = (float(v) for v in det[:4])
        lines = [
            "You are a multi-object tracking association module.",
            "",
            f"New detection #0: box=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}], "
            f"class_id={int(det[5])}, score={float(det[4]):.2f}.",
            "",
            "Retrieved candidate memories:",
        ]
        for _, t in candidates:
            cx = (float(t.last_box[0]) + float(t.last_box[2])) / 2
            cy = (float(t.last_box[1]) + float(t.last_box[3])) / 2
            lines.append(
                f"  track #{t.track_id}: class_id={t.class_id}, "
                f"appearance={candidate_scores.get(t.track_id, 0.0):.3f}, "
                f"last_center=({cx:.0f},{cy:.0f}), lost={t.lost}, age={t.age}"
            )
        lines.append(
            "Assign the detection to exactly one candidate and respond only "
            'with JSON like {"track_id": 3}, or {"track_id": null} for a new '
            "object."
        )
        return "\n".join(lines)

    def _chat(self, prompt: str) -> Optional[int]:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "stream": False,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        text = data["choices"][0]["message"]["content"]
        m = re.search(r'"track_id"\s*:\s*(\d+|null)', text)
        if not m:
            return None
        return None if m.group(1) == "null" else int(m.group(1))


def create_assignment_generator(name: str = "auto") -> object:
    """Select the generation head; ``auto`` = LLM if configured, else score."""
    name = (name or "auto").lower().strip()
    if name in ("llm", "gpt", "rag-llm"):
        return LLMGenerator()
    if name in ("score", "score-based", "builtin", "default"):
        return ScoreGenerator()
    if name == "auto":
        if os.environ.get("ZST_LLM_BASE_URL") or os.environ.get("ZST_LLM_API_KEY"):
            return LLMGenerator()
        return ScoreGenerator()
    raise ValueError(f"unknown assignment generator {name!r}; use score|llm|auto")


# ---------------------------------------------------------------------------
# The RAG tracker
# ---------------------------------------------------------------------------
class RAGTracker:
    """Retrieval-augmented tracker: embed -> retrieve -> generate assignment.

    The pipeline calls ``update(dets)`` like any other tracker; it also accepts
    the source ``frame`` and ``text_prompt`` so detection crops can be embedded
    (needed for the retrieval step). Without a frame the tracker degrades to
    positional-only retrieval and keeps running.
    """

    needs_frame = True

    def __init__(
        self,
        frame_rate: int = 30,
        track_thresh: float = 0.5,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
        min_box_area: int = 10,
        embedder: Optional[TrackEmbedder] = None,
        generator: object = None,
        w_appearance: float = 0.65,
        w_position: float = 0.35,
        memory_slots: int = 16,
        top_k: int = 8,
        min_hits: int = 1,
        **ignored,
    ) -> None:
        self.frame_rate = int(frame_rate)
        self.det_thresh = float(track_thresh)
        self.match_thresh = float(match_thresh)
        self.track_buffer = int(track_buffer)
        self.min_box_area = int(min_box_area)
        self.min_hits = int(min_hits)
        self.w_appearance = float(min(max(w_appearance, 0.0), 1.0))
        self.w_position = float(min(max(w_position, 0.0), 1.0))
        if self.w_appearance + self.w_position <= 0:
            self.w_appearance, self.w_position = 1.0, 0.0

        if embedder is None:
            from .embedder import create_embedder

            embedder = create_embedder(
                os.environ.get("ZST_EMBEDDER", "auto")
            )
        self.embedder = embedder
        self.generator = generator or create_assignment_generator(
            os.environ.get("ZST_RAG_GENERATOR", "score")
        )
        self.store = MemoryStore(memory_slots=memory_slots)
        self.top_k = int(max(1, top_k))
        self.frame_id = 0

    # -- helpers ------------------------------------------------------------
    def _filter(self, dets: np.ndarray) -> np.ndarray:
        if len(dets) == 0:
            return dets
        areas = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
        keep = (areas > self.min_box_area) & (dets[:, 4] >= self.det_thresh)
        return dets[keep]

    def _predict(self, tracks: List[TrackMemory]) -> None:
        for t in tracks:
            t.predict()

    # -- main loop ----------------------------------------------------------
    def update(
        self,
        dets: np.ndarray,
        frame: Optional[np.ndarray] = None,
        text_prompt: Optional[str] = None,
    ) -> List[Track]:
        """Embed, retrieve, generate assignments, and update memories."""
        self.frame_id += 1

        dets = self._filter(np.asarray(dets, dtype=np.float64))
        tracks_all = self.store.tracks

        if len(dets) == 0:
            for t in tracks_all:
                t.predict()
                t.lost += 1
            self.store.retire(self.track_buffer)
            return []

        boxes = dets[:, :4]

        # R: embed detection crops (retrieval queries)
        embeddings: Optional[np.ndarray] = None
        if frame is not None:
            try:
                embeddings = self.embedder.embed_crops(frame, boxes)
            except Exception:
                embeddings = None  # keep positional-only association

        self._predict(tracks_all)
        track_list = tracks_all
        m = len(track_list)

        # fused retrieval scores: appearance (cosine) + positional (IoU)
        if m:
            appear = np.full((len(boxes), m), 0.5, dtype=np.float64)
            position = _iou(boxes, np.asarray([t.predicted_box for t in track_list]))
            if embeddings is not None:
                for t_i, t in enumerate(track_list):
                    slots = np.asarray(list(t.embeddings), dtype=np.float32)
                    if slots.size == 0:
                        continue
                    appear[:, t_i] = np.max(
                        np.asarray(embeddings, dtype=np.float32) @ slots.T,
                        axis=1,
                    ).clip(0, 1)

            combined = self.w_position * position + self.w_appearance * appear
            if embeddings is None:
                # no frame/embedder available: degrade to position-only — IoU
                # becomes the full score so the match gate stays comparable.
                combined = position

            # G: generate the assignment from the retrieved context
            if getattr(self.generator, "name", "") == "llm":
                result = self.generator.assign(
                    combined, dets, track_list, embeddings, self.match_thresh,
                    top_k=self.top_k,
                )
            else:
                result = self.generator.assign(
                    combined, dets, track_list, self.match_thresh
                )
        else:
            # no memories yet: every detection starts a new track memory
            result = AssignResult([], list(range(len(boxes))))

        # commit assignments to memory
        for det_i, track in result.matches:
            emb = embeddings[det_i] if embeddings is not None else None
            track.update(boxes[det_i], dets[det_i, 4], self.frame_id, emb)

        # unmatched detections become new track memories
        for det_i in result.unmatched_dets:
            emb = embeddings[det_i] if embeddings is not None else None
            self.store.create(
                boxes[det_i], int(dets[det_i, 5]), dets[det_i, 4],
                self.frame_id, emb,
            )

        # pre-existing tracks not matched this frame age by one lost frame
        matched_ids = {t.track_id for _, t in result.matches}
        for t in track_list:
            if t.track_id not in matched_ids:
                t.lost += 1
        self.store.retire(self.track_buffer)

        out = [
            t.output(self.frame_id)
            for t in self.store.active()
            if t.hits >= self.min_hits
        ]
        out.sort(key=lambda t: t.track_id)
        return out