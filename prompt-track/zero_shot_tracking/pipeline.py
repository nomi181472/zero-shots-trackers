"""End-to-end zero-shot tracking pipeline: detect -> track -> annotate."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .detector import GroundingDINODetector
from .tracker import BYTETracker
from .visualizer import draw_frame_counter, draw_legend, draw_tracks


@dataclass
class PipelineConfig:
    config_path: str
    weights_path: str
    device: str = "cuda"
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    detect_interval: int = 1
    frame_rate: int = 30
    track_thresh: float = 0.5
    match_thresh: float = 0.8
    track_buffer: int = 30
    min_box_area: int = 10
    fuse_score: bool = True
    output: str = "output/tracked.mp4"
    show: bool = False
    max_frames: int = -1
    class_map: dict[str, str] = field(default_factory=dict)


class ZeroShotTrackerPipeline:
    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self.detector = GroundingDINODetector(
            config_path=cfg.config_path,
            weights_path=cfg.weights_path,
            device=cfg.device,
            box_threshold=cfg.box_threshold,
            text_threshold=cfg.text_threshold,
        )
        self.tracker = BYTETracker(
            frame_rate=cfg.frame_rate,
            track_thresh=cfg.track_thresh,
            match_thresh=cfg.match_thresh,
            track_buffer=cfg.track_buffer,
            min_box_area=cfg.min_box_area,
            fuse_score=cfg.fuse_score,
        )

    def track_video(
        self,
        video_path: str,
        text_prompt: str,
        output_path: str | None = None,
    ) -> int:
        """Track a video and write the annotated result.

        Returns the number of frames processed.
        """
        cfg = self.cfg
        output_path = output_path or cfg.output
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or cfg.frame_rate
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if cfg.max_frames > 0:
            total = min(total, cfg.max_frames)

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"cannot write video: {output_path}")

        frame_id = 0
        labels: list[str] = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if cfg.max_frames > 0 and frame_id >= cfg.max_frames:
                    break

                # ---- detect: every detect_interval-th frame ------------------
                if frame_id % cfg.detect_interval == 0:
                    dets, labels, _ = self.detector.detect(
                        frame, text_prompt, cfg.box_threshold, cfg.text_threshold
                    )
                    dets = self.detector.nms(dets)
                else:
                    dets = np.empty((0, 6), dtype=np.float32)

                # ---- track ----------------------------------------------------
                tracks = self.tracker.update(dets)

                # ---- annotate --------------------------------------------------
                annotated = draw_tracks(
                    frame, tracks, labels, cfg.class_map
                )
                annotated = draw_legend(annotated, labels, text_prompt)
                annotated = draw_frame_counter(
                    annotated, frame_id + 1, None if total <= 0 else total
                )

                writer.write(annotated)
                if cfg.show:
                    cv2.imshow("zero-shot tracking", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_id += 1
                if frame_id % 10 == 0:
                    print(
                        f"frame {frame_id}/{total if total > 0 else '?'} "
                        f"tracks={len(tracks)}"
                    )
        finally:
            cap.release()
            writer.release()
            if cfg.show:
                cv2.destroyAllWindows()

        print(f"saved annotated video -> {output_path}")
        return frame_id