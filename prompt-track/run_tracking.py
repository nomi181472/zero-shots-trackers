#!/usr/bin/env python3
"""CLI for the zero-shot tracking example.

Pipeline: GroundingDINO detects the prompt's classes -> ByteTrack tracks them.

Examples:
    python run_tracking.py --video clips/street.mp4 --text "person, car" \\
        -o output/demo.mp4

    python run_tracking.py --video clips/street.mp4 --text person \\
        --box-threshold 0.3 --text-threshold 0.2 --detect-interval 2 \\
        --device cuda --show
"""

from __future__ import annotations

import argparse

import yaml

from zero_shot_tracking.pipeline import PipelineConfig, ZeroShotTrackerPipeline


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, help="input video path")
    p.add_argument("--text", required=True,
                   help='free-text prompt, e.g. "person, car, traffic light"')
    p.add_argument("-o", "--output", default=None, help="output video path")

    p.add_argument("-c", "--config", default="configs/config.yaml",
                   help="YAML config (default: %(default)s)")
    p.add_argument("--device", default=None, choices=["cuda", "cpu"])

    # detection
    p.add_argument("--box-threshold", type=float, default=None)
    p.add_argument("--text-threshold", type=float, default=None)
    p.add_argument("--detect-interval", type=int, default=None)

    # tracking
    p.add_argument("--track-thresh", type=float, default=None)
    p.add_argument("--match-thresh", type=float, default=None)
    p.add_argument("--track-buffer", type=int, default=None)

    # io / misc
    p.add_argument("--show", action="store_true")
    p.add_argument("--max-frames", type=int, default=None)
    return p


def main() -> None:
    args = build_cli().parse_args()
    yml = load_config(args.config)

    model_cfg = yml.get("model", {})
    detect_cfg = yml.get("detector", {})
    track_cfg = yml.get("tracker", {})
    io_cfg = yml.get("io", {})

    cfg = PipelineConfig(
        config_path=model_cfg.get("config_path", "checkpoints/GroundingDINO_SwinT_OGC.cfg.py"),
        weights_path=model_cfg.get("weights_path", "checkpoints/groundingdino_swint_ogc.pth"),
        device=args.device or model_cfg.get("device", "cuda"),
        box_threshold=args.box_threshold if args.box_threshold is not None
        else detect_cfg.get("box_threshold", 0.35),
        text_threshold=args.text_threshold if args.text_threshold is not None
        else detect_cfg.get("text_threshold", 0.25),
        detect_interval=args.detect_interval if args.detect_interval is not None
        else detect_cfg.get("detect_interval", 1),
        frame_rate=track_cfg.get("frame_rate", 30),
        track_thresh=args.track_thresh if args.track_thresh is not None
        else track_cfg.get("track_thresh", 0.5),
        match_thresh=args.match_thresh if args.match_thresh is not None
        else track_cfg.get("match_thresh", 0.8),
        track_buffer=args.track_buffer if args.track_buffer is not None
        else track_cfg.get("track_buffer", 30),
        min_box_area=track_cfg.get("min_box_area", 10),
        fuse_score=track_cfg.get("fuse_score", True),
        output=args.output or io_cfg.get("output", "output/tracked.mp4"),
        show=args.show or io_cfg.get("show", False),
        max_frames=args.max_frames if args.max_frames is not None
        else io_cfg.get("max_frames", -1),
        class_map=io_cfg.get("class_map", {}),
    )

    pipeline = ZeroShotTrackerPipeline(cfg)
    pipeline.track_video(args.video, args.text)


if __name__ == "__main__":
    main()