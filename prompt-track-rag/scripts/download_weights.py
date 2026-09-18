#!/usr/bin/env python3
"""Download the GroundingDINO Swin-T config + weights from the official repo.

URLs are pinned to the v0.1.0-alpha release of IDEA-Research/GroundingDINO.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request

MODEL = "groundingdino_swint_ogc"
BASE = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha"
RAW = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/%s.cfg.py"

WIDTH = 80


def fetch(url: str, dest: str) -> None:
    sys.stdout.write(f"  downloading {url}\n  -> {dest}\n")
    urllib.request.urlretrieve(url, dest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="checkpoints")
    args = ap.parse_args()

    config_url = RAW % MODEL
    weights_url = f"{BASE}/{MODEL}.pth"

    print("Zero-shot tracking checkpoint download")
    print("-" * WIDTH)
    fetch(config_url, f"{args.out_dir}/{MODEL}.cfg.py")
    fetch(weights_url, f"{args.out_dir}/{MODEL}.pth")
    print("-" * WIDTH)
    print("done. Run: python run_tracking.py --video <video.mp4> --text 'person'")


if __name__ == "__main__":
    main()