#!/usr/bin/env python3
"""CLI wrapper for the red-blob demo clip generator.

    python scripts/make_demo_video.py -o runtime/demo-blobs.mp4
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zero_shot_tracking.demo_video import main  # noqa: E402

if __name__ == "__main__":
    main()