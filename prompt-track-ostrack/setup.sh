#!/usr/bin/env bash
set -euo pipefail

# zero-shots-trackers — prompt-track-ostrack (detector-free visual tracking)
# ---------------------------------------------------------------
# This project uses a detector-free single-object tracker (OSTrack / STARK /
# mock template-matching). No GroundingDINO weights are needed.
#
# Usage (full web app):
#
#   1. Create and activate a venv:
#      python3 -m venv .venv
#      source .venv/bin/activate
#
#   2. Install deps:
#      pip install --upgrade pip
#      pip install --break-system-packages numpy opencv-python fastapi uvicorn
#      pip install --break-system-packages httpx  # for tests
#
#   3. (Optional) Install OSTrack / STARK if desired:
#      # ostrack is not on PyPI; install from source or conda as needed
#      # pip install ostrack   # when available
#      # pip install stark-tracking   # when available
#
#   4. Start the backend:
#      python3 -m uvicorn backend.main:app --port 8000
#
#   5. Start the frontend:
#      cd frontend
#      npm install && npm run dev   # http://localhost:3000
#
#   6. Open http://localhost:3000, upload a video, draw a bounding box
#      on frame 1 (or upload an object crop), press "start tracking".
#      The OSTrack/mock tracker will follow the object appearance.

# --- venv bootstrap (optional, skip if venv already exists) ---
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate

# --- core Python deps ---
pip install --break-system-packages \
    numpy \
    opencv-python \
    fastapi \
    uvicorn[standard] \
    httpx

# --- project-agnostic helpers (shared with prompt-track / prompt-track-rag) ---
pip install --break-system-packages python-multipart

echo "Setup complete. Activate with: source .venv/bin/activate"