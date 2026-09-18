#!/usr/bin/env bash
# Start both backend (FastAPI) and frontend (Next.js) for prompt-track-ostrack.
# Detector-free visual tracking (OSTrack / STARK / template-matching).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- backend ---------------------------------------------------------------
TRACKER="${ZST_TRACKER:-ostrack}"
PORT_BACKEND="${PORT_BACKEND:-8000}"
PORT_FRONTEND="${PORT_FRONTEND:-3000}"

echo "==> Starting backend (FastAPI :${PORT_BACKEND}) [tracker=${TRACKER}]"
cd "$ROOT"
ZST_TRACKER="$TRACKER" python3 -m uvicorn backend.main:app --host 0.0.0.0 --port "${PORT_BACKEND}" &
BACKEND_PID=$!
sleep 2

# ---- frontend --------------------------------------------------------------
echo "==> Starting frontend (Next.js :${PORT_FRONTEND})"
cd "$ROOT/frontend"

# Link or install dependencies if missing
if [ ! -d "node_modules" ]; then
  if [ -d "$ROOT/../prompt-track/frontend/node_modules" ]; then
    echo "    linking node_modules from sibling project ..."
    ln -s "$ROOT/../prompt-track/frontend/node_modules" ./node_modules
  else
    echo "    installing node_modules ..."
    npm install --audit=false --fund=false 2>&1 | tail -1
  fi
fi

BACKEND_ORIGIN="http://127.0.0.1:${PORT_BACKEND}" npm run dev -- -p "${PORT_FRONTEND}" &
FRONTEND_PID=$!

echo ""
echo "========================================================================="
echo "==> Detector-Free Visual Tracking running:"
echo "    Backend API  : http://localhost:${PORT_BACKEND} (PID: $BACKEND_PID)"
echo "    Web UI       : http://localhost:${PORT_FRONTEND} (PID: $FRONTEND_PID)"
echo ""
echo "    1. Open http://localhost:${PORT_FRONTEND}"
echo "    2. Click 'Load Demo Video' (or upload an .mp4)"
echo "    3. Click and drag a bounding box on Frame 1 over the target object"
echo "    4. Press 'Start Tracking' — follows appearance via OSTrack / STARK"
echo "========================================================================="
echo ""

# keep alive until user presses Ctrl-C
trap 'echo "Stopping services..."; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null' EXIT INT TERM
wait $BACKEND_PID $FRONTEND_PID
