#!/usr/bin/env bash
# Start both backend (FastAPI) and frontend (Next.js) for prompt-track-rag.
# Defaults use the mock detector + histogram embedder so no heavyweights are needed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- backend ---------------------------------------------------------------
DETECTOR="${ZST_DETECTOR:-mock}"
EMBEDDER="${ZST_EMBEDDER:-histogram}"
DEVICE="${ZST_DEVICE:-cpu}"
echo "==> Starting backend (FastAPI :8000) [detector=$DETECTOR, embedder=$EMBEDDER, device=$DEVICE]"
cd "$ROOT"
ZST_DETECTOR="$DETECTOR" ZST_EMBEDDER="$EMBEDDER" ZST_DEVICE="$DEVICE" python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
sleep 3 # give the health endpoint a moment to appear

# ---- frontend --------------------------------------------------------------
echo "==> Starting frontend (Next.js :3000)"
cd "$ROOT/frontend"
#install deps if node_modules missing
if [ ! -d node_modules ]; then
  echo "    installing node_modules ..."
  npm install --audit=false --fund=false 2>&1 | tail -1
fi
# run dev, proxying /api/* to the backend we just started
BACKEND_ORIGIN=http://127.0.0.1:8000 npm run dev -- -p 3000 &
FRONTEND_PID=$!

echo "==> Both services running"
echo "  Backend PID : $BACKEND_PID  (http://127.0.0.1:8000)"
echo "  Frontend PID: $FRONTEND_PID (http://127.0.0.1:3000)"
echo "  Open http://localhost:3000, drop a video, type a prompt, press start tracking."

# keep the script alive so the background processes stay alive
wait $BACKEND_PID $FRONTEND_PID
trap 'kill $BACKEND_PID $FRONTEND_PID 2>/dev/null' EXIT