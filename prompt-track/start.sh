#!/usr/bin/env bash
# Start both backend (FastAPI) and frontend (Next.js) for prompt-track.
# Defaults use the ByteTrack algorithmic tracker (no neural re-identification).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- backend ---------------------------------------------------------------
echo "==> Starting backend (FastAPI :8000)"
cd "$ROOT"
ZST_DETECTOR=mock python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
sleep 3

# ---- frontend --------------------------------------------------------------
echo "==> Starting frontend (Next.js :3000)"
cd "$ROOT/frontend"
if [ ! -d node_modules ]; then
  echo "    installing node_modules ..."
  npm install --audit=false --fund=false 2>&1 | tail -1
fi
BACKEND_ORIGIN=http://127.0.0.1:8000 npm run dev -- -p 3000 &
FRONTEND_PID=$!

echo "==> Both services running"
echo "  Backend PID : $BACKEND_PID  (http://127.0.0.1:8000)"
echo "  Frontend PID: $FRONTEND_PID (http://127.0.0.1:3000)"
echo "  Open http://localhost:3000, drop a video, type a prompt, press start tracking."

# keep the script alive so the background processes stay alive
wait $BACKEND_PID $FRONTEND_PID
trap 'kill $BACKEND_PID $FRONTEND_PID 2>/dev/null' EXIT