#!/usr/bin/env bash
# Bootstrap script for the Zero-Shot Tracking example.
#
# Installs a virtualenv, clones the official GroundingDINO repo, installs the
# Python dependencies and downloads the Swin-T model weights + config.
#
# Usage:
#   ./setup.sh                # CPU build
#   CUDA=true ./setup.sh      # CUDA build (torch with cu121 wheels)
#
# Notes:
#   * A CUDA GPU is strongly recommended. Without it detection is very slow.
#   * Python 3.8-3.12 is supported by GroundingDINO (needs torch extension
#     compilation on first use). Python 3.13+ is not supported yet.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
GROUNDINGDINO_DIR="${ROOT_DIR}/third_party/GroundingDINO"
CHECKPOINT_DIR="${ROOT_DIR}/checkpoints"

# ---------------------------------------------------------------------------
# 1. Virtualenv
# ---------------------------------------------------------------------------
echo "==> Creating virtualenv at ${VENV_DIR}"
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------------------------
# 2. Torch (CPU or CUDA)
# ---------------------------------------------------------------------------
if [[ "${CUDA:-false}" == "true" ]]; then
    echo "==> Installing torch with CUDA 12.1 wheels"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
else
    echo "==> Installing torch (CPU)"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# ---------------------------------------------------------------------------
# 3. Base python deps
# ---------------------------------------------------------------------------
echo "==> Installing python dependencies"
pip install -r "${ROOT_DIR}/requirements.txt"

# ---------------------------------------------------------------------------
# 4. GroundingDINO (build from source)
# ---------------------------------------------------------------------------
if [[ ! -d "${GROUNDINGDINO_DIR}" ]]; then
    echo "==> Cloning GroundingDINO"
    mkdir -p "$(dirname "${GROUNDINGDINO_DIR}")"
    git clone https://github.com/IDEA-Research/GroundingDINO.git "${GROUNDINGDINO_DIR}"
fi

echo "==> Installing GroundingDINO"
pip install -e "${GROUNDINGDINO_DIR}"

# ---------------------------------------------------------------------------
# 5. Model weights + config
# ---------------------------------------------------------------------------
echo "==> Downloading GroundingDINO Swin-T weights and config"
python "${ROOT_DIR}/scripts/download_weights.py" --out-dir "${CHECKPOINT_DIR}"

echo ""
echo "Done. Run the example with:"
echo "  source ${VENV_DIR}/bin/activate"
echo "  python run_tracking.py --video path/to/video.mp4 --text 'person, car' -o output/demo.mp4"