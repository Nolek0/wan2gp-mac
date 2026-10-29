#!/bin/bash
# Mac port of WanGP - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt). Added 2026-10-29.
# One-time setup on Apple Silicon: Python 3.11 environment (~3 GB) and the Wan 2.1 1.3B text-to-video files (~10 GB).
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
    echo "This setup is for Macs with Apple Silicon (M1 or later)." >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
    echo "[1/4] Installing uv (Python package manager)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"

echo "[2/4] Creating the Python 3.11 environment..."
[ -d .venv ] || "$UV" venv --python 3.11 .venv

echo "[3/4] Installing PyTorch and WanGP's dependencies (a few minutes)..."
VIRTUAL_ENV=.venv "$UV" pip install torch==2.15.0 torchvision==0.29.2 torchaudio==2.11.1  # versions tested on Apple Silicon
VIRTUAL_ENV=.venv "$UV" pip install -r mac/requirements-mac.txt

if [ "${1:-}" != "--no-models" ]; then
    echo "[4/4] Downloading the model files (~10 GB)..."
    .venv/bin/python mac/download_models.py
fi
echo "Done. Try: .venv/bin/python mac/generate.py \"A red fox walking through fresh snow at sunrise\" --offline"
