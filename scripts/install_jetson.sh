#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ "${ALLOW_NON_JETSON:-0}" != "1" ] && [ ! -f /etc/nv_tegra_release ]; then
  echo "This installer is intended for Jetson Linux." >&2
  echo "Set ALLOW_NON_JETSON=1 only for dry-run testing on a non-Jetson Linux host." >&2
  exit 2
fi

if ! command -v apt-get >/dev/null 2>&1; then
  echo "apt-get is required. Run this script on Ubuntu-based Jetson Linux." >&2
  exit 1
fi

sudo apt-get update
sudo apt-get install -y \
  python3-venv \
  python3-opencv \
  python3-numpy \
  python3-pil

python3 -m venv --system-site-packages .venv

# shellcheck disable=SC1091
source .venv/bin/activate

python - <<'PY'
import cv2
import numpy
from PIL import Image

print("cv2", cv2.__version__)
print("numpy", numpy.__version__)
print("PIL", Image.__version__)
PY

python -m unittest discover -s tests -p 'test*.py' -v

echo "PolyV Jetson trial environment is ready: $ROOT/.venv"
