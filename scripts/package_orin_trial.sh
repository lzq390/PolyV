#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v git >/dev/null 2>&1; then
  echo "git is required to build the trial package." >&2
  exit 1
fi

if ! command -v tar >/dev/null 2>&1; then
  echo "tar is required to build the trial package." >&2
  exit 1
fi

COMMIT="$(git rev-parse --short HEAD)"
PACKAGE_NAME="polyv-orin-trial-${COMMIT}"
DIST_DIR="$ROOT/dist"
ARCHIVE_NAME="${PACKAGE_NAME}.tar.gz"
ARCHIVE_PATH="$DIST_DIR/$ARCHIVE_NAME"
CHECKSUM_PATH="$DIST_DIR/${PACKAGE_NAME}.sha256"
STAGE_ROOT="$(mktemp -d)"
STAGE_DIR="$STAGE_ROOT/$PACKAGE_NAME"

cleanup() {
  rm -rf "$STAGE_ROOT"
}
trap cleanup EXIT

mkdir -p "$DIST_DIR" "$STAGE_DIR"

copy_path() {
  local rel_path="$1"
  if [ ! -e "$ROOT/$rel_path" ]; then
    echo "Missing required package path: $rel_path" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$STAGE_DIR/$rel_path")"
  cp -a "$ROOT/$rel_path" "$STAGE_DIR/$rel_path"
}

copy_path "polyv_detector"
copy_path "tests"
copy_path "tools"
copy_path "README.md"
copy_path "environment.yml"
copy_path "scripts/install_jetson.sh"
copy_path "scripts/run_video.sh"
copy_path "docs/orin_nx_trial_run.md"

find "$STAGE_DIR" -type d -name "__pycache__" -prune -exec rm -rf {} +
find "$STAGE_DIR" -type f -name "*.pyc" -delete

export ROOT STAGE_DIR PACKAGE_NAME COMMIT
python3 - <<'PY'
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

root = Path(os.environ["ROOT"])
stage = Path(os.environ["STAGE_DIR"])
status = subprocess.run(
    ["git", "status", "--short"],
    cwd=root,
    check=True,
    capture_output=True,
    text=True,
).stdout.splitlines()
files = sorted(
    str(path.relative_to(stage))
    for path in stage.rglob("*")
    if path.is_file()
)
files = sorted({*files, "PACKAGING_MANIFEST.json"})
manifest = {
    "package": os.environ["PACKAGE_NAME"],
    "commit": os.environ["COMMIT"],
    "packaged_at": datetime.now(timezone.utc).isoformat(),
    "dirty": bool(status),
    "git_status_short": status,
    "excluded": [".git/", "dist/", "outputs/", "__pycache__/", ".venv/", "local video files"],
    "files": files,
}
(stage / "PACKAGING_MANIFEST.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY

rm -f "$ARCHIVE_PATH" "$CHECKSUM_PATH" "$DIST_DIR/${ARCHIVE_NAME}.sha256"
tar -czf "$ARCHIVE_PATH" -C "$STAGE_ROOT" "$PACKAGE_NAME"

if command -v sha256sum >/dev/null 2>&1; then
  (cd "$DIST_DIR" && sha256sum "$ARCHIVE_NAME" > "$(basename "$CHECKSUM_PATH")")
elif command -v shasum >/dev/null 2>&1; then
  (cd "$DIST_DIR" && shasum -a 256 "$ARCHIVE_NAME" > "$(basename "$CHECKSUM_PATH")")
else
  echo "Neither sha256sum nor shasum is available; checksum was not written." >&2
  exit 1
fi

echo "Wrote $ARCHIVE_PATH"
echo "Wrote $CHECKSUM_PATH"
