#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

usage() {
  cat <<'USAGE'
Usage:
  ./scripts/run_video.sh <video-path> [--sample-fps 1] [--output-dir /data/polyv/outputs] [--max-seconds N] [--overlay]

Examples:
  ./scripts/run_video.sh /data/polyv/videos/sample.mp4
  ./scripts/run_video.sh /data/polyv/videos/sample.mp4 --overlay
USAGE
}

VIDEO_PATH=""
SAMPLE_FPS="1"
OUTPUT_DIR="${POLYV_OUTPUT_DIR:-/data/polyv/outputs}"
MAX_SECONDS=""
WRITE_OVERLAY=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --sample-fps)
      SAMPLE_FPS="${2:?--sample-fps requires a value}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:?--output-dir requires a value}"
      shift 2
      ;;
    --max-seconds)
      MAX_SECONDS="${2:?--max-seconds requires a value}"
      shift 2
      ;;
    --overlay)
      WRITE_OVERLAY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      if [ -n "$VIDEO_PATH" ]; then
        echo "Only one video path is supported." >&2
        usage >&2
        exit 1
      fi
      VIDEO_PATH="$1"
      shift
      ;;
  esac
done

if [ -z "$VIDEO_PATH" ]; then
  echo "Missing video path." >&2
  usage >&2
  exit 1
fi

if [ ! -f "$VIDEO_PATH" ]; then
  echo "Video does not exist: $VIDEO_PATH" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

if [ -f "$ROOT/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

STEM="$(basename "$VIDEO_PATH")"
STEM="${STEM%.*}"
JSON_OUT="$OUTPUT_DIR/${STEM}.result.json"
TIME_LOG="$OUTPUT_DIR/${STEM}.time.log"

CMD=(
  python -m polyv_detector.cli
  --video "$VIDEO_PATH"
  --sample-fps "$SAMPLE_FPS"
  --json-out "$JSON_OUT"
)

if [ -n "$MAX_SECONDS" ]; then
  CMD+=(--max-seconds "$MAX_SECONDS")
fi

if [ "$WRITE_OVERLAY" -eq 1 ]; then
  OVERLAY_DIR="$OUTPUT_DIR/${STEM}_overlay"
  mkdir -p "$OVERLAY_DIR"
  CMD+=(--debug-overlay-dir "$OVERLAY_DIR")
fi

if [ -x /usr/bin/time ]; then
  /usr/bin/time -v -o "$TIME_LOG" "${CMD[@]}"
else
  "${CMD[@]}"
fi

echo "JSON result: $JSON_OUT"
if [ -f "$TIME_LOG" ]; then
  echo "Timing log: $TIME_LOG"
fi
if [ "$WRITE_OVERLAY" -eq 1 ]; then
  echo "Overlay dir: $OVERLAY_DIR"
fi
