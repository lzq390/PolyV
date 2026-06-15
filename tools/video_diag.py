from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from polyv_detector.cli import run_video
from polyv_detector.detector import DetectionConfig, detect_frame


PATHS = [
    "/mnt/c/Users/ASUS/Downloads/VID_20260601_142527.mp4",
    "/mnt/c/Users/ASUS/Downloads/VID_20260601_144516.mp4",
    "/mnt/c/Users/ASUS/Downloads/VID_20260601_150311.mp4",
    "/mnt/c/Users/ASUS/Downloads/VID_20260603_151103 - Trim.mp4",
]

PATHS += sorted(
    glob.glob("/mnt/c/Users/ASUS/Downloads/*20260604-111405.mp4")
    + glob.glob("/mnt/c/Users/ASUS/Downloads/*20260604-111732.mp4")
    + glob.glob("/mnt/c/Users/ASUS/Downloads/*20260604-111744.mp4")
)

SAMPLE_TIMES = [0, 30, 60, 120, 300, 540]
AUTO_CONFIG = DetectionConfig(roi_mode="auto")
FIXED_CONFIG = DetectionConfig(roi_mode="fixed")


def sample_metrics(path: str, duration_sec: float) -> list[dict]:
    cap = cv2.VideoCapture(path)
    samples = []
    for timestamp in SAMPLE_TIMES:
        if timestamp > duration_sec:
            continue
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ok, bgr = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        metrics = detect_frame(rgb, FIXED_CONFIG)
        samples.append({"t": timestamp, **metrics.to_dict()})
    cap.release()
    return samples


def inspect_video(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    duration_sec = frame_count / fps if fps else 0.0
    result = run_video(path, sample_fps=1.0, config=AUTO_CONFIG)
    return {
        "file": os.path.basename(path),
        "path": path,
        "duration_sec": round(duration_sec, 3),
        "fps": round(fps, 3),
        "size": f"{width}x{height}",
        "result": result.to_dict(),
        "fixed_samples": sample_metrics(path, duration_sec),
    }


def main() -> None:
    rows = [inspect_video(path) for path in PATHS]
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
