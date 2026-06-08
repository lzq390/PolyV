from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from polyv_detector.detector import DEFAULT_CONFIG, GelClimbDetector


COLORS = {
    "liquid_roi": (0, 180, 255),
    "sparse_roi": (255, 0, 255),
    "rod_roi": (0, 255, 0),
    "shaft_core_exclusion": (0, 0, 255),
}


def main() -> None:
    args = parse_args()
    videos = videos_from_manifest(args.source_manifest)
    args.outputs.mkdir(parents=True, exist_ok=True)

    items = []
    thumbs = []
    for index, video in enumerate(videos, 1):
        result, frame, error = analyze_video(video, args.time_sec, args.sample_fps)
        item = {"index": index, "video": str(video), "ok": error is None}
        if error is not None:
            item["error"] = error
            items.append(item)
            continue

        height, width = frame.shape[:2]
        image_path = args.outputs / f"{index:02d}_{safe_stem(video)}_liquid_roi_t{int(round(args.time_sec)):02d}.jpg"
        overlay = draw_overlay(frame, result, index, image_path)
        evidence = result.evidence
        rois = evidence.get("rois") or {}
        scores = evidence.get("roi_scores") or {}
        liquid = rois.get("liquid_roi") or [0.0, 0.0, 0.0, 0.0]
        item.update(
            {
                "time_sec": args.time_sec,
                "width": width,
                "height": height,
                "image": str(image_path),
                "state": result.state,
                "confidence": result.confidence,
                "liquid_roi": [round(float(v), 5) for v in liquid],
                "white_coverage": round(float(evidence.get("white_coverage", 0.0)), 4),
                "liquid_ellipse_score": round(float(scores.get("liquid_ellipse_score", 0.0)), 4),
                "roi_quality": round(float(evidence.get("roi_quality", 0.0)), 4),
            }
        )
        items.append(item)
        thumb = cv2.resize(overlay, (640, int(640 * height / width)), interpolation=cv2.INTER_AREA)
        thumbs.append((index, video.name, thumb, item))

    contact_path = args.outputs / "liquid_roi_contact_sheet.jpg"
    make_contact_sheet(thumbs, contact_path)
    manifest = {"contact_sheet": str(contact_path), "items": items}
    (args.outputs / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(contact_path)
    print(json.dumps(items, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate liquid_roi overlays for the manifest videos.")
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("outputs/liquid_roi_batch_review/manifest.json"),
        help="Manifest containing the video list to reprocess.",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path("outputs/liquid_roi_batch_review_v2"),
        help="Output directory for images and manifest.",
    )
    parser.add_argument("--time-sec", type=float, default=61.0)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    return parser.parse_args()


def videos_from_manifest(path: Path) -> list[Path]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    videos = []
    for item in payload.get("items", []):
        video = item.get("video")
        if video:
            videos.append(Path(video))
    return videos


def analyze_video(video: Path, time_sec: float, sample_fps: float):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None, None, "failed to open video"

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_step = max(1, int(round(fps / sample_fps)))
    detector = GelClimbDetector(config=replace(DEFAULT_CONFIG, roi_mode="auto"))
    result = None
    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        timestamp_sec = frame_idx / fps
        if timestamp_sec > time_sec:
            break
        if frame_idx % frame_step == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = detector.update(frame_rgb, timestamp_sec)
        frame_idx += 1

    cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000.0)
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(time_sec * fps))
        ok, frame = cap.read()
    cap.release()

    if result is None:
        return None, frame, "no detector result"
    if not ok:
        return result, None, "failed to read overlay frame"
    return result, frame, None


def draw_overlay(frame: np.ndarray, result, index: int, path: Path) -> np.ndarray:
    image = frame.copy()
    height, width = image.shape[:2]
    evidence = result.evidence
    rois = evidence.get("rois") or {}

    for name in ("liquid_roi", "sparse_roi", "rod_roi", "shaft_core_exclusion"):
        rect = rois.get(name)
        if not rect or len(rect) != 4:
            continue
        x0, y0, x1, y1 = rect_px(rect, width, height)
        color = COLORS.get(name, (255, 255, 255))
        cv2.rectangle(image, (x0, y0), (x1, y1), color, 4)
        cv2.putText(image, name, (x0, max(24, y0 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    liquid = rois.get("liquid_roi") or [0.0, 0.0, 0.0, 0.0]
    scores = evidence.get("roi_scores") or {}
    lines = [
        f"#{index} liquid=[{liquid[0]:.5f}, {liquid[1]:.5f}, {liquid[2]:.5f}, {liquid[3]:.5f}]",
        f"wc={evidence.get('white_coverage')} ellipse_score={scores.get('liquid_ellipse_score')}",
        f"state={result.state} roi_quality={evidence.get('roi_quality')}",
    ]
    for row, line in enumerate(lines):
        origin = (18, 34 + row * 30)
        cv2.putText(image, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 3)
        cv2.putText(image, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 1)

    cv2.imwrite(str(path), image)
    return image


def make_contact_sheet(thumbs: list[tuple[int, str, np.ndarray, dict]], path: Path) -> None:
    if not thumbs:
        return
    cols = 2
    pad = 18
    label_h = 72
    cell_w = 640
    cell_h = max(thumb.shape[0] for _, _, thumb, _ in thumbs) + label_h
    rows = (len(thumbs) + cols - 1) // cols
    sheet = np.full((rows * cell_h + (rows + 1) * pad, cols * cell_w + (cols + 1) * pad, 3), 245, np.uint8)

    for idx, (index, name, thumb, item) in enumerate(thumbs):
        row = idx // cols
        col = idx % cols
        x = pad + col * (cell_w + pad)
        y = pad + row * (cell_h + pad)
        title = (
            f"#{index} liquid={item['liquid_roi']} "
            f"wc={item['white_coverage']} score={item['liquid_ellipse_score']}"
        )
        cv2.putText(sheet, title, (x, y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 2)
        cv2.putText(sheet, name, (x, y + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (60, 60, 60), 1)
        thumb_h, thumb_w = thumb.shape[:2]
        sheet[y + label_h : y + label_h + thumb_h, x : x + thumb_w] = thumb

    cv2.imwrite(str(path), sheet)


def safe_stem(path: Path) -> str:
    return path.stem.replace("/", "_").replace("\\", "_").replace(" ", "_")


def rect_px(rect: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    return (
        int(round(rect[0] * width)),
        int(round(rect[1] * height)),
        int(round(rect[2] * width)),
        int(round(rect[3] * height)),
    )


if __name__ == "__main__":
    main()
