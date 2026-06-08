from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from polyv_detector.detector import DEFAULT_CONFIG


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv", ".webm"}
PREFIXES = ("VID", "飞书")


def main() -> None:
    args = parse_args()
    videos = list_videos(args.downloads, set(args.dates), include_current=args.include_current)
    outputs_root = args.outputs
    outputs_root.mkdir(parents=True, exist_ok=True)

    manifest = []
    for video in videos:
        out_dir = outputs_root / safe_stem(video)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[PolyV] processing {video.name} -> {out_dir}", flush=True)
        manifest.append(process_video(video, out_dir, args.snapshot_time))

    manifest_path = outputs_root / "batch_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate motion plot and ROI comparison images for recent Downloads videos."
    )
    parser.add_argument(
        "--downloads",
        type=Path,
        default=Path("/mnt/c/Users/ASUS/Downloads"),
        help="Downloads directory to scan.",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path("outputs"),
        help="Project output directory.",
    )
    parser.add_argument(
        "--dates",
        nargs="+",
        default=["2026-06-03", "2026-06-04"],
        help="Local file modified dates to include, YYYY-MM-DD.",
    )
    parser.add_argument(
        "--snapshot-time",
        type=float,
        default=60.0,
        help="Video timestamp used for ROI screenshots.",
    )
    parser.add_argument(
        "--include-current",
        action="store_true",
        help="Include VID_20260601_144516.mp4 instead of skipping it.",
    )
    return parser.parse_args()


def list_videos(downloads: Path, dates: set[str], *, include_current: bool) -> list[Path]:
    videos = []
    for path in downloads.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        if not path.name.startswith(PREFIXES):
            continue
        if not include_current and path.name == "VID_20260601_144516.mp4":
            continue
        modified = datetime.fromtimestamp(path.stat().st_mtime).date().isoformat()
        if modified in dates:
            videos.append(path)
    return sorted(videos, key=lambda p: p.name)


def process_video(video_path: Path, out_dir: Path, snapshot_time_sec: float) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "video": str(video_path),
            "output_dir": str(out_dir),
            "ok": False,
            "error": "failed to open video",
        }

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frame_count / fps if fps else 0.0

    motion_rows = compute_motion_rows(cap, duration, fps, width, height)
    motion_plot_path = out_dir / "motion_plot.jpg"
    make_motion_plot(motion_rows, motion_plot_path)

    frame = frame_at(cap, min(snapshot_time_sec, max(0.0, duration - 0.1)), fps)
    cap.release()
    if frame is None:
        return {
            "video": str(video_path),
            "output_dir": str(out_dir),
            "ok": False,
            "error": "failed to read ROI frame",
        }

    center_roi_path = out_dir / "motion_plot_center_roi_t60.jpg"
    detector_rois_path = out_dir / "detector_three_rois_t60.jpg"
    make_center_roi_image(frame, width, height, snapshot_time_sec, center_roi_path)
    detector_report = make_detector_roi_image(frame, width, height, snapshot_time_sec, detector_rois_path)

    return {
        "video": str(video_path),
        "output_dir": str(out_dir),
        "ok": True,
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": duration,
        "motion_samples": len(motion_rows),
        "files": [
            str(motion_plot_path),
            str(center_roi_path),
            str(detector_rois_path),
        ],
        "detector_rois": detector_report,
    }


def safe_stem(path: Path) -> str:
    return path.stem.replace("/", "_").replace("\\", "_").strip()


def frame_at(cap: cv2.VideoCapture, t_sec: float, fps: float) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_sec) * 1000.0)
    ok, frame = cap.read()
    if not ok:
        frame_idx = max(0, int(t_sec * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
    return frame if ok else None


def compute_motion_rows(
    cap: cv2.VideoCapture,
    duration: float,
    fps: float,
    width: int,
    height: int,
) -> list[dict]:
    roi = {
        "x0": int(width * 0.22),
        "x1": int(width * 0.78),
        "y0": int(height * 0.16),
        "y1": int(height * 0.90),
    }
    motion_rows = []
    prev = None
    motion_times = np.arange(0.0, max(0.0, duration), 2.0)
    if len(motion_times) > 360:
        motion_times = np.linspace(0.0, duration, 360)

    for t_sec in motion_times:
        frame = frame_at(cap, float(t_sec), fps)
        if frame is None:
            continue
        crop = frame[roi["y0"] : roi["y1"], roi["x0"] : roi["x1"]]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (180, 260), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (5, 5), 0)
        if prev is not None:
            diff = cv2.absdiff(small, prev)
            motion_rows.append(
                {
                    "t_sec": float(t_sec),
                    "mean_absdiff": float(diff.mean()),
                    "p95_absdiff": float(np.percentile(diff, 95)),
                }
            )
        prev = small
    return motion_rows


def make_motion_plot(rows: list[dict], path: Path) -> None:
    w, h = 1200, 360
    pad_l, pad_r, pad_t, pad_b = 70, 30, 25, 52
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([pad_l, pad_t, w - pad_r, h - pad_b], outline=(160, 160, 160))

    if rows:
        xs = [r["t_sec"] for r in rows]
        ys = [r["mean_absdiff"] for r in rows]
        x_min, x_max = min(xs), max(xs)
        y_max = max(1.0, float(np.percentile(ys, 98)) * 1.15)
    else:
        x_min, x_max, y_max = 0.0, 1.0, 1.0

    for i in range(6):
        x = pad_l + (w - pad_l - pad_r) * i / 5
        t = x_min + (x_max - x_min) * i / 5
        draw.line([x, pad_t, x, h - pad_b], fill=(235, 235, 235))
        draw.text((x - 24, h - pad_b + 12), f"{t / 60:.1f}m", fill=(70, 70, 70))
    for i in range(5):
        y = pad_t + (h - pad_t - pad_b) * i / 4
        val = y_max * (1 - i / 4)
        draw.line([pad_l, y, w - pad_r, y], fill=(235, 235, 235))
        draw.text((10, y - 8), f"{val:.1f}", fill=(70, 70, 70))

    pts = []
    for row in rows:
        x = pad_l + (row["t_sec"] - x_min) / max(1e-9, x_max - x_min) * (w - pad_l - pad_r)
        y = h - pad_b - min(row["mean_absdiff"], y_max) / y_max * (h - pad_t - pad_b)
        pts.append((x, y))
    if len(pts) > 1:
        draw.line(pts, fill=(18, 96, 186), width=2)
    draw.text(
        (pad_l, 4),
        "Center ROI motion energy: mean absolute gray-frame difference",
        fill=(20, 20, 20),
    )
    img.save(path, quality=90)


def make_center_roi_image(frame_bgr: np.ndarray, width: int, height: int, time_sec: float, path: Path) -> None:
    roi = {
        "x0": int(width * 0.22),
        "x1": int(width * 0.78),
        "y0": int(height * 0.16),
        "y1": int(height * 0.90),
    }
    full = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    crop = full.crop((roi["x0"], roi["y0"], roi["x1"], roi["y1"]))
    overlay = full.copy()
    draw = ImageDraw.Draw(overlay)
    for inset in range(5):
        draw.rectangle(
            [roi["x0"] - inset, roi["y0"] - inset, roi["x1"] + inset, roi["y1"] + inset],
            outline=(255, 36, 28),
        )

    left_w = 760
    left_h = int(left_w * height / width)
    right_w = 760
    right_h = int(right_w * crop.height / crop.width)
    label_h = 34
    pad = 18
    canvas_w = left_w + right_w + pad * 3
    canvas_h = max(left_h, right_h) + label_h + pad * 2
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    d = ImageDraw.Draw(canvas)
    canvas.paste(overlay.resize((left_w, left_h)), (pad, pad + label_h))
    canvas.paste(crop.resize((right_w, right_h)), (pad * 2 + left_w, pad + label_h))
    d.text((pad, pad), f"Full frame with fixed center ROI, t={time_sec:.1f}s", fill=(20, 20, 20))
    d.text((pad * 2 + left_w, pad), "Cropped ROI used for motion_plot", fill=(20, 20, 20))
    canvas.save(path, quality=92)


def make_detector_roi_image(frame_bgr: np.ndarray, width: int, height: int, time_sec: float, path: Path) -> list[dict]:
    rois = [
        ("liquid_roi", DEFAULT_CONFIG.liquid_roi, (255, 36, 28)),
        ("sparse_roi", DEFAULT_CONFIG.sparse_roi, (35, 118, 255)),
        ("rod_roi", DEFAULT_CONFIG.rod_roi, (32, 170, 83)),
    ]

    def rect_px(rect) -> dict:
        return {
            "x0": int(round(rect.x0 * width)),
            "y0": int(round(rect.y0 * height)),
            "x1": int(round(rect.x1 * width)),
            "y1": int(round(rect.y1 * height)),
        }

    full = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    overlay = full.copy()
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except OSError:
        font = None
        small_font = None

    roi_report = []
    for name, rect, color in rois:
        box = rect_px(rect)
        roi_report.append({"name": name, **box})
        for inset in range(5):
            draw.rectangle(
                [box["x0"] - inset, box["y0"] - inset, box["x1"] + inset, box["y1"] + inset],
                outline=color,
            )
        label = f"{name} ({box['x0']},{box['y0']})-({box['x1']},{box['y1']})"
        text_bbox = draw.textbbox((0, 0), label, font=small_font)
        tx, ty = box["x0"], max(0, box["y0"] - 28)
        draw.rectangle([tx, ty, tx + text_bbox[2] + 10, ty + 25], fill=(255, 255, 255))
        draw.text((tx + 5, ty + 4), label, fill=color, font=small_font)

    full_w = 1280
    full_h = int(full_w * height / width)
    thumb_w, thumb_h = 390, 260
    pad, label_h = 20, 36
    sheet_w = max(full_w + pad * 2, pad * 4 + thumb_w * 3)
    sheet_h = pad + label_h + full_h + pad + label_h + thumb_h + pad
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    sd = ImageDraw.Draw(sheet)
    sd.text((pad, pad), f"Detector ROIs on current video frame, t={time_sec:.1f}s", fill=(20, 20, 20), font=font)
    sheet.paste(overlay.resize((full_w, full_h)), (pad, pad + label_h))

    crop_y = pad + label_h + full_h + pad + label_h
    for idx, (name, rect, color) in enumerate(rois):
        box = rect_px(rect)
        crop = full.crop((box["x0"], box["y0"], box["x1"], box["y1"]))
        scale = min(thumb_w / crop.width, thumb_h / crop.height)
        cw, ch = max(1, int(crop.width * scale)), max(1, int(crop.height * scale))
        crop_resized = crop.resize((cw, ch))
        x = pad + idx * (thumb_w + pad)
        y = crop_y
        sd.text((x, y - label_h + 7), name, fill=color, font=font)
        sd.rectangle([x, y, x + thumb_w, y + thumb_h], outline=color, width=3)
        sheet.paste(crop_resized, (x + (thumb_w - cw) // 2, y + (thumb_h - ch) // 2))

    sheet.save(path, quality=92)
    return roi_report


if __name__ == "__main__":
    main()
