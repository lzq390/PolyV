from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from polyv_detector.detector import DEFAULT_CONFIG
from polyv_detector.dynamic_detector import (
    _detect_broad_liquid_ellipse_window,
    _detect_compact_liquid_ellipse_window,
    _milky_liquid_mask,
)


VIDEOS = [
    Path("/mnt/c/Users/ASUS/Downloads/VID_20260601_142527.mp4"),
    Path("/mnt/c/Users/ASUS/Downloads/VID_20260601_144516.mp4"),
    Path("/mnt/c/Users/ASUS/Downloads/VID_20260601_150311.mp4"),
    Path("/mnt/c/Users/ASUS/Downloads/VID_20260603_151103 - Trim.mp4"),
    Path("/mnt/c/Users/ASUS/Downloads/飞书20260604-111405.mp4"),
    Path("/mnt/c/Users/ASUS/Downloads/飞书20260604-111732.mp4"),
    Path("/mnt/c/Users/ASUS/Downloads/飞书20260604-111744.mp4"),
]


def main() -> None:
    out_dir = ROOT / "outputs" / "liquid_roi_strategy_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {}
    for strategy in ("compact", "broad"):
        contact, items = make_sheet(strategy, out_dir)
        manifest[strategy] = {"contact_sheet": str(contact), "items": items}
        print(f"{strategy}: {contact}")

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def make_sheet(strategy: str, out_dir: Path) -> tuple[Path, list[dict]]:
    thumbs = []
    items = []
    for index, video in enumerate(VIDEOS, 1):
        frame = frame_at(video, 61.0)
        if frame is None:
            items.append({"index": index, "video": str(video), "strategy": strategy, "ok": False})
            continue

        overlay, item = draw_overlay(frame, video, index, strategy, out_dir)
        thumbs.append((index, video.name, overlay, item))
        items.append({"ok": True, **item})

    contact = out_dir / f"{strategy}_contact_sheet.jpg"
    write_contact_sheet(thumbs, strategy, contact)
    return contact, items


def frame_at(video: Path, time_sec: float) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000.0)
    ok, frame = cap.read()
    if not ok:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(time_sec * fps))
        ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def draw_overlay(frame_bgr: np.ndarray, video: Path, index: int, strategy: str, out_dir: Path) -> tuple[np.ndarray, dict]:
    height, width = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    box = detect_box(strategy, frame_rgb, width, height)
    x0, y0, x1, y1 = [int(round(v)) for v in box[:4]]
    norm = [box[0] / width, box[1] / height, box[2] / width, box[3] / height, box[4]]

    image = frame_bgr.copy()
    color = (0, 180, 255) if strategy == "compact" else (0, 90, 255)
    cv2.rectangle(image, (x0, y0), (x1, y1), color, 4)
    cv2.putText(
        image,
        f"{strategy}_liquid_roi",
        (x0, max(28, y0 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        color,
        2,
    )

    lines = [
        f"#{index} {strategy}=[{norm[0]:.5f}, {norm[1]:.5f}, {norm[2]:.5f}, {norm[3]:.5f}] score={norm[4]:.4f}",
        video.name,
    ]
    for row, line in enumerate(lines):
        origin = (18, 34 + row * 30)
        cv2.putText(image, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 3)
        cv2.putText(image, line, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 1)

    image_path = out_dir / f"{index:02d}_{safe_stem(video)}_{strategy}.jpg"
    cv2.imwrite(str(image_path), image)
    item = {
        "index": index,
        "video": str(video),
        "image": str(image_path),
        "strategy": strategy,
        "liquid_roi": [round(float(v), 5) for v in norm[:4]],
        "score": round(float(norm[4]), 4),
        "width": width,
        "height": height,
    }
    return image, item


def detect_box(strategy: str, frame_rgb: np.ndarray, width: int, height: int) -> tuple[float, float, float, float, float]:
    milk_mask = _milky_liquid_mask(frame_rgb, cv2).astype(np.uint8)
    if strategy == "compact":
        box = _detect_compact_liquid_ellipse_window(frame_rgb, milk_mask, width, height, cv2)
    elif strategy == "broad":
        box = _detect_broad_liquid_ellipse_window(milk_mask, width, height, cv2)
    else:
        raise ValueError(strategy)

    if box is not None:
        return box

    fallback = DEFAULT_CONFIG.liquid_roi
    return (
        fallback.x0 * width,
        fallback.y0 * height,
        fallback.x1 * width,
        fallback.y1 * height,
        0.0,
    )


def write_contact_sheet(thumbs: list[tuple[int, str, np.ndarray, dict]], strategy: str, path: Path) -> None:
    if not thumbs:
        return
    cols = 2
    pad = 18
    label_h = 72
    cell_w = 640
    resized = []
    for index, name, image, item in thumbs:
        height, width = image.shape[:2]
        thumb = cv2.resize(image, (cell_w, int(cell_w * height / width)), interpolation=cv2.INTER_AREA)
        resized.append((index, name, thumb, item))

    cell_h = max(thumb.shape[0] for _, _, thumb, _ in resized) + label_h
    rows = (len(resized) + cols - 1) // cols
    sheet = np.full((rows * cell_h + (rows + 1) * pad, cols * cell_w + (cols + 1) * pad, 3), 245, np.uint8)

    for idx, (index, name, thumb, item) in enumerate(resized):
        row = idx // cols
        col = idx % cols
        x = pad + col * (cell_w + pad)
        y = pad + row * (cell_h + pad)
        title = f"#{index} {strategy}={item['liquid_roi']} score={item['score']}"
        cv2.putText(sheet, title, (x, y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 2)
        cv2.putText(sheet, name, (x, y + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (60, 60, 60), 1)
        thumb_h, thumb_w = thumb.shape[:2]
        sheet[y + label_h : y + label_h + thumb_h, x : x + thumb_w] = thumb

    cv2.imwrite(str(path), sheet)


def safe_stem(path: Path) -> str:
    return path.stem.replace("/", "_").replace("\\", "_").replace(" ", "_")


if __name__ == "__main__":
    main()
