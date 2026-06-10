from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .detector import DEFAULT_CONFIG, DetectionConfig, DetectionResult, GelClimbDetector


def main() -> None:
    args = _parse_args()
    result = run_video(
        video_path=args.video,
        sample_fps=args.sample_fps,
        config=_config_from_args(args),
        max_seconds=args.max_seconds,
        debug_overlay_dir=args.debug_overlay_dir,
    )
    payload = result.to_dict()
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n", encoding="utf-8")
    print(text)


def run_video(
    video_path: str,
    sample_fps: float,
    config: DetectionConfig | None = None,
    max_seconds: float | None = None,
    debug_overlay_dir: str | None = None,
) -> DetectionResult:
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("OpenCV is required. Use the conda CV environment.") from exc

    if config is None:
        config = replace(DEFAULT_CONFIG, roi_mode="auto")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_step = max(1, int(round(video_fps / sample_fps)))
    detector = GelClimbDetector(config=config)
    latest_result = None
    frame_idx = 0
    checkpoints = sorted({0.0, config.calibration_duration_sec, config.baseline_duration_sec, 60.0})
    saved_checkpoints: set[float] = set()
    saved_first_candidate = False
    overlay_dir = Path(debug_overlay_dir) if debug_overlay_dir else None
    if overlay_dir:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        timestamp_sec = frame_idx / video_fps
        if max_seconds is not None and timestamp_sec > max_seconds:
            break
        if frame_idx % frame_step == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            latest_result = detector.update(frame_rgb, timestamp_sec)
            if overlay_dir:
                for checkpoint in checkpoints:
                    if checkpoint not in saved_checkpoints and timestamp_sec >= checkpoint:
                        _write_overlay(
                            frame_bgr,
                            latest_result,
                            overlay_dir / f"t{int(round(checkpoint)):04d}.jpg",
                        )
                        saved_checkpoints.add(checkpoint)
                if latest_result.evidence.get("is_final_candidate") and not saved_first_candidate:
                    _write_overlay(frame_bgr, latest_result, overlay_dir / "first_candidate.jpg")
                    saved_first_candidate = True
            if latest_result.alert:
                if overlay_dir:
                    _write_overlay(frame_bgr, latest_result, overlay_dir / "alert.jpg")
                break
        frame_idx += 1

    cap.release()
    if latest_result is None:
        raise SystemExit("No frames were processed.")
    return latest_result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect LIQUID_STIRRING vs FINAL_GEL_ROD_CLIMBING in a fixed-camera video."
    )
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--json-out", help="Optional output JSON path.")
    parser.add_argument("--sample-fps", type=float, default=1.0, help="Frames per second to analyze.")
    parser.add_argument("--max-seconds", type=float, help="Optional processing limit for quick checks.")
    parser.add_argument("--roi-mode", choices=["auto", "dynamic", "fixed"], default="auto")
    parser.add_argument("--calibration-duration-sec", type=float, default=DEFAULT_CONFIG.calibration_duration_sec)
    parser.add_argument("--baseline-duration-sec", type=float, default=DEFAULT_CONFIG.baseline_duration_sec)
    parser.add_argument("--metric-smooth-sec", type=float, default=DEFAULT_CONFIG.metric_smooth_sec)
    parser.add_argument("--roi-quality-min", type=float, default=DEFAULT_CONFIG.roi_quality_min)
    parser.add_argument("--debug-overlay-dir", help="Optional directory for ROI/metric overlay images.")
    parser.add_argument("--stable-duration-sec", type=float, default=DEFAULT_CONFIG.stable_duration_sec)
    parser.add_argument("--white-coverage-min", type=float, default=DEFAULT_CONFIG.white_coverage_min)
    parser.add_argument("--sparse-hole-ratio-max", type=float, default=DEFAULT_CONFIG.sparse_hole_ratio_max)
    parser.add_argument("--rod-wrap-height-min-px", type=int, default=DEFAULT_CONFIG.rod_wrap_height_min_px)
    parser.add_argument("--rod-wrap-ratio-min", type=float, default=DEFAULT_CONFIG.rod_wrap_ratio_min)
    parser.add_argument("--rod-wrap-delta-ratio-min", type=float, default=DEFAULT_CONFIG.rod_wrap_delta_ratio_min)
    parser.add_argument("--connected-area-ratio-min", type=float, default=DEFAULT_CONFIG.connected_area_ratio_min)
    parser.add_argument(
        "--connected-area-delta-ratio-min",
        type=float,
        default=DEFAULT_CONFIG.connected_area_delta_ratio_min,
    )
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> DetectionConfig:
    return DetectionConfig(
        roi_mode=args.roi_mode,
        calibration_duration_sec=args.calibration_duration_sec,
        baseline_duration_sec=args.baseline_duration_sec,
        metric_smooth_sec=args.metric_smooth_sec,
        roi_quality_min=args.roi_quality_min,
        stable_duration_sec=args.stable_duration_sec,
        white_coverage_min=args.white_coverage_min,
        sparse_hole_ratio_max=args.sparse_hole_ratio_max,
        rod_wrap_height_min_px=args.rod_wrap_height_min_px,
        rod_wrap_ratio_min=args.rod_wrap_ratio_min,
        rod_wrap_delta_ratio_min=args.rod_wrap_delta_ratio_min,
        connected_area_ratio_min=args.connected_area_ratio_min,
        connected_area_delta_ratio_min=args.connected_area_delta_ratio_min,
    )


def _write_overlay(frame_bgr, result: DetectionResult, path: Path) -> None:
    import cv2

    image = frame_bgr.copy()
    height, width = image.shape[:2]
    evidence = result.evidence
    rois = evidence.get("rois") or {}
    colors = {
        "liquid_roi": (0, 180, 255),
        "rod_roi": (0, 255, 0),
        "shaft_core_exclusion": (0, 0, 255),
    }
    shaft_center_x = None
    for name in ("liquid_roi", "rod_roi", "shaft_core_exclusion"):
        rect = rois.get(name)
        if not rect:
            continue
        if len(rect) != 4:
            continue
        x0, y0, x1, y1 = _rect_px(rect, width, height)
        cv2.rectangle(image, (x0, y0), (x1, y1), colors.get(name, (255, 255, 255)), 2)
        cv2.putText(image, name, (x0, max(20, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors.get(name, (255, 255, 255)), 2)
        if name == "shaft_core_exclusion":
            shaft_center_x = int(round((x0 + x1) / 2))
    if shaft_center_x is not None:
        cv2.line(image, (shaft_center_x, 0), (shaft_center_x, height - 1), (0, 0, 255), 2)
        cv2.putText(image, "shaft_x", (shaft_center_x + 6, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    lines = [
        f"state={result.state} alert={result.alert}",
        f"roi_quality={evidence.get('roi_quality')} candidate={evidence.get('is_final_candidate')}",
        f"rod_wrap_ratio={evidence.get('rod_wrap_ratio')} delta={evidence.get('rod_wrap_delta_ratio')}",
        f"stable_ratio={evidence.get('stable_candidate_ratio')}",
    ]
    for idx, line in enumerate(lines):
        cv2.putText(image, line, (20, 32 + idx * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.imwrite(str(path), image)


def _rect_px(rect: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x0 = int(round(rect[0] * width))
    y0 = int(round(rect[1] * height))
    x1 = int(round(rect[2] * width))
    y1 = int(round(rect[3] * height))
    return x0, y0, x1, y1


if __name__ == "__main__":
    main()
