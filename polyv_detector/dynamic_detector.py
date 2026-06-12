from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from typing import Deque, NamedTuple

import numpy as np

from .states import FINAL_GEL_ROD_CLIMBING, LIQUID_STIRRING


class Rect(NamedTuple):
    """Normalized rectangle: x0, y0, x1, y1."""

    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(frozen=True)
class DetectionConfig:
    liquid_roi: Rect = Rect(0.50, 0.58, 0.74, 0.95)
    sparse_roi: Rect = Rect(0.55, 0.64, 0.68, 0.88)
    rod_roi: Rect = Rect(0.58, 0.36, 0.72, 0.90)
    shaft_core_exclusion: Rect = Rect(0.64, 0.36, 0.68, 0.90)

    roi_mode: str = "fixed"
    calibration_duration_sec: float = 30.0
    baseline_duration_sec: float = 60.0
    metric_smooth_sec: float = 10.0
    roi_quality_min: float = 0.60

    white_min_value: int = 145
    white_max_channel_span: int = 110
    white_min_red: int = 125
    white_min_green: int = 115
    white_min_blue: int = 70
    warm_min_red: int = 145
    warm_min_green: int = 105
    warm_min_blue: int = 35
    warm_min_red_blue_delta: int = 45
    warm_min_green_blue_delta: int = 25
    warm_max_red_green_delta: int = 95
    orange_min_red_blue_delta: int = 75
    orange_min_saturation: int = 60
    sparse_dark_value: int = 150

    white_coverage_min: float = 0.45
    sparse_hole_ratio_max: float = 0.05
    rod_wrap_height_min_px: int = 300

    rod_wrap_ratio_min: float = 0.70
    rod_wrap_delta_ratio_min: float = 0.30
    connected_area_ratio_min: float = 0.16
    connected_area_delta_ratio_min: float = 0.01
    dynamic_white_coverage_min: float = 0.40
    initial_final_white_coverage_min: float = 0.65
    rod_bottom_attachment_score_min: float = 0.80
    dynamic_final_min_elapsed_sec: float = 0.0
    rod_vertical_contrast_min: float = 0.45
    rod_white_path_top_density_min: float = 0.25
    rod_top_density_delta_min: float = 0.08
    rod_shape_progress_score_min: float = 1.0
    warm_material_coverage_min: float = 0.60
    rod_warm_material_ratio_min: float = 0.65
    rod_orange_maturity_ratio_min: float = 0.50
    rod_orange_maturity_delta_min: float = 0.25
    warm_connected_area_ratio_min: float = 0.35
    orange_material_path_score_min: float = 1.0

    stable_duration_sec: float = 60.0
    stable_min_ratio: float = 0.8
    window_time_tolerance_sec: float = 1.0

    bottom_seed_fraction: float = 0.35
    rod_bottom_band_fraction: float = 0.35
    rod_top_band_fraction: float = 0.30
    rod_center_exclusion_fraction: float = 0.24
    rod_flank_gap_ratio: float = 0.10
    rod_flank_width_ratio: float = 0.45
    rod_flank_bottom_density_max: float = 0.98
    sparse_component_stride: int = 2
    dynamic_roi_ema_alpha: float = 0.35
    dynamic_roi_update_interval_sec: float = 5.0
    rod_row_fill_min: float = 0.12


DEFAULT_CONFIG = DetectionConfig()


@dataclass(frozen=True)
class RoiSet:
    liquid_roi: Rect
    sparse_roi: Rect
    rod_roi: Rect
    shaft_core_exclusion: Rect | None
    source: str
    valid: bool
    quality: float
    failure_reason: str = ""
    shaft_x: float | None = None
    shaft_width: float | None = None
    bottle_left: float | None = None
    bottle_right: float | None = None
    bottle_bottom_y: float | None = None
    scores: dict[str, float] | None = None

    def rects_dict(self) -> dict[str, list[float]]:
        rects = {
            "liquid_roi": _rect_to_list(self.liquid_roi),
            "rod_roi": _rect_to_list(self.rod_roi),
        }
        if self.source == "fixed":
            rects["sparse_roi"] = _rect_to_list(self.sparse_roi)
            if self.shaft_core_exclusion is not None:
                rects["shaft_core_exclusion"] = _rect_to_list(self.shaft_core_exclusion)
        return rects


@dataclass(frozen=True)
class RoiGeometry:
    shaft_x_px: float
    shaft_width_px: float
    rod_top_y_px: float
    rod_bottom_y_px: float
    liquid_x0_px: float
    liquid_y0_px: float
    liquid_x1_px: float
    liquid_y1_px: float
    bottle_left_px: float
    bottle_right_px: float
    bottle_bottom_y_px: float
    frame_width: int
    frame_height: int
    rod_axis_score: float
    liquid_ellipse_score: float
    temporal_stability_score: float
    bottle_alignment_score: float
    roi_sanity_score: float
    bottle_detected: bool

    @property
    def quality(self) -> float:
        return _clamp01(
            0.30 * self.rod_axis_score
            + 0.20 * self.temporal_stability_score
            + 0.15 * self.bottle_alignment_score
            + 0.15 * self.liquid_ellipse_score
            + 0.20 * self.roi_sanity_score
        )

    @property
    def scores(self) -> dict[str, float]:
        return {
            "rod_axis_score": self.rod_axis_score,
            "liquid_ellipse_score": self.liquid_ellipse_score,
            "temporal_stability_score": self.temporal_stability_score,
            "bottle_alignment_score": self.bottle_alignment_score,
            "roi_sanity_score": self.roi_sanity_score,
        }


@dataclass(frozen=True)
class BaselineModel:
    rod_wrap_ratio: float
    rod_wrap_iqr: float
    connected_area_ratio: float
    connected_area_iqr: float
    white_coverage: float
    white_coverage_iqr: float
    rod_top_density: float
    rod_top_density_iqr: float
    rod_orange_maturity_ratio: float
    rod_orange_maturity_iqr: float
    initial_final_like: bool

    def to_evidence(self) -> dict:
        return {
            "baseline_rod_wrap_ratio": round(self.rod_wrap_ratio, 4),
            "baseline_rod_wrap_iqr": round(self.rod_wrap_iqr, 4),
            "baseline_connected_area_ratio": round(self.connected_area_ratio, 4),
            "baseline_connected_area_iqr": round(self.connected_area_iqr, 4),
            "baseline_white_coverage": round(self.white_coverage, 4),
            "baseline_white_coverage_iqr": round(self.white_coverage_iqr, 4),
            "baseline_rod_top_density": round(self.rod_top_density, 4),
            "baseline_rod_top_density_iqr": round(self.rod_top_density_iqr, 4),
            "baseline_rod_orange_maturity_ratio": round(self.rod_orange_maturity_ratio, 4),
            "baseline_rod_orange_maturity_iqr": round(self.rod_orange_maturity_iqr, 4),
            "initial_final_like": bool(self.initial_final_like),
        }


class RodAttachmentMetrics(NamedTuple):
    score: float
    bottom_density: float
    top_density: float
    vertical_contrast: float
    side_bottom_density: float
    flank_bottom_density: float
    local_contrast: float


@dataclass(frozen=True)
class FrameMetrics:
    white_coverage: float
    rod_wrap_height_px: int
    rod_wrap_ratio: float
    connected_area_ratio: float
    rod_connection_score: bool
    is_final_candidate: bool
    sparse_hole_ratio: float = 0.0
    rod_bottom_attachment_score: float = 0.0
    rod_bottom_density: float = 0.0
    rod_top_density: float = 0.0
    rod_vertical_contrast: float = 0.0
    rod_side_bottom_density: float = 0.0
    rod_flank_bottom_density: float = 0.0
    rod_local_contrast: float = 0.0
    rod_top_density_delta: float | None = None
    rod_shape_progress_score: float = 0.0
    warm_material_coverage: float = 0.0
    rod_warm_material_ratio: float = 0.0
    rod_orange_maturity_ratio: float = 0.0
    rod_orange_maturity_delta: float | None = None
    material_connected_area_ratio: float = 0.0
    orange_material_path_score: float = 0.0
    roi_source: str = "fixed"
    roi_valid: bool = True
    roi_quality: float = 1.0
    roi_failure_reason: str = ""
    rois: dict[str, list[float]] | None = None
    roi_scores: dict[str, float] | None = None
    baseline: BaselineModel | None = None
    rod_wrap_delta_ratio: float | None = None
    connected_area_delta_ratio: float | None = None
    white_coverage_delta: float | None = None

    def to_dict(self) -> dict:
        payload = {
            "white_coverage": round(float(self.white_coverage), 4),
            "rod_wrap_height_px": int(self.rod_wrap_height_px),
            "rod_wrap_ratio": round(float(self.rod_wrap_ratio), 4),
            "connected_area_ratio": round(float(self.connected_area_ratio), 4),
            "rod_bottom_attachment_score": round(float(self.rod_bottom_attachment_score), 4),
            "rod_bottom_density": round(float(self.rod_bottom_density), 4),
            "rod_top_density": round(float(self.rod_top_density), 4),
            "rod_vertical_contrast": round(float(self.rod_vertical_contrast), 4),
            "rod_side_bottom_density": round(float(self.rod_side_bottom_density), 4),
            "rod_flank_bottom_density": round(float(self.rod_flank_bottom_density), 4),
            "rod_local_contrast": round(float(self.rod_local_contrast), 4),
            "rod_shape_progress_score": round(float(self.rod_shape_progress_score), 4),
            "warm_material_coverage": round(float(self.warm_material_coverage), 4),
            "rod_warm_material_ratio": round(float(self.rod_warm_material_ratio), 4),
            "rod_orange_maturity_ratio": round(float(self.rod_orange_maturity_ratio), 4),
            "material_connected_area_ratio": round(float(self.material_connected_area_ratio), 4),
            "orange_material_path_score": round(float(self.orange_material_path_score), 4),
            "rod_connection_score": bool(self.rod_connection_score),
            "is_final_candidate": bool(self.is_final_candidate),
            "roi_source": self.roi_source,
            "roi_valid": bool(self.roi_valid),
            "roi_quality": round(float(self.roi_quality), 4),
            "roi_failure_reason": self.roi_failure_reason,
        }
        if self.roi_source == "fixed":
            payload["sparse_hole_ratio"] = round(float(self.sparse_hole_ratio), 4)
        if self.rois is not None:
            payload["rois"] = self.rois
        if self.roi_scores is not None:
            payload["roi_scores"] = {k: round(float(v), 4) for k, v in self.roi_scores.items()}
        if self.baseline is not None:
            payload.update(self.baseline.to_evidence())
        if self.rod_wrap_delta_ratio is not None:
            payload["rod_wrap_delta_ratio"] = round(float(self.rod_wrap_delta_ratio), 4)
        if self.connected_area_delta_ratio is not None:
            payload["connected_area_delta_ratio"] = round(float(self.connected_area_delta_ratio), 4)
        if self.white_coverage_delta is not None:
            payload["white_coverage_delta"] = round(float(self.white_coverage_delta), 4)
        if self.rod_top_density_delta is not None:
            payload["rod_top_density_delta"] = round(float(self.rod_top_density_delta), 4)
        if self.rod_orange_maturity_delta is not None:
            payload["rod_orange_maturity_delta"] = round(float(self.rod_orange_maturity_delta), 4)
        return payload


@dataclass(frozen=True)
class DetectionResult:
    state: str
    alert: bool
    transition_time_sec: float | None
    confidence: float
    evidence: dict

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "alert": self.alert,
            "transition_time_sec": self.transition_time_sec,
            "confidence": round(float(self.confidence), 4),
            "evidence": self.evidence,
        }


@dataclass
class DynamicRoiTracker:
    config: DetectionConfig
    geometry: RoiGeometry

    @classmethod
    def calibrate(cls, frames_rgb: list[np.ndarray], config: DetectionConfig) -> "DynamicRoiTracker":
        if not frames_rgb:
            raise ValueError("at least one calibration frame is required")
        detections = [_detect_single_geometry(_as_rgb_uint8(frame), config) for frame in frames_rgb]
        width = detections[0].frame_width
        height = detections[0].frame_height
        shaft_x_values = [d.shaft_x_px for d in detections]
        shaft_std = float(np.std(shaft_x_values)) if len(shaft_x_values) > 1 else 0.0
        temporal_stability = _clamp01(1.0 - shaft_std / max(1.0, 0.04 * width))
        geometry = RoiGeometry(
            shaft_x_px=_median(shaft_x_values),
            shaft_width_px=_median([d.shaft_width_px for d in detections]),
            rod_top_y_px=_median([d.rod_top_y_px for d in detections]),
            rod_bottom_y_px=_median([d.rod_bottom_y_px for d in detections]),
            liquid_x0_px=_median([d.liquid_x0_px for d in detections]),
            liquid_y0_px=_median([d.liquid_y0_px for d in detections]),
            liquid_x1_px=_median([d.liquid_x1_px for d in detections]),
            liquid_y1_px=_median([d.liquid_y1_px for d in detections]),
            bottle_left_px=_median([d.bottle_left_px for d in detections]),
            bottle_right_px=_median([d.bottle_right_px for d in detections]),
            bottle_bottom_y_px=_median([d.bottle_bottom_y_px for d in detections]),
            frame_width=width,
            frame_height=height,
            rod_axis_score=_median([d.rod_axis_score for d in detections]),
            liquid_ellipse_score=_median([d.liquid_ellipse_score for d in detections]),
            temporal_stability_score=temporal_stability,
            bottle_alignment_score=_median([d.bottle_alignment_score for d in detections]),
            roi_sanity_score=0.0,
            bottle_detected=any(d.bottle_detected for d in detections),
        )
        sanity = _roi_sanity_score(_roi_set_from_geometry(geometry, config, force_valid=True))
        return cls(config=config, geometry=replace(geometry, roi_sanity_score=sanity))

    def update(self, frame_rgb: np.ndarray) -> RoiSet:
        detected = _detect_single_geometry(_as_rgb_uint8(frame_rgb), self.config)
        if detected.rod_axis_score < 0.20:
            return self.rois()
        alpha = _clamp01(self.config.dynamic_roi_ema_alpha)
        self.geometry = RoiGeometry(
            shaft_x_px=_ema(self.geometry.shaft_x_px, detected.shaft_x_px, alpha),
            shaft_width_px=_ema(self.geometry.shaft_width_px, detected.shaft_width_px, alpha * 0.5),
            rod_top_y_px=_ema(self.geometry.rod_top_y_px, detected.rod_top_y_px, alpha * 0.35),
            rod_bottom_y_px=_ema(self.geometry.rod_bottom_y_px, detected.rod_bottom_y_px, alpha * 0.35),
            liquid_x0_px=_ema(self.geometry.liquid_x0_px, detected.liquid_x0_px, alpha * 0.35),
            liquid_y0_px=_ema(self.geometry.liquid_y0_px, detected.liquid_y0_px, alpha * 0.35),
            liquid_x1_px=_ema(self.geometry.liquid_x1_px, detected.liquid_x1_px, alpha * 0.35),
            liquid_y1_px=_ema(self.geometry.liquid_y1_px, detected.liquid_y1_px, alpha * 0.35),
            bottle_left_px=_ema(self.geometry.bottle_left_px, detected.bottle_left_px, alpha * 0.25),
            bottle_right_px=_ema(self.geometry.bottle_right_px, detected.bottle_right_px, alpha * 0.25),
            bottle_bottom_y_px=_ema(self.geometry.bottle_bottom_y_px, detected.bottle_bottom_y_px, alpha * 0.25),
            frame_width=self.geometry.frame_width,
            frame_height=self.geometry.frame_height,
            rod_axis_score=_ema(self.geometry.rod_axis_score, detected.rod_axis_score, alpha),
            liquid_ellipse_score=_ema(self.geometry.liquid_ellipse_score, detected.liquid_ellipse_score, alpha * 0.35),
            temporal_stability_score=self.geometry.temporal_stability_score,
            bottle_alignment_score=_ema(self.geometry.bottle_alignment_score, detected.bottle_alignment_score, alpha),
            roi_sanity_score=self.geometry.roi_sanity_score,
            bottle_detected=self.geometry.bottle_detected or detected.bottle_detected,
        )
        sanity = _roi_sanity_score(_roi_set_from_geometry(self.geometry, self.config, force_valid=True))
        self.geometry = replace(self.geometry, roi_sanity_score=sanity)
        return self.rois()

    def update_liquid(self, frame_rgb: np.ndarray) -> RoiSet:
        detected = _detect_single_geometry(_as_rgb_uint8(frame_rgb), self.config)
        if detected.liquid_ellipse_score < max(0.50, self.geometry.liquid_ellipse_score * 0.65):
            return self.rois()
        alpha = _clamp01(self.config.dynamic_roi_ema_alpha)
        if alpha <= 0.0:
            return self.rois()
        self.geometry = RoiGeometry(
            shaft_x_px=self.geometry.shaft_x_px,
            shaft_width_px=self.geometry.shaft_width_px,
            rod_top_y_px=self.geometry.rod_top_y_px,
            rod_bottom_y_px=self.geometry.rod_bottom_y_px,
            liquid_x0_px=_ema(self.geometry.liquid_x0_px, detected.liquid_x0_px, alpha),
            liquid_y0_px=_ema(self.geometry.liquid_y0_px, detected.liquid_y0_px, alpha),
            liquid_x1_px=_ema(self.geometry.liquid_x1_px, detected.liquid_x1_px, alpha),
            liquid_y1_px=_ema(self.geometry.liquid_y1_px, detected.liquid_y1_px, alpha),
            bottle_left_px=self.geometry.bottle_left_px,
            bottle_right_px=self.geometry.bottle_right_px,
            bottle_bottom_y_px=self.geometry.bottle_bottom_y_px,
            frame_width=self.geometry.frame_width,
            frame_height=self.geometry.frame_height,
            rod_axis_score=self.geometry.rod_axis_score,
            liquid_ellipse_score=_ema(self.geometry.liquid_ellipse_score, detected.liquid_ellipse_score, alpha),
            temporal_stability_score=self.geometry.temporal_stability_score,
            bottle_alignment_score=self.geometry.bottle_alignment_score,
            roi_sanity_score=self.geometry.roi_sanity_score,
            bottle_detected=self.geometry.bottle_detected,
        )
        sanity = _roi_sanity_score(_roi_set_from_geometry(self.geometry, self.config, force_valid=True))
        self.geometry = replace(self.geometry, roi_sanity_score=sanity)
        return self.rois()

    def rois(self) -> RoiSet:
        return _roi_set_from_geometry(self.geometry, self.config)


@dataclass
class GelClimbDetector:
    config: DetectionConfig = DEFAULT_CONFIG
    _window: Deque[tuple[float, FrameMetrics]] = field(default_factory=deque)
    _metric_window: Deque[tuple[float, FrameMetrics]] = field(default_factory=deque)
    _calibration_frames: list[np.ndarray] = field(default_factory=list)
    _roi_tracker: DynamicRoiTracker | None = None
    _baseline: BaselineModel | None = None
    _locked_result: DetectionResult | None = None
    _last_roi_update_sec: float | None = None

    def update(self, frame_rgb: np.ndarray, timestamp_sec: float) -> DetectionResult:
        if self._locked_result is not None:
            return self._locked_result
        mode = self.config.roi_mode.lower()
        if mode == "fixed":
            return self._update_stable_state(detect_frame(frame_rgb, self.config), float(timestamp_sec))
        if mode not in {"auto", "dynamic"}:
            raise ValueError(f"unsupported roi_mode: {self.config.roi_mode!r}")
        return self._update_dynamic(frame_rgb, float(timestamp_sec))

    def _update_dynamic(self, frame_rgb: np.ndarray, timestamp_sec: float) -> DetectionResult:
        frame = _as_rgb_uint8(frame_rgb)
        if self._roi_tracker is None or self._baseline is None:
            self._calibration_frames.append(frame.copy())
            if timestamp_sec < self.config.calibration_duration_sec:
                return _liquid_result(
                    0.0,
                    {
                        "roi_source": "calibrating",
                        "roi_valid": False,
                        "roi_quality": 0.0,
                        "calibration_duration_sec": self.config.calibration_duration_sec,
                        "calibration_observed_sec": round(timestamp_sec, 3),
                        "calibration_frame_count": len(self._calibration_frames),
                    },
                )
            if self._roi_tracker is None:
                self._roi_tracker = DynamicRoiTracker.calibrate(self._roi_calibration_frames(), self.config)
                self._last_roi_update_sec = None
            assert self._roi_tracker is not None
            baseline_duration_sec = max(self.config.baseline_duration_sec, self.config.calibration_duration_sec)
            if timestamp_sec < baseline_duration_sec:
                rois = self._dynamic_rois_for_frame(frame, timestamp_sec)
                metrics = detect_frame(frame, self.config, rois=rois)
                return _liquid_result(
                    0.0,
                    {
                        **metrics.to_dict(),
                        "baseline_source": "collecting",
                        "baseline_duration_sec": baseline_duration_sec,
                        "baseline_observed_sec": round(timestamp_sec, 3),
                        "baseline_frame_count": len(self._calibration_frames),
                    },
                )
            self._calibrate_dynamic()

        assert self._roi_tracker is not None
        assert self._baseline is not None
        rois = self._dynamic_rois_for_frame(frame, timestamp_sec)
        raw_metrics = detect_frame(frame, self.config, rois=rois)
        self._metric_window.append((timestamp_sec, raw_metrics))
        self._drop_old_metrics(timestamp_sec)
        metrics = _smooth_metrics([m for _, m in self._metric_window], raw_metrics)
        metrics = _apply_baseline(metrics, self._baseline, self.config)
        metrics = replace(
            metrics,
            is_final_candidate=_dynamic_final_candidate(metrics, self._baseline, self.config, timestamp_sec),
        )
        return self._update_stable_state(metrics, timestamp_sec)

    def _calibrate_dynamic(self) -> None:
        if self._roi_tracker is None:
            self._roi_tracker = DynamicRoiTracker.calibrate(self._roi_calibration_frames(), self.config)
            self._last_roi_update_sec = None
        rois = self._roi_tracker.rois()
        baseline_metrics = [detect_frame(frame, self.config, rois=rois) for frame in self._calibration_frames]
        self._baseline = _build_baseline(baseline_metrics, rois, self.config)

    def _roi_calibration_frames(self) -> list[np.ndarray]:
        if self.config.calibration_duration_sec < 10.0 or len(self._calibration_frames) < 9:
            return self._calibration_frames
        tail_count = max(3, len(self._calibration_frames) // 3)
        return self._calibration_frames[-tail_count:]

    def _dynamic_rois_for_frame(self, frame: np.ndarray, timestamp_sec: float) -> RoiSet:
        assert self._roi_tracker is not None
        return self._roi_tracker.rois()

    def _update_stable_state(self, metrics: FrameMetrics, timestamp_sec: float) -> DetectionResult:
        self._window.append((timestamp_sec, metrics))
        self._drop_old(timestamp_sec)
        stable_duration = self._stable_duration()
        candidate_ratio = self._candidate_ratio()
        evidence = {
            **metrics.to_dict(),
            "stable_duration_sec": round(stable_duration, 3),
            "stable_candidate_ratio": round(candidate_ratio, 4),
        }
        confidence = _confidence(metrics, self.config, candidate_ratio)
        is_stable_final = (
            stable_duration + self.config.window_time_tolerance_sec >= self.config.stable_duration_sec
            and candidate_ratio >= self.config.stable_min_ratio
        )
        if is_stable_final:
            result = DetectionResult(
                state=FINAL_GEL_ROD_CLIMBING,
                alert=True,
                transition_time_sec=round(timestamp_sec, 3),
                confidence=confidence,
                evidence=evidence,
            )
            self._locked_result = result
            return result
        return _liquid_result(confidence, evidence)

    def _drop_old(self, timestamp_sec: float) -> None:
        keep_since = timestamp_sec - self.config.stable_duration_sec
        while self._window and self._window[0][0] < keep_since:
            self._window.popleft()

    def _drop_old_metrics(self, timestamp_sec: float) -> None:
        keep_since = timestamp_sec - self.config.metric_smooth_sec
        while self._metric_window and self._metric_window[0][0] < keep_since:
            self._metric_window.popleft()

    def _stable_duration(self) -> float:
        if len(self._window) < 2:
            return 0.0
        return max(0.0, self._window[-1][0] - self._window[0][0])

    def _candidate_ratio(self) -> float:
        if not self._window:
            return 0.0
        return sum(1 for _, metrics in self._window if metrics.is_final_candidate) / len(self._window)


def detect_frame(
    frame_rgb: np.ndarray,
    config: DetectionConfig = DEFAULT_CONFIG,
    rois: RoiSet | None = None,
) -> FrameMetrics:
    frame = _as_rgb_uint8(frame_rgb)
    height, width = frame.shape[:2]
    rois = rois or _fixed_roi_set(config)
    white_mask = _white_gel_mask(frame, config)
    material_mask = _material_gel_mask(frame, white_mask, config)
    orange_mask = _orange_mature_mask(frame, material_mask, config)
    liquid_mask = _crop_bool(white_mask, rois.liquid_roi, width, height)
    liquid_material_mask = _crop_bool(material_mask, rois.liquid_roi, width, height)
    rod_material_mask = _crop_bool(material_mask, rois.rod_roi, width, height)
    rod_orange_mask = _crop_bool(orange_mask, rois.rod_roi, width, height)
    white_coverage = _safe_ratio(liquid_mask.sum(), liquid_mask.size)
    warm_material_coverage = _safe_ratio(liquid_material_mask.sum(), liquid_material_mask.size)
    rod_warm_material_ratio = _safe_ratio(rod_material_mask.sum(), rod_material_mask.size)
    rod_orange_maturity_ratio = _safe_ratio(rod_orange_mask.sum(), rod_orange_mask.size)
    sparse_hole_ratio = 0.0
    if rois.source == "fixed":
        sparse_frame = _crop_rgb(frame, rois.sparse_roi, width, height)
        sparse_white = _crop_bool(white_mask, rois.sparse_roi, width, height)
        sparse_hole_ratio = _sparse_hole_ratio(sparse_frame, sparse_white, config)
    rod_wrap_height, rod_wrap_ratio, connected_area_ratio = _rod_wrap_stats(
        white_mask=white_mask,
        frame_width=width,
        frame_height=height,
        rois=rois,
        config=config,
    )
    _, _, material_connected_area_ratio = _rod_wrap_stats(
        white_mask=material_mask,
        frame_width=width,
        frame_height=height,
        rois=rois,
        config=config,
    )
    rod_attachment = _rod_bottom_attachment_metrics(
        white_mask=white_mask,
        frame_width=width,
        frame_height=height,
        rois=rois,
        config=config,
    )
    if rois.source == "fixed":
        rod_connection_score = rod_wrap_height >= config.rod_wrap_height_min_px
        is_final_candidate = white_coverage >= config.white_coverage_min and rod_connection_score
    else:
        rod_connection_score = (
            (
                rod_attachment.score >= config.rod_bottom_attachment_score_min
                and connected_area_ratio >= config.connected_area_ratio_min
            )
            or (
                rod_warm_material_ratio >= config.rod_warm_material_ratio_min
                and material_connected_area_ratio >= config.warm_connected_area_ratio_min
            )
        )
        is_final_candidate = False
    return FrameMetrics(
        white_coverage=white_coverage,
        sparse_hole_ratio=sparse_hole_ratio,
        rod_wrap_height_px=rod_wrap_height,
        rod_wrap_ratio=rod_wrap_ratio,
        connected_area_ratio=connected_area_ratio,
        rod_connection_score=rod_connection_score,
        is_final_candidate=is_final_candidate,
        rod_bottom_attachment_score=rod_attachment.score,
        rod_bottom_density=rod_attachment.bottom_density,
        rod_top_density=rod_attachment.top_density,
        rod_vertical_contrast=rod_attachment.vertical_contrast,
        rod_side_bottom_density=rod_attachment.side_bottom_density,
        rod_flank_bottom_density=rod_attachment.flank_bottom_density,
        rod_local_contrast=rod_attachment.local_contrast,
        warm_material_coverage=warm_material_coverage,
        rod_warm_material_ratio=rod_warm_material_ratio,
        rod_orange_maturity_ratio=rod_orange_maturity_ratio,
        material_connected_area_ratio=material_connected_area_ratio,
        roi_source=rois.source,
        roi_valid=rois.valid,
        roi_quality=rois.quality,
        roi_failure_reason=rois.failure_reason,
        rois=rois.rects_dict(),
        roi_scores=rois.scores,
    )


def _build_baseline(metrics: list[FrameMetrics], rois: RoiSet, config: DetectionConfig) -> BaselineModel:
    baseline = BaselineModel(
        rod_wrap_ratio=_median([m.rod_wrap_ratio for m in metrics]),
        rod_wrap_iqr=_iqr([m.rod_wrap_ratio for m in metrics]),
        connected_area_ratio=_median([m.connected_area_ratio for m in metrics]),
        connected_area_iqr=_iqr([m.connected_area_ratio for m in metrics]),
        white_coverage=_median([m.white_coverage for m in metrics]),
        white_coverage_iqr=_iqr([m.white_coverage for m in metrics]),
        rod_top_density=_median([m.rod_top_density for m in metrics]),
        rod_top_density_iqr=_iqr([m.rod_top_density for m in metrics]),
        rod_orange_maturity_ratio=_median([m.rod_orange_maturity_ratio for m in metrics]),
        rod_orange_maturity_iqr=_iqr([m.rod_orange_maturity_ratio for m in metrics]),
        initial_final_like=False,
    )
    return baseline


def _apply_baseline(metrics: FrameMetrics, baseline: BaselineModel, config: DetectionConfig) -> FrameMetrics:
    rod_top_density_delta = metrics.rod_top_density - baseline.rod_top_density
    rod_orange_maturity_delta = metrics.rod_orange_maturity_ratio - baseline.rod_orange_maturity_ratio
    return replace(
        metrics,
        baseline=baseline,
        rod_wrap_delta_ratio=metrics.rod_wrap_ratio - baseline.rod_wrap_ratio,
        connected_area_delta_ratio=metrics.connected_area_ratio - baseline.connected_area_ratio,
        white_coverage_delta=metrics.white_coverage - baseline.white_coverage,
        rod_top_density_delta=rod_top_density_delta,
        rod_shape_progress_score=_rod_shape_progress_score(metrics, baseline, config),
        rod_orange_maturity_delta=rod_orange_maturity_delta,
        orange_material_path_score=_orange_material_path_score(metrics, baseline, config),
    )


def _rod_shape_progress_score(metrics: FrameMetrics, baseline: BaselineModel, config: DetectionConfig) -> float:
    vertical_score = _clamp01(
        metrics.rod_vertical_contrast / max(1e-9, config.rod_vertical_contrast_min)
    )
    top_density_delta = metrics.rod_top_density - baseline.rod_top_density
    top_progress_score = _clamp01(
        top_density_delta / max(1e-9, config.rod_top_density_delta_min)
    )
    return max(vertical_score, top_progress_score)


def _orange_material_path_score(metrics: FrameMetrics, baseline: BaselineModel, config: DetectionConfig) -> float:
    orange_delta = metrics.rod_orange_maturity_ratio - baseline.rod_orange_maturity_ratio
    scores = (
        metrics.warm_material_coverage / max(1e-9, config.warm_material_coverage_min),
        metrics.rod_warm_material_ratio / max(1e-9, config.rod_warm_material_ratio_min),
        metrics.rod_orange_maturity_ratio / max(1e-9, config.rod_orange_maturity_ratio_min),
        orange_delta / max(1e-9, config.rod_orange_maturity_delta_min),
        metrics.material_connected_area_ratio / max(1e-9, config.warm_connected_area_ratio_min),
    )
    return min(_clamp01(score) for score in scores)


def _dynamic_final_candidate(
    metrics: FrameMetrics,
    baseline: BaselineModel,
    config: DetectionConfig,
    timestamp_sec: float,
):
    if (not metrics.roi_valid) or metrics.roi_quality < config.roi_quality_min:
        return False
    white_path = (
        metrics.rod_bottom_attachment_score >= config.rod_bottom_attachment_score_min
        and metrics.rod_shape_progress_score >= config.rod_shape_progress_score_min
        and metrics.rod_top_density >= config.rod_white_path_top_density_min
        and metrics.connected_area_ratio >= config.connected_area_ratio_min
        and metrics.white_coverage >= config.dynamic_white_coverage_min
    )
    orange_path = metrics.orange_material_path_score >= config.orange_material_path_score_min
    return timestamp_sec >= config.dynamic_final_min_elapsed_sec and (white_path or orange_path)


def _smooth_metrics(metrics: list[FrameMetrics], latest: FrameMetrics) -> FrameMetrics:
    if not metrics:
        return latest
    return replace(
        latest,
        white_coverage=_median([m.white_coverage for m in metrics]),
        rod_wrap_height_px=int(round(_median([m.rod_wrap_height_px for m in metrics]))),
        rod_wrap_ratio=_median([m.rod_wrap_ratio for m in metrics]),
        connected_area_ratio=_median([m.connected_area_ratio for m in metrics]),
        rod_bottom_attachment_score=_median([m.rod_bottom_attachment_score for m in metrics]),
        rod_bottom_density=_median([m.rod_bottom_density for m in metrics]),
        rod_top_density=_median([m.rod_top_density for m in metrics]),
        rod_vertical_contrast=_median([m.rod_vertical_contrast for m in metrics]),
        rod_side_bottom_density=_median([m.rod_side_bottom_density for m in metrics]),
        rod_flank_bottom_density=_median([m.rod_flank_bottom_density for m in metrics]),
        rod_local_contrast=_median([m.rod_local_contrast for m in metrics]),
        warm_material_coverage=_median([m.warm_material_coverage for m in metrics]),
        rod_warm_material_ratio=_median([m.rod_warm_material_ratio for m in metrics]),
        rod_orange_maturity_ratio=_median([m.rod_orange_maturity_ratio for m in metrics]),
        material_connected_area_ratio=_median([m.material_connected_area_ratio for m in metrics]),
        roi_quality=_median([m.roi_quality for m in metrics]),
    )


def _fixed_roi_set(config: DetectionConfig) -> RoiSet:
    return RoiSet(
        liquid_roi=config.liquid_roi,
        sparse_roi=config.sparse_roi,
        rod_roi=config.rod_roi,
        shaft_core_exclusion=config.shaft_core_exclusion,
        source="fixed",
        valid=True,
        quality=1.0,
    )


def _detect_single_geometry(frame: np.ndarray, config: DetectionConfig) -> RoiGeometry:
    height, width = frame.shape[:2]
    cv2 = _cv2_module()
    gray = _rgb_to_gray(frame)
    if cv2 is not None:
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1]
        value = hsv[:, :, 2]
        bright_mask = ((value >= 125) & (sat <= 95)).astype(np.uint8)
        edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
    else:
        bright_mask = (gray >= 145).astype(np.uint8)
        edges = _simple_edges(gray)

    liquid_x0, liquid_y0, liquid_x1, liquid_y1, liquid_ellipse_score = _detect_liquid_ellipse_bounds(
        frame,
        config,
    )
    shaft_x, shaft_width, rod_column_score = _detect_liquid_guided_rod_axis(
        bright_mask=bright_mask,
        edges=edges,
        liquid_bounds=(liquid_x0, liquid_y0, liquid_x1, liquid_y1),
        width=width,
        height=height,
    )
    shaft_x_i = int(round(_clamp(shaft_x, 0, width - 1)))

    band_half = int(max(6, shaft_width * 1.8))
    bx0 = max(0, shaft_x_i - band_half)
    bx1 = min(width, shaft_x_i + band_half + 1)
    row_score = bright_mask[:, bx0:bx1].mean(axis=1) if bx1 > bx0 else bright_mask.mean(axis=1)
    active_rows = np.flatnonzero(row_score >= max(0.10, float(np.percentile(row_score, 80)) * 0.45))
    if active_rows.size:
        rod_top = float(np.percentile(active_rows, 5))
        rod_bottom = float(np.percentile(active_rows, 95))
    else:
        liquid_height = max(1.0, liquid_y1 - liquid_y0)
        rod_top = max(0.0, liquid_y0 - liquid_height * 1.30)
        rod_bottom = min(height, liquid_y1)

    bottle_left, bottle_right, bottle_detected = _detect_bottle_bounds(edges, shaft_x_i, width, height)
    bottle_bottom = height * 0.95
    bottle_alignment = _bottle_alignment_score(bottle_left, bottle_right, shaft_x, width, bottle_detected)
    length_score = _clamp01((rod_bottom - rod_top) / max(1.0, height * 0.35))
    rod_axis_score = _clamp01(0.75 * rod_column_score + 0.25 * length_score)
    return RoiGeometry(
        shaft_x_px=shaft_x,
        shaft_width_px=shaft_width,
        rod_top_y_px=rod_top,
        rod_bottom_y_px=rod_bottom,
        liquid_x0_px=liquid_x0,
        liquid_y0_px=liquid_y0,
        liquid_x1_px=liquid_x1,
        liquid_y1_px=liquid_y1,
        bottle_left_px=bottle_left,
        bottle_right_px=bottle_right,
        bottle_bottom_y_px=bottle_bottom,
        frame_width=width,
        frame_height=height,
        rod_axis_score=rod_axis_score,
        liquid_ellipse_score=liquid_ellipse_score,
        temporal_stability_score=1.0,
        bottle_alignment_score=bottle_alignment,
        roi_sanity_score=0.0,
        bottle_detected=bottle_detected,
    )


def _detect_liquid_guided_rod_axis(
    bright_mask: np.ndarray,
    edges: np.ndarray,
    liquid_bounds: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[float, float, float]:
    liquid_x0, liquid_y0, liquid_x1, liquid_y1 = liquid_bounds
    liquid_center_x = 0.5 * (liquid_x0 + liquid_x1)
    liquid_height = max(1.0, liquid_y1 - liquid_y0)
    fallback_width = float(_clamp(width * 0.012, 6, max(8, width * 0.05)))
    search_x0 = int(round(_clamp(liquid_x0 - width * 0.08, 0, width - 2)))
    search_x1 = int(round(_clamp(liquid_x1 + width * 0.20, search_x0 + 2, width)))
    search_y0 = int(round(_clamp(liquid_y0 - liquid_height * 2.40, 0, height - 2)))
    search_y1 = int(round(_clamp(liquid_y0 - liquid_height * 0.20, search_y0 + 2, height)))
    if search_y1 <= search_y0 + 4 or search_x1 <= search_x0 + 4:
        return liquid_center_x, fallback_width, 0.0

    edge_crop = (edges[search_y0:search_y1, search_x0:search_x1] > 0).astype(float)
    bright_crop = bright_mask[search_y0:search_y1, search_x0:search_x1].astype(float)
    if edge_crop.size == 0 or bright_crop.size == 0:
        return liquid_center_x, fallback_width, 0.0

    column_score = 0.65 * edge_crop.mean(axis=0) + 0.35 * bright_crop.mean(axis=0)
    column_score = _smooth_1d(column_score, max(9, width // 160))
    if column_score.size == 0 or float(column_score.max()) <= 0.0:
        return liquid_center_x, fallback_width, 0.0

    xs = (search_x0 + np.arange(column_score.size, dtype=float)) / max(1, width)
    center_prior = np.clip(
        1.0 - np.abs(xs - liquid_center_x / max(1, width)) / 0.28,
        0.20,
        1.0,
    )
    weighted = column_score * center_prior
    best_idx = int(np.argmax(weighted))
    shaft_x = float(search_x0 + best_idx)
    peak = float(column_score[best_idx])
    threshold = max(float(np.percentile(column_score, 80)), peak * 0.50)
    left = best_idx
    right = best_idx
    while left > 0 and column_score[left - 1] >= threshold:
        left -= 1
    while right + 1 < column_score.size and column_score[right + 1] >= threshold:
        right += 1
    shaft_width = float(_clamp(right - left + 1, 6, max(8, width * 0.05)))
    return shaft_x, shaft_width, _clamp01(float(weighted[best_idx]) / 0.35)


def _detect_liquid_ellipse_bounds(frame: np.ndarray, config: DetectionConfig) -> tuple[float, float, float, float, float]:
    height, width = frame.shape[:2]
    fallback = _rect_px(config.liquid_roi, width, height)
    cv2 = _cv2_module()
    if cv2 is None:
        return (*[float(v) for v in fallback], 0.0)

    milk_mask = _milky_liquid_mask(frame, cv2).astype(np.uint8)
    component = _detect_liquid_component_box(frame, milk_mask, width, height, cv2)
    broad = _detect_broad_liquid_ellipse_window(milk_mask, width, height, cv2)
    compact = _detect_compact_liquid_ellipse_window(frame, milk_mask, width, height, cv2)

    if component is not None:
        return component
    if broad is not None and _prefer_broad_liquid_box(broad, compact, width, height):
        return broad
    if compact is not None:
        return compact
    if broad is not None:
        return broad
    return (*[float(v) for v in fallback], 0.0)


def _detect_liquid_component_box(
    frame: np.ndarray,
    milk_mask: np.ndarray,
    width: int,
    height: int,
    cv2,
) -> tuple[float, float, float, float, float] | None:
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    gray = _rgb_to_gray(frame)
    edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
    soft_mask = _soft_milky_liquid_mask(frame, cv2).astype(np.uint8)
    core_mask = milk_mask.astype(np.uint8)
    search_x0 = int(round(width * 0.10))
    search_x1 = int(round(width * 0.90))
    search_y0 = int(round(height * 0.25))
    search_y1 = int(round(height * 0.96))
    if search_x1 <= search_x0 + 4 or search_y1 <= search_y0 + 4:
        return None

    search_mask = soft_mask[search_y0:search_y1, search_x0:search_x1].copy()
    kernel_size = max(3, int(round(width / 240)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    search_mask = cv2.morphologyEx(search_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    search_mask = cv2.morphologyEx(search_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(search_mask, 8)
    if num_labels <= 1:
        return None

    shaft_x, shaft_score = _estimate_global_shaft_axis(frame, edges, width, height, cv2)
    best: tuple[float, int, int, int, int] | None = None
    for label in range(1, num_labels):
        local_x, local_y, local_w, local_h, area = [int(v) for v in stats[label]]
        if area < width * height * 0.001:
            continue
        x0 = search_x0 + local_x
        y0 = search_y0 + local_y
        x1 = x0 + local_w
        y1 = y0 + local_h
        width_norm = local_w / max(1, width)
        height_norm = local_h / max(1, height)
        if not (0.06 <= width_norm <= 0.65 and 0.035 <= height_norm <= 0.45):
            continue
        component_mask = (labels[local_y:local_y + local_h, local_x:local_x + local_w] == label).astype(np.uint8)
        score = _score_liquid_component(
            component_mask,
            area,
            x0,
            y0,
            x1,
            y1,
            width,
            height,
            shaft_x,
            shaft_score,
            hsv,
            edges,
        )
        if best is None or score > best[0]:
            best = (score, x0, y0, x1, y1)

    if best is None or best[0] < 0.52:
        return None

    score, x0, y0, x1, y1 = best
    x0, y0, x1, y1 = _refine_liquid_bounds_from_mask(
        soft_mask,
        core_mask,
        hsv,
        edges,
        x0,
        y0,
        x1,
        y1,
        width,
        height,
        shaft_x=shaft_x,
    )
    if y0 / max(1, height) > 0.56 or y1 / max(1, height) > 0.74:
        return None
    return (float(x0), float(y0), float(x1), float(y1), float(score))


def _score_liquid_component(
    component_mask: np.ndarray,
    area: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    width: int,
    height: int,
    shaft_x: float,
    shaft_score: float,
    hsv: np.ndarray,
    edges: np.ndarray,
) -> float:
    box_area = max(1, (x1 - x0) * (y1 - y0))
    area_norm = area / max(1.0, width * height)
    density = area / box_area
    ellipse_density, corner_density = _ellipse_window_density(component_mask)
    ellipse_gain = _clamp01((ellipse_density - corner_density + 0.12) / 0.40)
    shaft_relation = _shaft_liquid_relation_score(shaft_x, shaft_score, x0, x1, width)
    center_x = 0.5 * (x0 + x1) / max(1, width)
    center_score = _clamp01(1.0 - abs(center_x - 0.50) / 0.34)
    label_penalty = _label_like_penalty(hsv, edges, x0, y0, x1, y1)
    return _clamp01(
        0.22 * _clamp01(area_norm / 0.035)
        + 0.18 * density
        + 0.16 * ellipse_density
        + 0.14 * ellipse_gain
        + 0.14 * shaft_relation
        + 0.08 * center_score
        - 0.20 * label_penalty
    )


def _detect_broad_liquid_ellipse_window(
    milk_mask: np.ndarray,
    width: int,
    height: int,
    cv2,
) -> tuple[float, float, float, float, float] | None:
    search_x0 = int(round(width * 0.18))
    search_x1 = int(round(width * 0.84))
    search_y0 = int(round(height * 0.62))
    search_y1 = int(round(height * 0.995))
    if search_x1 <= search_x0 or search_y1 <= search_y0:
        return None

    search_mask = milk_mask[search_y0:search_y1, search_x0:search_x1]
    kernel_size = max(3, int(round(width / 360)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    search_mask = cv2.morphologyEx(search_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    search_mask = cv2.morphologyEx(search_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    integral = cv2.integral(search_mask)
    search_h, search_w = search_mask.shape
    x_step = max(6, int(round(width * 0.025)))
    y_step = max(6, int(round(height * 0.025)))
    width_ratios = (0.24, 0.28, 0.30, 0.34, 0.38, 0.42, 0.44)
    height_ratios = (0.16, 0.20, 0.23, 0.26, 0.29, 0.30)
    coarse_candidates: list[tuple[float, int, int, int, int, float]] = []

    for window_w in {max(4, int(round(width * ratio))) for ratio in width_ratios}:
        if window_w >= search_w:
            continue
        for window_h in {max(4, int(round(height * ratio))) for ratio in height_ratios}:
            if window_h >= search_h:
                continue
            for local_y0 in range(0, search_h - window_h + 1, y_step):
                local_y1 = local_y0 + window_h
                global_cy = (search_y0 + local_y0 + window_h * 0.5) / max(1, height)
                lower_score = _clamp01(1.0 - abs(global_cy - 0.865) / 0.20)
                if lower_score <= 0.0:
                    continue
                for local_x0 in range(0, search_w - window_w + 1, x_step):
                    local_x1 = local_x0 + window_w
                    area = window_w * window_h
                    white_count = (
                        integral[local_y1, local_x1]
                        - integral[local_y0, local_x1]
                        - integral[local_y1, local_x0]
                        + integral[local_y0, local_x0]
                    )
                    density = float(white_count) / max(1, area)
                    if density < 0.12:
                        continue
                    global_cx = (search_x0 + local_x0 + window_w * 0.5) / max(1, width)
                    aspect = window_w / max(1, window_h)
                    center_score = _clamp01(1.0 - abs(global_cx - 0.50) / 0.28)
                    aspect_score = _clamp01(1.0 - abs(aspect - 2.35) / 1.80)
                    size_score = _clamp01(
                        1.0
                        - (
                            abs(window_w / max(1, width) - 0.34) / 0.20
                            + abs(window_h / max(1, height) - 0.24) / 0.16
                        )
                        * 0.5
                    )
                    coarse_score = (
                        0.36 * density
                        + 0.26 * center_score
                        + 0.18 * lower_score
                        + 0.10 * aspect_score
                        + 0.10 * size_score
                    )
                    coarse_candidates.append((coarse_score, local_x0, local_y0, local_x1, local_y1, density))

    if not coarse_candidates:
        return None

    best: tuple[float, int, int, int, int] | None = None
    for coarse_score, local_x0, local_y0, local_x1, local_y1, density in sorted(coarse_candidates, reverse=True)[:48]:
        candidate = search_mask[local_y0:local_y1, local_x0:local_x1]
        ellipse_density, corner_density = _ellipse_window_density(candidate)
        ellipse_gain = _clamp01((ellipse_density - corner_density + 0.25) / 0.50)
        final_score = _clamp01(
            0.90 * coarse_score
            + 0.06 * ellipse_density
            + 0.04 * ellipse_gain
        )
        if best is None or final_score > best[0]:
            best = (final_score, local_x0, local_y0, local_x1, local_y1)

    if best is None or best[0] < 0.30:
        return None

    score, local_x0, local_y0, local_x1, local_y1 = best
    local_x0, local_y0, local_x1, local_y1 = _refine_liquid_ellipse_box(
        search_mask,
        local_x0,
        local_y0,
        local_x1,
        local_y1,
        width,
        height,
    )
    return (
        float(search_x0 + local_x0),
        float(search_y0 + local_y0),
        float(search_x0 + local_x1),
        float(search_y0 + local_y1),
        float(score),
    )


def _detect_compact_liquid_ellipse_window(
    frame: np.ndarray,
    milk_mask: np.ndarray,
    width: int,
    height: int,
    cv2,
) -> tuple[float, float, float, float, float] | None:
    search_x0 = int(round(width * 0.12))
    search_x1 = int(round(width * 0.88))
    search_y0 = int(round(height * 0.28))
    search_y1 = int(round(height * 0.94))
    if search_x1 <= search_x0 or search_y1 <= search_y0:
        return None

    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    value = hsv[:, :, 2]
    r = frame[:, :, 0].astype(np.int16)
    g = frame[:, :, 1].astype(np.int16)
    b = frame[:, :, 2].astype(np.int16)
    max_ch = np.maximum.reduce([r, g, b])
    min_ch = np.minimum.reduce([r, g, b])
    channel_span = max_ch - min_ch
    core_mask = (
        (value >= 175)
        & (sat <= 85)
        & (r >= 175)
        & (g >= 168)
        & (b >= 125)
        & (channel_span <= 85)
    ).astype(np.uint8)
    gray = _rgb_to_gray(frame)
    edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
    shaft_x, shaft_score = _estimate_global_shaft_axis(frame, edges, width, height, cv2)

    masks = (core_mask, milk_mask)
    kernel_size = max(3, int(round(width / 360)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    candidates: list[tuple[float, int, int, int, int]] = []
    width_ratios = (0.14, 0.18, 0.20, 0.24, 0.28, 0.30, 0.32, 0.34)
    height_ratios = (0.08, 0.10, 0.12, 0.16, 0.18, 0.20)
    x_step = max(6, int(round(width * 0.025)))
    y_step = max(6, int(round(height * 0.025)))

    for base_mask in masks:
        search_mask = base_mask[search_y0:search_y1, search_x0:search_x1].copy()
        search_mask = cv2.morphologyEx(search_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        search_mask = cv2.morphologyEx(search_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        integral = cv2.integral(search_mask)
        search_h, search_w = search_mask.shape
        coarse_candidates: list[tuple[float, int, int, int, int, float, float, float, float, float]] = []

        for window_w in {max(4, int(round(width * ratio))) for ratio in width_ratios}:
            if window_w >= search_w:
                continue
            for window_h in {max(4, int(round(height * ratio))) for ratio in height_ratios}:
                if window_h >= search_h:
                    continue
                for local_y0 in range(0, search_h - window_h + 1, y_step):
                    local_y1 = local_y0 + window_h
                    for local_x0 in range(0, search_w - window_w + 1, x_step):
                        local_x1 = local_x0 + window_w
                        area = window_w * window_h
                        white_count = (
                            integral[local_y1, local_x1]
                            - integral[local_y0, local_x1]
                            - integral[local_y1, local_x0]
                            + integral[local_y0, local_x0]
                        )
                        density = float(white_count) / max(1, area)
                        if density < 0.18:
                            continue
                        global_x0 = search_x0 + local_x0
                        global_y0 = search_y0 + local_y0
                        global_x1 = search_x0 + local_x1
                        global_y1 = search_y0 + local_y1
                        global_cx = (global_x0 + global_x1) * 0.5 / max(1, width)
                        global_cy = (global_y0 + global_y1) * 0.5 / max(1, height)
                        aspect = window_w / max(1, window_h)
                        center_score = _clamp01(1.0 - abs(global_cx - 0.50) / 0.32)
                        aspect_score = _clamp01(1.0 - abs(aspect - 2.0) / 2.20)
                        expected_width = _clamp(0.05 + 0.29 * global_cy, 0.14, 0.34)
                        expected_height = _clamp(0.04 + 0.18 * global_cy, 0.08, 0.22)
                        size_score = _clamp01(
                            1.0
                            - (
                                abs(window_w / max(1, width) - expected_width) / 0.15
                                + abs(window_h / max(1, height) - expected_height) / 0.10
                            )
                            * 0.5
                        )
                        shaft_relation = _shaft_liquid_relation_score(
                            shaft_x,
                            shaft_score,
                            global_x0,
                            global_x1,
                            width,
                        )
                        coarse_score = _clamp01(
                            0.42 * density
                            + 0.22 * size_score
                            + 0.16 * shaft_relation
                            + 0.12 * center_score
                            + 0.08 * aspect_score
                        )
                        coarse_candidates.append(
                            (
                                coarse_score,
                                local_x0,
                                local_y0,
                                local_x1,
                                local_y1,
                                density,
                                size_score,
                                shaft_relation,
                                center_score,
                                aspect_score,
                            )
                        )

        for (
            _coarse_score,
            local_x0,
            local_y0,
            local_x1,
            local_y1,
            density,
            size_score,
            shaft_relation,
            center_score,
            aspect_score,
        ) in sorted(coarse_candidates, reverse=True)[:72]:
            global_x0 = search_x0 + local_x0
            global_y0 = search_y0 + local_y0
            global_x1 = search_x0 + local_x1
            global_y1 = search_y0 + local_y1
            ellipse_density, corner_density = _ellipse_window_density(
                search_mask[local_y0:local_y1, local_x0:local_x1]
            )
            ellipse_gain = _clamp01((ellipse_density - corner_density + 0.12) / 0.40)
            label_penalty = _label_like_penalty(
                hsv,
                edges,
                global_x0,
                global_y0,
                global_x1,
                global_y1,
            )
            score = _clamp01(
                0.27 * density
                + 0.21 * ellipse_density
                + 0.17 * ellipse_gain
                + 0.16 * size_score
                + 0.08 * shaft_relation
                + 0.06 * center_score
                + 0.05 * aspect_score
                - 0.12 * label_penalty
            )
            candidates.append((score, local_x0, local_y0, local_x1, local_y1))

    if not candidates:
        return None

    best = _select_compact_liquid_candidate(candidates, search_y0, width, height)
    if best[0] < 0.35:
        return None

    score, local_x0, local_y0, local_x1, local_y1 = best
    refine_mask = core_mask[search_y0:search_y1, search_x0:search_x1].copy()
    refine_mask = cv2.morphologyEx(refine_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    refine_mask = cv2.morphologyEx(refine_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    local_x0, local_y0, local_x1, local_y1 = _refine_compact_liquid_ellipse_box(
        refine_mask,
        local_x0,
        local_y0,
        local_x1,
        local_y1,
        width,
        height,
    )
    soft_mask = _soft_milky_liquid_mask(frame, cv2).astype(np.uint8)
    refined_global = _refine_liquid_bounds_from_mask(
        soft_mask,
        milk_mask.astype(np.uint8),
        hsv,
        edges,
        search_x0 + local_x0,
        search_y0 + local_y0,
        search_x0 + local_x1,
        search_y0 + local_y1,
        width,
        height,
        shaft_x=shaft_x,
    )
    global_x0, global_y0, global_x1, global_y1 = refined_global
    return (
        float(global_x0),
        float(global_y0),
        float(global_x1),
        float(global_y1),
        float(score),
    )


def _estimate_global_shaft_axis(
    frame: np.ndarray,
    edges: np.ndarray,
    width: int,
    height: int,
    cv2,
) -> tuple[float, float]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    value = hsv[:, :, 2]
    bright_mask = ((value >= 125) & (sat <= 95)).astype(np.uint8)
    y0 = int(round(height * 0.08))
    y1 = int(round(height * 0.62))
    x0 = int(round(width * 0.18))
    x1 = int(round(width * 0.82))
    if y1 <= y0 + 2 or x1 <= x0 + 2:
        return width * 0.5, 0.0

    bright_col = bright_mask[y0:y1, x0:x1].mean(axis=0)
    edge_col = (edges[y0:y1, x0:x1] > 0).mean(axis=0)
    column_score = _smooth_1d(0.65 * bright_col + 0.35 * edge_col, max(9, width // 160))
    if column_score.size == 0 or float(column_score.max()) <= 0.0:
        return width * 0.5, 0.0

    best_idx = int(np.argmax(column_score))
    best_x = float(x0 + best_idx)
    peak = float(column_score[best_idx])
    contrast = peak - float(np.median(column_score))
    score = _clamp01(contrast / 0.12)
    hough_x, hough_score = _hough_shaft_x(edges, width, height, best_x)
    if hough_x is not None and hough_score > 0.20:
        best_x = float(0.70 * hough_x + 0.30 * best_x)
        score = max(score, hough_score)
    return best_x, score


def _shaft_liquid_relation_score(
    shaft_x: float,
    shaft_score: float,
    x0: int,
    x1: int,
    frame_width: int,
) -> float:
    window_width = max(1.0, float(x1 - x0))
    center_x = 0.5 * (x0 + x1)
    if x0 <= shaft_x <= x1:
        overlap_score = 1.0
    else:
        distance = min(abs(shaft_x - x0), abs(shaft_x - x1))
        overlap_score = _clamp01(1.0 - distance / max(1.0, window_width * 0.75))
    center_score = _clamp01(1.0 - abs(center_x - shaft_x) / max(1.0, window_width * 0.75))
    confidence = 0.50 + 0.50 * _clamp01(shaft_score)
    return _clamp01(confidence * (0.65 * overlap_score + 0.35 * center_score))


def _label_like_penalty(
    hsv: np.ndarray,
    edges: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> float:
    crop_hsv = hsv[y0:y1, x0:x1]
    crop_edges = edges[y0:y1, x0:x1]
    if crop_hsv.size == 0 or crop_edges.size == 0:
        return 0.0
    hue = crop_hsv[:, :, 0]
    sat = crop_hsv[:, :, 1]
    value = crop_hsv[:, :, 2]
    red_or_yellow = (hue <= 15) | ((hue >= 18) & (hue <= 45)) | (hue >= 165)
    colored_density = float(((sat >= 80) & (value >= 90) & red_or_yellow).mean())
    edge_density = float((crop_edges > 0).mean())
    color_score = _clamp01((colored_density - 0.025) / 0.12)
    edge_score = _clamp01((edge_density - 0.055) / 0.16)
    return _clamp01(color_score * edge_score)


def _select_compact_liquid_candidate(
    candidates: list[tuple[float, int, int, int, int]],
    search_y0: int,
    frame_width: int,
    frame_height: int,
) -> tuple[float, int, int, int, int]:
    return max(
        candidates,
        key=lambda candidate: (
            candidate[0],
            -abs((search_y0 + candidate[2]) / max(1, frame_height) - 0.46),
            candidate[3] - candidate[1],
        ),
    )


def _refine_compact_liquid_ellipse_box(
    search_mask: np.ndarray,
    local_x0: int,
    local_y0: int,
    local_x1: int,
    local_y1: int,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    crop = search_mask[local_y0:local_y1, local_x0:local_x1]
    if crop.size == 0:
        return local_x0, local_y0, local_x1, local_y1

    column_density = _smooth_1d(crop.mean(axis=0), max(5, int(round(frame_width * 0.01))))
    row_density = _smooth_1d(crop.mean(axis=1), max(5, int(round(frame_height * 0.01))))
    if column_density.size == 0 or row_density.size == 0:
        return local_x0, local_y0, local_x1, local_y1

    column_threshold = max(0.12, float(column_density.max()) * 0.55)
    row_threshold = max(0.12, float(row_density.max()) * 0.50)
    active_columns = np.flatnonzero(column_density >= column_threshold)
    active_rows = np.flatnonzero(row_density >= row_threshold)
    if active_columns.size == 0 or active_rows.size == 0:
        return local_x0, local_y0, local_x1, local_y1

    refined_x0 = local_x0 + int(active_columns.min()) - int(round(frame_width * 0.012))
    refined_x1 = local_x0 + int(active_columns.max()) + 1 + int(round(frame_width * 0.012))
    refined_y0 = min(local_y0, local_y0 + int(active_rows.min()) - int(round(frame_height * 0.006)))
    refined_y1 = min(local_y1, local_y0 + int(active_rows.max()) + 1 + int(round(frame_height * 0.004)))

    search_h, search_w = search_mask.shape
    refined_x0 = max(0, refined_x0)
    refined_x1 = min(search_w, refined_x1)
    refined_y0 = max(0, refined_y0)
    refined_y1 = min(search_h, refined_y1)

    min_width = max(2, int(round(frame_width * 0.15)))
    min_height = max(2, int(round(frame_height * 0.10)))
    if refined_x1 - refined_x0 < min_width or refined_y1 - refined_y0 < min_height:
        return local_x0, local_y0, local_x1, local_y1
    return refined_x0, refined_y0, refined_x1, refined_y1


def _prefer_broad_liquid_box(
    broad: tuple[float, float, float, float, float],
    compact: tuple[float, float, float, float, float] | None,
    width: int,
    height: int,
) -> bool:
    broad_x0, broad_y0, broad_x1, broad_y1, broad_score = broad
    broad_left = broad_x0 / max(1, width)
    broad_right = broad_x1 / max(1, width)
    broad_bottom = broad_y1 / max(1, height)
    broad_width = (broad_x1 - broad_x0) / max(1, width)
    broad_height = (broad_y1 - broad_y0) / max(1, height)
    compact_score = 0.0 if compact is None else compact[4]

    return (
        broad_score >= max(0.55, compact_score * 0.80)
        and 0.42 <= broad_left <= 0.58
        and 0.62 <= broad_right <= 0.80
        and broad_bottom >= 0.94
        and 0.20 <= broad_width <= 0.36
        and 0.18 <= broad_height <= 0.30
    )


def _refine_liquid_ellipse_box(
    search_mask: np.ndarray,
    local_x0: int,
    local_y0: int,
    local_x1: int,
    local_y1: int,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    search_h, search_w = search_mask.shape
    xlo = max(0, local_x0 - int(round(frame_width * 0.05)))
    xhi = min(search_w, local_x1 + int(round(frame_width * 0.10)))
    y0 = max(0, local_y0)
    y1 = min(search_h, local_y1 + int(round(frame_height * 0.04)))
    if xhi <= xlo or y1 <= y0:
        return local_x0, local_y0, local_x1, local_y1

    column_density = search_mask[y0:y1, xlo:xhi].mean(axis=0)
    smoothed = _smooth_1d(column_density, 31)
    threshold = max(0.10, float(smoothed.max()) * 0.25)
    active = np.flatnonzero(smoothed >= threshold)
    if active.size == 0:
        return local_x0, local_y0, local_x1, local_y1

    refined_x0 = xlo + int(active.min())
    refined_x1 = xlo + int(active.max()) + 1
    refined_x0 = max(0, refined_x0 - int(round(frame_width * 0.01)))
    refined_x1 = min(search_w, refined_x1 + int(round(frame_width * 0.025)))
    min_width = max(2, int(round(frame_width * 0.12)))
    if refined_x1 - refined_x0 < min_width:
        return local_x0, local_y0, local_x1, local_y1
    return refined_x0, y0, refined_x1, y1


def _milky_liquid_mask(frame: np.ndarray, cv2) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    value = hsv[:, :, 2]
    r = frame[:, :, 0].astype(np.int16)
    g = frame[:, :, 1].astype(np.int16)
    b = frame[:, :, 2].astype(np.int16)
    max_ch = np.maximum.reduce([r, g, b])
    min_ch = np.minimum.reduce([r, g, b])
    channel_span = max_ch - min_ch
    return (
        (value >= 145)
        & (sat <= 120)
        & (r >= 150)
        & (g >= 145)
        & (b >= 105)
        & (channel_span <= 110)
    )


def _soft_milky_liquid_mask(frame: np.ndarray, cv2) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    value = hsv[:, :, 2]
    r = frame[:, :, 0].astype(np.int16)
    g = frame[:, :, 1].astype(np.int16)
    b = frame[:, :, 2].astype(np.int16)
    max_ch = np.maximum.reduce([r, g, b])
    min_ch = np.minimum.reduce([r, g, b])
    channel_span = max_ch - min_ch
    milky = (
        (value >= 105)
        & (r >= 105)
        & (g >= 95)
        & (b >= 65)
        & (sat <= 135)
        & (channel_span <= 125)
    )
    pale_yellow_liquid = (
        (value >= 125)
        & (r >= 125)
        & (g >= 110)
        & (b >= 70)
        & (hue >= 14)
        & (hue <= 35)
        & (sat <= 150)
        & (channel_span <= 145)
    )
    return milky | pale_yellow_liquid


def _refine_liquid_bounds_from_mask(
    soft_mask: np.ndarray,
    core_mask: np.ndarray,
    hsv: np.ndarray,
    edges: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    frame_width: int,
    frame_height: int,
    shaft_x: float | None = None,
) -> tuple[int, int, int, int]:
    pad_x = int(round(frame_width * 0.035))
    pad_y = int(round(frame_height * 0.055))
    rx0 = max(0, x0 - pad_x)
    rx1 = min(frame_width, x1 + pad_x)
    ry0 = max(0, y0 - pad_y)
    ry1 = min(frame_height, y1 + pad_y)
    if rx1 <= rx0 + 2 or ry1 <= ry0 + 2:
        return x0, y0, x1, y1

    region = soft_mask[ry0:ry1, rx0:rx1].astype(np.uint8)
    core_region = core_mask[ry0:ry1, rx0:rx1].astype(np.uint8)
    combined = ((region > 0) | (core_region > 0)).astype(np.uint8)
    if combined.size == 0 or float(combined.mean()) <= 0.02:
        return x0, y0, x1, y1

    column_density = _smooth_1d(combined.mean(axis=0), max(5, int(round(frame_width * 0.006))))
    row_density = _smooth_1d(combined.mean(axis=1), max(5, int(round(frame_height * 0.006))))
    if column_density.size == 0 or row_density.size == 0:
        return x0, y0, x1, y1

    local_x0 = max(0, x0 - rx0)
    local_x1 = min(rx1 - rx0, x1 - rx0)
    local_y0 = max(0, y0 - ry0)
    local_y1 = min(ry1 - ry0, y1 - ry0)
    col_threshold = max(0.10, float(column_density.max()) * 0.35)
    row_threshold = max(0.10, float(row_density.max()) * 0.35)
    active_cols = _active_span_touching(column_density >= col_threshold, local_x0, local_x1)
    active_rows = _active_span_touching(row_density >= row_threshold, local_y0, local_y1)
    if active_cols is None or active_rows is None:
        return x0, y0, x1, y1

    ax0, ax1 = active_cols
    ay0, ay1 = active_rows
    main_cols = _main_liquid_column_span(
        combined,
        local_y0,
        local_y1,
        ax0,
        ax1,
        shaft_x - rx0 if shaft_x is not None else None,
        frame_width,
    )
    if main_cols is not None:
        ax0, ax1 = main_cols
    refined_x0 = rx0 + ax0 - int(round(frame_width * 0.008))
    refined_x1 = rx0 + ax1 + int(round(frame_width * 0.008))
    refined_y0 = ry0 + ay0 - int(round(frame_height * 0.006))
    refined_y1 = ry0 + ay1 - int(round(frame_height * 0.006))

    refined_x0 = int(_clamp(refined_x0, 0, frame_width - 2))
    refined_x1 = int(_clamp(refined_x1, refined_x0 + 2, frame_width))
    refined_y0 = int(_clamp(refined_y0, 0, frame_height - 2))
    refined_y1 = int(_clamp(refined_y1, refined_y0 + 2, frame_height))
    refined_x0, refined_y0, refined_x1, refined_y1 = _trim_label_like_bottom(
        hsv,
        edges,
        refined_x0,
        refined_y0,
        refined_x1,
        refined_y1,
        frame_height,
    )
    refined_x0, refined_x1 = _trim_low_frame_liquid_width(
        refined_x0,
        refined_y0,
        refined_x1,
        refined_y1,
        frame_width,
        frame_height,
        shaft_x=shaft_x,
    )
    refined_y1 = _trim_horizontal_rim_bottom(
        edges,
        refined_x0,
        refined_y0,
        refined_x1,
        refined_y1,
        frame_width,
        frame_height,
    )
    refined_x0, refined_y0, refined_x1, refined_y1 = _trim_low_frame_bottom_drag(
        refined_x0,
        refined_y0,
        refined_x1,
        refined_y1,
        frame_width,
        frame_height,
    )
    min_width = max(2, int(round(frame_width * 0.08)))
    min_height = max(2, int(round(frame_height * 0.04)))
    if refined_x1 - refined_x0 < min_width or refined_y1 - refined_y0 < min_height:
        return x0, y0, x1, y1
    return refined_x0, refined_y0, refined_x1, refined_y1


def _main_liquid_column_span(
    combined_mask: np.ndarray,
    local_y0: int,
    local_y1: int,
    fallback_x0: int,
    fallback_x1: int,
    shaft_x_local: float | None,
    frame_width: int,
) -> tuple[int, int] | None:
    if combined_mask.size == 0:
        return None
    height, width = combined_mask.shape
    if width <= 2 or height <= 2:
        return None

    local_y0 = int(_clamp(local_y0, 0, height - 1))
    local_y1 = int(_clamp(local_y1, local_y0 + 1, height))
    liquid_h = max(1, local_y1 - local_y0)
    band_y0 = int(_clamp(local_y0 + liquid_h * 0.18, 0, height - 1))
    band_y1 = int(_clamp(local_y0 + liquid_h * 0.72, band_y0 + 1, height))
    band = combined_mask[band_y0:band_y1]
    if band.size == 0:
        return None

    column_density = _smooth_1d(band.mean(axis=0), max(5, int(round(frame_width * 0.006))))
    if column_density.size == 0 or float(column_density.max()) <= 0.0:
        return None
    active_runs = _active_runs(column_density >= max(0.10, float(column_density.max()) * 0.42))
    if not active_runs:
        return None

    fallback_width = max(1, fallback_x1 - fallback_x0)
    best: tuple[float, int, int] | None = None
    min_width = max(3, int(round(frame_width * 0.14)))
    for run_x0, run_x1 in active_runs:
        if run_x1 - run_x0 < min_width:
            continue
        overlap = max(0, min(run_x1, fallback_x1) - max(run_x0, fallback_x0))
        overlap_score = _clamp01(overlap / fallback_width)
        if shaft_x_local is None:
            shaft_score = overlap_score
        elif run_x0 <= shaft_x_local <= run_x1:
            shaft_score = 1.0
        else:
            distance = min(abs(shaft_x_local - run_x0), abs(shaft_x_local - run_x1))
            shaft_score = _clamp01(1.0 - distance / max(1.0, fallback_width * 0.55))
        density_score = float(column_density[run_x0:run_x1].mean())
        width_score = _clamp01((run_x1 - run_x0) / max(1.0, frame_width * 0.18))
        score = 0.42 * density_score + 0.28 * shaft_score + 0.20 * overlap_score + 0.10 * width_score
        if best is None or score > best[0]:
            best = (score, run_x0, run_x1)

    if best is None:
        return None
    _, run_x0, run_x1 = best
    return run_x0, run_x1


def _active_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    active = np.flatnonzero(mask)
    if active.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = int(active[0])
    prev = int(active[0])
    for value in active[1:]:
        value = int(value)
        if value == prev + 1:
            prev = value
            continue
        runs.append((start, prev + 1))
        start = prev = value
    runs.append((start, prev + 1))
    return runs


def _trim_low_frame_liquid_width(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    frame_width: int,
    frame_height: int,
    shaft_x: float | None = None,
) -> tuple[int, int]:
    top_norm = y0 / max(1, frame_height)
    bottom_norm = y1 / max(1, frame_height)
    width_norm = (x1 - x0) / max(1, frame_width)
    if not (0.34 <= top_norm <= 0.74 and bottom_norm >= 0.47 and width_norm > 0.27):
        return x0, x1

    original_center = 0.5 * (x0 + x1)
    if shaft_x is not None and x0 <= shaft_x <= x1:
        shaft_weight = 0.65 if top_norm >= 0.52 else 0.50
        center_x = shaft_weight * shaft_x + (1.0 - shaft_weight) * original_center
    else:
        center_x = original_center

    max_width_norm = 0.26 if top_norm >= 0.52 else 0.28
    max_width = frame_width * max_width_norm
    trimmed_x0 = int(round(center_x - max_width * 0.5))
    trimmed_x1 = int(round(center_x + max_width * 0.5))
    trimmed_x0 = max(x0, int(_clamp(trimmed_x0, 0, frame_width - 2)))
    trimmed_x1 = min(x1, int(_clamp(trimmed_x1, trimmed_x0 + 2, frame_width)))
    if trimmed_x1 - trimmed_x0 < frame_width * 0.12:
        return x0, x1
    return trimmed_x0, trimmed_x1


def _trim_horizontal_rim_bottom(
    edges: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    frame_width: int,
    frame_height: int,
) -> int:
    height = y1 - y0
    top_norm = y0 / max(1, frame_height)
    bottom_norm = y1 / max(1, frame_height)
    height_norm = height / max(1, frame_height)
    if not (0.50 <= top_norm <= 0.74 and bottom_norm >= 0.80 and height_norm >= 0.20):
        return y1

    search_y0 = int(round(y0 + height * 0.46))
    search_y1 = int(round(min(frame_height, y1 + frame_height * 0.035)))
    if search_y1 <= search_y0 + 3 or x1 <= x0 + 3:
        return y1

    edge_crop = (edges[search_y0:search_y1, x0:x1] > 0).astype(float)
    if edge_crop.size == 0:
        return y1
    row_score = _smooth_1d(edge_crop.mean(axis=1), max(7, int(round(frame_height * 0.006))))
    if row_score.size == 0 or float(row_score.max()) <= 0.0:
        return y1

    threshold = max(
        0.075,
        float(np.median(row_score)) + 0.025,
        float(row_score.max()) * 0.72,
    )
    active_rows = np.flatnonzero(row_score >= threshold)
    if active_rows.size == 0:
        return y1

    rim_y = search_y0 + int(active_rows[0])
    trim_pad = int(round(frame_height * 0.004))
    trimmed_y1 = rim_y - trim_pad
    if trimmed_y1 >= y1 - int(round(frame_height * 0.025)):
        return y1
    if trimmed_y1 - y0 < int(round(frame_height * 0.14)):
        return y1
    return int(_clamp(trimmed_y1, y0 + 2, y1))


def _active_span_touching(mask: np.ndarray, local_start: int, local_end: int) -> tuple[int, int] | None:
    active = np.flatnonzero(mask)
    if active.size == 0:
        return None
    local_start = int(_clamp(local_start, 0, len(mask) - 1))
    local_end = int(_clamp(local_end, local_start + 1, len(mask)))
    touched = active[(active >= local_start) & (active < local_end)]
    if touched.size == 0:
        pivot = int(round(0.5 * (local_start + local_end)))
        pivot = int(active[np.argmin(np.abs(active - pivot))])
    else:
        pivot = int(touched[len(touched) // 2])
    left = pivot
    right = pivot
    while left > 0 and mask[left - 1]:
        left -= 1
    while right + 1 < len(mask) and mask[right + 1]:
        right += 1
    return left, right + 1


def _trim_label_like_bottom(
    hsv: np.ndarray,
    edges: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    height = y1 - y0
    if height <= 4:
        return x0, y0, x1, y1
    bottom_start = y0 + int(round(height * 0.72))
    crop_hsv = hsv[bottom_start:y1, x0:x1]
    crop_edges = edges[bottom_start:y1, x0:x1]
    if crop_hsv.size == 0 or crop_edges.size == 0:
        return x0, y0, x1, y1
    hue = crop_hsv[:, :, 0]
    sat = crop_hsv[:, :, 1]
    value = crop_hsv[:, :, 2]
    red_or_yellow = (hue <= 15) | ((hue >= 18) & (hue <= 45)) | (hue >= 165)
    colored_density = float(((sat >= 90) & (value >= 90) & red_or_yellow).mean())
    edge_density = float((crop_edges > 0).mean())
    if colored_density < 0.10 or edge_density < 0.08:
        return x0, y0, x1, y1
    trimmed_y1 = bottom_start + int(round(frame_height * 0.006))
    if trimmed_y1 - y0 < frame_height * 0.04:
        return x0, y0, x1, y1
    return x0, y0, x1, int(_clamp(trimmed_y1, y0 + 2, y1))


def _trim_low_frame_bottom_drag(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    top_norm = y0 / max(1, frame_height)
    bottom_norm = y1 / max(1, frame_height)
    height_norm = (y1 - y0) / max(1, frame_height)
    if not (0.56 <= top_norm <= 0.73 and bottom_norm >= 0.90 and height_norm >= 0.24):
        return x0, y0, x1, y1

    width_norm = (x1 - x0) / max(1, frame_width)
    max_height_norm = _clamp(width_norm * 0.70, 0.14, 0.21)
    trimmed_y1 = y0 + int(round(frame_height * max_height_norm))
    if trimmed_y1 >= y1 - int(round(frame_height * 0.025)):
        return x0, y0, x1, y1
    if trimmed_y1 - y0 < int(round(frame_height * 0.08)):
        return x0, y0, x1, y1
    return x0, y0, x1, int(_clamp(trimmed_y1, y0 + 2, y1))


def _ellipse_window_density(mask: np.ndarray) -> tuple[float, float]:
    height, width = mask.shape
    if height == 0 or width == 0:
        return 0.0, 0.0
    yy, xx = np.ogrid[:height, :width]
    cy = (height - 1) * 0.5
    cx = (width - 1) * 0.5
    ellipse = ((xx - cx) / max(1.0, width * 0.5)) ** 2 + ((yy - cy) / max(1.0, height * 0.5)) ** 2 <= 1.0
    ellipse_density = float(mask[ellipse].mean()) if ellipse.any() else 0.0
    corner = ~ellipse
    corner_density = float(mask[corner].mean()) if corner.any() else 0.0
    return ellipse_density, corner_density


def _hough_shaft_x(edges: np.ndarray, width: int, height: int, expected_x: float) -> tuple[float | None, float]:
    cv2 = _cv2_module()
    if cv2 is None:
        return None, 0.0
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(40, width // 35),
        minLineLength=int(height * 0.25),
        maxLineGap=30,
    )
    if lines is None:
        return None, 0.0
    best_x = None
    best_score = 0.0
    max_allowed_dx = np.tan(np.deg2rad(14.0))
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = [float(v) for v in line]
        dy = abs(y2 - y1)
        dx = abs(x2 - x1)
        if dy < height * 0.20 or dx / max(1.0, dy) > max_allowed_dx:
            continue
        x_mid = 0.5 * (x1 + x2)
        if abs(x_mid - expected_x) > width * 0.18:
            continue
        center_score = _clamp01(1.0 - abs(x_mid - expected_x) / max(1.0, width * 0.12))
        vertical_score = _clamp01(1.0 - dx / max(1.0, dy * max_allowed_dx))
        length_score = _clamp01(dy / max(1.0, height * 0.55))
        score = 0.45 * vertical_score + 0.35 * length_score + 0.20 * center_score
        if score > best_score:
            best_score = score
            best_x = x_mid
    return best_x, _clamp01(best_score)


def _detect_bottle_bounds(edges: np.ndarray, shaft_x: int, width: int, height: int) -> tuple[float, float, bool]:
    y0 = int(round(height * 0.35))
    y1 = int(round(height * 0.98))
    edge_col = (edges[y0:y1] > 0).mean(axis=0) if y1 > y0 else (edges > 0).mean(axis=0)
    edge_col = _smooth_1d(edge_col, max(9, width // 90))
    min_gap = max(30, int(width * 0.04))
    left_start = max(0, int(width * 0.03))
    left_end = max(left_start + 1, shaft_x - min_gap)
    right_start = min(width - 1, shaft_x + min_gap)
    right_end = min(width, int(width * 0.97))
    left = float(left_start + np.argmax(edge_col[left_start:left_end])) if left_end > left_start + 3 else shaft_x - width * 0.18
    right = float(right_start + np.argmax(edge_col[right_start:right_end])) if right_end > right_start + 3 else shaft_x + width * 0.18
    bottle_width = right - left
    peak_ok = bool(
        bottle_width >= width * 0.40
        and bottle_width <= width * 0.78
        and edge_col[int(_clamp(left, 0, width - 1))] >= np.percentile(edge_col, 70)
        and edge_col[int(_clamp(right, 0, width - 1))] >= np.percentile(edge_col, 70)
    )
    if not peak_ok:
        fallback_half = width * 0.24
        left = shaft_x - fallback_half
        right = shaft_x + fallback_half
    return _clamp(left, 0, width - 1), _clamp(right, 1, width), peak_ok


def _roi_set_from_geometry(geometry: RoiGeometry, config: DetectionConfig, force_valid: bool = False) -> RoiSet:
    width = geometry.frame_width
    height = geometry.frame_height
    bottle_left = _clamp(geometry.bottle_left_px, 0, width - 2)
    bottle_right = _clamp(geometry.bottle_right_px, bottle_left + 2, width)
    bottle_bottom = _clamp(geometry.bottle_bottom_y_px, height * 0.55, height)
    liquid_left = _clamp(geometry.liquid_x0_px, 0, width - 2)
    liquid_top = _clamp(geometry.liquid_y0_px, 0, height - 2)
    liquid_right = _clamp(geometry.liquid_x1_px, liquid_left + 2, width)
    liquid_bottom = _clamp(geometry.liquid_y1_px, liquid_top + 2, height)
    liquid_width = max(1.0, liquid_right - liquid_left)
    liquid_height = max(1.0, liquid_bottom - liquid_top)
    liquid_center_x = 0.5 * (liquid_left + liquid_right)
    shaft_x = _clamp(geometry.shaft_x_px, 0, width)
    shaft_gap = max(liquid_left - shaft_x, shaft_x - liquid_right, 0.0)
    shaft_center_ok = (
        geometry.rod_axis_score >= 0.45
        and shaft_gap <= max(liquid_width * 0.35, width * 0.03)
    )
    rod_center_x = shaft_x if shaft_center_ok else liquid_center_x
    rod_roi_width = min(width * 0.18, max(width * 0.11, liquid_width * 0.48))
    rod_y0 = max(0.0, liquid_top - liquid_height * 0.30)
    rod_y1 = liquid_bottom
    sparse_width = liquid_width * 0.45
    sparse_height = liquid_height * 0.55
    sparse_center_x = liquid_center_x
    sparse_cy = liquid_top + liquid_height * 0.58
    liquid_roi = _rect_from_px(
        liquid_left,
        liquid_top,
        liquid_right,
        liquid_bottom,
        width,
        height,
    )
    rod_roi = _rect_from_px(
        rod_center_x - rod_roi_width * 0.5,
        rod_y0,
        rod_center_x + rod_roi_width * 0.5,
        rod_y1,
        width,
        height,
    )
    sparse_roi = _rect_from_px(
        sparse_center_x - sparse_width * 0.5,
        sparse_cy - sparse_height * 0.5,
        sparse_center_x + sparse_width * 0.5,
        sparse_cy + sparse_height * 0.5,
        width,
        height,
    )
    temp_rois = RoiSet(liquid_roi, sparse_roi, rod_roi, None, "dynamic", True, 0.0)
    sanity = _roi_sanity_score(temp_rois)
    quality = _clamp01(
        0.30 * geometry.rod_axis_score
        + 0.20 * geometry.temporal_stability_score
        + 0.15 * geometry.bottle_alignment_score
        + 0.15 * geometry.liquid_ellipse_score
        + 0.20 * sanity
    )
    valid = force_valid or quality >= config.roi_quality_min
    return RoiSet(
        liquid_roi=liquid_roi,
        sparse_roi=sparse_roi,
        rod_roi=rod_roi,
        shaft_core_exclusion=None,
        source="dynamic",
        valid=valid,
        quality=quality,
        failure_reason="" if valid else _roi_failure_reason(geometry, sanity),
        shaft_x=geometry.shaft_x_px / max(1, width),
        shaft_width=geometry.shaft_width_px / max(1, width),
        bottle_left=bottle_left / max(1, width),
        bottle_right=bottle_right / max(1, width),
        bottle_bottom_y=bottle_bottom / max(1, height),
        scores={**geometry.scores, "roi_sanity_score": sanity},
    )


def _roi_sanity_score(rois: RoiSet) -> float:
    checks = []
    rx0, ry0, rx1, ry1 = rois.rod_roi
    lx0, ly0, lx1, ly1 = rois.liquid_roi
    sx0, sy0, sx1, sy1 = rois.sparse_roi
    checks.append(0.04 <= rx1 - rx0 <= 0.28)
    checks.append(0.08 <= ry1 - ry0 <= 0.75)
    checks.append(0.15 <= lx1 - lx0 <= 0.85)
    checks.append(0.06 <= ly1 - ly0 <= 0.50)
    checks.append(0.04 <= sx1 - sx0 <= 0.45)
    checks.append(0.035 <= sy1 - sy0 <= 0.45)
    checks.append(0.28 <= ly0 <= 0.88 and ly1 >= ly0)
    checks.append(sy1 >= sy0 and ry1 >= ry0)
    checks.append(rx0 > 0.01 and rx1 < 0.99 and lx0 > 0.01 and lx1 < 0.99)
    if rois.source == "fixed" and rois.shaft_core_exclusion is not None:
        ex0, ey0, ex1, ey1 = rois.shaft_core_exclusion
        checks.append(rx0 <= ex0 < ex1 <= rx1)
        checks.append(abs(ey0 - ry0) < 0.02 and abs(ey1 - ry1) < 0.02)
    return sum(1 for item in checks if item) / len(checks)


def _roi_failure_reason(geometry: RoiGeometry, sanity: float) -> str:
    scores = {
        "rod_axis": geometry.rod_axis_score,
        "liquid_ellipse": geometry.liquid_ellipse_score,
        "temporal_stability": geometry.temporal_stability_score,
        "bottle_alignment": geometry.bottle_alignment_score,
        "roi_sanity": sanity,
    }
    weakest = min(scores, key=scores.get)
    return f"low_{weakest}_score"


def _bottle_alignment_score(left: float, right: float, shaft_x: float, width: int, detected: bool) -> float:
    bottle_width = max(1.0, right - left)
    inside = 1.0 if left < shaft_x < right else 0.0
    left_width = max(1.0, shaft_x - left)
    right_width = max(1.0, right - shaft_x)
    symmetry = _clamp01(1.0 - abs(left_width - right_width) / max(left_width, right_width))
    width_score = _clamp01((bottle_width - width * 0.16) / max(1.0, width * 0.36))
    detected_bonus = 1.0 if detected else 0.55
    return _clamp01(0.35 * inside + 0.30 * symmetry + 0.20 * width_score + 0.15 * detected_bonus)


def _rod_wrap_stats(
    white_mask: np.ndarray,
    frame_width: int,
    frame_height: int,
    rois: RoiSet,
    config: DetectionConfig,
) -> tuple[int, float, float]:
    rod_mask = _crop_bool(white_mask, rois.rod_roi, frame_width, frame_height).copy()
    if rois.source == "fixed" and rois.shaft_core_exclusion is not None:
        roi_x0, roi_y0, _, _ = _rect_px(rois.rod_roi, frame_width, frame_height)
        excl_x0, excl_y0, excl_x1, excl_y1 = _rect_px(rois.shaft_core_exclusion, frame_width, frame_height)
        x0 = max(0, excl_x0 - roi_x0)
        x1 = min(rod_mask.shape[1], excl_x1 - roi_x0)
        y0 = max(0, excl_y0 - roi_y0)
        y1 = min(rod_mask.shape[0], excl_y1 - roi_y0)
        if x0 < x1 and y0 < y1:
            rod_mask[y0:y1, x0:x1] = False
    if not rod_mask.any():
        return 0, 0.0, 0.0
    seed_rows = max(1, int(round(rod_mask.shape[0] * config.bottom_seed_fraction)))
    seeds = np.zeros_like(rod_mask, dtype=bool)
    seeds[-seed_rows:, :] = rod_mask[-seed_rows:, :]
    connected = _connected_from_seeds(rod_mask, seeds)
    if not connected.any():
        return 0, 0.0, 0.0
    row_fill = connected.mean(axis=1)
    significant_rows = np.flatnonzero(np.greater_equal(row_fill, config.rod_row_fill_min))
    if significant_rows.size == 0:
        return 0, 0.0, _safe_ratio(connected.sum(), connected.size)
    min_y = int(significant_rows.min())
    height_px = int(rod_mask.shape[0] - min_y)
    return height_px, _safe_ratio(height_px, rod_mask.shape[0]), _safe_ratio(connected.sum(), connected.size)


def _rod_bottom_attachment_metrics(
    white_mask: np.ndarray,
    frame_width: int,
    frame_height: int,
    rois: RoiSet,
    config: DetectionConfig,
) -> RodAttachmentMetrics:
    rx0, ry0, rx1, ry1 = _rect_px(rois.rod_roi, frame_width, frame_height)
    rod_mask = white_mask[ry0:ry1, rx0:rx1]
    rod_height, rod_width = rod_mask.shape
    if rod_height == 0 or rod_width == 0:
        return RodAttachmentMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    bottom_fraction = _clamp(config.rod_bottom_band_fraction, 0.01, 1.0)
    top_fraction = _clamp(config.rod_top_band_fraction, 0.01, 1.0)
    bottom_rows = max(1, int(round(rod_height * bottom_fraction)))
    top_rows = max(1, int(round(rod_height * top_fraction)))
    bottom_band = rod_mask[-bottom_rows:]
    top_band = rod_mask[:top_rows]

    bottom_density = _safe_ratio(bottom_band.sum(), bottom_band.size)
    top_density = _safe_ratio(top_band.sum(), top_band.size)
    vertical_contrast = bottom_density - top_density

    side_bottom = bottom_band.copy()
    center_fraction = _clamp(config.rod_center_exclusion_fraction, 0.0, 0.95)
    center_width = int(round(rod_width * center_fraction))
    if center_width > 0:
        center_x0 = max(0, (rod_width - center_width) // 2)
        center_x1 = min(rod_width, center_x0 + center_width)
        side_bottom[:, center_x0:center_x1] = False
    side_bottom_density = _safe_ratio(side_bottom.sum(), side_bottom.size)

    gap_px = max(1, int(round(rod_width * _clamp(config.rod_flank_gap_ratio, 0.0, 2.0))))
    flank_width_px = max(1, int(round(rod_width * _clamp(config.rod_flank_width_ratio, 0.01, 2.0))))
    flank_y0 = ry1 - bottom_rows
    flank_y1 = ry1
    flanks = []
    left_x1 = max(0, rx0 - gap_px)
    left_x0 = max(0, left_x1 - flank_width_px)
    if left_x1 > left_x0:
        flanks.append(white_mask[flank_y0:flank_y1, left_x0:left_x1])
    right_x0 = min(frame_width, rx1 + gap_px)
    right_x1 = min(frame_width, right_x0 + flank_width_px)
    if right_x1 > right_x0:
        flanks.append(white_mask[flank_y0:flank_y1, right_x0:right_x1])
    if flanks:
        flank_bottom = np.concatenate(flanks, axis=1)
        flank_bottom_density = _safe_ratio(flank_bottom.sum(), flank_bottom.size)
    else:
        flank_bottom_density = side_bottom_density

    local_contrast = side_bottom_density - flank_bottom_density
    flank_saturated = flank_bottom_density > _clamp(config.rod_flank_bottom_density_max, 0.0, 1.0)
    vertical_score = 0.0 if flank_saturated else _clamp01(vertical_contrast / 0.55)
    local_score = _clamp01(local_contrast / 0.20)
    score = max(vertical_score, local_score)
    return RodAttachmentMetrics(
        score=score,
        bottom_density=bottom_density,
        top_density=top_density,
        vertical_contrast=vertical_contrast,
        side_bottom_density=side_bottom_density,
        flank_bottom_density=flank_bottom_density,
        local_contrast=local_contrast,
    )


def _white_gel_mask(frame: np.ndarray, config: DetectionConfig) -> np.ndarray:
    r = frame[:, :, 0].astype(np.int16)
    g = frame[:, :, 1].astype(np.int16)
    b = frame[:, :, 2].astype(np.int16)
    max_ch = np.maximum.reduce([r, g, b])
    min_ch = np.minimum.reduce([r, g, b])
    channel_span = max_ch - min_ch
    return (
        (max_ch >= config.white_min_value)
        & (channel_span <= config.white_max_channel_span)
        & (r >= config.white_min_red)
        & (g >= config.white_min_green)
        & (b >= config.white_min_blue)
    )


def _material_gel_mask(frame: np.ndarray, white_mask: np.ndarray, config: DetectionConfig) -> np.ndarray:
    r = frame[:, :, 0].astype(np.int16)
    g = frame[:, :, 1].astype(np.int16)
    b = frame[:, :, 2].astype(np.int16)
    warm_mask = (
        (r >= config.warm_min_red)
        & (g >= config.warm_min_green)
        & (b >= config.warm_min_blue)
        & ((r - b) >= config.warm_min_red_blue_delta)
        & ((g - b) >= config.warm_min_green_blue_delta)
        & ((r - g) <= config.warm_max_red_green_delta)
    )
    return white_mask | warm_mask


def _orange_mature_mask(frame: np.ndarray, material_mask: np.ndarray, config: DetectionConfig) -> np.ndarray:
    r = frame[:, :, 0].astype(np.int16)
    g = frame[:, :, 1].astype(np.int16)
    b = frame[:, :, 2].astype(np.int16)
    max_ch = np.maximum.reduce([r, g, b])
    min_ch = np.minimum.reduce([r, g, b])
    saturation = np.zeros_like(max_ch, dtype=np.float32)
    positive = max_ch > 0
    saturation[positive] = (max_ch[positive] - min_ch[positive]) / max_ch[positive] * 255.0
    return (
        material_mask
        & ((r - b) >= config.orange_min_red_blue_delta)
        & (saturation >= config.orange_min_saturation)
        & (g >= config.warm_min_green)
    )


def _sparse_hole_ratio(sparse_frame: np.ndarray, sparse_white_mask: np.ndarray, config: DetectionConfig) -> float:
    value = sparse_frame.max(axis=2)
    candidate = (~sparse_white_mask) & (value < config.sparse_dark_value)
    stride = max(1, int(config.sparse_component_stride))
    if stride > 1:
        candidate = candidate[::stride, ::stride]
    return _safe_ratio(_largest_component_area(candidate), candidate.size)


def _connected_from_seeds(mask: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    connected = np.zeros_like(mask, dtype=bool)
    stack = [tuple(idx) for idx in np.argwhere(seeds)]
    height, width = mask.shape
    while stack:
        y, x = stack.pop()
        if connected[y, x] or not mask[y, x]:
            continue
        connected[y, x] = True
        if y > 0:
            stack.append((y - 1, x))
        if y + 1 < height:
            stack.append((y + 1, x))
        if x > 0:
            stack.append((y, x - 1))
        if x + 1 < width:
            stack.append((y, x + 1))
    return connected


def _largest_component_area(mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    visited = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    largest = 0
    for start_y, start_x in np.argwhere(mask):
        if visited[start_y, start_x]:
            continue
        area = 0
        stack = [(int(start_y), int(start_x))]
        while stack:
            y, x = stack.pop()
            if visited[y, x] or not mask[y, x]:
                continue
            visited[y, x] = True
            area += 1
            if y > 0:
                stack.append((y - 1, x))
            if y + 1 < height:
                stack.append((y + 1, x))
            if x > 0:
                stack.append((y, x - 1))
            if x + 1 < width:
                stack.append((y, x + 1))
        largest = max(largest, area)
    return largest


def _confidence(metrics: FrameMetrics, config: DetectionConfig, candidate_ratio: float) -> float:
    stable_score = _clamp01(candidate_ratio / max(1e-9, config.stable_min_ratio))
    if metrics.baseline is not None:
        roi_score = _clamp01(metrics.roi_quality / max(1e-9, config.roi_quality_min))
        attachment_score = _clamp01(
            metrics.rod_bottom_attachment_score / max(1e-9, config.rod_bottom_attachment_score_min)
        )
        shape_score = _clamp01(
            metrics.rod_shape_progress_score / max(1e-9, config.rod_shape_progress_score_min)
        )
        top_fill_score = _clamp01(
            metrics.rod_top_density / max(1e-9, config.rod_white_path_top_density_min)
        )
        white_path_score = min(attachment_score, shape_score, top_fill_score)
        orange_path_score = _clamp01(
            metrics.orange_material_path_score / max(1e-9, config.orange_material_path_score_min)
        )
        path_score = max(white_path_score, orange_path_score)
        connected_score = max(
            _clamp01(metrics.connected_area_ratio / max(1e-9, config.connected_area_ratio_min)),
            _clamp01(metrics.material_connected_area_ratio / max(1e-9, config.warm_connected_area_ratio_min)),
        )
        coverage_score = max(
            _clamp01(metrics.white_coverage / max(1e-9, config.dynamic_white_coverage_min)),
            _clamp01(metrics.warm_material_coverage / max(1e-9, config.warm_material_coverage_min)),
        )
        return _clamp01(
            0.25 * roi_score
            + 0.30 * path_score
            + 0.15 * connected_score
            + 0.15 * coverage_score
            + 0.15 * stable_score
        )
    white_score = _clamp01((metrics.white_coverage - 0.45) / max(1e-9, config.white_coverage_min - 0.45))
    sparse_score = _clamp01((0.12 - metrics.sparse_hole_ratio) / max(1e-9, 0.12 - config.sparse_hole_ratio_max))
    rod_score = _clamp01((metrics.rod_wrap_height_px - 50) / max(1e-9, config.rod_wrap_height_min_px - 50))
    return _clamp01(0.3 * white_score + 0.25 * sparse_score + 0.25 * rod_score + 0.2 * stable_score)


def _liquid_result(confidence: float, evidence: dict) -> DetectionResult:
    return DetectionResult(
        state=LIQUID_STIRRING,
        alert=False,
        transition_time_sec=None,
        confidence=confidence,
        evidence=evidence,
    )


def _as_rgb_uint8(frame_rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame_rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("frame_rgb must have shape (height, width, 3)")
    if arr.dtype == np.uint8:
        return arr
    return np.clip(arr, 0, 255).astype(np.uint8)


def _rgb_to_gray(frame: np.ndarray) -> np.ndarray:
    return (0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]).astype(np.uint8)


def _simple_edges(gray: np.ndarray) -> np.ndarray:
    gy = np.abs(np.diff(gray.astype(np.int16), axis=0, prepend=gray[:1].astype(np.int16)))
    gx = np.abs(np.diff(gray.astype(np.int16), axis=1, prepend=gray[:, :1].astype(np.int16)))
    return ((gx + gy) > 40).astype(np.uint8) * 255


def _cv2_module():
    try:
        import cv2  # type: ignore
    except ImportError:
        return None
    return cv2


def _smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window <= 1:
        return values.astype(float)
    if window % 2 == 0:
        window += 1
    return np.convolve(values.astype(float), np.ones(window, dtype=float) / window, mode="same")


def _rect_px(rect: Rect, width: int, height: int) -> tuple[int, int, int, int]:
    x0 = int(round(rect.x0 * width))
    y0 = int(round(rect.y0 * height))
    x1 = int(round(rect.x1 * width))
    y1 = int(round(rect.y1 * height))
    x0 = min(max(x0, 0), width)
    x1 = min(max(x1, 0), width)
    y0 = min(max(y0, 0), height)
    y1 = min(max(y1, 0), height)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"invalid ROI {rect!r} for frame size {width}x{height}")
    return x0, y0, x1, y1


def _rect_from_px(x0: float, y0: float, x1: float, y1: float, width: int, height: int) -> Rect:
    min_w = max(2.0, width * 0.01)
    min_h = max(2.0, height * 0.01)
    x0 = _clamp(x0, 0, width - min_w)
    y0 = _clamp(y0, 0, height - min_h)
    x1 = _clamp(x1, x0 + min_w, width)
    y1 = _clamp(y1, y0 + min_h, height)
    return Rect(x0 / width, y0 / height, x1 / width, y1 / height)


def _crop_rgb(frame: np.ndarray, rect: Rect, width: int, height: int) -> np.ndarray:
    x0, y0, x1, y1 = _rect_px(rect, width, height)
    return frame[y0:y1, x0:x1]


def _crop_bool(mask: np.ndarray, rect: Rect, width: int, height: int) -> np.ndarray:
    x0, y0, x1, y1 = _rect_px(rect, width, height)
    return mask[y0:y1, x0:x1]


def _rect_to_list(rect: Rect) -> list[float]:
    return [round(float(v), 5) for v in rect]


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _median(values) -> float:
    if not values:
        return 0.0
    return float(np.median(np.asarray(values, dtype=float)))


def _iqr(values) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=float)
    return float(np.percentile(arr, 75) - np.percentile(arr, 25))


def _ema(old: float, new: float, alpha: float) -> float:
    alpha = _clamp01(alpha)
    return float(old * (1.0 - alpha) + new * alpha)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))
