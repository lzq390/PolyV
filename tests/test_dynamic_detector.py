from dataclasses import replace
import unittest

import numpy as np

from polyv_detector import FINAL_GEL_ROD_CLIMBING, LIQUID_STIRRING
from polyv_detector.detector import (
    BaselineModel,
    DetectionConfig,
    DynamicRoiTracker,
    FrameMetrics,
    GelClimbDetector,
    Rect,
    RoiSet,
    detect_frame,
)
from polyv_detector.dynamic_detector import _dynamic_final_candidate


def dynamic_config(**kwargs):
    defaults = dict(
        roi_mode="auto",
        calibration_duration_sec=2.0,
        baseline_duration_sec=2.0,
        metric_smooth_sec=1.0,
        stable_duration_sec=3.0,
        stable_min_ratio=0.8,
        window_time_tolerance_sec=0.0,
        roi_quality_min=0.45,
        rod_wrap_ratio_min=0.65,
        rod_wrap_delta_ratio_min=0.25,
        connected_area_ratio_min=0.10,
        connected_area_delta_ratio_min=0.01,
        sparse_component_stride=1,
    )
    defaults.update(kwargs)
    return DetectionConfig(**defaults)


def _paint(frame, rect, color):
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = _rect_px(rect, width, height)
    frame[y0:y1, x0:x1] = color


def _rect_px(rect, width, height):
    x0 = int(round(rect.x0 * width))
    y0 = int(round(rect.y0 * height))
    x1 = int(round(rect.x1 * width))
    y1 = int(round(rect.y1 * height))
    return x0, y0, x1, y1


def synthetic_frame(width=480, height=270, shaft_x=240, gel=False, white_background=False):
    frame = np.full((height, width, 3), (45, 45, 42), dtype=np.uint8)
    bottle_left = int(width * 0.28)
    bottle_right = int(width * 0.72)
    bottle_top = int(height * 0.18)
    bottle_bottom = int(height * 0.95)

    frame[bottle_top:bottle_bottom, bottle_left:bottle_left + 3] = (190, 190, 185)
    frame[bottle_top:bottle_bottom, bottle_right - 3:bottle_right] = (190, 190, 185)
    frame[bottle_bottom - 3:bottle_bottom, bottle_left:bottle_right] = (190, 190, 185)
    frame[int(height * 0.74):bottle_bottom, bottle_left + 12:bottle_right - 12] = (195, 188, 150)

    shaft_half = 3
    frame[int(height * 0.10):bottle_bottom, shaft_x - shaft_half:shaft_x + shaft_half] = (235, 235, 232)

    if white_background:
        frame[int(height * 0.40):int(height * 0.96), int(width * 0.70):int(width * 0.92)] = (235, 235, 232)

    if gel:
        gel_top = int(height * 0.30)
        gel_left = shaft_x - 24
        gel_right = shaft_x + 24
        frame[gel_top:bottle_bottom, gel_left:shaft_x - shaft_half] = (235, 228, 180)
        frame[gel_top:bottle_bottom, shaft_x + shaft_half:gel_right] = (235, 228, 180)
        frame[int(height * 0.68):bottle_bottom, bottle_left + 15:bottle_right - 15] = (235, 228, 180)
    return frame


def synthetic_ellipse_frame(
    width=480,
    height=270,
    shaft_x=240,
    liquid_center_x=0.50,
    liquid_center_y=0.84,
    liquid_radius_x=0.16,
    liquid_radius_y=0.10,
    misleading_white=False,
):
    frame = np.full((height, width, 3), (45, 45, 42), dtype=np.uint8)
    bottle_left = int(width * 0.28)
    bottle_right = int(width * 0.72)
    bottle_top = int(height * 0.18)
    bottle_bottom = int(height * 0.95)

    frame[bottle_top:bottle_bottom, bottle_left:bottle_left + 3] = (190, 190, 185)
    frame[bottle_top:bottle_bottom, bottle_right - 3:bottle_right] = (190, 190, 185)
    frame[bottle_bottom - 3:bottle_bottom, bottle_left:bottle_right] = (190, 190, 185)

    shaft_half = 3
    frame[int(height * 0.10):bottle_bottom, shaft_x - shaft_half:shaft_x + shaft_half] = (235, 235, 232)

    if misleading_white:
        frame[int(height * 0.62):int(height * 0.92), int(width * 0.72):int(width * 0.94)] = (235, 235, 232)

    yy, xx = np.ogrid[:height, :width]
    cx = width * liquid_center_x
    cy = height * liquid_center_y
    rx = width * liquid_radius_x
    ry = height * liquid_radius_y
    ellipse = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
    frame[ellipse] = (235, 228, 180)
    return frame


def synthetic_wide_view_frame(width=272, height=480, shaft_x=None, liquid_center_x=0.52, liquid_radius_x=0.10):
    if shaft_x is None:
        shaft_x = int(width * 0.52)
    frame = np.full((height, width, 3), (70, 70, 66), dtype=np.uint8)

    pot_top = int(height * 0.26)
    frame[pot_top:height, int(width * 0.05):int(width * 0.95)] = (112, 108, 92)
    frame[int(height * 0.30):int(height * 0.34), int(width * 0.05):int(width * 0.95)] = (175, 170, 150)

    bottle_left = int(width * 0.33)
    bottle_right = int(width * 0.68)
    bottle_top = int(height * 0.12)
    bottle_bottom = int(height * 0.62)
    frame[bottle_top:bottle_bottom, bottle_left:bottle_left + 2] = (188, 188, 184)
    frame[bottle_top:bottle_bottom, bottle_right - 2:bottle_right] = (188, 188, 184)
    frame[bottle_bottom - 2:bottle_bottom, bottle_left:bottle_right] = (188, 188, 184)

    shaft_half = max(2, int(width * 0.008))
    frame[int(height * 0.06):int(height * 0.64), shaft_x - shaft_half:shaft_x + shaft_half] = (238, 238, 234)

    yy, xx = np.ogrid[:height, :width]
    cx = width * liquid_center_x
    cy = height * 0.42
    rx = width * liquid_radius_x
    ry = height * 0.045
    liquid = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
    frame[liquid] = (235, 228, 180)

    label_x0 = int(width * 0.28)
    label_x1 = int(width * 0.72)
    label_y0 = int(height * 0.64)
    label_y1 = int(height * 0.79)
    frame[label_y0:label_y1, label_x0:label_x1] = (232, 226, 185)
    frame[label_y0:label_y1, label_x0:label_x0 + int(width * 0.08)] = (215, 30, 35)
    frame[label_y0:label_y0 + int(height * 0.035), label_x0:label_x1] = (235, 205, 22)
    for y in range(label_y0 + 12, label_y1 - 4, 9):
        frame[y:y + 2, label_x0 + int(width * 0.10):label_x1 - 4] = (45, 45, 42)
    for x in range(label_x0 + int(width * 0.15), label_x1 - 3, 16):
        frame[label_y0 + 4:label_y1 - 4, x:x + 2] = (245, 245, 238)
    return frame


def synthetic_lower_glare_frame(width=480, height=270, shaft_x=None):
    if shaft_x is None:
        shaft_x = int(width * 0.42)
    frame = np.full((height, width, 3), (48, 48, 45), dtype=np.uint8)
    bottle_left = int(width * 0.28)
    bottle_right = int(width * 0.62)
    bottle_top = int(height * 0.18)
    bottle_bottom = int(height * 0.94)

    frame[bottle_top:bottle_bottom, bottle_left:bottle_left + 3] = (188, 188, 184)
    frame[bottle_top:bottle_bottom, bottle_right - 3:bottle_right] = (188, 188, 184)
    frame[bottle_bottom - 3:bottle_bottom, bottle_left:bottle_right] = (188, 188, 184)
    shaft_half = 3
    frame[int(height * 0.08):bottle_bottom, shaft_x - shaft_half:shaft_x + shaft_half] = (238, 238, 234)

    yy, xx = np.ogrid[:height, :width]
    cx = width * 0.42
    cy = height * 0.73
    rx = width * 0.16
    ry = height * 0.11
    liquid = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
    frame[liquid] = (235, 228, 185)

    lower_glare = ((xx - cx) / (width * 0.15)) ** 2 + ((yy - height * 0.84) / (height * 0.08)) ** 2 <= 1.0
    frame[lower_glare] = (255, 255, 250)
    frame[int(height * 0.90):int(height * 0.99), int(width * 0.32):int(width * 0.74)] = (230, 220, 165)
    return frame


def synthetic_pot_rim_frame(width=480, height=270, shaft_x=None):
    frame = synthetic_lower_glare_frame(width=width, height=height, shaft_x=shaft_x)
    rim_y = int(height * 0.80)
    rim_x0 = int(width * 0.18)
    rim_x1 = int(width * 0.80)
    frame[rim_y - 2:rim_y + 2, rim_x0:rim_x1] = (236, 236, 228)
    frame[rim_y + 2:rim_y + 9, rim_x0:rim_x1] = (96, 94, 82)
    return frame


ATTACHMENT_TEST_ROIS = RoiSet(
    liquid_roi=Rect(0.30, 0.62, 0.72, 0.95),
    sparse_roi=Rect(0.40, 0.68, 0.62, 0.90),
    rod_roi=Rect(0.44, 0.30, 0.58, 0.92),
    shaft_core_exclusion=None,
    source="dynamic",
    valid=True,
    quality=1.0,
)


def attachment_base_frame(width=480, height=270):
    frame = np.full((height, width, 3), (45, 45, 42), dtype=np.uint8)
    _paint(frame, ATTACHMENT_TEST_ROIS.liquid_roi, (235, 228, 180))
    return frame


def attachment_frame_bottom_heavy(width=480, height=270):
    frame = attachment_base_frame(width, height)
    rx0, ry0, rx1, ry1 = _rect_px(ATTACHMENT_TEST_ROIS.rod_roi, width, height)
    rod_h = ry1 - ry0
    rod_w = rx1 - rx0
    gap = max(1, int(round(rod_w * 0.10)))
    flank_w = max(1, int(round(rod_w * 0.45)))
    bottom_y0 = ry1 - int(round(rod_h * 0.35))
    frame[bottom_y0:ry1, rx0:rx1] = (235, 228, 180)
    frame[bottom_y0:ry1, max(0, rx0 - gap - flank_w):max(0, rx0 - gap)] = (45, 45, 42)
    frame[bottom_y0:ry1, min(width, rx1 + gap):min(width, rx1 + gap + flank_w)] = (45, 45, 42)
    return frame


def attachment_frame_uniform_rod_and_flanks(width=480, height=270):
    frame = attachment_base_frame(width, height)
    rx0, ry0, rx1, ry1 = _rect_px(ATTACHMENT_TEST_ROIS.rod_roi, width, height)
    rod_w = rx1 - rx0
    gap = max(1, int(round(rod_w * 0.10)))
    flank_w = max(1, int(round(rod_w * 0.45)))
    frame[ry0:ry1, rx0:rx1] = (235, 228, 180)
    frame[ry0:ry1, max(0, rx0 - gap - flank_w):max(0, rx0 - gap)] = (235, 228, 180)
    frame[ry0:ry1, min(width, rx1 + gap):min(width, rx1 + gap + flank_w)] = (235, 228, 180)
    return frame


def attachment_frame_local_rod_bottom(width=480, height=270):
    frame = attachment_base_frame(width, height)
    rx0, ry0, rx1, ry1 = _rect_px(ATTACHMENT_TEST_ROIS.rod_roi, width, height)
    rod_h = ry1 - ry0
    bottom_y0 = ry1 - int(round(rod_h * 0.35))
    frame[bottom_y0:ry1, rx0:rx1] = (235, 228, 180)
    frame[bottom_y0:ry1, max(0, rx0 - 40):rx0 - 8] = (45, 45, 42)
    frame[bottom_y0:ry1, rx1 + 8:min(width, rx1 + 40)] = (45, 45, 42)
    frame[ry0:bottom_y0, rx0:rx1] = (45, 45, 42)
    return frame


def positive_attachment_ellipse_frame(width=480, height=270):
    return synthetic_ellipse_frame(
        width=width,
        height=height,
        liquid_center_x=0.50,
        liquid_center_y=0.76,
        liquid_radius_x=0.20,
        liquid_radius_y=0.12,
    )


def final_candidate_metrics(**kwargs):
    defaults = dict(
        white_coverage=0.70,
        rod_wrap_height_px=120,
        rod_wrap_ratio=1.0,
        connected_area_ratio=0.50,
        rod_connection_score=True,
        is_final_candidate=False,
        rod_bottom_attachment_score=1.0,
        rod_vertical_contrast=0.60,
        rod_top_density=0.30,
        rod_top_density_delta=0.10,
        rod_shape_progress_score=1.0,
        warm_material_coverage=0.0,
        rod_warm_material_ratio=0.0,
        rod_orange_maturity_ratio=0.0,
        rod_orange_maturity_delta=0.0,
        material_connected_area_ratio=0.0,
        orange_material_path_score=0.0,
        roi_source="dynamic",
        roi_valid=True,
        roi_quality=1.0,
    )
    defaults.update(kwargs)
    return FrameMetrics(**defaults)


def baseline_model():
    return BaselineModel(
        rod_wrap_ratio=1.0,
        rod_wrap_iqr=0.0,
        connected_area_ratio=0.5,
        connected_area_iqr=0.0,
        white_coverage=0.7,
        white_coverage_iqr=0.0,
        rod_top_density=0.20,
        rod_top_density_iqr=0.0,
        rod_orange_maturity_ratio=0.0,
        rod_orange_maturity_iqr=0.0,
        initial_final_like=False,
    )


class DynamicDetectorTests(unittest.TestCase):
    def test_dynamic_rois_follow_liquid_guided_rod_axis(self):
        cfg = dynamic_config()
        left_tracker = DynamicRoiTracker.calibrate(
            [synthetic_ellipse_frame(shaft_x=int(480 * 0.42), liquid_center_x=0.42) for _ in range(4)],
            cfg,
        )
        right_tracker = DynamicRoiTracker.calibrate(
            [synthetic_ellipse_frame(shaft_x=int(480 * 0.58), liquid_center_x=0.58) for _ in range(4)],
            cfg,
        )

        left_rois = left_tracker.rois()
        right_rois = right_tracker.rois()
        left_rod = left_rois.rod_roi
        right_rod = right_rois.rod_roi
        left_liquid = left_rois.liquid_roi
        right_liquid = right_rois.liquid_roi
        left_center = (left_rod.x0 + left_rod.x1) / 2
        right_center = (right_rod.x0 + right_rod.x1) / 2
        left_liquid_center = (left_liquid.x0 + left_liquid.x1) / 2
        right_liquid_center = (right_liquid.x0 + right_liquid.x1) / 2

        self.assertGreater(right_center, left_center + 0.10)
        self.assertAlmostEqual(left_center, left_liquid_center, delta=0.004)
        self.assertAlmostEqual(right_center, right_liquid_center, delta=0.004)
        self.assertGreater(left_rois.scores["liquid_ellipse_score"], 0.50)
        self.assertGreaterEqual(left_rod.x1 - left_rod.x0, 0.10)
        self.assertLessEqual(left_rod.x1 - left_rod.x0, 0.19)
        left_liquid_height = left_liquid.y1 - left_liquid.y0
        self.assertAlmostEqual(left_liquid.y0 - left_rod.y0, left_liquid_height * 0.30, delta=0.004)
        self.assertLessEqual(left_rod.y0, left_liquid.y0)
        self.assertAlmostEqual(left_rod.y1, left_liquid.y1, delta=0.004)
        self.assertIsNone(left_rois.shaft_core_exclusion)
        self.assertNotIn("sparse_roi", left_rois.rects_dict())
        self.assertNotIn("shaft_core_exclusion", left_rois.rects_dict())

    def test_liquid_roi_tracks_lower_white_ellipse(self):
        cfg = dynamic_config()
        left_tracker = DynamicRoiTracker.calibrate(
            [synthetic_ellipse_frame(liquid_center_x=0.42) for _ in range(4)],
            cfg,
        )
        right_tracker = DynamicRoiTracker.calibrate(
            [synthetic_ellipse_frame(liquid_center_x=0.58) for _ in range(4)],
            cfg,
        )

        left_liquid = left_tracker.rois().liquid_roi
        right_liquid = right_tracker.rois().liquid_roi
        left_center = (left_liquid.x0 + left_liquid.x1) / 2
        right_center = (right_liquid.x0 + right_liquid.x1) / 2

        self.assertGreater(right_center, left_center + 0.10)
        self.assertAlmostEqual(left_center, 0.42, delta=0.08)
        self.assertAlmostEqual(right_center, 0.58, delta=0.08)
        self.assertGreater(left_tracker.geometry.liquid_ellipse_score, 0.50)

    def test_liquid_roi_prefers_lower_ellipse_over_misaligned_white_region(self):
        cfg = dynamic_config()
        tracker = DynamicRoiTracker.calibrate(
            [synthetic_ellipse_frame(liquid_center_x=0.46, misleading_white=True) for _ in range(4)],
            cfg,
        )

        liquid = tracker.rois().liquid_roi
        center = (liquid.x0 + liquid.x1) / 2

        self.assertLess(center, 0.62)
        self.assertLess(liquid.x1, 0.68)
        self.assertLessEqual(liquid.x1 - liquid.x0, 0.36)
        self.assertGreater(tracker.geometry.liquid_ellipse_score, 0.50)

    def test_wide_view_liquid_roi_rejects_colored_label_region(self):
        cfg = dynamic_config()
        frame = synthetic_wide_view_frame()
        tracker = DynamicRoiTracker.calibrate([frame for _ in range(4)], cfg)

        rois = tracker.rois()
        liquid = rois.liquid_roi
        rod = rois.rod_roi
        liquid_center = (liquid.x0 + liquid.x1) / 2
        rod_center = (rod.x0 + rod.x1) / 2
        shaft_center = tracker.geometry.shaft_x_px / frame.shape[1]

        self.assertLess(liquid.y0, 0.56)
        self.assertLess(liquid.y1, 0.58)
        self.assertAlmostEqual(liquid_center, 0.52, delta=0.12)
        self.assertAlmostEqual(liquid_center, shaft_center, delta=0.035)
        self.assertLessEqual(liquid.x1 - liquid.x0, 0.29)
        self.assertAlmostEqual(rod_center, shaft_center, delta=0.004)
        self.assertGreater(tracker.geometry.liquid_ellipse_score, 0.50)
        self.assertTrue(rois.valid)
        self.assertNotIn("sparse_roi", rois.rects_dict())

    def test_liquid_roi_keeps_upper_boundary_with_lower_glare(self):
        cfg = dynamic_config()
        frame = synthetic_lower_glare_frame()
        tracker = DynamicRoiTracker.calibrate([frame for _ in range(4)], cfg)

        liquid = tracker.rois().liquid_roi

        self.assertLessEqual(liquid.y0, 0.70)
        self.assertLessEqual(liquid.y1, 0.88)
        self.assertLessEqual(liquid.x1 - liquid.x0, 0.28)
        self.assertGreater(tracker.geometry.liquid_ellipse_score, 0.50)

    def test_liquid_roi_trims_horizontal_pot_rim_edge(self):
        cfg = dynamic_config()
        frame = synthetic_pot_rim_frame()
        tracker = DynamicRoiTracker.calibrate([frame for _ in range(4)], cfg)

        liquid = tracker.rois().liquid_roi

        self.assertLessEqual(liquid.y1, 0.82)
        self.assertGreater(tracker.geometry.liquid_ellipse_score, 0.50)

    def test_rod_roi_uses_shaft_axis_when_liquid_center_is_offset(self):
        cfg = dynamic_config()
        frame = synthetic_wide_view_frame(
            shaft_x=int(272 * 0.56),
            liquid_center_x=0.47,
            liquid_radius_x=0.14,
        )
        tracker = DynamicRoiTracker.calibrate([frame for _ in range(4)], cfg)

        rois = tracker.rois()
        liquid_center = (rois.liquid_roi.x0 + rois.liquid_roi.x1) / 2
        rod_center = (rois.rod_roi.x0 + rois.rod_roi.x1) / 2
        shaft_center = tracker.geometry.shaft_x_px / frame.shape[1]

        self.assertGreater(rod_center, liquid_center + 0.03)
        self.assertAlmostEqual(rod_center, shaft_center, delta=0.004)
        self.assertAlmostEqual(shaft_center, 0.56, delta=0.06)

    def test_rod_roi_uses_liquid_guided_axis_not_right_background(self):
        cfg = dynamic_config()
        frame = synthetic_ellipse_frame(
            liquid_center_x=0.42,
            shaft_x=int(480 * 0.42),
            misleading_white=True,
        )
        tracker = DynamicRoiTracker.calibrate([frame for _ in range(4)], cfg)

        rois = tracker.rois()
        rod = rois.rod_roi
        liquid = rois.liquid_roi
        rod_center = (rod.x0 + rod.x1) / 2
        liquid_center = (liquid.x0 + liquid.x1) / 2
        shaft_center = tracker.geometry.shaft_x_px / frame.shape[1]

        self.assertAlmostEqual(rod_center, shaft_center, delta=0.004)
        self.assertAlmostEqual(rod_center, liquid_center, delta=0.004)
        self.assertLess(rod.x1, 0.62)
        liquid_height = liquid.y1 - liquid.y0
        self.assertAlmostEqual(liquid.y0 - rod.y0, liquid_height * 0.30, delta=0.004)
        self.assertLessEqual(rod.y0, liquid.y0)
        self.assertAlmostEqual(rod.y1, liquid.y1, delta=0.004)

    def test_liquid_roi_falls_back_when_ellipse_missing(self):
        cfg = dynamic_config(liquid_roi=Rect(0.21, 0.62, 0.63, 0.91))
        dark = np.full((270, 480, 3), (45, 45, 42), dtype=np.uint8)
        tracker = DynamicRoiTracker.calibrate([dark for _ in range(4)], cfg)

        liquid = tracker.rois().liquid_roi
        for actual, expected in zip(liquid, cfg.liquid_roi):
            self.assertAlmostEqual(actual, expected, delta=0.004)
        self.assertEqual(tracker.geometry.liquid_ellipse_score, 0.0)

    def test_dynamic_rod_metrics_ignore_shaft_exclusion_region(self):
        cfg = dynamic_config()
        frame = np.full((120, 160, 3), (40, 40, 38), dtype=np.uint8)
        rois = RoiSet(
            liquid_roi=Rect(0.20, 0.60, 0.80, 0.95),
            sparse_roi=Rect(0.38, 0.68, 0.62, 0.88),
            rod_roi=Rect(0.45, 0.50, 0.55, 0.95),
            shaft_core_exclusion=Rect(0.45, 0.50, 0.55, 0.95),
            source="dynamic",
            valid=True,
            quality=1.0,
        )
        x0, y0, x1, y1 = (
            int(round(rois.rod_roi.x0 * frame.shape[1])),
            int(round(rois.rod_roi.y0 * frame.shape[0])),
            int(round(rois.rod_roi.x1 * frame.shape[1])),
            int(round(rois.rod_roi.y1 * frame.shape[0])),
        )
        frame[y0:y1, x0:x1] = (235, 228, 180)

        dynamic_metrics = detect_frame(frame, cfg, rois=rois)
        fixed_metrics = detect_frame(frame, cfg, rois=replace(rois, source="fixed"))

        self.assertEqual(dynamic_metrics.rod_wrap_ratio, 1.0)
        self.assertEqual(fixed_metrics.rod_wrap_ratio, 0.0)

    def test_rod_bottom_attachment_detects_bottom_heavy_wrap(self):
        cfg = dynamic_config()
        metrics = detect_frame(attachment_frame_bottom_heavy(), cfg, rois=ATTACHMENT_TEST_ROIS)

        self.assertGreaterEqual(metrics.rod_bottom_attachment_score, 0.80)
        self.assertGreater(metrics.rod_vertical_contrast, 0.50)
        self.assertGreater(metrics.rod_bottom_density, metrics.rod_top_density)

    def test_rod_bottom_attachment_rejects_uniform_white_rod_and_liquid(self):
        cfg = dynamic_config()
        metrics = detect_frame(attachment_frame_uniform_rod_and_flanks(), cfg, rois=ATTACHMENT_TEST_ROIS)

        self.assertLess(metrics.rod_bottom_attachment_score, 0.30)
        self.assertAlmostEqual(metrics.rod_bottom_density, metrics.rod_top_density, delta=0.01)
        self.assertLess(metrics.rod_local_contrast, 0.05)

    def test_rod_bottom_attachment_detects_local_rod_bottom_blob(self):
        cfg = dynamic_config()
        metrics = detect_frame(attachment_frame_local_rod_bottom(), cfg, rois=ATTACHMENT_TEST_ROIS)

        self.assertGreaterEqual(metrics.rod_bottom_attachment_score, 0.80)
        self.assertGreater(metrics.rod_local_contrast, 0.20)

    def test_dynamic_final_candidate_waits_until_min_elapsed_time(self):
        cfg = dynamic_config(dynamic_final_min_elapsed_sec=420.0)
        metrics = final_candidate_metrics()
        baseline = baseline_model()

        self.assertFalse(_dynamic_final_candidate(metrics, baseline, cfg, 419.0))
        self.assertTrue(_dynamic_final_candidate(metrics, baseline, cfg, 420.0))

    def test_default_dynamic_final_candidate_has_no_elapsed_time_gate(self):
        cfg = dynamic_config()
        metrics = final_candidate_metrics()
        baseline = baseline_model()

        self.assertTrue(_dynamic_final_candidate(metrics, baseline, cfg, 0.0))

    def test_dynamic_final_candidate_rejects_bottom_attachment_without_shape_progress(self):
        cfg = dynamic_config()
        metrics = final_candidate_metrics(
            rod_vertical_contrast=0.15,
            rod_top_density=0.22,
            rod_top_density_delta=0.02,
            rod_shape_progress_score=0.45,
        )
        baseline = baseline_model()

        self.assertFalse(_dynamic_final_candidate(metrics, baseline, cfg, 120.0))

    def test_dynamic_final_candidate_rejects_hollow_white_path(self):
        cfg = dynamic_config()
        metrics = final_candidate_metrics(
            rod_bottom_attachment_score=1.0,
            rod_vertical_contrast=0.60,
            rod_top_density=0.22,
            rod_top_density_delta=0.10,
            rod_shape_progress_score=1.0,
        )
        baseline = baseline_model()

        self.assertFalse(_dynamic_final_candidate(metrics, baseline, cfg, 120.0))

    def test_dynamic_final_candidate_accepts_top_density_progress(self):
        cfg = dynamic_config()
        metrics = final_candidate_metrics(
            rod_vertical_contrast=0.15,
            rod_top_density=0.29,
            rod_top_density_delta=0.09,
            rod_shape_progress_score=1.0,
        )
        baseline = baseline_model()

        self.assertTrue(_dynamic_final_candidate(metrics, baseline, cfg, 120.0))

    def test_dynamic_final_candidate_accepts_orange_material_path(self):
        cfg = dynamic_config()
        metrics = final_candidate_metrics(
            white_coverage=0.20,
            connected_area_ratio=0.02,
            rod_bottom_attachment_score=0.0,
            rod_vertical_contrast=-0.20,
            rod_shape_progress_score=0.0,
            warm_material_coverage=0.72,
            rod_warm_material_ratio=0.74,
            rod_orange_maturity_ratio=0.54,
            rod_orange_maturity_delta=0.54,
            material_connected_area_ratio=0.62,
            orange_material_path_score=1.0,
        )
        baseline = baseline_model()

        self.assertTrue(_dynamic_final_candidate(metrics, baseline, cfg, 420.0))

    def test_dynamic_final_candidate_rejects_immature_warm_material(self):
        cfg = dynamic_config()
        metrics = final_candidate_metrics(
            white_coverage=0.20,
            connected_area_ratio=0.02,
            rod_bottom_attachment_score=0.0,
            rod_vertical_contrast=-0.20,
            rod_shape_progress_score=0.0,
            warm_material_coverage=0.78,
            rod_warm_material_ratio=0.77,
            rod_orange_maturity_ratio=0.39,
            rod_orange_maturity_delta=0.39,
            material_connected_area_ratio=0.66,
            orange_material_path_score=0.78,
        )
        baseline = baseline_model()

        self.assertFalse(_dynamic_final_candidate(metrics, baseline, cfg, 420.0))

    def test_dynamic_detector_alerts_after_attachment_is_stable_and_allowed(self):
        cfg = dynamic_config(
            calibration_duration_sec=2.0,
            baseline_duration_sec=2.0,
            metric_smooth_sec=0.1,
            stable_duration_sec=3.0,
            stable_min_ratio=0.8,
            dynamic_final_min_elapsed_sec=4.0,
            rod_bottom_attachment_score_min=0.60,
            connected_area_ratio_min=0.05,
            dynamic_white_coverage_min=0.30,
            initial_final_white_coverage_min=1.1,
        )
        detector = GelClimbDetector(config=cfg)
        result = None
        for t in range(8):
            result = detector.update(positive_attachment_ellipse_frame(), float(t))

        self.assertIsNotNone(result)
        self.assertEqual(result.state, FINAL_GEL_ROD_CLIMBING)
        self.assertTrue(result.alert)
        self.assertGreaterEqual(result.transition_time_sec, 7.0)
        self.assertGreaterEqual(result.evidence["rod_bottom_attachment_score"], 0.60)

    def test_dynamic_detector_ignores_misaligned_white_background(self):
        cfg = dynamic_config(initial_final_white_coverage_min=1.1)
        detector = GelClimbDetector(config=cfg)
        result = None
        for t in range(8):
            result = detector.update(synthetic_frame(white_background=True), float(t))

        self.assertIsNotNone(result)
        self.assertEqual(result.state, LIQUID_STIRRING)
        self.assertFalse(result.alert)

    def test_no_initial_final_shortcut_without_delta(self):
        cfg = dynamic_config(rod_wrap_ratio_min=0.60, initial_final_white_coverage_min=0.35, sparse_hole_ratio_max=0.08)
        detector = GelClimbDetector(config=cfg)
        result = None
        for t in range(7):
            result = detector.update(synthetic_frame(gel=True), float(t))

        self.assertIsNotNone(result)
        self.assertEqual(result.state, LIQUID_STIRRING)
        self.assertFalse(result.alert)
        self.assertFalse(result.evidence["initial_final_like"])

    def test_dynamic_roi_freezes_after_calibration(self):
        cfg = dynamic_config(calibration_duration_sec=2.0, initial_final_white_coverage_min=1.1)
        detector = GelClimbDetector(config=cfg)
        result = None
        for t in range(3):
            result = detector.update(synthetic_frame(shaft_x=300), float(t))

        self.assertIsNotNone(result)
        calibrated_rod_roi = result.evidence["rois"]["rod_roi"]
        calibrated_liquid_roi = result.evidence["rois"]["liquid_roi"]
        self.assertNotIn("sparse_roi", result.evidence["rois"])

        for t in range(3, 7):
            result = detector.update(synthetic_frame(shaft_x=340, white_background=True), float(t))

        self.assertEqual(result.evidence["rois"]["rod_roi"], calibrated_rod_roi)
        self.assertEqual(result.evidence["rois"]["liquid_roi"], calibrated_liquid_roi)
        self.assertNotIn("sparse_roi", result.evidence["rois"])
        self.assertNotIn("sparse_hole_ratio", result.evidence)

    def test_liquid_roi_freezes_after_calibration_for_reliable_ellipse(self):
        cfg = dynamic_config(
            calibration_duration_sec=2.0,
            baseline_duration_sec=2.0,
            dynamic_roi_ema_alpha=0.70,
            dynamic_roi_update_interval_sec=0.0,
            initial_final_white_coverage_min=1.1,
        )
        detector = GelClimbDetector(config=cfg)
        result = None
        for t in range(3):
            result = detector.update(synthetic_ellipse_frame(liquid_center_y=0.70), float(t))

        self.assertIsNotNone(result)
        calibrated_rod_roi = result.evidence["rois"]["rod_roi"]
        calibrated_liquid_roi = result.evidence["rois"]["liquid_roi"]

        for t in range(3, 9):
            result = detector.update(synthetic_ellipse_frame(liquid_center_y=0.84), float(t))

        self.assertEqual(result.evidence["rois"]["rod_roi"], calibrated_rod_roi)
        self.assertEqual(result.evidence["rois"]["liquid_roi"], calibrated_liquid_roi)

    def test_default_dynamic_roi_calibration_uses_first_30_seconds(self):
        self.assertEqual(DetectionConfig().calibration_duration_sec, 30.0)

    def test_30_second_roi_calibration_uses_recent_stable_frames(self):
        cfg = dynamic_config(
            calibration_duration_sec=30.0,
            baseline_duration_sec=30.0,
            initial_final_white_coverage_min=1.1,
        )
        detector = GelClimbDetector(config=cfg)
        result = None
        for t in range(31):
            liquid_center_y = 0.70 if t <= 20 else 0.84
            result = detector.update(synthetic_ellipse_frame(liquid_center_y=liquid_center_y), float(t))

        self.assertIsNotNone(result)
        final_liquid = result.evidence["rois"]["liquid_roi"]
        self.assertGreater(final_liquid[1], 0.70)
        self.assertGreater(final_liquid[3], 0.85)

if __name__ == "__main__":
    unittest.main()
