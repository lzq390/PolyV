import unittest

import numpy as np

from polyv_detector import FINAL_GEL_ROD_CLIMBING, LIQUID_STIRRING
from polyv_detector.detector import DetectionConfig, GelClimbDetector, Rect, detect_frame


TEST_CONFIG = DetectionConfig(
    liquid_roi=Rect(0.30, 0.62, 0.72, 0.95),
    sparse_roi=Rect(0.40, 0.68, 0.62, 0.90),
    rod_roi=Rect(0.44, 0.30, 0.58, 0.92),
    shaft_core_exclusion=Rect(0.49, 0.30, 0.53, 0.92),
    rod_wrap_height_min_px=35,
    stable_duration_sec=10.0,
    stable_min_ratio=0.9,
    window_time_tolerance_sec=0.0,
    sparse_component_stride=1,
)


def _rect_px(rect, width, height):
    return (
        int(round(rect.x0 * width)),
        int(round(rect.y0 * height)),
        int(round(rect.x1 * width)),
        int(round(rect.y1 * height)),
    )


def _paint(frame, rect, color):
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = _rect_px(rect, width, height)
    frame[y0:y1, x0:x1] = color


def liquid_frame(width=480, height=270):
    frame = np.full((height, width, 3), (115, 105, 88), dtype=np.uint8)
    _paint(frame, TEST_CONFIG.liquid_roi, (205, 198, 155))

    # Large dark sparse vortex remains in the center, so this must not be final.
    _paint(frame, TEST_CONFIG.sparse_roi, (55, 52, 45))

    # The stirring shaft is white, but shaft_core_exclusion prevents this from
    # being counted as gel wrapping the rod.
    _paint(frame, TEST_CONFIG.shaft_core_exclusion, (230, 230, 225))
    return frame


def final_gel_frame(width=480, height=270):
    frame = np.full((height, width, 3), (115, 105, 88), dtype=np.uint8)
    _paint(frame, TEST_CONFIG.liquid_roi, (235, 228, 180))
    _paint(frame, TEST_CONFIG.sparse_roi, (235, 228, 180))
    _paint(frame, TEST_CONFIG.shaft_core_exclusion, (230, 230, 225))

    height_px, width_px = frame.shape[:2]
    rx0, ry0, rx1, ry1 = _rect_px(TEST_CONFIG.rod_roi, width_px, height_px)
    ex0, _, ex1, _ = _rect_px(TEST_CONFIG.shaft_core_exclusion, width_px, height_px)
    gel_top = ry0 + 20
    frame[gel_top:ry1, rx0:ex0] = (235, 228, 180)
    frame[gel_top:ry1, ex1:rx1] = (235, 228, 180)
    return frame


def near_final_frame(width=480, height=270):
    frame = final_gel_frame(width, height)
    _paint(frame, TEST_CONFIG.sparse_roi, (70, 65, 55))
    return frame


class DetectorTests(unittest.TestCase):
    def test_liquid_frame_is_not_final_candidate(self):
        metrics = detect_frame(liquid_frame(), TEST_CONFIG)
        self.assertFalse(metrics.is_final_candidate)
        self.assertGreater(metrics.sparse_hole_ratio, TEST_CONFIG.sparse_hole_ratio_max)

    def test_final_frame_is_final_candidate(self):
        metrics = detect_frame(final_gel_frame(), TEST_CONFIG)
        self.assertTrue(metrics.is_final_candidate)
        self.assertGreaterEqual(metrics.white_coverage, TEST_CONFIG.white_coverage_min)
        self.assertLessEqual(metrics.sparse_hole_ratio, TEST_CONFIG.sparse_hole_ratio_max)
        self.assertGreaterEqual(metrics.rod_wrap_height_px, TEST_CONFIG.rod_wrap_height_min_px)

    def test_near_final_with_sparse_hole_stays_liquid(self):
        metrics = detect_frame(near_final_frame(), TEST_CONFIG)
        self.assertFalse(metrics.is_final_candidate)
        self.assertGreater(metrics.sparse_hole_ratio, TEST_CONFIG.sparse_hole_ratio_max)

    def test_state_machine_requires_stable_window(self):
        detector = GelClimbDetector(config=TEST_CONFIG)
        result = None
        for t in range(10):
            result = detector.update(final_gel_frame(), float(t))
        self.assertIsNotNone(result)
        self.assertEqual(result.state, LIQUID_STIRRING)
        self.assertFalse(result.alert)

        result = detector.update(final_gel_frame(), 10.0)
        self.assertEqual(result.state, FINAL_GEL_ROD_CLIMBING)
        self.assertTrue(result.alert)
        self.assertEqual(result.transition_time_sec, 10.0)

    def test_state_machine_does_not_alert_for_7_to_8_min_like_near_final(self):
        detector = GelClimbDetector(config=TEST_CONFIG)
        result = None
        for t in range(40):
            result = detector.update(near_final_frame(), float(t))
        self.assertIsNotNone(result)
        self.assertEqual(result.state, LIQUID_STIRRING)
        self.assertFalse(result.alert)


if __name__ == "__main__":
    unittest.main()
