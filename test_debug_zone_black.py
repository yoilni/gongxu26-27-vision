"""Regression checks: cancelled safety-zone line preview cannot run again."""

import ast
from pathlib import Path
import unittest

import test_vision_rules  # Host-only Maix stub.
import app_config as config
import debug_zone_black
import vision_overlay

ROOT = Path(__file__).resolve().parent


class ZoneLinePreviewRemovedTests(unittest.TestCase):
    def test_retired_module_has_no_detector_or_dependencies(self):
        self.assertFalse(hasattr(debug_zone_black, "DebugZoneBlackPreview"))
        tree = ast.parse((ROOT / "debug_zone_black.py").read_text(encoding="utf8"))
        self.assertEqual(len(tree.body), 1)
        self.assertIsInstance(tree.body[0], ast.Expr)
        self.assertIsInstance(tree.body[0].value, ast.Constant)

    def test_line_renderer_is_removed_without_removing_other_debug_geometry(self):
        self.assertFalse(hasattr(vision_overlay, "draw_debug_zone_black_lines"))
        self.assertTrue(callable(vision_overlay.draw_debug_frame_center))
        self.assertTrue(callable(vision_overlay.draw_roi_border))
        self.assertTrue(callable(vision_overlay.draw_debug_zone_obstacle_region))
        self.assertTrue(callable(vision_overlay.draw_debug_yolo_detections))

    def test_all_line_preview_configuration_is_removed(self):
        self.assertFalse(any(name.startswith("DEBUG_ZONE_BLACK") for name in vars(config)))
        self.assertEqual(config.DETECT_BLACK_CONFIDENCE, .7)
        self.assertEqual(config.DETECT_SAFETY_ZONE_CONFIDENCE, .4)
        self.assertFalse(config.VERIFY_COLOR_ENABLED)

    def test_runtime_package_excludes_retired_source_but_keeps_regression_checks(self):
        manifest = (ROOT / "app.yaml").read_text(encoding="utf8")
        self.assertNotIn("  - debug_zone_black.py", manifest)
        self.assertIn("  - test_debug_zone_black.py", manifest)
        self.assertNotIn("  - debug_lab_preview.py", manifest)
        self.assertTrue((ROOT / "debug_zone_black.py").is_file())

    def test_main_has_no_feature_construction_scan_or_draw(self):
        source = (ROOT / "main.py").read_text(encoding="utf8")
        for retired in ("debug_zone_black", "DebugZoneBlackPreview", "debug_black_lines",
                        "draw_debug_zone_black_lines", "HoughLines", "image2cv", "cv2image"):
            self.assertNotIn(retired, source)
        self.assertIn("zone_line_preview_removed_v8", source)

    def test_remaining_debug_photo_fps_and_uart_entry_points_are_preserved(self):
        source = (ROOT / "main.py").read_text(encoding="utf8")
        for retained in ("draw_debug_yolo_detections(",
                         "draw_roi_border(", "debug_fps=profiler.fps", "DatasetCapture(",
                         "dataset_capture.save(img)", "VisionSerialController(",
                         "raw_objects=objects", "disp.show(img"):
            self.assertIn(retained, source)
        for removed in ("draw_debug_frame_center(", "draw_debug_zone_obstacle_region(",
                        "draw_debug_zone_realign_point(", "draw_debug_object_area(",
                        "draw_debug_safety_zone_area(", "debug_arrangement_objects("):
            self.assertNotIn(removed, source)

    def test_packaged_runtime_never_imports_opencv_numpy_or_retired_preview(self):
        manifest = (ROOT / "app.yaml").read_text(encoding="utf8")
        for line in manifest.splitlines():
            item = line.strip().removeprefix("- ")
            if not item.endswith(".py") or item.startswith("test_"):
                continue
            with self.subTest(file=item):
                tree = ast.parse((ROOT / item).read_text(encoding="utf8"))
                for node in ast.walk(tree):
                    modules = ([name.name for name in node.names] if isinstance(node, ast.Import)
                               else [node.module] if isinstance(node, ast.ImportFrom) else [])
                    for module in modules:
                        self.assertNotIn((module or "").split(".")[0],
                                         ("cv2", "numpy", "debug_zone_black"))


if __name__ == "__main__":
    unittest.main()
