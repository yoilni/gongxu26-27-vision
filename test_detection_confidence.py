"""Per-class confidence gates run before every raw/verified YOLO consumer."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_vision_rules import block, controller, write_near_view_request, SingleFrameTrackingScenario
from detection_confidence import DetectionConfidenceFilter
from model_class_mapping import ModelClassMapper
import serial_controller as s


NEW_LABELS = ["bluesafety", "redsafety", "sqareblue", "sqaregreen",
              "sqarered", "startzone", "triangualrblack"]


def scored(class_id, score, x=450, y=100):
    obj = block(x, y, class_id)
    obj.score = score
    return obj


class DetectionConfidenceTests(unittest.TestCase):
    def setUp(self):
        self.gate = DetectionConfidenceFilter(.5, .7, .4)
        self.mapper = ModelClassMapper(NEW_LABELS)

    def accepted(self, raw):
        return self.gate.apply(self.mapper.normalize(raw))

    def test_detector_floor_keeps_zones_below_the_default_threshold(self):
        self.assertEqual(self.gate.detector_threshold, .4)
        self.assertEqual(DetectionConfidenceFilter(.3, .7, .4).detector_threshold, .3)

    def test_black_requires_at_least_point7_including_the_boundary(self):
        for score, expected in ((.4, False), (.5, False), (.6999, False), (.7, True), (.95, True)):
            with self.subTest(score=score):
                self.assertEqual(bool(self.accepted([scored(6, score)])), expected)

    def test_both_safety_zones_accept_point4_but_not_below(self):
        for model_id in (0, 1):
            for score, expected in ((.3999, False), (.4, True), (.45, True), (.9, True)):
                with self.subTest(model_id=model_id, score=score):
                    result = self.accepted([scored(model_id, score)])
                    self.assertEqual(bool(result), expected)
                    if result:
                        self.assertEqual(result[0].class_id, model_id + 1)

    def test_red_green_blue_keep_the_existing_point5_threshold(self):
        for model_id, semantic_id in ((2, 3), (3, 5), (4, 4)):
            for score, expected in ((.4, False), (.4999, False), (.5, True), (.7, True)):
                with self.subTest(model_id=model_id, score=score):
                    result = self.accepted([scored(model_id, score)])
                    self.assertEqual(bool(result), expected)
                    if result:
                        self.assertEqual(result[0].class_id, semantic_id)

    def test_thresholds_apply_after_model_mapping_not_native_index(self):
        raw = [scored(0, .4), scored(1, .4), scored(2, .49), scored(3, .5),
               scored(4, .5), scored(5, .99), scored(6, .69)]
        self.assertEqual([obj.class_id for obj in self.accepted(raw)], [1, 2, 5, 4])

    def test_old_eight_and_seven_class_layouts_use_the_same_thresholds(self):
        layouts = [
            (["blue safety", "bluesafety", "redsafety", "sqareblue", "sqarered", "sqareredgreen", "triangualrblack"], 6),
            (["blue safety", "bluesafety", "redsafety", "sqareblue", "sqarered", "sqareredgreen", "startzone", "triangualrblack"], 7),
        ]
        for labels, black_id in layouts:
            with self.subTest(classes=len(labels)):
                mapper = ModelClassMapper(labels)
                raw = [scored(1, .4), scored(2, .4), scored(black_id, .69)]
                self.assertEqual([obj.class_id for obj in self.gate.apply(mapper.normalize(raw))], [1, 2])
                self.assertEqual(self.gate.apply(mapper.normalize([scored(black_id, .7)]))[0].class_id, 6)

    def test_filter_keeps_object_identity_score_geometry_and_order(self):
        objects = [scored(5, .5), scored(6, .7), scored(4, .9)]
        before = [vars(obj).copy() for obj in objects]
        accepted = self.gate.apply(objects)
        self.assertEqual([id(obj) for obj in accepted], [id(obj) for obj in objects])
        self.assertEqual([vars(obj) for obj in objects], before)

    def test_invalid_or_missing_scores_are_not_valid_detections(self):
        raw = [scored(6, float("nan")), scored(6, float("inf")), scored(6, float("-inf"))]
        missing = block(450, 100, 6)
        del missing.score
        self.assertEqual(self.gate.apply(raw + [missing]), [])

    def test_ignored_class_ids_cannot_pass_even_at_high_confidence(self):
        self.assertEqual(self.gate.apply([scored(i, .99) for i in (-1, 0, 7)]), [])
        self.assertEqual(self.accepted([scored(5, .99)]), [])  # model5=startzone

    def test_empty_or_weak_frame_does_not_reuse_previous_detections(self):
        self.assertEqual(len(self.accepted([scored(6, .9)])), 1)
        self.assertEqual(self.accepted([scored(6, .69)]), [])
        self.assertEqual(self.accepted([]), [])

    def test_custom_black_and_safety_thresholds_are_independent(self):
        gate = DetectionConfidenceFilter(.55, .8, .3)
        self.assertEqual(gate.detector_threshold, .3)
        raw = [scored(1, .3), scored(2, .3), scored(3, .54), scored(5, .55), scored(6, .79)]
        self.assertEqual([obj.class_id for obj in gate.apply(raw)], [1, 2, 5])

    def test_invalid_threshold_settings_fail_explicitly(self):
        for values in ((-.1, .7, .4), (.5, 1.1, .4), (.5, .7, float("nan"))):
            with self.subTest(values=values), self.assertRaises(ValueError):
                DetectionConfidenceFilter(*values)

    def test_main_applies_confidence_after_mapping_before_all_consumers(self):
        root = Path(__file__).resolve().parent
        tree = ast.parse((root / "main.py").read_text(encoding="utf8"))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute)]
        normalize = next(node for node in calls if node.func.attr == "normalize")
        filtered = [node for node in calls if node.func.attr == "apply"
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "confidence_filter"]
        self.assertEqual(len(filtered), 1)
        self.assertLess(normalize.lineno, filtered[0].lineno)
        for node in calls:
            if (isinstance(node.func.value, ast.Name) and node.func.value.id in ("verifier", "serial_controller")
                    and node.func.attr in ("process", "submit", "arrangement_side_objects")):
                self.assertLess(filtered[0].lineno, node.lineno)
        manifest = (root / "app.yaml").read_text(encoding="utf8")
        self.assertIn("  - detection_confidence.py", manifest)
        self.assertIn("  - test_detection_confidence.py", manifest)


class ConfidencePipelineTests(SingleFrameTrackingScenario):
    accepted = DetectionConfidenceTests.accepted

    def setUp(self):
        self.gate = DetectionConfidenceFilter(.5, .7, .4)
        self.mapper = ModelClassMapper(NEW_LABELS)
        helper = patch("builtins.print")
        helper.start()
        self.addCleanup(helper.stop)

    def submit(self, current, raw, now=1000):
        objects = self.accepted(raw)
        with patch.object(s, "_now_ms", return_value=now):
            return current.submit(objects, self.mapper.labels, raw_objects=objects)

    def command(self, current, command, now=1000):
        with patch.object(s, "_now_ms", return_value=now):
            current.process_received_bytes(s.build_command_packet(command))

    def test_weak_black_cannot_publish_coordinates_through_an_existing_raw_lock(self):
        for mode in (s.CMD_SEARCH, s.CMD_TRACK, s.MODE_FINAL_ROI_TRACK):
            with self.subTest(mode=mode):
                current = controller(mode)
                self.command(current, s.CMD_SELECT_BLACK_OR_GREEN)
                current.mode = mode
                self.assertEqual(self.submit(current, [scored(6, .7)]).class_id, 6)
                self.assertEqual(current._latest_packet[2], 6)
                self.assertIsNone(self.submit(current, [scored(6, .69)], 1100))
                self.assertIsNone(current._latest_packet)

    def test_weak_black_is_not_a_spacing_neighbor(self):
        current = controller(s.MODE_FINAL_ROI_TRACK)
        objects = self.accepted([scored(3, .5, 300, 200), scored(6, .69, 301, 200)])
        selected = current._select_final_roi_target(objects, self.mapper.labels, "sqareredgreen", 1000)
        self.assertEqual(selected.class_id, 5)

    def test05_does_not_count_weak_black_as_a_second_supply(self):
        current = controller(s.MODE_FINAL_ROI_TRACK)
        self.command(current, s.CMD_SELECT_BLACK_OR_GREEN)
        self.command(current, s.CMD_SEARCH_BLUE_ZONE, 2000)
        self.assertEqual(write_near_view_request(current, 2000), s.build_event_packet(5))
        for now in (2500, 2510, 2520):
            self.submit(current, [scored(3, .5, 120, 200), scored(6, .69, 200, 200)], now)
        self.assertEqual(current.mode, s.MODE_WAIT_SUPPLY_SEARCH)
        self.assertEqual(current._next_outgoing_packet(2520), s.build_event_packet(3))

    def test06_weak_large_zone_cannot_trigger26_or_displace_valid_point4_zone(self):
        current = controller(s.MODE_SEARCH_ZONE)
        current.zone_label = "bluesafety"
        weak = SimpleNamespace(x=0, y=0, w=576, h=448, class_id=0, score=.3999)
        valid = SimpleNamespace(x=50, y=50, w=300, h=100, class_id=0, score=.4)
        selected = self.submit(current, [weak, valid])
        self.assertEqual((selected.class_id, selected.w, selected.score), (1, 300, .4))
        self.assertEqual(current._latest_packet[2], 1)
        self.assertNotIn(s.build_event_packet(0x26), current._pending_events)

    def test_weak_zone_cannot_exclude_an_otherwise_valid_green(self):
        current = controller(s.CMD_SEARCH)
        weak = SimpleNamespace(x=0, y=0, w=576, h=448, class_id=0, score=.3999)
        selected = self.submit(current, [weak, scored(3, .5)])
        self.assertEqual(selected.class_id, 5)
        self.assertEqual(current._latest_packet[2], 5)


if __name__ == "__main__":
    unittest.main()
