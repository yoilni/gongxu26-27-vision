"""New/legacy model layouts must preserve the existing MCU category contract."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from test_vision_rules import LABELS, block, near_view_block, controller, write_near_view_request, SingleFrameTrackingScenario
import app_config as config
import serial_controller as s
from detection_verifier import DetectionVerifier
from detection_confidence import DetectionConfidenceFilter
from model_class_mapping import ModelClassMapper, VISION_LABELS
from vision_overlay import draw_debug_yolo_detections


MODEL_LABELS = [
    "blue safety", "bluesafety", "redsafety", "sqareblue",
    "sqarered", "sqareredgreen", "startzone", "triangualrblack",
]

REORDERED_SEVEN_LABELS = [
    "bluesafety", "redsafety", "sqareblue", "sqaregreen",
    "sqarered", "startzone", "triangualrblack",
]


def model_near_view_block(x, y, model_id, labels):
    """Only05 fixtures meet red/black area gates, preserving native model IDs."""
    label = labels[model_id]
    semantic_id = 6 if label == "triangualrblack" else 4 if label == "sqarered" else 5
    obj = near_view_block(x, y, semantic_id)
    obj.class_id = model_id
    return obj


class ModelClassMappingTests(SingleFrameTrackingScenario):
    model_labels = MODEL_LABELS

    def setUp(self):
        self.mapper = ModelClassMapper(MODEL_LABELS)
        for setting in (
            patch.object(config, "VISION_DIAGNOSTICS_ENABLED", False),
            patch("builtins.print"),
        ):
            setting.start()
            self.addCleanup(setting.stop)

    def submit(self, current, objects, now=1000, verified=True):
        # Exactly the production boundary: normalized objects AND labels go to
        # every downstream consumer; raw association must not see model IDs.
        normalized = self.mapper.normalize(objects)
        with patch.object(s, "_now_ms", return_value=now):
            return current.submit(
                normalized if verified else [], self.mapper.labels,
                raw_objects=normalized,
                frame_generation=current.frame_generation(),
            )

    def command(self, current, command, now=1000):
        with patch.object(s, "_now_ms", return_value=now):
            current.process_received_bytes(s.build_command_packet(command))

    def near_check(self, current, model_ids, now=2000):
        self.command(current, s.CMD_SEARCH_BLUE_ZONE, now)
        self.assertEqual(write_near_view_request(current, now), s.build_event_packet(5))
        objects = [model_near_view_block(120 + index * 50, 200, model_id, self.model_labels)
                   for index, model_id in enumerate(model_ids)]
        for index in range(3):
            self.submit(current, objects, now + 500 + index * 10)
        return now + 520

    def test_eight_class_mapping_ignores_startzone_and_deprecated_class(self):
        raw = [block(100, 100, model_id) for model_id in range(8)]
        mapped = self.mapper.normalize(raw)
        self.assertEqual([obj.class_id for obj in mapped], [1, 2, 3, 4, 5, 6])
        self.assertEqual(self.mapper.labels, tuple(LABELS))
        self.assertEqual(mapped[-1].model_class_id, 7)
        self.assertEqual(self.mapper.labels[mapped[-1].class_id], "triangualrblack")
        self.assertNotIn("startzone", self.mapper.labels)

    def test_mapping_uses_label_not_class_count_or_fixed_source_index(self):
        mapper = ModelClassMapper(["triangualrblack", "startzone", "sqarered", "other"])
        mapped = mapper.normalize([block(100, 100, i) for i in range(4)])
        self.assertEqual([obj.class_id for obj in mapped], [6, 4])
        self.assertEqual([obj.model_class_id for obj in mapped], [0, 2])

    def test_old_seven_class_model_keeps_black6_and_existing_objects(self):
        mapper = ModelClassMapper(LABELS)
        raw = [block(100, 100, i) for i in range(7)]
        mapped = mapper.normalize(raw)
        self.assertEqual([obj.class_id for obj in mapped], [1, 2, 3, 4, 5, 6])
        for obj, source in zip(mapped, raw[1:]):
            self.assertIs(obj, source)

    def test_mapping_does_not_mutate_native_boxes_or_geometry_confidence(self):
        source = block(190.5, 202.5, 7)
        mapped = self.mapper.normalize([source])[0]
        self.assertIsNot(mapped, source)
        self.assertEqual(source.class_id, 7)
        for field in ("x", "y", "w", "h", "score"):
            self.assertEqual(getattr(mapped, field), getattr(source, field))
        self.assertEqual(mapped.class_id, 6)

    def test_bad_model_ids_and_unknown_labels_are_not_invented_as_targets(self):
        mapper = ModelClassMapper(MODEL_LABELS + ["unknown"])
        self.assertEqual(mapper.normalize([block(100, 100, i) for i in (-1, 0, 6, 8, 9)]), [])
        self.assertEqual(mapper.normalize([]), [])

    def test_mapper_does_not_cache_boxes_between_frames(self):
        first = self.mapper.normalize([block(100, 100, 7)])[0]
        second = self.mapper.normalize([block(190, 180, 7)])[0]
        self.assertNotEqual((first.x, first.y), (second.x, second.y))
        self.assertEqual(self.mapper.normalize([block(100, 100, 6)]), [])

    def test_startup_summary_distinguishes_model_index_from_uart_id(self):
        self.assertEqual(self.mapper.summary(),
                         "0->ignore 1->1 2->2 3->3 4->4 5->5 6->ignore 7->6")

    def test_verifier_confirms_black7_not_startzone6(self):
        verifier = DetectionVerifier(576, 448)
        raw = [block(100, 100, 6), block(300, 180, 7)]
        img = Mock()
        confirmed = []
        for _ in range(config.VERIFY_REQUIRED_HITS):
            confirmed = verifier.process(img, self.mapper.normalize(raw), self.mapper.labels)
        self.assertEqual([obj.class_id for obj in confirmed], [6])
        self.assertEqual(set(verifier._class_states), {"triangualrblack"})
        self.assertEqual(confirmed[0].model_class_id, 7)

    def test_disabled_verification_still_cannot_let_startzone_into_state_machine(self):
        raw = self.mapper.normalize([block(100, 100, 6)])
        with patch.object(config, "VERIFY_ENABLED", False):
            self.assertEqual(DetectionVerifier(576, 448).process(Mock(), raw, VISION_LABELS), [])

    def test_black_tracking_sends_legacy_id6_in03_04_and24(self):
        for mode in (s.CMD_SEARCH, s.CMD_TRACK, s.MODE_FINAL_ROI_TRACK):
            with self.subTest(mode=mode):
                current = controller(mode)
                self.command(current, s.CMD_SELECT_BLACK_OR_GREEN)
                current.mode = mode
                black = self.submit(current, [block(450, 100, 7)])
                self.assertEqual(black.class_id, 6)
                self.assertEqual(black.model_class_id, 7)
                self.assertEqual(current._latest_packet[2], 6)
                self.assertEqual(current._last_tracked_class_id, 6)

    def test_31_and51_commands_do_not_select_startzone(self):
        for command in (s.CMD_SELECT_BLACK_OBJECT, s.CMD_SELECT_BLACK_OR_GREEN):
            with self.subTest(command=command):
                current = controller(s.MODE_FINAL_ROI_TRACK)
                self.command(current, command)
                self.assertIsNone(self.submit(current, [block(288, 380, 6)]))
                self.assertIsNone(current._latest_packet)
                self.assertNotIn(s.build_event_packet(0x34), current._pending_events)
                selected = self.submit(current, [block(450, 100, 7)], 1100)
                self.assertEqual(selected.class_id, 6)

    def test_startzone_does_not_reject_green_or_red_as_close_neighbor(self):
        for class_id, label in ((4, "sqarered"), (5, "sqareredgreen"), (7, "triangualrblack")):
            with self.subTest(class_id=class_id):
                current = controller(s.MODE_FINAL_ROI_TRACK)
                objects = self.mapper.normalize([block(300, 200, class_id), block(301, 200, 6)])
                selected = current._select_final_roi_target(
                    objects, self.mapper.labels, label, 1000, check_front_block=False,
                )
                self.assertIsNotNone(selected)
                self.assertEqual(selected.class_id, 6 if class_id == 7 else class_id)

    def test_startzone_does_not_hide_red_as_false_safety_area(self):
        current = controller(s.CMD_SEARCH)
        self.command(current, s.CMD_SELECT_RED_OBJECT)
        start = block(100, 180, 6)
        start.w = start.h = 300
        selected = self.submit(current, [block(100, 180, 4), start])
        self.assertEqual(selected.class_id, 4)
        self.assertEqual(current._latest_packet[2], 4)

    def test_black_lock_associates_raw7_and_keeps_existing_loss_delay(self):
        current = controller(s.CMD_SEARCH)
        self.command(current, s.CMD_SELECT_BLACK_OR_GREEN)
        self.submit(current, [block(450, 100, 7)])
        for now in (1100, 1300, 1500, 1700):
            self.assertIsNone(self.submit(current, [block(450, 100, 6)], now))
            self.assertIsNotNone(current._search_target_lock)
            self.assertIsNone(current._latest_packet)
        recovered = self.submit(current, [block(460, 110, 7), block(460, 110, 6)], 1800,
                                verified=False)
        self.assertEqual(recovered.model_class_id, 7)
        self.assertEqual(current._latest_packet, s.build_coordinate_packet(6, 172, -114))
        self.assertIsNone(current._tracking_missing_since_ms)

    @patch.object(config, "UART_BLACK_TO_RED_PRIORITY_ENABLED", True)
    def test_real_black7_still_uses_red11_handshake_after_confirmed_loss(self):
        current = controller(s.CMD_SEARCH)
        self.command(current, s.CMD_SELECT_BLACK_OR_GREEN)
        self.submit(current, [block(450, 100, 7)])
        for now in range(1100, 2100, 100):
            self.submit(current, [block(100, 150, 4), block(450, 100, 6)], now,
                        verified=False)
        self.assertEqual(current.mode, s.MODE_WAIT_RED_SWITCH)
        self.assertIsNone(current._latest_packet)
        self.assertEqual(current._next_outgoing_packet(2100), s.build_event_packet(0x11))
        self.command(current, 0x11, 2200)
        self.command(current, 0x03, 2201)
        selected = self.submit(current, [block(105, 150, 4)], 2202, verified=False)
        self.assertEqual(selected.class_id, 4)
        self.assertEqual(current._latest_packet[2], 4)

    def test_near05_black7_with_startzone_is_one_supply_not_foreign_cargo(self):
        current = controller(s.MODE_FINAL_ROI_TRACK)
        self.command(current, s.CMD_SELECT_BLACK_OR_GREEN)
        decision = self.near_check(current, [7, 6])
        self.assertEqual(current.mode, s.MODE_WAIT_SUPPLY_SEARCH)
        self.assertEqual(current._supply_collect_attempts, 1)
        self.assertEqual(current._next_outgoing_packet(decision), s.build_event_packet(0x03))

    def test_near05_two_real_supplies_plus_startzone_requests_final05(self):
        current = controller(s.MODE_FINAL_ROI_TRACK)
        self.command(current, s.CMD_SELECT_BLACK_OR_GREEN)
        self.near_check(current, [5, 7, 6])
        self.assertEqual(current._zone_load_recheck_source, "before06")
        self.assertEqual(current._pending_events, [s.build_event_packet(0x05)])
        self.assertTrue(current._supply_collect_finished)

    def test_final05_black7_reports_supply_id5_then06(self):
        current = controller(s.MODE_FINAL_ROI_TRACK)
        self.command(current, s.CMD_SELECT_BLACK_OBJECT)
        self.near_check(current, [7, 6])
        self.assertEqual(current._zone_load_recheck_source, "before06")
        packet = current._next_outgoing_packet(2600)
        self.assertEqual(packet, s.build_event_packet(5))
        serial = Mock()
        serial.write.side_effect = lambda data: len(data)
        with patch.object(current, "_serial", serial), patch.object(s, "_now_ms", return_value=2600):
            current._write_packet(packet, current._outgoing_generation)
        for now in (3100, 3110, 3120):
            self.submit(current, [model_near_view_block(120, 200, 7, MODEL_LABELS),
                                  model_near_view_block(170, 200, 6, MODEL_LABELS)], now)
        packet = current._next_outgoing_packet(3120)
        self.assertEqual(packet, s.build_coordinate_packet(5, 0, 0))
        with patch.object(current, "_serial", serial), patch.object(s, "_now_ms", return_value=3120):
            current._write_packet(packet, current._outgoing_generation)
        self.assertEqual(current._next_outgoing_packet(3170), s.build_event_packet(6))

    def test_near05_startzone_only_counts_empty_not_black(self):
        current = controller(s.MODE_FINAL_ROI_TRACK)
        self.command(current, s.CMD_SELECT_BLACK_OR_GREEN)
        decision = self.near_check(current, [6])
        self.assertEqual(current.mode, s.MODE_FINAL_ROI_TRACK)
        self.assertEqual(current._next_outgoing_packet(decision), s.build_event_packet(0x24))
        self.assertEqual(current._supply_collect_attempts, 0)

    def test_debug_draws_black7_in_black_and_does_not_draw_startzone(self):
        img = Mock()
        mapped = self.mapper.normalize([block(100, 100, 6), block(200, 100, 7), block(300, 100, 5)])
        draw_debug_yolo_detections(img, mapped, 576, 448)
        self.assertEqual(img.draw_rect.call_count, 2)
        self.assertEqual([call.kwargs["color"] for call in img.draw_rect.call_args_list],
                         ["black", "green"])

    def test_main_normalizes_once_before_every_consumer_and_packages_mapper(self):
        root = Path(__file__).resolve().parent
        tree = ast.parse((root / "main.py").read_text(encoding="utf8"))
        calls = [node for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        normalize = [node for node in calls
                     if isinstance(node.func.value, ast.Name)
                     and node.func.value.id == "class_mapper" and node.func.attr == "normalize"]
        self.assertEqual(len(normalize), 1)
        detect = next(node for node in calls if node.func.attr == "detect")
        verify = next(node for node in calls if node.func.attr == "process"
                      and isinstance(node.func.value, ast.Name) and node.func.value.id == "verifier")
        submit = next(node for node in calls if node.func.attr == "submit")
        self.assertLess(detect.lineno, normalize[0].lineno)
        self.assertLess(normalize[0].lineno, verify.lineno)
        self.assertLess(normalize[0].lineno, submit.lineno)
        raw_argument = next(keyword.value for keyword in submit.keywords if keyword.arg == "raw_objects")
        self.assertEqual(raw_argument.id, "objects")
        manifest = (root / "app.yaml").read_text(encoding="utf8")
        self.assertIn("  - model_class_mapping.py", manifest)


class ReorderedSevenClassMappingTests(SingleFrameTrackingScenario):
    model_labels = REORDERED_SEVEN_LABELS

    submit = ModelClassMappingTests.submit
    command = ModelClassMappingTests.command
    near_check = ModelClassMappingTests.near_check

    def setUp(self):
        self.mapper = ModelClassMapper(REORDERED_SEVEN_LABELS)
        for setting in (patch.object(config, "VISION_DIAGNOSTICS_ENABLED", False),
                        patch("builtins.print")):
            setting.start()
            self.addCleanup(setting.stop)

    def test_exact_new_table_maps_to_nonzero_existing_uart_ids(self):
        raw = [block(100, 100, index) for index in range(7)]
        mapped = self.mapper.normalize(raw)
        self.assertEqual([obj.class_id for obj in mapped], [1, 2, 3, 5, 4, 6])
        self.assertEqual([self.mapper.labels[obj.class_id] for obj in mapped], [
            "bluesafety", "redsafety", "sqareblue", "sqareredgreen", "sqarered", "triangualrblack",
        ])
        self.assertEqual(self.mapper.summary(), "0->1 1->2 2->3 3->5 4->4 5->ignore 6->6")
        self.assertEqual(self.mapper.labels, tuple(LABELS))

    def test_model0_blue_safety_is_valid_not_the_deprecated_label(self):
        blue_zone = block(200, 100, 0)
        mapped = self.mapper.normalize([blue_zone])[0]
        self.assertEqual((mapped.class_id, mapped.model_class_id), (1, 0))
        self.assertEqual(blue_zone.class_id, 0)

    def test_green_alias_maps_by_label_even_when_model_order_changes(self):
        mapper = ModelClassMapper(["startzone", "sqaregreen", "sqareredgreen", "blue safety"])
        raw = [block(200, 100, index) for index in range(4)]
        mapped = mapper.normalize(raw)
        self.assertEqual([obj.class_id for obj in mapped], [5, 5])
        self.assertEqual([obj.model_class_id for obj in mapped], [1, 2])

    def test_remapping_keeps_source_geometry_and_native_ids_unchanged(self):
        for model_id, semantic_id in ((0, 1), (1, 2), (2, 3), (3, 5), (4, 4), (6, 6)):
            with self.subTest(model_id=model_id):
                source = block(190.5, 202.5, model_id)
                mapped = self.mapper.normalize([source])[0]
                self.assertEqual(source.class_id, model_id)
                self.assertEqual(mapped.class_id, semantic_id)
                for field in ("x", "y", "w", "h", "score"):
                    self.assertEqual(getattr(mapped, field), getattr(source, field))
                if model_id == semantic_id:
                    self.assertIs(mapped, source)
                else:
                    self.assertIsNot(mapped, source)
                    self.assertEqual(mapped.model_class_id, model_id)

    def test_new_green_runs_green_lab_verification_not_blue(self):
        verifier = DetectionVerifier(576, 448)
        img = Mock()
        img.find_blobs.return_value = [SimpleNamespace(
            pixels=lambda: 100, code=lambda: 1, density=lambda: .5,
        )]
        with patch.object(config, "VERIFY_COLOR_ENABLED", True), \
                patch.object(config, "VERIFY_SKIP_REDUNDANT_COLOR_ON_HIGH_CONFIDENCE", False):
            for _ in range(config.VERIFY_REQUIRED_HITS):
                confirmed = verifier.process(img, self.mapper.normalize([block(200, 100, 3)]), self.mapper.labels)
        self.assertEqual([obj.class_id for obj in confirmed], [5])
        self.assertEqual(set(verifier._class_states), {"sqareredgreen"})
        self.assertTrue(img.find_blobs.call_count)
        for call in img.find_blobs.call_args_list:
            self.assertEqual(call.args[0], [config.VERIFY_LAB_GREEN])

    def test_existing_target_commands_keep_their_original_meaning(self):
        for command, model_id, wire_id in ((0x01, 2, 3), (0x11, 4, 4), (0x21, 3, 5),
                                          (0x31, 6, 6), (0x51, 3, 5), (0x51, 6, 6)):
            with self.subTest(command=command, model_id=model_id):
                current = controller(s.CMD_SEARCH)
                self.command(current, command)
                selected = self.submit(current, [block(450, 100, model_id)])
                self.assertEqual(selected.class_id, wire_id)
                self.assertEqual(current._latest_packet[2], wire_id)

    def test_green_coordinates_and_raw_lock_recovery_in03_04_24(self):
        for mode in (s.CMD_SEARCH, s.CMD_TRACK, s.MODE_FINAL_ROI_TRACK):
            with self.subTest(mode=mode):
                current = controller(mode)
                selected = self.submit(current, [block(450, 100, 3)])
                self.assertEqual(selected.class_id, 5)
                self.assertEqual(current._latest_packet[2], 5)
                selected = self.submit(current, [block(460, 110, 3), block(451, 100, 2)],
                                       1100, verified=False)
                self.assertEqual(selected.class_id, 5)
                self.assertEqual(selected.model_class_id, 3)
                self.assertEqual(current._latest_packet[2], 5)

    def test_new_green03_to04_to14_to24_to34_keeps_green_identity(self):
        current = controller(s.CMD_SEARCH)
        self.submit(current, [block(288, 210, 3)])
        self.assertEqual(current.mode, s.MODE_WAIT_04_HANDSHAKE)
        self.assertEqual(current._track_target_lock[0], 5)
        self.command(current, 0x04, 1010)
        self.submit(current, [block(288, 224, 3)], 1020)
        self.assertEqual(current.mode, s.MODE_WAIT_14_HANDSHAKE)
        self.assertIn(s.build_event_packet(0x14), current._pending_events)
        self.command(current, 0x14, 1030)
        for now in (1040, 1050, 1060):
            self.submit(current, [block(288, 224, 3), block(289, 224, 5)], now)
        self.assertEqual(current.mode, s.MODE_FINAL_ROI_TRACK)
        self.assertEqual(current._final_target_lock[0], 5)
        self.assertIn(s.build_event_packet(0x24), current._pending_events)
        self.submit(current, [block(288, 390, 3)], 1070)
        self.assertIn(s.build_event_packet(0x34), current._pending_events)

    def test_startzone5_is_not_green_and_cannot_generate_coordinates_or34(self):
        for command in (0x21, 0x31, 0x51):
            with self.subTest(command=command):
                current = controller(s.MODE_FINAL_ROI_TRACK)
                self.command(current, command)
                current.mode = s.MODE_FINAL_ROI_TRACK
                self.assertIsNone(self.submit(current, [block(288, 380, 5)]))
                self.assertIsNone(current._latest_packet)
                self.assertNotIn(s.build_event_packet(0x34), current._pending_events)

    def test_startzone_is_not_a_neighbor_or_a_false_safety_zone(self):
        current = controller(s.MODE_FINAL_ROI_TRACK)
        raw = [block(300, 200, 3), block(301, 200, 5)]
        normalized = self.mapper.normalize(raw)
        selected = current._select_final_roi_target(normalized, self.mapper.labels, "sqareredgreen", 1000)
        self.assertIsNotNone(selected)
        self.assertEqual(selected.class_id, 5)

    def test_new_safety_ids_exclude_nearby_targets_for_both_colors(self):
        for model_id in (0, 1):
            with self.subTest(zone_model_id=model_id):
                current = controller(s.CMD_SEARCH)
                zone = SimpleNamespace(x=100, y=50, w=100, h=200, class_id=model_id, score=.9)
                selected = self.submit(current, [zone, block(250, 100, 3), block(450, 100, 3)])
                self.assertEqual(selected.class_id, 5)
                self.assertEqual(selected.x + selected.w / 2, 450)

    def test06_selects_largest_commanded_zone_and_sends_nonzero_zone_id(self):
        for zone_id, label, wire_id in ((0, "bluesafety", 1), (1, "redsafety", 2)):
            for target_label, expected_x in (("sqareredgreen", -148), ("sqarered", -58)):
                with self.subTest(zone=label, task=target_label):
                    current = controller(s.MODE_SEARCH_ZONE)
                    current.zone_label, current.target_label = label, target_label
                    selected_zone = SimpleNamespace(x=50, y=50, w=300, h=100, class_id=zone_id, score=.5)
                    smaller = SimpleNamespace(x=200, y=50, w=50, h=50, class_id=zone_id, score=.99)
                    other_color = SimpleNamespace(x=0, y=0, w=576, h=448, class_id=1-zone_id, score=.99)
                    selected = self.submit(current, [smaller, other_color, selected_zone], verified=False)
                    self.assertEqual((selected.x, selected.w, selected.class_id), (50, 300, wire_id))
                    self.assertEqual(current._latest_packet, s.build_coordinate_packet(wire_id, expected_x, -304))

    def test_first_green05_new_green_with_startzone_requests_final05(self):
        current = controller(s.MODE_FINAL_ROI_TRACK)
        self.near_check(current, [3, 5])
        self.assertEqual(current._zone_load_recheck_source, "before06")
        self.assertEqual(current._pending_events, [s.build_event_packet(5)])
        self.assertEqual(current._near_view_green_missing_cycles, 0)

    def test05_startzone_only_is_empty_not_a_green_load(self):
        current = controller(s.MODE_FINAL_ROI_TRACK)
        now = self.near_check(current, [5])
        self.assertEqual(current.mode, s.MODE_FINAL_ROI_TRACK)
        self.assertEqual(current._next_outgoing_packet(now), s.build_event_packet(0x24))
        self.assertEqual(current._near_view_green_missing_cycles, 1)

    def test05_new_blue_with_green_still_requires_arrangement(self):
        current = controller(s.MODE_FINAL_ROI_TRACK)
        self.near_check(current, [2, 3, 5])
        self.assertEqual(current.mode, s.MODE_ARRANGE_RIGHTMOST)
        self.assertIn(s.build_event_packet(0x02), current._pending_events)
        self.assertTrue(current._first_green_both_sides_required)

    def test05_new_green_and_black_are_two_supplies_and_report_id5_before06(self):
        current = controller(s.MODE_FINAL_ROI_TRACK)
        self.command(current, s.CMD_SELECT_BLACK_OR_GREEN)
        self.near_check(current, [3, 6, 5])
        self.assertEqual(current._zone_load_recheck_source, "before06")
        packet = current._next_outgoing_packet(2600)
        self.assertEqual(packet, s.build_event_packet(5))
        serial = Mock()
        serial.write.side_effect = lambda data: len(data)
        with patch.object(current, "_serial", serial), patch.object(s, "_now_ms", return_value=2600):
            current._write_packet(packet, current._outgoing_generation)
        for now in (3100, 3110, 3120):
            self.submit(current, [model_near_view_block(120, 200, 3, REORDERED_SEVEN_LABELS),
                                  model_near_view_block(170, 200, 6, REORDERED_SEVEN_LABELS),
                                  model_near_view_block(220, 200, 5, REORDERED_SEVEN_LABELS)], now)
        packet = current._next_outgoing_packet(3120)
        self.assertEqual(packet, s.build_coordinate_packet(5, 0, 0))
        with patch.object(current, "_serial", serial), patch.object(s, "_now_ms", return_value=3120):
            current._write_packet(packet, current._outgoing_generation)
        self.assertEqual(current._next_outgoing_packet(3170), s.build_event_packet(6))

    def test_debug_colors_follow_new_model_mapping_and_hide_startzone(self):
        img = Mock()
        normalized = self.mapper.normalize([block(200, 200, i) for i in range(7)])
        draw_debug_yolo_detections(img, normalized, 576, 448)
        self.assertEqual([call.kwargs["color"] for call in img.draw_rect.call_args_list],
                         ["blue", "red", (0, 191, 255), "green", "red", "black"])


class StartzonePreviewTests(unittest.TestCase):
    def test_preview_follows_label_across_old_new_and_custom_layouts(self):
        for labels in (MODEL_LABELS, REORDERED_SEVEN_LABELS,
                       REORDERED_SEVEN_LABELS + ["blackbar"],
                       ["startzone", "sqaregreen"], ["other"] * 8 + ["startzone"]):
            mapper = ModelClassMapper(labels)
            model_id = labels.index("startzone")
            native = block(100, 200, model_id)
            mapped = mapper.startzones([native])
            self.assertEqual(len(mapped), 1)
            self.assertEqual(mapped[0].class_id, 8)
            self.assertEqual((mapped[0].x, mapped[0].y, mapped[0].w, mapped[0].h, mapped[0].score),
                             (native.x, native.y, native.w, native.h, native.score))
            self.assertEqual(native.class_id, model_id)
            self.assertEqual(mapper.normalize([native]), [])
            self.assertEqual(mapper.blackbars([native]), [])
            self.assertNotIn("startzone", mapper.labels)

    def test_legacy_models_without_startzone_have_no_preview(self):
        mapper = ModelClassMapper(LABELS)
        self.assertEqual(mapper.startzones([block(100, 200, i) for i in range(7)]), [])

    def test_source_startzone5_never_replaces_actual_green5_or_bar7(self):
        mapper = ModelClassMapper(REORDERED_SEVEN_LABELS + ["blackbar"])
        green, start, bar = block(100, 200, 3), block(200, 200, 5), block(300, 200, 7)
        self.assertEqual([obj.class_id for obj in mapper.normalize([green, start, bar])], [5])
        self.assertEqual([obj.class_id for obj in mapper.startzones([green, start, bar])], [8])
        self.assertEqual([obj.class_id for obj in mapper.blackbars([green, start, bar])], [7])
        self.assertEqual([green.class_id, start.class_id, bar.class_id], [3, 5, 7])

    def test_preview_confidence_uses_default_finite_threshold(self):
        gate = DetectionConfidenceFilter(.5, .7, .4)
        objects = [block(100, 200, 8) for _ in range(6)]
        for obj, score in zip(objects, (.4999, .5, .7, float("nan"), float("inf"), -.1)):
            obj.score = score
        self.assertEqual(gate.apply_startzones(objects), objects[1:3])
        self.assertEqual(gate.apply(objects), [])
        self.assertEqual(gate.apply_blackbars(objects), [])
        self.assertEqual(gate.apply_startzones([block(100, 200, 5), block(100, 200, 7)]), [])
        self.assertEqual(DetectionConfidenceFilter(.6).apply_startzones([objects[1]]), [])

    def test_preview_is_gray_original_clipped_box_with_black_area_label(self):
        import vision_overlay as overlay
        obj = SimpleNamespace(x=-10, y=20, w=100, h=80, class_id=8, score=.9)
        img = Mock()
        overlay.draw_debug_yolo_detections(img, [obj], 576, 448)
        img.draw_rect.assert_called_once_with(0, 20, 90, 80, color=(128, 128, 128), thickness=3)
        overlay.draw_debug_object_area(img, [obj], 576, 448)
        self.assertEqual(img.draw_string.call_args.args[2], "START AREA: 2.8%")
        self.assertEqual(img.draw_string.call_args.kwargs, dict(color="black", scale=1.2, thickness=1))
        self.assertEqual((obj.x, obj.w), (-10, 100))

    def test_startzone_percentage_is_visible_box_over_whole_image_not_confidence(self):
        import vision_overlay as overlay
        for rect, expected in (((100, 100, 288, 224), "25.0%"),
                               ((-288, 0, 576, 448), "50.0%"),
                               ((0, 0, 576, 448), "100.0%")):
            obj = SimpleNamespace(x=rect[0], y=rect[1], w=rect[2], h=rect[3], class_id=8, score=.91)
            img = Mock()
            overlay.draw_debug_object_area(img, [obj], 576, 448)
            self.assertEqual(img.draw_string.call_args.args[2], "START AREA: " + expected)
            self.assertGreaterEqual(img.draw_string.call_args.args[1], 84)
            self.assertLessEqual(img.draw_string.call_args.args[0], 576-224)

    def test_startzone_area_missing_invalid_offscreen_and_blackbar_remain_hidden(self):
        import vision_overlay as overlay
        objects = [SimpleNamespace(x=800, y=20, w=100, h=100, class_id=8),
                   SimpleNamespace(x=10, y=20, w=0, h=100, class_id=8),
                   SimpleNamespace(x=float("nan"), y=20, w=100, h=100, class_id=8),
                   block(100, 200, 7)]
        img = Mock()
        overlay.draw_debug_object_area(img, objects, 576, 448)
        img.draw_string.assert_not_called()
        overlay.draw_debug_object_area(img, [block(100, 200, 8)], 0, 448)
        img.draw_string.assert_not_called()

    def test_preview_does_not_become_cargo_neighbor_or_false_safety_zone(self):
        from target_selector import directional_target_neighbors
        current = controller(s.MODE_NEAR_VIEW_CHECK)
        preview, target = block(288, 224, 8), block(288, 224, 5)
        self.assertEqual(current._near_view_target_ids([preview], LABELS), [])
        self.assertEqual(current._movable_target_ids([preview], LABELS), [])
        self.assertEqual(directional_target_neighbors(target, [target, preview], 576, 448), ([], []))
        self.assertEqual(current._exclude_objects_in_safety_zones([target], LABELS, [preview]), [target])


if __name__ == "__main__":
    unittest.main()
