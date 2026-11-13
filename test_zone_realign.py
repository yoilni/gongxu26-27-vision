"""Auxiliary YOLO blackbar and06/46 one-correction delivery handshake."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from test_vision_rules import LABELS, block, controller, near_view_block, confirm_final_load_check, SingleFrameTrackingScenario
from test_model_class_mapping import MODEL_LABELS, REORDERED_SEVEN_LABELS
from detection_confidence import DetectionConfidenceFilter
from model_class_mapping import ModelClassMapper, VISION_LABELS
from zone_blackbar import blackbar_in_zone, blackbar_side, select_zone_blackbar
import app_config as config
import serial_controller as s
import vision_overlay as overlay


def box(x, y, w, h, class_id=1, score=.9):
    return SimpleNamespace(x=x, y=y, w=w, h=h, class_id=class_id, score=score)


class AuxiliaryBlackbarTests(unittest.TestCase):
    def test_new_label_is_auxiliary_and_never_changes_cargo_ids(self):
        mapper = ModelClassMapper(REORDERED_SEVEN_LABELS + ["blackbar"])
        raw = [block(100, 100, i) for i in range(8)]
        self.assertEqual([o.class_id for o in mapper.normalize(raw)], [1, 2, 3, 5, 4, 6])
        self.assertEqual(mapper.labels, VISION_LABELS)
        self.assertEqual(mapper.blackbars(raw), [raw[7]])
        self.assertIn("7->blackbar(aux)", mapper.summary())
        self.assertEqual([o.class_id for o in raw], list(range(8)))

    def test_old_triangle_at_native7_is_not_a_bar(self):
        for labels in (LABELS, MODEL_LABELS, REORDERED_SEVEN_LABELS):
            mapper = ModelClassMapper(labels)
            raw = [block(100, 100, i) for i in range(len(labels))]
            self.assertEqual(mapper.blackbars(raw), [])
            self.assertEqual(mapper.normalize(raw)[-1].class_id, 6)

    def test_auxiliary_mapping_is_by_label_not_native_index(self):
        mapper = ModelClassMapper(["blackbar", "startzone", "triangualrblack"])
        raw = [block(100, 100, i) for i in range(3)]
        mapped = mapper.blackbars(raw)
        self.assertEqual([o.class_id for o in mapped], [7])
        self.assertEqual(mapped[0].model_class_id, 0)
        self.assertEqual(mapper.normalize(raw)[0].class_id, 6)
        self.assertEqual(raw[0].class_id, 0)

    def test_bar_confidence_uses_default_not_triangle_gate(self):
        gate = DetectionConfidenceFilter(.5, .7, .4)
        scores = (.4999, .5, .6999, .7, float("nan"), float("inf"))
        bars = [box(10, 10, 5, 40, 7, score) for score in scores]
        self.assertEqual(gate.apply_blackbars(bars), bars[1:4])
        self.assertEqual(gate.apply(bars), [])
        triangle = box(10, 10, 5, 40, 6, .6999)
        self.assertEqual(gate.apply([triangle]), [])
        self.assertEqual(gate.apply_blackbars([triangle]), [])
        gate = DetectionConfidenceFilter(.6)
        self.assertEqual(gate.apply_blackbars([bars[1]]), [])

    def test_bar_has_no_inventory_spacing_or_uart_cargo_role(self):
        current = controller(s.CMD_TRACK)
        bar = box(200, 100, 20, 300, 7)
        with patch("builtins.print"), patch.object(config, "VISION_DIAGNOSTICS_ENABLED", False):
            self.assertEqual(current._near_view_target_ids([bar], LABELS), [])
            self.assertEqual(current._movable_target_ids([bar], LABELS), [])
            self.assertIsNone(current.submit([bar], LABELS, raw_objects=[bar], blackbar_objects=[bar]))
        self.assertIsNone(current._latest_packet)

    def test_debug_auxiliary_border_is_yellow_no_cargo_area_label(self):
        img = Mock()
        bar = box(250, 100, 10, 250, 7)
        overlay.draw_debug_yolo_detections(img, [bar], 576, 448)
        img.draw_rect.assert_called_once_with(250, 100, 10, 250, color=overlay.SELECTED_COLOR, thickness=3)
        overlay.draw_debug_object_area(img, [bar], 576, 448)
        img.draw_string.assert_not_called()

    def test_packaged_without_restoring_cv_line_feature(self):
        root = Path(__file__).resolve().parent
        manifest = (root / "app.yaml").read_text(encoding="utf8")
        self.assertIn("  - zone_blackbar.py", manifest)
        self.assertIn("  - test_zone_realign.py", manifest)
        source = (root / "zone_blackbar.py").read_text(encoding="utf8")
        for forbidden in ("import cv2", "import numpy", "get_pixel", "find_lines", "find_blobs"):
            self.assertNotIn(forbidden, source)


class BlackbarGeometryTests(unittest.TestCase):
    def test_reference_is_zone_box_center_not_frame_or_clipped_center(self):
        zone = box(-100, 0, 400, 448)
        # Raw zone center100; visible center150; frame center288.
        bar = box(115, 100, 10, 100, 7)
        self.assertTrue(blackbar_in_zone(bar, zone, 576, 448))
        self.assertEqual(blackbar_side(bar, zone), 1)
        self.assertEqual(blackbar_side(box(95, 100, 10, 100, 7), zone), 0)
        self.assertEqual(blackbar_side(box(75, 100, 10, 100, 7), zone), -1)
        self.assertEqual(blackbar_side(None, zone), 0)

    def test_bar_center_must_be_in_same_current_visible_zone(self):
        zone = box(100, 100, 300, 250)
        for x, y, expected in ((100, 100, True), (399.99, 349.99, True),
                                (400, 200, False), (200, 350, False),
                                (99.99, 200, False), (200, 99.99, False)):
            self.assertEqual(blackbar_in_zone(block(x, y, 7), zone, 576, 448), expected)
        for invalid in (box(100, 100, 0, 20, 7), box(800, 100, 10, 20, 7),
                        box(float("nan"), 100, 10, 20, 7), block(200, 200, 6)):
            self.assertFalse(blackbar_in_zone(invalid, zone, 576, 448))

    def test_multiple_bars_choose_closest_to_zone_center_only(self):
        zone = box(100, 100, 300, 250)
        left, right, outside = block(130, 200, 7), block(270, 200, 7), block(450, 200, 7)
        for bars in ([left, right, outside], [outside, right, left]):
            self.assertIs(select_zone_blackbar(bars, zone, 576, 448), right)
        self.assertIsNone(select_zone_blackbar([outside], zone, 576, 448))
        self.assertIsNone(select_zone_blackbar([right], None, 576, 448))


class ZoneCloseFiveFrameTests(unittest.TestCase):
    """Real default5 close frames; legacy one-frame fixtures stay separate."""
    def setUp(self):
        for setting in (patch("builtins.print"), patch.object(config, "VISION_DIAGNOSTICS_ENABLED", False)):
            setting.start()
            self.addCleanup(setting.stop)
        self.current = controller(s.MODE_SEARCH_ZONE)
        self.current.frame_width = self.current.frame_height = 100
        self.current.tracking_roi = [0, 80, 100, 20]
        self.current.zone_label = LABELS[1]
        self.current._serial = SimpleNamespace(write=Mock(side_effect=lambda packet: len(packet)))
        self.zone = box(0, 0, 100, 50)
        self.left, self.right = box(20, 10, 10, 30, 7), box(70, 10, 10, 30, 7)

    def submit(self, bars=(), now=1000, zones=None, generation=None):
        with patch.object(s, "_now_ms", return_value=now):
            return self.current.submit([], LABELS, raw_objects=[self.zone] if zones is None else zones,
                                       blackbar_objects=bars, frame_generation=(
                                           self.current.frame_generation() if generation is None else generation))

    def write(self, packet, now):
        with patch.object(s, "_now_ms", return_value=now):
            self.current._write_packet(packet, self.current.frame_generation())

    def observe_five(self, bars=(), start=1000):
        for index in range(5):
            self.submit(bars, start + index * 50)

    def highlight(self, bars, zones=None, generation=None):
        return self.current.debug_zone_realign_bar(
            [self.zone] if zones is None else zones, LABELS, bars,
            frame_generation=self.current.frame_generation() if generation is None else generation)

    def test_default_five_all_missing_bar_then26_and_first_four_keep_coordinates(self):
        self.assertEqual(config.UART_ZONE_CLOSE_CONFIRM_FRAMES, 5)
        for index in range(5):
            now = 1000 + index * 50
            self.submit(now=now)
            self.assertEqual(self.current._zone_close_confirm_frames, index + 1)
            self.assertIsNotNone(self.current._latest_packet)
            self.assertEqual(self.current._zone_close_active, index == 4)
            if index == 0:
                self.assertEqual(self.current._next_outgoing_packet(now), s.build_event_packet(0x16))
            outgoing = self.current._next_outgoing_packet(now)
            self.assertEqual(outgoing, s.build_event_packet(0x26) if index == 4 else self.current._latest_packet)

    def test_five_correct_centered_or_mixed_missing_bar_frames_allow26(self):
        for pattern in ((0, 0, 0, 0, 0), (1, 1, 1, 1, 1), (0, 1, 2, 0, 1)):
            self.current._reset_zone_approach()
            self.current._pending_events.clear()
            for index, choice in enumerate(pattern):
                bars = ([], [self.right], [box(45, 10, 10, 30, 7)])[choice]
                self.submit(bars, 1000 + index * 50)
                self.assertEqual(s.build_event_packet(0x26) in self.current._pending_events, index == 4)
            self.assertNotIn(s.build_event_packet(0x46), self.current._pending_events)

    def test_one_wrong_bar_in_any_of_five_frames_requests46_not26(self):
        for wrong_index in range(5):
            self.current._reset_zone_approach()
            self.current._pending_events.clear()
            for index in range(wrong_index + 1):
                self.submit([self.left] if index == wrong_index else [], 1000 + index * 50)
            self.assertEqual(self.current._zone_realign_phase, "pending")
            self.assertIn(s.build_event_packet(0x46), self.current._pending_events)
            self.assertNotIn(s.build_event_packet(0x26), self.current._pending_events)
            self.assertIsNone(self.current._latest_packet)
            self.submit([self.right], 1050 + wrong_index * 50)
            self.assertEqual(self.current._zone_realign_phase, "pending")
            self.assertNotIn(s.build_event_packet(0x26), self.current._pending_events)

    def test_red_and_all_supply_colors_use_inverse46_direction(self):
        for target in (LABELS[3], LABELS[4], LABELS[5], LABELS[6], s.TARGET_BLACK_OR_GREEN):
            for zone_id in (1, 2):
                self.current._reset_zone_approach()
                self.current._pending_events.clear()
                self.current.target_label, self.current.zone_label = target, LABELS[zone_id]
                self.zone.class_id = zone_id
                safe, wrong = (self.left, self.right) if target == LABELS[4] else (self.right, self.left)
                self.submit([safe])
                self.assertEqual(self.current._zone_close_confirm_frames, 1)
                self.submit([wrong], 1050)
                self.assertIn(s.build_event_packet(0x46), self.current._pending_events)
                self.assertNotIn(s.build_event_packet(0x26), self.current._pending_events)

    def test_small_missing_wrong_color_and_stale_gap_restart_five(self):
        for interruption in ("small", "missing", "wrong_color", "gap"):
            self.current._reset_zone_approach()
            self.current._pending_events.clear()
            for now in (1000, 1050, 1100):
                self.submit(now=now)
            if interruption != "gap":
                zones = ([box(0, 0, 100, 49)] if interruption == "small" else []
                         if interruption == "missing" else [box(0, 0, 100, 50, 2)])
                self.submit(now=1150, zones=zones)
                self.assertEqual(self.current._zone_close_confirm_frames, 0)
            for index in range(5):
                self.submit(now=1401 + index * 50)
                self.assertEqual(self.current._zone_close_confirm_frames, index + 1)
                self.assertEqual(self.current._zone_close_active, index == 4)

    def test_timers_without_new_images_never_complete_five(self):
        for now in (1000, 1050, 1100, 1150):
            self.submit(now=now)
        for now in (1200, 1300, 1500, 3000):
            self.assertNotEqual(self.current._next_outgoing_packet(now), s.build_event_packet(0x26))
        self.assertEqual(self.current._zone_close_confirm_frames, 4)

    def test_actual26_skips_reconfirmation_repeats_and_keeps_coordinates(self):
        self.observe_five()
        self.write(self.current._next_outgoing_packet(1200), 1200)  #16
        self.write(self.current._next_outgoing_packet(1250), 1250)  #26
        self.assertEqual(self.current._zone_realign_phase, "done")
        self.submit([self.left], 1750)
        self.assertEqual(self.current._next_outgoing_packet(1750), s.build_event_packet(0x26))
        self.assertEqual(len(self.current._next_outgoing_packet(1750)), 11)
        self.assertNotIn(s.build_event_packet(0x46), self.current._pending_events)
        self.assertIsNone(self.highlight([self.left]))

    def test_mcu46_completion_keeps_existing_immediate26_rule_and_fresh_frame(self):
        self.submit([self.left])
        self.write(self.current._next_outgoing_packet(1000), 1000)
        self.write(self.current._next_outgoing_packet(1050), 1050)
        old_generation = self.current.frame_generation()
        with patch.object(s, "_now_ms", return_value=1600):
            self.current.process_received_bytes(s.build_command_packet(0x46))
        self.assertIsNone(self.current._next_outgoing_packet(1600))
        self.submit([self.left], 1610, generation=old_generation)
        self.assertFalse(self.current._zone_close_active)
        self.submit([self.left], 1620)
        self.assertTrue(self.current._zone_close_active)
        self.assertEqual(self.current._next_outgoing_packet(1620), s.build_event_packet(0x16))
        self.assertEqual(self.current._next_outgoing_packet(1670), s.build_event_packet(0x26))

    def test_late_wrong_bar_cancels_prepared26_before_actual_write(self):
        self.observe_five()
        self.current._next_outgoing_packet(1200)
        prepared = self.current._next_outgoing_packet(1250)
        self.assertEqual(prepared, s.build_event_packet(0x26))
        self.submit([self.left], 1260)
        self.write(prepared, 1260)
        self.current._serial.write.assert_not_called()
        self.assertIn(s.build_event_packet(0x46), self.current._pending_events)

    def test_fe_discards_confirmation_and_old_frames(self):
        for now in (1000, 1050, 1100, 1150):
            self.submit(now=now)
        generation = self.current.frame_generation()
        with patch.object(s, "_now_ms", return_value=1200):
            self.current.process_received_bytes(s.build_command_packet(0xFE))
        self.assertEqual(self.current._zone_close_confirm_frames, 0)
        self.submit(now=1250, generation=generation)
        self.assertEqual(self.current._zone_close_confirm_frames, 0)
        self.assertIsNone(self.current._next_outgoing_packet(1250))

    def test_missing_confirm_config_defaults_to_five(self):
        with patch.dict(vars(config), {}, clear=False):
            delattr(config, "UART_ZONE_CLOSE_CONFIRM_FRAMES")
            self.observe_five()
            self.assertEqual(self.current._zone_close_confirm_frames, 5)
            self.assertIn(s.build_event_packet(0x26), self.current._pending_events)

    def test_debug_red_highlight_obeys_area_phase_color_side_and_generation(self):
        for target in (LABELS[4], LABELS[5], LABELS[6], s.TARGET_BLACK_OR_GREEN):
            self.current.target_label = target
            wrong, safe = (self.right, self.left) if target == LABELS[4] else (self.left, self.right)
            self.assertIs(self.highlight([wrong]), wrong)
            self.assertIsNone(self.highlight([safe]))
            self.assertIsNone(self.highlight([wrong], zones=[box(0, 0, 100, 49)]))
            self.assertIsNone(self.highlight([wrong], zones=[]))
            self.assertIsNone(self.highlight([wrong], zones=[box(0, 0, 100, 50, 2)]))
            self.assertIsNone(self.highlight([wrong], generation=-1))
            self.current._zone_realign_phase = "done"
            self.assertIsNone(self.highlight([wrong]))
            self.current._zone_realign_phase = None
        for mode in (s.CMD_SEARCH, s.CMD_TRACK, s.MODE_FINAL_ROI_TRACK, s.MODE_NEAR_VIEW_CHECK):
            self.current.mode = mode
            self.assertIsNone(self.highlight([self.left]))

    def test_debug_uses_same_largest_zone_and_nearest_valid_bar_as_uart(self):
        smaller = box(0, 60, 100, 40)
        outside = box(20, 65, 10, 30, 7)
        closer_right = box(50, 10, 10, 30, 7)
        self.assertIsNone(self.highlight([outside], zones=[smaller, self.zone]))
        self.assertIsNone(self.highlight([self.left, closer_right]))
        self.assertIs(self.highlight([self.left, self.right]), self.left)
        self.assertIsNone(self.highlight([box(45, 10, 10, 30, 7)]))

    def test_debug_is_read_only_and_changes_only_one_bar_to_red(self):
        before = dict(vars(self.current))
        selected = self.highlight([self.left, self.right])
        self.assertEqual(vars(self.current), before)
        img = Mock()
        overlay.draw_debug_yolo_detections(img, [self.left, self.right], 100, 100, realign_bar=selected)
        self.assertEqual([call.kwargs['color'] for call in img.draw_rect.call_args_list],
                         ["red", overlay.SELECTED_COLOR])
        img.draw_string.assert_not_called()


class DebugRealignPointTests(unittest.TestCase):
    def test_default_roi_center_plus165_minus90_is453_293half(self):
        roi = overlay.make_bottom_center_roi(576, 448)
        original = list(roi)
        img = Mock()
        overlay.draw_debug_zone_realign_point(img, roi, 576, 448)
        yellow = [call.args for call in img.draw_line.call_args_list
                  if call.kwargs["color"] == overlay.SELECTED_COLOR]
        self.assertEqual(yellow, [(445, 294, 461, 294), (453, 286, 453, 302)])
        self.assertEqual(img.draw_line.call_count, 4)
        img.draw_string.assert_called_once_with(
            465, 274, "46", color="black", scale=1.2, thickness=1)
        self.assertEqual(roi, original)

    def test_point_follows_custom_roi_not_image_center_or_hardcoded_pixels(self):
        img = Mock()
        overlay.draw_debug_zone_realign_point(img, [40, 250, 400, 100], 576, 448)
        self.assertEqual(img.draw_line.call_args_list[2].args, (397, 210, 413, 210))
        self.assertEqual(img.draw_line.call_args_list[3].args, (405, 202, 405, 218))

    def test_missing_offset_settings_use_same165_minus90_defaults(self):
        img = Mock()
        with patch.object(overlay, "config", SimpleNamespace()):
            overlay.draw_debug_zone_realign_point(img, [0, 319, 576, 129], 576, 448)
        self.assertEqual(img.draw_line.call_args_list[2].args, (445, 294, 461, 294))

    def test_offsets_are_display_only_and_field_tunable(self):
        img = Mock()
        with patch.object(config, "DEBUG_ZONE_REALIGN_POINT_OFFSET_X", -60), \
                patch.object(config, "DEBUG_ZONE_REALIGN_POINT_OFFSET_Y", 60):
            overlay.draw_debug_zone_realign_point(img, [0, 319, 576, 129], 576, 448)
        self.assertEqual(img.draw_line.call_args_list[2].args, (220, 444, 236, 444))
        self.assertEqual(img.draw_line.call_args_list[3].args, (228, 436, 228, 447))

    def test_invalid_or_missing_roi_and_empty_image_do_not_draw(self):
        for roi in (None, [], [0, 319, 0, 129], [0, 319, 576, -1],
                    [0, float("nan"), 576, 129], [0, 0, float("inf"), 129]):
            with self.subTest(roi=roi):
                img = Mock()
                overlay.draw_debug_zone_realign_point(img, roi, 576, 448)
                self.assertEqual(img.mock_calls, [])
        for width, height in ((0, 448), (576, 0), (-1, 448)):
            img = Mock()
            overlay.draw_debug_zone_realign_point(img, [0, 319, 576, 129], width, height)
            self.assertEqual(img.mock_calls, [])

    def test_offscreen_reference_is_not_silently_moved_to_another_point(self):
        img = Mock()
        overlay.draw_debug_zone_realign_point(img, [0, 0, 40, 40], 576, 448)
        self.assertEqual(img.mock_calls, [])

    def test_cross_endpoints_and_label_stay_inside_image(self):
        img = Mock()
        with patch.object(config, "DEBUG_ZONE_REALIGN_POINT_OFFSET_X", 0), \
                patch.object(config, "DEBUG_ZONE_REALIGN_POINT_OFFSET_Y", 0):
            overlay.draw_debug_zone_realign_point(img, [0, 0, 2, 2], 576, 448)
        for call in img.draw_line.call_args_list:
            x1, y1, x2, y2 = call.args
            self.assertTrue(0 <= x1 < 576 and 0 <= x2 < 576)
            self.assertTrue(0 <= y1 < 448 and 0 <= y2 < 448)
        self.assertEqual(img.draw_string.call_args.args[:2], (13, 0))


class ZoneRealignTests(SingleFrameTrackingScenario):
    def setUp(self):
        for setting in (patch("builtins.print"), patch.object(config, "VISION_DIAGNOSTICS_ENABLED", False)):
            setting.start()
            self.addCleanup(setting.stop)
        self.current = controller(s.MODE_SEARCH_ZONE)
        self.current.zone_label = LABELS[1]
        self.current.frame_width = self.current.frame_height = 100
        self.current.tracking_roi = [0, 80, 100, 20]
        self.current._serial = SimpleNamespace(write=Mock(side_effect=lambda packet: len(packet)))
        self.zone = box(0, 0, 100, 50)  # EXACT50%, not51%.
        self.left = box(20, 10, 10, 30, 7)
        self.right = box(70, 10, 10, 30, 7)

    def submit(self, bars=(), zones=None, now=1000, generation=None):
        with patch.object(s, "_now_ms", return_value=now):
            return self.current.submit([], LABELS, raw_objects=[self.zone] if zones is None else zones,
                                       blackbar_objects=bars, frame_generation=(
                                           self.current.frame_generation() if generation is None else generation))

    def write(self, packet, now=1000, generation=None):
        with patch.object(s, "_now_ms", return_value=now):
            self.current._write_packet(packet, self.current.frame_generation() if generation is None else generation)

    def command(self, command, now=1000):
        with patch.object(s, "_now_ms", return_value=now):
            self.current.process_received_bytes(s.build_command_packet(command))

    def start_maneuver(self, now=1000):
        self.submit([self.left], now=now)
        self.write(self.current._next_outgoing_packet(now), now)
        event = self.current._next_outgoing_packet(now + 50)
        self.assertEqual(event, s.build_event_packet(0x46))
        self.write(event, now + 50)
        self.assertEqual(self.current._zone_realign_phase, "wait_ack")

    def test_wrong_side_table_for_all_cargo_and_zone_colors(self):
        for target in (LABELS[3], LABELS[4], LABELS[5], LABELS[6], s.TARGET_BLACK_OR_GREEN):
            for zone_id in (1, 2):
                for side, bar in ((-1, self.left), (1, self.right)):
                    with self.subTest(target=target, zone=zone_id, side=side):
                        self.setUp_current(target, zone_id)
                        self.submit([bar])
                        self.assertEqual(self.current._next_outgoing_packet(1000), s.build_event_packet(0x16))
                        wrong = side == (1 if target == LABELS[4] else -1)
                        expected = 0x46 if wrong else 0x26
                        self.assertEqual(self.current._next_outgoing_packet(1050), s.build_event_packet(expected))
                        self.assertEqual(self.current._latest_packet is None, wrong)

    def setUp_current(self, target, zone_id):
        self.current = controller(s.MODE_SEARCH_ZONE)
        self.current.target_label = target
        self.current.zone_label = LABELS[zone_id]
        self.current.frame_width = self.current.frame_height = 100
        self.current.tracking_roi = [0, 80, 100, 20]
        self.current._serial = SimpleNamespace(write=Mock(side_effect=lambda packet: len(packet)))
        self.zone.class_id = zone_id

    def test_missing_centered_invalid_or_outside_bar_falls_back_to26(self):
        for bars in ([], [box(45, 10, 10, 30, 7)], [box(20, 60, 10, 30, 7)],
                     [box(20, 10, 0, 30, 7)], [box(20, 10, 10, 30, 6)]):
            self.setUp_current(LABELS[5], 1)
            self.submit(bars)
            self.assertEqual(self.current._next_outgoing_packet(1000), s.build_event_packet(0x16))
            self.assertEqual(self.current._next_outgoing_packet(1050), s.build_event_packet(0x26))
            self.assertEqual(len(self.current._next_outgoing_packet(1050)), 11)

    def test_below50_does_not_realign_and_keeps_normal_coordinates(self):
        self.zone.h = 49.999
        self.submit([self.left])
        self.assertIsNone(self.current._zone_realign_phase)
        self.assertEqual(self.current._next_outgoing_packet(1000), s.build_event_packet(0x16))
        self.assertEqual(len(self.current._next_outgoing_packet(1050)), 11)

    def test_only_selected_largest_commanded_zone_supplies_bar(self):
        other = box(0, 60, 100, 40, 1)
        wrong_color = box(0, 0, 100, 100, 2)
        # Wrong bar is in smaller selected-color zone, right bar in largest.
        bar = box(20, 65, 10, 30, 7)
        self.assertIs(self.submit([bar, self.right], [other, wrong_color, self.zone]), self.zone)
        self.assertIn(s.build_event_packet(0x26), self.current._pending_events)
        self.assertNotIn(s.build_event_packet(0x46), self.current._pending_events)

    def test_prepare_is_not_send_and_unsolicited_ack_cannot_skip_decision(self):
        self.submit([self.left])
        self.current._next_outgoing_packet(1000)  #16
        prepared = self.current._next_outgoing_packet(1050)
        generation = self.current.frame_generation()
        self.command(0x46, 1050)
        self.assertEqual(self.current._zone_realign_phase, "pending")
        self.assertEqual(self.current.frame_generation(), generation)
        self.write(prepared, 1060)
        self.assertEqual(self.current._zone_realign_phase, "wait_ack")

    def test_one_wrong_frame_latches_pending46_despite_missing_correct_or_small_observation(self):
        for replacement in ("no_bar", "correct", "small", "no_zone"):
            self.setUp_current(LABELS[5], 1)
            self.submit([self.left])
            self.current._next_outgoing_packet(1000)
            prepared = self.current._next_outgoing_packet(1050)
            bars = [] if replacement == "no_bar" else [self.right] if replacement == "correct" else [self.left]
            zones = [] if replacement == "no_zone" else [self.zone]
            if replacement == "small":
                zones = [box(0, 0, 100, 49)]
            self.submit(bars, zones, 1060)
            self.write(prepared, 1060)
            self.current._serial.write.assert_called_once_with(prepared)
            self.assertEqual(self.current._zone_realign_phase, "wait_ack")
            self.assertNotIn(s.build_event_packet(0x26), self.current._pending_events)

    def test_pending46_expires_if_camera_stops_before_write(self):
        self.submit([self.left])
        self.current._next_outgoing_packet(1000)
        prepared = self.current._next_outgoing_packet(1050)
        now = 1001 + config.UART_TARGET_FRESHNESS_MS
        self.write(prepared, now)
        self.current._serial.write.assert_not_called()
        self.assertIsNone(self.current._next_outgoing_packet(now))
        self.assertIsNone(self.current._zone_realign_phase)

    def test_short_or_failed_first_write_retries_without_premature_wait(self):
        for sent in (-1, 3):
            self.setUp_current(LABELS[5], 1)
            self.submit([self.left])
            self.current._next_outgoing_packet(1000)
            event = self.current._next_outgoing_packet(1050)
            self.current._serial.write = Mock(return_value=sent)
            generation = self.current.frame_generation()
            self.write(event, 1050)
            self.assertEqual(self.current._zone_realign_phase, "pending")
            self.assertIsNone(self.current._last_zone_realign_write_ms)
            self.assertEqual(self.current.frame_generation(), generation)
            self.command(0x46)
            self.assertEqual(self.current._zone_realign_phase, "pending")
            self.assertEqual(self.current._next_outgoing_packet(1060), event)
            self.current._serial.write = Mock(return_value=5)
            self.write(event, 1070)
            self.assertEqual(self.current._zone_realign_phase, "wait_ack")

    def test_retry500ms_uses_actual_write_time_and_never_restarts_generation(self):
        self.start_maneuver()
        generation = self.current.frame_generation()
        self.assertIsNone(self.current._next_outgoing_packet(1549))
        event = self.current._next_outgoing_packet(1550)
        self.assertEqual(event, s.build_event_packet(0x46))
        self.write(event, 1600)
        self.assertEqual(self.current._last_zone_realign_write_ms, 1600)
        self.assertEqual(self.current.frame_generation(), generation)
        self.assertIsNone(self.current._next_outgoing_packet(2099))
        self.assertEqual(self.current._next_outgoing_packet(2100), event)

    def test_retry_write_failure_keeps_wait_and_original_deadline(self):
        self.start_maneuver()
        self.current._serial.write = Mock(return_value=-1)
        self.write(self.current._next_outgoing_packet(1550), 1600)
        self.assertEqual(self.current._zone_realign_phase, "wait_ack")
        self.assertEqual(self.current._last_zone_realign_write_ms, 1050)
        self.assertEqual(self.current._next_outgoing_packet(1601), s.build_event_packet(0x46))

    def test_wait_ack_blocks_coordinates_close_missing_and_old_phase_commands(self):
        self.start_maneuver()
        self.current._serial.write.reset_mock()
        generation = self.current.frame_generation()
        for command in (0x14, 0x25, 0x36, 0x24, 0x03):
            self.command(command, 1100)
        self.assertEqual(self.current.frame_generation(), generation)
        for bars, zones in (([self.right], [self.zone]), ([], [])):
            self.assertIsNone(self.submit(bars, zones, 1100))
            self.assertEqual(self.current._last_frame_seen_ms, 1100)
        for packet in (s.build_coordinate_packet(1, 0, 0), s.build_event_packet(0x26),
                       s.build_event_packet(0x16), s.build_event_packet(0xE6), s.build_event_packet(0x36)):
            self.write(packet, 1100)
        self.current._serial.write.assert_not_called()
        self.assertIsNone(self.current._next_outgoing_packet(1100))

    def test_ack_requires_new_frame_then_direct26_even_with_same_wrong_bar(self):
        self.start_maneuver()
        old_generation = self.current.frame_generation()
        retry = self.current._next_outgoing_packet(1550)
        self.command(0x46, 1600)
        self.assertEqual(self.current._zone_realign_phase, "done")
        self.assertIsNone(self.current._latest_packet)
        self.assertIsNone(self.current._next_outgoing_packet(1600))
        self.assertIsNone(self.submit([self.left], now=1610, generation=old_generation))
        self.current._serial.write.reset_mock()
        self.write(retry, 1610, old_generation)
        self.current._serial.write.assert_not_called()
        self.assertIs(self.submit([self.left], now=1620), self.zone)
        self.write(self.current._next_outgoing_packet(1620), 1620)  #fresh16
        self.write(self.current._next_outgoing_packet(1670), 1670)  #26
        packet = self.current._next_outgoing_packet(1670)
        self.assertEqual(len(packet), 11)
        self.assertEqual(packet[2], 1)
        self.assertEqual(self.current._zone_realign_phase, "done")

    def test_duplicate_ack_does_not_clear_fresh_coordinates_or_generation(self):
        self.start_maneuver()
        self.command(0x46, 1600)
        self.submit([self.left], now=1610)
        generation, packet, events = (self.current.frame_generation(), self.current._latest_packet,
                                       list(self.current._pending_events))
        self.command(0x46, 1620)
        self.assertEqual((self.current.frame_generation(), self.current._latest_packet, self.current._pending_events),
                         (generation, packet, events))

    def test_prepared_old_coordinates_and26_blocked_by_new_wrong_side(self):
        self.submit([self.right])
        self.current._next_outgoing_packet(1000)
        old26 = self.current._next_outgoing_packet(1050)
        old_coordinate = self.current._next_outgoing_packet(1050)
        self.submit([self.left], now=1060)
        self.write(old26, 1060)
        self.write(old_coordinate, 1060)
        self.current._serial.write.assert_not_called()
        self.assertEqual(self.current._pending_events, [s.build_event_packet(0x46)])

    def test_actual26_latches_decision_no_late46_from_rod_flicker(self):
        self.submit()
        self.write(self.current._next_outgoing_packet(1000), 1000)
        self.write(self.current._next_outgoing_packet(1050), 1050)
        self.assertEqual(self.current._zone_realign_phase, "done")
        self.submit([self.left], now=1600)
        self.assertEqual(self.current._next_outgoing_packet(1600), s.build_event_packet(0x26))
        self.assertEqual(len(self.current._next_outgoing_packet(1600)), 11)
        self.assertNotIn(s.build_event_packet(0x46), self.current._pending_events)

    def test_latched_pending46_does_not_release_an_even_older_prepared26(self):
        for zones in ([], [box(0, 0, 100, 49)]):
            self.setUp_current(LABELS[5], 1)
            self.submit([self.right])
            self.current._next_outgoing_packet(1000)
            prepared26 = self.current._next_outgoing_packet(1050)
            self.submit([self.left], now=1060)
            self.submit([], zones=zones, now=1070)
            self.assertEqual(self.current._zone_realign_phase, "pending")
            self.write(prepared26, 1070)
            self.current._serial.write.assert_not_called()

    def test_ack_alone_or_small_fresh_zone_never_fabricates26(self):
        self.start_maneuver()
        self.command(0x46, 1600)
        self.submit([self.left], zones=[box(0, 0, 100, 49)], now=1610)
        self.assertEqual(self.current._next_outgoing_packet(1610), s.build_event_packet(0x16))
        self.assertEqual(len(self.current._next_outgoing_packet(1660)), 11)
        self.assertNotIn(s.build_event_packet(0x26), self.current._pending_events)
        self.submit([self.left], now=1700)
        self.assertEqual(self.current._next_outgoing_packet(1700), s.build_event_packet(0x26))

    def test_old_bar_is_not_cached_into_a_later_frame(self):
        self.submit([self.left], zones=[box(0, 0, 100, 49)])
        self.submit(now=1010)
        self.assertIsNone(self.current._zone_realign_phase)
        self.assertIn(s.build_event_packet(0x26), self.current._pending_events)

    def test_missing_repeat_config_defaults_to500ms(self):
        self.start_maneuver()
        fallback = SimpleNamespace(**{key: value for key, value in vars(config).items()
                                      if key != "UART_ZONE_REALIGN_REPEAT_MS"})
        with patch.object(s, "config", fallback):
            self.assertIsNone(self.current._next_outgoing_packet(1549))
            self.assertEqual(self.current._next_outgoing_packet(1550), s.build_event_packet(0x46))

    def test_done_survives_same_delivery05_and_final_inventory(self):
        # Use normal image geometry so actual final05 fixtures remain eligible.
        self.current.frame_width, self.current.frame_height = 576, 448
        self.zone = box(0, 0, 576, 224)
        self.start_maneuver()
        self.command(0x46, 1600)
        self.command(0x05, 1700)
        self.assertEqual(self.current._zone_realign_phase, "done")
        with patch.object(s, "_now_ms", return_value=1700):
            self.current._write_packet(self.current._next_outgoing_packet(1700))
        green = near_view_block(288, 224, 5)
        for now in (2200, 2210, 2220):
            with patch.object(s, "_now_ms", return_value=now):
                self.current.submit([green], LABELS, raw_objects=[green])
        self.assertEqual(self.current._zone_load_recheck, "wait05tx")
        with patch.object(s, "_now_ms", return_value=2300):
            confirm_final_load_check(self.current, [5], 2300)
        self.assertEqual(self.current._zone_realign_phase, "done")

    def test_fe_and_new_task_clear_all46_history_and_old_writes(self):
        for command in (0xFE, 0x11, 0x21, 0x31, 0x51):
            self.setUp_current(LABELS[5], 1)
            self.start_maneuver()
            generation = self.current.frame_generation()
            self.command(command, 1600)
            self.assertIsNone(self.current._zone_realign_phase)
            self.assertIsNone(self.current._last_zone_realign_write_ms)
            self.current._serial.write.reset_mock()
            self.write(s.build_event_packet(0x46), 1610, generation)
            self.current._serial.write.assert_not_called()

    def test_unrelated_modes_ignore_auxiliary_and_unsolicited46(self):
        for mode in (s.CMD_SEARCH, s.CMD_TRACK, s.MODE_FINAL_ROI_TRACK,
                     s.MODE_ARRANGE_RIGHTMOST, s.MODE_NEAR_VIEW_CHECK):
            self.current.mode = mode
            self.command(0x46)
            self.assertEqual(self.current.mode, mode)
            self.assertIsNone(self.current._zone_realign_phase)
            self.write(s.build_event_packet(0x46))
        self.current._serial.write.assert_not_called()


if __name__ == "__main__":
    unittest.main()
