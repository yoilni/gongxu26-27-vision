"""14 directional strips,32/42 fresh-target motion and completion handshakes."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from test_vision_rules import LABELS, block, controller, SingleFrameTrackingScenario
import app_config as config
import serial_controller as s
import vision_overlay as overlay
from target_selector import directional_target_neighbors, expanded_target_neighbors, casualty_vertical_grab_allowed, selected_target_rectangles


class MicroAdjustTests(SingleFrameTrackingScenario):
    def setUp(self):
        for helper in (patch("builtins.print"), patch.object(config, "VISION_DIAGNOSTICS_ENABLED", False)):
            helper.start()
            self.addCleanup(helper.stop)
        self.current = controller(s.MODE_WAIT_14_HANDSHAKE)
        self.current.target_label, self.current._post07_started = LABELS[4], True
        self.current._serial = SimpleNamespace(write=Mock(side_effect=lambda packet: len(packet)))
        self.target = block(288, 224, 4)
        self.horizontal = block(350, 224, 3)
        self.vertical = block(288, 280, 3)
        self.corner = block(350, 280, 3)
        self.current._track_target_lock = self.current._final_target_identity(self.target)
        self.command(0x14)

    def command(self, cmd, now=1000):
        with patch.object(s, "_now_ms", return_value=now):
            self.current.process_received_bytes(s.build_command_packet(cmd))

    def submit(self, raw, now=1050, generation=None):
        with patch.object(s, "_now_ms", return_value=now):
            return self.current.submit([], LABELS, raw_objects=raw,
                                       frame_generation=self.current.frame_generation() if generation is None else generation)

    def write(self, packet, now=1050, generation=None):
        with patch.object(s, "_now_ms", return_value=now):
            self.current._write_packet(packet, self.current.frame_generation() if generation is None else generation)

    def decide(self, event, objects=None, now=1050):
        self.submit([self.target] + (objects or []), now)
        packet = self.current._next_outgoing_packet(now)
        self.assertEqual(packet, s.build_event_packet(event))
        return packet

    def start(self, cmd, now=1050):
        packet = self.decide(cmd, [self.horizontal if cmd == 0x32 else self.vertical], now)
        self.write(packet, now)
        self.assertTrue(self.current._micro_request_sent)
        return packet

    def test_directional_table_and_casualty_upper_vs_lower_corner_rules(self):
        cases = (([], 0x24), ([self.horizontal], 0x32), ([self.vertical], 0x42),
                 ([self.horizontal, self.vertical], 0x02), ([block(290, 225, 3)], 0x02))
        for neighbors, expected in cases:
            self.assertEqual(self.current._directional_arrangement_event(self.target, [self.target] + neighbors), expected)
        for x, y in ((226, 168), (350, 168), (226, 280), (350, 280)):
            other = block(x, y, 3)
            self.assertTrue(expanded_target_neighbors(self.target, [self.target, other], 576, 448))
            self.assertEqual(directional_target_neighbors(self.target, [self.target, other], 576, 448),
                             ([], [other]) if y < 224 else ([], []))
            self.assertEqual(self.current._directional_arrangement_event(self.target, [self.target, other]), 0x24)

    def test_directional_bands_are_edge_exclusive_and_neighbor_is_not_expanded(self):
        for x, y, expected in ((362, 224, 0x24), (361, 224, 0x32),
                                (288, 298, 0x24), (288, 297, 0x42), (350, 280, 0x24)):
            other = block(x, y, 3)
            self.assertEqual(self.current._directional_arrangement_event(self.target, [self.target, other]), expected)

    def test_red_vertical_only_grabs_when_target_original_width_over_height_gt1p5(self):
        for width, height, expected in ((29.99, 20, 0x42), (30, 20, 0x42), (30.01, 20, 0x24),
                                        (39.99, 20, 0x24), (40, 20, 0x24), (40.01, 20, 0x24),
                                        (60, 20, 0x24), (20, 60, 0x42)):
            target = SimpleNamespace(x=288-width/2, y=224-height/2,
                                     w=width, h=height, class_id=4, score=.9)
            vertical = block(288, 280, 3)
            self.assertEqual(self.current._directional_arrangement_event(target, [target, vertical]), expected)

    def test_wide_red_vertical_exception_keeps_horizontal_and_two_axis_rules(self):
        target = SimpleNamespace(x=272, y=214, w=32, h=20, class_id=4, score=.9)
        horizontal, vertical = block(350, 224, 3), block(288, 280, 6)
        for objects, expected in (([horizontal], 0x32), ([vertical], 0x24),
                                   ([horizontal, vertical], 2), ([], 0x24)):
            self.assertEqual(self.current._directional_arrangement_event(target, [target] + objects), expected)

    def test_red_front_of_all_vertical_neighbors_grabs_but_one_in_front_still42(self):
        for class_id in (3, 4, 5, 6):
            above = block(288, 190, class_id)
            farther_above = block(288, 170, class_id)
            below = block(288, 270, class_id)
            for neighbors, event in (([above], 0x24), ([above, farther_above], 0x24),
                                      ([above, below], 0x42), ([below], 0x42)):
                with self.subTest(class_id=class_id, event=event):
                    self.assertEqual(self.current._directional_arrangement_event(
                        self.target, [self.target] + neighbors), event)

    def test_red_front_exception_never_bypasses_horizontal_or_two_axis_congestion(self):
        behind = block(288, 190, 5)
        horizontal = block(350, 224, 3)
        for width, height in ((20, 20), (40, 20)):
            red = SimpleNamespace(x=288-width/2, y=224-height/2, w=width, h=height, class_id=4, score=.9)
            self.assertEqual(self.current._directional_arrangement_event(red, [red, behind]), 0x24)
            self.assertEqual(self.current._directional_arrangement_event(red, [red, horizontal]), 0x32)
            self.assertEqual(self.current._directional_arrangement_event(red, [red, behind, horizontal]), 0x02)

    def test_casualty_upper_diagonals_are_vertical_only_inside_original_expansion(self):
        for x, y in ((280, 220), (296, 220), (230, 220), (350, 220),
                      (240, 190), (336, 190), (288, 223)):
            for neighbor_id in (3, 4, 5, 6):
                other = block(x, y, neighbor_id)
                raw = [self.target, other]
                original = vars(other).copy()
                self.assertEqual(directional_target_neighbors(self.target, raw, 576, 448), ([], [other]))
                self.assertEqual(self.current._directional_arrangement_result(self.target, raw), (0x24, []))
                self.assertEqual(vars(other), original)
        for x, y in ((214, 190), (362, 190), (288, 150), (100, 100)):
            other = block(x, y, 5)
            self.assertEqual(directional_target_neighbors(self.target, [self.target, other], 576, 448), ([], []))
        # Same-height sides and lower-half/cargo-class rules are unchanged.
        for class_id in (3, 5, 6):
            target = block(288, 224, class_id)
            near_upper = block(280, 220, 5)
            corner = block(240, 190, 5)
            self.assertEqual(directional_target_neighbors(target, [target, near_upper], 576, 448),
                             ([near_upper], [near_upper]))
            self.assertEqual(directional_target_neighbors(target, [target, corner], 576, 448), ([], []))

    def test_red14_and44_upper_diagonals_grab24_but_real_side_or_front_neighbor_still_blocks(self):
        uppers = [block(280, 220, 5), block(296, 220, 6)]
        for others, event in ((uppers, 0x24), (uppers + [self.horizontal], 0x02),
                               (uppers + [self.vertical], 0x42)):
            for command in (0x14, 0x44):
                self.current = controller(s.MODE_WAIT_14_HANDSHAKE if command == 0x14 else s.CMD_TRACK)
                self.current.target_label, self.current._post07_started = LABELS[4], True
                self.current._track_target_lock = self.current._final_target_identity(self.target)
                self.command(command)
                self.decide(event, others)
                if event == 0x24:
                    self.assertEqual(self.current._final_target_lock,
                                     self.current._final_target_identity(self.target))

    def test_front_condition_is_strict_all_neighbors_valid_and_only_red(self):
        for center_y, allowed in ((223.9, True), (224, False), (224.1, False)):
            self.assertEqual(casualty_vertical_grab_allowed(self.target, [], [block(288, center_y, 3)]), allowed)
        for invalid in (SimpleNamespace(y=float("nan"), h=20), SimpleNamespace(h=20),
                        SimpleNamespace(y=170, h=0), SimpleNamespace(y=170, h=float("inf"))):
            self.assertFalse(casualty_vertical_grab_allowed(self.target, [], [invalid]))
        for class_id in (3, 5, 6):
            target = block(288, 224, class_id)
            self.assertEqual(self.current._directional_arrangement_event(target, [target, block(288, 190, 3)]), 0x42)

    def test_red_front14_and44_seed_same_target_even_after_three_micro_rounds(self):
        for command in (0x14, 0x44):
            for rounds in (0, 3):
                self.current = controller(s.MODE_WAIT_14_HANDSHAKE if command == 0x14 else s.CMD_TRACK)
                self.current.target_label, self.current._post07_started = LABELS[4], True
                self.current._track_target_lock = self.current._final_target_identity(self.target)
                self.current._micro_rounds = rounds
                self.command(command)
                self.submit([self.target, block(288, 190, 5)])
                self.assertEqual(self.current._next_outgoing_packet(1050), s.build_event_packet(0x24))
                self.assertEqual(self.current._final_target_lock, self.current._final_target_identity(self.target))

    def test_non_red_target_and_wide_neighbor_do_not_enable_red_exception(self):
        wide_neighbor = SimpleNamespace(x=258, y=270, w=60, h=20, class_id=4, score=.9)
        for class_id in (3, 4, 5, 6):
            target = block(288, 224, class_id)
            self.assertEqual(self.current._directional_arrangement_event(target, [target, wide_neighbor]), 0x42)
        for class_id in (3, 5, 6):
            target = SimpleNamespace(x=258, y=214, w=60, h=20, class_id=class_id, score=.9)
            self.assertEqual(self.current._directional_arrangement_event(target, [target, self.vertical]), 0x42)

    def test_red_exception_checks_native_ratio_and_rejects_invalid_dimensions(self):
        for width, height in ((float('nan'), 20), (float('inf'), 20), (60, 0),
                              (60, -1), (-60, 20), (60, float('nan'))):
            target = SimpleNamespace(class_id=4, w=width, h=height)
            self.assertFalse(casualty_vertical_grab_allowed(target, [], [self.vertical]))
        target = SimpleNamespace(x=-40, y=214, w=60, h=20, class_id=4, score=.9)
        vertical = block(10, 280, 5)
        self.assertEqual(self.current._directional_arrangement_event(target, [target, vertical]), 0x24)
        self.assertFalse(casualty_vertical_grab_allowed(target, [], []))
        self.assertFalse(casualty_vertical_grab_allowed(None, [], [vertical]))

    def test_red14_and_recovery44_wide_vertical_seed24_with_approached_identity(self):
        for command in (0x14, 0x44):
            for rounds in (0, 3):
                self.current = controller(s.MODE_WAIT_14_HANDSHAKE if command == 0x14 else s.CMD_TRACK)
                self.current.target_label, self.current._post07_started = LABELS[4], True
                target = SimpleNamespace(x=272, y=214, w=32, h=20, class_id=4, score=.9)
                self.current._track_target_lock = self.current._final_target_identity(target)
                self.current._micro_rounds = rounds
                self.command(command)
                self.submit([target, block(288, 280, 5)])
                self.assertEqual(self.current._next_outgoing_packet(1050), s.build_event_packet(0x24))
                self.assertEqual(self.current.mode, s.MODE_FINAL_ROI_TRACK)
                self.assertEqual(self.current._final_target_lock, self.current._final_target_identity(target))
                self.assertEqual(self.current._micro_rounds, 0)

    def test_all_cargo_ids_are_neighbors_but_bar_zone_and_self_are_not(self):
        for neighbor_id in (3, 4, 5, 6):
            self.horizontal.class_id = self.vertical.class_id = neighbor_id
            self.assertEqual(self.current._directional_arrangement_event(
                self.target, [self.target, self.horizontal, self.vertical]), 2)
        for class_id in (0, 1, 2, 7, 99):
            self.assertEqual(directional_target_neighbors(self.target,
                [self.target, block(290, 224, class_id)], 576, 448), ([], []))

    def test_opponent_zone_excludes_locked_micro_target_and_prepared_coordinates(self):
        for command in (0x32, 0x42):
            for destination_id in (1, 2):
                for target_id in (4, 5, 6):
                    with self.subTest(command=command, destination=destination_id, target=target_id):
                        self.current = controller(s.MODE_WAIT_14_HANDSHAKE)
                        self.current._serial = SimpleNamespace(write=Mock(side_effect=lambda packet: len(packet)))
                        self.current.target_label = LABELS[target_id]
                        self.current._post07_started = True
                        self.current.zone_label = LABELS[destination_id]
                        self.target.class_id = target_id
                        self.current._track_target_lock = self.current._final_target_identity(self.target)
                        self.command(0x14)
                        self.start(command)
                        self.assertIs(self.submit([self.target], 1100), self.target)
                        prepared = self.current._latest_packet
                        self.assertIsNotNone(prepared)
                        opponent_zone = SimpleNamespace(x=278, y=214, w=20, h=20,
                                                        class_id=3 - destination_id, score=.9)
                        self.assertIsNone(self.submit([opponent_zone, self.target], 1150))
                        self.assertIsNone(self.current._latest_packet)
                        self.assertIsNone(self.current._next_outgoing_packet(1150))
                        self.current._serial.write.reset_mock()
                        self.write(prepared, 1151)
                        self.current._serial.write.assert_not_called()
                        self.assertEqual(self.current.mode, command)
                        # Exclusion uses current detections, never cached zones.
                        self.assertIs(self.submit([self.target], 1200), self.target)

    def test_opponent_zone_cargo_is_not_a_horizontal_or_vertical_adjustment_neighbor(self):
        for destination_id in (1, 2):
            for neighbor, zone_rect in ((self.horizontal, (440, 214, 20, 20)),
                                        (self.vertical, (278, 274, 20, 20))):
                with self.subTest(destination=destination_id, zone=zone_rect):
                    self.current = controller(s.MODE_WAIT_14_HANDSHAKE)
                    self.current._serial = SimpleNamespace(write=Mock(side_effect=lambda packet: len(packet)))
                    self.current.target_label, self.current._post07_started = LABELS[4], True
                    self.current.zone_label = LABELS[destination_id]
                    self.current._track_target_lock = self.current._final_target_identity(self.target)
                    self.command(0x14)
                    opponent_zone = SimpleNamespace(**dict(zip(("x", "y", "w", "h"), zone_rect)),
                                                    class_id=3 - destination_id, score=.9)
                    self.decide(0x24, [neighbor, opponent_zone])
                    self.assertEqual(self.current._final_target_lock,
                                     self.current._final_target_identity(self.target))

    def test32_request_before_coordinates_then_new_frame_roi_center(self):
        old_generation = self.current.frame_generation()
        packet = self.decide(0x32, [self.horizontal])
        self.assertIsNone(self.current._latest_packet)
        self.assertIsNone(self.submit([self.target], 1051))
        self.command(0x32, 1052)  # Preparing is not successful transmission.
        self.assertEqual(self.current.mode, 0x32)
        generation = self.current.frame_generation()
        self.write(packet, 1060)
        self.assertEqual(self.current._micro_rounds, 1)
        self.assertIsNone(self.submit([self.target], 1070, generation))
        self.assertIsNone(self.submit([self.target], 1070, old_generation))
        self.assertIs(self.submit([self.target], 1080), self.target)
        self.assertEqual(self.current._latest_packet, s.build_coordinate_packet(4, 0, -180))
        self.assertEqual(self.current._next_outgoing_packet(1080), self.current._latest_packet)

    def test32_tracks_in_roi_without34_14_or05_and_preserves_original_target(self):
        self.start(0x32)
        moved, distractor = block(300, 420, 4), block(400, 400, 4)
        self.assertIs(self.submit([distractor, moved, self.horizontal], 1100), moved)
        self.assertEqual(self.current.mode, 0x32)
        self.assertEqual(self.current._latest_packet, s.build_coordinate_packet(4, 12, 16))
        self.assertEqual(self.current._track_target_lock, self.current._final_target_identity(moved))
        self.assertFalse(self.current._final_roi_sent)
        self.assertEqual(self.current._pending_events, [])
        for event in (0x34, 0x14, 0x05, 0x24):
            self.current._serial.write.reset_mock()
            self.write(s.build_event_packet(event), 1100)
            self.current._serial.write.assert_not_called()

    def test42_roi_center_reference_stays_same_when_first_turn_alternates(self):
        self.start(0x42)
        self.assertFalse(self.current._micro42_right)
        self.assertEqual(self.current._reference_point(0x42), (288, 403.5))
        self.assertIs(self.submit([self.target], 1800), self.target)
        self.assertEqual(self.current._latest_packet, s.build_coordinate_packet(4, 0, -180))
        self.assertEqual(self.current.mode, 0x42)  # Vision does not finish at800ms.
        self.command(0x42, 2900)  # Simulate completion of ALL MCU motion.
        self.assertIsNone(self.current._latest_packet)
        self.assertTrue(self.current._initial_arrangement_decision_pending)
        self.start(0x42, 2950)
        self.assertTrue(self.current._micro42_right)
        self.assertEqual(self.current._reference_point(0x42), (288, 403.5))
        self.submit([self.target], 3000)
        self.assertEqual(self.current._latest_packet, s.build_coordinate_packet(4, 0, -180))
        self.assertEqual(self.current._micro_rounds, 2)

    def test42_custom_roi_center_ignores_retired_side_reference_settings(self):
        self.current.tracking_roi = [20, 100, 400, 200]
        # Keep this test's intended vertical neighbor outside its custom ROI;
        # ROI cargo is no longer an arrangement neighbor.
        self.target = block(288, 330, 4)
        self.vertical = block(288, 350, 3)
        with patch.object(config, "UART_MICRO42_SIDE_INSET_PX", 50, create=True), \
                patch.object(config, "UART_MICRO42_Y_RATIO", .75, create=True):
            self.start(0x42)
            self.assertEqual(self.current._reference_point(0x42), (220, 200))
            self.submit([self.target], 1100)
            self.assertEqual(self.current._latest_packet, s.build_coordinate_packet(4, 68, 130))
            self.command(0x42, 2900)
            self.start(0x42, 2950)
            self.assertTrue(self.current._micro42_right)
            self.assertEqual(self.current._reference_point(0x42), (220, 200))

    def test42_uses_production_roi_center_and_image_center_only_without_roi(self):
        self.current.tracking_roi = [0, 319, 576, 129]
        self.start(0x42)
        self.assertEqual(self.current._reference_point(0x42), (288, 383.5))
        self.submit([block(300, 390, 4)], 1100)
        self.assertEqual(self.current._latest_packet, s.build_coordinate_packet(4, 12, 6))
        self.current.tracking_roi = None
        self.assertEqual(self.current._reference_point(0x42), (288, 224))

    def test42_center_tracking_keeps_raw_identity_inside_roi_and_never_emits34(self):
        self.start(0x42)
        moved = block(300, 420, 4)
        with patch.object(self.current, "_directional_arrangement_event",
                          side_effect=AssertionError("do not re-screen an active42 lock")):
            self.assertIs(self.submit([block(400, 400, 4), moved], 1100), moved)
        self.assertEqual(self.current._latest_packet, s.build_coordinate_packet(4, 12, 16))
        self.assertEqual(self.current._track_target_lock, self.current._final_target_identity(moved))
        self.assertFalse(self.current._final_roi_sent)
        self.assertEqual(self.current._pending_events, [])
        for event in (0x34, 0x14, 0x05, 0x24):
            self.current._serial.write.reset_mock()
            self.write(s.build_event_packet(event), 1100)
            self.current._serial.write.assert_not_called()

    def test_three_rounds_total_then_fallback02_without_fourth_request(self):
        for index, cmd in enumerate((0x32, 0x42, 0x32)):
            now = 1050 + index * 2200
            self.start(cmd, now)
            self.command(cmd, now + (1800 if cmd == 0x42 else 900))
        packet = self.decide(0x02, [self.vertical], 7650)
        self.assertEqual(self.current.mode, s.MODE_ARRANGE_RIGHTMOST)
        self.assertEqual(self.current._micro_rounds, 0)
        self.assertEqual(packet, s.build_event_packet(2))
        self.command(0x12, 7700)
        self.assertEqual(self.current._next_outgoing_packet(7700), s.build_event_packet(0x12))
        self.command(0x22, 7750)
        self.assertEqual(self.current.mode, s.MODE_POST_ARRANGE_CHECK)

    def test_clear_after_third_round_still24_not_forced02(self):
        for index in range(3):
            self.start(0x32, 1050 + index * 1000)
            self.command(0x32, 1950 + index * 1000)
        self.decide(0x24, [self.corner], 4050)
        self.assertEqual(self.current._micro_rounds, 0)

    def test_wrong_duplicate_and_late_ack_do_not_reset_fresh_output(self):
        self.start(0x32)
        generation = self.current.frame_generation()
        for code in (0x42, 0x14, 0x24, 0x12, 0x04):
            self.command(code, 1100)
            self.assertEqual(self.current.mode, 0x32)
            self.assertEqual(self.current.frame_generation(), generation)
        self.command(0x32, 1700)
        generation = self.current.frame_generation()
        self.command(0x32, 1701)
        self.assertEqual(self.current.frame_generation(), generation)
        self.decide(0x24, now=1750)
        events = list(self.current._pending_events)
        self.command(0x32, 1800)
        self.assertEqual(self.current._pending_events, events)

    def test_no_successful_request_retries_after_sent_and_short_write_not_counted(self):
        packet = self.decide(0x42, [self.vertical])
        self.current._serial.write = Mock(return_value=3)
        self.write(packet)
        self.assertFalse(self.current._micro_request_sent)
        self.assertEqual(self.current._micro_rounds, 0)
        self.assertEqual(self.current._next_outgoing_packet(1060), packet)
        self.current._serial.write = Mock(return_value=5)
        self.write(packet, 1060)
        self.write(packet, 1070)
        self.current._serial.write.assert_called_once_with(packet)
        self.assertEqual(self.current._micro_rounds, 1)
        self.assertIsNone(self.current._next_outgoing_packet(1600))

    def test_loss5_frames_sends_e4_then_fresh_coordinates_switch_to_image_center(self):
        self.start(0x32)
        for index in range(5):
            self.submit([], 1100 + index * 20)
            if index < 4:
                self.assertIsNone(self.current._next_outgoing_packet(1100 + index * 20))
        event = self.current._next_outgoing_packet(1180)
        self.assertEqual(event, s.build_event_packet(0xE4))
        old_generation = self.current.frame_generation()
        self.write(event, 1180)
        self.assertEqual(self.current.mode, s.CMD_TRACK)
        self.assertEqual(self.current._micro_rounds, 1)
        self.assertIsNone(self.submit([self.target], 1200, old_generation))
        self.submit([block(350, 250, 4)], 1210)
        self.assertEqual(self.current._latest_packet, s.build_coordinate_packet(4, 62, 26))
        self.assertNotIn(s.build_event_packet(0xE4), self.current._pending_events)
        self.command(0x32, 1220)  # E4 aborted that old action.
        self.assertEqual(self.current.mode, s.CMD_TRACK)

    def test_request_ack_timeout_e4_does_not_depend_on_camera_frames(self):
        self.start(0x42)
        for now in (6049, 6050, 8850, 11850, 12049):
            self.assertIsNone(self.current._next_outgoing_packet(now))
        event = self.current._next_outgoing_packet(12050)
        self.assertEqual(event, s.build_event_packet(0xE4))
        self.write(event, 12050)
        self.assertEqual(self.current.mode, s.CMD_TRACK)
        self.assertIsNone(self.current._next_outgoing_packet(12051))

    def test32_still_times_out_at_five_seconds_without_completion(self):
        self.start(0x32)
        self.assertIsNone(self.current._next_outgoing_packet(6049))
        self.assertEqual(self.current._next_outgoing_packet(6050), s.build_event_packet(0xE4))

    def test42_missing_timeout_config_defaults_to11_seconds_and_custom_is_respected(self):
        self.start(0x42)
        settings = {key: value for key, value in vars(config).items()
                    if key != "UART_MICRO42_ACK_TIMEOUT_MS"}
        self.assertEqual(config.UART_MICRO42_ACK_TIMEOUT_MS, 11000)
        with patch.object(s, "config", SimpleNamespace(**settings)):
            self.assertIsNone(self.current._next_outgoing_packet(12049))
            self.assertEqual(self.current._next_outgoing_packet(12050), s.build_event_packet(0xE4))
        # A new request, not the already-timed-out one, tests custom tuning.
        self.command(0xFE, 13000)
        self.current.mode = s.MODE_WAIT_14_HANDSHAKE
        self.current.target_label, self.current._post07_started = LABELS[4], True
        self.current._track_target_lock = self.current._final_target_identity(self.target)
        self.command(0x14, 13000)
        self.start(0x42, 13050)
        with patch.object(config, "UART_MICRO42_ACK_TIMEOUT_MS", 12000):
            self.assertIsNone(self.current._next_outgoing_packet(25049))
            self.assertEqual(self.current._next_outgoing_packet(25050), s.build_event_packet(0xE4))

    def test42_late_valid_completion_is_accepted_without_auto_completing_locally(self):
        self.start(0x42)
        self.assertIsNone(self.current._next_outgoing_packet(10000))
        self.assertEqual(self.current.mode, 0x42)
        self.command(0x42, 12049)
        self.assertTrue(self.current._initial_arrangement_decision_pending)
        self.assertFalse(self.current._micro_request_sent)
        self.decide(0x24, now=12100)

    def test42_completion_timeout_starts_at_successful_uart_write_not_decision(self):
        packet = self.decide(0x42, [self.vertical], 1050)
        self.current._serial.write = Mock(return_value=3)
        self.write(packet, 1050)
        self.current._serial.write = Mock(return_value=5)
        self.write(packet, 1080)
        self.assertIsNone(self.current._next_outgoing_packet(12079))
        self.assertEqual(self.current._next_outgoing_packet(12080), s.build_event_packet(0xE4))

    def test_stale_unsent_request_becomes_e4_without_motion_or_round_charge(self):
        packet = self.decide(0x32, [self.horizontal])
        self.write(packet, 1300)
        self.current._serial.write.assert_not_called()
        event = self.current._next_outgoing_packet(1300)
        self.assertEqual(event, s.build_event_packet(0xE4))
        self.write(event, 1300)
        self.assertEqual(self.current._micro_rounds, 0)

    def test_fe_and_new_task_cancel_micro_events_prepared_coordinates_and_counters(self):
        self.start(0x42)
        self.submit([self.target], 1100)
        packet = self.current._latest_packet
        generation = self.current.frame_generation()
        self.command(0xFE, 1200)
        self.current._serial.write.reset_mock()
        self.write(packet, 1300, generation)
        self.current._serial.write.assert_not_called()
        self.assertEqual(self.current._micro_rounds, 0)
        self.assertFalse(self.current._micro_request_sent)

    def test32_and42_completion_at400ms_rechecks_on_new_frame_without_minimum_gate(self):
        self.start(0x32)
        generation = self.current.frame_generation()
        self.command(0x32, 1450)
        self.assertTrue(self.current._initial_arrangement_decision_pending)
        self.assertGreater(self.current.frame_generation(), generation)
        self.assertIsNone(self.submit([self.target], 1500, generation))
        self.start(0x42, 1700)
        generation = self.current.frame_generation()
        self.command(0x42, 2100)
        self.assertTrue(self.current._initial_arrangement_decision_pending)
        self.assertGreater(self.current.frame_generation(), generation)
        self.assertIsNone(self.submit([self.target], 2110, generation))

    def test32_no_minimum_elapsed_gate_even_with_stale_600ms_config(self):
        with patch.object(config, "UART_MICRO32_MIN_ACK_MS", 600, create=True):
            for index, delay in enumerate((0, 1, 399)):
                now = 1050 + index * 1000
                self.start(0x32, now)
                self.command(0x32, now + delay)
                self.assertTrue(self.current._initial_arrangement_decision_pending)
                self.assertFalse(self.current._micro_request_sent)

    def test32_unsent_request_cannot_accept_completion(self):
        request = self.decide(0x32, [self.horizontal])
        generation = self.current.frame_generation()
        self.command(0x32, 1100)
        self.assertEqual(self.current.mode, 0x32)
        self.assertFalse(self.current._micro_request_sent)
        self.assertEqual(self.current.frame_generation(), generation)
        self.write(request, 1100)
        self.command(0x32, 1500)
        self.assertTrue(self.current._initial_arrangement_decision_pending)

    def test42_unsent_or_short_request_cannot_accept_completion(self):
        packet = self.decide(0x42, [self.vertical])
        generation = self.current.frame_generation()
        self.command(0x42, 1100)
        self.assertEqual(self.current.mode, 0x42)
        self.assertEqual(self.current.frame_generation(), generation)
        self.current._serial.write.side_effect = None
        self.current._serial.write.return_value = 2
        self.write(packet, 1110)
        self.command(0x42, 1120)
        self.assertEqual(self.current.mode, 0x42)
        self.assertFalse(self.current._micro_request_sent)
        self.current._serial.write.return_value = 5
        self.write(self.current._next_outgoing_packet(1130), 1130)
        sent_generation = self.current.frame_generation()
        self.command(0x42, 1131)
        self.assertTrue(self.current._initial_arrangement_decision_pending)
        self.assertGreater(self.current.frame_generation(), sent_generation)

    def test_fast42_completion_resumes_fresh_decision_instead_of_waiting11s_then_e4(self):
        self.start(0x42, 1050)
        generation = self.current.frame_generation()
        self.command(0x42, 1350)
        self.assertIsNone(self.submit([self.target], 1351, generation))
        self.submit([self.target], 1400)
        self.assertEqual(self.current._next_outgoing_packet(1400), s.build_event_packet(0x24))
        self.assertEqual(self.current.mode, s.MODE_FINAL_ROI_TRACK)
        self.assertIsNone(self.current._next_outgoing_packet(12100))

    def test42_has_no_minimum_gate_with_missing_or_stale1600ms_setting(self):
        self.assertFalse(hasattr(config, "UART_MICRO42_MIN_ACK_MS"))
        for setting in (None, 1200, 1600):
            for delay in (0, 1, 300, 1000, 1599):
                with self.subTest(setting=setting, delay=delay):
                    self.current = controller(s.MODE_WAIT_14_HANDSHAKE)
                    self.current.target_label, self.current._post07_started = LABELS[4], True
                    self.current._serial = SimpleNamespace(write=Mock(side_effect=lambda packet: len(packet)))
                    self.current._track_target_lock = self.current._final_target_identity(self.target)
                    self.command(0x14)
                    self.start(0x42, 1050)
                    settings = {key: value for key, value in vars(config).items()
                                if key != "UART_MICRO42_MIN_ACK_MS"}
                    if setting is not None:
                        settings["UART_MICRO42_MIN_ACK_MS"] = setting
                    with patch.object(s, "config", SimpleNamespace(**settings)):
                        generation = self.current.frame_generation()
                        self.command(0x42, 1050 + delay)
                        self.assertTrue(self.current._initial_arrangement_decision_pending)
                        self.assertGreater(self.current.frame_generation(), generation)
                        self.assertFalse(self.current._micro_request_sent)

    def test_actual_coordinates_keep_same_class_despite_large_jump_area_and_new_neighbor(self):
        self.start(0x32)
        moved = block(500, 350, 4)
        moved.w, moved.h = 60, 100
        other_color = block(289, 224, 5)
        with patch.object(self.current, "_directional_arrangement_event", side_effect=AssertionError("no repeat screening")):
            self.assertIs(self.submit([other_color, moved], 1100), moved)
        self.assertEqual(self.current._latest_packet[2], 4)
        self.assertEqual(self.current._track_target_lock, self.current._final_target_identity(moved))

    def test_micro32_and42_all_tasks_use_raw_yolo_and_ignore_corner_only(self):
        for label, class_id in ((LABELS[4], 4), (LABELS[5], 5), (LABELS[6], 6), (s.TARGET_BLACK_OR_GREEN, 5)):
            for neighbors, event in (([self.horizontal], 0x32), ([self.vertical], 0x42), ([self.corner], 0x24)):
                self.current = controller(s.MODE_WAIT_14_HANDSHAKE)
                self.current._serial = SimpleNamespace(write=Mock(side_effect=lambda packet: len(packet)))
                self.current.target_label = label
                self.current._post07_started = True
                self.current.mode = s.MODE_WAIT_14_HANDSHAKE
                self.current._reset_micro_adjust()
                self.target.class_id = class_id
                self.current._track_target_lock = self.current._final_target_identity(self.target)
                self.command(0x14)
                self.decide(event, neighbors)
                if event in (0x32, 0x42):
                    self.write(s.build_event_packet(event))
                    self.assertIs(self.submit([self.target], 1100), self.target)
                    self.assertEqual(self.current._latest_packet[2], class_id)

    def test_black_missing_obeys_existing800ms_confirmation(self):
        self.current.target_label, self.target.class_id = LABELS[6], 6
        self.current.mode = s.MODE_WAIT_14_HANDSHAKE
        self.current._track_target_lock = self.current._final_target_identity(self.target)
        self.command(0x14)
        self.start(0x32)
        for index in range(5):
            self.submit([], 1100 + index * 20)
        self.assertIsNone(self.current._next_outgoing_packet(1899))
        self.assertEqual(self.current._next_outgoing_packet(1900), s.build_event_packet(0xE4))

    def test_post_ack_target_missing_is_e4_not_02_or24(self):
        self.start(0x42)
        self.command(0x42, 2900)
        for index in range(5):
            self.submit([], 2950 + index * 20)
            if index < 4:
                self.assertEqual(self.current._pending_events, [])
        self.assertEqual(self.current._next_outgoing_packet(3030), s.build_event_packet(0xE4))
        self.assertEqual(self.current.mode, s.CMD_TRACK)
        self.assertIsNotNone(self.current._track_target_lock)

    def test_post_ack_camera_stall_has_watchdog_without_forcing24(self):
        self.start(0x32)
        self.command(0x32, 1700)
        self.assertIsNone(self.current._next_outgoing_packet(6699))
        self.assertEqual(self.current._next_outgoing_packet(6700), s.build_event_packet(0xE4))

    def test_e4_interrupted42_keeps_alternation_and_round_budget(self):
        self.start(0x42)
        for index in range(5):
            self.submit([], 1100 + index * 20)
        self.write(self.current._next_outgoing_packet(1180), 1180)
        self.assertEqual(self.current._micro_rounds, 1)
        self.assertEqual(self.current._micro42_rounds, 1)
        self.command(0x44, 2500)
        self.start(0x42, 2550)
        self.assertTrue(self.current._micro42_right)
        self.assertEqual(self.current._micro_rounds, 2)

    def test42_alternates_left_right_left_without_toggling_on_duplicate_command(self):
        for index, right in enumerate((False, True, False)):
            now = 1050 + index * 2200
            packet = self.start(0x42, now)
            self.assertEqual(self.current._micro42_right, right)
            self.write(packet, now + 10)
            self.assertEqual(self.current._micro42_right, right)
            self.assertEqual(self.current._micro42_rounds, index + 1)
            self.assertEqual(self.current._reference_point(0x42), (288, 403.5))
            self.command(0x42, now + 1800)

    def test32_between42_rounds_does_not_change42_turn_alternation(self):
        for index, command in enumerate((0x42, 0x32, 0x42)):
            now = 1050 + index * 2200
            self.start(command, now)
            if command == 0x42:
                self.assertEqual(self.current._micro42_right, index == 2)
                self.assertEqual(self.current._reference_point(command), (288, 403.5))
            self.command(command, now + (1800 if command == 0x42 else 900))
        self.assertEqual(self.current._micro42_rounds, 2)
        self.assertEqual(self.current._micro_rounds, 3)

    def test_mcu05_and_supply_timeout_can_interrupt_adjustment_for_final_inventory(self):
        for source in ("mcu", "timer"):
            self.current = controller(s.MODE_WAIT_14_HANDSHAKE)
            self.current._serial = SimpleNamespace(write=Mock(side_effect=lambda packet: len(packet)))
            self.current._post07_started = True
            self.current.target_label, self.target.class_id = s.TARGET_BLACK_OR_GREEN, 5
            self.current.mode = s.MODE_WAIT_14_HANDSHAKE
            self.current._reset_micro_adjust()
            self.current._track_target_lock = self.current._final_target_identity(self.target)
            self.command(0x14)
            self.start(0x32)
            self.submit([self.target], 1100)
            old_packet, generation = self.current._latest_packet, self.current.frame_generation()
            if source == "mcu":
                self.command(5, 1200)
            else:
                self.current._supply_collect_started_ms = 1000
                self.current._supply_collect_finished = False
                self.current._next_outgoing_packet(16000)
            self.assertEqual(self.current.mode, s.MODE_NEAR_VIEW_CHECK)
            self.assertEqual(self.current._zone_load_recheck_source, "before06")
            self.current._serial.write.reset_mock()
            self.write(old_packet, 16100, generation)
            self.current._serial.write.assert_not_called()
            self.assertEqual(self.current._micro_rounds, 0)

    def test51_original_micro_target_can_enter_inner_roi_without34_or_old_cargo_swap(self):
        self.current = controller(s.MODE_WAIT_14_HANDSHAKE)
        self.current._serial = SimpleNamespace(write=Mock(side_effect=lambda packet: len(packet)))
        self.current.target_label, self.current._post07_started = s.TARGET_BLACK_OR_GREEN, True
        self.target = block(288, 350, 5)
        self.horizontal = block(350, 350, 3)
        self.current._track_target_lock = self.current._final_target_identity(self.target)
        self.command(0x14)
        cargo = block(450, 430, 5)
        packet = self.decide(0x32, [self.horizontal, cargo])
        self.write(packet)
        for now, y in ((1100, 390), (1150, 410), (1200, 425)):
            moving = block(288, y, 5)
            self.assertIs(self.submit([cargo, moving], now), moving)
            self.assertEqual(self.current._track_target_lock[1:3], (288, y))
            self.assertIsNotNone(self.current._latest_packet)
            self.assertFalse(self.current._final_roi_sent)
            self.assertEqual(self.current._pending_events, [])
        self.command(0x32, 1700)
        self.submit([cargo, block(288, 425, 5)], 1750)
        self.assertEqual(self.current._next_outgoing_packet(1750), s.build_event_packet(0x24))
        self.assertEqual(self.current._final_target_lock[1:3], (288, 425))
        self.submit([cargo, block(288, 425, 5)], 1800)
        self.assertEqual(self.current._next_outgoing_packet(1800), s.build_event_packet(0x34))

    def test_micro_e4_center_reference_waits_mcu04_or44_before_sending14(self):
        self.start(0x42)
        for index in range(5):
            self.submit([], 1100 + index * 20)
        self.write(self.current._next_outgoing_packet(1180), 1180)
        self.assertTrue(self.current._track_green_recovery_waiting)
        self.submit([self.target], 1200)
        self.assertEqual(self.current.mode, s.CMD_TRACK)
        self.assertEqual(self.current._latest_packet, s.build_coordinate_packet(4, 0, 0))
        self.assertNotIn(s.build_event_packet(0x14), self.current._pending_events)
        self.command(0x04, 1250)
        self.assertFalse(self.current._track_green_recovery_waiting)
        self.submit([self.target], 1300)
        self.assertEqual(self.current._next_outgoing_packet(1300), s.build_event_packet(0x14))


class DebugDirectionalPreviewTests(unittest.TestCase):
    def setUp(self):
        self.current = controller(s.CMD_TRACK)
        self.current.target_label = LABELS[4]
        self.current._post07_started = True
        self.target = block(288, 224, 4)

    def preview(self, objects, selected=None, generation=None, target_label=None):
        before = {key: repr(value) for key, value in vars(self.current).items()}
        with patch.object(s, "_now_ms", return_value=1000):
            result = self.current.debug_arrangement_objects(
                objects, LABELS, actual_selected=selected, frame_generation=generation,
                target_label=target_label)
        self.assertEqual({key: repr(value) for key, value in vars(self.current).items()}, before)
        return result

    def test_preview_matches14_for_all_directions_and_marks_every_unique_blocker(self):
        left, right = block(230, 224, 5), block(350, 224, 6)
        vertical, corner = block(288, 280, 3), block(350, 280, 3)
        for others, expected, blockers in (([], 0x24, []), ([corner], 0x24, []),
                ([left, right], 0x32, [left, right]), ([vertical], 0x42, [vertical]),
                ([left, right, vertical], 0x02, [left, right, vertical])):
            with self.subTest(event=expected):
                raw = [self.target] + others
                candidate, event, neighbors = self.preview(raw, self.target)
                self.assertIs(candidate, self.target)
                self.assertEqual(event, expected)
                self.assertEqual(event, self.current._directional_arrangement_event(self.target, raw))
                self.assertEqual(neighbors, blockers)

    def test_fixed_red_upper_diagonal_preview_is_yellow_and_matches_real14(self):
        self.current.target_label = LABELS[5]
        upper_left, upper_right = block(280, 220, 5), block(296, 220, 6)
        side = block(350, 224, 3)
        for others, event in (([upper_left, upper_right], 0x24),
                               ([upper_left, upper_right, side], 0x02)):
            raw = [self.target] + others
            result = self.preview(raw, target_label="sqarered")
            self.assertIs(result[0], self.target)
            self.assertEqual(result[1], event)
            self.assertEqual(result[1], self.current._directional_arrangement_event(self.target, raw))
            img = Mock()
            overlay.draw_debug_yolo_detections(img, raw, 576, 448, arrangement_preview=result)
            if event == 0x24:
                self.assertEqual(result[2], [])
                self.assertTrue(all(call.kwargs["color"] == overlay.SELECTED_COLOR
                                    for call in img.draw_rect.call_args_list))
            else:
                self.assertEqual({id(obj) for obj in result[2]}, {id(obj) for obj in others})

    def test_casualty_front_or_ratio_exception_removes_red_blockers_but_not_horizontal(self):
        behind, ahead, horizontal = block(288, 190, 4), block(288, 280, 3), block(350, 224, 5)
        for width, others, expected in ((20, [behind], 0x24), (20, [behind, ahead], 0x42),
                (30, [ahead], 0x42), (30.01, [ahead], 0x24),
                (32, [behind, horizontal], 0x02), (32, [horizontal], 0x32)):
            target = SimpleNamespace(x=288-width*.5, y=214, w=width, h=20, class_id=4, score=.9)
            raw = [target] + others
            result = self.preview(raw, target)
            self.assertEqual(result[1], expected)
            if expected == 0x24:
                self.assertEqual(result[2], [])
            img = Mock()
            overlay.draw_debug_yolo_detections(img, raw, 576, 448, arrangement_preview=result)
            expected = [(tuple(rect), overlay.SELECTED_COLOR)
                        for rect in selected_target_rectangles(target, 576, 448)]
            expected += [(tuple(overlay.debug_yolo_rect(obj, 576, 448)), "red") for obj in result[2]]
            self.assertCountEqual([(call.args, call.kwargs["color"])
                                   for call in img.draw_rect.call_args_list], expected)
            img.draw_string.assert_not_called()

    def test_render_omits_clear_other_targets_and_clips_target_expansion(self):
        target, neighbor, unrelated = block(4, 224, 4), block(50, 224, 5), block(500, 100, 6)
        raw = [unrelated, target, neighbor]
        result = self.preview(raw, target)
        self.assertEqual(result[1:], (0x32, [neighbor]))
        img = Mock()
        overlay.draw_debug_yolo_detections(img, raw, 576, 448, arrangement_preview=result)
        self.assertEqual([call.args for call in img.draw_rect.call_args_list],
                         [(0, 214, 14, 20), (0, 160, 68, 128), (40, 214, 20, 20)])
        self.assertEqual([call.kwargs["color"] for call in img.draw_rect.call_args_list],
                         [overlay.SELECTED_COLOR, overlay.SELECTED_COLOR, "red"])
        img.reset_mock()
        overlay.draw_debug_yolo_detections(img, raw, 576, 448, arrangement_preview=(None, None, []))
        img.draw_rect.assert_not_called()

    def test_preview_preserves_approached_identity_instead_of_choosing_a_clearer_target(self):
        approached, clear = block(100, 224, 4), self.target
        blocker = block(100, 280, 5)
        self.current._track_target_lock = self.current._final_target_identity(approached)
        candidate, event, neighbors = self.preview([clear, approached, blocker])
        self.assertIs(candidate, approached)
        self.assertEqual(event, 0x42)
        self.assertEqual(neighbors, [blocker])
        self.current._track_target_lock = self.current._final_target_identity(clear)
        self.assertIs(self.preview([clear, approached, blocker], approached)[0], approached)
        self.assertEqual(self.preview([block(100, 224, 5)]), (None, None, []))
        self.assertEqual(self.preview([]), (None, None, []))

    def test_roi_cargo_and_both_zone_colors_side_margins_are_not_red_neighbors(self):
        target, inside = block(288, 340, 4), block(288, 410, 5)
        for zone_id in (1, 2):
            blocker = block(350, 340, 5)
            zone = SimpleNamespace(x=400, y=330, w=30, h=20, class_id=zone_id)
            candidate, event, neighbors = self.preview([target, inside, blocker, zone], target)
            self.assertIs(candidate, target)
            self.assertEqual((event, neighbors), (0x24, []))
        self.assertEqual(self.preview([inside], inside), (None, None, []))
        candidate, event, neighbors = self.preview([target, inside], target)
        self.assertIs(candidate, target)
        self.assertEqual((event, neighbors), (0x24, []))

    def test_missing_zone_color_memory_is_read_without_extending_or_mutating_it(self):
        blocker = block(350, 224, 5)
        for zone_id in (1, 2):
            self.current._supply_recent_zones = [(zone_id, (340, 210, 400, 240), 750)]
            self.assertEqual(self.preview([self.target, blocker], self.target)[1:], (0x24, []))
            self.current._supply_recent_zones = [(zone_id, (340, 210, 400, 240), 0)]
            self.assertEqual(self.preview([self.target, blocker], self.target)[1:], (0x32, [blocker]))

    def test51_supply_neighbors_are_exempt_but_other_tasks_and_foreign_cargo_still_block(self):
        target, supply, foreign = block(288, 224, 6), block(350, 224, 5), block(288, 280, 4)
        self.current.target_label = s.TARGET_BLACK_OR_GREEN
        self.assertEqual(self.preview([target, supply], target)[1:], (0x24, []))
        self.assertEqual(self.preview([target, supply, foreign], target)[1:], (0x42, [foreign]))
        self.current.target_label = LABELS[6]
        self.assertEqual(self.preview([target, supply], target)[1:], (0x32, [supply]))

    def test_default_green_preview_needs_no_uart_confirmation_and_stale_frames_are_ignored(self):
        self.current.target_label = None
        target, other = block(288, 224, 5), block(350, 224, 3)
        self.assertEqual(self.preview([target, other]), (target, 0x32, [other]))
        self.assertEqual(self.preview([target, other], generation=-1), (None, None, []))

    def test_forced_red_preview_ignores_mcu_color_and_foreign_actual_or_locked_target(self):
        red, neighbor, old_green = self.target, block(350, 224, 3), block(288, 250, 5)
        old_identity = self.current._final_target_identity(old_green)
        self.assertEqual(config.DEBUG_ARRANGEMENT_TARGET_LABEL, "sqarered")
        for label in (None, LABELS[3], LABELS[5], LABELS[6], s.TARGET_BLACK_OR_GREEN):
            for mode in (s.CMD_SEARCH, s.CMD_TRACK, s.MODE_WAIT_14_HANDSHAKE, s.MODE_FINAL_ROI_TRACK):
                with self.subTest(task=label, mode=mode):
                    self.current.target_label, self.current.mode = label, mode
                    self.current._search_target_lock = self.current._track_target_lock = old_identity
                    self.current._final_target_lock = self.current._initial_confirmation_anchor = old_identity
                    # The green is visible and closer to the old green anchor,
                    # but it must not become this casualty preview's target.
                    result = self.preview([red, neighbor, old_green], old_green, target_label="sqarered")
                    self.assertIs(result[0], red)
                    self.assertEqual(result[1], 0x02)
                    self.assertEqual({id(obj) for obj in result[2]}, {id(neighbor), id(old_green)})
                    self.assertEqual(self.preview([neighbor, old_green], old_green,
                                                 target_label="sqarered"), (None, None, []))

    def test_forced_red_retains_matching_identity_and_applies_front_and_ratio_exceptions(self):
        self.current.target_label = s.TARGET_BLACK_OR_GREEN
        actual_green = block(500, 200, 5)
        behind, ahead = block(288, 190, 5), block(288, 280, 6)
        for width, neighbor, event in ((20, behind, 0x24), (20, ahead, 0x42),
                                       (30, ahead, 0x42), (30.01, ahead, 0x24)):
            red = SimpleNamespace(x=288-width*.5, y=214, w=width, h=20, class_id=4, score=.9)
            result = self.preview([red, neighbor, actual_green], actual_green, target_label="sqarered")
            self.assertIs(result[0], red)
            self.assertEqual(result[1:], (event, [] if event == 0x24 else [neighbor]))
        approached, clear = block(100, 224, 4), self.target
        self.current._track_target_lock = self.current._final_target_identity(approached)
        self.assertIs(self.preview([clear, approached], target_label="sqarered")[0], approached)


if __name__ == "__main__":
    unittest.main()
