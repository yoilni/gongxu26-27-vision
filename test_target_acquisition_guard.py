"""Default target confirmation and 500ms zone hold across cargo tasks.

Unlike legacy single-scene fixtures, these tests submit distinct camera frames
with advancing timestamps. No timing gate is disabled for these scenarios.
"""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from test_vision_rules import LABELS, block, near_view_block, write_near_view_request
import app_config as config
import serial_controller as s


class TargetAcquisitionGuardTests(unittest.TestCase):
    def setUp(self):
        for helper in (patch("builtins.print"),
                       patch.object(config, "VISION_DIAGNOSTICS_ENABLED", False)):
            helper.start()
            self.addCleanup(helper.stop)

    def make(self, mode=s.CMD_SEARCH, class_id=5, post07=False):
        with patch.object(config, "UART_COORDINATE_ENABLED", False):
            current = s.VisionSerialController(576, 448, tracking_roi=[0, 319, 576, 129])
        current.mode, current.target_label = mode, LABELS[class_id]
        current._post07_started = post07
        current._serial = SimpleNamespace(write=Mock(side_effect=lambda packet: len(packet)))
        return current

    def submit(self, current, objects=(), now=1000, verified=None, generation=None):
        with patch.object(s, "_now_ms", return_value=now):
            return current.submit(list(objects) if verified is None else verified, LABELS,
                                  raw_objects=list(objects), frame_generation=generation)

    def command(self, current, command, now=1000):
        with patch.object(s, "_now_ms", return_value=now):
            current.process_received_bytes(s.build_command_packet(command))

    def warm(self, current, objects, now=1000, verified=None):
        return [self.submit(current, objects, now + index * 50, verified)
                for index in range(3)]

    @staticmethod
    def zone(class_id=1, x=200, y=100, w=100, h=100):
        return SimpleNamespace(class_id=class_id, x=x, y=y, w=w, h=h, score=.9)

    def test_defaults_and_every_initial_task_stage_require_fresh_frames(self):
        self.assertEqual(config.UART_NEW_TARGET_CONFIRM_FRAMES, 3)
        self.assertEqual(config.UART_POST07_SUPPLY_CONFIRM_FRAMES, 2)
        self.assertEqual(config.UART_TARGET_ZONE_HOLD_MS, 500)
        for mode in (s.CMD_SEARCH, s.CMD_TRACK, s.MODE_FINAL_ROI_TRACK):
            for class_id, post07 in ((5, False), (3, True), (4, True), (5, True), (6, True)):
                with self.subTest(mode=mode, class_id=class_id, post07=post07):
                    current, target = self.make(mode, class_id, post07), block(160, 180, class_id)
                    fast_supply = post07 and class_id in (5, 6)
                    verified = [] if fast_supply else None
                    required = 2 if fast_supply else 3
                    for index in range(required - 1):
                        now = 1000 + 50 * index
                        self.assertIsNone(self.submit(current, [target], now, verified=verified))
                        self.assertIsNone(current._latest_packet)
                        self.assertEqual(current._pending_events, [])
                        self.assertIsNone(current._next_outgoing_packet(now))
                    self.assertIs(self.submit(current, [target], 1000 + 50 * (required - 1),
                                              verified=verified), target)
                    self.assertEqual(current._latest_packet[2], class_id)

    def test_new_raw_yolo24_cannot_bypass_task_confirmation(self):
        for class_id in (4, 5, 6):
            current = self.make(s.MODE_FINAL_ROI_TRACK, class_id, True)
            target, neighbor = block(160, 180, class_id), block(165, 180, 3)
            expected = [None, None if class_id == 4 else target, target]
            self.assertEqual(self.warm(current, [target, neighbor], verified=[]), expected)
            self.assertEqual(current._final_target_lock[0], class_id)

    def test_post07_51_locks_crowded_supply_on_second_raw_frame(self):
        current = self.make(s.CMD_SEARCH, 5, True)
        current.target_label = s.TARGET_BLACK_OR_GREEN
        target, neighbor = block(110, 280, 6), block(145, 260, 5)
        self.assertFalse(current._matches_final_primary_spacing(target, [target, neighbor]))
        self.assertIsNone(self.submit(current, [target, neighbor], 1000, verified=[]))
        self.assertIsNone(current._latest_packet)
        self.assertIs(self.submit(current, [target, neighbor], 1050, verified=[]), target)
        self.assertEqual(current._latest_packet[2], 6)

    def test_first_green03_emits04_only_on_third_observation(self):
        current, target = self.make(), block(313, 340)
        for now in (1000, 1050):
            self.assertIsNone(self.submit(current, [target], now))
            self.assertEqual(current.mode, s.CMD_SEARCH)
            self.assertIsNone(current._next_outgoing_packet(now))
        self.assertIs(self.submit(current, [target], 1100), target)
        self.assertEqual(current.mode, s.MODE_WAIT_04_HANDSHAKE)
        self.assertEqual(current._next_outgoing_packet(1100), s.build_event_packet(4))
        self.command(current, 4, 1150)
        self.assertIs(self.submit(current, [target], 1200), target)
        self.assertEqual(current._next_outgoing_packet(1200), s.build_event_packet(0x14))

    def test_new04_cannot_send14_during_warmup(self):
        current, target = self.make(s.CMD_TRACK), block(313, 320)
        self.assertEqual(self.warm(current, [target]), [None, None, target])
        self.assertEqual(current._pending_events, [s.build_event_packet(0x14)])

    def test_initial_roi_entry34_also_requires_task_confirmation(self):
        for class_id, post07 in ((5, False), (4, True), (5, True), (6, True)):
            current = self.make(s.MODE_FINAL_ROI_TRACK, class_id, post07)
            target = block(313 if class_id == 4 else 288, 420, class_id)
            confirmed_at = 1050 if post07 and class_id in (5, 6) else 1100
            for now in range(1000, confirmed_at, 50):
                self.submit(current, [target], now)
                self.assertFalse(current._final_roi_sent)
                self.assertIsNone(current._next_outgoing_packet(now))
            self.submit(current, [target], confirmed_at)
            # For post07 red, the same frames may satisfy acquisition and
            # deep-ROI confirmation. Neither condition bypasses the other.
            self.assertTrue(current._final_roi_sent)
            self.assertEqual(current._next_outgoing_packet(confirmed_at), s.build_event_packet(0x34))

    def test_actual51_capture_line_uses_only_confirmed_new_target(self):
        for class_id in (5, 6):
            current = self.make(s.MODE_FINAL_ROI_TRACK, class_id, True)
            current.target_label = s.TARGET_BLACK_OR_GREEN
            cargo, new = block(288, 425, 5), block(70, 360, class_id)
            self.assertIsNone(self.submit(current, [cargo, new], 1000))
            self.assertIsNone(current._supply_roi_handoff)
            self.assertIsNone(current._next_outgoing_packet(1000))
            self.assertIsNone(self.submit(current, [cargo, new], 1050))
            self.assertTrue(current._final_roi_sent)
            self.assertEqual(current._supply_roi_handoff[0], class_id)
            self.assertEqual(current._next_outgoing_packet(1050), s.build_event_packet(0x34))
            self.submit(current, [cargo, block(288, 420, class_id)], 1150)
            self.assertTrue(current._final_roi_sent)
            self.assertIsNone(current._latest_packet)

    def test_distinct_red_in_deep_roi_can_capture_after_three_confirmed_frames(self):
        current = self.make(s.MODE_FINAL_ROI_TRACK, 4, True)
        outside, inside = block(100, 180, 4), block(313, 400, 4)
        self.warm(current, [outside])
        for now in (1150, 1200):
            self.assertIs(self.submit(current, [outside, inside], now), outside)
            self.assertFalse(current._final_roi_sent)
        self.assertIs(self.submit(current, [outside, inside], 1250), outside)
        self.assertTrue(current._final_roi_sent)
        self.assertEqual(current._final_target_lock[:3], (4, 100, 180))
        self.assertEqual(current._next_outgoing_packet(1250), s.build_event_packet(0x34))

    def test_inherited_locks_follow_raw_immediately_without_jump_area_gate(self):
        for mode in (s.CMD_SEARCH, s.CMD_TRACK, s.MODE_WAIT_04_HANDSHAKE,
                     s.MODE_WAIT_14_HANDSHAKE, s.MODE_FINAL_ROI_TRACK):
            for class_id in (3, 4, 5, 6):
                current = self.make(mode, class_id, class_id != 5)
                old, moved = block(100, 140, class_id), block(450, 170, class_id)
                moved.w, moved.h = 70, 50
                attribute = ("_search_target_lock" if mode == s.CMD_SEARCH else
                             "_final_target_lock" if mode == s.MODE_FINAL_ROI_TRACK else "_track_target_lock")
                setattr(current, attribute, current._final_target_identity(old))
                self.assertIs(self.submit(current, [moved], 1000, verified=[]), moved)
                self.assertEqual(current._latest_packet[2], class_id)

    def test_raw_miss_never_sends_cached_coordinates(self):
        current, target = self.make(), block(160, 180)
        self.warm(current, [target])
        self.assertIsNone(self.submit(current, [], 1150))
        self.assertIsNone(current._latest_packet)
        self.assertIsNone(current._next_outgoing_packet(1150))
        self.assertIs(self.submit(current, [target], 1200, verified=[]), target)

    def test_frame_miss_motion_class_and_stale_gap_restart_new_evidence(self):
        for change in ("miss", "jump", "class", "gap"):
            current = self.make(s.MODE_FINAL_ROI_TRACK)
            first, second = block(100, 180), block(110, 185)
            self.submit(current, [first], 1000)
            self.submit(current, [second], 1050)
            target, start = block(110, 185), 1150
            if change == "miss":
                self.submit(current, [], 1100)
            elif change == "jump":
                target = block(300, 185)
            elif change == "class":
                current.target_label = LABELS[4]
                target.class_id = 4
            else:
                start = 1301
            self.assertEqual(self.warm(current, [target], start), [None, None, target])

    def test_one_prior_box_cannot_confirm_two_current_boxes(self):
        current = self.make()
        original = block(100, 180)
        _, previous = current._confirm_supply_outside_targets([original], [], 1000)
        _, previous = current._confirm_supply_outside_targets([original], previous, 1050)
        near, other = block(101, 180), block(120, 180)
        confirmed, records = current._confirm_supply_outside_targets([near, other], previous, 1100)
        self.assertEqual(confirmed, {current._final_target_identity(near)})
        self.assertEqual([hits for _, hits, _ in records], [3, 1])

    def test_both_safety_colors_block_all_tasks_through500ms_inclusive(self):
        for color in (1, 2):
            for class_id, post07 in ((5, False), (3, True), (4, True), (5, True), (6, True)):
                current = self.make(s.MODE_FINAL_ROI_TRACK, class_id, post07)
                target = block(288, 150, class_id)
                self.submit(current, [target, self.zone(color)], 1000)
                for now in (1100, 1300, 1500):
                    self.assertIsNone(self.submit(current, [target], now))
                    self.assertIsNone(current._latest_packet)
                expected = [None, target if post07 and class_id in (5, 6) else None, target]
                self.assertEqual(self.warm(current, [target], 1501), expected)

    def test_zone_memory_has100px_inclusive_x_margin_and_no_y_expansion(self):
        current = self.make(s.MODE_FINAL_ROI_TRACK)
        self.submit(current, [self.zone()], 1000)
        for x, y, excluded in ((100, 150, True), (400, 150, True), (99, 150, False),
                               (401, 150, False), (288, 99, False), (288, 201, False)):
            target = block(x, y)
            filtered = current._exclude_recent_supply_zones([target], current._supply_recent_zones)
            self.assertEqual(filtered, [] if excluded else [target])

    def test_fresh_color_replaces_its_old_geometry_independently(self):
        current = self.make(s.MODE_FINAL_ROI_TRACK)
        self.submit(current, [self.zone(1), self.zone(2, y=250)], 1000)
        self.submit(current, [self.zone(1, x=450)], 1100)
        self.assertEqual(len(current._supply_recent_zones), 2)
        by_color = {color: bounds for color, bounds, _ in current._supply_recent_zones}
        self.assertEqual(by_color[1][0], 350)
        self.assertEqual(by_color[2][1], 250)
        self.submit(current, [self.zone(1, x=450)], 1501)
        self.assertEqual(len(current._supply_recent_zones), 1)

    def test_no_zone_ever_seen_still_allows_outside_targets_after_three(self):
        current, target = self.make(), block(160, 180)
        self.assertEqual(self.warm(current, [target]), [None, None, target])
        self.assertEqual(current._supply_recent_zones, [])

    def test_safety_zone_overrides_an_active_lock_and_prepared_coordinate(self):
        for mode in (s.CMD_SEARCH, s.CMD_TRACK, s.MODE_FINAL_ROI_TRACK):
            current, target = self.make(mode), block(288, 150)
            self.warm(current, [target])
            prepared = current._latest_packet
            self.assertIsNone(self.submit(current, [target, self.zone()], 1150))
            self.assertIsNone(self.submit(current, [target], 1200))
            with patch.object(s, "_now_ms", return_value=1201):
                current._write_packet(prepared, current.frame_generation())
            current._serial.write.assert_not_called()

    def test_generation_mismatch_cannot_warm_history_or_refresh_zones(self):
        current, target = self.make(), block(160, 180)
        old = current.frame_generation()
        self.command(current, s.CMD_SEARCH)
        self.submit(current, [target, self.zone()], 1050, generation=old)
        self.assertEqual(current._supply_outside_history, [])
        self.assertEqual(current._supply_recent_zones, [])
        self.assertEqual(self.warm(current, [target], 1100), [None, None, target])

    def test_fe_discards_evidence_and_zone_history_for_first_green(self):
        current = self.make(s.MODE_FINAL_ROI_TRACK, 4, True)
        self.submit(current, [self.zone(), block(50, 180, 4)], 1000)
        self.command(current, 0xFE, 1050)
        self.assertFalse(current._post07_started)
        self.assertEqual(current._supply_recent_zones, [])
        self.assertEqual(current._supply_outside_history, [])
        self.command(current, 0x21, 1100)
        self.command(current, 3, 1150)
        target = block(160, 180)
        self.assertEqual(self.warm(current, [target], 1200), [None, None, target])

    def test_post22_does_not_send_e2_or_completion_while_confirming(self):
        for class_id, post07 in ((5, False), (4, True), (5, True), (6, True)):
            current = self.make(s.MODE_POST_ARRANGE_CHECK, class_id, post07)
            target = block(160, 180, class_id)
            for now in (1000, 1050):
                self.submit(current, [target], now)
                self.assertIsNone(current._next_outgoing_packet(now))
                self.assertEqual(current.mode, s.MODE_POST_ARRANGE_CHECK)
            self.submit(current, [target], 1100)
            self.assertEqual(current.mode, s.MODE_FINAL_ROI_TRACK)
            self.assertEqual(current._next_outgoing_packet(1100), s.build_event_packet(0x24))

    def test_post22_rejected_spacing_uses04_after_three_and_not_e2(self):
        for class_id, post07 in ((5, False), (4, True)):
            current = self.make(s.MODE_POST_ARRANGE_CHECK, class_id, post07)
            target, neighbor = block(160, 180, class_id), block(165, 180, 3)
            for now in (1000, 1050):
                self.submit(current, [target, neighbor], now, verified=[])
                self.assertIsNone(current._next_outgoing_packet(now))
            self.submit(current, [target, neighbor], 1100, verified=[])
            self.assertEqual(current.mode, s.MODE_WAIT_04_HANDSHAKE)
            self.assertEqual(current._latest_packet[2], class_id)
            self.assertEqual(current._next_outgoing_packet(1100), s.build_event_packet(4))

    def test_post22_prepared_e2_canceled_by_pending_candidate_but_stall_not_permanent(self):
        current = self.make(s.MODE_POST_ARRANGE_CHECK, 4, True)
        self.submit(current, [], 1000)
        prepared = current._next_outgoing_packet(1000)
        self.assertEqual(prepared, s.build_event_packet(0xE2))
        self.submit(current, [block(160, 180, 4)], 1050)
        with patch.object(s, "_now_ms", return_value=1051):
            current._write_packet(prepared, current.frame_generation())
        current._serial.write.assert_not_called()
        self.assertEqual(current._next_outgoing_packet(1600), prepared)

    @patch.object(config, "UART_BLACK_TO_RED_PRIORITY_ENABLED", True)
    def test_black_red_switch_requires_confirmed_red_then_keeps_handshake_identity(self):
        current = self.make(s.MODE_FINAL_ROI_TRACK, 6, True)
        red = block(160, 180, 4)
        for now in (1000, 1050):
            self.submit(current, [red], now)
            self.assertNotEqual(current.mode, s.MODE_WAIT_RED_SWITCH)
        self.submit(current, [red], 1100)
        self.assertEqual(current.mode, s.MODE_WAIT_RED_SWITCH)
        self.assertEqual(current._next_outgoing_packet(1100), s.build_event_packet(0x11))
        self.command(current, 0x11, 1150)
        self.command(current, 3, 1200)
        self.assertIs(self.submit(current, [red], 1250, verified=[]), red)
        self.assertEqual(current._latest_packet[2], 4)

    def test_first_green14_preserves_existing_lock_without_extra_three_frame_acquisition(self):
        current, target = self.make(s.MODE_WAIT_14_HANDSHAKE), block(288, 224)
        current._track_target_lock = current._final_target_identity(target)
        with patch.object(config, "UART_DIRECT_CAPTURE_AFTER_14", False):
            self.command(current, 0x14)
        self.submit(current, [target], 1050)
        self.assertEqual(current._initial_confirmation_frames, 1)
        self.assertIsNone(current._next_outgoing_packet(1050))
        self.submit(current, [target], 1100)
        self.assertEqual(current._next_outgoing_packet(1100), s.build_event_packet(0x24))

    def test_wide_red_vertical_exception_still_requires_new_target_confirmation(self):
        current = self.make(s.MODE_WAIT_14_HANDSHAKE, 4, True)
        with patch.object(config, "UART_DIRECT_CAPTURE_AFTER_14", False):
            self.command(current, 0x14)
        target = SimpleNamespace(x=258, y=214, w=60, h=20, class_id=4, score=.9)
        objects = [target, block(288, 280, 5)]
        for now in (1050, 1100):
            self.submit(current, objects, now, verified=[])
            self.assertIsNone(current._next_outgoing_packet(now))
            self.assertTrue(current._initial_arrangement_decision_pending)
        self.submit(current, objects, 1150, verified=[])
        self.assertEqual(current._next_outgoing_packet(1150), s.build_event_packet(0x24))
        self.assertEqual(current._final_target_lock, current._final_target_identity(target))

    def test_raw_unverified03_does_not_gain_verification_from_three_frame_guard(self):
        current, target = self.make(), block(160, 180)
        self.assertEqual(self.warm(current, [target], verified=[]), [None, None, None])
        self.assertIsNone(current._search_target_lock)
        self.assertIs(self.submit(current, [target], 1150), target)

    def test03_roi_filter_still_precedes_confirmation(self):
        current, target = self.make(), block(288, 360)
        self.assertEqual(self.warm(current, [target]), [None, None, None])
        self.assertEqual(current._supply_outside_history, [])

    def test06_zone_is_immediate_without_cargo_confirmation_or_zone_memory(self):
        current = self.make(s.MODE_SEARCH_ZONE, 4, True)
        current.zone_label = LABELS[1]
        zone = self.zone(1, x=100, y=100, w=200, h=100)
        self.assertIs(self.submit(current, [zone], verified=[]), zone)
        self.assertEqual(current._latest_packet[2], 1)
        self.assertIsNone(self.submit(current, [], 1050))
        self.assertIsNone(current._latest_packet)
        self.assertEqual(current._supply_recent_zones, [])

    def test05_full_image_inventory_does_not_exclude_cargo_inside_safety_zone(self):
        current = self.make(s.MODE_NEAR_VIEW_CHECK)
        current._near_view_started_ms = 1000
        target = near_view_block(288, 180, 5)
        for now in (1500, 1550, 1600):
            self.submit(current, [target, self.zone()], now)
        self.assertEqual(current._zone_load_recheck_source, "before06")
        self.assertEqual(current._near_view_green_missing_cycles, 0)

    def test_missing_config_defaults_remain_three_and500(self):
        with patch.dict(vars(config), {}, clear=False):
            for key in ("UART_NEW_TARGET_CONFIRM_FRAMES", "UART_TARGET_ZONE_HOLD_MS",
                        "UART_NEW_TARGET_ASSOC_MAX_PX"):
                delattr(config, key)
            current, target = self.make(), block(160, 180)
            self.assertEqual(self.warm(current, [target]), [None, None, target])
            self.submit(current, [target, self.zone()], 1150)
            self.assertIsNone(self.submit(current, [target], 1200))

    def test_confirmation_frames_are_not_counted_as_missing(self):
        for mode, counter in ((s.CMD_SEARCH, "_search_no_target_frames"),
                              (s.CMD_TRACK, "_track_no_target_frames"),
                              (s.MODE_FINAL_ROI_TRACK, "_final_no_target_frames")):
            current = self.make(mode)
            for now in (1000, 1050):
                self.submit(current, [block(160, 180)], now)
                self.assertEqual(getattr(current, counter), 0)
            for index in range(5):
                self.submit(current, [], 1100 + index * 50)
                self.assertEqual(getattr(current, counter), index + 1)

    def test_prepared_e3_waits_while_a_new_target_is_confirming(self):
        current = self.make()
        for now in (1000, 1050, 1100, 1150, 1200):
            self.submit(current, [], now)
        event = current._next_outgoing_packet(1200)
        self.assertEqual(event, s.build_event_packet(0xE3))
        self.submit(current, [block(160, 180)], 1250)
        with patch.object(s, "_now_ms", return_value=1251):
            current._write_packet(event, current.frame_generation())
        current._serial.write.assert_not_called()

    def test24_loss_and_new_acquisition_repeats_three_not_old_evidence(self):
        current, first = self.make(s.MODE_FINAL_ROI_TRACK), block(160, 180)
        self.warm(current, [first])
        self.submit(current, [], 1150)
        self.assertIsNone(current._final_target_lock)
        new = block(300, 180)
        self.assertEqual(self.warm(current, [new], 1200), [None, None, new])

    def test_prepared34_is_canceled_after_new_zone_exclusion_for_first_green(self):
        current = self.make(s.MODE_FINAL_ROI_TRACK)
        target = block(288, 420)
        self.warm(current, [target])
        old34 = current._next_outgoing_packet(1100)
        self.assertEqual(old34, s.build_event_packet(0x34))
        self.submit(current, [target, self.zone(y=360)], 1150)
        with patch.object(s, "_now_ms", return_value=1151):
            current._write_packet(old34, current.frame_generation())
        current._serial.write.assert_not_called()

    def test_micro_association_is_immediate_but_missing_zone_remains_protected(self):
        for mode in s.MICRO_MODES:
            current, target = self.make(mode, 4, True), block(288, 150, 4)
            current._track_target_lock = current._final_target_identity(target)
            current._micro_request_sent = True
            current._micro_started_ms = 1000
            self.assertIs(self.submit(current, [target], 1050, verified=[]), target)
            self.submit(current, [target, self.zone(2)], 1100, verified=[])
            self.assertIsNone(self.submit(current, [target], 1150, verified=[]))
            self.assertIsNone(current._latest_packet)
            self.assertEqual(current.mode, mode)

    def test_active51_direct03_keeps_budget_and_confirms_only_new_outside_cargo(self):
        current = self.make(s.MODE_FINAL_ROI_TRACK, 5, True)
        self.command(current, s.CMD_SELECT_BLACK_OR_GREEN, 1000)
        self.command(current, s.CMD_SEARCH_BLUE_ZONE, 1000)
        write_near_view_request(current, 1000)
        for now in (1500, 1550, 1600):
            self.submit(current, [near_view_block(200, 180, 5)], now)
        self.assertEqual(current.mode, s.MODE_WAIT_SUPPLY_SEARCH)
        self.assertEqual(current._next_outgoing_packet(1600), s.build_event_packet(3))
        self.command(current, s.CMD_SELECT_BLACK_OR_GREEN, 1650)
        self.command(current, s.CMD_SEARCH, 1700)
        self.assertEqual(current.mode, s.CMD_SEARCH)
        self.assertIsNone(current._zone_load_recheck)
        target, cargo = block(160, 180, 6), block(288, 425, 5)
        self.assertIsNone(self.submit(current, [cargo, target], 1949))
        self.assertEqual(self.warm(current, [cargo, target], 1950, verified=[]), [None, target, target])
        self.assertEqual(current._supply_collect_attempts, 1)
        self.assertEqual(current._supply_collect_started_ms, 1600)
        self.assertEqual(current._latest_packet[2], 6)
        self.assertIsNone(current._zone_load_recheck)

    def test_same_view04_keeps_zone_memory_and_new_camera_view_clears_it(self):
        current = self.make(s.MODE_WAIT_04_HANDSHAKE)
        current._track_target_lock = current._final_target_identity(block(288, 150))
        self.submit(current, [self.zone()], 1000)
        self.command(current, 4, 1050)
        self.assertTrue(current._supply_recent_zones)
        self.assertIsNone(self.submit(current, [block(288, 150)], 1100))
        self.command(current, s.CMD_SEARCH_BLUE_ZONE, 1150)
        self.assertEqual(current._supply_recent_zones, [])
        self.assertEqual(current._supply_outside_history, [])


if __name__ == "__main__":
    unittest.main()
