"""Host regression tests for black loss and the MCU11/03 red handshake."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from test_vision_rules import LABELS, block, near_view_block, controller, confirm_final_load_check, write_near_view_request, SingleFrameTrackingScenario
import app_config as config
import serial_controller as s


class RedSwitchHandshakeTests(SingleFrameTrackingScenario):
    """Retain opt-in legacy handshake coverage; deployed default is OFF."""
    def setUp(self):
        for setting in (patch.object(config, "UART_BLACK_TO_RED_PRIORITY_ENABLED", True),
                        patch.object(config, "VISION_DIAGNOSTICS_ENABLED", False),
                        patch("builtins.print")):
            setting.start()
            self.addCleanup(setting.stop)

    def make_tracked(self, mode=s.CMD_SEARCH, command=s.CMD_SELECT_BLACK_OR_GREEN,
                     class_id=6):
        current = controller(mode)
        target = block(450, 100, class_id)
        with patch.object(s, "_now_ms", return_value=1000):
            current._handle_command(command)
            current.mode = mode
            self.assertIs(current.submit([target], LABELS, raw_objects=[target]), target)
        self.assertEqual(current._last_tracked_class_id, class_id)
        return current

    def submit_at(self, current, now, objects=(), raw=None, generation=None):
        objects = list(objects)
        with patch.object(s, "_now_ms", return_value=now):
            return current.submit(objects, LABELS, raw_objects=list(raw) if raw is not None else objects,
                                  frame_generation=generation)

    def start_switch(self, mode=s.CMD_SEARCH, command=s.CMD_SELECT_BLACK_OR_GREEN):
        current = self.make_tracked(mode, command)
        red = block(100, 150, 4)
        # Satisfy both the elapsed-time and fresh-frame loss minima.
        for now in range(1100, 2100, 100):
            self.submit_at(current, now, raw=[red])
        self.assertEqual(current.mode, s.MODE_WAIT_RED_SWITCH)
        self.assertEqual(current._red_switch_stage, "wait11")
        return current, red

    def acknowledge_switch(self, current):
        with patch.object(s, "_now_ms", return_value=2200):
            current.process_received_bytes(s.build_command_packet(0x11) + s.build_command_packet(0x03))
        self.assertEqual(current.mode, s.CMD_SEARCH)
        self.assertIsNone(current._red_switch_stage)
        self.assertEqual(current._search_target_lock[0], 4)
        self.assertIsNone(current._latest_packet)

    def test_black_hold_requires_time_and_frames_in_03_04_and24(self):
        modes = ((s.CMD_SEARCH, "_search_target_lock", s.EVENT_SEARCH_NO_TARGET),
                 (s.CMD_TRACK, "_track_target_lock", s.EVENT_TRACK_NO_TARGET),
                 (s.MODE_FINAL_ROI_TRACK, "_final_target_lock", s.EVENT_NO_TARGET))
        for mode, anchor_name, event in modes:
            with self.subTest(mode=mode):
                current = self.make_tracked(mode, s.CMD_SELECT_BLACK_OBJECT)
                anchor = getattr(current, anchor_name)
                for now in range(1100, 1180):
                    self.assertIsNone(self.submit_at(current, now))
                    self.assertIsNone(current._latest_packet)
                    self.assertEqual(getattr(current, anchor_name), anchor)
                    self.assertIsNone(current._next_outgoing_packet(now))
                self.assertIsNone(current._next_outgoing_packet(1899))
                self.submit_at(current, 1900)
                self.assertIsNone(getattr(current, anchor_name))
                self.assertEqual(current._next_outgoing_packet(1900), s.build_event_packet(event))

    def test_elapsed800ms_alone_does_not_bypass_frame_minimum(self):
        for mode, minimum in ((s.CMD_SEARCH, 5), (s.CMD_TRACK, 5), (s.MODE_FINAL_ROI_TRACK, 10)):
            with self.subTest(mode=mode):
                current = self.make_tracked(mode, s.CMD_SELECT_BLACK_OBJECT)
                self.submit_at(current, 1100)
                for index in range(minimum - 2):
                    self.submit_at(current, 2000 + index)
                self.assertIsNone(current._next_outgoing_packet(2100))
                self.submit_at(current, 2200)
                self.assertIsNotNone(current._next_outgoing_packet(2200))

    def test_black_reappearing_restarts_entire_loss_clock_without_stale_output(self):
        current = self.make_tracked()
        for now in range(1100, 1800, 100):
            self.submit_at(current, now)
        black = block(460, 110, 6)
        self.assertIs(self.submit_at(current, 1800, raw=[black]), black)
        self.assertIsNone(current._tracking_missing_since_ms)
        self.assertEqual(current._latest_packet, s.build_coordinate_packet(6, 172, -114))
        self.submit_at(current, 1900)
        for now in range(2000, 2600, 100):
            self.submit_at(current, now)
            self.assertIsNone(current._next_outgoing_packet(now))
        self.submit_at(current, 2700)
        self.assertEqual(current._next_outgoing_packet(2700), s.build_event_packet(s.EVENT_SEARCH_NO_TARGET))

    def test_short_black_misses_do_not_switch_even_with_red_visible(self):
        current = self.make_tracked()
        red = block(100, 150, 4)
        for now in range(1100, 1800, 100):
            self.submit_at(current, now, raw=[red])
            self.assertEqual(current.target_label, s.TARGET_BLACK_OR_GREEN)
            self.assertIsNone(current._red_switch_stage)
            self.assertIsNone(current._next_outgoing_packet(now))

    def test_31_unlocked_red_priority_requests_handshake_without_coordinates(self):
        current = controller(s.CMD_SEARCH)
        with patch.object(s, "_now_ms", return_value=1000):
            current._handle_command(s.CMD_SELECT_BLACK_OBJECT)
        self.assertIsNone(self.submit_at(current, 1100, raw=[block(288, 410, 4)]))
        self.assertEqual(current.mode, s.MODE_WAIT_RED_SWITCH)
        self.assertIsNone(current._latest_packet)
        self.assertEqual(current._next_outgoing_packet(1100), s.build_event_packet(0x11))

    def test_51_without_actual_black_or_last_id5_never_requests11(self):
        for acquired in (False, True):
            with self.subTest(acquired_green=acquired):
                current = (self.make_tracked(class_id=5) if acquired else controller(s.CMD_SEARCH))
                if not acquired:
                    current._handle_command(s.CMD_SELECT_BLACK_OR_GREEN)
                for now in range(1100, 2300, 100):
                    self.submit_at(current, now, raw=[block(100, 150, 4)])
                self.assertIsNone(current._red_switch_stage)
                self.assertEqual(current.target_label, s.TARGET_BLACK_OR_GREEN)
                self.assertNotIn(s.build_event_packet(0x11), current._pending_events)

    def test_visible_locked_black_is_not_preempted_in_03_04_or24(self):
        for mode in (s.CMD_SEARCH, s.CMD_TRACK, s.MODE_FINAL_ROI_TRACK):
            current = self.make_tracked(mode)
            for now in range(1100, 2300, 100):
                black, red = block(460, 100, 6), block(100, 150, 4)
                self.assertIs(self.submit_at(current, now, raw=[black, red]), black)
            self.assertIsNone(current._red_switch_stage)

    def test_red_inside_safety_zone_is_not_a_switch_candidate(self):
        current = self.make_tracked()
        red = block(100, 150, 4)
        zone = SimpleNamespace(x=50, y=100, w=100, h=100, class_id=1, score=.9)
        for now in range(1100, 2300, 100):
            self.submit_at(current, now, raw=[red, zone])
        self.assertIsNone(current._red_switch_stage)
        self.assertEqual(current.target_label, s.TARGET_BLACK_OR_GREEN)

    def test_red_in_safety_zone_side_margins_is_not_a_switch_candidate(self):
        for x in (150, 350):
            with self.subTest(x=x):
                current = self.make_tracked()
                red = block(x, 150, 4)
                zone = SimpleNamespace(x=200, y=100, w=100, h=100, class_id=1, score=.9)
                for now in range(1100, 2300, 100):
                    self.submit_at(current, now, raw=[red, zone])
                self.assertIsNone(current._red_switch_stage)
                self.assertEqual(current.target_label, s.TARGET_BLACK_OR_GREEN)
                self.assertNotIn(s.build_event_packet(0x11), current._pending_events)

    def test_first_green_arrangement_and05_06_cannot_request11(self):
        modes = (s.CMD_SEARCH, s.CMD_TRACK, s.MODE_FINAL_ROI_TRACK,
                 s.MODE_ARRANGE_RIGHTMOST, s.MODE_ARRANGE_LEFTMOST,
                 s.MODE_NEAR_VIEW_CHECK, s.MODE_SEARCH_ZONE)
        for mode in modes:
            for command in (s.CMD_SELECT_GREEN_OBJECT, s.CMD_SELECT_BLACK_OR_GREEN):
                current = controller(mode)
                current._handle_command(command)
                current.mode = mode
                # No valid black history. Seeing red alone is not permission.
                self.submit_at(current, 1000, raw=[block(100, 150, 4)])
                self.assertNotIn(s.build_event_packet(0x11), current._pending_events)
                self.assertIsNone(current._red_switch_stage)

    def test_request_discards_old34_and_emits11_once_then_stays_silent(self):
        current = self.make_tracked(s.MODE_FINAL_ROI_TRACK)
        current._pending_events = [s.build_event_packet(0x34), s.build_event_packet(0xEE)]
        red = block(288, 410, 4)
        for now in range(1100, 2100, 100):
            self.submit_at(current, now, raw=[red])
        self.assertEqual(current._pending_events, [s.build_event_packet(0x11)])
        self.assertEqual(current._next_outgoing_packet(2100), s.build_event_packet(0x11))
        for now in range(2200, 4200, 100):
            self.submit_at(current, now, raw=[red])
            current.send_event(0x34)
            self.assertIsNone(current._latest_packet)
            self.assertIsNone(current._next_outgoing_packet(now))

    def test_switch_allowed_during24_EE_wait_and_after_F1_recovery(self):
        for acknowledge in (False, True):
            current = self.make_tracked(s.MODE_FINAL_ROI_TRACK)
            for now in range(1100, 2100, 100):
                self.submit_at(current, now)
            self.assertEqual(current._next_outgoing_packet(2100), s.build_event_packet(0xEE))
            self.assertTrue(current._final_search_waiting_ack)
            self.assertEqual(current._last_tracked_class_id, 6)
            if acknowledge:
                current._handle_command(s.CMD_FINAL_SEARCH_DONE)
            for now in range(2200, 3300, 100):
                self.submit_at(current, now, raw=[block(100, 150, 4)])
            self.assertEqual(current.mode, s.MODE_WAIT_RED_SWITCH)
            self.assertFalse(current._final_search_waiting_ack)
            self.assertEqual(current._next_outgoing_packet(3300), s.build_event_packet(0x11))

    def test_RX11_then03_preserves_candidate_and_requires_post_handshake_frame(self):
        current, candidate = self.start_switch(s.MODE_FINAL_ROI_TRACK)
        self.assertEqual(current._next_outgoing_packet(2100), s.build_event_packet(0x11))
        old_generation = current.frame_generation()
        current._handle_command(s.CMD_SEARCH)  # Out-of-order03 must not resume.
        self.assertEqual(current._red_switch_stage, "wait11")
        current._handle_command(s.CMD_SELECT_RED_OBJECT)
        self.assertEqual(current._red_switch_stage, "wait03")
        moved, other = block(110, 160, 4), block(288, 224, 4)
        self.assertIsNone(self.submit_at(current, 2150, raw=[moved], generation=old_generation))
        self.assertEqual(current._red_switch_candidate[1:3], (100.0, 150.0))
        self.assertIs(self.submit_at(current, 2160, raw=[other, moved]), moved)
        before03 = current.frame_generation()
        current._handle_command(s.CMD_SEARCH)
        self.assertEqual(current._search_target_lock[1:3], (110.0, 160.0))
        self.assertIsNone(self.submit_at(current, 2300, raw=[moved], generation=before03))
        self.assertIsNone(current._next_outgoing_packet(2300))
        # Association uses the preserved candidate, not the more central red.
        self.assertIs(self.submit_at(current, 2400, raw=[other, moved],
                                     generation=current.frame_generation()), moved)
        self.assertEqual(current._next_outgoing_packet(2400), s.build_coordinate_packet(4, -178, -64))

    def test_duplicate_RX11_does_not_reset_red_lock_or_recovery_round(self):
        current, red = self.start_switch()
        self.acknowledge_switch(current)
        self.submit_at(current, 2300, raw=[red])
        identity = current._search_target_lock
        current._search_no_target_initial_sent = True
        generation = current.frame_generation()
        current._handle_command(s.CMD_SELECT_RED_OBJECT)
        self.assertEqual(current.frame_generation(), generation)
        self.assertEqual(current._search_target_lock, identity)
        self.assertTrue(current._search_no_target_initial_sent)

    def test_unrelated_old_commands_cannot_unpause_switch_new_task_can_cancel(self):
        current, red = self.start_switch()
        for command in (0x04, 0x14, 0x24, 0x25, 0xF1):
            current._handle_command(command)
            self.assertEqual(current.mode, s.MODE_WAIT_RED_SWITCH)
        current._handle_command(s.CMD_SELECT_GREEN_OBJECT)
        self.assertIsNone(current._red_switch_stage)
        self.assertFalse(current._red_priority_tracking)
        self.assertEqual(current.target_label, "sqareredgreen")
        self.assertEqual(current.mode, s.CMD_SEARCH)
        self.assertNotIn(s.build_event_packet(0x11), current._pending_events)
        self.assertIsNone(self.submit_at(current, 2300, raw=[red]))

    def test_prepared_old_coordinate_or34_cannot_be_written_after_switch(self):
        for packet_kind in ("coordinate", "event34"):
            current = self.make_tracked(s.MODE_FINAL_ROI_TRACK)
            if packet_kind == "event34":
                current._pending_events.append(s.build_event_packet(0x34))
            prepared = current._next_outgoing_packet(1000)
            generation = current._outgoing_generation
            current._serial = SimpleNamespace(write=Mock(side_effect=lambda packet: len(packet)))
            for now in range(1100, 2100, 100):
                self.submit_at(current, now, raw=[block(100, 150, 4)])
            current._write_packet(prepared, generation=generation)
            current._serial.write.assert_not_called()
            request = current._next_outgoing_packet(2100)
            with patch.object(s, "_now_ms", return_value=2100):
                current._write_packet(request, generation=current._outgoing_generation)
            current._serial.write.assert_called_once_with(s.build_event_packet(0x11))

    def test_command_during_submit_discards_in_progress_old_frame(self):
        current, red = self.start_switch()
        self.acknowledge_switch(current)
        original = current._select_locked_target

        def command_during_selection(*args, **kwargs):
            result = original(*args, **kwargs)
            current._handle_command(s.CMD_SELECT_GREEN_OBJECT)
            return result

        with patch.object(current, "_select_locked_target", side_effect=command_during_selection):
            self.assertIsNone(self.submit_at(current, 2300, raw=[red]))
        self.assertEqual(current.target_label, "sqareredgreen")
        self.assertIsNone(current._latest_packet)
        self.assertIsNone(current._search_target_lock)

    def test_red_association_failure_uses_E3_then_recovers_without_random_rescreen(self):
        current, red = self.start_switch()
        self.acknowledge_switch(current)
        anchor = current._search_target_lock
        for now in range(2300, 2800, 100):
            self.submit_at(current, now)
            self.assertIsNone(current._latest_packet)
        self.assertEqual(current._search_target_lock, anchor)
        self.assertEqual(current._next_outgoing_packet(2800), s.build_event_packet(s.EVENT_SEARCH_NO_TARGET))
        self.assertIsNone(current._next_outgoing_packet(3300))
        with patch.object(s, "_now_ms", return_value=3400):
            current._handle_command(s.CMD_SEARCH_SWEEP_DONE)
        # RX13 discards the pre-sweep observation; a fresh frame must confirm
        # that this same red identity is still absent before anotherE3.
        self.submit_at(current, 3500)
        self.assertIsNone(current._next_outgoing_packet(3899))
        self.assertEqual(current._next_outgoing_packet(3900), s.build_event_packet(s.EVENT_SEARCH_NO_TARGET))
        with patch.object(current, "_select_final_roi_target", side_effect=AssertionError("must associate, not rescreen")):
            self.assertIs(self.submit_at(current, 4000, raw=[red]), red)
        self.assertEqual(current._next_outgoing_packet(4000), s.build_coordinate_packet(4, -188, -74))
        self.assertFalse(current._search_no_target_initial_sent)

    def test_red_recovery04_keeps_identity_waits_new_frame_and_uses_image_center(self):
        current, red = self.start_switch()
        self.acknowledge_switch(current)
        self.submit_at(current, 2300, raw=[red])
        # Explicit recovery04 can inherit this same red search identity.
        current._handle_command(s.CMD_TRACK)
        self.assertEqual(current._track_target_lock[0], 4)
        self.submit_at(current, 2400, raw=[red])
        for now in range(2500, 3000, 100):
            self.submit_at(current, now)
        self.assertEqual(current._next_outgoing_packet(3000), s.build_event_packet(s.EVENT_TRACK_NO_TARGET))
        anchor = current._track_target_lock
        self.assertIsNotNone(anchor)
        old_generation = current.frame_generation()
        current._handle_command(s.CMD_TRACK)
        self.assertEqual(current._track_target_lock, anchor)
        self.assertIsNone(current._latest_packet)
        moved = block(110, 160, 4)
        self.assertIsNone(self.submit_at(current, 3100, raw=[moved], generation=old_generation))
        with patch.object(current, "_select_final_roi_target", side_effect=AssertionError("no recovery rescreen")):
            self.assertIs(self.submit_at(current, 3200, raw=[block(288, 224, 4), moved]), moved)
        self.assertEqual(current._next_outgoing_packet(3200), s.build_coordinate_packet(4, -178, -64))

    def test_switched_red_03_to04_to24_then05_uses_red_rules(self):
        for ids, expected in (([4], s.EVENT_ZONE_SEARCH_STARTED),
                              ([5], s.EVENT_ROI_READY), ([6], s.EVENT_ROI_READY),
                              ([5, 6], s.EVENT_ROI_READY), ([4, 5], s.EVENT_ROI_READY),
                              ([4, 4], s.EVENT_ROI_READY), ([], s.EVENT_SKIP_TO_FINAL_TRACK)):
            with self.subTest(ids=ids):
                current, _ = self.start_switch()
                self.acknowledge_switch(current)
                centered = block(288, 224, 4)
                self.submit_at(current, 2300, raw=[centered])
                self.assertEqual(current._next_outgoing_packet(2300), s.build_event_packet(0x04))
                current._handle_command(s.CMD_TRACK)
                self.submit_at(current, 2400, raw=[centered])
                self.assertEqual(current._next_outgoing_packet(2400), s.build_event_packet(0x14))
                current._handle_command(s.CMD_TRACK_CENTER_DONE)
                self.submit_at(current, 2500, [centered])
                self.assertEqual(current._next_outgoing_packet(2500), s.build_event_packet(0x24))
                for now in (2600, 2620, 2640):
                    self.submit_at(current, now, raw=[block(288, 410, 4)])
                self.assertEqual(current._next_outgoing_packet(2640), s.build_event_packet(0x34))
                with patch.object(s, "_now_ms", return_value=2700):
                    current._handle_command(s.CMD_SEARCH_BLUE_ZONE)
                self.assertEqual(write_near_view_request(current, 2700), s.build_event_packet(0x05))
                objects = [near_view_block(100 + i * 30, 200, class_id) for i, class_id in enumerate(ids)]
                for now in (3300, 3400, 3500):
                    self.submit_at(current, now, objects)
                result_at = 3500
                if expected == s.EVENT_ZONE_SEARCH_STARTED:
                    result_at = confirm_final_load_check(current, ids, result_at)
                    self.assertEqual(current._next_outgoing_packet(result_at), s.build_coordinate_packet(4, 0, 0))
                self.assertEqual(current._next_outgoing_packet(result_at), s.build_event_packet(expected))
                self.assertEqual(current.target_label, "sqarered")
                self.assertFalse(current._post07_supply_tracking)
                self.assertFalse(current._first_green_both_sides_required)

    def test_switched_red_06_uses_casualty_aim_point(self):
        current, _ = self.start_switch()
        self.acknowledge_switch(current)
        with patch.object(s, "_now_ms", return_value=2300):
            current._handle_command(s.CMD_SEARCH_BLUE_ZONE)
        write_near_view_request(current, 2300)
        for now in (2900, 3000, 3100):
            self.submit_at(current, now, [near_view_block(100, 150, 4)])
        ready = confirm_final_load_check(current, [4], 3100)
        report = current._next_outgoing_packet(ready)
        self.assertEqual(report, s.build_coordinate_packet(4, 0, 0))
        current._serial = SimpleNamespace(write=lambda packet: len(packet))
        with patch.object(s, "_now_ms", return_value=ready):
            current._write_packet(report, current._outgoing_generation)
        done = current._next_outgoing_packet(ready + 50)
        self.assertEqual(done, s.build_event_packet(0x06))
        with patch.object(s, "_now_ms", return_value=ready + 50):
            current._write_packet(done, current._outgoing_generation)
        zone_at = ready + 50 + config.UART_WIDE_VIEW_SETTLE_MS
        current._next_outgoing_packet(zone_at)
        zone = SimpleNamespace(x=20, y=20, w=200, h=100, class_id=1, score=.9)
        self.submit_at(current, zone_at + 100, raw=[zone])
        self.assertEqual(current._next_outgoing_packet(zone_at + 100), s.build_event_packet(0x16))
        self.assertEqual(current._next_outgoing_packet(zone_at + 101), s.build_coordinate_packet(1, -148, -334))
        zone.h = 400  # Above25% after16: casualty now uses width/2.
        self.submit_at(current, zone_at + 200, raw=[zone])
        self.assertEqual(current._next_outgoing_packet(zone_at + 200), s.build_coordinate_packet(1, -168, -184))


class AutomaticRedPriorityDisabledTests(unittest.TestCase):
    """Default11 switching is restricted to a post07 material E3 search."""

    def setUp(self):
        for setting in (patch.object(config, "VISION_DIAGNOSTICS_ENABLED", False), patch("builtins.print")):
            setting.start()
            self.addCleanup(setting.stop)

    def make(self, mode=s.MODE_FINAL_ROI_TRACK, label="triangualrblack"):
        current = controller(mode)
        current.target_label, current._post07_started = label, True
        current._serial = SimpleNamespace(write=Mock(side_effect=lambda packet: len(packet)))
        return current

    def submit(self, current, raw=(), now=1000, generation=None):
        with patch.object(s, "_now_ms", return_value=now):
            return current.submit(list(raw), LABELS, raw_objects=list(raw), frame_generation=generation)

    def warm(self, current, raw, now=1000):
        return [self.submit(current, raw, now + index * 50) for index in range(3)]

    def prepare_e3(self, current, start=1000):
        for index in range(config.UART_SEARCH_NO_TARGET_FRAMES):
            self.submit(current, now=start + index * 200)
        packet = current._next_outgoing_packet(start + 850)
        self.assertEqual(packet, s.build_event_packet(0xE3))
        return packet

    def test_post07_material_e3_requests_red_on_one_frame_and_resumes_e3_after_loss(self):
        for label in ("sqareredgreen", "triangualrblack", s.TARGET_BLACK_OR_GREEN):
            with self.subTest(label=label):
                current = self.make(s.CMD_SEARCH, label)
                e3 = self.prepare_e3(current)
                with patch.object(s, "_now_ms", return_value=1850):
                    current._write_packet(e3, current._outgoing_generation)
                red = block(160, 150, 4)
                self.submit(current, [red], 1900)
                self.assertEqual(current.mode, s.MODE_WAIT_RED_SWITCH)
                self.assertIsNone(current._latest_packet)
                request = current._next_outgoing_packet(1900)
                self.assertEqual(request, s.build_event_packet(0x11))
                with patch.object(s, "_now_ms", return_value=1900):
                    current._write_packet(request, current._outgoing_generation)
                current.process_received_bytes(s.build_command_packet(0x03))
                self.assertEqual(current._red_switch_stage, "wait11")
                current.process_received_bytes(s.build_command_packet(0x11))
                self.assertEqual(current._red_switch_stage, "wait03")
                self.submit(current, [red], 2050)
                self.assertIsNone(current._latest_packet)
                self.assertIsNone(current._next_outgoing_packet(2050))
                current.process_received_bytes(s.build_command_packet(0x03))
                self.assertEqual(current.target_label, "sqarered")
                self.assertEqual(current.mode, s.CMD_SEARCH)
                self.assertIs(self.submit(current, [red], 2100), red)
                self.assertEqual(current._latest_packet[2], 4)
                for index in range(config.UART_SEARCH_NO_TARGET_FRAMES):
                    now = 2200 + index * 50
                    self.submit(current, now=now)
                    if index < config.UART_SEARCH_NO_TARGET_FRAMES - 1:
                        self.assertIsNone(current._next_outgoing_packet(now))
                self.assertEqual(current._next_outgoing_packet(now), e3)
                self.assertEqual((current.mode, current.target_label), (s.CMD_SEARCH, "sqarered"))

    def test_e3_red_switch_retries_once_100ms_after_the_first_successful_write(self):
        current = self.make(s.CMD_SEARCH, s.TARGET_BLACK_OR_GREEN)
        e3 = self.prepare_e3(current)
        with patch.object(s, "_now_ms", return_value=1850):
            current._write_packet(e3, current._outgoing_generation)
        self.submit(current, [block(160, 150, 4)], 1900)
        request = current._next_outgoing_packet(1950)
        self.assertEqual(request, s.build_event_packet(0x11))
        with patch.object(s, "_now_ms", return_value=1950):
            current._write_packet(request, current._outgoing_generation)
        self.assertIsNone(current._next_outgoing_packet(2049))
        with patch.object(s, "_now_ms", return_value=2049):
            current._write_packet(request, current._outgoing_generation)
        self.assertEqual(current._next_outgoing_packet(2050), request)
        with patch.object(s, "_now_ms", return_value=2050):
            current._write_packet(request, current._outgoing_generation)
        self.assertIsNone(current._next_outgoing_packet(2150))
        self.assertEqual([call.args[0] for call in current._serial.write.call_args_list],
                         [e3, request, request])
        self.assertEqual(current._red_switch_stage, "wait11")
        self.assertIsNone(current._latest_packet)

    def test_e3_red_switch_ack_discards_prepared_retry(self):
        current = self.make(s.CMD_SEARCH, s.TARGET_BLACK_OR_GREEN)
        e3 = self.prepare_e3(current)
        with patch.object(s, "_now_ms", return_value=1850):
            current._write_packet(e3, current._outgoing_generation)
        self.submit(current, [block(160, 150, 4)], 1900)
        request = current._next_outgoing_packet(1900)
        with patch.object(s, "_now_ms", return_value=1900):
            current._write_packet(request, current._outgoing_generation)
        retry = current._next_outgoing_packet(2000)
        self.assertEqual(retry, request)
        generation = current._outgoing_generation
        with patch.object(s, "_now_ms", return_value=2000):
            current.process_received_bytes(s.build_command_packet(0x11))
            current._write_packet(retry, generation)
        self.assertEqual([call.args[0] for call in current._serial.write.call_args_list], [e3, request])
        self.assertEqual(current._red_switch_stage, "wait03")

    def test_material_reply_cancels_current11_and_next_e3_rearms_without_fe(self):
        for command, label in ((0x51, s.TARGET_BLACK_OR_GREEN),
                               (0x31, "triangualrblack")):
            with self.subTest(command=command):
                current = self.make(s.CMD_SEARCH, label)
                red = block(160, 150, 4)
                material_commands = s.build_command_packet(command) + s.build_command_packet(0x03)
                request = s.build_event_packet(0x11)
                e3 = self.prepare_e3(current)
                with patch.object(s, "_now_ms", return_value=1850):
                    current._write_packet(e3, current._outgoing_generation)
                self.warm(current, [red], 1900)
                self.assertEqual(current._next_outgoing_packet(2000), request)
                with patch.object(s, "_now_ms", return_value=2000):
                    current._write_packet(request, current._outgoing_generation)
                with patch.object(s, "_now_ms", return_value=2010):
                    current.process_received_bytes(material_commands)
                self.assertEqual((current.mode, current.target_label), (s.CMD_SEARCH, label))
                self.submit(current, [red], 2100)
                self.assertIsNone(current._next_outgoing_packet(2100))

                self.assertEqual(self.prepare_e3(current, 2100), e3)
                with patch.object(s, "_now_ms", return_value=2950):
                    current._write_packet(e3, current._outgoing_generation)
                self.submit(current, [red], 3000)
                self.assertEqual(current.mode, s.MODE_WAIT_RED_SWITCH)
                self.assertEqual(current._next_outgoing_packet(3000), request)
                with patch.object(s, "_now_ms", return_value=3000):
                    current._write_packet(request, current._outgoing_generation)
                self.assertEqual([call.args[0] for call in current._serial.write.call_args_list],
                                 [e3, request, e3, request])

                with patch.object(s, "_now_ms", return_value=3310):
                    current.process_received_bytes(s.build_command_packet(0x11) + s.build_command_packet(0x03))
                self.warm(current, [red], 3320)
                self.assertEqual(current._latest_packet[2], 4)

    def test_material_command_after_short11_cancels_the_handshake(self):
        current = self.make(s.CMD_SEARCH, s.TARGET_BLACK_OR_GREEN)
        e3 = self.prepare_e3(current)
        with patch.object(s, "_now_ms", return_value=1850):
            current._write_packet(e3, current._outgoing_generation)
        self.warm(current, [block(160, 150, 4)], 1900)
        request = current._next_outgoing_packet(2000)
        self.assertEqual(request, s.build_event_packet(0x11))
        current._serial.write.side_effect = None
        current._serial.write.return_value = 2
        with patch.object(s, "_now_ms", return_value=2000):
            current._write_packet(request, current._outgoing_generation)
        current.process_received_bytes(s.build_command_packet(0x51) + s.build_command_packet(0x03))
        self.assertIsNone(current._red_switch_stage)
        self.assertEqual(current.mode, s.CMD_SEARCH)
        self.assertIsNone(current._next_outgoing_packet(2010))

    def test_material_switch_requires_active_sent_e3_and_no_supplement(self):
        for case in ("short_write", "completed_sweep", "before07", "supplement"):
            with self.subTest(case=case):
                current = self.make(s.CMD_SEARCH, s.TARGET_BLACK_OR_GREEN)
                current._post07_started = case != "before07"
                e3 = self.prepare_e3(current)
                if case == "short_write":
                    current._serial.write.side_effect = None
                    current._serial.write.return_value = 2
                with patch.object(s, "_now_ms", return_value=1850):
                    current._write_packet(e3, current._outgoing_generation)
                    if case == "completed_sweep":
                        current.process_received_bytes(s.build_command_packet(0x13))
                if case == "supplement":
                    current._supply_collect_started_ms = 1850
                self.warm(current, [block(160, 150, 4)], 1900)
                self.assertEqual(current.mode, s.CMD_SEARCH)
                self.assertIsNone(current._red_switch_stage)
                self.assertNotIn(s.build_event_packet(0x11), current._pending_events)

    def test_default_off_and_missing_configuration_cannot_initiate_switch(self):
        self.assertIs(config.UART_BLACK_TO_RED_PRIORITY_ENABLED, False)
        current = self.make()
        settings = {key: value for key, value in vars(config).items()
                    if key != "UART_BLACK_TO_RED_PRIORITY_ENABLED"}
        with patch.object(s, "config", SimpleNamespace(**settings)):
            self.warm(current, [block(160, 150, 4)])
        self.assertEqual(current.target_label, "triangualrblack")
        self.assertNotEqual(current.mode, s.MODE_WAIT_RED_SWITCH)
        self.assertIsNone(current._red_switch_stage)

    def test_unlocked31_red_only_does_not_request11_or_pause_for_handshake(self):
        current = self.make()
        for now in range(1000, 2201, 100):
            self.submit(current, [block(160, 150, 4)], now)
            self.assertEqual(current.mode, s.MODE_FINAL_ROI_TRACK)
            self.assertIsNone(current._red_switch_stage)
            self.assertIsNone(current._red_switch_candidate)
            self.assertIsNone(current._latest_packet)
            self.assertNotIn(s.build_event_packet(0x11), current._pending_events)
        self.assertEqual(current._next_outgoing_packet(2200), s.build_event_packet(0xEE))

    def test31_and51_tracked_black_loss_with_visible_red_keep_normal_loss_and_recovery(self):
        for label in ("triangualrblack", s.TARGET_BLACK_OR_GREEN):
            for mode, event in ((s.CMD_SEARCH, 0xE3), (s.CMD_TRACK, 0xE4),
                                (s.MODE_WAIT_04_HANDSHAKE, None), (s.MODE_WAIT_14_HANDSHAKE, None),
                                (s.MODE_FINAL_ROI_TRACK, 0xEE)):
                with self.subTest(label=label, mode=mode):
                    current, black = self.make(mode, label), block(400, 100, 6)
                    self.assertIs(self.warm(current, [black])[-1], black)
                    for now in range(1200, 2501, 100):
                        self.submit(current, [block(160, 150, 4)], now)
                        self.assertEqual(current.target_label, label)
                        self.assertEqual(current.mode, mode)
                        self.assertIsNone(current._red_switch_stage)
                        self.assertIsNone(current._latest_packet)
                    expected = s.build_event_packet(event) if event is not None else None
                    self.assertEqual(current._next_outgoing_packet(2500), expected)
                    self.assertIs(self.warm(current, [black], 2600)[-1], black)
                    self.assertEqual(current._latest_packet[2], 6)
                    self.assertFalse(current._red_priority_tracking)

    def test51_can_follow_green_with_red_visible_without_switching(self):
        current = self.make(s.CMD_SEARCH, s.TARGET_BLACK_OR_GREEN)
        green, red = block(400, 100, 5), block(160, 150, 4)
        self.assertIs(self.warm(current, [green, red])[-1], green)
        self.assertEqual(current._latest_packet[2], 5)
        self.assertEqual(current.target_label, s.TARGET_BLACK_OR_GREEN)
        self.assertIsNone(current._red_switch_stage)

    def test31_can_continue_black_with_red_visible_without_switching(self):
        current = self.make()
        black, red = block(400, 100, 6), block(160, 150, 4)
        self.assertIs(self.warm(current, [black, red])[-1], black)
        self.assertEqual(current._latest_packet[2], 6)
        self.assertIsNone(current._red_switch_stage)

    def test_pending_manual_and_prepared_tx11_are_blocked_at_actual_wire(self):
        current = self.make(s.CMD_SEARCH)
        request = s.build_event_packet(0x11)
        current.send_event(0x11)
        self.assertEqual(current._pending_events, [])
        current._pending_events.append(request)
        self.assertIsNone(current._next_outgoing_packet(1000))
        self.assertEqual(current._pending_events, [])
        current._write_packet(request)
        current._serial.write.assert_not_called()

    def test_mcu11_03_still_selects_red_and_publishes_only_new_fresh_red_frames(self):
        for label in ("triangualrblack", s.TARGET_BLACK_OR_GREEN):
            current = self.make(label=label)
            self.warm(current, [block(400, 100, 6)])
            old_generation = current.frame_generation()
            with patch.object(s, "_now_ms", return_value=1200):
                current.process_received_bytes(s.build_command_packet(0x11) + s.build_command_packet(0x03))
            self.assertEqual(current.target_label, "sqarered")
            self.assertEqual(current.mode, s.CMD_SEARCH)
            self.assertIsNone(current._red_switch_stage)
            self.assertIsNone(current._latest_packet)
            red = block(160, 150, 4)
            self.assertIsNone(self.submit(current, [red], 1250, old_generation))
            self.assertIs(self.warm(current, [red], 1300)[-1], red)
            self.assertEqual(current._latest_packet, s.build_coordinate_packet(4, -153, -229))
            self.assertNotIn(s.build_event_packet(0x11), current._pending_events)


if __name__ == "__main__":
    unittest.main()
