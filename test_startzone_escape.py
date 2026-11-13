"""AE turn handshake: current startzone >4%, no steering until actual RXAE."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from test_vision_rules import LABELS, block, near_view_block
from detection_confidence import DetectionConfidenceFilter
from model_class_mapping import ModelClassMapper
import app_config as config
import serial_controller as s


class StartzoneEscapeTests(unittest.TestCase):
    """Opt-in AE protocol coverage; the deployed experiment defaults to OFF."""
    def setUp(self):
        for helper in (patch("builtins.print"), patch.object(config, "VISION_DIAGNOSTICS_ENABLED", False),
                       patch.object(config, "UART_STARTZONE_ESCAPE_ENABLED", True)):
            helper.start()
            self.addCleanup(helper.stop)

    def make(self, mode=s.CMD_SEARCH, label="sqareredgreen", post07=False):
        with patch.object(config, "UART_COORDINATE_ENABLED", False):
            current = s.VisionSerialController(576, 448, [0, 319, 576, 129])
        current.mode, current.target_label, current._post07_started = mode, label, post07
        current._serial = SimpleNamespace(write=Mock(side_effect=lambda packet: len(packet)))
        return current

    @staticmethod
    def start(x=0, y=0, w=200, h=100, score=.9, class_id=8):
        return SimpleNamespace(x=x, y=y, w=w, h=h, score=score, class_id=class_id)

    def submit(self, current, starts=(), objects=(), now=1000, generation=None):
        with patch.object(s, "_now_ms", return_value=now):
            return current.submit(list(objects), LABELS, raw_objects=list(objects),
                                  startzone_objects=list(starts), frame_generation=generation)

    def write_next(self, current, now=1000):
        with patch.object(s, "_now_ms", return_value=now):
            packet = current._next_outgoing_packet(now)
            if packet is not None:
                current._write_packet(packet, current._outgoing_generation)
            return packet

    def command(self, current, command=0xAE, now=1100):
        with patch.object(s, "_now_ms", return_value=now):
            current.process_received_bytes(s.build_command_packet(command))

    def trigger(self, current, now=1000):
        self.assertIsNone(self.submit(current, [self.start()], now=now))
        self.assertEqual(self.write_next(current, now), s.build_event_packet(0xAE))
        self.assertEqual(current._startzone_escape_phase, "wait_ack")

    def test_wire_format_and_default_threshold(self):
        self.assertEqual(config.UART_STARTZONE_ESCAPE_MIN_AREA_RATIO, .04)
        self.assertEqual(s.build_event_packet(s.EVENT_STARTZONE_ESCAPE), b"\xF1\xF2\xAE\x1F\x2F")
        self.assertEqual(s.build_command_packet(s.CMD_STARTZONE_ESCAPE_DONE), b"\xE1\xE2\xAE\x1E\x2E")
        self.assertIn(s.EVENT_STARTZONE_ESCAPE, s.EVENT_NAMES)

    def test_tracking_and_center_handshake_phases_preempt_same_frame(self):
        for mode in (3, 4, 0x24, s.MODE_WAIT_04_HANDSHAKE, s.MODE_WAIT_14_HANDSHAKE):
            for class_id in (3, 4, 5, 6):
                current = self.make(mode, LABELS[class_id], class_id != 5)
                self.submit(current, [self.start()], [block(288, 340 if mode == 0x24 else 224, class_id)])
                self.assertEqual(current.mode, s.MODE_WAIT_STARTZONE_ESCAPE)
                self.assertIsNone(current._latest_packet)
                self.assertEqual(current._pending_events, [])
                self.assertEqual(self.write_next(current), s.build_event_packet(0xAE))

    def test_other_phases_and_camera_settling_never_start_escape(self):
        for mode in (2, 5, 6, s.MODE_ARRANGE_RIGHTMOST, s.MODE_ARRANGE_LEFTMOST,
                     s.MODE_POST_ARRANGE_CHECK, 0x32, 0x42, s.MODE_WAIT_RED_SWITCH,
                     s.MODE_WAIT_SUPPLY_SEARCH, s.MODE_WAIT_TASK_START):
            current = self.make(mode)
            if mode == s.MODE_WAIT_RED_SWITCH:
                current._red_switch_stage = "wait11"
            self.submit(current, [self.start()])
            self.assertIsNone(current._startzone_escape_phase, hex(mode))
        for phase in ("wait05tx", "count05", "wait06tx", "wait24tx", "wide_settle"):
            current = self.make(0x24)
            current._zone_load_recheck, current._zone_load_resume_ready_ms = phase, 99999
            self.submit(current, [self.start()])
            self.assertIsNone(current._startzone_escape_phase)

    def test_strict_four_percent_float_boundary_and_largest_not_sum(self):
        h = 576 * 448 * .04 / 100
        for starts in ([self.start(w=100, h=h)], [self.start(w=100, h=h-.001)],
                       [self.start(w=100, h=100), self.start(x=150, w=100, h=100)]):
            current = self.make()
            self.submit(current, starts)
            self.assertIsNone(current._startzone_escape_phase)
        current = self.make()
        self.submit(current, [self.start(w=100, h=h+.001)])
        self.assertEqual(current._startzone_escape_phase, "pending")

    def test_float_area_clips_to_whole_image_and_invalid_aux_boxes_are_ignored(self):
        current = self.make()
        self.assertAlmostEqual(current._startzone_area_ratio([self.start(-100, 0, 200, 100)]), 10000/(576*448))
        self.assertEqual(current._startzone_area_ratio([self.start(-100, -100, 1000, 1000)]), 1)
        invalid = [self.start(w=0), self.start(h=-1), self.start(x=float("nan")),
                   self.start(h=float("inf")), self.start(x=600), self.start(class_id=5),
                   self.start(class_id=7), SimpleNamespace(class_id=8)]
        self.assertEqual(current._startzone_area_ratio(invalid), 0)
        self.submit(current, invalid)
        self.assertIsNone(current._startzone_escape_phase)

    def test_threshold_missing_config_fallback_and_custom_override(self):
        current = self.make()
        with patch.object(config, "UART_STARTZONE_ESCAPE_MIN_AREA_RATIO", .1):
            self.submit(current, [self.start()])
            self.assertIsNone(current._startzone_escape_phase)
        value = config.UART_STARTZONE_ESCAPE_MIN_AREA_RATIO
        del config.UART_STARTZONE_ESCAPE_MIN_AREA_RATIO
        try:
            self.submit(current, [self.start()])
            self.assertEqual(current._startzone_escape_phase, "pending")
        finally:
            config.UART_STARTZONE_ESCAPE_MIN_AREA_RATIO = value

    def test_label_and_score_gate_keep_auxiliary_out_of_cargo(self):
        for labels in (["startzone", "sqaregreen"],
                       ["bluesafety", "redsafety", "sqareblue", "sqaregreen", "sqarered", "startzone", "triangualrblack"]):
            mapper, gate = ModelClassMapper(labels), DetectionConfidenceFilter(.5)
            for score, trigger in ((.49, False), (.5, True), (float("nan"), False)):
                native = self.start(class_id=labels.index("startzone"), score=score)
                self.assertEqual(mapper.normalize([native]), [])
                current = self.make()
                self.submit(current, gate.apply_startzones(mapper.startzones([native])))
                self.assertEqual(current._startzone_escape_phase == "pending", trigger)
                self.assertEqual(native.class_id, labels.index("startzone"))

    def test_rotation_silent_with_targets_manual_events_and_long_timers(self):
        current = self.make()
        self.trigger(current)
        for now in (1500, 3000, 20000, 100000):
            self.assertIsNone(self.submit(current, [], [block(288, 224)], now))
            self.assertEqual(current._last_frame_seen_ms, now)
            current.send_event(4)
            self.assertIsNone(self.write_next(current, now))
            self.assertEqual(current.mode, s.MODE_WAIT_STARTZONE_ESCAPE)
        self.assertEqual(current._serial.write.call_count, 1)

    def test_uart_guard_blocks_all_old_packets_even_without_generation_argument(self):
        current = self.make()
        self.submit(current, [self.start()])
        for phase in ("pending", "wait_ack"):
            current._startzone_escape_phase = phase
            for event in s.EVENT_NAMES:
                if event != s.EVENT_STARTZONE_ESCAPE:
                    current._write_packet(s.build_event_packet(event))
            for class_id in range(1, 9):
                current._write_packet(s.build_coordinate_packet(class_id, 1, -2))
        current._serial.write.assert_not_called()

    def test_old_commands_including_inventory_and_target_changes_cannot_release_turn(self):
        current = self.make()
        self.trigger(current)
        for command in (3, 4, 5, 0x14, 0x44, 0x34, 0x24, 0x13, 0xF1,
                        0x11, 0x21, 0x31, 0x51, 0x15, 0x25, 0x32, 0x42, 0x46):
            self.command(current, command)
            self.assertEqual(current._startzone_escape_phase, "wait_ack")
            self.assertEqual(current.target_label, "sqareredgreen")
            self.assertIsNone(self.write_next(current, 2000))
        self.assertEqual(current._startzone_deferred_zone_command, 0x25)

    def test_deferred15_or25_runs_normal05_only_after_ack_for_every_task(self):
        tasks = (("sqareredgreen", False), ("sqarered", True),
                 ("triangualrblack", True), (s.TARGET_BLACK_OR_GREEN, True))
        for command, zone in ((0x15, "redsafety"), (0x25, "bluesafety")):
            for label, post07 in tasks:
                with self.subTest(command=command, label=label):
                    current = self.make(0x24, label, post07)
                    current.zone_label = "bluesafety" if command == 0x15 else "redsafety"
                    current._first_green_near_view_decided = True
                    current._near_view_green_missing_cycles = 1
                    current._supply_collect_attempts, current._supply_collect_started_ms = 1, 500
                    self.trigger(current)
                    generation = current.frame_generation()
                    self.command(current, command)
                    self.assertEqual(current._startzone_deferred_zone_command, command)
                    self.assertEqual(current._startzone_escape_phase, "wait_ack")
                    self.assertEqual(current.frame_generation(), generation)
                    self.assertNotEqual(current.zone_label, zone)
                    self.assertIsNone(self.write_next(current, 1150))
                    self.command(current, now=1200)
                    self.assertIsNone(current._startzone_escape_phase)
                    self.assertIsNone(current._startzone_deferred_zone_command)
                    self.assertEqual(current.mode, s.MODE_NEAR_VIEW_CHECK)
                    self.assertEqual((current.target_label, current._post07_started, current.zone_label),
                                     (label, post07, zone))
                    self.assertTrue(current._first_green_near_view_decided)
                    self.assertEqual(current._near_view_green_missing_cycles, 1)
                    self.assertEqual((current._supply_collect_attempts, current._supply_collect_started_ms), (1, 500))
                    self.assertEqual(current._pending_events, [s.build_event_packet(0x05)])
                    self.assertIsNone(current._near_view_started_ms)
                    for attribute in ("_latest_packet", "_search_target_lock", "_track_target_lock", "_final_target_lock"):
                        self.assertIsNone(getattr(current, attribute))
                    self.assertEqual(current._serial.write.call_count, 1)
                    self.assertEqual(self.write_next(current, 1300), s.build_event_packet(0x05))
                    self.assertEqual(current._near_view_started_ms, 1300)

    def test_cached_zone_command_without_ack_remains_silent_indefinitely(self):
        current = self.make()
        self.trigger(current)
        self.command(current, 0x15)
        for now in (2000, 20000, 100000):
            self.submit(current, [], [near_view_block(150, 200)], now)
            for event in (0x05, 0x06, 0xE3, 0x34):
                current.send_event(event)
            self.assertIsNone(self.write_next(current, now))
            self.assertEqual(current.mode, s.MODE_WAIT_STARTZONE_ESCAPE)
            self.assertEqual(current._startzone_deferred_zone_command, 0x15)
            self.assertIsNone(current._near_view_started_ms)
            self.assertEqual(current._near_view_stable_frames, 0)
        self.assertEqual(current._serial.write.call_count, 1)

    def test_latest_zone_selection_wins_and_duplicate_ack_does_not_restart05(self):
        for commands in ((0x15, 0x25, 0x25), (0x25, 0x15, 0x15)):
            with self.subTest(commands=commands):
                current = self.make()
                self.trigger(current)
                for command in commands:
                    self.command(current, command)
                self.assertEqual(current._startzone_deferred_zone_command, commands[-1])
                self.command(current, now=1200)
                self.assertEqual(current.zone_label, s.ZONE_SEARCH_COMMANDS[commands[-1]])
                self.assertEqual(self.write_next(current, 1300), s.build_event_packet(5))
                self.submit(current, objects=[near_view_block(150, 200)], now=1550)
                self.assertEqual(current._near_view_stable_frames, 1)
                generation = current.frame_generation()
                self.command(current, now=1551)
                self.assertEqual(current.frame_generation(), generation)
                self.assertEqual(current._near_view_started_ms, 1300)
                self.assertEqual(current._near_view_stable_frames, 1)
                self.assertEqual(current._pending_events, [])
                self.assertIsNone(current._startzone_deferred_zone_command)

    def test_zone_cached_before_ae_write_survives_short_write_and_premature_ack(self):
        for sent in (-1, 0, 2):
            with self.subTest(sent=sent):
                current = self.make()
                self.submit(current, [self.start()])
                self.command(current, 0x15)
                self.assertEqual(current._startzone_escape_phase, "pending")
                current._serial.write.side_effect = [sent, 5]
                self.write_next(current, 1100)
                self.command(current, now=1150)
                self.assertEqual(current._startzone_escape_phase, "pending")
                self.assertEqual(current._startzone_deferred_zone_command, 0x15)
                self.assertEqual(current._pending_events, [])
                self.assertIsNone(current._near_view_started_ms)
                self.write_next(current, 1200)
                self.assertEqual(current._startzone_escape_phase, "wait_ack")
                self.command(current, now=1300)
                self.assertEqual(current.mode, s.MODE_NEAR_VIEW_CHECK)
                self.assertEqual(current.zone_label, "redsafety")
                self.assertEqual(current._pending_events, [s.build_event_packet(0x05)])

    def test_deferred05_fences_old_frames_and_starts250ms_only_on_actual05_write(self):
        current = self.make()
        old = current.frame_generation()
        self.submit(current, [self.start()], generation=old)
        pending = current.frame_generation()
        self.write_next(current)
        rotating = current.frame_generation()
        self.command(current, 0x15)
        self.command(current, now=1200)
        after_ack = current.frame_generation()
        target = near_view_block(150, 200)
        for generation in (old, pending, rotating):
            self.submit(current, objects=[target], now=1700, generation=generation)
            for packet in (s.build_coordinate_packet(5, 10, 20),
                           s.build_event_packet(0x34), s.build_event_packet(0x05)):
                current._write_packet(packet, generation)
        self.assertEqual(current._serial.write.call_count, 1)
        for now in (1750, 1760, 1770):
            self.submit(current, objects=[target], now=now, generation=after_ack)
        self.assertEqual(current._near_view_stable_frames, 0)
        current._serial.write.side_effect = [2, 5]
        self.assertEqual(self.write_next(current, 1800), s.build_event_packet(5))
        self.assertIsNone(current._near_view_started_ms)
        self.assertEqual(self.write_next(current, 2000), s.build_event_packet(5))
        self.assertEqual(current._near_view_started_ms, 2000)
        self.submit(current, objects=[target], now=2250, generation=after_ack)
        self.submit(current, objects=[target], now=2249)
        self.assertEqual(current._near_view_stable_frames, 0)
        for now, count in ((2250, 1), (2260, 2)):
            self.submit(current, objects=[target], now=now)
            self.assertEqual(current._near_view_stable_frames, count)
            self.assertEqual(current._pending_events, [])
        self.submit(current, objects=[target], now=2270)
        self.assertEqual(current._zone_load_recheck_source, "before06")
        self.assertEqual(current._pending_events, [s.build_event_packet(5)])

    def test_fragmented_and_coalesced_zone_and_ack_commands_preserve_order(self):
        for commands in ((0x15, 0xAE), (0xAE, 0x15), (0x25, 0x15, 0xAE)):
            for fragmented in (False, True):
                with self.subTest(commands=commands, fragmented=fragmented):
                    current = self.make()
                    self.trigger(current)
                    data = b"".join(s.build_command_packet(command) for command in commands)
                    with patch.object(s, "_now_ms", return_value=1200):
                        if fragmented:
                            received = []
                            for byte in data:
                                received.extend(current.process_received_bytes(bytes([byte])))
                        else:
                            received = current.process_received_bytes(data)
                    self.assertEqual(received, list(commands))
                    self.assertEqual(current.mode, s.MODE_NEAR_VIEW_CHECK)
                    self.assertEqual(current.zone_label, "redsafety")
                    self.assertIsNone(current._startzone_deferred_zone_command)
                    self.assertEqual(current._pending_events, [s.build_event_packet(0x05)])
                    self.assertEqual(current._serial.write.call_count, 1)

    def test_fe_clears_deferred_zone_in_pending_and_wait_ack_and_late_ack_cannot_replay(self):
        for sent in (False, True):
            with self.subTest(sent=sent):
                current = self.make()
                self.submit(current, [self.start()])
                if sent:
                    self.write_next(current)
                self.command(current, 0x15)
                self.command(current, 0xFE, now=1200)
                self.assertIsNone(current._startzone_deferred_zone_command)
                self.command(current, now=1300)
                self.command(current, 0x21, now=1400)
                self.command(current, 0x03, now=1401)
                self.command(current, now=1402)
                self.assertEqual(current.mode, s.CMD_SEARCH)
                self.assertEqual(current.target_label, "sqareredgreen")
                self.assertEqual(current._pending_events, [])
                self.assertIsNone(current._startzone_deferred_zone_command)

    def test_deferred_zone_does_not_reset_or_extend_supply_deadline(self):
        current = self.make(0x24, s.TARGET_BLACK_OR_GREEN, True)
        current._supply_collect_started_ms, current._supply_collect_attempts = 100, 1
        self.trigger(current)
        self.command(current, 0x25)
        self.assertIsNone(self.write_next(current, 16000))
        self.command(current, now=17000)
        self.assertEqual(current.mode, s.MODE_NEAR_VIEW_CHECK)
        self.assertEqual((current._supply_collect_started_ms, current._supply_collect_attempts), (100, 1))
        self.assertEqual(self.write_next(current, 17001), s.build_event_packet(0x05))
        self.assertTrue(current._supply_collect_finished)
        self.assertEqual(current._supply_collect_finish_reason, "total_timeout")
        self.assertEqual(current._zone_load_recheck_source, "before06")
        self.assertEqual(current._near_view_started_ms, 17001)

    def test_short_failed_writes_retry_and_ack_requires_full_request_write(self):
        for sent in (-1, 0, 3):
            current = self.make()
            self.submit(current, [self.start()])
            current._serial.write.side_effect = [sent, 5]
            self.write_next(current)
            self.assertEqual(current._startzone_escape_phase, "pending")
            self.command(current)
            self.assertEqual(current._startzone_escape_phase, "pending")
            self.write_next(current, 1100)
            self.assertEqual(current._startzone_escape_phase, "wait_ack")
            self.assertIsNone(self.write_next(current, 9000))

    def test_serial_exception_never_pretends_request_was_delivered(self):
        current = self.make()
        self.submit(current, [self.start()])
        current._serial.write.side_effect = [OSError("uart"), 5]
        with self.assertRaises(OSError):
            self.write_next(current)
        self.assertEqual(current._startzone_escape_phase, "pending")
        self.command(current)
        self.assertEqual(current._startzone_escape_phase, "pending")
        # Existing transport exceptions stop the UART worker; AE does not
        # hide that failure or falsely accept completion of an unsent turn.

    def test_event_spacing_and_no_successful_request_retransmission(self):
        current = self.make()
        current._last_event_write_ms = 990
        self.submit(current, [self.start()])
        self.assertIsNone(self.write_next(current, 1039))
        self.assertEqual(self.write_next(current, 1040), s.build_event_packet(0xAE))
        self.assertIsNone(self.write_next(current, 100000))

    def test_ack_resumes03_even_black_preserving_task_budgets_and_histories(self):
        for label, post07 in (("sqareredgreen", False), ("sqarered", True),
                              ("triangualrblack", True), (s.TARGET_BLACK_OR_GREEN, True)):
            current = self.make(0x24, label, post07)
            current.zone_label = "redsafety"
            current._first_green_near_view_decided = True
            current._near_view_green_missing_cycles = 1
            current._supply_collect_attempts, current._supply_collect_started_ms = 2, 500
            current._supply_collect_finished, current._supply_collect_finish_reason = True, "attempt_limit"
            current._search_target_lock = (5, 200., 200., 400.)
            self.trigger(current)
            self.command(current)
            self.assertEqual(current.mode, 3)
            self.assertEqual((current.target_label, current._post07_started, current.zone_label), (label, post07, "redsafety"))
            self.assertEqual((current._supply_collect_attempts, current._supply_collect_started_ms), (2, 500))
            self.assertEqual((current._supply_collect_finished, current._supply_collect_finish_reason), (True, "attempt_limit"))
            self.assertEqual(current._near_view_green_missing_cycles, 1)
            self.assertTrue(current._first_green_near_view_decided)
            for name in ("_search_target_lock", "_track_target_lock", "_final_target_lock", "_latest_packet"):
                self.assertIsNone(getattr(current, name))

    def test_stale_frames_and_prepared_packets_cannot_cross_request_or_ack(self):
        current = self.make()
        old = current.frame_generation()
        self.submit(current, [self.start()], generation=old)
        pending = current.frame_generation()
        self.write_next(current)
        rotating = current.frame_generation()
        self.command(current)
        fresh = current.frame_generation()
        for stale in (old, pending, rotating):
            self.assertIsNone(self.submit(current, [self.start()], [block(100, 180)], 1200, stale))
            current._write_packet(s.build_coordinate_packet(5, 12, 34), stale)
            current._write_packet(s.build_event_packet(0x34), stale)
            current._write_packet(s.build_event_packet(0xAE), stale)
        self.assertEqual(current.frame_generation(), fresh)
        self.assertEqual(current._serial.write.call_count, 1)

    def test_after_ack_requires_three_new_frames_and_reticle_reference(self):
        current = self.make(0x24)
        self.trigger(current)
        self.command(current)
        target = block(150, 150)
        self.assertIsNone(current._next_outgoing_packet(1150))
        for now in (1200, 1250):
            self.assertIsNone(self.submit(current, [], [target], now))
        self.assertIs(self.submit(current, [], [target], 1300), target)
        self.assertEqual(current._latest_packet, s.build_coordinate_packet(5, -163, -189))

    def test_persistent_zone_does_not_loop_until_fresh_exit_rearms(self):
        current = self.make()
        self.trigger(current)
        self.command(current)
        for now in (1200, 1250, 1300):
            self.submit(current, [self.start()], now=now)
            self.assertIsNone(current._startzone_escape_phase)
        self.submit(current, [], now=1400)
        self.assertTrue(current._startzone_escape_armed)
        self.trigger(current, 1500)

    def test_unsolicited_or_duplicate_ack_never_resets_fresh_tracking(self):
        current = self.make()
        lock = (5, 100., 180., 400.)
        current._search_target_lock = lock
        generation = current.frame_generation()
        self.command(current)
        self.assertEqual(current._search_target_lock, lock)
        self.assertEqual(current.frame_generation(), generation)

    def test_fe_cancels_both_pending_and_sent_escape_and_rearms_new_task(self):
        for sent in (False, True):
            current = self.make()
            self.submit(current, [self.start()])
            generation = current.frame_generation()
            if sent:
                self.write_next(current)
            self.command(current, 0xFE)
            self.assertEqual(current.mode, s.MODE_WAIT_TASK_START)
            self.assertIsNone(current._startzone_escape_phase)
            self.assertTrue(current._startzone_escape_armed)
            self.command(current)
            current._write_packet(s.build_event_packet(0xAE), generation)
            self.assertEqual(current.mode, s.MODE_WAIT_TASK_START)
            self.assertEqual(current._serial.write.call_count, int(sent))

    def test_51_deadline_deferred_during_turn_but_not_extended_or_reset(self):
        current = self.make(0x24, s.TARGET_BLACK_OR_GREEN, True)
        current._supply_collect_started_ms, current._supply_collect_attempts = 100, 1
        self.trigger(current)
        self.assertIsNone(self.write_next(current, 16000))
        self.assertFalse(current._supply_collect_finished)
        self.command(current, now=17000)
        self.assertEqual((current._supply_collect_started_ms, current._supply_collect_attempts), (100, 1))
        self.assertEqual(self.write_next(current, 17001), s.build_event_packet(5))
        self.assertTrue(current._supply_collect_finished)
        self.assertEqual(current._supply_collect_finish_reason, "total_timeout")
        # Existing final-inventory handling clears the active timer/counter,
        # while its delivery-level exhausted latch prevents a fresh budget.
        self.assertEqual((current._supply_collect_started_ms, current._supply_collect_attempts), (None, 0))


class StartzoneEscapeDisabledTests(unittest.TestCase):
    """Real default-OFF behavior, without bypassing normal acquisition/loss."""

    make = StartzoneEscapeTests.make
    start = staticmethod(StartzoneEscapeTests.start)
    submit = StartzoneEscapeTests.submit
    write_next = StartzoneEscapeTests.write_next
    command = StartzoneEscapeTests.command

    def setUp(self):
        for helper in (patch("builtins.print"), patch.object(config, "VISION_DIAGNOSTICS_ENABLED", False)):
            helper.start()
            self.addCleanup(helper.stop)

    def test_default_and_missing_flag_disable_trigger_before_any_state_mutation(self):
        self.assertFalse(config.UART_STARTZONE_ESCAPE_ENABLED)
        current = self.make()
        generation = current.frame_generation()
        current._search_target_lock = (5, 150., 150., 400.)
        with patch.object(current, "_startzone_area_ratio", side_effect=AssertionError("disabled AE must not scan")):
            self.submit(current, [self.start(w=576, h=448)], [block(150, 150)])
        self.assertEqual(current.frame_generation(), generation)
        self.assertIsNone(current._startzone_escape_phase)
        current._begin_startzone_escape(1.0)
        self.assertEqual(current.mode, s.CMD_SEARCH)
        self.assertIsNotNone(current._search_target_lock)
        missing = SimpleNamespace(**{name: value for name, value in vars(config).items()
                                     if name != "UART_STARTZONE_ESCAPE_ENABLED"})
        with patch.object(s, "config", missing), \
                patch.object(current, "_startzone_area_ratio", side_effect=AssertionError("missing flag defaults off")):
            self.submit(current, [self.start(w=576, h=448)], [block(150, 150)], 1050)
            current._begin_startzone_escape(1.0)
            current.send_event(0xAE)
            current._write_packet(s.build_event_packet(0xAE))
        self.assertEqual(current.mode, s.CMD_SEARCH)
        self.assertEqual(current.frame_generation(), generation)
        self.assertIsNone(current._startzone_escape_phase)
        self.assertNotIn(s.build_event_packet(0xAE), current._pending_events)
        current._serial.write.assert_not_called()

    def test_full_startzone_does_not_interrupt_any_tracking_phase_or_cargo_task(self):
        tasks = (("sqareblue", 3), ("sqarered", 4), ("sqareredgreen", 5),
                 ("triangualrblack", 6), (s.TARGET_BLACK_OR_GREEN, 5))
        for mode in (s.CMD_SEARCH, s.CMD_TRACK, s.MODE_WAIT_04_HANDSHAKE,
                     s.MODE_WAIT_14_HANDSHAKE, s.MODE_FINAL_ROI_TRACK):
            for label, class_id in tasks:
                with self.subTest(mode=mode, label=label):
                    current = self.make(mode, label, label != "sqareredgreen")
                    target = block(150, 150, class_id)
                    for now in (1000, 1050, 1100):
                        selected = self.submit(current, [self.start(w=576, h=448)], [target], now)
                    self.assertIs(selected, target)
                    self.assertEqual(current.mode, mode)
                    self.assertIsNone(current._startzone_escape_phase)
                    self.assertIsNone(current._startzone_deferred_zone_command)
                    self.assertEqual(current._latest_packet[2], class_id)
                    packet = self.write_next(current, 1101)
                    self.assertEqual(len(packet), 11)
                    self.assertEqual(packet[2], class_id)

    def test_startzone_does_not_suppress_normal_e3_e4_or_ee_loss_handshakes(self):
        for mode, event, frames in ((s.CMD_SEARCH, 0xE3, config.UART_SEARCH_NO_TARGET_FRAMES),
                                     (s.CMD_TRACK, 0xE4, config.UART_TRACK_LOST_CONFIRM_FRAMES),
                                     (s.MODE_FINAL_ROI_TRACK, 0xEE, config.UART_FINAL_NO_TARGET_FRAMES)):
            with self.subTest(mode=mode):
                current = self.make(mode)
                for index in range(frames):
                    now = 1000 + index * 50
                    self.submit(current, [self.start(w=576, h=448)], now=now)
                    self.write_next(current, now)
                packets = [call.args[0] for call in current._serial.write.call_args_list]
                self.assertIn(s.build_event_packet(event), packets)
                self.assertNotIn(s.build_event_packet(0xAE), packets)
                self.assertIsNone(current._startzone_escape_phase)

    def test_manual_queued_and_prepared_ae_packets_are_blocked_at_actual_wire(self):
        current = self.make()
        current.send_event(0xAE)
        self.assertEqual(current._pending_events, [])
        current._pending_events = [s.build_event_packet(0xAE), s.build_event_packet(0x02)]
        self.assertEqual(current._next_outgoing_packet(1000), s.build_event_packet(0x02))
        self.assertEqual(current._pending_events, [])
        for phase in (None, "pending", "wait_ack"):
            current._startzone_escape_phase = phase
            current._write_packet(s.build_event_packet(0xAE))
            current._write_packet(s.build_event_packet(0xAE), current.frame_generation())
            if phase is not None:
                self.assertIsNone(current._next_outgoing_packet(2000))
        current._serial.write.assert_not_called()

    def test_late_ae_is_ignored_and15_25_enter05_without_deferral(self):
        for command in (0x15, 0x25):
            with self.subTest(command=command):
                current = self.make()
                lock = (5, 150., 150., 400.)
                current._search_target_lock = lock
                generation = current.frame_generation()
                self.command(current, 0xAE)
                self.assertEqual(current.frame_generation(), generation)
                self.assertEqual(current._search_target_lock, lock)
                self.command(current, command, 1200)
                self.assertEqual(current.mode, s.MODE_NEAR_VIEW_CHECK)
                self.assertEqual(current.zone_label, s.ZONE_SEARCH_COMMANDS[command])
                self.assertIsNone(current._startzone_deferred_zone_command)
                self.assertEqual(self.write_next(current, 1300), s.build_event_packet(0x05))
                self.assertEqual(current._near_view_started_ms, 1300)

    def test_supply_deadline_remains_active_even_when_startzone_is_visible(self):
        current = self.make(s.MODE_FINAL_ROI_TRACK, s.TARGET_BLACK_OR_GREEN, True)
        current._supply_collect_started_ms, current._supply_collect_attempts = 100, 1
        self.submit(current, [self.start(w=576, h=448)], now=16000)
        self.assertIsNone(current._startzone_escape_phase)
        self.assertTrue(current._supply_collect_finished)
        self.assertEqual(current._supply_collect_finish_reason, "total_timeout")
        self.assertEqual(current._zone_load_recheck_source, "before06")
        self.assertEqual(self.write_next(current, 16000), s.build_event_packet(0x05))


if __name__ == "__main__":
    unittest.main()
