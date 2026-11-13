"""Post07 command51 supplemental collection, without camera/UART hardware."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from test_vision_rules import LABELS, block, near_view_block, controller, confirm_final_load_check, SingleFrameTrackingScenario
import app_config as config
import serial_controller as s
from vision_overlay import make_bottom_center_roi


DEFAULT_SUPPLY_SEARCH_SETTLE_MS = config.UART_SUPPLY_SEARCH_SETTLE_MS


class SupplyCollectionTests(SingleFrameTrackingScenario):
    def setUp(self):
        for setting in (patch.object(config, "VISION_DIAGNOSTICS_ENABLED", False),
                        patch.object(config, "UART_SUPPLY_COLLECT_MAX_ATTEMPTS", 2),
                        patch.object(config, "UART_SUPPLY_COLLECT_LOADED_ROI", None),
                        patch.object(config, "UART_SUPPLY_SEARCH_SETTLE_MS", 0),
                        patch("builtins.print")):
            setting.start()
            self.addCleanup(setting.stop)

    def command_at(self, current, command, now):
        with patch.object(s, "_now_ms", return_value=now):
            current.process_received_bytes(s.build_command_packet(command))

    def submit_at(self, current, now, objects=(), raw=None, generation=None):
        objects = list(objects)
        with patch.object(s, "_now_ms", return_value=now):
            return current.submit(objects, LABELS,
                                  raw_objects=list(raw) if raw is not None else objects,
                                  frame_generation=generation)

    def check_near(self, current, ids, start):
        self.command_at(current, s.CMD_SEARCH_BLUE_ZONE, start)
        self.write_next(current, start, s.build_event_packet(s.EVENT_TARGET_READY))
        objects = [near_view_block(120 + 40 * index, 200, class_id)
                   for index, class_id in enumerate(ids)]
        for index in range(3):
            self.submit_at(current, start + 500 + index * 10, objects)
        return start + 520

    def start_collection(self, class_id=5):
        current = controller(s.MODE_FINAL_ROI_TRACK)
        self.command_at(current, s.CMD_SELECT_BLACK_OR_GREEN, 1000)
        requested_at = self.check_near(current, [class_id], 1000)
        self.assertEqual(current.mode, s.MODE_WAIT_SUPPLY_SEARCH)
        self.assertEqual(current._next_outgoing_packet(requested_at),
                         s.build_event_packet(s.EVENT_COLLECT_MORE_SUPPLIES))
        return current, requested_at

    def acknowledge_search(self, current, now):
        self.command_at(current, s.CMD_SELECT_BLACK_OR_GREEN, now)
        self.command_at(current, s.CMD_SEARCH, now + 1)
        # One RX51/RX03 handshake enters search without an inventory detour.
        self.assertEqual(current.mode, s.CMD_SEARCH)

    def write_next(self, current, now, expected):
        packet = current._next_outgoing_packet(now)
        self.assertEqual(packet, expected)
        serial = SimpleNamespace(write=Mock(side_effect=lambda data: len(data)))
        with patch.object(current, "_serial", serial), patch.object(s, "_now_ms", return_value=now):
            current._write_packet(packet, generation=current._outgoing_generation)
        serial.write.assert_called_once_with(packet)

    @patch.object(config, "UART_TRACKING_ROI_SHAPE", "triangle")
    @patch.object(config, "UART_DIRECT_CAPTURE_AFTER_14", True)
    @patch.object(config, "UART_SUPPLY_SEARCH_SETTLE_MS", DEFAULT_SUPPLY_SEARCH_SETTLE_MS)
    def test_first_supply_rx34_skips05_then_second_capture_checks_load(self):
        current = controller(s.MODE_FINAL_ROI_TRACK)
        current.tracking_roi = make_bottom_center_roi(576, 448)
        self.command_at(current, 0x31, 1000)
        first = block(313, 370, 6)
        self.submit_at(current, 1100, [first])
        self.write_next(current, 1100, s.build_event_packet(0x34))
        old_generation = current.frame_generation()

        self.command_at(current, 0x34, 1400)
        self.assertEqual(current.mode, s.MODE_WAIT_SUPPLY_SEARCH)
        self.assertEqual(current.target_label, s.TARGET_BLACK_OR_GREEN)
        self.assertGreater(current.frame_generation(), old_generation)
        self.assertIsNone(current._latest_packet)
        self.assertEqual(current._pending_events, [])
        self.command_at(current, 0x34, 1450)
        self.assertEqual(current._supply_collect_attempts, 1)
        self.assertEqual(current._supply_collect_started_ms, 1400)

        self.acknowledge_search(current, 1600)
        carried, new = block(313, 430, 6), block(313, 340, 5)
        self.assertIsNone(self.submit_at(current, 1850, [carried, new]))
        self.assertIsNone(current._next_outgoing_packet(1850))
        self.assertIs(self.submit_at(current, 1900, [carried, new]), new)
        self.write_next(current, 1900, s.build_event_packet(0x04))
        self.command_at(current, 0x04, 1950)
        self.submit_at(current, 2000, [carried, new])
        self.write_next(current, 2000, s.build_event_packet(0x14))
        self.command_at(current, 0x14, 2050)
        self.write_next(current, 2050, s.build_event_packet(0x24))
        self.submit_at(current, 2350, [carried, block(313, 350, 5)])
        self.write_next(current, 2350, s.build_event_packet(0x34))
        self.command_at(current, 0x34, 2400)
        self.assertEqual(current.mode, s.MODE_FINAL_ROI_TRACK)
        self.assertEqual(current._supply_collect_attempts, 1)

        decision = self.check_near(current, [5, 6], 3000)
        self.assertEqual(current._zone_load_recheck_source, "before06")
        self.assert_delivery(current, decision)

    def test_first_supply_rx34_requires_a_written34_and_a_post07_supply_task(self):
        for label, post07, class_id, sent in (
            (LABELS[5], False, 5, True), (LABELS[4], True, 4, True),
            (LABELS[6], True, 6, False),
        ):
            with self.subTest(label=label, sent=sent):
                current = controller(s.MODE_FINAL_ROI_TRACK)
                current.target_label, current._post07_started = label, post07
                observed_at = (1060, 1080, 1100) if class_id == 4 else (1100,)
                for now in observed_at:
                    self.submit_at(current, now, [block(288, 410, class_id)])
                if sent:
                    self.write_next(current, 1100, s.build_event_packet(0x34))
                self.command_at(current, 0x34, 1200)
                self.assertEqual(current.mode, s.MODE_FINAL_ROI_TRACK)
                self.assertIsNone(current._supply_collect_started_ms)
                self.assertFalse(current._supply_first_capture_bypassed)

    @patch.object(config, "UART_SUPPLY_SEARCH_SETTLE_MS", DEFAULT_SUPPLY_SEARCH_SETTLE_MS)
    def test_supplement_rx51_03_waits250ms_for_a_fresh_frame_without_extra_handshake(self):
        self.assertEqual(config.UART_SUPPLY_SEARCH_SETTLE_MS, 250)
        for class_id in (5, 6):
            with self.subTest(class_id=class_id):
                current, first_request = self.start_collection(class_id)
                self.acknowledge_search(current, 1600)
                self.assertEqual(current._supply_collect_attempts, 1)
                self.assertEqual(current._supply_collect_started_ms, first_request)
                self.assertIsNone(current._zone_load_recheck)
                self.assertEqual(current._pending_events, [])
                new, cargo = block(220, 180, class_id), block(288, 430)
                self.acknowledge_search(current, 1700)  # Duplicate does not extend settle.
                settling_generation = current.frame_generation()
                self.assertIsNone(self.submit_at(current, 1850, [cargo, new], generation=settling_generation))
                self.assertIsNone(current._next_outgoing_packet(1850))
                self.assertEqual(current._supply_outside_history, [])
                with patch.object(s, "_now_ms", return_value=1851):
                    generation, _ = current.frame_context()
                self.assertIsNone(self.submit_at(current, 1851, [cargo, new], generation=settling_generation))
                self.assertIs(self.submit_at(current, 1852, [cargo, new], generation=generation), new)
                self.assertEqual(current._next_outgoing_packet(1852),
                                 s.build_coordinate_packet(class_id, -68, -44))
                self.assertEqual(current.mode, s.CMD_SEARCH)

    def test_direct_supplement_does_not_inventory_loaded_supplies_before_search(self):
        for ids in ([5], [6], [5, 6], [6, 6, 5], [5, 5, 6, 6]):
            with self.subTest(ids=ids):
                current, first_request = self.start_collection()
                self.acknowledge_search(current, 1600)
                cargo = [block(80 + index * 80, 430, class_id)
                         for index, class_id in enumerate(ids)]
                for now in (1602, 1700, 2200):
                    self.assertIsNone(self.submit_at(current, now, cargo))
                self.assertEqual(current.mode, s.CMD_SEARCH)
                self.assertIsNone(current._latest_packet)
                self.assertIsNone(current._zone_load_recheck)
                self.assertEqual(current._pending_events, [])
                self.assertEqual(current._supply_collect_started_ms, first_request)

    def test_second_supplement_enters03_directly_and_preserves_total_budget(self):
        current, first_request = self.start_collection()
        self.acknowledge_search(current, 1600)
        requested = self.check_near(current, [6], 3000)
        self.write_next(current, requested, s.build_event_packet(0x03))
        self.acknowledge_search(current, 3600)
        self.assertEqual(current._supply_collect_attempts, 2)
        self.assertEqual(current._supply_collect_started_ms, first_request)
        self.assertFalse(current._supply_collect_delivery_pending)
        self.assertEqual(current._pending_events, [])
        new = block(220, 180, 6)
        self.assertIs(self.submit_at(current, 3602, [block(288, 430), new]), new)
        self.assertEqual(current._latest_packet[2], 6)

    def test_ordinary05_after_direct_capture_still_arranges_foreign_or_excess_supplies(self):
        for ids in ([5, 4], [5, 3], [5, 5, 6, 6]):
            with self.subTest(ids=ids):
                current, first_request = self.start_collection()
                self.acknowledge_search(current, 1600)
                decision = self.check_near(current, ids, 3000)
                self.write_next(current, decision, s.build_event_packet(0x02))
                self.assertEqual(current.mode, s.MODE_ARRANGE_RIGHTMOST)
                self.command_at(current, 0x12, decision + 1000)
                self.assertEqual(current.mode, s.MODE_ARRANGE_LEFTMOST)
                self.write_next(current, decision + 1000, s.build_event_packet(0x12))
                self.command_at(current, 0x22, decision + 2000)
                self.assertEqual(current.mode, s.MODE_POST_ARRANGE_CHECK)
                self.assertEqual(current._supply_collect_started_ms, first_request)

    def test_ordinary05_after_direct_capture_empty_or_blue_only_still_returns24(self):
        for ids in ([], [3], [3, 3], [3, 4]):
            with self.subTest(ids=ids):
                current, first_request = self.start_collection()
                self.acknowledge_search(current, 1600)
                decision = self.check_near(current, ids, 3000)
                self.write_next(current, decision, s.build_event_packet(0x24))
                self.assertEqual(current.mode, s.MODE_FINAL_ROI_TRACK)
                self.assertIsNone(current._supply_collect_started_ms)
                self.assertIsNone(current._next_outgoing_packet(first_request + config.UART_SUPPLY_COLLECT_TIMEOUT_MS))

    def test_ordinary05_after_direct_capture_enough_supplies_still_requires_final05(self):
        for ids in ([5, 5], [6, 6], [5, 6], [5, 6, 6]):
            with self.subTest(ids=ids):
                current, _ = self.start_collection()
                self.acknowledge_search(current, 1600)
                decision = self.check_near(current, ids, 3000)
                self.assertEqual(current._zone_load_recheck_source, "before06")
                self.assertEqual(current._pending_events, [s.build_event_packet(0x05)])
                self.assert_delivery(current, decision)

    def test_waiting_supplement03_still_pauses_until_mcu_search_ack(self):
        current, first_request = self.start_collection()
        self.command_at(current, 0x51, 1600)
        for now in (1601, 1700, 2000):
            self.assertIsNone(self.submit_at(current, now, [block(220, 180)]))
            self.assertEqual(current.mode, s.MODE_WAIT_SUPPLY_SEARCH)
            self.assertIsNone(current._next_outgoing_packet(now))
        self.command_at(current, 0x03, 2001)
        self.assertEqual(current.mode, s.CMD_SEARCH)
        self.assertEqual(current._supply_collect_started_ms, first_request)
        self.assertEqual(current._pending_events, [])

    def test_supplement03_alone_for_already_selected51_also_searches_directly(self):
        current, _ = self.start_collection()
        self.command_at(current, 0x03, 1600)
        self.assertEqual(current.mode, s.CMD_SEARCH)
        self.assertEqual(current._pending_events, [])
        new = block(220, 180)
        self.assertIs(self.submit_at(current, 1601, [new]), new)

    def test_direct_search_ack_invalidates_old_frames_and_prepared_coordinates(self):
        current, _ = self.start_collection()
        generation = current.frame_generation()
        self.acknowledge_search(current, 1600)
        packet = s.build_coordinate_packet(6, 7, 8)
        serial = SimpleNamespace(write=Mock(return_value=len(packet)))
        with patch.object(current, "_serial", serial), patch.object(s, "_now_ms", return_value=1602):
            current._write_packet(packet, generation=generation)
        serial.write.assert_not_called()
        self.assertIsNone(self.submit_at(current, 1602, [block(220, 180)], generation=generation))
        self.assertIsNone(current._latest_packet)
        self.assertEqual(current._pending_events, [])
        new = block(220, 180)
        self.assertIs(self.submit_at(current, 1603, [new]), new)
        self.assertEqual(current._latest_packet[2], 5)

    def test_direct_supplement_keeps12s_timeout_through_search_track_and_recovery(self):
        self.assertEqual(config.UART_SUPPLY_COLLECT_TIMEOUT_MS, 12000)
        for mode in (s.CMD_SEARCH, s.CMD_TRACK, s.MODE_WAIT_04_HANDSHAKE,
                     s.MODE_WAIT_14_HANDSHAKE, s.MODE_FINAL_ROI_TRACK):
            with self.subTest(mode=mode):
                current, first_request = self.start_collection()
                self.acknowledge_search(current, 1600)
                current.mode = mode
                self.assert_delivery(current, first_request + config.UART_SUPPLY_COLLECT_TIMEOUT_MS)

    def test_final24_restart_keeps_supplement_budget_and_final05_priority(self):
        current, first_request = self.start_collection()
        self.acknowledge_search(current, 1600)
        current.mode = s.MODE_FINAL_ROI_TRACK
        self.command_at(current, 0x03, first_request + 6000)
        self.assertEqual(current.mode, s.CMD_SEARCH)
        self.assertEqual(current.target_label, s.TARGET_BLACK_OR_GREEN)
        self.assertEqual(current._supply_collect_started_ms, first_request)
        self.assertEqual(current._supply_collect_attempts, 1)
        self.assertIsNone(current._next_outgoing_packet(first_request + config.UART_SUPPLY_COLLECT_TIMEOUT_MS - 1))

        deadline = first_request + config.UART_SUPPLY_COLLECT_TIMEOUT_MS
        self.command_at(current, 0x05, deadline)
        self.command_at(current, 0x03, deadline)
        self.assertEqual(current.mode, s.MODE_NEAR_VIEW_CHECK)
        self.assertEqual(current._next_outgoing_packet(deadline), s.build_event_packet(0x05))

    def test_supplement03_missing_raw_targets_keeps_e3_not_an_inventory05(self):
        current, _ = self.start_collection()
        self.acknowledge_search(current, 1600)
        for index in range(config.UART_SEARCH_NO_TARGET_FRAMES):
            self.submit_at(current, 1610 + index * 10, [])
        self.assertEqual(current._next_outgoing_packet(1800), s.build_event_packet(0xE3))
        self.assertEqual(current.mode, s.CMD_SEARCH)
        self.assertIsNone(current._zone_load_recheck)
        self.assertNotIn(s.build_event_packet(0x05), current._pending_events)

    def test_mcu13_recovery_in_direct_supplement_never_adds_precheck05(self):
        current, first_request = self.start_collection()
        self.acknowledge_search(current, 1600)
        for index in range(config.UART_SEARCH_NO_TARGET_FRAMES):
            self.submit_at(current, 1610 + index * 10, [])
        self.assertEqual(current._next_outgoing_packet(1800), s.build_event_packet(0xE3))
        self.command_at(current, 0x13, 1900)
        new = block(220, 180)
        self.assertIs(self.submit_at(current, 1910, [new]), new)
        self.assertEqual(current._latest_packet[2], 5)
        self.assertEqual(current._supply_collect_started_ms, first_request)
        self.assertNotIn(s.build_event_packet(0x05), current._pending_events)

    def test_explicit_other_task_cancels_direct_supplement_without_extra05(self):
        for code in (0x11, 0x21, 0x31, 0x01):
            with self.subTest(code=code):
                current, first_request = self.start_collection()
                self.acknowledge_search(current, 1600)
                self.command_at(current, code, 1700)
                self.command_at(current, 0x03, 1701)
                self.assertIsNone(current._supply_collect_started_ms)
                self.assertNotIn(s.build_event_packet(0x05), current._pending_events)
                self.assertIsNone(current._next_outgoing_packet(first_request + config.UART_SUPPLY_COLLECT_TIMEOUT_MS))

    def test_initial51_03_and_other_normal_searches_still_do_not_precheck(self):
        for code in (0x51, 0x21, 0x31, 0x11):
            with self.subTest(code=code):
                current = controller(s.CMD_SEARCH)
                self.command_at(current, code, 1000)
                self.command_at(current, 0x03, 1001)
                self.assertNotEqual(current.mode, s.MODE_NEAR_VIEW_CHECK)
                self.assertIsNone(current._zone_load_recheck)
                self.assertEqual(current._pending_events, [])

    def assert_delivery(self, current, now):
        first_packet = current._next_outgoing_packet(now)
        self.assertEqual(first_packet, s.build_event_packet(0x05))
        # Requeue the inspected request; the helper verifies its real UART
        # write,500ms settling and three fresh final inventory observations.
        current._pending_events.insert(0, first_packet)
        now = confirm_final_load_check(current, [5], now)
        first_packet = current._next_outgoing_packet(now)
        self.assertEqual(current.mode, s.MODE_SEARCH_ZONE)
        self.assertEqual(first_packet, s.build_coordinate_packet(5, 0, 0))
        serial = SimpleNamespace(write=Mock(side_effect=lambda packet: len(packet)))
        with patch.object(current, "_serial", serial), patch.object(s, "_now_ms", return_value=now):
            current._write_packet(first_packet, current._outgoing_generation)
        self.assertIsNone(current._next_outgoing_packet(now + 49))
        self.write_next(current, now + 50, s.build_event_packet(s.EVENT_ZONE_SEARCH_STARTED))
        current._next_outgoing_packet(now + 50 + config.UART_WIDE_VIEW_SETTLE_MS)

    def test_mcu05_accepts_all_supply_capture_and_recovery_stages(self):
        for mode in (s.CMD_SEARCH, s.CMD_TRACK, s.MODE_WAIT_04_HANDSHAKE,
                     s.MODE_WAIT_14_HANDSHAKE, s.MODE_FINAL_ROI_TRACK,
                     s.MODE_WAIT_SUPPLY_SEARCH, s.MODE_NEAR_VIEW_CHECK,
                     s.MODE_ARRANGE_RIGHTMOST, s.MODE_ARRANGE_LEFTMOST,
                     s.MODE_POST_ARRANGE_CHECK):
            with self.subTest(mode=mode):
                current, _ = self.start_collection()
                current.mode = mode
                current.zone_label = "redsafety"
                identity = current._final_target_identity(block(200, 200, 6))
                current._search_target_lock = identity
                current._track_target_lock = identity
                current._final_target_lock = identity
                current._latest_packet = s.build_coordinate_packet(6, 10, -20)
                current._pending_events = [s.build_event_packet(code) for code in (0xE3, 0xE4, 0xEE, 0x34)]
                current._final_search_waiting_ack = True
                current._track_green_recovery_waiting = True
                generation = current.frame_generation()
                self.command_at(current, 0x05, 1800)
                self.assertEqual(current.mode, s.MODE_NEAR_VIEW_CHECK)
                self.assertEqual(current._zone_load_recheck_source, "before06")
                self.assertEqual(current._zone_load_recheck, "wait05tx")
                self.assertEqual((current.target_label, current.zone_label), (s.TARGET_BLACK_OR_GREEN, "redsafety"))
                self.assertEqual(current._pending_events, [s.build_event_packet(0x05)])
                self.assertIsNone(current._latest_packet)
                self.assertIsNone(current._search_target_lock)
                self.assertIsNone(current._track_target_lock)
                self.assertIsNone(current._final_target_lock)
                self.assertIsNone(current._supply_collect_started_ms)
                self.assertEqual(current._supply_collect_attempts, 0)
                self.assertFalse(current._final_search_waiting_ack)
                self.assertFalse(current._track_green_recovery_waiting)
                self.assertGreater(current.frame_generation(), generation)

    def test_mcu05_after_empty05_still_checks_when_visual_budget_was_cancelled(self):
        current, _ = self.start_collection()
        self.acknowledge_search(current, 1600)
        self.command_at(current, 0x24, 2500)
        emptied_at = self.check_near(current, [], 3000)
        self.assertEqual(current.mode, s.MODE_FINAL_ROI_TRACK)
        self.assertIsNone(current._supply_collect_started_ms)
        self.command_at(current, 0x05, emptied_at + 10)
        self.assertEqual(current._zone_load_recheck_source, "before06")
        self.assert_delivery(current, emptied_at + 60)

    def test_mcu05_final_inventory_accepts_one_to_three_supplies_without_extra05_or03(self):
        for ids in ([5], [6], [5, 6], [6, 6, 5]):
            with self.subTest(ids=ids):
                current, _ = self.start_collection()
                current.mode = s.MODE_FINAL_ROI_TRACK
                self.command_at(current, 0x05, 1800)
                now = confirm_final_load_check(current, ids, 1800)
                self.assertEqual(current._zone_load_recheck, "wait06tx")
                self.assertEqual(current._pending_events, [s.build_coordinate_packet(5, 0, 0),
                                                           s.build_event_packet(0x06)])
                self.assertEqual(current._supply_collect_attempts, 0)
                self.write_next(current, now, s.build_coordinate_packet(5, 0, 0))
                self.assertIsNone(current._next_outgoing_packet(now + 49))
                self.write_next(current, now + 50, s.build_event_packet(0x06))
                self.assertEqual(current._zone_load_recheck, "wide_settle")

    def test_mcu05_final_inventory_returns02_or24_for_unacceptable_load(self):
        for ids, event in (([], 0x24), ([3], 0x24), ([4], 0x24), ([3, 3], 0x24),
                           ([5, 4], 0x02), ([6, 3], 0x02), ([5, 5, 6, 6], 0x02)):
            with self.subTest(ids=ids):
                current, _ = self.start_collection()
                current.mode = s.MODE_FINAL_ROI_TRACK
                self.command_at(current, 0x05, 1800)
                now = confirm_final_load_check(current, ids, 1800)
                self.assertEqual(current._pending_events, [s.build_event_packet(event)])
                self.write_next(current, now, s.build_event_packet(event))
                self.assertEqual(current.mode, s.MODE_FINAL_ROI_TRACK if event == 0x24
                                 else s.MODE_ARRANGE_RIGHTMOST)
                self.assertIsNone(current._supply_collect_started_ms)

    def test_mcu05_is_not_accepted_as_supply_timeout_for_other_tasks_or_red_switch(self):
        for label, post07, red_switch in (("sqareredgreen", False, None),
                                          ("sqarered", True, None),
                                          ("triangualrblack", True, None),
                                          ("sqareredgreen", True, None),
                                          (s.TARGET_BLACK_OR_GREEN, False, None),
                                          (s.TARGET_BLACK_OR_GREEN, True, "wait11")):
            for mode in (s.CMD_SEARCH, s.CMD_TRACK, s.MODE_FINAL_ROI_TRACK):
                with self.subTest(label=label, post07=post07, switch=red_switch, mode=mode):
                    current = controller(mode)
                    current.target_label, current._post07_started = label, post07
                    current._red_switch_stage = red_switch
                    generation = current.frame_generation()
                    self.command_at(current, 0x05, 1800)
                    self.assertEqual(current.mode, mode)
                    self.assertIsNone(current._zone_load_recheck)
                    self.assertEqual(current._pending_events, [])
                    self.assertEqual(current.frame_generation(), generation)

    def test_mcu05_duplicates_and_old_handshakes_do_not_restart_final_count(self):
        current, _ = self.start_collection()
        self.command_at(current, 0x05, 1800)
        self.command_at(current, 0x05, 1810)
        self.assertEqual(current._pending_events, [s.build_event_packet(0x05)])
        self.write_next(current, 1850, s.build_event_packet(0x05))
        self.submit_at(current, 2350, [block(200, 400)])
        self.assertEqual(current._near_view_stable_frames, 1)
        generation = current.frame_generation()
        self.command_at(current, 0x05, 2351)
        for command in (0x51, 0x03, 0x04, 0x14, 0x12, 0x22, 0x24, 0x13, 0xF1, 0x25):
            self.command_at(current, command, 2352)
        self.write_next(current, 2353, s.build_event_packet(0x05))
        self.assertEqual(current._near_view_started_ms, 1850)
        self.assertEqual(current._near_view_stable_frames, 1)
        self.assertEqual(current.frame_generation(), generation)
        self.submit_at(current, 2360, [block(200, 400)])
        self.submit_at(current, 2370, [block(200, 400)])
        self.assertEqual(current._pending_events, [s.build_coordinate_packet(5, 0, 0),
                                                   s.build_event_packet(0x06)])

    def test_mcu05_and_visual_timeout_coalesce_into_one_final_inventory(self):
        for mcu_first in (True, False):
            with self.subTest(mcu_first=mcu_first):
                current, first_request = self.start_collection()
                deadline = first_request + config.UART_SUPPLY_COLLECT_TIMEOUT_MS
                if not mcu_first:
                    self.assertEqual(current._next_outgoing_packet(deadline), s.build_event_packet(0x05))
                self.command_at(current, 0x05, deadline)
                generation = current.frame_generation()
                self.assert_delivery(current, deadline + 1)
                self.assertGreater(current.frame_generation(), generation)
                self.assertEqual(current._pending_events, [])
                self.assertIsNone(current._supply_collect_started_ms)

    def test_mcu05_invalidates_inflight_frames_and_prepared_old_commands_coordinates(self):
        for packet in ([s.build_coordinate_packet(6, 10, -20)]
                       + [s.build_event_packet(code) for code in (0x03, 0x14, 0x24, 0x34, 0xE3, 0xE4, 0xEE)]):
            with self.subTest(packet=packet):
                current, _ = self.start_collection()
                current.mode = s.MODE_FINAL_ROI_TRACK
                generation = current.frame_generation()
                self.command_at(current, 0x05, 1800)
                current._serial = SimpleNamespace(write=Mock(return_value=len(packet)))
                with patch.object(s, "_now_ms", return_value=1801):
                    current._write_packet(packet, generation=generation)
                current._serial.write.assert_not_called()
                self.assertIsNone(self.submit_at(current, 1802, [block(200, 400)], generation=generation))
                self.assertEqual(current._near_view_stable_frames, 0)
                self.assertEqual(current._pending_events, [s.build_event_packet(0x05)])

    def test_mcu05_failed_reply_retries_before_settling_and_fresh_three_frame_count(self):
        current, _ = self.start_collection()
        self.command_at(current, 0x05, 1800)
        packet = current._next_outgoing_packet(1800)
        current._serial = SimpleNamespace(write=Mock(return_value=0))
        with patch.object(s, "_now_ms", return_value=1800):
            current._write_packet(packet, current._outgoing_generation)
        self.assertEqual(current._zone_load_recheck, "wait05tx")
        self.assertIsNone(current._near_view_started_ms)
        generation = current.frame_generation()
        for now in (1810, 1820, 1830):
            self.submit_at(current, now, [block(200, 400)])
        self.assertEqual(current._near_view_stable_frames, 0)
        self.write_next(current, 1900, s.build_event_packet(0x05))
        self.assertIsNone(self.submit_at(current, 2150, [block(200, 400)], generation=generation))
        self.submit_at(current, 2149, [block(200, 400)])
        self.assertEqual(current._near_view_stable_frames, 0)
        for now in (2150, 2160):
            self.submit_at(current, now, [block(200, 400)])
            self.assertEqual(current._zone_load_recheck, "count05")
        self.submit_at(current, 2170, [block(200, 400)])
        self.assertEqual(current._zone_load_recheck, "wait06tx")
        self.assertEqual(current._pending_events, [s.build_coordinate_packet(5, 0, 0),
                                                   s.build_event_packet(0x06)])

    def test_mcu05_final_inventory_allows_explicit_new_task_and_ignores_late_duplicate05(self):
        current, _ = self.start_collection()
        self.command_at(current, 0x05, 1800)
        self.command_at(current, 0x11, 1801)
        self.assertEqual(current.target_label, "sqarered")
        self.assertIsNone(current._zone_load_recheck)
        self.assertNotIn(s.build_event_packet(0x05), current._pending_events)
        current, _ = self.start_collection()
        self.command_at(current, 0x05, 1800)
        confirm_final_load_check(current, [5], 1800)
        generation = current.frame_generation()
        packets = list(current._pending_events)
        self.command_at(current, 0x05, 2350)
        self.assertEqual(current._zone_load_recheck, "wait06tx")
        self.assertEqual(current.frame_generation(), generation)
        self.assertEqual(current._pending_events, packets)

    def test_single_green_or_black_requests03_once_and_pauses_coordinates(self):
        for class_id in (5, 6):
            with self.subTest(class_id=class_id):
                current, requested_at = self.start_collection(class_id)
                self.assertEqual(current._supply_collect_attempts, 1)
                self.assertEqual(current._supply_collect_started_ms, requested_at)
                for now in (requested_at + 20, requested_at + 500, requested_at + 1000):
                    self.assertIsNone(self.submit_at(current, now, [block(288, 224, class_id)]))
                    self.assertIsNone(current._next_outgoing_packet(now))
                    self.assertIsNone(current._latest_packet)

    def test_settling_and_single_vs_multiple_stable_buckets(self):
        current = controller(s.MODE_FINAL_ROI_TRACK)
        self.command_at(current, s.CMD_SELECT_BLACK_OR_GREEN, 1000)
        self.command_at(current, s.CMD_SEARCH_BLUE_ZONE, 1000)
        self.write_next(current, 1000, s.build_event_packet(0x05))
        for now in (1001, 1200, 1249):
            self.submit_at(current, now, [block(200, 200)])
        self.assertEqual(current._pending_events, [])
        for index in range(8):
            objects = [block(200, 200)]
            if index % 2:
                objects.append(near_view_block(250, 200, 6))
            self.submit_at(current, 1500 + index * 10, objects)
        self.assertEqual(current.mode, s.MODE_NEAR_VIEW_CHECK)
        self.assertEqual(current._pending_events, [])
        for index in range(3):
            self.submit_at(current, 1700 + index * 10, [block(200, 200)])
        self.assertEqual(current.mode, s.MODE_WAIT_SUPPLY_SEARCH)

    def test_two_additional_attempts_keep_one_budget_third_single_delivers(self):
        current, first_request = self.start_collection()
        self.acknowledge_search(current, 1600)
        self.assertEqual(current._supply_collect_started_ms, first_request)
        second_request = self.check_near(current, [6], 3000)
        self.assertEqual(current._supply_collect_attempts, 2)
        self.assertEqual(current._next_outgoing_packet(second_request), s.build_event_packet(0x03))
        self.acknowledge_search(current, 3600)
        self.assertEqual(current._supply_collect_started_ms, first_request)
        finished = self.check_near(current, [5], 5000)
        self.assert_delivery(current, finished)
        self.assertIsNone(current._supply_collect_started_ms)
        self.assertEqual(current._supply_collect_attempts, 0)

    def test_two_or_three_supplies_finish_immediately_and_report_id5(self):
        for ids in ([5, 5], [6, 6], [5, 6], [5, 6, 6]):
            with self.subTest(ids=ids):
                current, _ = self.start_collection()
                self.acknowledge_search(current, 1600)
                finished = self.check_near(current, ids, 2000)
                self.assert_delivery(current, finished)

    def test_foreign_colors_or_over_three_keep_existing02_12_flow(self):
        for ids in ([5, 4], [5, 3], [5, 5, 6, 6]):
            with self.subTest(ids=ids):
                current, first_request = self.start_collection()
                self.acknowledge_search(current, 1600)
                decision = self.check_near(current, ids, 2000)
                self.assertEqual(current.mode, s.MODE_ARRANGE_RIGHTMOST)
                self.assertEqual(current._next_outgoing_packet(decision), s.build_event_packet(0x02))
                self.command_at(current, s.CMD_ARRANGE_FIRST_DONE, decision + 1000)
                self.assertEqual(current.mode, s.MODE_ARRANGE_LEFTMOST)
                self.assertEqual(current._next_outgoing_packet(decision + 1000), s.build_event_packet(0x12))
                self.command_at(current, s.CMD_ARRANGE_LEFT_DONE, decision + 2000)
                self.assertEqual(current.mode, s.MODE_POST_ARRANGE_CHECK)
                self.assertEqual(current._supply_collect_started_ms, first_request)
                self.assertEqual(current._supply_collect_attempts, 1)

    def test_confirmed_empty_load_cancels_collection_and_returns24(self):
        current, first_request = self.start_collection()
        self.acknowledge_search(current, 1600)
        finished = self.check_near(current, [], 2000)
        self.assertEqual(current.mode, s.MODE_FINAL_ROI_TRACK)
        self.assertEqual(current._next_outgoing_packet(finished), s.build_event_packet(0x24))
        self.assertIsNone(current._supply_collect_started_ms)
        self.assertEqual(current._supply_collect_attempts, 0)
        self.assertIsNone(current._next_outgoing_packet(first_request + config.UART_SUPPLY_COLLECT_TIMEOUT_MS))

    @patch.object(config, "UART_SUPPLY_SEARCH_SETTLE_MS", DEFAULT_SUPPLY_SEARCH_SETTLE_MS)
    def test_empty_supplement_waits250ms_after_return24_is_written(self):
        current, _ = self.start_collection()
        self.acknowledge_search(current, 1600)
        decision = self.check_near(current, [], 2000)
        target = block(220, 180, 6)
        self.assertIsNone(self.submit_at(current, decision, [target]))
        request = s.build_event_packet(0x24)
        current._serial = SimpleNamespace(write=Mock(return_value=2))
        self.assertEqual(current._next_outgoing_packet(decision), request)
        with patch.object(s, "_now_ms", return_value=decision):
            current._write_packet(request, current._outgoing_generation)
        self.assertIsNone(self.submit_at(current, decision + 300, [target]))
        sent_at = decision + 300
        self.write_next(current, sent_at, request)
        settling_generation = current.frame_generation()
        self.assertIsNone(self.submit_at(current, sent_at + 249, [target]))
        self.assertIsNone(self.submit_at(current, sent_at + 250, [target], generation=settling_generation))
        self.assertIs(self.submit_at(current, sent_at + 251, [target]), target)
        self.assertEqual(current._latest_packet[2], 6)

    def test_blue_without_supply05_cancels_budget_returns24_no_delivery(self):
        for ids in ([3], [3, 3], [3, 4], [3] * 5):
            with self.subTest(ids=ids):
                current, first_request = self.start_collection()
                self.acknowledge_search(current, 1600)
                decision = self.check_near(current, ids, 2000)
                self.assertEqual(current._next_outgoing_packet(decision), s.build_event_packet(0x24))
                self.assertEqual(current.mode, s.MODE_FINAL_ROI_TRACK)
                self.assertEqual(current.target_label, s.TARGET_BLACK_OR_GREEN)
                self.assertIsNone(current._supply_collect_started_ms)
                self.assertEqual(current._supply_collect_attempts, 0)
                self.assertEqual(current._pending_events, [])
                self.assertIsNone(current._latest_packet)
                self.assertFalse(current._supply_collect_delivery_pending)
                self.assertIsNone(current._next_outgoing_packet(first_request + config.UART_SUPPLY_COLLECT_TIMEOUT_MS))

    def test_blue_no_supply_return24_keeps_roi_exclusion_for_next_search(self):
        current, _ = self.start_collection()
        self.acknowledge_search(current, 1600)
        decision = self.check_near(current, [3], 2000)
        current._next_outgoing_packet(decision)
        self.assertIsNone(self.submit_at(current, 2600, [block(288, 430)]))
        self.assertIsNone(current._latest_packet)
        self.assertFalse(current._final_roi_sent)
        outside = block(250, 200, 6)
        self.assertIs(self.submit_at(current, 2610, [block(288, 430), outside]), outside)
        self.assertEqual(current._latest_packet[2], 6)
        self.assertIsNone(current._supply_collect_started_ms)

    def test_10_seconds_is_total_not_reset_by51_03_or_second_attempt(self):
        current, first_request = self.start_collection()
        self.acknowledge_search(current, 5000)
        second_request = self.check_near(current, [6], 9000)
        current._next_outgoing_packet(second_request)
        self.acknowledge_search(current, 10000)
        self.assertIsNone(current._next_outgoing_packet(first_request + config.UART_SUPPLY_COLLECT_TIMEOUT_MS - 1))
        self.assert_delivery(current, first_request + config.UART_SUPPLY_COLLECT_TIMEOUT_MS)

    def test_uart_timer_finishes_from_wait_search_track24_and_arrange_without_camera(self):
        for mode in (s.MODE_WAIT_SUPPLY_SEARCH, s.CMD_SEARCH, s.CMD_TRACK,
                     s.MODE_FINAL_ROI_TRACK, s.MODE_ARRANGE_RIGHTMOST,
                     s.MODE_ARRANGE_LEFTMOST, s.MODE_POST_ARRANGE_CHECK):
            with self.subTest(mode=mode):
                current, first_request = self.start_collection()
                current.mode = mode
                current._pending_events = [s.build_event_packet(code) for code in (0xE3, 0xE4, 0x34, 0x12)]
                current._latest_packet = s.build_coordinate_packet(6, 2, 3)
                self.assert_delivery(current, first_request + config.UART_SUPPLY_COLLECT_TIMEOUT_MS)
                self.assertIsNone(current._latest_packet)
                self.assertEqual(current._pending_events, [])

    def test_limit_overrides_in_progress_old_frame_and_prepared_uart_packet(self):
        current, first_request = self.start_collection()
        self.acknowledge_search(current, 1600)
        generation = current.frame_generation()
        old_packet = s.build_coordinate_packet(6, 7, 9)
        current._serial = SimpleNamespace(write=Mock(return_value=len(old_packet)))
        with patch.object(s, "_now_ms", return_value=first_request + config.UART_SUPPLY_COLLECT_TIMEOUT_MS):
            current._write_packet(old_packet, generation=generation)
        current._serial.write.assert_not_called()
        self.assertIsNone(self.submit_at(current, first_request + config.UART_SUPPLY_COLLECT_TIMEOUT_MS,
                                        [block(288, 224)], generation=generation))
        self.assert_delivery(current, first_request + config.UART_SUPPLY_COLLECT_TIMEOUT_MS)

    def test_timeout_cargo_report_precedes06_by_existing_command_gap(self):
        current, first_request = self.start_collection()
        now = first_request + config.UART_SUPPLY_COLLECT_TIMEOUT_MS
        self.assertEqual(current._next_outgoing_packet(now), s.build_event_packet(0x05))
        current._pending_events.insert(0, s.build_event_packet(0x05))
        now = confirm_final_load_check(current, [5], now)
        report = current._next_outgoing_packet(now)
        current._serial = SimpleNamespace(write=Mock(side_effect=lambda data: len(data)))
        with patch.object(s, "_now_ms", return_value=now):
            current._write_packet(report, generation=current._outgoing_generation)
        self.assertEqual(report, s.build_coordinate_packet(5, 0, 0))
        self.assertIsNone(current._next_outgoing_packet(now + 49))
        self.assertEqual(current._next_outgoing_packet(now + 50), s.build_event_packet(0x06))

    def test_new_tasks_cancel_budget_but_same51_ack_preserves_it(self):
        for command in (s.CMD_SELECT_RED_OBJECT, s.CMD_SELECT_BLACK_OBJECT,
                        s.CMD_SELECT_GREEN_OBJECT):
            with self.subTest(command=command):
                current, first_request = self.start_collection()
                self.command_at(current, command, 1600)
                self.assertIsNone(current._supply_collect_started_ms)
                self.assertEqual(current._supply_collect_attempts, 0)
                self.assertNotEqual(current._next_outgoing_packet(first_request + config.UART_SUPPLY_COLLECT_TIMEOUT_MS),
                                    s.build_event_packet(0x06))

    def test_late_capture_handshakes_cannot_cancel_delivery_new_task_can(self):
        current, first_request = self.start_collection()
        now = first_request + config.UART_SUPPLY_COLLECT_TIMEOUT_MS
        self.assert_delivery(current, now)
        for code in (0x51, 0x03, 0x04, 0x14, 0x44, 0x12, 0x22, 0x24, 0x25, 0xF1):
            generation = current.frame_generation()
            self.command_at(current, code, now + 1)
            self.assertEqual(current.mode, s.MODE_SEARCH_ZONE)
            self.assertEqual(current.frame_generation(), generation)
        self.command_at(current, 0x11, now + 10)
        self.command_at(current, 0x03, now + 11)
        self.assertEqual(current.target_label, "sqarered")
        self.assertEqual(current.mode, s.CMD_SEARCH)

    def test_next51_task_allowed_after_zone_close(self):
        current, first_request = self.start_collection()
        self.assert_delivery(current, first_request + config.UART_SUPPLY_COLLECT_TIMEOUT_MS)
        current._zone_close_sent = True
        self.acknowledge_search(current, first_request + 11000)
        self.assertFalse(current._supply_collect_delivery_pending)
        self.assertIsNone(current._supply_collect_started_ms)

    def finish_collection_for_test(self, reason="total_timeout"):
        current, first_request = self.start_collection(6)
        self.acknowledge_search(current, 1600)
        if reason == "attempt_limit":
            second = self.check_near(current, [6], 3000)
            self.write_next(current, second, s.build_event_packet(0x03))
            self.acknowledge_search(current, 3600)
            now = self.check_near(current, [6], 4500)
        elif reason == "enough_supplies":
            now = self.check_near(current, [5, 6], 3000)
        elif reason == "final_inventory":
            now = 1800
            self.command_at(current, s.CMD_ZONE_LOAD_RECHECK, now)
        else:
            now = first_request + config.UART_SUPPLY_COLLECT_TIMEOUT_MS
            self.submit_at(current, now)
        self.assertTrue(current._supply_collect_finished)
        self.assertEqual(current._supply_collect_finish_reason, reason)
        self.assertEqual(current._zone_load_recheck, "wait05tx")
        return current, now

    def assert_finished_single_delivers(self, current, start, class_id=6):
        decision = self.check_near(current, [class_id], start)
        self.assertNotEqual(current.mode, s.MODE_WAIT_SUPPLY_SEARCH)
        self.assertTrue(current._supply_collect_finished)
        self.assertIsNone(current._supply_collect_started_ms)
        self.assertEqual(current._zone_load_recheck_source, "before06")
        self.assertEqual(current._pending_events, [s.build_event_packet(0x05)])
        final = confirm_final_load_check(current, [class_id], decision)
        self.write_next(current, final, s.build_coordinate_packet(5, 0, 0))
        self.assertIsNone(current._next_outgoing_packet(final + 49))
        self.write_next(current, final + 50, s.build_event_packet(0x06))
        self.assertEqual(current.mode, s.MODE_SEARCH_ZONE)
        self.assertTrue(current._supply_collect_finished)

    def test_finished_budget_survives_foreign_final05_arrange_and_single_recapture(self):
        for reason in ("total_timeout", "attempt_limit", "enough_supplies", "final_inventory"):
            for class_id in (5, 6):
                with self.subTest(reason=reason, class_id=class_id):
                    current, now = self.finish_collection_for_test(reason)
                    decision = confirm_final_load_check(current, [3, class_id], now)
                    self.write_next(current, decision, s.build_event_packet(0x02))
                    self.command_at(current, 0x12, decision + 1000)
                    self.write_next(current, decision + 1000, s.build_event_packet(0x12))
                    self.command_at(current, 0x22, decision + 2000)
                    self.assertEqual(current.mode, s.MODE_POST_ARRANGE_CHECK)
                    self.assertTrue(current._supply_collect_finished)
                    self.acknowledge_search(current, decision + 2100)
                    self.assertTrue(current._supply_collect_finished)
                    self.assertEqual(current._supply_collect_finish_reason, reason)
                    self.assert_finished_single_delivers(current, decision + 3000, class_id)

    def test_reported_timeout_blue_black_push_search_capture_recheck_never_reopens_budget(self):
        current, now = self.finish_collection_for_test()
        decision = confirm_final_load_check(current, [3, 6], now)
        self.write_next(current, decision, s.build_event_packet(0x02))
        self.command_at(current, 0x12, decision + 1000)
        self.write_next(current, decision + 1000, s.build_event_packet(0x12))
        self.command_at(current, 0x22, decision + 2000)
        self.submit_at(current, decision + 2010)
        self.write_next(current, decision + 2010, s.build_event_packet(0xE2))
        # MCU post22 E2 recovery explicitly returns51/03, not a new delivery.
        self.acknowledge_search(current, decision + 2110)
        target = block(288, 224, 6)
        self.submit_at(current, decision + 2210, [target])
        self.write_next(current, decision + 2210, s.build_event_packet(0x04))
        self.command_at(current, 0x04, decision + 2310)
        self.submit_at(current, decision + 2410, [target])
        self.write_next(current, decision + 2410, s.build_event_packet(0x14))
        self.command_at(current, 0x14, decision + 2510)
        self.submit_at(current, decision + 2610, [target])
        self.write_next(current, decision + 2610, s.build_event_packet(0x24))
        self.submit_at(current, decision + 2910, [block(288, 390, 6)])
        self.write_next(current, decision + 2910, s.build_event_packet(0x34))
        self.assertEqual(current._supply_collect_finish_reason, "total_timeout")
        self.assert_finished_single_delivers(current, decision + 3010)

    def test_finished_budget_survives_empty_or_foreign_only_recheck_and_recapture(self):
        for ids in ([], [3], [3, 3], [4], [3, 4]):
            with self.subTest(ids=ids):
                current, now = self.finish_collection_for_test()
                decision = confirm_final_load_check(current, ids, now)
                self.write_next(current, decision, s.build_event_packet(0x24))
                resume = decision + config.UART_WIDE_VIEW_SETTLE_MS
                self.assertIsNone(current._next_outgoing_packet(resume))
                self.assertIsNone(current._zone_load_recheck)
                self.assertTrue(current._supply_collect_finished)
                self.acknowledge_search(current, resume + 100)
                self.assert_finished_single_delivers(current, resume + 1000)

    def test_finished_budget_survives_repeated_ordinary_empty_or_blue_only05(self):
        for ids in ([], [3], [3, 3]):
            with self.subTest(ids=ids):
                current, now = self.finish_collection_for_test()
                decision = confirm_final_load_check(current, [6, 3], now)
                self.write_next(current, decision, s.build_event_packet(0x02))
                for offset in (1000, 2000):
                    empty = self.check_near(current, ids, decision + offset)
                    self.write_next(current, empty, s.build_event_packet(0x24))
                    self.assertTrue(current._supply_collect_finished)
                    self.assertIsNone(current._supply_collect_started_ms)
                self.assert_finished_single_delivers(current, decision + 4000)

    def test_finished_budget_mixed_or_excess_recapture_still_arranges_not_blind06(self):
        for ids in ([6, 3], [5, 4], [6, 5, 5, 6]):
            with self.subTest(ids=ids):
                current, now = self.finish_collection_for_test()
                decision = confirm_final_load_check(current, [6, 3], now)
                self.write_next(current, decision, s.build_event_packet(0x02))
                recapture = self.check_near(current, ids, decision + 1000)
                self.write_next(current, recapture, s.build_event_packet(0x02))
                self.assertTrue(current._supply_collect_finished)
                self.assertNotIn(s.build_event_packet(0x06), current._pending_events)
                self.assertNotIn(s.build_event_packet(0x03), current._pending_events)

    def test_finished_budget_not_rearmed_by_same_task_commands_or_expired_timers(self):
        current, now = self.finish_collection_for_test()
        decision = confirm_final_load_check(current, [6, 3], now)
        self.write_next(current, decision, s.build_event_packet(0x02))
        for code in (0x12, 0x22, 0x51, 0x03, 0x04, 0x14, 0x44, 0x13, 0x24, 0xF1):
            self.command_at(current, code, decision + 1000)
            self.assertTrue(current._supply_collect_finished)
            self.assertEqual(current._supply_collect_finish_reason, "total_timeout")
        self.assertNotEqual(current._next_outgoing_packet(decision + 100000), s.build_event_packet(0x05))
        self.assertTrue(current._supply_collect_finished)

    def test_finished_budget_blocks_manual_queued_or_prepared_new_supplement03(self):
        current, now = self.finish_collection_for_test()
        decision = confirm_final_load_check(current, [6, 3], now)
        self.write_next(current, decision, s.build_event_packet(0x02))
        self.acknowledge_search(current, decision + 1000)
        request = s.build_event_packet(0x03)
        current.send_event(0x03)
        self.assertIsNone(current._next_outgoing_packet(decision + 1010))
        self.assertNotIn(request, current._pending_events)
        serial = SimpleNamespace(write=Mock(return_value=len(request)))
        with patch.object(current, "_serial", serial), patch.object(s, "_now_ms", return_value=decision + 1010):
            current._write_packet(request, generation=current.frame_generation())
        serial.write.assert_not_called()

    def test_fe_resets_finished_delivery_and_allows_new_first_supplement(self):
        current, now = self.finish_collection_for_test()
        self.command_at(current, s.CMD_RESET_TASK, now + 1)
        self.assertFalse(current._supply_collect_finished)
        self.assertIsNone(current._supply_collect_finish_reason)
        self.command_at(current, 0x51, now + 2)
        self.command_at(current, 0x03, now + 3)
        first = self.check_near(current, [6], now + 1000)
        self.write_next(current, first, s.build_event_packet(0x03))
        self.assertEqual(current._supply_collect_attempts, 1)
        self.assertEqual(current._supply_collect_started_ms, first)

    def test_explicit_different_task_then51_resets_finished_delivery(self):
        for command in (0x01, 0x11, 0x21, 0x31):
            with self.subTest(command=command):
                current, now = self.finish_collection_for_test()
                self.command_at(current, command, now + 1)
                self.assertFalse(current._supply_collect_finished)
                self.assertIsNone(current._supply_collect_finish_reason)
                self.command_at(current, 0x51, now + 2)
                first = self.check_near(current, [6], now + 1000)
                self.write_next(current, first, s.build_event_packet(0x03))
                self.assertEqual(current._supply_collect_attempts, 1)

    def test_new51_after_delivery26_resets_finished_budget_for_next_round(self):
        current, first_request = self.start_collection(6)
        self.assert_delivery(current, first_request + config.UART_SUPPLY_COLLECT_TIMEOUT_MS)
        self.assertTrue(current._supply_collect_finished)
        current._zone_close_sent = True
        self.acknowledge_search(current, first_request + 18000)
        self.assertFalse(current._supply_collect_finished)
        self.assertIsNone(current._supply_collect_finish_reason)
        first = self.check_near(current, [6], first_request + 19000)
        self.write_next(current, first, s.build_event_packet(0x03))
        self.assertEqual(current._supply_collect_attempts, 1)

    def test_failed_final05_write_does_not_unlock_budget_or_emit03(self):
        current, now = self.finish_collection_for_test()
        packet = current._next_outgoing_packet(now)
        current._serial = SimpleNamespace(write=Mock(return_value=0))
        with patch.object(s, "_now_ms", return_value=now):
            current._write_packet(packet, current._outgoing_generation)
        self.assertEqual(current._zone_load_recheck, "wait05tx")
        self.assertTrue(current._supply_collect_finished)
        self.assertEqual(current._pending_events, [s.build_event_packet(0x05)])
        for time in (now + 500, now + 1000, now + 20000):
            self.submit_at(current, time, [block(200, 200, 6)])
            self.assertEqual(current._near_view_stable_frames, 0)
            self.assertNotIn(s.build_event_packet(0x03), current._pending_events)

    def test_loaded_roi_excluded_from_tracking_new_entry_is34_only(self):
        current, _ = self.start_collection()
        self.acknowledge_search(current, 1600)
        carried, new = block(288, 430), block(250, 180, 6)
        self.assertIsNone(self.submit_at(current, 1700, [carried]))
        self.assertIsNone(current._latest_packet)
        self.assertIs(self.submit_at(current, 1710, [carried, new]), new)
        self.assertEqual(current._latest_packet[2], 6)
        # A new outside lock entering ROI gets34, but never ROI coordinates
        # or a tracking/display target. The tracking anchor stays outside.
        current.mode = s.MODE_FINAL_ROI_TRACK
        current._final_target_lock = current._search_target_lock
        entering = block(250, 390, 6)
        self.assertIsNone(self.submit_at(current, 1800, [carried, entering]))
        self.assertIsNone(current._latest_packet)
        self.assertEqual(current._final_target_lock[2], 180)
        self.assertEqual(current._supply_roi_handoff[2], 390)
        self.assertEqual(current._next_outgoing_packet(1800), s.build_event_packet(0x34))

    def test_supplement24_can_select_lowest_supply_outside_the_roi(self):
        current, _ = self.start_collection()
        current.tracking_roi = make_bottom_center_roi(576, 448)
        self.acknowledge_search(current, 1600)
        current.mode = s.MODE_FINAL_ROI_TRACK
        carried, new, farther = block(313, 430), block(300, 335, 6), block(70, 130)
        with patch.object(config, "UART_TRACKING_ROI_SHAPE", "triangle"):
            self.assertFalse(current._target_center_in_tracking_roi(new))
            self.assertIs(self.submit_at(current, 1700, [carried, new, farther]), new)
            self.assertEqual(current._latest_packet, s.build_coordinate_packet(6, -13, -24))
            self.assertNotIn(s.build_event_packet(0x34), current._pending_events)
            entered = block(300, 360, 6)
            self.assertIsNone(self.submit_at(current, 1720, [carried, entered, farther]))
            self.assertEqual(current._next_outgoing_packet(1720), s.build_event_packet(0x34))

    def test_supplement24_keeps_new_lock_when_y_order_changes_or_cargo_disappears(self):
        current, _ = self.start_collection()
        current.tracking_roi = make_bottom_center_roi(576, 448)
        self.acknowledge_search(current, 1600)
        current.mode = s.MODE_FINAL_ROI_TRACK
        cargo, new = block(230, 210), block(290, 160)
        with patch.object(config, "UART_TRACKING_ROI_SHAPE", "triangle"):
            self.assertIs(self.submit_at(current, 1700, [new]), new)
            moved = block(290, 215)
            self.assertIs(self.submit_at(current, 1710, [cargo, moved]), moved)
            self.assertIs(self.submit_at(current, 1720, [moved]), moved)
            self.assertEqual(current._latest_packet, s.build_coordinate_packet(5, -23, -144))

    def test_supplement24_protects_inherited_black_target_on_first_frame(self):
        for cargo in ([], [block(70, 190)]):
            with self.subTest(cargo_visible=bool(cargo)), \
                    patch.object(config, "UART_TRACKING_ROI_SHAPE", "triangle"), \
                    patch.object(config, "UART_DIRECT_CAPTURE_AFTER_14", True):
                current, _ = self.start_collection()
                current.tracking_roi = make_bottom_center_roi(576, 448)
                self.acknowledge_search(current, 1600)
                approaching = block(313, 340, 6)
                self.submit_at(current, 1610, [approaching])
                self.write_next(current, 1610, s.build_event_packet(0x04))
                self.command_at(current, 0x04, 1620)
                self.submit_at(current, 1630, [approaching])
                self.write_next(current, 1660, s.build_event_packet(0x14))
                self.command_at(current, 0x14, 1700)
                self.write_next(current, 1710, s.build_event_packet(0x24))

                first = block(313, 205, 6)
                self.assertIs(self.submit_at(current, 1720, cargo + [first]), first)
                self.assertEqual(current._latest_packet, s.build_coordinate_packet(6, 0, -154))
                self.submit_at(current, 2010, cargo + [block(313, 360, 6)])
                self.write_next(current, 2010, s.build_event_packet(0x34))

    def test51_roi_mask_applies_before_collection_budget_and_in_all_tracking_stages(self):
        for mode in (s.CMD_SEARCH, s.CMD_TRACK, s.MODE_WAIT_04_HANDSHAKE,
                     s.MODE_WAIT_14_HANDSHAKE, s.MODE_FINAL_ROI_TRACK):
            with self.subTest(mode=mode):
                current = controller(mode)
                self.command_at(current, 0x51, 1000)
                current.mode = mode
                self.assertIsNone(current._supply_collect_started_ms)
                self.assertIsNone(self.submit_at(current, 1100, [block(288, 430), block(200, 420, 6)]))
                self.assertIsNone(current._latest_packet)
                self.assertFalse(current._final_roi_sent)
                self.assertNotIn(s.build_event_packet(0x34), current._pending_events)

    def test_roi_cargo_cannot_steal_any_raw_tracking_lock(self):
        for mode, attribute in ((s.CMD_SEARCH, "_search_target_lock"),
                                (s.CMD_TRACK, "_track_target_lock"),
                                (s.MODE_WAIT_04_HANDSHAKE, "_track_target_lock"),
                                (s.MODE_WAIT_14_HANDSHAKE, "_track_target_lock"),
                                (s.MODE_FINAL_ROI_TRACK, "_final_target_lock")):
            with self.subTest(mode=mode):
                current, _ = self.start_collection()
                self.acknowledge_search(current, 1600)
                current.mode = mode
                setattr(current, attribute, current._final_target_identity(block(288, 350)))
                carried, outside = block(288, 410), block(100, 250)
                # Cargo is nearer to the previous target than the outside box.
                # Lock association must still use the outside raw detection.
                self.assertIs(self.submit_at(current, 1700, raw=[carried, outside]), outside)
                self.assertEqual(getattr(current, attribute)[2], 250)
                self.assertIsNotNone(current._latest_packet)
                self.assertNotIn(s.build_event_packet(0x34), current._pending_events)
                self.assertIsNone(self.submit_at(current, 1710, raw=[carried]))
                self.assertIsNone(current._latest_packet)
                self.assertFalse(current._final_roi_sent)

    def test03_04_new_target_entering_inner_roi_is_not_tracked_and_does_not_send34(self):
        for mode, attribute in ((s.CMD_SEARCH, "_search_target_lock"),
                                (s.CMD_TRACK, "_track_target_lock")):
            with self.subTest(mode=mode):
                current, _ = self.start_collection()
                self.acknowledge_search(current, 1600)
                current.mode = mode
                setattr(current, attribute, current._final_target_identity(block(200, 350)))
                self.assertIsNone(self.submit_at(current, 1700, [block(200, 410)]))
                self.assertIsNone(current._latest_packet)
                self.assertIsNone(current._supply_roi_handoff)
                self.assertNotIn(s.build_event_packet(0x34), current._pending_events)

    def test_existing_roi_cargo_is_remembered_through_detection_flicker(self):
        current, _ = self.start_collection()
        self.acknowledge_search(current, 1600)
        current.mode = s.MODE_FINAL_ROI_TRACK
        carried, approaching = block(288, 420), block(288, 340)
        self.submit_at(current, 1700, [carried, approaching])
        self.submit_at(current, 1710, [approaching])  # carried box flickers
        self.assertIsNone(self.submit_at(current, 1720, [carried]))
        self.assertFalse(current._final_roi_sent)
        self.assertIsNone(current._supply_roi_handoff)
        self.assertIsNone(current._latest_packet)

    def test_same_color_entry_repeats34_only_with_fresh_observations(self):
        current, _ = self.start_collection()
        self.acknowledge_search(current, 1600)
        current.mode = s.MODE_FINAL_ROI_TRACK
        carried, approaching = block(400, 430), block(200, 340)
        self.assertIs(self.submit_at(current, 1700, [carried, approaching]), approaching)
        entered = block(200, 390)
        self.assertIsNone(self.submit_at(current, 1800, [carried, entered]))
        self.assertEqual(current._next_outgoing_packet(1800), s.build_event_packet(0x34))
        self.assertIsNone(current._next_outgoing_packet(2299))
        self.submit_at(current, 2300, raw=[carried, entered])
        self.assertEqual(current._next_outgoing_packet(2300), s.build_event_packet(0x34))
        self.assertIsNone(current._latest_packet)
        self.assertEqual(current._final_no_target_frames, 0)
        # Camera stall must not endlessly repeat an old entry observation.
        self.assertIsNone(current._next_outgoing_packet(2300 + config.UART_TARGET_FRESHNESS_MS + 1))
        self.assertFalse(current._final_roi_sent)

    def test_handoff_disappearing_cannot_be_replaced_by_old_cargo(self):
        current, _ = self.start_collection()
        self.acknowledge_search(current, 1600)
        current.mode = s.MODE_FINAL_ROI_TRACK
        carried, approaching = block(400, 430), block(200, 340)
        self.submit_at(current, 1700, [carried, approaching])
        self.submit_at(current, 1800, [carried, block(200, 390)])
        self.assertTrue(current._final_roi_sent)
        self.submit_at(current, 1810, [carried])
        self.assertIsNone(current._latest_packet)
        self.assertFalse(current._final_roi_sent)
        self.assertIsNone(current._supply_roi_handoff)
        self.assertNotIn(s.build_event_packet(0x34), current._pending_events)
        self.assertIsNone(current._next_outgoing_packet(2300))

    def test_new_handoff_target_leaving_roi_resumes_raw_outside_tracking(self):
        current, _ = self.start_collection()
        self.acknowledge_search(current, 1600)
        current.mode = s.MODE_FINAL_ROI_TRACK
        carried, approaching = block(400, 430), block(200, 340)
        self.submit_at(current, 1700, [carried, approaching])
        self.submit_at(current, 1800, [carried, block(200, 390)])
        left = block(200, 350)
        self.assertIs(self.submit_at(current, 1810, raw=[carried, left]), left)
        self.assertIsNotNone(current._latest_packet)
        self.assertFalse(current._final_roi_sent)
        self.assertIsNone(current._supply_roi_handoff)
        self.assertNotIn(s.build_event_packet(0x34), current._pending_events)

    def test_custom_loaded_area_does_not_replace_actual_inner_half(self):
        current, _ = self.start_collection()
        self.acknowledge_search(current, 1600)
        with patch.object(config, "UART_SUPPLY_COLLECT_LOADED_ROI", [0, 0, 100, 100]):
            roi_cargo, extra_cargo, outside = block(288, 430), block(50, 50, 6), block(200, 180)
            self.assertIsNone(self.submit_at(current, 1700, [roi_cargo, extra_cargo]))
            self.assertIs(self.submit_at(current, 1710, [roi_cargo, extra_cargo, outside]), outside)

    def test_roi_only51_loss_uses_existing_ee_f1_flow(self):
        current, first_request = self.start_collection()
        self.acknowledge_search(current, 1600)
        current.mode = s.MODE_FINAL_ROI_TRACK
        for index in range(config.UART_FINAL_NO_TARGET_FRAMES):
            self.submit_at(current, 1700 + index * 10, [block(288, 430)])
        self.assertEqual(current._next_outgoing_packet(1800), s.build_event_packet(0xEE))
        self.assertIsNone(current._next_outgoing_packet(2300))
        self.command_at(current, 0xF1, 2400)
        outside = block(200, 250)
        self.assertIs(self.submit_at(current, 2410, [block(288, 430), outside]), outside)
        self.assertIsNotNone(current._latest_packet)
        self.assertEqual(current._supply_collect_started_ms, first_request)

    def test_near05_still_counts_supplies_inside_roi(self):
        current, _ = self.start_collection()
        self.acknowledge_search(current, 1600)
        self.command_at(current, 0x25, 2000)
        self.write_next(current, 2000, s.build_event_packet(0x05))
        for index in range(3):
            self.submit_at(current, 2500 + index * 10, [near_view_block(200, 400), near_view_block(350, 400, 6)])
        self.assert_delivery(current, 2520)

    def test_roi_exclusion_does_not_change_single_class_tasks_or_first_green(self):
        for command in (0x21, 0x31, 0x11):
            with self.subTest(command=command):
                current = controller(s.MODE_FINAL_ROI_TRACK)
                self.command_at(current, command, 1000)
                class_id = {0x21: 5, 0x31: 6, 0x11: 4}[command]
                target = block(288, 400, class_id)
                observed_at = (1060, 1080, 1100) if class_id == 4 else (1100,)
                for now in observed_at:
                    self.assertIs(self.submit_at(current, now, [target]), target)
                self.assertEqual(current._next_outgoing_packet(1100), s.build_event_packet(0x34))

    def test_old_generation_roi_frame_cannot_create_cargo_or_handoff(self):
        current, _ = self.start_collection()
        self.acknowledge_search(current, 1600)
        old_generation = current.frame_generation()
        self.command_at(current, 0x03, 1700)
        self.assertIsNone(self.submit_at(current, 1710, [block(288, 400)], generation=old_generation))
        self.assertEqual(current._supply_loaded_roi_anchors, [])
        self.assertIsNone(current._supply_roi_handoff)
        self.assertEqual(current._pending_events, [])

    def test_prepared_coordinate_and_canceled34_cannot_cross_handoff(self):
        current, _ = self.start_collection()
        self.acknowledge_search(current, 1600)
        current.mode = s.MODE_FINAL_ROI_TRACK
        carried, approaching = block(400, 430), block(200, 340)
        self.submit_at(current, 1700, [carried, approaching])
        old_coordinate = current._next_outgoing_packet(1700)
        current._serial = SimpleNamespace(write=Mock(side_effect=lambda data: len(data)))
        self.submit_at(current, 1800, [carried, block(200, 390)])
        with patch.object(s, "_now_ms", return_value=1800):
            current._write_packet(old_coordinate, generation=current.frame_generation())
        current._serial.write.assert_not_called()
        old34 = current._next_outgoing_packet(1800)
        self.submit_at(current, 1810, [carried])
        with patch.object(s, "_now_ms", return_value=1810):
            current._write_packet(old34, generation=current.frame_generation())
        current._serial.write.assert_not_called()

    def test_post14_decision_and_post22_exclude_roi_even_before_extra_attempt(self):
        for mode in (s.MODE_ARRANGE_RIGHTMOST, s.MODE_POST_ARRANGE_CHECK):
            with self.subTest(mode=mode):
                current = controller(mode)
                self.command_at(current, 0x51, 1000)
                current.mode = mode
                current._post07_supply_tracking = True
                current._initial_arrangement_decision_pending = mode == s.MODE_ARRANGE_RIGHTMOST
                self.submit_at(current, 1100, [block(288, 430), block(200, 420, 6)])
                self.assertNotEqual(current.mode, s.MODE_FINAL_ROI_TRACK)
                self.assertNotIn(s.build_event_packet(0x24), current._pending_events)
                self.assertIsNone(current._latest_packet)

    def test04_full_roi_center_boundary_uses_center_not_box_overlap(self):
        current, _ = self.start_collection()
        self.acknowledge_search(current, 1600)
        #04 now excludes the WHOLE ROI too, not just its inner half.
        current.mode = s.CMD_TRACK
        self.assertIsNone(self.submit_at(current, 1700, [block(200, 403.5)]))
        self.assertIsNone(self.submit_at(current, 1705, [block(200, 403)]))
        above = block(200, 358)  # native box overlaps ROI, center is outside
        self.assertIs(self.submit_at(current, 1710, [above]), above)
        self.assertIsNotNone(current._latest_packet)

    def test_default04_roi_excludes_front_half_as_well_as_inner_half(self):
        current, _ = self.start_collection()
        current.tracking_roi = make_bottom_center_roi(576, 448)
        self.acknowledge_search(current, 1600)
        current.mode = s.CMD_TRACK
        self.assertIsNone(self.submit_at(current, 1700, [block(200, 383.5)]))
        self.assertIsNone(current._latest_packet)
        self.assertIsNone(self.submit_at(current, 1705, [block(200, 383)]))
        above = block(200, 318)
        self.assertIs(self.submit_at(current, 1710, [above]), above)
        self.assertIsNotNone(current._latest_packet)

    def test51_inner_half_geometry_float_midline_and_xy_bounds(self):
        current = controller(s.MODE_FINAL_ROI_TRACK)
        current.tracking_roi = make_bottom_center_roi(576, 448)
        for x, y, loaded in ((200, 319, False), (200, 383, False),
                             (200, 383.5, True), (200, 384, True),
                             (0, 447, True), (575, 447, True),
                             (-1, 400, False), (576, 400, False),
                             (200, 448, False)):
            with self.subTest(x=x, y=y):
                self.assertEqual(current._target_center_in_supply_loaded_roi(block(x, y)), loaded)
        current.tracking_roi = [20, 100, 400, 200]
        self.assertFalse(current._target_center_in_supply_loaded_roi(block(200, 199.5)))
        self.assertTrue(current._target_center_in_supply_loaded_roi(block(200, 200)))
        self.assertFalse(current._target_center_in_supply_loaded_roi(block(420, 250)))
        current.tracking_roi = None
        self.assertFalse(current._target_center_in_supply_loaded_roi(block(200, 400)))

    def test51_initial_front_half_candidate_below_capture_line_gets34(self):
        for class_id in (5, 6):
            for y in (364, 370, 383):
                with self.subTest(class_id=class_id, y=y):
                    current = controller(s.MODE_FINAL_ROI_TRACK)
                    current.tracking_roi = make_bottom_center_roi(576, 448)
                    self.command_at(current, 0x51, 1000)
                    current.mode = s.MODE_FINAL_ROI_TRACK
                    cargo = block(450, 420, class_id)
                    new = block(200, y, class_id)
                    self.assertIsNone(self.submit_at(current, 1100, [cargo, new]))
                    self.assertIsNone(current._latest_packet)
                    self.assertEqual(current._supply_roi_handoff[2], y)
                    self.assertEqual(current._next_outgoing_packet(1100), s.build_event_packet(0x34))
                    self.assertEqual(current._final_no_target_frames, 0)
                    self.assertEqual(len(current._supply_loaded_roi_anchors), 1)
                    self.assertEqual(current._supply_loaded_roi_anchors[0][2], 420)

    def test51_raw_front_half_candidate_can_use24_fallback_and34(self):
        current = controller(s.MODE_FINAL_ROI_TRACK)
        current.tracking_roi = make_bottom_center_roi(576, 448)
        self.command_at(current, 0x51, 1000)
        current.mode = s.MODE_FINAL_ROI_TRACK
        new = block(200, 370, 6)
        self.assertIsNone(self.submit_at(current, 1100, raw=[new, block(450, 420)]))
        self.assertEqual(current._supply_roi_handoff[0], 6)
        self.assertEqual(current._next_outgoing_packet(1100), s.build_event_packet(0x34))
        self.assertIsNone(current._latest_packet)

    def test51_inner_midline_cargo_alone_never_acquires_or_triggers34(self):
        for class_id in (5, 6):
            for y in (383.5, 384, 420, 447):
                with self.subTest(class_id=class_id, y=y):
                    current = controller(s.MODE_FINAL_ROI_TRACK)
                    current.tracking_roi = make_bottom_center_roi(576, 448)
                    self.command_at(current, 0x51, 1000)
                    current.mode = s.MODE_FINAL_ROI_TRACK
                    self.assertIsNone(self.submit_at(current, 1100, [block(200, y, class_id)]))
                    self.assertIsNone(current._latest_packet)
                    self.assertIsNone(current._supply_roi_handoff)
                    self.assertFalse(current._final_roi_sent)
                    self.assertNotIn(s.build_event_packet(0x34), current._pending_events)

    def test51_front_half_excluded_in03_04_and_both_handshake_waits(self):
        for mode in (s.CMD_SEARCH, s.CMD_TRACK, s.MODE_WAIT_04_HANDSHAKE,
                     s.MODE_WAIT_14_HANDSHAKE):
            for class_id in (5, 6):
                with self.subTest(mode=mode, class_id=class_id):
                    current = controller(mode)
                    current.tracking_roi = make_bottom_center_roi(576, 448)
                    self.command_at(current, 0x51, 1000)
                    current.mode = mode
                    new, cargo = block(200, 330, class_id), block(450, 420)
                    selected = self.submit_at(current, 1100, [cargo, new])
                    self.assertIsNone(selected)
                    self.assertIsNone(current._latest_packet)
                    self.assertNotIn(s.build_event_packet(0x34), current._pending_events)
                    self.assertTrue(all(anchor[2] >= 383.5 for anchor in current._supply_loaded_roi_anchors))

    def test51_outside03_lock_is_inherited_by24_for_front_half34(self):
        current, budget_start = self.start_collection()
        current.tracking_roi = make_bottom_center_roi(576, 448)
        self.acknowledge_search(current, 1600)
        cargo, new = block(450, 420), block(200, 270, 6)
        self.assertIs(self.submit_at(current, 1700, [cargo, new]), new)
        current.mode = s.MODE_FINAL_ROI_TRACK
        current._final_target_lock = current._search_target_lock
        entered = block(200, 370, 6)
        self.assertIsNone(self.submit_at(current, 1710, raw=[cargo, entered]))
        self.assertEqual(current._next_outgoing_packet(1710), s.build_event_packet(0x34))
        self.assertIsNone(current._latest_packet)
        self.assertEqual(current._supply_collect_started_ms, budget_start)
        self.assertEqual(current._supply_collect_attempts, 1)

    def test51_front_half34_continues_into_inner_half_then_stops_on_old_cargo_only(self):
        current = controller(s.MODE_FINAL_ROI_TRACK)
        current.tracking_roi = make_bottom_center_roi(576, 448)
        self.command_at(current, 0x51, 1000)
        current.mode = s.MODE_FINAL_ROI_TRACK
        cargo, new = block(450, 420), block(200, 370, 6)
        self.submit_at(current, 1100, [cargo, new])
        self.assertEqual(current._next_outgoing_packet(1100), s.build_event_packet(0x34))
        entered_inner = block(200, 390, 6)
        self.submit_at(current, 1600, raw=[cargo, entered_inner])
        self.assertEqual(current._supply_roi_handoff[2], 390)
        self.assertEqual(current._next_outgoing_packet(1600), s.build_event_packet(0x34))
        self.assertIsNone(current._latest_packet)
        self.submit_at(current, 1610, raw=[cargo])
        self.assertFalse(current._final_roi_sent)
        self.assertIsNone(current._supply_roi_handoff)
        self.assertIsNone(current._next_outgoing_packet(2100))

    def test51_front_half_handoff_exiting_full_roi_restores_coordinates(self):
        current = controller(s.MODE_FINAL_ROI_TRACK)
        current.tracking_roi = make_bottom_center_roi(576, 448)
        self.command_at(current, 0x51, 1000)
        current.mode = s.MODE_FINAL_ROI_TRACK
        cargo = block(450, 420)
        self.submit_at(current, 1100, [cargo, block(200, 370, 6)])
        current._next_outgoing_packet(1100)
        outside = block(200, 298, 6)
        self.assertIs(self.submit_at(current, 1110, raw=[cargo, outside]), outside)
        self.assertFalse(current._final_roi_sent)
        self.assertIsNotNone(current._latest_packet)
        self.assertNotIn(s.build_event_packet(0x34), current._pending_events)

    def test51_custom_mask_still_extends_full_tracking_roi_exclusion(self):
        current = controller(s.CMD_SEARCH)
        current.tracking_roi = make_bottom_center_roi(576, 448)
        self.command_at(current, 0x51, 1000)
        current.mode = s.CMD_TRACK
        with patch.object(config, "UART_SUPPLY_COLLECT_LOADED_ROI", [0, 0, 100, 100]):
            new = block(200, 298, 6)
            self.assertIs(self.submit_at(current, 1100, [block(450, 420), block(50, 50), new]), new)
            self.assertEqual(current._latest_packet[2], 6)

    def test51_front_half_only_capture_still_waits05_and_counts_whole_image(self):
        current, _ = self.start_collection()
        current.tracking_roi = make_bottom_center_roi(576, 448)
        self.acknowledge_search(current, 1600)
        self.submit_at(current, 1610, [block(288, 224)])
        self.write_next(current, 1610, s.build_event_packet(0x04))
        self.command_at(current, 0x04, 1620)
        self.submit_at(current, 1630, [block(288, 224)])
        self.write_next(current, 1660, s.build_event_packet(0x14))
        self.command_at(current, 0x14, 1700)
        self.submit_at(current, 1710, [block(200, 330), block(350, 420, 6)])
        self.write_next(current, 1710, s.build_event_packet(0x24))
        self.submit_at(current, 1720, [block(200, 370), block(350, 420, 6)])
        self.assertIsNone(current._next_outgoing_packet(1760))
        self.submit_at(current, 2010, [block(200, 370), block(350, 420, 6)])
        self.write_next(current, 2010, s.build_event_packet(0x34))
        self.assertIsNone(current._latest_packet)
        ready = self.check_near(current, [5, 6], 2100)
        self.assert_delivery(current, ready)

    def test51_post14_and_post22_allow_front_half_to_enter24(self):
        for mode in (s.MODE_ARRANGE_RIGHTMOST, s.MODE_POST_ARRANGE_CHECK):
            with self.subTest(mode=mode):
                current = controller(mode)
                current.tracking_roi = make_bottom_center_roi(576, 448)
                self.command_at(current, 0x51, 1000)
                current.mode = mode
                current._post07_supply_tracking = True
                current._initial_arrangement_decision_pending = mode == s.MODE_ARRANGE_RIGHTMOST
                new, cargo = block(200, 330, 6), block(450, 420)
                self.submit_at(current, 1100, [cargo, new])
                self.assertEqual(current.mode, s.MODE_FINAL_ROI_TRACK)
                self.assertEqual(current._next_outgoing_packet(1100), s.build_event_packet(0x24))
                self.assertIsNone(self.submit_at(current, 1110, raw=[cargo, block(200, 370, 6)]))
                self.assertEqual(current._next_outgoing_packet(1110), s.build_event_packet(0x34))
                self.assertIsNone(current._latest_packet)

    def test51_inner_cargo_afterEE_F1_cannot_hide_new_front_half_target(self):
        current = controller(s.MODE_FINAL_ROI_TRACK)
        current.tracking_roi = make_bottom_center_roi(576, 448)
        self.command_at(current, 0x51, 1000)
        current.mode = s.MODE_FINAL_ROI_TRACK
        cargo = block(450, 420)
        for index in range(config.UART_FINAL_NO_TARGET_FRAMES):
            self.submit_at(current, 1100 + index * 10, [cargo])
        self.assertEqual(current._next_outgoing_packet(1200), s.build_event_packet(0xEE))
        self.command_at(current, 0xF1, 1500)
        self.assertIsNone(self.submit_at(current, 1510, [cargo, block(200, 370, 6)]))
        self.assertEqual(current._next_outgoing_packet(1510), s.build_event_packet(0x34))
        self.assertIsNone(current._latest_packet)

    def test_relaxed51_raw_association_excludes_both_cargo_colors(self):
        current, _ = self.start_collection()
        self.acknowledge_search(current, 1600)
        current.mode = s.MODE_FINAL_ROI_TRACK
        current._post07_supply_tracking = True
        current._final_target_lock = current._final_target_identity(block(288, 350, 6))
        outside = block(100, 300, 5)
        self.assertIs(self.submit_at(current, 1700, raw=[block(288, 410, 6), block(300, 430, 5), outside]), outside)
        self.assertEqual(current._final_target_lock[0], 5)
        self.assertEqual(current._latest_packet[2], 5)
        self.assertFalse(current._final_roi_sent)

    def test_fresh_black_roi_entry_is_not_black_loss_or_red_switch(self):
        current = controller(s.MODE_FINAL_ROI_TRACK)
        self.command_at(current, 0x51, 1000)
        carried = block(400, 430)
        self.submit_at(current, 1700, [carried, block(200, 340, 6)])
        self.assertEqual(current._final_target_lock[0], 6)
        current._final_no_target_frames = config.UART_FINAL_NO_TARGET_FRAMES
        current._tracking_missing_class_id = 6
        current._tracking_missing_since_ms = 1800
        self.submit_at(current, 2700, [carried, block(200, 390, 6), block(450, 200, 4)])
        self.assertIsNone(current._red_switch_stage)
        self.assertEqual(current.mode, s.MODE_FINAL_ROI_TRACK)
        self.assertEqual(current._final_no_target_frames, 0)
        self.assertEqual(current._next_outgoing_packet(2700), s.build_event_packet(0x34))

    def test_mid_frame_command_cannot_commit_old_roi_handoff_or_loaded_history(self):
        current, _ = self.start_collection()
        self.acknowledge_search(current, 1600)
        current.mode = s.MODE_FINAL_ROI_TRACK
        self.submit_at(current, 1700, [block(400, 430), block(200, 340)])
        observe = current._observe_supply_roi_entry
        def command_during_observation(*args):
            result = observe(*args)
            self.command_at(current, 0x03, 1800)
            return result
        with patch.object(current, "_observe_supply_roi_entry", side_effect=command_during_observation):
            self.assertIsNone(self.submit_at(current, 1800, [block(400, 430), block(200, 365)]))
        self.assertEqual(current.mode, s.CMD_SEARCH)
        self.assertFalse(current._supply_roi_observed)
        self.assertEqual(current._supply_loaded_roi_anchors, [])
        self.assertIsNone(current._supply_roi_handoff)
        self.assertIsNone(current._latest_packet)
        self.assertNotIn(s.build_event_packet(0x34), current._pending_events)

    def test_configurable_loaded_area_and_raw_yolo_acquisition_fallback(self):
        current, _ = self.start_collection()
        self.acknowledge_search(current, 1600)
        with patch.object(config, "UART_SUPPLY_COLLECT_LOADED_ROI", [0, 250, 576, 198]):
            carried, new = block(288, 280), block(200, 100, 6)
            self.assertIsNone(self.submit_at(current, 1700, raw=[carried]))
            self.assertIs(self.submit_at(current, 1710, raw=[carried, new]), new)

    def test_inherited_supply_survives04_14_24_without_reranking_carried_load(self):
        current, _ = self.start_collection()
        self.acknowledge_search(current, 1600)
        carried = block(288, 430)
        approaching = block(260, 224)
        self.assertIs(self.submit_at(current, 1700, [carried, approaching]), approaching)
        centered = block(288, 224)
        self.assertIs(self.submit_at(current, 1710, [carried, centered]), centered)
        self.assertEqual(current.mode, s.MODE_WAIT_04_HANDSHAKE)
        self.assertEqual(current._next_outgoing_packet(1710), s.build_event_packet(0x04))
        self.command_at(current, 0x04, 1720)
        self.submit_at(current, 1730, [carried, centered])
        self.assertEqual(current._next_outgoing_packet(1730), s.build_event_packet(0x14))
        self.command_at(current, 0x14, 1740)
        closer = block(288, 260)
        # A stage transition deliberately withholds coordinates until a new
        # frame; its final lock, not the display return, identifies the target.
        self.assertIsNone(self.submit_at(current, 1750, [carried, closer]))
        self.assertEqual(current._next_outgoing_packet(1750), s.build_event_packet(0x24))
        self.assertEqual(current._final_target_lock[2], 260)

    def test_held_load_does_not_enable_black_to_red_switch(self):
        current, _ = self.start_collection(6)
        self.acknowledge_search(current, 1600)
        self.submit_at(current, 1700, [block(200, 150, 6)])
        for now in range(1800, 3000, 100):
            self.submit_at(current, now, raw=[block(200, 150, 4)])
        self.assertIsNone(current._red_switch_stage)
        self.assertEqual(current.target_label, s.TARGET_BLACK_OR_GREEN)
        self.assertNotIn(s.build_event_packet(0x11), current._pending_events)

    def test_search_loss_and13_recovery_do_not_reset_extra_collection_budget(self):
        current, first_request = self.start_collection()
        self.acknowledge_search(current, 1600)
        carried = block(288, 430)
        for now in range(1700, 1750, 10):
            self.submit_at(current, now, [carried])
        self.assertEqual(current._next_outgoing_packet(1750), s.build_event_packet(0xE3))
        self.command_at(current, 0x13, 2000)
        self.assertEqual(current._supply_collect_attempts, 1)
        self.assertEqual(current._supply_collect_started_ms, first_request)
        new = block(250, 160)
        self.assertIs(self.submit_at(current, 2010, [carried, new]), new)
        self.assertEqual(current._next_outgoing_packet(2010),
                         s.build_coordinate_packet(5, -38, -64))

    def test_post22_reacquisition_does_not_return24_for_carried_supply_alone(self):
        current, first_request = self.start_collection()
        self.acknowledge_search(current, 1600)
        current.mode = s.MODE_POST_ARRANGE_CHECK
        self.assertIsNone(self.submit_at(current, 1700, [block(288, 430)]))
        self.assertEqual(current.mode, s.MODE_POST_ARRANGE_CHECK)
        self.assertEqual(current._next_outgoing_packet(1700), s.build_event_packet(0xE2))
        new = block(250, 160, 6)
        self.submit_at(current, 1710, [block(288, 430), new])
        self.assertEqual(current._next_outgoing_packet(1710), s.build_event_packet(0x04))
        self.assertEqual(current._track_target_lock[0], 6)
        self.assertEqual(current._track_target_lock[2], 160)
        self.assertEqual(current._supply_collect_started_ms, first_request)

    def test_wait_for_search_ignores_duplicate_old_near_and_stage_commands(self):
        current, first_request = self.start_collection()
        for code in (0x25, 0x15, 0x24, 0x04, 0x14, 0x44, 0x12, 0x22, 0xF1):
            generation = current.frame_generation()
            self.command_at(current, code, 1600)
            self.assertEqual(current.mode, s.MODE_WAIT_SUPPLY_SEARCH)
            self.assertEqual(current.frame_generation(), generation)
        self.assertEqual(current._supply_collect_attempts, 1)
        self.assertEqual(current._supply_collect_started_ms, first_request)


if __name__ == "__main__":
    unittest.main()
