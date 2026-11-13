"""06: visible commanded-zone area must exceed2%; oneE6 per absence episode."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from test_vision_rules import LABELS, controller, SingleFrameTrackingScenario
import app_config as config
import serial_controller as serial


def zone(x=100, y=100, w=100, h=80, class_id=1):
    return SimpleNamespace(x=x, y=y, w=w, h=h, class_id=class_id, score=.4)


class SafetyZoneMinimumAreaTests(SingleFrameTrackingScenario):
    def setUp(self):
        for setting in (patch.object(config, "VISION_DIAGNOSTICS_ENABLED", False),
                        patch("builtins.print")):
            setting.start()
            self.addCleanup(setting.stop)

    def current(self, zone_id=1):
        current = controller(serial.MODE_SEARCH_ZONE)
        current.zone_label = LABELS[zone_id]
        return current

    def submit(self, current, raw, now=1000, generation=None):
        with patch.object(serial, "_now_ms", return_value=now):
            return current.submit([], LABELS, raw_objects=raw,
                                  frame_generation=current.frame_generation() if generation is None else generation)

    def write(self, current, packet, now=1000, writer=None, generation=None):
        writer = writer if writer is not None else Mock(side_effect=lambda data: len(data))
        with patch.object(current, "_serial", SimpleNamespace(write=writer)), \
                patch.object(serial, "_now_ms", return_value=now):
            current._write_packet(packet, current.frame_generation() if generation is None else generation)
        return writer

    def test_new_e6_is_a_normal_five_byte_logged_event(self):
        event = serial.build_event_packet(serial.EVENT_ZONE_NO_TARGET)
        self.assertEqual(event, bytes.fromhex("F1 F2 E6 1F 2F"))
        self.assertNotIn(0x6E, serial.EVENT_NAMES)
        current = self.current()
        self.submit(current, [])
        with patch("builtins.print") as log:
            self.write(current, current._next_outgoing_packet(1000))
        self.assertTrue(any("TX command 0xE6" in str(call) for call in log.call_args_list))
        self.assertEqual(current._zone_no_target_phase, "sent")

    def test_exact2percent_and_below_are_missing_but_above_is_immediate(self):
        for zone_id in (1, 2):
            for height, valid in ((9, False), (10, False), (11, True)):
                with self.subTest(zone_id=zone_id, height=height):
                    current = self.current(zone_id)
                    current.frame_width = current.frame_height = 100
                    current.tracking_roi = [0, 80, 100, 20]
                    obj = zone(10, 10, 20, height, zone_id)
                    selected = self.submit(current, [obj])
                    self.assertEqual(current._zone_area_ratio, 20 * height / 10_000)
                    self.assertIs(selected, obj if valid else None)
                    self.assertEqual(current._zone_found_sent, valid)
                    event = serial.EVENT_ZONE_FOUND if valid else serial.EVENT_ZONE_NO_TARGET
                    self.assertEqual(current._next_outgoing_packet(1000), serial.build_event_packet(event))
                    packet = current._next_outgoing_packet(1000)
                    self.assertEqual(packet is not None, valid)
                    if valid:
                        self.assertEqual(len(packet), 11)
                        self.assertEqual(packet[2], zone_id)

    def test_default_threshold_is_two_percent_and_missing_config_is_same(self):
        self.assertEqual(config.UART_ZONE_MIN_AREA_RATIO, .02)
        fallback = SimpleNamespace(**{key: value for key, value in vars(config).items()
                                      if key != "UART_ZONE_MIN_AREA_RATIO"})
        with patch.object(serial, "config", fallback):
            current = self.current()
            self.assertIsNone(self.submit(current, [zone(w=10, h=10)]))
            self.assertEqual(current._next_outgoing_packet(1000),
                             serial.build_event_packet(serial.EVENT_ZONE_NO_TARGET))
            valid = zone(w=80, h=80)  #2.48% proves fallback is not the old3%.
            self.assertIs(self.submit(current, [valid], 1100), valid)
            self.assertEqual(current._latest_packet[2], 1)

    def test_two_to_three_percent_zones_allow16_and_coordinates_for_both_colors_and_loads(self):
        for zone_id in (1, 2):
            for target_label in ("sqareredgreen", "triangualrblack", serial.TARGET_BLACK_OR_GREEN, "sqarered"):
                current = self.current(zone_id)
                current.target_label = target_label
                obj = zone(w=80, h=80, class_id=zone_id)
                self.assertIs(self.submit(current, [obj]), obj)
                self.assertGreater(current._zone_area_ratio, .02)
                self.assertLess(current._zone_area_ratio, .03)
                self.assertEqual(current._next_outgoing_packet(1000), serial.build_event_packet(serial.EVENT_ZONE_FOUND))
                self.assertEqual(len(current._next_outgoing_packet(1000)), 11)
                self.assertNotIn(serial.build_event_packet(serial.EVENT_ZONE_NO_TARGET), current._pending_events)

    def test_visible_clipped_area_not_full_box_or_sum_of_small_zones(self):
        current = self.current()
        clipped = zone(-1000, 0, 1010, 448)
        parts = [zone(50 * index, 100, 50, 100) for index in range(3)]
        self.assertGreater(clipped.w * clipped.h, 576 * 448 * .02)
        self.assertGreater(sum(obj.w * obj.h for obj in parts), 576 * 448 * .02)
        self.assertIsNone(self.submit(current, [clipped] + parts))
        self.assertIsNone(current._latest_packet)
        self.assertEqual(current._zone_area_ratio, 5000 / (576 * 448))

    def test_largest_commanded_color_wins_before_area_gate(self):
        for zone_id in (1, 2):
            current = self.current(zone_id)
            small = zone(w=10, h=10, class_id=zone_id)
            valid = zone(class_id=zone_id)
            wrong = zone(0, 0, 576, 448, 3 - zone_id)
            self.assertIs(self.submit(current, [wrong, small, valid]), valid)
            self.assertEqual(current._latest_packet[2], zone_id)
            self.assertEqual(current._pending_events, [serial.build_event_packet(serial.EVENT_ZONE_FOUND)])

    def test_missing_wrong_color_and_tiny_zone_share_one_episode(self):
        current = self.current()
        event = serial.build_event_packet(serial.EVENT_ZONE_NO_TARGET)
        self.submit(current, [])
        self.assertEqual(current._next_outgoing_packet(1000), event)
        self.write(current, event)
        for index, raw in enumerate(([zone(w=5, h=5)], [zone(class_id=2)], [], [zone(w=5, h=5)])):
            now = 2000 + index * 10_000
            self.assertIsNone(self.submit(current, raw, now))
            self.assertIsNone(current._next_outgoing_packet(now))
            self.assertEqual(current._zone_no_target_phase, "sent")

    def test_queued_and_prepared_e6_cancel_on_fresh_valid_detection(self):
        for prepared in (False, True):
            current = self.current()
            self.submit(current, [])
            event = (current._next_outgoing_packet(1000) if prepared
                     else serial.build_event_packet(serial.EVENT_ZONE_NO_TARGET))
            valid = zone()
            self.assertIs(self.submit(current, [valid], 1010), valid)
            self.assertNotIn(event, current._pending_events)
            writer = self.write(current, event, 1010)
            writer.assert_not_called()
            self.assertEqual(current._next_outgoing_packet(1010), serial.build_event_packet(serial.EVENT_ZONE_FOUND))
            self.assertEqual(len(current._next_outgoing_packet(1010)), 11)

    def test_valid_zone_rearms_one_new_missing_event(self):
        current = self.current()
        event = serial.build_event_packet(serial.EVENT_ZONE_NO_TARGET)
        self.submit(current, [])
        self.write(current, current._next_outgoing_packet(1000))
        self.submit(current, [zone()], 1100)
        self.write(current, current._next_outgoing_packet(1100), 1100)
        self.submit(current, [zone(w=5, h=5)], 1200)
        self.assertEqual(current._next_outgoing_packet(1200), event)
        self.write(current, event, 1200)
        self.assertIsNone(current._next_outgoing_packet(1800))

    def test_new_small_or_missing_frame_cancels_prepared_old_coordinates(self):
        for raw in ([], [zone(w=5, h=5)]):
            current = self.current()
            self.submit(current, [zone()])
            self.write(current, current._next_outgoing_packet(1000))
            packet = current._next_outgoing_packet(1050)
            self.assertEqual(len(packet), 11)
            self.submit(current, raw, 1100)
            writer = self.write(current, packet, 1100)
            writer.assert_not_called()
            self.assertIsNone(current._latest_packet)
            self.assertEqual(current._next_outgoing_packet(1100),
                             serial.build_event_packet(serial.EVENT_ZONE_NO_TARGET))

    def test_uninterrupted_valid_coordinates_still_send_once_per_new_frame(self):
        current = self.current()
        obj = zone()
        self.submit(current, [obj])
        self.write(current, current._next_outgoing_packet(1000))
        for now in (1050, 1100, 1200):
            self.submit(current, [obj], now)
            packet = current._next_outgoing_packet(now)
            self.assertEqual(len(packet), 11)
            self.write(current, packet, now).assert_called_once_with(packet)
            self.assertIsNone(current._next_outgoing_packet(now))

    def test_stale_coordinate_packet_cannot_write_after_camera_stops(self):
        current = self.current()
        self.submit(current, [zone()])
        current._next_outgoing_packet(1000)
        packet = current._next_outgoing_packet(1000)
        writer = self.write(current, packet, 1001 + config.UART_TARGET_FRESHNESS_MS)
        writer.assert_not_called()

    def test_failed_and_short_e6_writes_retry_until_one_success(self):
        current = self.current()
        self.submit(current, [])
        writer = Mock(side_effect=(-1, 2, 5))
        event = serial.build_event_packet(serial.EVENT_ZONE_NO_TARGET)
        for index in range(3):
            packet = current._next_outgoing_packet(1000 + index)
            self.assertEqual(packet, event)
            self.write(current, packet, 1000 + index, writer)
        self.assertEqual(writer.call_count, 3)
        self.assertEqual(current._zone_no_target_phase, "sent")
        self.assertIsNone(current._next_outgoing_packet(2000))

    def test_mode_change_and_fe_reset_cancel_old_e6_and_clear_area_history(self):
        for command in (serial.CMD_RESET_TASK, serial.CMD_SEARCH, serial.CMD_TRACK):
            current = self.current()
            self.submit(current, [zone(w=50, h=100)])
            self.assertGreater(current._zone_area_ratio, 0)
            packet = current._next_outgoing_packet(1000)
            generation = current.frame_generation()
            current._handle_command(command)
            self.assertIsNone(current._zone_no_target_phase)
            self.assertEqual(current._zone_area_ratio, 0)
            self.write(current, packet, generation=generation).assert_not_called()

    def test_prepared_old_e6_cannot_interrupt_load_recheck(self):
        current = self.current()
        self.submit(current, [])
        packet = current._next_outgoing_packet(1000)
        generation = current.frame_generation()
        current._begin_zone_load_recheck()
        self.assertIsNone(current._zone_no_target_phase)
        self.assertEqual(current._zone_area_ratio, 0)
        self.write(current, packet, 1010, generation=generation).assert_not_called()
        self.assertEqual(current._next_outgoing_packet(1010),
                         serial.build_event_packet(serial.EVENT_TARGET_READY))

    def test_inventory_cargo_metadata_and_no_zone_event_scope_are_unchanged(self):
        for mode in (serial.CMD_SEARCH, serial.CMD_TRACK, serial.MODE_FINAL_ROI_TRACK,
                     serial.MODE_ARRANGE_RIGHTMOST, serial.MODE_ARRANGE_LEFTMOST,
                     serial.MODE_NEAR_VIEW_CHECK):
            current = controller(mode)
            self.submit(current, [])
            self.assertNotIn(serial.build_event_packet(serial.EVENT_ZONE_NO_TARGET), current._pending_events)
        current = self.current()
        current._zone_load_recheck = "wait06tx"
        current._zone_load_recheck_source = "before06"
        metadata = serial.build_coordinate_packet(5, 0, 0)
        self.write(current, metadata).assert_called_once_with(metadata)

    def test_26_keeps_original_half_image_boundary_and_coordinate_continuation(self):
        for zone_id in (1, 2):
            current = self.current(zone_id)
            obj = zone(0, 0, 576, 224, zone_id)
            self.assertIs(self.submit(current, [obj]), obj)
            self.assertEqual(current._next_outgoing_packet(1000), serial.build_event_packet(serial.EVENT_ZONE_FOUND))
            self.assertEqual(current._next_outgoing_packet(1000), serial.build_event_packet(serial.EVENT_ZONE_CLOSE))
            self.assertEqual(len(current._next_outgoing_packet(1000)), 11)
            self.submit(current, [obj], 1500)
            self.assertEqual(current._next_outgoing_packet(1500), serial.build_event_packet(serial.EVENT_ZONE_CLOSE))
            self.assertEqual(len(current._next_outgoing_packet(1500)), 11)

    def test_diagnostics_show_actual_small_area_and_e6_reason(self):
        for verbose in (False, True):
            current = self.current()
            with patch.object(config, "VISION_DIAGNOSTICS_ENABLED", True), \
                    patch.object(config, "VISION_DIAGNOSTICS_VERBOSE", verbose), patch("builtins.print") as log:
                self.submit(current, [zone(w=50, h=100)])
            messages = [str(call) for call in log.call_args_list]
            if verbose:
                self.assertTrue(any("ZONE_AREA_TOO_SMALL" in msg for msg in messages))
                self.assertTrue(any("min_area>0.02" in msg and "missingE6=pending" in msg for msg in messages))
            else:
                self.assertTrue(any("area=1.9%" in msg and "missingE6=pending" in msg for msg in messages))


if __name__ == "__main__":
    unittest.main()
