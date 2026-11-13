"""FE starts a new mission without restarting the camera or UART worker."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from test_vision_rules import LABELS, block, controller, SingleFrameTrackingScenario
from detection_verifier import DetectionVerifier
from tracking_roi import near_view_roi_vertices
import app_config as config
import serial_controller as s


class TaskResetTests(SingleFrameTrackingScenario):
    def setUp(self):
        quiet = patch.object(config, "VISION_DIAGNOSTICS_ENABLED", False)
        quiet.start()
        self.addCleanup(quiet.stop)

    def test_fe_restores_every_task_field_to_shared_initial_defaults(self):
        # Future task fields added to the common initializer are checked too.
        transport = {
            "frame_width", "frame_height", "tracking_roi", "near_view_roi", "_lock",
            "_stop_event", "_tx_wakeup", "_parser", "_serial", "_worker",
            "enabled", "_command_generation", "_outgoing_generation",
            "_task_reset_epoch", "_last_io_health_ms", "_diagnostics",
            "_claw_geometry", "_capture_event_packet",
        }
        defaults = controller(s.CMD_SEARCH)
        defaults.mode = s.MODE_WAIT_TASK_START
        defaults.target_label = "sqareredgreen"
        modes = (
            s.CMD_SORT, s.CMD_SEARCH, s.CMD_TRACK, s.MODE_WAIT_04_HANDSHAKE,
            s.MODE_WAIT_14_HANDSHAKE, s.MODE_FINAL_ROI_TRACK,
            s.MODE_ARRANGE_RIGHTMOST, s.MODE_ARRANGE_LEFTMOST,
            s.MODE_POST_ARRANGE_CHECK, s.MODE_NEAR_VIEW_CHECK,
            s.MODE_SEARCH_ZONE, s.MODE_WAIT_RED_SWITCH,
            s.MODE_WAIT_SUPPLY_SEARCH, s.MODE_WAIT_TASK_START,
            s.MODE_MICRO_FORWARD, s.MODE_MICRO_SIDE,
        )
        for mode in modes:
            with self.subTest(mode=mode):
                current = controller(mode)
                for name, value in vars(defaults).items():
                    if name in transport or name in ("mode", "target_label"):
                        continue
                    if isinstance(value, bool):
                        dirty = not value
                    elif isinstance(value, int):
                        dirty = value + 99
                    elif isinstance(value, list):
                        dirty = ["old"]
                    elif isinstance(value, str):
                        dirty = "old"
                    else:
                        dirty = object()
                    setattr(current, name, dirty)
                current.target_label = s.TARGET_BLACK_OR_GREEN
                current._diagnostics._last_ms = 900
                current._command_generation = 100
                current._outgoing_generation = 99
                current._handle_command(s.CMD_RESET_TASK)
                for name, expected in vars(defaults).items():
                    if name not in transport:
                        self.assertEqual(getattr(current, name), expected, name)
                self.assertEqual(vars(current._diagnostics), vars(defaults._diagnostics))
                self.assertEqual(current.frame_context(), (101, 1))
                # Do not relabel a previously prepared packet as generation101.
                self.assertEqual(current._outgoing_generation, 99)

    def test_fe_preserves_live_resources_and_configuration(self):
        current = controller(s.MODE_SEARCH_ZONE)
        serial, worker = Mock(), Mock()
        current._serial, current._worker = serial, worker
        current.enabled = True
        current.near_view_roi = near_view_roi_vertices(576, 448)
        names = ("_lock", "_parser", "_stop_event", "_tx_wakeup", "tracking_roi", "near_view_roi")
        resources = {name: getattr(current, name) for name in names}
        obstacle_enabled = config.UART_ZONE_OBSTACLE_ENABLED
        current._handle_command(s.CMD_RESET_TASK)
        self.assertIs(current._serial, serial)
        self.assertIs(current._worker, worker)
        self.assertTrue(current.enabled)
        for name, value in resources.items():
            self.assertIs(getattr(current, name), value)
        serial.close.assert_not_called()
        worker.join.assert_not_called()
        self.assertFalse(current._stop_event.is_set())
        self.assertTrue(current._tx_wakeup.is_set())
        self.assertEqual(config.UART_ZONE_OBSTACLE_ENABLED, obstacle_enabled)

    def test_fe_overrides_all_inventory_switch_and_collection_guards(self):
        for phase in (None, "wait05tx", "count05", "wait06tx", "wait24tx", "wide_settle"):
            with self.subTest(phase=phase):
                current = controller(s.MODE_NEAR_VIEW_CHECK)
                current._zone_load_recheck = phase
                current._zone_load_recheck_source = "before06"
                current._zone_load_resume_ready_ms = 99999
                current._red_switch_stage = "wait11"
                current._supply_collect_started_ms = 1
                current._supply_collect_delivery_pending = True
                current._track_green_recovery_waiting = True
                current.process_received_bytes(s.build_command_packet(s.CMD_RESET_TASK))
                self.assertEqual(current.mode, s.MODE_WAIT_TASK_START)
                self.assertIsNone(current._zone_load_recheck)
                self.assertIsNone(current._red_switch_stage)
                self.assertIsNone(current._supply_collect_started_ms)
                self.assertFalse(current._track_green_recovery_waiting)

    def test_wait_after_fe_is_silent_even_with_objects_timers_or_manual_events(self):
        current = controller(s.MODE_FINAL_ROI_TRACK)
        current._handle_command(s.CMD_RESET_TASK)
        objects = [block(288, 224), block(200, 200, 1)]
        for now in (1000, 16000, 100000):
            with patch.object(s, "_now_ms", return_value=now):
                current.send_event(s.EVENT_SEARCH_NO_TARGET)
                self.assertIsNone(current.submit(objects, LABELS, raw_objects=objects))
                self.assertIsNone(current._next_outgoing_packet(now))
        self.assertEqual(current._pending_events, [])
        self.assertEqual(current._search_no_target_frames, 0)
        self.assertEqual(current._latest_submit_sequence, 0)
        self.assertIsNone(current._latest_packet)
        serial = SimpleNamespace(write=Mock())
        with patch.object(current, "_serial", serial):
            current._write_packet(s.build_event_packet(s.EVENT_ZONE_CLOSE))
            current._write_packet(s.build_coordinate_packet(5, 0, 0))
        serial.write.assert_not_called()

    def test_stale_stage_handshakes_cannot_leave_fe_wait(self):
        current = controller(s.CMD_SEARCH)
        current._handle_command(s.CMD_RESET_TASK)
        generation = current.frame_generation()
        for command in (s.CMD_SORT, s.CMD_TRACK, s.CMD_ZONE_LOAD_RECHECK,
                        s.CMD_SEARCH_SWEEP_DONE, s.CMD_TRACK_CENTER_DONE,
                        s.CMD_TRACK_RECOVERY_DONE, s.CMD_FINAL_SEARCH_DONE,
                        s.CMD_ARRANGE_FIRST_DONE, s.CMD_ARRANGE_LEFT_DONE,
                        s.CMD_ARRANGE_SECOND_DONE, s.CMD_SEARCH_RED_ZONE,
                        s.CMD_SEARCH_BLUE_ZONE, s.CMD_ZONE_OBSTACLE_DONE,
                        s.CMD_RECHECK_SORT, 0x99):
            current._handle_command(command)
            self.assertEqual(current.mode, s.MODE_WAIT_TASK_START)
            self.assertEqual(current.frame_generation(), generation)
            self.assertEqual(current._pending_events, [])

    def test_target_selection_waits_for_new03_then_keeps_normal_task_behavior(self):
        for command, label in s.TARGET_SELECTION_COMMANDS.items():
            with self.subTest(command=command):
                current = controller(s.CMD_SEARCH)
                current._handle_command(s.CMD_RESET_TASK)
                current._handle_command(command)
                self.assertEqual(current.mode, s.MODE_WAIT_TASK_START)
                self.assertEqual(current.target_label, label)
                self.assertIsNone(current._next_outgoing_packet(1000))
                current._handle_command(s.CMD_SEARCH)
                expected = s.MODE_FINAL_ROI_TRACK if command == s.CMD_SELECT_BLACK_OBJECT else s.CMD_SEARCH
                self.assertEqual(current.mode, expected)
                self.assertEqual(current._post07_started, command in (
                    s.CMD_SELECT_RED_OBJECT, s.CMD_SELECT_BLACK_OBJECT,
                    s.CMD_SELECT_BLACK_OR_GREEN,
                ))

    def test_fe_then03_without21_uses_fresh_first_green_default(self):
        current = controller(s.MODE_SEARCH_ZONE)
        current._post07_started = True
        current.target_label = "sqarered"
        current.process_received_bytes(s.build_command_packet(s.CMD_RESET_TASK)
                                       + s.build_command_packet(s.CMD_SEARCH))
        self.assertEqual(current.mode, s.CMD_SEARCH)
        self.assertTrue(current._uses_first_green_center_selection(current.target_label))

    def test_fragmented_fe_and_following_selection_are_not_dropped(self):
        current = controller(s.MODE_SEARCH_ZONE)
        current._post07_started = True
        reset = s.build_command_packet(s.CMD_RESET_TASK)
        green = s.build_command_packet(s.CMD_SELECT_GREEN_OBJECT)
        search = s.build_command_packet(s.CMD_SEARCH)
        self.assertEqual(current.process_received_bytes(b"noise" + reset[:3]), [])
        self.assertEqual(current.process_received_bytes(reset[3:] + green[:2]), [s.CMD_RESET_TASK])
        self.assertEqual(current.mode, s.MODE_WAIT_TASK_START)
        self.assertEqual(current.process_received_bytes(green[2:] + search[:4]), [s.CMD_SELECT_GREEN_OBJECT])
        self.assertEqual(current.mode, s.MODE_WAIT_TASK_START)
        self.assertEqual(current.process_received_bytes(search[4:]), [s.CMD_SEARCH])
        self.assertFalse(current._post07_started)
        self.assertTrue(current._uses_first_green_center_selection(current.target_label))

    def test_duplicate_fe_is_harmless_and_never_replies_or_restarts_uart(self):
        current = controller(s.MODE_SEARCH_ZONE)
        current._command_generation = 77
        for index in range(3):
            current.process_received_bytes(s.build_command_packet(s.CMD_RESET_TASK))
            self.assertEqual(current.frame_context(), (78 + index, 1 + index))
            self.assertEqual(current._pending_events, [])
            self.assertIsNone(current._next_outgoing_packet(1000 + index * 500))
            self.assertFalse(current._post07_started)

    def test_old_frames_and_prepared_packets_stay_invalid_after_new03(self):
        current = controller(s.CMD_SEARCH)
        current._command_generation = 100
        generation = current.frame_generation()
        current._handle_command(s.CMD_RESET_TASK)
        current._handle_command(s.CMD_SELECT_GREEN_OBJECT)
        current._handle_command(s.CMD_SEARCH)
        target = block(300, 200)
        self.assertIsNone(current.submit([target], LABELS, raw_objects=[target], frame_generation=generation))
        self.assertIsNone(current._latest_packet)
        serial = SimpleNamespace(write=Mock(return_value=5))
        with patch.object(current, "_serial", serial):
            for event in s.EVENT_NAMES:
                current._write_packet(s.build_event_packet(event), generation=generation)
            current._write_packet(s.build_coordinate_packet(5, 12, -24), generation=generation)
        serial.write.assert_not_called()
        self.assertIs(current.submit([target], LABELS, raw_objects=[target],
                                    frame_generation=current.frame_generation()), target)
        self.assertEqual(current._latest_packet, s.build_coordinate_packet(5, 12, -24))

    def test_fe_during_inference_discards_the_old_selection(self):
        current = controller(s.CMD_TRACK)
        target = block(288, 224)
        current._track_target_lock = current._final_target_identity(target)
        generation = current.frame_generation()

        def reset_while_selecting(*args, **kwargs):
            current._handle_command(s.CMD_RESET_TASK)
            return target

        with patch.object(current, "_select_locked_target", side_effect=reset_while_selecting):
            self.assertIsNone(current.submit([target], LABELS, raw_objects=[target], frame_generation=generation))
        self.assertEqual(current.mode, s.MODE_WAIT_TASK_START)
        self.assertIsNone(current._latest_packet)
        self.assertIsNone(current._track_target_lock)
        self.assertEqual(current._pending_events, [])

    def _first_green_decision_after_old_post07(self, crowded):
        current = controller(s.CMD_SEARCH)
        current._handle_command(s.CMD_SELECT_BLACK_OR_GREEN)
        current._first_green_near_view_decided = True
        current._near_view_green_missing_cycles = 1
        current.process_received_bytes(s.build_command_packet(s.CMD_RESET_TASK)
                                       + s.build_command_packet(s.CMD_SELECT_GREEN_OBJECT)
                                       + s.build_command_packet(s.CMD_SEARCH))
        target = block(288, 224)
        objects = [target, block(300, 224, 3)] if crowded else [target]
        with patch.object(s, "_now_ms", return_value=1000):
            current.submit(objects, LABELS, raw_objects=objects)
            self.assertEqual(current._next_outgoing_packet(1000), s.build_event_packet(s.EVENT_SEARCH_CENTER_REACHED))
            current._handle_command(s.CMD_TRACK)
            current.submit(objects, LABELS, raw_objects=objects)
            self.assertEqual(current._next_outgoing_packet(1020), s.build_event_packet(s.EVENT_TRACK_CENTER_REACHED))
            current._handle_command(s.CMD_TRACK_CENTER_DONE)
        for index in range(config.UART_FIRST_GREEN_DECISION_CONFIRM_FRAMES):
            now = 1040 + index * 10
            with patch.object(s, "_now_ms", return_value=now):
                current.submit(objects, LABELS, raw_objects=objects)
                packet = current._next_outgoing_packet(now)
            if index < config.UART_FIRST_GREEN_DECISION_CONFIRM_FRAMES - 1:
                self.assertTrue(packet is None or len(packet) == 11)
            else:
                event = s.EVENT_ROI_READY if crowded else s.EVENT_SKIP_TO_FINAL_TRACK
                self.assertEqual(packet, s.build_event_packet(event))
        self.assertFalse(current._post07_started)
        self.assertFalse(current._post07_supply_tracking)
        self.assertEqual(current._near_view_green_missing_cycles, 0)

    def test_reset_corrects_reported_first_green_crowding_bug(self):
        self._first_green_decision_after_old_post07(crowded=True)

    def test_reset_still_allows_clear_first_green_to_enter24(self):
        self._first_green_decision_after_old_post07(crowded=False)

    def test_only_fe_changes_verifier_reset_epoch(self):
        current = controller(s.CMD_SEARCH)
        self.assertEqual(current.frame_context(), (0, 0))
        current._handle_command(s.CMD_SELECT_RED_OBJECT)
        current._handle_command(s.CMD_SEARCH)
        self.assertEqual(current.frame_context()[1], 0)
        current._handle_command(s.CMD_RESET_TASK)
        self.assertEqual(current.frame_context(), (3, 1))
        current._handle_command(s.CMD_SELECT_GREEN_OBJECT)
        current._handle_command(s.CMD_SEARCH)
        self.assertEqual(current.frame_context(), (5, 1))

    def test_verifier_reset_drops_old_confirmation_but_preserves_capabilities(self):
        verifier = DetectionVerifier(576, 448)
        target = block(288, 200)
        with patch.object(config, "VERIFY_ENABLED", True), \
                patch.object(config, "VERIFY_REQUIRED_HITS", 3), \
                patch.object(config, "VERIFY_REQUIRED_HITS_WITHOUT_COLOR", 3), \
                patch.object(verifier, "_validate_candidate", return_value=(True, {})):
            for _ in range(3):
                verified = verifier.process(None, [target], LABELS)
            self.assertEqual(verified, [target])
            verifier._color_available = False
            verifier._color_error_reported = True
            verifier.reset_task_history()
            self.assertEqual(verifier._class_states, {})
            self.assertEqual(verifier._frame_index, 0)
            self.assertEqual(verifier.frame_diagnostics(), [])
            self.assertFalse(verifier._color_available)
            self.assertTrue(verifier._color_error_reported)
            self.assertEqual(verifier.process(None, [target], LABELS), [])


if __name__ == "__main__":
    unittest.main()
