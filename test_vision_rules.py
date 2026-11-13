"""Host-side regression checks for the agreed MaixCAM tracking behavior.

Run with ``python -m unittest -v test_vision_rules``. No camera or UART is
opened: the test supplies a tiny ``maix`` import stub and disables UART.
"""

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch


if "maix" not in sys.modules:
    maix_stub = types.ModuleType("maix")
    maix_stub.comm = types.ModuleType("comm")
    maix_stub.uart = types.ModuleType("uart")
    maix_stub.time = SimpleNamespace(ticks_us=lambda: 0)
    maix_stub.image = SimpleNamespace(
        COLOR_BLUE="blue", COLOR_RED="red", COLOR_GREEN="green", COLOR_BLACK="black",
        Color=SimpleNamespace(from_rgb=lambda r, g, b: (r, g, b)),
        Format=SimpleNamespace(FMT_RGB888=0, FMT_BGR888=1, FMT_GRAYSCALE=2),
    )
    sys.modules["maix"] = maix_stub

import app_config as config
import serial_controller as serial_module
from debug_target_preview import DebugTargetPreview
from detection_verifier import DetectionVerifier
from performance_debug import PerformanceProfiler
import performance_debug as performance_module
from vision_overlay import (
    draw_debug_yolo_detections, draw_debug_zone_obstacle_region,
    draw_debug_safety_zone_area, draw_roi_border, make_bottom_center_roi,
)
from zone_obstacle_region import zone_obstacle_polygon, point_in_zone_obstacle_polygon
from serial_controller import (
    CMD_TRACK,
    EVENT_ARRANGEMENT_NO_OBJECT,
    EVENT_ROI_READY,
    EVENT_SKIP_TO_FINAL_TRACK,
    EVENT_TRACK_CENTER_REACHED,
    MODE_ARRANGE_RIGHTMOST,
    MODE_FINAL_ROI_TRACK,
    MODE_POST_ARRANGE_CHECK,
    MODE_WAIT_14_HANDSHAKE,
    VisionSerialController,
    build_event_packet,
)


LABELS = [
    "blue safety", "bluesafety", "redsafety", "sqareblue",
    "sqarered", "sqareredgreen", "triangualrblack",
]


class SingleFrameTrackingScenario(unittest.TestCase):
    """Isolate legacy geometry/protocol assertions from new temporal evidence.

    These scenarios intentionally supply one observation per scene. Run them
    with one-frame cargo/zone acquisition, legacy arrangement and rectangle,
    and no zone hold, without changing assertions. test_tracking_roi covers
    real default-triangle geometry and tracking/inventory transitions.
    test_target_acquisition_guard separately exercises the real timing defaults
    and integrated state transitions with distinct fresh camera observations.
    """
    def run(self, result=None):
        with patch.object(config, "UART_NEW_TARGET_CONFIRM_FRAMES", 1), \
                patch.object(config, "UART_POST07_SUPPLY_CONFIRM_FRAMES", 1), \
                patch.object(config, "UART_TARGET_ZONE_HOLD_MS", 0), \
                patch.object(config, "UART_ZONE_CLOSE_CONFIRM_FRAMES", 1), \
                patch.object(config, "UART_DIRECT_CAPTURE_AFTER_14", False), \
                patch.object(config, "UART_TRACKING_ROI_SHAPE", "rectangle"):
            return super().run(result)


def block(center_x, center_y, class_id=5):
    return SimpleNamespace(
        x=center_x - 10,
        y=center_y - 10,
        w=20,
        h=20,
        class_id=class_id,
        score=0.9,
    )


def near_view_block(center_x, center_y, class_id=5):
    """Eligible05 fixture: red25%, black>3%; generic tracking boxes stay20px."""
    obj = block(center_x, center_y, class_id)
    if class_id == 4:
        obj.x = max(0, min(288, center_x - 144))
        obj.y = max(0, min(224, center_y - 112))
        obj.w, obj.h = 288, 224
    elif class_id == 6:
        obj.x = max(0, min(480, center_x - 48))
        obj.y = max(0, min(364, center_y - 42))
        obj.w, obj.h = 96, 84
    return obj


def controller(mode):
    with patch.object(config, "UART_COORDINATE_ENABLED", False):
        result = VisionSerialController(
            576, 448, tracking_roi=[0, 359, 576, 89]
        )
    result.mode = mode
    result.target_label = "sqareredgreen"
    return result


def write_near_view_request(current, now):
    """Actually write05; merely dequeuing it no longer starts an inventory."""
    packet = current._next_outgoing_packet(now)
    if packet != build_event_packet(serial_module.EVENT_TARGET_READY):
        raise AssertionError("expected a pending05 camera request")
    serial = SimpleNamespace(write=Mock(side_effect=lambda data: len(data)))
    with patch.object(current, "_serial", serial), \
            patch.object(serial_module, "_now_ms", return_value=now):
        current._write_packet(packet, current._outgoing_generation)
    serial.write.assert_called_once_with(packet)
    return packet


def confirm_final_load_check(current, ids, now, settle_ms=500):
    """Drive the newly required05 write and fresh confirmation, not a bypass.

    Old flow tests can compress only this additional delay; dedicated final
    inventory tests exercise actual writes, settling and generation boundaries.
    Returns the time the metadata/result is ready, without consuming them.
    """
    if current.mode != serial_module.MODE_NEAR_VIEW_CHECK:
        raise AssertionError("final inventory must precede06")
    if current._zone_load_recheck_source != "before06":
        raise AssertionError("missing final inventory source")
    if current._zone_load_recheck != "wait05tx":
        raise AssertionError("must wait for actual final05 write")
    sent_at = max(now, (current._last_event_write_ms + config.UART_EVENT_MIN_INTERVAL_MS)
                  if current._last_event_write_ms is not None else now)
    reply = current._next_outgoing_packet(sent_at)
    if reply != build_event_packet(serial_module.EVENT_TARGET_READY):
        raise AssertionError("expected the final05 camera request")
    serial = SimpleNamespace(write=Mock(side_effect=lambda packet: len(packet)))
    with patch.object(current, "_serial", serial), \
            patch.object(serial_module, "_now_ms", return_value=sent_at):
        current._write_packet(reply, current._outgoing_generation)
    objects = [near_view_block(120 + index * 40, 200, class_id) for index, class_id in enumerate(ids)]
    decision_at = sent_at + settle_ms + 20
    with patch.object(config, "UART_NEAR_VIEW_SETTLE_MS", settle_ms):
        for index in range(3):
            with patch.object(serial_module, "_now_ms", return_value=sent_at + settle_ms + index * 10):
                current.submit(objects, LABELS, raw_objects=objects,
                               frame_generation=current.frame_generation())
    return decision_at


class VisionRuleRegressionTests(SingleFrameTrackingScenario):
    def setUp(self):
        # Existing behavioral tests need not print diagnostic snapshots.
        quiet = patch.object(config, "VISION_DIAGNOSTICS_ENABLED", False)
        quiet.start()
        self.addCleanup(quiet.stop)
        # Detailed diagnostic tests explicitly exercise the opt-in mode.
        for name, value in (("VISION_DIAGNOSTICS_VERBOSE", True),
                            ("VISION_DIAGNOSTICS_INTERVAL_MS", 500),
                            ("VISION_COORDINATE_LOG_ENABLED", True),
                            ("VISION_COORDINATE_LOG_INTERVAL_MS", 500)):
            setting = patch.object(config, name, value)
            setting.start()
            self.addCleanup(setting.stop)

    def test_default_roi_height_reduced20_and_keeps_bottom_and_width(self):
        roi = make_bottom_center_roi(576, 448)
        self.assertEqual(roi, [0, 319, 576, 129])
        self.assertEqual(448 // 5 + 60 - roi[3], 20)
        self.assertEqual(roi[1] + roi[3], 448)
        self.assertEqual(make_bottom_center_roi(20, 10), [0, 0, 20, 10])

    def test_enlarged_roi_display_reference_and_obstacle_share_geometry(self):
        roi = make_bottom_center_roi(576, 448)
        img = Mock()
        draw_roi_border(img, roi)
        self.assertEqual(img.draw_rect.call_args.args, tuple(roi))
        current = controller(MODE_FINAL_ROI_TRACK)
        current.tracking_roi = roi
        self.assertEqual(current._reference_point(MODE_FINAL_ROI_TRACK), (288.0, 363.5))
        for mode in (MODE_WAIT_14_HANDSHAKE, serial_module.MODE_SEARCH_ZONE):
            self.assertEqual(current._reference_point(mode), (288.0, 383.5))
        for mode in (serial_module.CMD_SEARCH, CMD_TRACK):
            self.assertEqual(current._reference_point(mode), (288.0, 224.0))
        self.assertEqual(zone_obstacle_polygon(roi, 576, 448), (
            (238.0 - 238.0 * 100.0 / 239.0, 200.0),
            (338.0 + 237.0 * 100.0 / 239.0, 200.0),
            (575.0, 339.0), (0.0, 339.0),
        ))

    def test_rectangle_roi_34_starts_on_entry_even_above_offset_reference(self):
        for y, inside in ((318.99, False), (319, True), (363.49, True), (363.5, True),
                          (363.51, True), (447, True)):
            with self.subTest(y=y):
                current = controller(MODE_FINAL_ROI_TRACK)
                current.tracking_roi = make_bottom_center_roi(576, 448)
                target = block(288, y)
                with patch.object(serial_module, "_now_ms", return_value=1000):
                    self.assertIs(current.submit([target], LABELS), target)
                self.assertEqual(current._final_roi_sent, inside)
                if inside:
                    self.assertIsNone(current._latest_packet)
                    self.assertEqual(current._next_outgoing_packet(1000),
                                     build_event_packet(serial_module.EVENT_FINAL_TARGET_IN_ROI))
                else:
                    self.assertIsNotNone(current._latest_packet)

    def test_compact_summary_rate_limited_despite_detection_flicker(self):
        current = controller(serial_module.CMD_SEARCH)
        target = block(100, 150)
        with patch.object(config, "VISION_DIAGNOSTICS_ENABLED", True), \
                patch.object(config, "VISION_DIAGNOSTICS_VERBOSE", False), \
                patch.object(config, "VISION_DIAGNOSTICS_INTERVAL_MS", 2000), \
                patch("builtins.print") as log:
            for index in range(20):
                objects = [target] if index % 2 == 0 else []
                with patch.object(serial_module, "_now_ms", return_value=1000 + index * 90):
                    current.submit(objects, LABELS, raw_objects=objects)
            with patch.object(serial_module, "_now_ms", return_value=3000):
                current.submit([target], LABELS)
            messages = [str(call.args[0]) for call in log.call_args_list]
            self.assertEqual(sum("[VISION]" in text for text in messages), 2)
            self.assertFalse(any(any(tag in text for tag in (
                "[RAW]", "[VOBJ]", "[CANDIDATE]", "[SELECT]", "[TX-COORD]")) for text in messages))

    def test_compact_tracking_flicker_has_no_per_frame_lock_loss_logs(self):
        for mode in (serial_module.CMD_SEARCH, CMD_TRACK, MODE_FINAL_ROI_TRACK):
            with self.subTest(mode=mode):
                current = controller(mode)
                target = block(100, 150)
                with patch.object(config, "VISION_DIAGNOSTICS_ENABLED", True), \
                        patch.object(config, "VISION_DIAGNOSTICS_VERBOSE", False), \
                        patch.object(config, "VISION_DIAGNOSTICS_INTERVAL_MS", 5000), \
                        patch("builtins.print") as log:
                    for index in range(40):
                        objects = [target] if index % 2 == 0 else []
                        with patch.object(serial_module, "_now_ms", return_value=1000 + index * 80):
                            current.submit(objects, LABELS, raw_objects=objects)
                    messages = [str(call.args[0]) for call in log.call_args_list]
                    self.assertEqual(sum("[VISION]" in message for message in messages), 1)
                    self.assertFalse(any(any(tag in message for tag in (
                        "[TRACK-04]", "[FINAL-24]", "[SEARCH-03]")) for message in messages))

    def test_compact_diagnostics_returns_before_building_unused_snapshot(self):
        current = controller(CMD_TRACK)
        target = block(100, 150)
        with patch.object(config, "VISION_DIAGNOSTICS_ENABLED", True), \
                patch.object(config, "VISION_DIAGNOSTICS_VERBOSE", False), \
                patch.object(config, "VISION_DIAGNOSTICS_INTERVAL_MS", 5000), \
                patch.object(serial_module, "_now_ms", return_value=1000), \
                patch("builtins.print"):
            current.submit([target], LABELS)
            # Iterating events or reading the diagnostic anchor would fail;
            # neither is needed by a throttled report.
            current._pending_events = None
            current._diagnostics.report(
                current, LABELS, [target], [target], [target], target,
                None, CMD_TRACK, "track_lock", None, None, False, [],
                1100, "unused_anchor_must_not_be_read",
            )

    def test_compact_and_verbose_tracking_have_identical_state_and_uart_traces(self):
        for mode in (serial_module.CMD_SEARCH, CMD_TRACK, MODE_FINAL_ROI_TRACK):
            histories = []
            target, moved = block(100, 150), block(120, 180)
            frames = [[target]] + [[]] * 6 + [[moved]] + [[]] * 10 + [[target]]
            for verbose in (False, True):
                current = controller(mode)
                history = []
                with patch.object(config, "VISION_DIAGNOSTICS_ENABLED", True), \
                        patch.object(config, "VISION_DIAGNOSTICS_VERBOSE", verbose), \
                        patch("builtins.print"):
                    for index, objects in enumerate(frames):
                        now = 1000 + index * 30
                        with patch.object(serial_module, "_now_ms", return_value=now):
                            selected = current.submit(objects, LABELS, raw_objects=objects)
                            packet = current._next_outgoing_packet(now)
                        history.append((current.mode, getattr(selected, "class_id", None),
                                        packet, current._latest_packet,
                                        current._search_target_lock, current._track_target_lock,
                                        current._final_target_lock, current._search_no_target_frames,
                                        current._track_no_target_frames, current._final_no_target_frames,
                                        current._final_search_waiting_ack))
                histories.append(history)
            with self.subTest(mode=mode):
                self.assertEqual(histories[0], histories[1])

    def test_non_arrangement_overlay_skips_safety_filter_work(self):
        for mode in (serial_module.CMD_SEARCH, CMD_TRACK, MODE_FINAL_ROI_TRACK,
                     serial_module.MODE_SEARCH_ZONE, serial_module.MODE_ARRANGE_LEFTMOST):
            with self.subTest(mode=mode):
                current = controller(mode)
                with patch.object(current, "_exclude_objects_in_safety_zones",
                                  side_effect=AssertionError("unused overlay filtering")):
                    self.assertEqual(current.arrangement_side_objects([block(100, 150)], LABELS), [])

    def test_compact_debug_profiler_skips_stage_clocks_but_reports_total_fps(self):
        with patch.object(config, "VISION_DIAGNOSTICS_VERBOSE", False), \
                patch.object(config, "DEBUG_REPORT_INTERVAL_US", 5_000_000), \
                patch.object(performance_module.time, "ticks_us", return_value=1000) as clock, \
                patch("builtins.print") as log:
            profiler = PerformanceProfiler()
            profiler.set_enabled(True)
            start = profiler.begin_frame()
            clock.reset_mock()
            stage_start = start
            for stage in profiler.STAGES:
                stage_start = profiler.mark(stage, stage_start)
            clock.assert_not_called()
            self.assertEqual(stage_start, start)
            clock.return_value = 5_001_000
            profiler.end_frame(start)
            self.assertTrue(any("[DEBUG] total=" in str(call) for call in log.call_args_list))

    def test_verbose_debug_profiler_retains_stage_timings(self):
        with patch.object(config, "VISION_DIAGNOSTICS_VERBOSE", True), \
                patch.object(performance_module.time, "ticks_us", return_value=1000) as clock, \
                patch("builtins.print"):
            profiler = PerformanceProfiler()
            profiler.set_enabled(True)
            start = profiler.begin_frame()
            clock.reset_mock()
            clock.return_value = 2000
            self.assertEqual(profiler.mark("camera", start), 2000)
            clock.assert_called_once()
            self.assertEqual(profiler._stage_us["camera"], 1000)

    def test_high_confidence_color_fast_path_keeps_temporal_confirmation(self):
        targets = [block(100 + 80 * index, 150, class_id)
                   for index, class_id in enumerate((3, 4, 5, 6))]
        for target in targets:
            target.score = config.VERIFY_HIGH_CONFIDENCE_BYPASS
        img = SimpleNamespace(find_blobs=Mock(side_effect=AssertionError("redundant color scan")))
        verifier = DetectionVerifier(576, 448)
        with patch.object(config, "VERIFY_COLOR_ENABLED", True), \
                patch.object(config, "VISION_DIAGNOSTICS_VERBOSE", False), \
                patch.object(config, "VERIFY_SKIP_REDUNDANT_COLOR_ON_HIGH_CONFIDENCE", True), \
                patch("builtins.print"):
            for _ in range(config.VERIFY_REQUIRED_HITS - 1):
                self.assertEqual(verifier.process(img, targets, LABELS), [])
            self.assertEqual(verifier.process(img, targets, LABELS), targets)
        img.find_blobs.assert_not_called()
        self.assertTrue(verifier._color_available)

    def test_high_confidence_color_fast_path_never_approves_weak_neighbor(self):
        strong, weak = block(100, 150), block(300, 150)
        weak.score = .6
        img = SimpleNamespace(find_blobs=Mock(return_value=[]))
        verifier = DetectionVerifier(576, 448)
        with patch.object(config, "VERIFY_COLOR_ENABLED", True), \
                patch.object(config, "VISION_DIAGNOSTICS_VERBOSE", False), \
                patch.object(config, "VERIFY_SKIP_REDUNDANT_COLOR_ON_HIGH_CONFIDENCE", True):
            for _ in range(config.VERIFY_REQUIRED_HITS):
                verified = verifier.process(img, [strong, weak], LABELS)
            self.assertEqual(verified, [strong])
            self.assertEqual(img.find_blobs.call_count, config.VERIFY_REQUIRED_HITS)

    def test_color_fast_and_full_paths_preserve_verification_history_and_objects(self):
        strong, weak = block(100, 150), block(300, 150)
        weak.score = .6
        blob = SimpleNamespace(pixels=lambda: 80, code=lambda: 1, density=lambda: .5)
        frames = [[strong, weak]] * 4 + [[]] * 2 + [[strong, weak]] * 3
        for color_passes in (False, True):
            histories, calls, states = [], [], []
            for fast in (False, True):
                verifier = DetectionVerifier(576, 448)
                img = SimpleNamespace(find_blobs=Mock(return_value=[blob] if color_passes else []))
                with patch.object(config, "VERIFY_COLOR_ENABLED", True), \
                        patch.object(config, "VISION_DIAGNOSTICS_VERBOSE", False), \
                        patch.object(config, "VERIFY_SKIP_REDUNDANT_COLOR_ON_HIGH_CONFIDENCE", fast):
                    histories.append([verifier.process(img, objects, LABELS) for objects in frames])
                states.append(verifier._class_states)
                calls.append(img.find_blobs.call_count)
            with self.subTest(color_passes=color_passes):
                self.assertEqual(histories[0], histories[1])
                self.assertEqual(states[0], states[1])
                self.assertLess(calls[1], calls[0])

    def test_verbose_verification_still_collects_high_confidence_color_evidence(self):
        verifier = DetectionVerifier(576, 448)
        target = block(100, 150)
        img = SimpleNamespace(find_blobs=Mock(return_value=[]))
        with patch.object(config, "VERIFY_COLOR_ENABLED", True), \
                patch.object(config, "VISION_DIAGNOSTICS_ENABLED", True), \
                patch.object(config, "VISION_DIAGNOSTICS_VERBOSE", True), \
                patch.object(config, "VERIFY_SKIP_REDUNDANT_COLOR_ON_HIGH_CONFIDENCE", True):
            for _ in range(config.VERIFY_REQUIRED_HITS):
                verified = verifier.process(img, [target], LABELS)
            self.assertEqual(verified, [target])
            self.assertEqual(img.find_blobs.call_count, config.VERIFY_REQUIRED_HITS)
            self.assertEqual(verifier.frame_diagnostics()[0]["reason"], "color_shape_rejected")
            self.assertTrue(verifier.frame_diagnostics()[0]["high_conf_bypass"])

    def test_coordinates_not_logged_but_every_uart_packet_is_written(self):
        current = controller(CMD_TRACK)
        current._serial = SimpleNamespace(write=Mock(side_effect=lambda packet: len(packet)))
        packets = [serial_module.build_coordinate_packet(5, index, -index) for index in range(40)]
        with patch.object(config, "VISION_DIAGNOSTICS_ENABLED", True), \
                patch.object(config, "VISION_COORDINATE_LOG_ENABLED", False), \
                patch.object(serial_module, "_now_ms", return_value=1000), \
                patch("builtins.print") as log:
            for packet in packets:
                current._write_packet(packet)
            self.assertEqual(current._serial.write.call_count, 40)
            self.assertEqual(current._tx_coordinate_count, 40)
            self.assertEqual([call.args[0] for call in current._serial.write.call_args_list], packets)
            log.assert_not_called()

    def test_compact_command_and_error_logs_retained_without_duplicate_tx_event(self):
        current = controller(CMD_TRACK)
        current._serial = SimpleNamespace(write=lambda packet: len(packet))
        with patch.object(config, "VISION_DIAGNOSTICS_ENABLED", True), \
                patch.object(config, "VISION_DIAGNOSTICS_VERBOSE", False), \
                patch.object(serial_module, "_now_ms", return_value=1000), \
                patch("builtins.print") as log:
            current.process_received_bytes(bytes((0xE1, 0xE2, 0x03, 0x1E, 0x2E)))
            current._write_packet(build_event_packet(EVENT_TRACK_CENTER_REACHED))
            current._serial.write = lambda packet: -1
            current._write_packet(serial_module.build_coordinate_packet(5, 0, 0))
            messages = [str(call.args[0]) for call in log.call_args_list]
            self.assertTrue(any("RX command 0x03" in text for text in messages))
            self.assertTrue(any("TX command 0x14" in text for text in messages))
            self.assertTrue(any("write failed" in text for text in messages))
            self.assertFalse(any("[TX-EVENT]" in text or "[RX-CONTEXT]" in text for text in messages))

    def test_compact_verifier_skips_evidence_dump_without_changing_results(self):
        histories = []
        target = block(100, 150)
        with patch.object(config, "VISION_DIAGNOSTICS_ENABLED", True):
            for verbose in (False, True):
                verifier = DetectionVerifier(576, 448)
                metrics = {"ok": True, "reason": "color_shape_pass", "red": 0,
                           "green": .8, "blue": 0, "black": 0, "density": .8}
                with patch.object(config, "VISION_DIAGNOSTICS_VERBOSE", verbose), \
                        patch.object(verifier, "_validate_candidate", return_value=(True, metrics.copy())):
                    histories.append([verifier.process(None, [target], LABELS)
                                      for _ in range(config.VERIFY_REQUIRED_HITS)])
                    self.assertEqual(bool(verifier.frame_diagnostics()), verbose)
            self.assertEqual(histories[0], histories[1])

    def test_diagnostics_explain_screening_and_throttle_without_changing_selection(self):
        current = controller(CMD_TRACK)
        current.target_label = "sqarered"
        target, neighbor = block(100, 150, 4), block(150, 150, 3)
        with patch.object(config, "VISION_DIAGNOSTICS_ENABLED", True), \
                patch.object(serial_module, "_now_ms", return_value=1000), \
                patch("builtins.print") as log:
            self.assertIsNone(current.submit([target, neighbor], LABELS))
            messages = [str(call.args[0]) for call in log.call_args_list]
            self.assertTrue(any("[CANDIDATE]" in text and "dx=50.0<=120" in text
                                and "applied=yes" in text for text in messages))
            self.assertTrue(any("NO_ELIGIBLE_SELECTION" in text for text in messages))
            frame_logs = sum("[FRAME]" in str(call.args[0]) for call in log.call_args_list)
            current.submit([target, neighbor], LABELS)
            self.assertEqual(sum("[FRAME]" in str(call.args[0]) for call in log.call_args_list), frame_logs)
        with patch.object(config, "VISION_DIAGNOSTICS_ENABLED", True), \
                patch.object(serial_module, "_now_ms", return_value=1500), \
                patch("builtins.print") as log:
            current.submit([target, neighbor], LABELS)
            self.assertTrue(any("[FRAME]" in str(call.args[0]) for call in log.call_args_list))

    def test_diagnostics_show_verification_vs_safety_exclusion_and_inherited_lock(self):
        current = controller(serial_module.CMD_SEARCH)
        current.target_label = "sqarered"
        target, zone = block(288, 224, 4), block(288, 224, 2)
        with patch.object(config, "VISION_DIAGNOSTICS_ENABLED", True), \
                patch.object(serial_module, "_now_ms", return_value=1000), \
                patch("builtins.print") as log:
            current.submit([], LABELS, raw_objects=[target])
            self.assertTrue(any("reason=VERIFICATION_REJECTED" in str(call) for call in log.call_args_list))
            current._diagnostics._last_signature = None
            current.submit([target], LABELS, raw_objects=[target, zone])
            self.assertTrue(any("reason=SAFE_ZONE_EXCLUDED" in str(call) for call in log.call_args_list))
            self.assertTrue(any("[SAFE-REASON]" in str(call) and "id2" in str(call) for call in log.call_args_list))
            current.submit([target], LABELS, raw_objects=[target])
            self.assertTrue(any("lock_after=(4, 288.0, 224.0, 400.0)" in str(call) for call in log.call_args_list))

    def test_actual_coordinate_tx_logs_signed_data_and_counts_only_successful_writes(self):
        current = controller(CMD_TRACK)
        packet = serial_module.build_coordinate_packet(4, -134, -130)
        current._coordinate_in_flight = (packet, 7, 1000, CMD_TRACK)
        current._serial = SimpleNamespace(write=lambda data: len(data))
        with patch.object(config, "VISION_DIAGNOSTICS_ENABLED", True), \
                patch.object(serial_module, "_now_ms", return_value=1010), \
                patch("builtins.print") as log:
            current._write_packet(packet)
            self.assertEqual(current._tx_coordinate_count, 1)
            self.assertTrue(any("[TX-COORD]" in str(call) and "f=7" in str(call)
                                and "dx=-134 dy=-130" in str(call) and "age_ms=10" in str(call)
                                for call in log.call_args_list))
            current._write_packet(packet)
            self.assertEqual(current._tx_coordinate_count, 2)
            self.assertEqual(sum("[TX-COORD]" in str(call) for call in log.call_args_list), 1)
            current._serial.write = lambda data: 2
            current._write_packet(packet)
            self.assertEqual(current._tx_coordinate_count, 2)

    def test_verifier_diagnostics_report_temporal_confirmation_color_and_class_limit(self):
        verifier = DetectionVerifier(576, 448)
        target, weak, zone = block(100, 150, 4), block(400, 150, 4), block(100, 150, 2)
        weak.score = .6
        metrics = {"ok": False, "reason": "color_shape_rejected", "red": .001,
                   "green": 0, "blue": 0, "black": 0, "density": 0,
                   "hits": 0, "evidence": 0}
        with patch.object(config, "VISION_DIAGNOSTICS_ENABLED", True), \
                patch.object(config, "VERIFY_MAX_CANDIDATES_PER_CLASS", 1), \
                patch.object(verifier, "_validate_candidate", side_effect=lambda *args: (False, dict(metrics))):
            self.assertEqual(verifier.process(None, [target, weak, zone], LABELS), [])
            records = verifier.frame_diagnostics()
            self.assertEqual([item["status"] for item in records],
                             ["wait_temporal", "class_candidate_limit", "not_movable_class"])
            self.assertTrue(records[0]["high_conf_bypass"])
            for _ in range(config.VERIFY_REQUIRED_HITS - 1):
                result = verifier.process(None, [target, weak, zone], LABELS)
            self.assertEqual(result, [target])
            self.assertEqual(verifier.frame_diagnostics()[0]["status"], "pass")
            target.score = .6
            self.assertEqual(verifier.process(None, [target], LABELS), [])
            self.assertEqual(verifier.frame_diagnostics()[0]["status"], "current_evidence_rejected")

    def test_io_health_reports_frozen_frame_without_synthesizing_coordinates(self):
        current = controller(CMD_TRACK)
        current._last_frame_seen_ms = 1000
        with patch.object(config, "VISION_DIAGNOSTICS_ENABLED", True), patch("builtins.print") as log:
            current._report_io_health(2200)
            self.assertTrue(any("no_new_frame_ms=1200" in str(call) for call in log.call_args_list))
            self.assertIsNone(current._latest_packet)

    def test_diagnostics_enabled_does_not_change_selected_target_states_or_packets(self):
        target, moved, neighbor = block(288, 224, 4), block(100, 150, 4), block(140, 150, 3)
        histories = []
        for enabled in (False, True):
            current = controller(serial_module.CMD_SEARCH)
            current.target_label = "sqarered"
            history = []
            with patch.object(config, "VISION_DIAGNOSTICS_ENABLED", enabled), \
                    patch.object(serial_module, "_now_ms", return_value=1000), \
                    patch("builtins.print"):
                frames = [[target], [moved, neighbor]] + [[]] * 5 + [[moved]] + [[]] * 5
                for index, objects in enumerate(frames):
                    selected = current.submit(objects, LABELS, raw_objects=objects)
                    packet = current._next_outgoing_packet(1000 + index)
                    history.append((current.mode, getattr(selected, "class_id", None),
                                    packet, current._track_target_lock, current._track_no_target_frames))
                    if index == 0:
                        current._handle_command(CMD_TRACK)
            histories.append(history)
        self.assertEqual(histories[0], histories[1])

    def test_diagnostics_zone_and_near_view_describe_actual_rules(self):
        zone_cam = controller(serial_module.MODE_SEARCH_ZONE)
        zone = block(450, 200, 1)
        blocker = block(288, 200, 3)
        near_cam = controller(serial_module.MODE_NEAR_VIEW_CHECK)
        with patch.object(config, "VISION_DIAGNOSTICS_ENABLED", True), \
                patch.object(serial_module, "_now_ms", return_value=1000), \
                patch("builtins.print") as log:
            zone_cam.submit([blocker], LABELS, raw_objects=[zone, blocker])
            near_cam.submit([blocker], LABELS, raw_objects=[blocker])
            self.assertTrue(any("[ZONE-DETAIL]" in str(call) and ", 200.0)" in str(call)
                                and "id3" in str(call) for call in log.call_args_list))
            self.assertTrue(any("[NEAR-DETAIL]" in str(call) and "ids=[3]" in str(call)
                                for call in log.call_args_list))

    def test_zone_26_keeps_fresh_coordinates_between_repeats(self):
        for class_id in (1, 2):
            with self.subTest(zone_id=class_id):
                current = controller(serial_module.MODE_SEARCH_ZONE)
                current.zone_label = LABELS[class_id]
                zone = SimpleNamespace(x=0, y=0, w=576, h=224,
                                       class_id=class_id, score=0.9)
                with patch.object(serial_module, "_now_ms", return_value=1000):
                    current.submit([], LABELS, raw_objects=[zone])
                self.assertEqual(current._next_outgoing_packet(1000),
                                 build_event_packet(serial_module.EVENT_ZONE_FOUND))
                self.assertEqual(current._next_outgoing_packet(1000),
                                 build_event_packet(serial_module.EVENT_ZONE_CLOSE))
                self.assertIsNotNone(current._latest_packet)
                self.assertEqual(current._next_outgoing_packet(1000), current._latest_packet)
                self.assertIsNone(current._next_outgoing_packet(1001))
                # Fresh coordinates continue both above and below 50%, without
                # another 26 before 500ms or an MCU acknowledgment.
                # Supplies use horizontal center above25% after16.
                for index, height in enumerate((300, 200, 300)):
                    zone.h = height
                    zone.x = 10
                    zone.w = 550
                    now = 1100 + index * 100
                    with patch.object(serial_module, "_now_ms", return_value=now):
                        current.submit([], LABELS, raw_objects=[zone])
                    expected = serial_module.build_coordinate_packet(
                        class_id, -3, int(round(height * 0.5 - 403.5)))
                    self.assertEqual(current._next_outgoing_packet(now), expected)
                    self.assertEqual(current._pending_events, [])

    def test_zone_26_repeats_every_500ms_without_starving_coordinates(self):
        for class_id in (1, 2):
            with self.subTest(zone_id=class_id):
                current = controller(serial_module.MODE_SEARCH_ZONE)
                current.zone_label = LABELS[class_id]
                zone = SimpleNamespace(x=0, y=0, w=576, h=224,
                                       class_id=class_id, score=0.9)
                event = build_event_packet(serial_module.EVENT_ZONE_CLOSE)
                with patch.object(serial_module, "_now_ms", return_value=1000):
                    current.submit([], LABELS, raw_objects=[zone])
                self.assertEqual(current._next_outgoing_packet(1000),
                                 build_event_packet(serial_module.EVENT_ZONE_FOUND))
                self.assertEqual(current._next_outgoing_packet(1000), event)
                self.assertEqual(current._next_outgoing_packet(1000), current._latest_packet)
                for now in (1100, 1499, 1500, 1501, 1999, 2000):
                    with patch.object(serial_module, "_now_ms", return_value=now):
                        current.submit([], LABELS, raw_objects=[zone])
                    if now in (1500, 2000):
                        self.assertEqual(current._next_outgoing_packet(now), event)
                    self.assertEqual(current._next_outgoing_packet(now), current._latest_packet)
                    self.assertIsNone(current._next_outgoing_packet(now))
                    self.assertEqual(current._pending_events, [])

    def test_zone_26_stops_below_threshold_on_loss_or_stale_frame(self):
        current = controller(serial_module.MODE_SEARCH_ZONE)
        zone = SimpleNamespace(x=0, y=0, w=576, h=224, class_id=1, score=0.9)
        event = build_event_packet(serial_module.EVENT_ZONE_CLOSE)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            current.submit([], LABELS, raw_objects=[zone])
        for _ in range(3):
            current._next_outgoing_packet(1000)
        zone.h = 223
        with patch.object(serial_module, "_now_ms", return_value=1500):
            current.submit([], LABELS, raw_objects=[zone])
        self.assertFalse(current._zone_close_active)
        self.assertEqual(current._next_outgoing_packet(1500), current._latest_packet)
        self.assertIsNone(current._next_outgoing_packet(1500))
        zone.h = 224
        with patch.object(serial_module, "_now_ms", return_value=1600):
            current.submit([], LABELS, raw_objects=[zone])
        self.assertEqual(current._next_outgoing_packet(1600), event)
        self.assertEqual(current._next_outgoing_packet(1600), current._latest_packet)
        # A new frame is required; a stopped camera cannot keep emitting 26.
        self.assertIsNone(current._next_outgoing_packet(2100))
        with patch.object(serial_module, "_now_ms", return_value=2200):
            current.submit([], LABELS, raw_objects=[zone])
        self.assertEqual(current._next_outgoing_packet(2200), event)
        self.assertEqual(current._next_outgoing_packet(2200), current._latest_packet)
        with patch.object(serial_module, "_now_ms", return_value=2700):
            current.submit([], LABELS, raw_objects=[block(288, 224, 2)])
        self.assertEqual(current._next_outgoing_packet(2700),
                         build_event_packet(serial_module.EVENT_ZONE_NO_TARGET))
        with patch.object(serial_module, "_now_ms", return_value=2800):
            current.submit([], LABELS, raw_objects=[zone])
        self.assertEqual(current._next_outgoing_packet(2800), event)
        self.assertEqual(current._next_outgoing_packet(2800), current._latest_packet)
        current._handle_command(serial_module.CMD_SEARCH)
        self.assertFalse(current._zone_close_sent)
        self.assertFalse(current._zone_close_active)
        self.assertIsNone(current._last_zone_close_event_ms)
        self.assertIsNone(current._next_outgoing_packet(3300))

    def test_zone_26_cancels_queued_first_event_after_loss_or_staleness(self):
        for invalid in ("missing", "small", "stale"):
            with self.subTest(invalid=invalid):
                current = controller(serial_module.MODE_SEARCH_ZONE)
                zone = SimpleNamespace(x=0, y=0, w=576, h=224, class_id=1, score=0.9)
                event = build_event_packet(serial_module.EVENT_ZONE_CLOSE)
                with patch.object(serial_module, "_now_ms", return_value=1000):
                    current.submit([], LABELS, raw_objects=[zone])
                self.assertEqual(current._next_outgoing_packet(1000),
                                 build_event_packet(serial_module.EVENT_ZONE_FOUND))
                if invalid != "stale":
                    zone.h = 223
                    with patch.object(serial_module, "_now_ms", return_value=1100):
                        current.submit([], LABELS, raw_objects=[] if invalid == "missing" else [zone])
                packet = current._next_outgoing_packet(1500)
                if invalid == "missing":
                    self.assertEqual(packet, build_event_packet(serial_module.EVENT_ZONE_NO_TARGET))
                else:
                    self.assertIsNone(packet)
                self.assertNotIn(event, current._pending_events)
                self.assertIsNone(current._last_zone_close_event_ms)
                zone.h = 224
                with patch.object(serial_module, "_now_ms", return_value=1600):
                    current.submit([], LABELS, raw_objects=[zone])
                self.assertEqual(current._next_outgoing_packet(1600), event)
                self.assertEqual(current._next_outgoing_packet(1600), current._latest_packet)

    def test_zone_26_repeat_respects_command_spacing(self):
        current = controller(serial_module.MODE_SEARCH_ZONE)
        zone = SimpleNamespace(x=0, y=0, w=576, h=224, class_id=1, score=0.9)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            current.submit([], LABELS, raw_objects=[zone])
        for _ in range(3):
            current._next_outgoing_packet(1000)
        with patch.object(serial_module, "_now_ms", return_value=1500):
            current.submit([], LABELS, raw_objects=[zone])
        current._last_event_write_ms = 1460
        self.assertIsNone(current._next_outgoing_packet(1500))
        self.assertEqual(current._next_outgoing_packet(1510),
                         build_event_packet(serial_module.EVENT_ZONE_CLOSE))
        self.assertEqual(current._next_outgoing_packet(1510), current._latest_packet)

    def test_zone_after_26_missing_frame_drops_coordinates_and_reacquires(self):
        current = controller(serial_module.MODE_SEARCH_ZONE)
        zone = SimpleNamespace(x=0, y=0, w=576, h=300, class_id=1, score=0.9)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            current.submit([], LABELS, raw_objects=[zone])
        for _ in range(3):
            current._next_outgoing_packet(1000)
        with patch.object(serial_module, "_now_ms", return_value=1100):
            current.submit([], LABELS, raw_objects=[])
        self.assertIsNone(current._latest_packet)
        self.assertEqual(current._next_outgoing_packet(1100),
                         build_event_packet(serial_module.EVENT_ZONE_NO_TARGET))
        with patch.object(serial_module, "_now_ms", return_value=1200):
            current.submit([], LABELS, raw_objects=[zone])
        self.assertEqual(current._next_outgoing_packet(1200),
                                 serial_module.build_coordinate_packet(1, 0, -254))
        self.assertEqual(current._pending_events, [])
        current._handle_command(serial_module.CMD_SEARCH)
        self.assertIsNone(current._latest_packet)
        self.assertFalse(current._zone_close_sent)

    def test_zone_obstacle_bypass_defaults_disabled(self):
        self.assertFalse(config.UART_ZONE_OBSTACLE_ENABLED)
        self.assertFalse(controller(serial_module.MODE_SEARCH_ZONE)._zone_obstacle_enabled())

    def test_zone_obstacle_missing_enable_config_also_defaults_disabled(self):
        with patch.dict(config.__dict__):
            config.__dict__.pop("UART_ZONE_OBSTACLE_ENABLED", None)
            current = controller(serial_module.MODE_SEARCH_ZONE)
            self.assertFalse(current._zone_obstacle_enabled())
            current.send_event(serial_module.EVENT_ZONE_OBSTACLE)
            self.assertIsNone(current._next_outgoing_packet(1000))

    def test_disabled_obstacle_range_never_pauses_zone_coordinates_or_requests36(self):
        for zone_id in (1, 2):
            for target_id in (3, 4, 5, 6):
                with self.subTest(zone=zone_id, target=target_id):
                    current = controller(serial_module.MODE_SEARCH_ZONE)
                    current.zone_label = LABELS[zone_id]
                    current.target_label = LABELS[target_id]
                    zone = SimpleNamespace(x=0, y=300, w=576, h=100, class_id=zone_id, score=0.9)
                    blocker = block(288, 240, target_id)
                    with patch.object(current, "_find_roi_edge_blocker",
                                      side_effect=AssertionError("disabled obstacle scanning must not run")):
                        with patch.object(serial_module, "_now_ms", return_value=1000):
                            current.submit([blocker], LABELS, raw_objects=[zone, blocker])
                        self.assertEqual(current._next_outgoing_packet(1000),
                                         build_event_packet(serial_module.EVENT_ZONE_FOUND))
                        for now in range(1100, 1200, 10):
                            with patch.object(serial_module, "_now_ms", return_value=now):
                                current.submit([blocker], LABELS, raw_objects=[zone, blocker])
                            self.assertEqual(current._next_outgoing_packet(now)[2], zone_id)
                            self.assertFalse(current._zone_obstacle_waiting)
                            self.assertEqual(current._zone_obstacle_frames, 0)
                            self.assertNotIn(build_event_packet(serial_module.EVENT_ZONE_OBSTACLE),
                                             current._pending_events)

    def test_disabled_obstacle_never_defers26_or_coordinates_after_zone_area_grows(self):
        current = controller(serial_module.MODE_SEARCH_ZONE)
        zone = SimpleNamespace(x=0, y=300, w=576, h=100, class_id=1, score=0.9)
        blocker = block(288, 240, 3)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            current.submit([blocker], LABELS, raw_objects=[zone, blocker])
        self.assertEqual(current._next_outgoing_packet(1000), build_event_packet(serial_module.EVENT_ZONE_FOUND))
        zone.y, zone.h = 200, 248
        for now in (1100, 1600, 2100):
            with patch.object(serial_module, "_now_ms", return_value=now):
                current.submit([blocker], LABELS, raw_objects=[zone, blocker])
            self.assertEqual(current._next_outgoing_packet(now), build_event_packet(serial_module.EVENT_ZONE_CLOSE))
            self.assertEqual(current._next_outgoing_packet(now), serial_module.build_coordinate_packet(1, 0, -80))
            self.assertFalse(current._zone_obstacle_waiting)

    def test_disabled_obstacle_drops_manual_queued_and_prepared36_in_every_stage(self):
        for mode in (serial_module.MODE_SEARCH_ZONE, serial_module.CMD_SEARCH,
                     MODE_FINAL_ROI_TRACK):
            with self.subTest(mode=mode):
                current = controller(mode)
                current.send_event(serial_module.EVENT_ZONE_OBSTACLE)
                self.assertIsNone(current._next_outgoing_packet(1000))
                request = build_event_packet(serial_module.EVENT_ZONE_OBSTACLE)
                current._serial = SimpleNamespace(write=Mock(return_value=len(request)))
                with patch.object(serial_module, "_now_ms", return_value=1000):
                    current._write_packet(request, current.frame_generation())
                current._serial.write.assert_not_called()
                self.assertEqual(current._pending_events, [])

    def test_disabling_obstacle_cancels_old_wait_and_wire36_without_ack(self):
        with patch.object(config, "UART_ZONE_OBSTACLE_ENABLED", True):
            current, zone, blocker = self._start_zone_obstacle_wait()
        request = build_event_packet(serial_module.EVENT_ZONE_OBSTACLE)
        current._zone_obstacle_request_sent = True
        current._pending_events.append(request)
        current._serial = SimpleNamespace(write=Mock(return_value=len(request)))
        with patch.object(serial_module, "_now_ms", return_value=1200):
            current._write_packet(request, current.frame_generation())
        current._serial.write.assert_not_called()
        self.assertFalse(current._zone_obstacle_waiting)
        self.assertFalse(current._zone_obstacle_request_sent)
        self.assertEqual(current._pending_events, [])
        zone.y, zone.h = 200, 248
        with patch.object(serial_module, "_now_ms", return_value=1300):
            current.submit([blocker], LABELS, raw_objects=[zone, blocker])
        self.assertEqual(current._next_outgoing_packet(1300), build_event_packet(serial_module.EVENT_ZONE_CLOSE))
        self.assertIsNotNone(current._latest_packet)

    def test_disabled_obstacle_ack_does_not_clear_fresh_zone_coordinates_or_generation(self):
        current = controller(serial_module.MODE_SEARCH_ZONE)
        zone = SimpleNamespace(x=0, y=200, w=576, h=248, class_id=1, score=0.9)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            current.submit([], LABELS, raw_objects=[zone])
        packet = current._latest_packet
        generation = current.frame_generation()
        self.assertTrue(current._zone_close_active)
        with patch.object(serial_module, "_now_ms", return_value=1001):
            current.process_received_bytes(serial_module.build_command_packet(serial_module.CMD_ZONE_OBSTACLE_DONE))
        self.assertEqual(current._latest_packet, packet)
        self.assertEqual(current.frame_generation(), generation)
        self.assertTrue(current._zone_close_active)
        self.assertEqual(current.mode, serial_module.MODE_SEARCH_ZONE)

    @patch.object(config, "UART_ZONE_OBSTACLE_ENABLED", True)
    def test_zone_close_still_waits_for_obstacle_ack_before_coordinates(self):
        current = controller(serial_module.MODE_SEARCH_ZONE)
        zone = SimpleNamespace(x=0, y=300, w=576, h=100, class_id=1, score=0.9)
        blocker = block(288, 240, 3)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            current.submit([], LABELS, raw_objects=[zone])
        self.assertEqual(current._next_outgoing_packet(1000),
                         build_event_packet(serial_module.EVENT_ZONE_FOUND))
        for index in range(config.UART_ZONE_OBSTACLE_CONFIRM_FRAMES):
            with patch.object(serial_module, "_now_ms", return_value=1100 + index * 10):
                current.submit([blocker], LABELS, raw_objects=[zone, blocker])
        self.assertTrue(current._zone_obstacle_waiting)
        self.assertFalse(current._zone_close_sent)
        self.assertIsNone(current._latest_packet)
        request = current._next_outgoing_packet(1120)
        self.assertEqual(request, build_event_packet(serial_module.EVENT_ZONE_OBSTACLE))
        current._serial = SimpleNamespace(write=Mock(return_value=len(request)))
        with patch.object(serial_module, "_now_ms", return_value=1120):
            current._write_packet(request, current._outgoing_generation)
        self.assertTrue(current._zone_obstacle_request_sent)
        zone.y, zone.h = 200, 248
        with patch.object(serial_module, "_now_ms", return_value=1200):
            current.submit([blocker], LABELS, raw_objects=[zone, blocker])
        # A successfully sent36 must finish its MCU handshake even if the
        # zone grows past the new25% gate while that maneuver is in progress.
        self.assertTrue(current._zone_obstacle_waiting)
        self.assertFalse(current._zone_close_sent)
        self.assertIsNone(current._latest_packet)
        self.assertIsNone(current._next_outgoing_packet(1200))
        self.assertEqual(current._next_outgoing_packet(1620), request)
        with patch.object(serial_module, "_now_ms", return_value=1650):
            current._handle_command(serial_module.CMD_ZONE_OBSTACLE_DONE)
        self.assertIsNone(current._latest_packet)
        with patch.object(serial_module, "_now_ms", return_value=1700):
            current.submit([blocker], LABELS, raw_objects=[zone, blocker])
        # New bypasses above25% are forbidden; the same blocker no longer
        # defers26 once the previously started maneuver has been acknowledged.
        self.assertTrue(current._zone_close_sent)
        self.assertEqual(current._next_outgoing_packet(1700),
                         build_event_packet(serial_module.EVENT_ZONE_CLOSE))
        self.assertEqual(current._next_outgoing_packet(1700),
                         serial_module.build_coordinate_packet(1, 0, -80))
        for now in range(1710, 1800, 10):
            with patch.object(serial_module, "_now_ms", return_value=now):
                current.submit([blocker], LABELS, raw_objects=[zone, blocker])
            self.assertNotEqual(current._next_outgoing_packet(now), request)
        self.assertFalse(current._zone_obstacle_waiting)
        self.assertEqual(current._zone_obstacle_frames, 0)

    @patch.object(config, "UART_ZONE_OBSTACLE_ENABLED", True)
    def test_zone36_only_triggers_at_or_below25_percent_for_both_zone_colors(self):
        self.assertEqual(config.UART_ZONE_OBSTACLE_MAX_AREA_RATIO, .25)
        for zone_id in (1, 2):
            for cargo_id in (4, 5):
                for height in (111, 112, 113):
                    with self.subTest(zone=zone_id, cargo=cargo_id, h=height):
                        current = controller(serial_module.MODE_SEARCH_ZONE)
                        current.zone_label = LABELS[zone_id]
                        current.target_label = LABELS[cargo_id]
                        zone = SimpleNamespace(x=0, y=300, w=576, h=height,
                                               class_id=zone_id, score=.9)
                        blocker = block(288, 240, 3)
                        with patch.object(serial_module, "_now_ms", return_value=1000):
                            current.submit([], LABELS, raw_objects=[zone])
                        self.assertEqual(current._next_outgoing_packet(1000),
                                         build_event_packet(serial_module.EVENT_ZONE_FOUND))
                        for now in range(1100, 1170, 10):
                            with patch.object(serial_module, "_now_ms", return_value=now):
                                current.submit([blocker], LABELS, raw_objects=[zone, blocker])
                        if height <= 112:
                            self.assertTrue(current._zone_obstacle_waiting)
                            self.assertEqual(current._next_outgoing_packet(1160),
                                             build_event_packet(serial_module.EVENT_ZONE_OBSTACLE))
                        else:
                            self.assertFalse(current._zone_obstacle_waiting)
                            self.assertEqual(current._zone_obstacle_frames, 0)
                            self.assertEqual(len(current._next_outgoing_packet(1160)), 11)

    @patch.object(config, "UART_ZONE_OBSTACLE_ENABLED", True)
    def test_zone36_area_cap_uses_visible_clipped_area(self):
        current = controller(serial_module.MODE_SEARCH_ZONE)
        zone = SimpleNamespace(x=-576, y=300, w=1152, h=112, class_id=1, score=.9)
        blocker = block(288, 240, 3)
        self.assertEqual(current._visible_box_area_ratio(zone), .25)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            current.submit([], LABELS, raw_objects=[zone])
        current._next_outgoing_packet(1000)
        for now in (1100, 1110, 1120):
            with patch.object(serial_module, "_now_ms", return_value=now):
                current.submit([blocker], LABELS, raw_objects=[zone, blocker])
        self.assertEqual(current._next_outgoing_packet(1120),
                         build_event_packet(serial_module.EVENT_ZONE_OBSTACLE))

    @patch.object(config, "UART_ZONE_OBSTACLE_ENABLED", True)
    def test_zone36_area_above25_clears_hits_before_returning_to_far_view(self):
        current = controller(serial_module.MODE_SEARCH_ZONE)
        zone = SimpleNamespace(x=0, y=300, w=576, h=112, class_id=1, score=.9)
        blocker = block(288, 240, 3)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            current.submit([], LABELS, raw_objects=[zone])
        current._next_outgoing_packet(1000)
        for now in (1100, 1110):
            with patch.object(serial_module, "_now_ms", return_value=now):
                current.submit([blocker], LABELS, raw_objects=[zone, blocker])
        self.assertEqual(current._zone_obstacle_frames, 2)
        zone.h = 113
        with patch.object(serial_module, "_now_ms", return_value=1120):
            current.submit([blocker], LABELS, raw_objects=[zone, blocker])
        self.assertEqual(current._zone_obstacle_frames, 0)
        zone.h = 112
        for index, now in enumerate((1130, 1140, 1150), start=1):
            with patch.object(serial_module, "_now_ms", return_value=now):
                current.submit([blocker], LABELS, raw_objects=[zone, blocker])
            self.assertEqual(current._zone_obstacle_frames, index)
            self.assertEqual(current._zone_obstacle_waiting, index == 3)

    @patch.object(config, "UART_ZONE_OBSTACLE_ENABLED", True)
    def test_zone36_area_above25_cancels_queued_and_prepared_unsent_requests(self):
        for prepared in (False, True):
            current, zone, blocker = self._start_zone_obstacle_wait()
            request = build_event_packet(serial_module.EVENT_ZONE_OBSTACLE)
            generation = current.frame_generation()
            if not prepared:
                current._pending_events.append(request)
            zone.h = 113
            with patch.object(serial_module, "_now_ms", return_value=1100):
                current.submit([blocker], LABELS, raw_objects=[zone, blocker])
            self.assertFalse(current._zone_obstacle_waiting)
            self.assertNotIn(request, current._pending_events)
            current._serial = SimpleNamespace(write=Mock(return_value=len(request)))
            with patch.object(serial_module, "_now_ms", return_value=1100):
                current._write_packet(request, generation)
            current._serial.write.assert_not_called()
            self.assertEqual(len(current._next_outgoing_packet(1100)), 11)

    @patch.object(config, "UART_ZONE_OBSTACLE_ENABLED", True)
    def test_zone36_short_write_does_not_count_as_started_bypass_above25(self):
        current, zone, blocker = self._start_zone_obstacle_wait()
        request = build_event_packet(serial_module.EVENT_ZONE_OBSTACLE)
        current._serial = SimpleNamespace(write=Mock(return_value=2))
        with patch.object(serial_module, "_now_ms", return_value=1040):
            current._write_packet(request, current.frame_generation())
        self.assertFalse(current._zone_obstacle_request_sent)
        zone.h = 113
        with patch.object(serial_module, "_now_ms", return_value=1100):
            current.submit([blocker], LABELS, raw_objects=[zone, blocker])
        self.assertFalse(current._zone_obstacle_waiting)
        self.assertEqual(len(current._next_outgoing_packet(1100)), 11)

    def _start_zone_obstacle_wait(self, start=1000):
        current = controller(serial_module.MODE_SEARCH_ZONE)
        zone = SimpleNamespace(x=0, y=300, w=576, h=100, class_id=1, score=0.9)
        blocker = block(288, 240, 3)
        with patch.object(serial_module, "_now_ms", return_value=start):
            current.submit([], LABELS, raw_objects=[zone])
        self.assertEqual(current._next_outgoing_packet(start),
                         build_event_packet(serial_module.EVENT_ZONE_FOUND))
        for index in range(3):
            with patch.object(serial_module, "_now_ms", return_value=start + 10 + index * 10):
                current.submit([blocker], LABELS, raw_objects=[zone, blocker])
        self.assertEqual(current._next_outgoing_packet(start + 30),
                         build_event_packet(serial_module.EVENT_ZONE_OBSTACLE))
        return current, zone, blocker

    @patch.object(config, "UART_ZONE_OBSTACLE_ENABLED", True)
    def test_zone36_same_blocker_can_trigger_multiple_rounds_after_each_ack(self):
        current, zone, blocker = self._start_zone_obstacle_wait()
        for cycle in range(3):
            now = 1100 + cycle * 100
            with patch.object(serial_module, "_now_ms", return_value=now):
                current.process_received_bytes(serial_module.build_command_packet(0x36))
            self.assertFalse(current._zone_obstacle_waiting)
            self.assertTrue(current._zone_obstacle_armed)
            self.assertEqual(current._zone_obstacle_frames, 0)
            self.assertIsNone(current._next_outgoing_packet(now))
            for index in range(3):
                with patch.object(serial_module, "_now_ms", return_value=now + 10 + index * 10):
                    current.submit([blocker], LABELS, raw_objects=[zone, blocker])
                packet = current._next_outgoing_packet(now + 10 + index * 10)
                if index < 2:
                    self.assertFalse(current._zone_obstacle_waiting)
                    self.assertEqual(len(packet), 11)
                    self.assertEqual(packet[2], 1)
                else:
                    self.assertTrue(current._zone_obstacle_waiting)
                    self.assertEqual(packet, build_event_packet(serial_module.EVENT_ZONE_OBSTACLE))
                    self.assertIsNone(current._latest_packet)
            self.assertFalse(current._zone_close_sent)

    @patch.object(config, "UART_ZONE_OBSTACLE_ENABLED", True)
    def test_zone36_rearmed_hits_are_accumulated_not_required_consecutive(self):
        current, zone, blocker = self._start_zone_obstacle_wait()
        current._handle_command(serial_module.CMD_ZONE_OBSTACLE_DONE)
        samples = ((1100, [blocker], [zone, blocker]),
                   (1110, [], [zone]),
                   (1120, [blocker], [zone, blocker]),
                   (1130, [], []),
                   (1140, [blocker], [zone, blocker]))
        for now, objects, raw in samples:
            with patch.object(serial_module, "_now_ms", return_value=now):
                current.submit(objects, LABELS, raw_objects=raw)
            packet = current._next_outgoing_packet(now)
            if now < 1140:
                self.assertNotEqual(packet, build_event_packet(serial_module.EVENT_ZONE_OBSTACLE))
            else:
                self.assertEqual(packet, build_event_packet(serial_module.EVENT_ZONE_OBSTACLE))
        self.assertEqual(current._zone_obstacle_frames, 3)

    @patch.object(config, "UART_ZONE_OBSTACLE_ENABLED", True)
    def test_zone36_waiting_repeats500ms_without_resuming_or_new_round(self):
        current, zone, blocker = self._start_zone_obstacle_wait()
        for now in range(1100, 1530, 50):
            with patch.object(serial_module, "_now_ms", return_value=now):
                current.submit([blocker], LABELS, raw_objects=[zone, blocker])
            self.assertIsNone(current._next_outgoing_packet(now))
            self.assertIsNone(current._latest_packet)
        self.assertEqual(current._next_outgoing_packet(1530),
                         build_event_packet(serial_module.EVENT_ZONE_OBSTACLE))
        self.assertTrue(current._zone_obstacle_waiting)
        self.assertIsNone(current._latest_packet)
        current._handle_command(serial_module.CMD_ZONE_OBSTACLE_DONE)
        self.assertEqual(current._zone_obstacle_frames, 0)
        self.assertEqual(current._pending_events, [])

    @patch.object(config, "UART_ZONE_OBSTACLE_ENABLED", True)
    def test_zone36_ack_discards_old_frame_and_prepared_packet_before_new_hits(self):
        current, zone, blocker = self._start_zone_obstacle_wait()
        old_generation = current.frame_generation()
        old_packet = build_event_packet(serial_module.EVENT_ZONE_OBSTACLE)
        current._serial = SimpleNamespace(write=Mock(return_value=len(old_packet)))
        current._handle_command(serial_module.CMD_ZONE_OBSTACLE_DONE)
        current._write_packet(old_packet, generation=old_generation)
        current._serial.write.assert_not_called()
        with patch.object(serial_module, "_now_ms", return_value=1100):
            self.assertIsNone(current.submit([blocker], LABELS, raw_objects=[zone, blocker],
                                             frame_generation=old_generation))
        self.assertEqual(current._zone_obstacle_frames, 0)
        self.assertIsNone(current._latest_packet)
        with patch.object(serial_module, "_now_ms", return_value=1110):
            current.submit([blocker], LABELS, raw_objects=[zone, blocker],
                           frame_generation=current.frame_generation())
        self.assertEqual(current._zone_obstacle_frames, 1)

    @patch.object(config, "UART_ZONE_OBSTACLE_ENABLED", True)
    def test_zone36_duplicate_ack_before_new_request_does_not_reset_hits(self):
        current, zone, blocker = self._start_zone_obstacle_wait()
        current._handle_command(serial_module.CMD_ZONE_OBSTACLE_DONE)
        with patch.object(serial_module, "_now_ms", return_value=1100):
            current.submit([blocker], LABELS, raw_objects=[zone, blocker])
        generation = current.frame_generation()
        current._handle_command(serial_module.CMD_ZONE_OBSTACLE_DONE)
        self.assertEqual(current.frame_generation(), generation)
        self.assertEqual(current._zone_obstacle_frames, 1)
        for now in (1110, 1120):
            with patch.object(serial_module, "_now_ms", return_value=now):
                current.submit([blocker], LABELS, raw_objects=[zone, blocker])
        self.assertEqual(current._next_outgoing_packet(1120),
                         build_event_packet(serial_module.EVENT_ZONE_OBSTACLE))

    @patch.object(config, "UART_ZONE_OBSTACLE_ENABLED", True)
    def test_zone36_never_restarts_after_first26_even_if_area_shrinks(self):
        current = controller(serial_module.MODE_SEARCH_ZONE)
        zone = SimpleNamespace(x=0, y=200, w=576, h=248, class_id=1, score=0.9)
        blocker = block(288, 240, 3)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            current.submit([], LABELS, raw_objects=[zone])
        self.assertEqual(current._next_outgoing_packet(1000),
                         build_event_packet(serial_module.EVENT_ZONE_FOUND))
        self.assertEqual(current._next_outgoing_packet(1000),
                         build_event_packet(serial_module.EVENT_ZONE_CLOSE))
        self.assertTrue(current._zone_close_sent)
        zone.y, zone.h = 300, 100
        for now in range(1100, 1200, 10):
            with patch.object(serial_module, "_now_ms", return_value=now):
                current.submit([blocker], LABELS, raw_objects=[zone, blocker])
            self.assertNotEqual(current._next_outgoing_packet(now),
                                build_event_packet(serial_module.EVENT_ZONE_OBSTACLE))
        self.assertFalse(current._zone_obstacle_waiting)
        self.assertEqual(current._zone_obstacle_frames, 0)

    def test_zone_raw_yolo_detection_sends_without_jump_or_verification_gates(self):
        current = controller(serial_module.MODE_SEARCH_ZONE)
        # Raw-zone tracking still bypasses jump/color verification, but valid
        # zones now need visible area >2% before coordinates/16 are allowed.
        blue_zone = SimpleNamespace(x=50, y=160, w=100, h=80, class_id=1, score=.9)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            current.submit([], LABELS, raw_objects=[blue_zone])
        self.assertEqual(current._next_outgoing_packet(1000),
                         build_event_packet(serial_module.EVENT_ZONE_FOUND))
        self.assertIsNotNone(current._next_outgoing_packet(1000))
        jumped_zone = SimpleNamespace(x=400, y=160, w=100, h=80, class_id=1, score=.9)
        jumped_zone.score = 0.01
        wrong_color = block(288, 200, 2)
        with patch.object(serial_module, "_now_ms", return_value=1010), \
                patch.object(current, "_reject_large_target_jump",
                             side_effect=AssertionError("zone raw YOLO must bypass jump gate")):
            self.assertIs(current.submit([], LABELS, raw_objects=[wrong_color, jumped_zone]), jumped_zone)
        self.assertEqual(current._next_outgoing_packet(1010),
                         serial_module.build_coordinate_packet(1, 142, -204))
        with patch.object(serial_module, "_now_ms", return_value=1020):
            current.submit([], LABELS, raw_objects=[wrong_color])
        self.assertEqual(current._next_outgoing_packet(1020),
                         build_event_packet(serial_module.EVENT_ZONE_NO_TARGET))

    def test_zone_supply_and_casualty_center_after_25percent(self):
        for target_label, small_x, large_x in (
                ("sqareredgreen", -115, 0), ("triangualrblack", -115, 0),
                ("sqareblue", -115, 0), (serial_module.TARGET_BLACK_OR_GREEN, -115, 0),
                ("sqarered", 58, 0)):
            with self.subTest(target=target_label):
                current = controller(serial_module.MODE_SEARCH_ZONE)
                current.target_label = target_label
                zone = SimpleNamespace(x=0, y=0, w=576, h=112, class_id=1, score=0.9)
                with patch.object(serial_module, "_now_ms", return_value=1000):
                    current.submit([], LABELS, raw_objects=[zone])
                self.assertEqual(current._next_outgoing_packet(1000),
                                 build_event_packet(serial_module.EVENT_ZONE_FOUND))
                self.assertEqual(current._next_outgoing_packet(1000),
                                 serial_module.build_coordinate_packet(1, small_x, -348))
                # Exactly25% keeps the original point, even after16.
                with patch.object(serial_module, "_now_ms", return_value=1100):
                    current.submit([], LABELS, raw_objects=[zone])
                self.assertEqual(current._next_outgoing_packet(1100),
                                 serial_module.build_coordinate_packet(1, small_x, -348))
                zone.h = 113
                with patch.object(serial_module, "_now_ms", return_value=1200):
                    current.submit([], LABELS, raw_objects=[zone])
                self.assertEqual(current._next_outgoing_packet(1200),
                                 serial_module.build_coordinate_packet(1, large_x, -347))
                zone.h = 224
                with patch.object(serial_module, "_now_ms", return_value=1300):
                    current.submit([], LABELS, raw_objects=[zone])
                self.assertEqual(current._next_outgoing_packet(1300),
                                 build_event_packet(serial_module.EVENT_ZONE_CLOSE))
                self.assertEqual(current._next_outgoing_packet(1300),
                                 serial_module.build_coordinate_packet(1, large_x, -292))

    def test_supply_near_center_with_offset_zone_both_colors_and16_gate(self):
        self.assertEqual(config.UART_ZONE_SUPPLY_NEAR_X_RATIO, .5)
        for target_label in ("sqareredgreen", "triangualrblack", "sqareblue", serial_module.TARGET_BLACK_OR_GREEN):
            for zone_id in (1, 2):
                current = controller(serial_module.MODE_SEARCH_ZONE)
                current.target_label, current.zone_label = target_label, LABELS[zone_id]
                current.tracking_roi = make_bottom_center_roi(576, 448)
                zone = SimpleNamespace(x=60, y=20, w=400, h=164, class_id=zone_id, score=.9)
                with patch.object(serial_module, "_now_ms", return_value=1000):
                    current.submit([], LABELS, raw_objects=[zone])
                self.assertEqual(current._next_outgoing_packet(1000), build_event_packet(serial_module.EVENT_ZONE_FOUND))
                # First recognition still uses3/10 even with area>25%.
                self.assertEqual(current._next_outgoing_packet(1000), serial_module.build_coordinate_packet(zone_id, -108, -282))
                for now, height, expected_x, expected_y in ((1100, 164, -28, -282),
                                                           (1200, 160, -108, -284),
                                                           (1300, 164, -28, -282)):
                    zone.h = height
                    with patch.object(serial_module, "_now_ms", return_value=now):
                        current.submit([], LABELS, raw_objects=[zone])
                    self.assertEqual(current._next_outgoing_packet(now), serial_module.build_coordinate_packet(zone_id, expected_x, expected_y))

    def test_casualty_right_inward_points_with_offset_zone_and_enlarged_roi(self):
        for zone_id in (1, 2):
            with self.subTest(zone_id=zone_id):
                current = controller(serial_module.MODE_SEARCH_ZONE)
                current.target_label = "sqarered"
                current.zone_label = LABELS[zone_id]
                current.tracking_roi = make_bottom_center_roi(576, 448)
                zone = SimpleNamespace(x=60, y=20, w=400, h=160,
                                       class_id=zone_id, score=0.9)
                # Area <25%: right460 minus width*2/5 is X300.
                with patch.object(serial_module, "_now_ms", return_value=1000):
                    current.submit([], LABELS, raw_objects=[zone])
                self.assertEqual(current._next_outgoing_packet(1000),
                                 build_event_packet(serial_module.EVENT_ZONE_FOUND))
                self.assertEqual(current._next_outgoing_packet(1000),
                                 serial_module.build_coordinate_packet(zone_id, 12, -284))
                # After16 and area >25%: right460 minus width/2 is X260.
                zone.h = 164
                with patch.object(serial_module, "_now_ms", return_value=1100):
                    current.submit([], LABELS, raw_objects=[zone])
                self.assertEqual(current._next_outgoing_packet(1100),
                                 serial_module.build_coordinate_packet(zone_id, -28, -282))

    def test_casualty_center_point_requires16_and_strictly_over25percent(self):
        current = controller(serial_module.MODE_SEARCH_ZONE)
        current.target_label = "sqarered"
        zone = SimpleNamespace(x=0, y=0, w=576, h=113, class_id=1, score=0.9)
        # Even above25%, the first observation uses 2/5 inward from the right.
        with patch.object(serial_module, "_now_ms", return_value=1000):
            current.submit([], LABELS, raw_objects=[zone])
        self.assertEqual(current._next_outgoing_packet(1000),
                         build_event_packet(serial_module.EVENT_ZONE_FOUND))
        self.assertEqual(current._next_outgoing_packet(1000),
                         serial_module.build_coordinate_packet(1, 58, -347))
        with patch.object(serial_module, "_now_ms", return_value=1100):
            current.submit([], LABELS, raw_objects=[zone])
        self.assertEqual(current._next_outgoing_packet(1100),
                         serial_module.build_coordinate_packet(1, 0, -347))
        # Returning to exactly25% restores the unchanged far aim point.
        zone.h = 112
        with patch.object(serial_module, "_now_ms", return_value=1200):
            current.submit([], LABELS, raw_objects=[zone])
        self.assertEqual(current._next_outgoing_packet(1200),
                         serial_module.build_coordinate_packet(1, 58, -348))

    def test_zone_far_points_for_supplies_and_casualties_on_both_zone_colors(self):
        for zone_id in (1, 2):
            for target_label, error_x in (("sqareredgreen", -118), ("triangualrblack", -118),
                                           (serial_module.TARGET_BLACK_OR_GREEN, -118),
                                           ("sqarered", -28)):
                with self.subTest(zone_id=zone_id, target=target_label):
                    current = controller(serial_module.MODE_SEARCH_ZONE)
                    current.tracking_roi = make_bottom_center_roi(576, 448)
                    current.target_label = target_label
                    current.zone_label = LABELS[zone_id]
                    zone = SimpleNamespace(x=80, y=50, w=300, h=120,
                                           class_id=zone_id, score=.9)
                    with patch.object(serial_module, "_now_ms", return_value=1000):
                        current.submit([], LABELS, raw_objects=[zone])
                    self.assertEqual(current._next_outgoing_packet(1000),
                                     build_event_packet(serial_module.EVENT_ZONE_FOUND))
                    # Supply X170; casualty right380-120=X260. Both use Y110.
                    self.assertEqual(current._next_outgoing_packet(1000),
                                     serial_module.build_coordinate_packet(zone_id, error_x, -274))

    def test_zone_point_helper_default_matches_supply_config(self):
        zone = SimpleNamespace(x=80, y=50, w=300, h=120)
        self.assertEqual(serial_module.VisionSerialController._zone_supply_point_error(
            zone, 288, 373.5), (-118, -264))

    def test_debug_raw_yolo_class_color_boxes_include_all_classes(self):
        img = SimpleNamespace(draw_rect=Mock())
        objects = [block(200, 300, class_id) for class_id in range(7)]
        draw_debug_yolo_detections(img, objects, 576, 448)
        self.assertEqual(img.draw_rect.call_count, 7)
        colors = ["blue", "blue", "red", (0, 191, 255), "red", "green", "black"]
        for call, color in zip(img.draw_rect.call_args_list, colors):
            self.assertEqual(call.args, (190, 290, 20, 20))
            self.assertEqual(call.kwargs, {"color": color, "thickness": 3})

    def test_debug_raw_yolo_boxes_clip_and_do_not_draw_missing_detections(self):
        img = SimpleNamespace(draw_rect=Mock())
        draw_debug_yolo_detections(img, [block(0, 0)], 576, 448)
        img.draw_rect.assert_called_once_with(
            0, 0, 10, 10, color="green", thickness=3
        )
        img.draw_rect.reset_mock()
        draw_debug_yolo_detections(img, [], 576, 448)
        draw_debug_yolo_detections(img, [block(700, 600)], 576, 448)
        invalid = block(100, 100)
        invalid.w = 0
        draw_debug_yolo_detections(img, [invalid], 576, 448)
        img.draw_rect.assert_not_called()

    def test_debug_zone_obstacle_trapezoid_matches_actual_center_bounds(self):
        roi = [0, 359, 576, 89]
        img = SimpleNamespace(draw_line=Mock())
        draw_debug_zone_obstacle_region(img, roi, 576, 448)
        vertices = ((238.0 - 238.0 * 100.0 / 279.0, 200.0),
                    (338.0 + 237.0 * 100.0 / 279.0, 200.0),
                    (575, 379), (0, 379))
        self.assertEqual(zone_obstacle_polygon(roi, 576, 448), vertices)
        self.assertEqual(img.draw_line.call_count, 4)
        for index, call in enumerate(img.draw_line.call_args_list):
            self.assertEqual(call.args, tuple(int(round(value)) for value in
                                             vertices[index] + vertices[(index + 1) % 4]))
            self.assertEqual(call.kwargs, {"color": "red", "thickness": 2})
        current = controller(serial_module.MODE_SEARCH_ZONE)
        for x, y in vertices + ((288, 200), (288, 250), (200, 279), (200, 359)):
            target = block(x, y)
            self.assertIs(current._find_roi_edge_blocker([target], LABELS), target)
        for x, y in ((288, 199), (288, 380), (152, 200), (424, 200), (0, 279), (575, 279)):
            self.assertIsNone(current._find_roi_edge_blocker([block(x, y)], LABELS))

    def test_debug_zone_obstacle_trapezoid_uses_config_and_clips_to_image(self):
        img = SimpleNamespace(draw_line=Mock())
        with patch.object(config, "UART_ZONE_OBSTACLE_FRONT_Y", -10), \
                patch.object(config, "UART_ZONE_OBSTACLE_CLIP_TOP_Y", -10), \
                patch.object(config, "UART_ZONE_OBSTACLE_FRONT_WIDTH_PX", 1000), \
                patch.object(config, "UART_ZONE_OBSTACLE_EDGE_INSIDE_PX", 25):
            draw_debug_zone_obstacle_region(img, [10, 10, 600, 89], 576, 448)
            self.assertEqual(zone_obstacle_polygon([10, 10, 600, 89], 576, 448), (
                (0, 0), (575, 0), (575, 35), (10, 35)
            ))
            current = controller(serial_module.MODE_SEARCH_ZONE)
            current.tracking_roi = [10, 10, 600, 89]
            target = block(100, 30)
            self.assertIs(current._find_roi_edge_blocker([target], LABELS), target)
        self.assertEqual(img.draw_line.call_count, 4)
        img.draw_line.reset_mock()
        draw_debug_zone_obstacle_region(img, None, 576, 448)
        draw_debug_zone_obstacle_region(img, [600, 359, 10, 89], 576, 448)
        draw_debug_zone_obstacle_region(img, [0, 0, 576, 89], 576, 448)
        img.draw_line.assert_not_called()

    def test_zone_obstacle_trapezoid_sloped_edges_are_inclusive(self):
        polygon = zone_obstacle_polygon([0, 359, 576, 89], 576, 448)
        current = controller(serial_module.MODE_SEARCH_ZONE)
        # Halfway down, the sloped edges are X=119 and X=456.5.
        for x in (119, 120, 288, 456, 456.5):
            target = block(x, 239.5)
            self.assertTrue(point_in_zone_obstacle_polygon(x, 239.5, polygon))
            self.assertIs(current._find_roi_edge_blocker([target], LABELS), target)
        for x in (118.9, 456.6):
            self.assertFalse(point_in_zone_obstacle_polygon(x, 239.5, polygon))
            self.assertIsNone(current._find_roi_edge_blocker([block(x, 239.5)], LABELS))

    def test_zone_obstacle_trapezoid_only_counts_movable_classes(self):
        current = controller(serial_module.MODE_SEARCH_ZONE)
        for class_id in (0, 1, 2, -1, 7):
            self.assertIsNone(current._find_roi_edge_blocker([block(288, 200, class_id)], LABELS))
        for class_id in (3, 4, 5, 6):
            target = block(288, 200, class_id)
            self.assertIs(current._find_roi_edge_blocker([target], LABELS), target)

    def test_zone_obstacle_y200_clip_keeps_original_width_and_debug_outline(self):
        roi = make_bottom_center_roi(576, 448)
        polygon = zone_obstacle_polygon(roi, 576, 448)
        self.assertEqual(config.UART_ZONE_OBSTACLE_CLIP_TOP_Y, 200)
        self.assertAlmostEqual(polygon[0][0], 138.418410041841)
        self.assertAlmostEqual(polygon[1][0], 437.163179916318)
        self.assertAlmostEqual(polygon[1][0] - polygon[0][0], 298.744769874477)
        self.assertEqual(tuple(point[1] for point in polygon), (200, 200, 339, 339))
        self.assertEqual(polygon[2:], ((575, 339), (0, 339)))
        img = SimpleNamespace(draw_line=Mock())
        draw_debug_zone_obstacle_region(img, roi, 576, 448)
        self.assertEqual(img.draw_line.call_args_list[0].args, (138, 200, 437, 200))
        self.assertEqual(img.draw_line.call_args_list[2].args, (575, 339, 0, 339))
        for x, y, inside in ((288, 199, False), (139, 200, True),
                             (437, 200, True), (138, 200, False),
                             (438, 200, False), (0, 339, True),
                             (575, 339, True), (288, 340, False)):
            self.assertEqual(point_in_zone_obstacle_polygon(x, y, polygon), inside)

    def test_zone_obstacle_clip_preserves_original_lower_region_for_custom_rois(self):
        for roi in (make_bottom_center_roi(576, 448), [0, 359, 576, 89],
                    [30, 300, 500, 100]):
            with patch.object(config, "UART_ZONE_OBSTACLE_CLIP_TOP_Y", 0):
                original = zone_obstacle_polygon(roi, 576, 448)
            clipped = zone_obstacle_polygon(roi, 576, 448)
            self.assertEqual(clipped[2:], original[2:])
            for y in (100, 199, 200, 225, 250, 300, 319, 320, 379, 380):
                for x in range(0, 576, 13):
                    expected = y >= 200 and point_in_zone_obstacle_polygon(x, y, original)
                    self.assertEqual(point_in_zone_obstacle_polygon(x, y, clipped), expected,
                                     (roi, x, y))
        for cutoff in (339, 340, 500):
            with patch.object(config, "UART_ZONE_OBSTACLE_CLIP_TOP_Y", cutoff):
                self.assertIsNone(zone_obstacle_polygon(make_bottom_center_roi(576, 448),
                                                        576, 448))

    @patch.object(config, "UART_ZONE_OBSTACLE_ENABLED", True)
    def test_zone36_ignores_above200_then_counts_new_upper_edge_hits(self):
        current = controller(serial_module.MODE_SEARCH_ZONE)
        current.tracking_roi = make_bottom_center_roi(576, 448)
        zone = SimpleNamespace(x=400, y=310, w=100, h=80, class_id=1, score=.9)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            current.submit([], LABELS, raw_objects=[zone])
        self.assertEqual(current._next_outgoing_packet(1000),
                         build_event_packet(serial_module.EVENT_ZONE_FOUND))
        for index, (x, y) in enumerate(((288, 100), (288, 199), (138, 200), (438, 200))):
            rejected = block(x, y, 3)
            with patch.object(serial_module, "_now_ms", return_value=1100 + index * 10):
                current.submit([rejected], LABELS, raw_objects=[zone, rejected])
            self.assertEqual(current._zone_obstacle_frames, 0)
            self.assertFalse(current._zone_obstacle_waiting)
        for index in range(3):
            accepted = block(139, 200, 3)
            with patch.object(serial_module, "_now_ms", return_value=1200 + index * 10):
                current.submit([accepted], LABELS, raw_objects=[zone, accepted])
            self.assertEqual(current._zone_obstacle_frames, index + 1)
            self.assertEqual(current._zone_obstacle_waiting, index == 2)
        self.assertEqual(current._next_outgoing_packet(1220),
                         build_event_packet(serial_module.EVENT_ZONE_OBSTACLE))

    def test_fast_uart_sends_each_new_frame_once_even_with_identical_values(self):
        current = controller(CMD_TRACK)
        target = block(100, 150)
        with patch.object(config, "UART_SEND_RATE_HZ", 0):
            with patch.object(serial_module, "_now_ms", return_value=1000):
                current.submit([target], LABELS)
                self.assertTrue(current._tx_wakeup.is_set())
                packet = current._latest_packet
                self.assertEqual(current._next_outgoing_packet(1000), packet)
                self.assertIsNone(current._next_outgoing_packet(1001))
                # Identical timestamp and coordinates still represent a new frame.
                current.submit([target], LABELS)
                self.assertEqual(current._next_outgoing_packet(1001), packet)
                self.assertIsNone(current._next_outgoing_packet(1002))

    def test_fast_uart_keeps_only_newest_frame_and_rejects_stale_coordinates(self):
        current = controller(CMD_TRACK)
        with patch.object(config, "UART_SEND_RATE_HZ", 0):
            with patch.object(serial_module, "_now_ms", return_value=1000):
                current.submit([block(100, 150)], LABELS)
                current.submit([block(110, 155)], LABELS)
                newest = current._latest_packet
            self.assertEqual(current._next_outgoing_packet(1001), newest)
            self.assertIsNone(current._next_outgoing_packet(1002))
            with patch.object(serial_module, "_now_ms", return_value=1100):
                current.submit([block(120, 160)], LABELS)
            self.assertIsNone(current._next_outgoing_packet(
                1100 + config.UART_TARGET_FRESHNESS_MS + 1
            ))

    def test_fast_uart_preserves_event_order_and_queued_command_spacing(self):
        current = controller(CMD_TRACK)
        first = build_event_packet(serial_module.EVENT_ZONE_FOUND)
        second = build_event_packet(serial_module.EVENT_ZONE_CLOSE)
        current._serial = SimpleNamespace(write=lambda packet: len(packet))
        current._pending_events.extend([first, second])
        self.assertEqual(current._next_outgoing_packet(1000), first)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            current._write_packet(first)
        self.assertIsNone(current._next_outgoing_packet(1001))
        self.assertEqual(current._next_outgoing_packet(
            1000 + config.UART_EVENT_MIN_INTERVAL_MS
        ), second)
        # With no command pending, coordinates need not wait for event spacing.
        with patch.object(serial_module, "_now_ms", return_value=1001):
            current.submit([block(100, 150)], LABELS)
        self.assertEqual(current._next_outgoing_packet(1001), current._latest_packet)

    def test_near05_cargo_report_precedes06_and_waits50ms(self):
        for cargo_id in (4, 5):
            current = controller(serial_module.MODE_SEARCH_ZONE)
            report = serial_module.build_coordinate_packet(cargo_id, 0, 0)
            done = build_event_packet(serial_module.EVENT_ZONE_SEARCH_STARTED)
            current._pending_events.extend([report, done])
            current._serial = SimpleNamespace(write=lambda packet: len(packet))
            self.assertEqual(current._next_outgoing_packet(1000), report)
            with patch.object(serial_module, "_now_ms", return_value=1000), patch("builtins.print") as output:
                current._write_packet(report)
            self.assertIn("NEAR05_LOAD", str(output.call_args_list))
            self.assertIsNone(current._next_outgoing_packet(1049))
            self.assertEqual(current._next_outgoing_packet(1050), done)
            self.assertIsNone(current._next_outgoing_packet(1200))

    def test_fast_uart_worker_does_not_wait_50ms_for_updated_coordinates(self):
        current = controller(CMD_TRACK)
        clock = {"ms": 1000, "waits": 0}
        writes = []

        def write(packet, generation=None):
            self.assertEqual(generation, current.frame_generation())
            writes.append((clock["ms"], packet))
            if len(writes) == 2:
                current._stop_event.set()

        def wait(timeout):
            clock["waits"] += 1
            clock["ms"] += 1
            if clock["waits"] == 1:
                current.submit([block(110, 155)], LABELS)
            if clock["waits"] >= 10:
                current._stop_event.set()

        with patch.object(config, "UART_SEND_RATE_HZ", 0), \
                patch.object(serial_module, "_now_ms", side_effect=lambda: clock["ms"]), \
                patch.object(serial_module.python_time, "monotonic",
                             side_effect=lambda: clock["ms"] / 1000.0), \
                patch.object(current, "_read_available"), \
                patch.object(current, "_write_packet", side_effect=write), \
                patch.object(current._tx_wakeup, "wait", side_effect=wait):
            current.submit([block(100, 150)], LABELS)
            current._io_loop()
        self.assertEqual([stamp for stamp, _ in writes], [1000, 1001])
        self.assertNotEqual(writes[0][1], writes[1][1])

    def test_04_lock_ignores_spacing_xy_area_and_keeps_frame_center_errors(self):
        for class_id, label in ((4, "sqarered"), (5, "sqareredgreen")):
            with self.subTest(class_id=class_id):
                current = controller(CMD_TRACK)
                current.target_label = label
                first = block(40, 80, class_id)
                moved = block(400, 260, class_id)
                moved.x, moved.y, moved.w, moved.h = 360, 220, 80, 80
                nearby = block(430, 270, 3)
                with patch.object(serial_module, "_now_ms", return_value=1000):
                    self.assertIs(current.submit([first], LABELS), first)
                self.assertFalse(current._matches_final_primary_spacing(moved, [moved, nearby]))
                with patch.object(serial_module, "_now_ms", return_value=1100):
                    with patch.object(current, "_select_final_roi_target",
                                      side_effect=AssertionError("04 lock must not rescreen")):
                        self.assertIs(current.submit([moved, nearby], LABELS), moved)
                self.assertEqual(current.mode, CMD_TRACK)
                self.assertEqual(current._track_target_lock, (class_id, 400.0, 260.0, 6400.0))
                self.assertEqual(current._latest_packet,
                                 serial_module.build_coordinate_packet(class_id, 112, 36))
                self.assertNotIn(build_event_packet(serial_module.EVENT_TRACK_NO_TARGET),
                                 current._pending_events)

    def test_04_nearest_same_class_and_lock_survives_tx14_until_handshake(self):
        current = controller(CMD_TRACK)
        current.target_label = "sqarered"
        first = block(100, 100, 4)
        near = block(110, 150, 4)
        lower = block(450, 330, 4)
        centered = block(288, 224, 4)
        blocker = block(300, 250, 3)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            current.submit([first], LABELS)
            self.assertIs(current.submit([lower, near], LABELS), near)
            self.assertIs(current.submit([centered, blocker], LABELS), centered)
            self.assertEqual(current.mode, MODE_WAIT_14_HANDSHAKE)
            self.assertEqual(current._track_target_lock[:3], (4, 288.0, 224.0))
            self.assertIn(build_event_packet(EVENT_TRACK_CENTER_REACHED), current._pending_events)
            self.assertIs(current.submit([centered, blocker], LABELS), centered)
            self.assertEqual(current._latest_packet,
                             serial_module.build_coordinate_packet(4, 0, -180))
            current._handle_command(serial_module.CMD_TRACK_CENTER_DONE)
            self.assertIsNone(current._track_target_lock)
            self.assertEqual(current._middle_target_anchor, (288.0, None, None))
            current.submit([centered, blocker], LABELS)
            self.assertIn(build_event_packet(serial_module.EVENT_MICRO_SIDE), current._pending_events)

    def test_04_requires_initial_screening_and_rescreens_after_loss(self):
        current = controller(CMD_TRACK)
        current.target_label = "sqarered"
        first = block(100, 150, 4)
        nearby = block(130, 145, 3)  # same (100,150] band, still rejects dx
        replacement = block(450, 260, 4)
        replacement_neighbor = block(480, 270, 3)
        e4 = build_event_packet(serial_module.EVENT_TRACK_NO_TARGET)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            for _ in range(config.UART_TRACK_LOST_CONFIRM_FRAMES):
                self.assertIsNone(current.submit([first, nearby], LABELS))
            self.assertIsNone(current._track_target_lock)
            self.assertEqual(current._next_outgoing_packet(1000), e4)
            self.assertIs(current.submit([first], LABELS), first)
            for _ in range(config.UART_TRACK_LOST_CONFIRM_FRAMES):
                self.assertIsNone(current.submit([], LABELS))
            self.assertIsNone(current._track_target_lock)
            self.assertEqual(current._next_outgoing_packet(1500), e4)
            self.assertIsNone(current.submit([replacement, replacement_neighbor], LABELS))
            self.assertIsNone(current._track_target_lock)
            self.assertIsNone(current._next_outgoing_packet(1500))
            self.assertIs(current.submit([replacement], LABELS), replacement)
            self.assertEqual(current._track_target_lock[:3], (4, 450.0, 260.0))
            current._handle_command(serial_module.CMD_SELECT_GREEN_OBJECT)
            self.assertIsNone(current._track_target_lock)

    def test_24_roi_entry_34_does_not_require_center_alignment(self):
        for x, y in ((0, 384), (575, 384), (351, 402), (575, 447)):
            with self.subTest(x=x, y=y):
                current = controller(MODE_FINAL_ROI_TRACK)
                target = block(x, y)
                with patch.object(serial_module, "_now_ms", return_value=1000):
                    self.assertIs(current.submit([target], LABELS), target)
                self.assertIsNone(current._latest_packet)
                self.assertEqual(current._next_outgoing_packet(1000),
                                 build_event_packet(serial_module.EVENT_FINAL_TARGET_IN_ROI))
        for x, y in ((288, 358), (288, 448), (-1, 400), (576, 400)):
            with self.subTest(outside_x=x, outside_y=y):
                current = controller(MODE_FINAL_ROI_TRACK)
                target = block(x, y)
                self.assertFalse(current._target_center_in_tracking_roi(target))
                current.submit([target], LABELS)
                self.assertFalse(current._final_roi_sent)
                self.assertIsNotNone(current._latest_packet)
        current = controller(MODE_FINAL_ROI_TRACK)
        current.tracking_roi = None
        self.assertFalse(current._target_center_in_tracking_roi(block(288, 400)))

    def test_24_34_repeats_500ms_and_stops_when_center_exits_capture_region(self):
        current = controller(MODE_FINAL_ROI_TRACK)
        event = build_event_packet(serial_module.EVENT_FINAL_TARGET_IN_ROI)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            current.submit([block(351, 384)], LABELS)
        self.assertEqual(current._next_outgoing_packet(1000), event)
        with patch.object(serial_module, "_now_ms", return_value=1400):
            current.submit([block(351, 384)], LABELS)
        self.assertIsNone(current._next_outgoing_packet(1499))
        self.assertEqual(current._next_outgoing_packet(1500), event)
        with patch.object(serial_module, "_now_ms", return_value=1600):
            current.submit([block(351, 358.99)], LABELS)
        self.assertFalse(current._final_roi_sent)
        self.assertIsNotNone(current._latest_packet)
        self.assertNotEqual(current._next_outgoing_packet(1600), event)
        with patch.object(serial_module, "_now_ms", return_value=1650):
            current.submit([block(351, 384)], LABELS)
        self.assertEqual(current._next_outgoing_packet(1650), event)
        with patch.object(serial_module, "_now_ms", return_value=1700):
            current.submit([], LABELS)
        self.assertFalse(current._final_roi_sent)
        self.assertIsNone(current._next_outgoing_packet(2150))

    def test_24_keeps_locked_target_without_rescreening(self):
        current = controller(MODE_FINAL_ROI_TRACK)
        initial = block(100, 300)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            self.assertIs(current.submit([initial], LABELS), initial)
        moved = block(110, 310)
        lower = block(450, 330)
        nearby = block(160, 320, 3)
        blocking = block(110, 340, 6)
        objects = [lower, nearby, blocking, moved]
        # The old screening would reject moved, and prefer lower instead.
        self.assertFalse(current._matches_final_primary_spacing(moved, objects))
        self.assertTrue(current._final_candidate_is_blocked(moved, objects))
        with patch.object(serial_module, "_now_ms", return_value=2000):
            self.assertIs(current.submit(objects, LABELS), moved)
        self.assertEqual(current._final_target_lock[:3], (5, 110.0, 310.0))
        self.assertIsNotNone(current._latest_packet)

    def test_24_loss_immediately_rescreens_and_locks_replacement(self):
        current = controller(MODE_FINAL_ROI_TRACK)
        current.target_label = serial_module.TARGET_BLACK_OR_GREEN
        initial = block(100, 300)
        replacement = block(450, 320, 6)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            current.submit([initial], LABELS)
            # Loss must permit a jump to a freshly screened target in this frame.
            self.assertIs(current.submit([replacement], LABELS), replacement)
        self.assertEqual(current._final_target_lock[:3], (6, 450.0, 320.0))
        nearby = block(480, 330, 3)
        with patch.object(serial_module, "_now_ms", return_value=1100):
            self.assertIs(current.submit([replacement, nearby], LABELS), replacement)

    def test_24_loss_with_no_matching_yolo_target_keeps_ee_flow(self):
        current = controller(MODE_FINAL_ROI_TRACK)
        current.target_label = serial_module.TARGET_BLACK_OR_GREEN
        initial = block(100, 300)
        distant = block(450, 310, 6)
        nearby = block(480, 310, 3)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            current.submit([initial], LABELS)
            for _ in range(config.UART_FINAL_NO_TARGET_FRAMES):
                self.assertIsNone(current.submit([nearby], LABELS))
        self.assertIsNone(current._final_target_lock)
        ee = build_event_packet(serial_module.EVENT_NO_TARGET)
        self.assertEqual(current._next_outgoing_packet(1000), ee)
        self.assertIsNone(current._next_outgoing_packet(2000))
        current._handle_command(serial_module.CMD_FINAL_SEARCH_DONE)
        with patch.object(serial_module, "_now_ms", return_value=2100):
            self.assertIs(current.submit([distant, nearby], LABELS), distant)
        self.assertEqual(current._selection_diagnostic_policy, "supply_no_spacing24")

    def test_24_large_xy_and_area_changes_do_not_drop_or_rescreen_lock(self):
        current = controller(MODE_FINAL_ROI_TRACK)
        initial = block(40, 100)
        moved = block(400, 320)
        moved.x, moved.y, moved.w, moved.h = 360, 280, 80, 80
        nearby = block(430, 330, 3)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            self.assertIs(current.submit([initial], LABELS), initial)
        # 360px X / 220px Y movement and 16x area change would fail the
        # former association and generic jump gate. Spacing now fails too,
        # but must not be rechecked while the locked class is detected.
        self.assertFalse(current._matches_final_primary_spacing(moved, [moved, nearby]))
        with patch.object(serial_module, "_now_ms", return_value=1100):
            with patch.object(current, "_select_final_roi_target",
                              side_effect=AssertionError("locked target must not rescreen")):
                self.assertIs(current.submit([moved, nearby], LABELS), moved)
        self.assertIsNotNone(current._latest_packet)
        self.assertEqual(current._final_target_lock, (5, 400.0, 320.0, 6400.0))

    def test_24_associates_nearest_same_class_ignoring_area(self):
        current = controller(MODE_FINAL_ROI_TRACK)
        initial = block(100, 300)
        near = block(110, 310)
        near.x, near.y, near.w, near.h = 70, 270, 80, 80
        far = block(450, 330)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            current.submit([initial], LABELS)
            self.assertIs(current.submit([far, near], LABELS), near)

    def test_24_empty_frame_unlocks_and_allows_fresh_screening(self):
        current = controller(MODE_FINAL_ROI_TRACK)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            current.submit([block(100, 300)], LABELS)
            self.assertIsNone(current.submit([], LABELS))
            self.assertIsNone(current._final_target_lock)
            target = block(450, 320)
            blocker = block(470, 330, 3)
            self.assertIs(current.submit([target, blocker], LABELS), target)
            self.assertEqual(current._final_target_lock[:3], (5, 450.0, 320.0))
            self.assertIs(current.submit([target], LABELS), target)

    def test24_ignores_front_gate_while_green14_xy_overlap_and_red14_reject(self):
        target = block(100, 310)
        front = block(350, 330, 3)
        front.x, front.w = 100, 500
        objects = [target, front]
        current = controller(MODE_FINAL_ROI_TRACK)
        self.assertTrue(current._final_candidate_is_blocked(target, objects))
        self.assertTrue(current._matches_final_primary_spacing(target, objects))
        with patch.object(serial_module, "_now_ms", return_value=1000):
            self.assertIsNone(current._select_final_roi_target(objects, LABELS, current.target_label, 1000))
            self.assertIs(current.submit(objects, LABELS), target)
        self.assertEqual(current._selection_diagnostic_policy, "supply_no_spacing24")
        for command in (serial_module.CMD_TRACK_CENTER_DONE, serial_module.CMD_TRACK_RECOVERY_DONE):
            with self.subTest(command=command):
                decision = controller(MODE_WAIT_14_HANDSHAKE)
                decision._first_green_near_view_decided = True
                decision._post07_started = True  # Later-green44 still judges neighbors.
                decision._handle_command(command)
                with patch.object(serial_module, "_now_ms", return_value=1000):
                    decision.submit(objects, LABELS)
                # Native boxes only touchY320, but the new Y expansion now
                # overlaps the front box.14 arranges;24 selection stays unchanged.
                self.assertEqual(decision.mode, serial_module.MODE_MICRO_SIDE)
                self.assertEqual(decision._next_outgoing_packet(1000),
                                 build_event_packet(serial_module.EVENT_MICRO_SIDE))
                later = controller(MODE_WAIT_14_HANDSHAKE if command == serial_module.CMD_TRACK_CENTER_DONE else CMD_TRACK)
                later.target_label, later._post07_started = "sqarered", True
                red = block(100, 310, 4)
                later._handle_command(command)
                with patch.object(serial_module, "_now_ms", return_value=1000):
                    later.submit([red, front], LABELS)
                self.assertEqual(later.mode, serial_module.MODE_MICRO_SIDE)
                self.assertEqual(later._next_outgoing_packet(1000), build_event_packet(serial_module.EVENT_MICRO_SIDE))

    def test_first_green24_prefers_nearer_target_despite_close_neighbor(self):
        current = controller(MODE_FINAL_ROI_TRACK)
        qualified = block(30, 280)
        nearer = block(310, 310)
        neighbor = block(300, 315, 3)
        objects = [qualified, nearer, neighbor]
        self.assertTrue(current._matches_final_primary_spacing(qualified, objects))
        self.assertFalse(current._matches_final_primary_spacing(nearer, objects))
        with patch.object(serial_module, "_now_ms", return_value=1000):
            self.assertIs(current.submit(objects, LABELS, raw_objects=objects), nearer)
        self.assertEqual(current._selection_diagnostic_policy, "supply_no_spacing24")

    def test_24_fallback_uses_raw_matching_yolo_nearest_roi_center_2d_then_locks(self):
        current = controller(MODE_FINAL_ROI_TRACK)
        near = block(288, 340)
        y_nearer_but_far_x = block(30, 350)
        objects = [y_nearer_but_far_x, near, block(300, 340, 3), block(50, 350, 3)]
        with patch.object(serial_module, "_now_ms", return_value=1000):
            self.assertIs(current.submit([], LABELS, raw_objects=objects), near)
        self.assertEqual(current._selection_diagnostic_policy, "final24_yolo_fallback")
        self.assertEqual(current._final_target_lock[:3], (5, 288.0, 340.0))
        self.assertEqual(current._latest_packet, serial_module.build_coordinate_packet(5, 0, -44))
        moved, other = block(295, 345), block(288, 350)
        with patch.object(serial_module, "_now_ms", return_value=1100), \
                patch.object(current, "_select_final_roi_target", side_effect=AssertionError("fallback lock must not rescreen")):
            self.assertIs(current.submit([], LABELS, raw_objects=[other, moved, block(300, 350, 3)]), moved)

    def test_24_fallback_does_not_switch_color_select_safe_zone_or_send_stale_box(self):
        current = controller(MODE_FINAL_ROI_TRACK)
        target, neighbor = block(100, 300), block(110, 310, 3)
        zone = block(100, 300, 1)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            self.assertIsNone(current.submit([], LABELS, raw_objects=[neighbor, block(288, 330, 4)]))
            self.assertIsNone(current.submit([], LABELS, raw_objects=[target, neighbor, zone]))
            self.assertIs(current.submit([], LABELS, raw_objects=[target, neighbor]), target)
            self.assertIsNone(current.submit([], LABELS, raw_objects=[]))
        self.assertIsNone(current._final_target_lock)
        self.assertIsNone(current._latest_packet)

    def test_24_fallback_roi_entry_still_emits_34_and_repeats(self):
        current = controller(MODE_FINAL_ROI_TRACK)
        target, neighbor = block(288, 400), block(300, 405, 3)
        event = build_event_packet(serial_module.EVENT_FINAL_TARGET_IN_ROI)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            self.assertIs(current.submit([], LABELS, raw_objects=[target, neighbor]), target)
        self.assertEqual(current._next_outgoing_packet(1000), event)
        self.assertIsNone(current._next_outgoing_packet(1499))
        with patch.object(serial_module, "_now_ms", return_value=1500):
            current.submit([], LABELS, raw_objects=[target, neighbor])
        self.assertEqual(current._next_outgoing_packet(1500), event)

    def test_24_lost_lock_immediately_uses_unverified_raw_fallback_replacement(self):
        current = controller(MODE_FINAL_ROI_TRACK)
        current.target_label = serial_module.TARGET_BLACK_OR_GREEN
        initial, replacement = block(100, 300), block(450, 310, 6)
        neighbor = block(480, 310, 3)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            current.submit([initial], LABELS)
        with patch.object(serial_module, "_now_ms", return_value=1100):
            self.assertIs(current.submit([], LABELS, raw_objects=[replacement, neighbor]), replacement)
        self.assertEqual(current._selection_diagnostic_policy, "final24_yolo_fallback")
        self.assertEqual(current._final_target_lock[:3], (6, 450.0, 310.0))
        self.assertEqual(current._final_no_target_frames, 0)
        self.assertEqual(current._latest_packet, serial_module.build_coordinate_packet(6, 162, -74))
        self.assertNotIn(build_event_packet(serial_module.EVENT_NO_TARGET), current._pending_events)

    def test_24_diagnostics_report_front_gate_removed_and_raw_fallback(self):
        target, front = block(100, 310), block(350, 330, 3)
        front.x, front.w = 100, 500
        current = controller(MODE_FINAL_ROI_TRACK)
        with patch.object(config, "VISION_DIAGNOSTICS_ENABLED", True), \
                patch.object(serial_module, "_now_ms", return_value=1000), \
                patch("builtins.print") as log:
            self.assertIs(current.submit([target, front], LABELS), target)
            messages = [str(call.args[0]) for call in log.call_args_list]
            self.assertTrue(any("[CANDIDATE]" in text and "merged=skip applied=no" in text
                                and "front=[none]" in text and "front_applied=no" in text for text in messages))
            fallback = controller(MODE_FINAL_ROI_TRACK)
            self.assertIs(fallback.submit([], LABELS, raw_objects=[target, block(110, 310, 3)]), target)
            self.assertTrue(any("[FINAL24-FALLBACK]" in str(call) and "nearest=roi_center_2d" in str(call)
                                and "verification_applied=no" in str(call) for call in log.call_args_list))

    def test_controller_24_debug_preview_uses_spacing_not_front_overlap(self):
        current = controller(serial_module.CMD_SEARCH)
        target, front = block(100, 310), block(350, 330, 3)
        front.x, front.w = 100, 500
        candidates, neighbors, selected, annotations = current.final_track_debug_objects([target, front], LABELS)
        self.assertEqual(candidates, [target])
        self.assertEqual(neighbors, [])
        self.assertIs(selected, target)
        self.assertEqual(annotations, [])

    def test_24_transition_locks_decision_target_before_next_frame(self):
        for source_mode in (MODE_POST_ARRANGE_CHECK, MODE_WAIT_14_HANDSHAKE):
            with self.subTest(source_mode=source_mode):
                current = controller(source_mode)
                first = block(100, 300)
                with patch.object(serial_module, "_now_ms", return_value=1000):
                    if source_mode == MODE_WAIT_14_HANDSHAKE:
                        current._handle_command(serial_module.CMD_TRACK_CENTER_DONE)
                        for _ in range(config.UART_FIRST_GREEN_DECISION_CONFIRM_FRAMES - 1):
                            current.submit([first], LABELS)
                    current.submit([first], LABELS)
                    self.assertEqual(current.mode, MODE_FINAL_ROI_TRACK)
                    self.assertEqual(current._final_target_lock[:3], (5, 100.0, 300.0))
                    lower = block(450, 330)
                    self.assertIs(current.submit([lower, first], LABELS), first)

    def test_24_locked_black_is_not_preempted_and_new_commands_reset_lock(self):
        current = controller(MODE_FINAL_ROI_TRACK)
        current.target_label = "triangualrblack"
        black = block(100, 300, 6)
        red = block(450, 330, 4)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            current.submit([black], LABELS)
            self.assertIs(current.submit([black, red], LABELS), black)
            self.assertEqual(current.target_label, "triangualrblack")
            self.assertNotIn(build_event_packet(serial_module.EVENT_SWITCH_TO_RED_TARGET),
                             current._pending_events)
            current._handle_command(serial_module.CMD_SELECT_GREEN_OBJECT)
            self.assertIsNone(current._final_target_lock)
            green = block(450, 330)
            self.assertIs(current.submit([green], LABELS), green)

    def test_first_green_e4_waits_44_and_does_not_repeat_on_reacquisition(self):
        current = controller(CMD_TRACK)
        e4 = build_event_packet(serial_module.EVENT_TRACK_NO_TARGET)
        target = block(288, 224)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            for _ in range(config.UART_TRACK_LOST_CONFIRM_FRAMES):
                current.submit([], LABELS, raw_objects=[])
            self.assertEqual(current._next_outgoing_packet(1000), e4)
            self.assertIsNone(current._next_outgoing_packet(4000))
            # Coordinates can recover before RX44, but 14 must still wait.
            current.submit([target], LABELS, raw_objects=[target])
            self.assertIsNotNone(current._latest_packet)
            self.assertEqual(current.mode, CMD_TRACK)
            for _ in range(config.UART_TRACK_LOST_CONFIRM_FRAMES):
                current.submit([], LABELS, raw_objects=[])
            self.assertIsNone(current._next_outgoing_packet(5000))
            current._handle_command(serial_module.CMD_TRACK_RECOVERY_DONE)
            self.assertFalse(current._track_green_recovery_waiting)
            self.assertEqual(current.mode, serial_module.CMD_SEARCH)
            self.assertFalse(current._initial_arrangement_decision_pending)
            self.assertIsNone(current._middle_target_anchor)
            self.assertIsNone(current._next_outgoing_packet(1000))
            middle = block(288, 300)
            side = block(290, 310, 3)  # Crowding must not bypass the new03 tracking.
            for _ in range(config.UART_FIRST_GREEN_DECISION_CONFIRM_FRAMES):
                current.submit([middle, side], LABELS, raw_objects=[middle, side])
            self.assertNotIn(
                build_event_packet(EVENT_TRACK_CENTER_REACHED),
                current._pending_events,
            )
            self.assertEqual(
                current._next_outgoing_packet(1001),
                serial_module.build_coordinate_packet(5, 0, 76),
            )
            self.assertIsNone(current._next_outgoing_packet(6501))
            # A fresh search/04 round may report its own first loss.
            current._handle_command(serial_module.CMD_SEARCH)
            current._handle_command(CMD_TRACK)
            for _ in range(config.UART_TRACK_LOST_CONFIRM_FRAMES):
                current.submit([], LABELS, raw_objects=[])
            self.assertEqual(current._next_outgoing_packet(7000), e4)

    def test_first_green_e4_resume04_keeps_reacquired_lock_and_requires_fresh_frame(self):
        current = controller(CMD_TRACK)
        target = block(100, 150)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            for _ in range(config.UART_TRACK_LOST_CONFIRM_FRAMES):
                current.submit([], LABELS, raw_objects=[])
        self.assertEqual(current._next_outgoing_packet(1000),
                         build_event_packet(serial_module.EVENT_TRACK_NO_TARGET))
        with patch.object(serial_module, "_now_ms", return_value=1100):
            current.submit([target], LABELS, raw_objects=[target])
        anchor = current._track_target_lock
        self.assertTrue(current._track_green_recovery_waiting)
        self.assertIsNotNone(current._latest_packet)
        # A queued obsolete E4 must not be sent after MCU resumes tracking.
        current.send_event(serial_module.EVENT_TRACK_NO_TARGET)
        with patch.object(serial_module, "_now_ms", return_value=1200), patch("builtins.print") as log:
            current.process_received_bytes(bytes((0xE1, 0xE2, 0x04, 0x1E, 0x2E)))
        self.assertEqual(current.mode, CMD_TRACK)
        self.assertFalse(current._track_green_recovery_waiting)
        self.assertFalse(current._track_no_target_sent)
        self.assertFalse(current._track_center_sent)
        self.assertEqual(current._track_target_lock, anchor)
        self.assertEqual((current._last_target_center_x, current._last_target_center_y), (100, 150))
        self.assertEqual(current._pending_events, [])
        self.assertIsNone(current._latest_packet)
        self.assertIsNone(current._next_outgoing_packet(1200))
        self.assertTrue(any("recovery_resume=RX04" in str(call) for call in log.call_args_list))
        # A verifier dropout/close neighbor must not rerank the inherited lock
        # to another green closer to the image center after the resume command.
        moved, nearer, adjacent = block(110, 160), block(288, 224), block(120, 170, 3)
        with patch.object(serial_module, "_now_ms", return_value=1250):
            self.assertIs(current.submit([], LABELS, raw_objects=[nearer, moved, adjacent]), moved)
        self.assertEqual(current._next_outgoing_packet(1250),
                         serial_module.build_coordinate_packet(5, -178, -64))
        with patch.object(serial_module, "_now_ms", return_value=1300):
            current.submit([], LABELS, raw_objects=[nearer])
        self.assertEqual(current.mode, MODE_WAIT_14_HANDSHAKE)
        self.assertEqual(current._next_outgoing_packet(1300), build_event_packet(EVENT_TRACK_CENTER_REACHED))
        self.assertIsNone(current._next_outgoing_packet(1301))

    def test_first_green_e4_centered_target_waits_ack_then_emits14_on_next_image(self):
        current = controller(CMD_TRACK)
        target = block(288, 224)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            for _ in range(config.UART_TRACK_LOST_CONFIRM_FRAMES):
                current.submit([], LABELS, raw_objects=[])
        current._next_outgoing_packet(1000)
        with patch.object(serial_module, "_now_ms", return_value=1100):
            current.submit([target], LABELS)
        self.assertTrue(current._track_green_recovery_waiting)
        self.assertEqual(current.mode, CMD_TRACK)
        self.assertNotIn(build_event_packet(EVENT_TRACK_CENTER_REACHED), current._pending_events)
        lock = current._track_target_lock
        with patch.object(serial_module, "_now_ms", return_value=1200):
            current._handle_command(CMD_TRACK)
        self.assertEqual(current._track_target_lock, lock)
        self.assertIsNone(current._next_outgoing_packet(1200))
        with patch.object(serial_module, "_now_ms", return_value=1250):
            current.submit([], LABELS, raw_objects=[])
        self.assertEqual(current.mode, CMD_TRACK)
        self.assertIsNone(current._next_outgoing_packet(1250))
        with patch.object(serial_module, "_now_ms", return_value=1300):
            current.submit([], LABELS, raw_objects=[target])
        self.assertEqual(current._next_outgoing_packet(1300), build_event_packet(EVENT_TRACK_CENTER_REACHED))

    def test_first_green_resume04_without_lock_keeps_verification_and_rearms_one_e4(self):
        current = controller(CMD_TRACK)
        target = block(288, 224)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            for _ in range(config.UART_TRACK_LOST_CONFIRM_FRAMES):
                current.submit([], LABELS, raw_objects=[])
        self.assertEqual(current._next_outgoing_packet(1000),
                         build_event_packet(serial_module.EVENT_TRACK_NO_TARGET))
        self.assertIsNone(current._track_target_lock)
        with patch.object(serial_module, "_now_ms", return_value=1100):
            current._handle_command(CMD_TRACK)
        self.assertFalse(current._track_green_recovery_waiting)
        self.assertIsNone(current._next_outgoing_packet(1100))
        for index in range(config.UART_TRACK_LOST_CONFIRM_FRAMES):
            now = 1200 + index * 50
            with patch.object(serial_module, "_now_ms", return_value=now):
                current.submit([], LABELS, raw_objects=[target])
            self.assertIsNone(current._track_target_lock)
            self.assertIsNone(current._latest_packet)
            self.assertNotIn(build_event_packet(EVENT_TRACK_CENTER_REACHED), current._pending_events)
            if index < config.UART_TRACK_LOST_CONFIRM_FRAMES - 1:
                self.assertIsNone(current._next_outgoing_packet(now))
        self.assertEqual(current._next_outgoing_packet(1400),
                         build_event_packet(serial_module.EVENT_TRACK_NO_TARGET))
        self.assertTrue(current._track_green_recovery_waiting)
        self.assertIsNone(current._next_outgoing_packet(2000))

    def test_first_green_decision_timeout_e4_can_resume04_without_old_deadline(self):
        current = controller(MODE_WAIT_14_HANDSHAKE)
        target = block(288, 224)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            current._handle_command(serial_module.CMD_TRACK_CENTER_DONE)
        self.assertEqual(current._next_outgoing_packet(1500),
                         build_event_packet(serial_module.EVENT_TRACK_NO_TARGET))
        with patch.object(serial_module, "_now_ms", return_value=1550):
            current.submit([target], LABELS)
        self.assertTrue(current._track_green_recovery_waiting)
        with patch.object(serial_module, "_now_ms", return_value=1600):
            current._handle_command(CMD_TRACK)
        self.assertFalse(current._track_green_recovery_waiting)
        self.assertIsNone(current._initial_confirmation_started_ms)
        self.assertFalse(current._initial_arrangement_decision_pending)
        self.assertIsNone(current._next_outgoing_packet(2000))
        with patch.object(serial_module, "_now_ms", return_value=2050):
            current.submit([], LABELS, raw_objects=[target])
        self.assertEqual(current._next_outgoing_packet(2050), build_event_packet(EVENT_TRACK_CENTER_REACHED))

    def test_03_target_lock_survives_wait_and_rx04_without_rescreening(self):
        for class_id, label in ((4, "sqarered"), (5, "sqareredgreen")):
            with self.subTest(class_id=class_id):
                current = controller(serial_module.CMD_SEARCH)
                current.target_label = label
                target = block(288, 224, class_id)
                neighbor = block(400, 224, 3)
                self.assertFalse(current._matches_final_primary_spacing(target, [target, neighbor]))
                with patch.object(serial_module, "_now_ms", return_value=1000):
                    self.assertIs(current.submit([target, neighbor], LABELS), target)
                    self.assertEqual(current.mode, serial_module.MODE_WAIT_04_HANDSHAKE)
                    self.assertEqual(current._track_target_lock[:3], (class_id, 288.0, 224.0))
                    self.assertEqual(current._next_outgoing_packet(1000),
                                     build_event_packet(serial_module.EVENT_SEARCH_CENTER_REACHED))
                    moved = block(500, 100, class_id)
                    moved.x, moved.y, moved.w, moved.h = 460, 60, 80, 80
                    adjacent = block(550, 110, 3)
                    with patch.object(current, "_select_final_roi_target",
                                      side_effect=AssertionError("inherited lock must not rescreen")), \
                            patch.object(current, "_reject_large_target_jump",
                                         side_effect=AssertionError("wait04 must not reject motion")):
                        self.assertIs(current.submit([moved, adjacent], LABELS), moved)
                        self.assertEqual(current._latest_packet,
                                         serial_module.build_coordinate_packet(class_id, 212, -124))
                        lock = current._track_target_lock
                        current._handle_command(CMD_TRACK)
                        self.assertEqual(current._track_target_lock, lock)
                        self.assertEqual(current._last_target_center_x, 500)
                        self.assertIs(current.submit([moved, adjacent], LABELS), moved)
                        self.assertEqual(current._latest_packet,
                                         serial_module.build_coordinate_packet(class_id, 212, -124))
                    self.assertFalse(current._track_no_target_sent)
                    self.assertEqual(current._track_no_target_frames, 0)

    def test_wait04_short_miss_preserves_identity_and_new_commands_reset_it(self):
        current = controller(serial_module.CMD_SEARCH)
        current.target_label = "sqarered"
        target = block(288, 224, 4)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            current.submit([target], LABELS)
            current._next_outgoing_packet(1000)
            lock = current._track_target_lock
            self.assertIsNone(current.submit([], LABELS))
            self.assertEqual(current._track_target_lock, lock)
            self.assertIsNone(current._latest_packet)
            current._handle_command(CMD_TRACK)
            self.assertEqual(current._track_target_lock, lock)
            self.assertEqual(current._track_no_target_frames, 0)
            self.assertIsNone(current._latest_packet)
            self.assertIsNone(current._next_outgoing_packet(1100))
            current._handle_command(serial_module.CMD_SELECT_GREEN_OBJECT)
            self.assertIsNone(current._track_target_lock)
            self.assertEqual(current._track_no_target_frames, 0)

    def test_04_short_misses_keep_lock_but_never_send_stale_coordinates(self):
        current = controller(CMD_TRACK)
        current.target_label = "sqarered"
        first = block(100, 150, 4)
        moved = block(450, 260, 4)
        neighbor = block(470, 270, 3)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            current.submit([first], LABELS)
            lock = current._track_target_lock
            for missing in range(1, config.UART_TRACK_LOST_CONFIRM_FRAMES):
                # Only absence from the current raw frame counts as loss.
                self.assertIsNone(current.submit([], LABELS, raw_objects=[]))
                self.assertEqual(current._track_no_target_frames, missing)
                self.assertEqual(current._track_target_lock, lock)
                self.assertIsNone(current._latest_packet)
                self.assertIsNone(current._next_outgoing_packet(1000 + missing))
            with patch.object(current, "_select_final_roi_target",
                              side_effect=AssertionError("brief miss must not rescreen")):
                self.assertIs(current.submit([moved, neighbor], LABELS), moved)
            self.assertEqual(current._track_no_target_frames, 0)
            self.assertEqual(current._latest_packet,
                             serial_module.build_coordinate_packet(4, 162, 36))

    def test_04_e4_once_per_confirmed_loss_rearms_after_fresh_red_coordinates(self):
        current = controller(CMD_TRACK)
        current.target_label = "sqarered"
        target = block(100, 150, 4)
        e4 = build_event_packet(serial_module.EVENT_TRACK_NO_TARGET)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            for _ in range(config.UART_TRACK_LOST_CONFIRM_FRAMES - 1):
                current.submit([], LABELS)
                self.assertIsNone(current._next_outgoing_packet(1000))
            # Time alone must not count as additional missing frames.
            self.assertIsNone(current._next_outgoing_packet(10000))
            current.submit([], LABELS)
            self.assertEqual(current._next_outgoing_packet(10001), e4)
            self.assertIsNone(current._next_outgoing_packet(11000))
            current.submit([], LABELS)
            self.assertIsNone(current._next_outgoing_packet(12000))
            current.submit([target], LABELS)
            self.assertFalse(current._track_no_target_sent)
            self.assertIsNotNone(current._latest_packet)
            for _ in range(config.UART_TRACK_LOST_CONFIRM_FRAMES):
                current.submit([], LABELS)
            self.assertIsNone(current._track_target_lock)
            self.assertEqual(current._next_outgoing_packet(13000), e4)
            self.assertIsNone(current._next_outgoing_packet(14000))

    def test_04_miss_logs_distinguish_raw_verification_and_zone_exclusion(self):
        current = controller(CMD_TRACK)
        current.target_label = "sqarered"
        target = block(100, 150, 4)
        zone = block(100, 150, 2)
        with patch.object(config, "VISION_DIAGNOSTICS_ENABLED", True), \
                patch.object(serial_module, "_now_ms", return_value=1000), \
                patch("builtins.print") as log:
            current.submit([], LABELS, raw_objects=[target])
            self.assertTrue(any("raw=1 verified=0 outside_zone=0" in str(call)
                                for call in log.call_args_list))
            current.submit([target], LABELS)
            current.submit([target], LABELS, raw_objects=[target, zone])
            self.assertTrue(any("raw=1 verified=1 outside_zone=0" in str(call)
                                for call in log.call_args_list))
            self.assertIsNone(current._latest_packet)

    def test_locked_03_04_24_follow_raw_boxes_without_verification_or_rescreening(self):
        for mode in (serial_module.CMD_SEARCH, CMD_TRACK, MODE_FINAL_ROI_TRACK):
            with self.subTest(mode=mode):
                current = controller(mode)
                current.target_label = serial_module.TARGET_BLACK_OR_GREEN
                initial = block(493, 81, 6)
                moved = block(494, 82, 6)
                neighbor = block(545, 76, 3)
                with patch.object(serial_module, "_now_ms", return_value=1000):
                    self.assertIs(current.submit([initial], LABELS, raw_objects=[initial]), initial)
                    with patch.object(current, "_select_final_roi_target",
                                      side_effect=AssertionError("locked raw target must not rescreen")), \
                            patch.object(current, "_reject_large_target_jump",
                                         side_effect=AssertionError("locked raw target must not reject jumps")):
                        self.assertIs(current.submit([neighbor], LABELS,
                                                     raw_objects=[moved, neighbor]), moved)
                    self.assertIsNotNone(current._latest_packet)
                    self.assertEqual(current._latest_packet[2], 6)

    def test_search_lock_keeps_nearest_same_class_not_new_preferred_target(self):
        current = controller(serial_module.CMD_SEARCH)
        current.target_label = serial_module.TARGET_BLACK_OR_GREEN
        initial, moved = block(450, 100, 6), block(440, 110, 6)
        far_same = block(10, 300, 6)
        new_green = block(288, 350)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            current.submit([initial], LABELS)
            self.assertIs(current.submit([far_same, new_green], LABELS,
                                         raw_objects=[far_same, moved, new_green]), moved)
            self.assertEqual(current._search_target_lock[:3], (6, 440.0, 110.0))
            self.assertEqual(current._latest_packet[2], 6)

    def test_search_black_lock_missing5_and800ms_no_stale_then_reacquires(self):
        current = controller(serial_module.CMD_SEARCH)
        current.target_label = serial_module.TARGET_BLACK_OR_GREEN
        initial, replacement = block(450, 100, 6), block(100, 300)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            current.submit([initial], LABELS)
            lock = current._search_target_lock
            for index in range(1, config.UART_TRACK_LOST_CONFIRM_FRAMES + 1):
                self.assertIsNone(current.submit([], LABELS, raw_objects=[]))
                self.assertIsNone(current._latest_packet)
                if index < config.UART_TRACK_LOST_CONFIRM_FRAMES:
                    self.assertEqual(current._search_target_lock, lock)
            self.assertEqual(current._search_target_lock, lock)
            self.assertIsNone(current._next_outgoing_packet(1000))
            with patch.object(serial_module, "_now_ms", return_value=1800):
                self.assertIsNone(current.submit([], LABELS, raw_objects=[]))
                self.assertIsNone(current._search_target_lock)
                self.assertEqual(current._next_outgoing_packet(1800),
                                 build_event_packet(serial_module.EVENT_SEARCH_NO_TARGET))
                self.assertIs(current.submit([replacement], LABELS), replacement)
            self.assertEqual(current._search_target_lock[0], 5)
            self.assertEqual(current._search_no_target_frames, 0)

    def test_search_lock_preserved_by_13_reset_by_new_target_or_search(self):
        current = controller(serial_module.CMD_SEARCH)
        initial = block(450, 100)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            current.submit([initial], LABELS)
            lock = current._search_target_lock
            current._handle_command(serial_module.CMD_SEARCH_SWEEP_DONE)
            self.assertEqual(current._search_target_lock, lock)
            current._handle_command(serial_module.CMD_SEARCH)
            self.assertIsNone(current._search_target_lock)
            current.submit([initial], LABELS)
            current._handle_command(serial_module.CMD_SELECT_RED_OBJECT)
            self.assertIsNone(current._search_target_lock)

    def test_search_e3_wait13_retries_at8s_and_restarts_each_timeout(self):
        self.assertEqual(config.UART_SEARCH_NO_TARGET_ACK_TIMEOUT_MS, 8000)
        current = controller(serial_module.CMD_SEARCH)
        event = build_event_packet(serial_module.EVENT_SEARCH_NO_TARGET)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            for _ in range(config.UART_SEARCH_NO_TARGET_FRAMES - 1):
                current.submit([], LABELS)
            self.assertIsNone(current._next_outgoing_packet(1000))
            current.submit([], LABELS)
        self.assertEqual(current._next_outgoing_packet(1000), event)
        self.assertFalse(current._search_no_target_acknowledged)
        for now in (1001, 1500, 8999):
            self.assertIsNone(current._next_outgoing_packet(now))
        self.assertEqual(current._next_outgoing_packet(9000), event)
        self.assertFalse(current._search_no_target_acknowledged)
        self.assertIsNone(current._next_outgoing_packet(16999))
        self.assertEqual(current._next_outgoing_packet(17000), event)

    def test_search_e3_timeout_starts_at_actual_successful_write(self):
        current = controller(serial_module.CMD_SEARCH)
        event = build_event_packet(serial_module.EVENT_SEARCH_NO_TARGET)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            for _ in range(config.UART_SEARCH_NO_TARGET_FRAMES):
                current.submit([], LABELS)
        self.assertEqual(current._next_outgoing_packet(1000), event)
        current._serial = SimpleNamespace(write=Mock(return_value=len(event)))
        with patch.object(serial_module, "_now_ms", return_value=1200):
            current._write_packet(event, current._outgoing_generation)
        self.assertEqual(current._last_search_no_target_ms, 1200)
        self.assertIsNone(current._next_outgoing_packet(9000))
        self.assertIsNone(current._next_outgoing_packet(9199))
        self.assertEqual(current._next_outgoing_packet(9200), event)

    def test_search_e3_rx13_preserves500ms_retry_and_requires_a_new_frame(self):
        current = controller(serial_module.CMD_SEARCH)
        event = build_event_packet(serial_module.EVENT_SEARCH_NO_TARGET)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            for _ in range(config.UART_SEARCH_NO_TARGET_FRAMES):
                current.submit([], LABELS)
        self.assertEqual(current._next_outgoing_packet(1000), event)
        with patch.object(serial_module, "_now_ms", return_value=4000):
            current.process_received_bytes(serial_module.build_command_packet(
                serial_module.CMD_SEARCH_SWEEP_DONE))
        self.assertTrue(current._search_no_target_acknowledged)
        self.assertIsNone(current._next_outgoing_packet(4500))
        with patch.object(serial_module, "_now_ms", return_value=4510):
            current.submit([], LABELS)
        self.assertEqual(current._next_outgoing_packet(4510), event)
        self.assertIsNone(current._next_outgoing_packet(5009))
        self.assertEqual(current._next_outgoing_packet(5010), event)

    def test_search_e3_recovery_before13_cancels8s_retry_and_rearms_next_loss(self):
        current = controller(serial_module.CMD_SEARCH)
        event = build_event_packet(serial_module.EVENT_SEARCH_NO_TARGET)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            for _ in range(config.UART_SEARCH_NO_TARGET_FRAMES):
                current.submit([], LABELS)
        self.assertEqual(current._next_outgoing_packet(1000), event)
        target = block(100, 270)
        with patch.object(serial_module, "_now_ms", return_value=8900):
            self.assertIs(current.submit([target], LABELS), target)
        self.assertFalse(current._search_no_target_initial_sent)
        self.assertEqual(current._search_no_target_frames, 0)
        self.assertEqual(current._next_outgoing_packet(9000), current._latest_packet)
        self.assertIsNone(current._next_outgoing_packet(10000))
        with patch.object(serial_module, "_now_ms", return_value=11000):
            for _ in range(config.UART_SEARCH_NO_TARGET_FRAMES):
                current.submit([], LABELS)
        self.assertEqual(current._next_outgoing_packet(11000), event)
        self.assertIsNone(current._next_outgoing_packet(18999))
        self.assertEqual(current._next_outgoing_packet(19000), event)

    def test_search_e3_new_task_or_stage_cancels_old8s_timeout(self):
        for command in (serial_module.CMD_SEARCH, CMD_TRACK,
                        serial_module.CMD_SELECT_RED_OBJECT,
                        serial_module.CMD_SELECT_BLACK_OBJECT):
            with self.subTest(command=command):
                current = controller(serial_module.CMD_SEARCH)
                with patch.object(serial_module, "_now_ms", return_value=1000):
                    for _ in range(config.UART_SEARCH_NO_TARGET_FRAMES):
                        current.submit([], LABELS)
                self.assertEqual(current._next_outgoing_packet(1000),
                                 build_event_packet(serial_module.EVENT_SEARCH_NO_TARGET))
                with patch.object(serial_module, "_now_ms", return_value=1100):
                    current.process_received_bytes(serial_module.build_command_packet(command))
                self.assertFalse(current._search_no_target_initial_sent)
                self.assertIsNone(current._next_outgoing_packet(9000))

    def test_search_e3_8s_timeout_applies_to_red_and_mixed03(self):
        for label in ("sqarered", serial_module.TARGET_BLACK_OR_GREEN):
            with self.subTest(target=label):
                current = controller(serial_module.CMD_SEARCH)
                current.target_label = label
                current._post07_started = True
                with patch.object(serial_module, "_now_ms", return_value=1000):
                    for _ in range(config.UART_SEARCH_NO_TARGET_FRAMES):
                        current.submit([], LABELS)
                event = build_event_packet(serial_module.EVENT_SEARCH_NO_TARGET)
                self.assertEqual(current._next_outgoing_packet(1000), event)
                self.assertIsNone(current._next_outgoing_packet(8999))
                self.assertEqual(current._next_outgoing_packet(9000), event)

    def test_search_e3_prepared_timeout_retry_cannot_send_after_fresh_recovery(self):
        current = controller(serial_module.CMD_SEARCH)
        event = build_event_packet(serial_module.EVENT_SEARCH_NO_TARGET)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            for _ in range(config.UART_SEARCH_NO_TARGET_FRAMES):
                current.submit([], LABELS)
        self.assertEqual(current._next_outgoing_packet(1000), event)
        self.assertEqual(current._next_outgoing_packet(9000), event)
        generation = current._outgoing_generation
        current._serial = SimpleNamespace(write=Mock(return_value=len(event)))
        with patch.object(serial_module, "_now_ms", return_value=9001):
            current.submit([block(100, 270)], LABELS)
            current._write_packet(event, generation)
        current._serial.write.assert_not_called()
        self.assertFalse(current._search_no_target_initial_sent)
        self.assertEqual(current._next_outgoing_packet(9001), current._latest_packet)

    def test_safety_zone_side_margin_boundaries_all_colors_and_movable_ids(self):
        self.assertEqual(config.UART_SAFE_ZONE_TARGET_MARGIN_X, 100)
        samples = ((99, 150, False), (100, 150, True), (150, 150, True),
                   (350, 150, True), (400, 150, True), (401, 150, False),
                   (150, 99, False), (150, 100, True),
                   (350, 200, True), (350, 201, False))
        for zone_id in (1, 2):
            zone = SimpleNamespace(x=200, y=100, w=100, h=100,
                                   class_id=zone_id, score=.9)
            for target_id in (3, 4, 5, 6):
                for x, y, excluded in samples:
                    with self.subTest(zone=zone_id, target=target_id, x=x, y=y):
                        target = block(x, y, target_id)
                        result = VisionSerialController._exclude_objects_in_safety_zones(
                            [target, zone], LABELS, [zone],
                        )
                        self.assertEqual(result, [zone] if excluded else [target, zone])
        blue = SimpleNamespace(x=0, y=0, w=50, h=50, class_id=1)
        red = SimpleNamespace(x=450, y=300, w=100, h=100, class_id=2)
        target = block(400, 350)
        self.assertEqual(VisionSerialController._exclude_objects_in_safety_zones(
            [target], LABELS, [blue, red]), [])
        self.assertEqual(VisionSerialController._exclude_objects_in_safety_zones(
            [target], LABELS, []), [target])

    def test_safety_zone_side_margin_excludes_verified_and_raw_initial_targets(self):
        for mode in (serial_module.CMD_SEARCH, CMD_TRACK, MODE_FINAL_ROI_TRACK):
            for zone_id in (1, 2):
                for center_x in (150, 350):
                    with self.subTest(mode=mode, zone=zone_id, x=center_x):
                        current = controller(mode)
                        target = block(center_x, 150)
                        zone = SimpleNamespace(x=200, y=100, w=100, h=100,
                                               class_id=zone_id, score=.9)
                        with patch.object(serial_module, "_now_ms", return_value=1000):
                            self.assertIsNone(current.submit([target], LABELS,
                                                             raw_objects=[zone, target]))
                            self.assertIsNone(current.submit([], LABELS,
                                                             raw_objects=[zone, target]))
                        self.assertIsNone(current._latest_packet)
                        self.assertEqual(current._pending_events, [])
                        outside = block(99 if center_x == 150 else 401, 150)
                        with patch.object(serial_module, "_now_ms", return_value=1100):
                            self.assertIs(current.submit([outside], LABELS,
                                                         raw_objects=[zone, outside]), outside)

    def test_safety_zone_side_margin_also_stops_raw_locked_coordinates(self):
        modes = (serial_module.CMD_SEARCH, CMD_TRACK, MODE_WAIT_14_HANDSHAKE,
                 serial_module.MODE_WAIT_04_HANDSHAKE, MODE_FINAL_ROI_TRACK)
        for mode in modes:
            for center_x in (150, 350):
                with self.subTest(mode=mode, x=center_x):
                    current = controller(mode)
                    target = block(center_x, 150)
                    zone = SimpleNamespace(x=200, y=100, w=100, h=100,
                                           class_id=1, score=.9)
                    with patch.object(serial_module, "_now_ms", return_value=1000):
                        self.assertIs(current.submit([target], LABELS), target)
                    self.assertIsNotNone(current._latest_packet)
                    with patch.object(serial_module, "_now_ms", return_value=1100):
                        self.assertIsNone(current.submit([], LABELS, raw_objects=[zone, target]))
                    self.assertIsNone(current._latest_packet)
                    self.assertIsNone(current._next_outgoing_packet(1100))

    def test_safety_zone_side_margin_cancels_target_in_roi34(self):
        current = controller(MODE_FINAL_ROI_TRACK)
        target = block(150, 400)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            current.submit([target], LABELS)
        self.assertTrue(current._final_roi_sent)
        zone = SimpleNamespace(x=200, y=360, w=100, h=80, class_id=2, score=.9)
        with patch.object(serial_module, "_now_ms", return_value=1100):
            self.assertIsNone(current.submit([], LABELS, raw_objects=[zone, target]))
        self.assertFalse(current._final_roi_sent)
        self.assertIsNone(current._latest_packet)
        self.assertIsNone(current._next_outgoing_packet(1500))

    def test_safety_zone_side_margin_is_used_by_post22_e2_recovery(self):
        for center_x in (150, 350):
            current = controller(MODE_POST_ARRANGE_CHECK)
            current._handle_command(serial_module.CMD_SELECT_RED_OBJECT)
            current.mode = MODE_POST_ARRANGE_CHECK
            with patch.object(serial_module, "_now_ms", return_value=1000):
                current.submit([], LABELS, raw_objects=[])
            self.assertEqual(current._next_outgoing_packet(1000),
                             build_event_packet(EVENT_ARRANGEMENT_NO_OBJECT))
            target = block(center_x, 150, 4)
            zone = SimpleNamespace(x=200, y=100, w=100, h=100, class_id=2, score=.9)
            with patch.object(serial_module, "_now_ms", return_value=1100):
                current.submit([target], LABELS, raw_objects=[zone, target])
            self.assertEqual(current.mode, MODE_POST_ARRANGE_CHECK)
            self.assertIsNone(current._latest_packet)
            self.assertIsNone(current._next_outgoing_packet(1100))
            with patch.object(serial_module, "_now_ms", return_value=1200):
                current.submit([], LABELS, raw_objects=[target])
            self.assertEqual(current._next_outgoing_packet(1200),
                             build_event_packet(serial_module.EVENT_SEARCH_CENTER_REACHED))

    def test_safety_zone_side_margin_does_not_hide_near05_cargo(self):
        current = controller(MODE_FINAL_ROI_TRACK)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            current._handle_command(serial_module.CMD_SEARCH_BLUE_ZONE)
        self.assertEqual(write_near_view_request(current, 1000),
                         build_event_packet(serial_module.EVENT_TARGET_READY))
        target = block(150, 150)
        zone = SimpleNamespace(x=200, y=100, w=100, h=100, class_id=1, score=.9)
        for now in (1500, 1510, 1520):
            with patch.object(serial_module, "_now_ms", return_value=now):
                current.submit([target], LABELS, raw_objects=[zone, target])
        self.assertEqual(current._zone_load_recheck_source, "before06")
        self.assertEqual(current._pending_events,
                         [build_event_packet(serial_module.EVENT_TARGET_READY)])
        confirm_final_load_check(current, [5], 1530)
        self.assertEqual(current._next_outgoing_packet(2100),
                         serial_module.build_coordinate_packet(5, 0, 0))

    @patch.object(config, "UART_ZONE_OBSTACLE_ENABLED", True)
    def test_safety_zone_side_margin_does_not_relax_zone06_obstacle_detection(self):
        current = controller(serial_module.MODE_SEARCH_ZONE)
        zone = SimpleNamespace(x=340, y=100, w=100, h=100, class_id=1, score=.9)
        blocker = block(300, 200, 3)
        self.assertEqual(current._exclude_objects_in_safety_zones([blocker], LABELS, [zone]), [])
        self.assertEqual(current._exclude_objects_in_safety_zones(
            [blocker], LABELS, [zone], horizontal_padding=0), [blocker])
        with patch.object(serial_module, "_now_ms", return_value=1000):
            self.assertIs(current.submit([], LABELS, raw_objects=[zone]), zone)
        self.assertEqual(current._next_outgoing_packet(1000),
                         build_event_packet(serial_module.EVENT_ZONE_FOUND))
        for now in (1010, 1020, 1030):
            with patch.object(serial_module, "_now_ms", return_value=now):
                current.submit([blocker], LABELS, raw_objects=[zone, blocker])
        self.assertEqual(current._zone_obstacle_frames, 3)
        self.assertTrue(current._zone_obstacle_waiting)
        self.assertEqual(current._next_outgoing_packet(1030),
                         build_event_packet(serial_module.EVENT_ZONE_OBSTACLE))

    def test_safety_zone_side_margin_diagnostic_uses_same_expanded_bounds(self):
        current = controller(serial_module.CMD_SEARCH)
        target = block(150, 150)
        zone = SimpleNamespace(x=200, y=100, w=100, h=100, class_id=2, score=.9)
        with patch.object(config, "VISION_DIAGNOSTICS_ENABLED", True), \
                patch.object(serial_module, "_now_ms", return_value=1000), \
                patch("builtins.print") as log:
            current.submit([target], LABELS, raw_objects=[zone, target])
        self.assertTrue(any("[SAFE-REASON]" in str(call) and "id2" in str(call)
                            and "x_padding=100" in str(call) for call in log.call_args_list))

    def test_raw_yolo_24_fallback_and_other_acquisition_still_exclude_safety_zones(self):
        for mode in (serial_module.CMD_SEARCH, CMD_TRACK, MODE_FINAL_ROI_TRACK):
            with self.subTest(mode=mode):
                current = controller(mode)
                target, zone = block(100, 150), block(100, 150, 1)
                with patch.object(serial_module, "_now_ms", return_value=1000):
                    selected = current.submit([], LABELS, raw_objects=[target])
                    if mode == MODE_FINAL_ROI_TRACK:
                        self.assertIs(selected, target)
                    else:
                        self.assertIsNone(selected)
                    self.assertIs(current.submit([target], LABELS, raw_objects=[target]), target)
                    self.assertIsNone(current.submit([], LABELS, raw_objects=[target, zone]))
                    self.assertIsNone(current._latest_packet)

    def test_raw_locked_target_still_triggers_04_14_and_34_with_existing_references(self):
        for mode in (serial_module.CMD_SEARCH, CMD_TRACK, MODE_FINAL_ROI_TRACK):
            with self.subTest(mode=mode):
                current = controller(mode)
                initial = block(100, 150)
                destination = block(288, 224 if mode != MODE_FINAL_ROI_TRACK else 400)
                with patch.object(serial_module, "_now_ms", return_value=1000):
                    current.submit([initial], LABELS)
                    self.assertIs(current.submit([], LABELS, raw_objects=[destination]), destination)
                    expected = {serial_module.CMD_SEARCH: serial_module.EVENT_SEARCH_CENTER_REACHED,
                                CMD_TRACK: EVENT_TRACK_CENTER_REACHED,
                                MODE_FINAL_ROI_TRACK: serial_module.EVENT_FINAL_TARGET_IN_ROI}[mode]
                    self.assertIn(build_event_packet(expected), current._pending_events)

    def test_diagnostics_explain_raw_yolo_lock_follow(self):
        current = controller(serial_module.CMD_SEARCH)
        current.target_label = serial_module.TARGET_BLACK_OR_GREEN
        target, neighbor = block(493, 81, 6), block(545, 76, 3)
        with patch.object(config, "VISION_DIAGNOSTICS_ENABLED", True), \
                patch.object(serial_module, "_now_ms", return_value=1000), \
                patch("builtins.print") as log:
            current.submit([target], LABELS)
            current._diagnostics._last_signature = None
            current.submit([], LABELS, raw_objects=[target, neighbor])
            self.assertTrue(any("source=current_raw_yolo" in str(call) for call in log.call_args_list))
            self.assertTrue(any("policy=search_lock" in str(call) for call in log.call_args_list))
            self.assertTrue(any("applied=no" in str(call) for call in log.call_args_list))
            self.assertFalse(any("reason=VERIFICATION_REJECTED" in str(call) for call in log.call_args_list))

    def test_later_14_and_44_use_merged_selector_for_24_or_02(self):
        lone = block(200, 310)
        high = block(30, 300)
        low = block(450, 310)
        close_blue = block(205, 310, 3)  # Actual overlap hits BOTH strips.
        far_red = block(450, 80, 4)
        for command, source_mode in (
            (serial_module.CMD_TRACK_CENTER_DONE, MODE_WAIT_14_HANDSHAKE),
            (serial_module.CMD_TRACK_RECOVERY_DONE, CMD_TRACK),
        ):
            for objects, expected_target in (
                ([lone], lone),
                ([high, low], low),
                ([lone, far_red], lone),
                ([lone, close_blue], None),
                ([], None),
            ):
                with self.subTest(command=command, objects=len(objects),
                                  target=expected_target):
                    current = controller(source_mode)
                    current._first_green_near_view_decided = True
                    current._post07_started = True  # Scope: unchanged later-green decision.
                    # A moved target must not fail because the old side-only
                    # middle-X anchor cannot match it after MCU motion.
                    current._last_target_center_x = 10
                    with patch.object(serial_module, "_now_ms", return_value=1000):
                        current._handle_command(command)
                        self.assertEqual(current._pending_events, [])
                        selected = current.submit(objects, LABELS, raw_objects=objects)
                    # The decision frame emits only its state event; the
                    # next fresh state-24 frame starts coordinates/overlay.
                    self.assertIsNone(selected)
                    event = (EVENT_SKIP_TO_FINAL_TRACK if expected_target
                             is not None else EVENT_ROI_READY)
                    expected_mode = (MODE_FINAL_ROI_TRACK if expected_target
                                     is not None else MODE_ARRANGE_RIGHTMOST)
                    self.assertEqual(current.mode, expected_mode)
                    self.assertFalse(current._initial_arrangement_decision_pending)
                    self.assertIsNone(current._latest_packet)
                    self.assertEqual(current._next_outgoing_packet(1000),
                                     build_event_packet(event))
                    for now_ms in (1500, 6000, 10000):
                        self.assertIsNone(current._next_outgoing_packet(now_ms))
                    if expected_target is not None:
                        with patch.object(serial_module, "_now_ms", return_value=10100):
                            self.assertIs(
                                current.submit(objects, LABELS, raw_objects=objects),
                                expected_target,
                            )
                        self.assertIsNotNone(current._latest_packet)

    def test_first_green_14_requires_two_clear_frames_and_seed_lock(self):
        for command, mode in (
            (serial_module.CMD_TRACK_CENTER_DONE, MODE_WAIT_14_HANDSHAKE),
        ):
            with self.subTest(command=command):
                current = controller(mode)
                target = block(288, 310)
                with patch.object(serial_module, "_now_ms", return_value=1000):
                    current._handle_command(command)
                for index in range(config.UART_FIRST_GREEN_DECISION_CONFIRM_FRAMES):
                    now = 1050 + index * 50
                    with patch.object(serial_module, "_now_ms", return_value=now):
                        self.assertIsNone(current.submit([target], LABELS))
                    self.assertIsNone(current._latest_packet)
                    if index < config.UART_FIRST_GREEN_DECISION_CONFIRM_FRAMES - 1:
                        self.assertTrue(current._initial_arrangement_decision_pending)
                        self.assertIsNone(current._next_outgoing_packet(now))
                    else:
                        self.assertEqual(current.mode, MODE_FINAL_ROI_TRACK)
                        self.assertEqual(current._final_target_lock[:3], (5, 288.0, 310.0))
                        self.assertEqual(current._next_outgoing_packet(now),
                                         build_event_packet(EVENT_SKIP_TO_FINAL_TRACK))
                self.assertIsNone(current._next_outgoing_packet(3000))
                # A later raw frame follows the confirmed target, not a newly
                # ranked green target; verifier dropouts do not break that lock.
                nearer = block(40, 350)
                with patch.object(serial_module, "_now_ms", return_value=3050):
                    self.assertIs(current.submit([], LABELS, raw_objects=[nearer, target]), target)
                self.assertIsNotNone(current._latest_packet)

    def test_first_green_transient_neighbor_and_unverified_frame_do_not_emit02(self):
        current = controller(MODE_WAIT_14_HANDSHAKE)
        target, neighbor = block(288, 310), block(350, 310, 3)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            current._handle_command(serial_module.CMD_TRACK_CENTER_DONE)
            current.submit([target, neighbor], LABELS)
        with patch.object(serial_module, "_now_ms", return_value=1050):
            current.submit([], LABELS, raw_objects=[target])
        self.assertEqual(current._initial_confirmation_frames, 1)
        self.assertIsNone(current._next_outgoing_packet(1050))
        with patch.object(serial_module, "_now_ms", return_value=1100):
            current.submit([target], LABELS)
        self.assertEqual(current._next_outgoing_packet(1100),
                         build_event_packet(EVENT_SKIP_TO_FINAL_TRACK))

    def test_first_green_crowding_requires_two_consecutive_same_target_frames(self):
        current = controller(MODE_WAIT_14_HANDSHAKE)
        target, neighbor = block(288, 310), block(290, 310, 3)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            current._handle_command(serial_module.CMD_TRACK_CENTER_DONE)
        for index, objects in enumerate(([target, neighbor], [], [target, neighbor],
                                         [target, neighbor])):
            now = 1050 + index * 50
            with patch.object(serial_module, "_now_ms", return_value=now):
                current.submit(objects, LABELS)
            if index < 3:
                self.assertIsNone(current._next_outgoing_packet(now))
        self.assertEqual(current._next_outgoing_packet(1200), build_event_packet(EVENT_ROI_READY))
        self.assertTrue(current._arrangement_02_ack_pending)
        self.assertIsNone(current._next_outgoing_packet(2000))

    def test_first_green_confirmation_does_not_combine_different_green_targets(self):
        current = controller(MODE_WAIT_14_HANDSHAKE)
        left, right = block(70, 310), block(450, 310)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            current._handle_command(serial_module.CMD_TRACK_CENTER_DONE)
        for index, target in enumerate((left, right, left, right)):
            now = 1050 + index * 50
            with patch.object(serial_module, "_now_ms", return_value=now):
                current.submit([target], LABELS)
            self.assertEqual(current._initial_confirmation_frames, 1)
            self.assertIsNone(current._next_outgoing_packet(now))
        for now in (1250, 1300):
            with patch.object(serial_module, "_now_ms", return_value=now):
                current.submit([right], LABELS)
        self.assertEqual(current._next_outgoing_packet(1300), build_event_packet(EVENT_SKIP_TO_FINAL_TRACK))
        self.assertEqual(current._final_target_lock[:3], (5, 450.0, 310.0))

    def test_first_green_deadline_missing_unverified_or_stale_sends_one_e4(self):
        target = block(288, 310)
        for observations in ([], [(1200, [], [target])], [(1100, [target], [target])]):
            with self.subTest(observations=observations):
                current = controller(MODE_WAIT_14_HANDSHAKE)
                with patch.object(serial_module, "_now_ms", return_value=1000):
                    current._handle_command(serial_module.CMD_TRACK_CENTER_DONE)
                for now, verified, raw in observations:
                    with patch.object(serial_module, "_now_ms", return_value=now):
                        current.submit(verified, LABELS, raw_objects=raw)
                self.assertIsNone(current._next_outgoing_packet(1499))
                missing_deadline = (observations[-1][0] if observations else 1000) + 500
                self.assertIsNone(current._next_outgoing_packet(missing_deadline - 1))
                self.assertEqual(current._next_outgoing_packet(missing_deadline),
                                 build_event_packet(serial_module.EVENT_TRACK_NO_TARGET))
                self.assertEqual(current.mode, CMD_TRACK)
                self.assertTrue(current._track_green_recovery_waiting)
                self.assertIsNone(current._latest_packet)
                self.assertIsNone(current._next_outgoing_packet(5000))
                with patch.object(serial_module, "_now_ms", return_value=5050):
                    current.submit([target], LABELS)
                self.assertIsNotNone(current._latest_packet)
                self.assertNotIn(build_event_packet(EVENT_TRACK_CENTER_REACHED), current._pending_events)
                with patch.object(serial_module, "_now_ms", return_value=5100):
                    current._handle_command(serial_module.CMD_TRACK_RECOVERY_DONE)
                for now in (5150, 5200, 5250):
                    with patch.object(serial_module, "_now_ms", return_value=now):
                        current.submit([target], LABELS)
                self.assertEqual(current._next_outgoing_packet(5250),
                                 serial_module.build_coordinate_packet(5, 0, 86))
                self.assertEqual(current.mode, serial_module.CMD_SEARCH)

    def test_first_green_timeout_crowding_needs_multiple_fresh_observations(self):
        target, neighbor = block(288, 310), block(290, 310, 3)
        # Missing frames break the streak. At the deadline, only repeatedly
        # observed crowding, with no clear result, can justify arrangement.
        for count, event in ((1, serial_module.EVENT_TRACK_NO_TARGET), (3, EVENT_ROI_READY)):
            with self.subTest(count=count):
                current = controller(MODE_WAIT_14_HANDSHAKE)
                with patch.object(serial_module, "_now_ms", return_value=1000):
                    current._handle_command(serial_module.CMD_TRACK_CENTER_DONE)
                for index in range(count):
                    now = 1250 + index * 80
                    with patch.object(serial_module, "_now_ms", return_value=now):
                        current.submit([target, neighbor], LABELS)
                    if index < count - 1:
                        with patch.object(serial_module, "_now_ms", return_value=now + 40):
                            current.submit([], LABELS)
                if count == 1:
                    self.assertIsNone(current._next_outgoing_packet(1500))
                    self.assertEqual(current._next_outgoing_packet(1750), build_event_packet(event))
                else:
                    self.assertEqual(current._next_outgoing_packet(1500), build_event_packet(event))

    def test_first_green_new_search_cancels_confirmation_and_duplicate14_keeps_deadline(self):
        current = controller(MODE_WAIT_14_HANDSHAKE)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            current._handle_command(serial_module.CMD_TRACK_CENTER_DONE)
        with patch.object(serial_module, "_now_ms", return_value=1300):
            current._handle_command(serial_module.CMD_TRACK_CENTER_DONE)
        self.assertEqual(current._initial_confirmation_started_ms, 1000)
        current._handle_command(serial_module.CMD_SEARCH)
        self.assertIsNone(current._initial_confirmation_started_ms)
        self.assertFalse(current._initial_arrangement_decision_pending)
        self.assertIsNone(current._next_outgoing_packet(1500))

    def test_first_green_confirmation_restarts_after_frame_gap_and_cannot_pass_deadline(self):
        target = block(288, 310)
        current = controller(MODE_WAIT_14_HANDSHAKE)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            current._handle_command(serial_module.CMD_TRACK_CENTER_DONE)
        for now, expected_frames in ((1010, 1), (1250, 1)):
            with patch.object(serial_module, "_now_ms", return_value=now):
                current.submit([target], LABELS)
            self.assertEqual(current._initial_confirmation_frames, expected_frames)
            self.assertIsNone(current._next_outgoing_packet(now))
        # A fresh detection beyond the old deadline must not generate E4.
        with patch.object(serial_module, "_now_ms", return_value=1800):
            current.submit([target], LABELS)
        self.assertIsNone(current._next_outgoing_packet(1800))
        self.assertIsNone(current._final_target_lock)
        self.assertIsNone(current._latest_packet)
        for now in (1850, 1900):
            with patch.object(serial_module, "_now_ms", return_value=now):
                current.submit([], LABELS, raw_objects=[target])
        self.assertEqual(current._next_outgoing_packet(1900),
                         build_event_packet(EVENT_SKIP_TO_FINAL_TRACK))

    def test_first_green14_decision_retains_raw_identity_and_counts_raw_neighbors(self):
        for command in (serial_module.CMD_TRACK_CENTER_DONE,):
            for crowded in (False, True):
                with self.subTest(command=command, crowded=crowded):
                    current = controller(MODE_WAIT_14_HANDSHAKE if command == serial_module.CMD_TRACK_CENTER_DONE else CMD_TRACK)
                    target, other = block(90, 310), block(480, 380)
                    current._track_target_lock = current._final_target_identity(target)
                    raw = [other, target]
                    if crowded:
                        raw.append(block(95, 310, 3))  # Both axes, not horizontal-only32.
                    with patch.object(serial_module, "_now_ms", return_value=1000):
                        current._handle_command(command)
                    for now in (1050, 1100, 1150):
                        with patch.object(serial_module, "_now_ms", return_value=now):
                            current.submit([], LABELS, raw_objects=raw)
                    self.assertEqual(current._next_outgoing_packet(1150), build_event_packet(
                        EVENT_ROI_READY if crowded else EVENT_SKIP_TO_FINAL_TRACK))
                    if not crowded:
                        self.assertEqual(current._final_target_lock[:3], (5, 90.0, 310.0))

    def test_debug_safety_zone_area_text_is_black_and_clipped_not_confidence(self):
        img = SimpleNamespace(draw_string=Mock())
        blue = SimpleNamespace(x=-100, y=0, w=388, h=448, class_id=1, score=.99)
        red = SimpleNamespace(x=0, y=0, w=576, h=224, class_id=2, score=.1)
        deprecated = SimpleNamespace(x=0, y=0, w=576, h=448, class_id=0)
        draw_debug_safety_zone_area(img, [blue, red, deprecated, block(288, 310)], LABELS, 576, 448)
        self.assertEqual(img.draw_string.call_count, 2)
        for call in img.draw_string.call_args_list:
            self.assertIn("AREA: 50.0%", call.args[2])
            self.assertEqual(call.kwargs["color"], "black")
            self.assertEqual(call.kwargs["thickness"], 1)
            self.assertEqual(call.kwargs["scale"], 1.5)
        img.draw_string.reset_mock()
        draw_debug_safety_zone_area(img, [], LABELS, 576, 448)
        img.draw_string.assert_not_called()

        roi = make_bottom_center_roi(576, 448)
        current = controller(serial_module.MODE_SEARCH_ZONE)
        current.tracking_roi = roi
        reference_x, reference_y = current._reference_point(serial_module.MODE_SEARCH_ZONE)
        for zone in (blue, red, block(288, 383.5, 1), block(288, 440, 2)):
            img.draw_string.reset_mock()
            before = vars(zone).copy()
            draw_debug_safety_zone_area(img, [zone], LABELS, 576, 448, roi_rect=roi)
            error_y = current._zone_supply_point_error(zone, reference_x, reference_y)[1]
            call = img.draw_string.call_args
            self.assertTrue(call.args[2].endswith(" dY:{:+d}px".format(error_y)))
            self.assertEqual(call.args[:2], (8, 84))
            self.assertEqual(call.kwargs["color"], "black")
            self.assertEqual(vars(zone), before)

    def test_confirmed_red_thresholds(self):
        self.assertEqual(config.UART_FINAL_RED_MIN_DX_Y_GT_290, 240)
        self.assertEqual(config.UART_FINAL_RED_MIN_DX_Y_GT_100, 120)
        self.assertEqual(config.UART_FINAL_RED_MIN_DX_Y_GT_90, 70)
        self.assertEqual(config.UART_FINAL_RED_MIN_DX_Y_LE_90, 50)
        red = block(100, 310, 4)
        self.assertFalse(
            VisionSerialController._matches_final_primary_spacing(
                red, [red, block(340, 310)]
            )
        )
        self.assertTrue(
            VisionSerialController._matches_final_primary_spacing(
                red, [red, block(341, 310)]
            )
        )

    def test_red_spacing_y_le_90_uses_50px_inclusive_neighbor_range(self):
        for center_y in (80, 90):
            self.assertEqual(
                VisionSerialController._final_spacing_limits(center_y, True),
                (50, None),
            )
            for target_id, neighbor_id in ((4, 5), (5, 4), (4, 4)):
                target = block(100, center_y, target_id)
                for delta_x in (49, 50, 51, 110):
                    with self.subTest(y=center_y, target_id=target_id,
                                      neighbor_id=neighbor_id, dx=delta_x):
                        neighbor = block(100 + delta_x, center_y, neighbor_id)
                        self.assertEqual(
                            VisionSerialController._matches_final_primary_spacing(
                                target, [target, neighbor],
                            ),
                            delta_x > 50,
                        )
        self.assertEqual(
            VisionSerialController._final_spacing_limits(90.1, True),
            (70, None),
        )
        self.assertEqual(
            VisionSerialController._final_spacing_limits(90, False),
            (60, None),
        )

    def test_spacing_checks_every_neighbor_and_accepts_singleton(self):
        current = controller(MODE_POST_ARRANGE_CHECK)
        target = block(100, 200)
        blue = block(225, 200, 3)  # clears normal dx > 120
        red = block(240, 200, 4)  # fails red-pair dx > 150
        self.assertIsNone(
            current._select_final_roi_target(
                [target, blue, red], LABELS, "sqareredgreen", 1000
            )
        )
        self.assertIs(
            current._select_final_roi_target(
                [target], LABELS, "sqareredgreen", 1000
            ),
            target,
        )

    def test_qualified_target_with_y_nearest_roi_center_wins(self):
        current = controller(MODE_POST_ARRANGE_CHECK)
        near = block(30, 400)
        lower = block(450, 430)
        self.assertIs(
            current._select_final_roi_target(
                [near, lower], LABELS, "sqareredgreen", 1000
            ),
            near,
        )

    def test_roi_y_ranking_is_absolute_and_uses_configured_roi(self):
        current = controller(MODE_POST_ARRANGE_CHECK)
        above = block(30, 370)
        below = block(450, 420)
        self.assertIs(current._select_final_roi_target(
            [above, below], LABELS, "sqareredgreen", 1000
        ), below)
        current.tracking_roi = [0, 200, 576, 100]
        near = block(30, 255)
        farther = block(450, 300)
        self.assertIs(current._select_final_roi_target(
            [near, farther], LABELS, "sqareredgreen", 1000
        ), near)
        current.tracking_roi = None
        self.assertIs(current._select_final_roi_target(
            [near, farther], LABELS, "sqareredgreen", 1000
        ), near)

    def test_debug_ranks_clear_targets_by_roi_y_distance(self):
        preview = DebugTargetPreview(tracking_roi=[0, 359, 576, 89])
        near = block(30, 400)
        lower = block(450, 430)
        for frame in range(1, 4):
            _, _, selected, _ = preview.select(
                [near, lower], LABELS, [near, lower], "sqareredgreen"
            )
            self.assertIs(selected, near if frame == 3 else None)

    def test_04_and_24_roi_y_ranking_only_before_lock_or_after_loss(self):
        for mode in (CMD_TRACK, MODE_FINAL_ROI_TRACK):
            with self.subTest(mode=mode):
                current = controller(mode)
                # First-green 04 now uses image-center distance, not this
                # merged selector. Later-green 04 and all 24 still use it.
                current._first_green_near_view_decided = True
                target = block(30, 300)
                farther = block(450, 200)
                with patch.object(serial_module, "_now_ms", return_value=1000):
                    self.assertIs(current.submit([farther, target], LABELS), target)
                    now_closer = block(450, 310)
                    self.assertIs(current.submit([now_closer, target], LABELS), target)
                    current.submit([], LABELS)
                    self.assertIs(current.submit([farther, target], LABELS), target)

    def test_first_green_03_uses_image_center_2d_not_roi_or_xy_area_filters(self):
        current = controller(serial_module.CMD_SEARCH)
        near = block(360, 250)
        x_aligned = block(288, 400)
        neighbor = block(380, 260, 3)
        current._last_target_center_x = 10
        current._last_target_center_y = 10
        current._last_target_area = 1
        current._last_target_accept_ms = 1000
        with patch.object(serial_module, "_now_ms", return_value=1000), \
                patch.object(current, "_select_final_roi_target", side_effect=AssertionError("no merged first-green03")), \
                patch.object(current, "_reject_large_target_jump", side_effect=AssertionError("no jump first-green03")):
            self.assertIs(current.submit([x_aligned, neighbor, near], LABELS), near)
        self.assertEqual(current._latest_packet, serial_module.build_coordinate_packet(5, 72, 26))
        self.assertEqual(current._selection_diagnostic_policy, "first_green_frame_center")

    def test_first_green_04_acquires_and_reacquires_by_frame_center_then_keeps_lock(self):
        current = controller(CMD_TRACK)
        near, lower, neighbor = block(360, 250), block(288, 400), block(380, 260, 3)
        with patch.object(serial_module, "_now_ms", return_value=1000), \
                patch.object(current, "_select_final_roi_target", side_effect=AssertionError("no merged first-green04")):
            self.assertIs(current.submit([lower, neighbor, near], LABELS), near)
            self.assertEqual(current._latest_packet, serial_module.build_coordinate_packet(5, 72, 26))
            # A newly appearing center-nearer green must not replace a lock.
            self.assertIs(current.submit([block(300, 225), near, neighbor], LABELS), near)
            for _ in range(config.UART_TRACK_LOST_CONFIRM_FRAMES):
                current.submit([], LABELS)
            self.assertIsNone(current._track_target_lock)
            replacement = block(350, 260)
            self.assertIs(current.submit([replacement, block(360, 270, 3)], LABELS), replacement)
            self.assertEqual(current._latest_packet, serial_module.build_coordinate_packet(5, 62, 36))

    def test_first_green24_and_later04_ignore_spacing_neighbors(self):
        target, neighbor = block(360, 250), block(380, 260, 3)
        for mode, decided in ((MODE_FINAL_ROI_TRACK, False), (CMD_TRACK, True)):
            with self.subTest(mode=mode, decided=decided):
                current = controller(mode)
                current._first_green_near_view_decided = decided
                selected = current.submit([target, neighbor], LABELS)
                self.assertIs(selected, target)
                self.assertEqual(current._selection_diagnostic_policy,
                                 "supply_no_spacing24" if mode == MODE_FINAL_ROI_TRACK else "supply_no_spacing")

    def test_post07_supply_14_uses_overlap_then24_tracks_raw_group_actual_id(self):
        for command in (serial_module.CMD_SELECT_GREEN_OBJECT,
                        serial_module.CMD_SELECT_BLACK_OBJECT,
                        serial_module.CMD_SELECT_BLACK_OR_GREEN):
            with self.subTest(command=command):
                current = controller(MODE_WAIT_14_HANDSHAKE)
                current._handle_command(serial_module.CMD_SELECT_RED_OBJECT)
                current._handle_command(command)
                current.mode = MODE_WAIT_14_HANDSHAKE
                current._handle_command(serial_module.CMD_TRACK_CENTER_DONE)
                # Distant Y boxes lie outside even the expanded XY range,
                # despite failing the old center-dx table's all-Y comparison.
                black, green, blue = block(200, 150, 6), block(205, 320), block(210, 100, 3)
                with patch.object(serial_module, "_now_ms", return_value=1000), \
                        patch.object(current, "_select_final_roi_target",
                                     side_effect=AssertionError("post14 supplies must use overlap, not dx table")):
                    current.submit([], LABELS, raw_objects=[black, green, blue])
                self.assertEqual(current._next_outgoing_packet(1000), build_event_packet(EVENT_SKIP_TO_FINAL_TRACK))
                self.assertEqual(current.mode, MODE_FINAL_ROI_TRACK)
                self.assertEqual(current._final_target_lock[:3], (5, 205.0, 320.0))
                blue = block(210, 330, 3)  # New crowding must not re-screen24.
                # Green disappears, but nearby black remains: follow it as the
                # same supply group, transmit its actual model ID, no EE.
                black = block(208, 323, 6)
                with patch.object(serial_module, "_now_ms", return_value=1100):
                    self.assertIs(current.submit([], LABELS, raw_objects=[black, blue]), black)
                self.assertEqual(current._next_outgoing_packet(1100),
                                 serial_module.build_coordinate_packet(6, -80, -60))
                with patch.object(serial_module, "_now_ms", return_value=1200):
                    current.submit([], LABELS, raw_objects=[])
                self.assertIsNone(current._latest_packet)
                # Black now retains identity during its 800ms confirmation,
                # while still suppressing coordinates on every raw miss.
                self.assertEqual(current._final_target_lock[:3], (6, 208.0, 323.0))
                # Reacquisition also skips spacing and uses current YOLO only.
                with patch.object(serial_module, "_now_ms", return_value=1300):
                    current.submit([], LABELS, raw_objects=[green, blue])
                self.assertEqual(current._next_outgoing_packet(1300),
                                 serial_module.build_coordinate_packet(5, -83, -64))
                current._handle_command(serial_module.CMD_SELECT_RED_OBJECT)
                self.assertFalse(current._post07_supply_tracking)

    def test_spacing_relaxation_preserves_overlap_decisions_and_mixed_front_guard(self):
        target, neighbor = block(200, 300), block(210, 310, 3)
        for after07, target_command in ((False, serial_module.CMD_SELECT_GREEN_OBJECT),
                                        (True, serial_module.CMD_SELECT_RED_OBJECT)):
            current = controller(MODE_WAIT_14_HANDSHAKE)
            if after07:
                current._handle_command(serial_module.CMD_SELECT_RED_OBJECT)
            current._handle_command(target_command)
            current.mode = MODE_WAIT_14_HANDSHAKE
            current._handle_command(serial_module.CMD_TRACK_CENTER_DONE)
            objects = [block(200, 300, 4), neighbor] if after07 else [target, neighbor]
            for _ in range(1 if after07 else config.UART_FIRST_GREEN_DECISION_CONFIRM_FRAMES):
                current.submit(objects, LABELS, raw_objects=objects)
            self.assertEqual(current._pending_events, [build_event_packet(EVENT_ROI_READY)])
        current = controller(serial_module.CMD_SEARCH)
        current._handle_command(serial_module.CMD_SELECT_BLACK_OR_GREEN)
        current._handle_command(serial_module.CMD_SEARCH)
        current.submit([target, neighbor], LABELS, raw_objects=[target, neighbor])
        self.assertEqual(current._selection_diagnostic_policy, "supply_no_spacing")
        self.assertIsNone(current._latest_packet)

    def test_post07_supply_tracking_still_excludes_safety_zone_contents(self):
        current = controller(MODE_WAIT_14_HANDSHAKE)
        current._handle_command(serial_module.CMD_SELECT_BLACK_OR_GREEN)
        current.mode = MODE_WAIT_14_HANDSHAKE
        current._handle_command(serial_module.CMD_TRACK_CENTER_DONE)
        target = block(200, 300)
        zone = SimpleNamespace(x=150, y=250, w=100, h=100, class_id=1, score=0.9)
        current.submit([], LABELS, raw_objects=[target, zone])
        self.assertEqual(current._pending_events, [build_event_packet(EVENT_ROI_READY)])
        self.assertIsNone(current._latest_packet)

    def test_first_green_second_empty05_sends_e5_waits44_then_restarts_count(self):
        current = controller(MODE_FINAL_ROI_TRACK)

        def inspect_empty(start):
            with patch.object(serial_module, "_now_ms", return_value=start):
                current._handle_command(serial_module.CMD_SEARCH_BLUE_ZONE)
            self.assertEqual(write_near_view_request(current, start),
                             build_event_packet(serial_module.EVENT_TARGET_READY))
            for index in range(3):
                with patch.object(serial_module, "_now_ms", return_value=start + 500 + index * 50):
                    current.submit([], LABELS, raw_objects=[])
                if index < 2:
                    self.assertIsNone(current._next_outgoing_packet(start + 500 + index * 50))
            return current._next_outgoing_packet(start + 600)

        self.assertEqual(inspect_empty(1000), build_event_packet(EVENT_SKIP_TO_FINAL_TRACK))
        self.assertEqual(current._near_view_empty_cycles, 1)
        self.assertEqual(current._near_view_green_missing_cycles, 1)
        self.assertTrue(current._first_green_near_view_decided)
        # Tracking and same-target synchronization must not reset the cycle.
        current._handle_command(serial_module.CMD_SELECT_GREEN_OBJECT)
        current._handle_command(serial_module.CMD_SEARCH)
        self.assertEqual(current._near_view_empty_cycles, 1)
        self.assertEqual(inspect_empty(2000), build_event_packet(serial_module.EVENT_FIRST_GREEN_EMPTY_RECOVERY))
        self.assertEqual(current.mode, CMD_TRACK)
        self.assertTrue(current._track_green_recovery_waiting)
        self.assertFalse(current._first_green_both_sides_required)
        self.assertEqual(current._near_view_empty_cycles, 0)
        self.assertEqual(current._near_view_green_missing_cycles, 0)
        self.assertIsNone(current._latest_packet)
        self.assertEqual(current._pending_events, [])
        # Genuine current coordinates may reappear, but cannot release the
        # timed MCU pushes or generate14/E4 while E5 recovery waits for44.
        with patch.object(serial_module, "_now_ms", return_value=2800):
            current.submit([block(288, 224)], LABELS)
        self.assertEqual(current.mode, CMD_TRACK)
        self.assertTrue(current._track_green_recovery_waiting)
        self.assertEqual(current._next_outgoing_packet(2800),
                         serial_module.build_coordinate_packet(5, 0, 0))
        for now in (2900, 3000, 3100, 3200, 3300, 3400):
            with patch.object(serial_module, "_now_ms", return_value=now):
                current.submit([], LABELS)
            self.assertIsNone(current._next_outgoing_packet(now))
        with patch.object(serial_module, "_now_ms", return_value=3500):
            current._handle_command(serial_module.CMD_TRACK_RECOVERY_DONE)
            current.submit([block(288, 310)], LABELS)
        self.assertEqual(current._next_outgoing_packet(3500),
                         serial_module.build_coordinate_packet(5, 0, 86))
        self.assertEqual(current.mode, serial_module.CMD_SEARCH)
        self.assertFalse(current._track_green_recovery_waiting)
        self.assertEqual(inspect_empty(4000), build_event_packet(EVENT_SKIP_TO_FINAL_TRACK))
        self.assertEqual(current._near_view_empty_cycles, 1)
        self.assertEqual(current._near_view_green_missing_cycles, 1)

    def _confirm_near05_round(self, current, ids, start):
        with patch.object(serial_module, "_now_ms", return_value=start):
            current._handle_command(serial_module.CMD_SEARCH_BLUE_ZONE)
        self.assertEqual(write_near_view_request(current, start),
                         build_event_packet(serial_module.EVENT_TARGET_READY))
        objects = [near_view_block(120 + index * 30, 200, class_id)
                   for index, class_id in enumerate(ids)]
        for index in range(3):
            with patch.object(serial_module, "_now_ms", return_value=start + 500 + index * 50):
                current.submit(objects, LABELS, raw_objects=objects)
        result_at = start + 600
        if current._zone_load_recheck_source == "before06":
            result_at = confirm_final_load_check(current, ids, result_at)
        packet = current._next_outgoing_packet(result_at)
        if current._zone_load_recheck_source == "supply_empty":
            serial = SimpleNamespace(write=Mock(return_value=len(packet)))
            with patch.object(current, "_serial", serial), patch.object(serial_module, "_now_ms", return_value=result_at):
                current._write_packet(packet, current._outgoing_generation)
            self.assertIsNone(current._next_outgoing_packet(result_at + config.UART_SUPPLY_SEARCH_SETTLE_MS))
        return packet

    def _complete_near05_arrangement(self, current, start):
        with patch.object(serial_module, "_now_ms", return_value=start):
            current._handle_command(serial_module.CMD_ARRANGE_FIRST_DONE)
        self.assertEqual(current._next_outgoing_packet(start),
                         build_event_packet(serial_module.EVENT_ARRANGE_FIRST_ACK))
        with patch.object(serial_module, "_now_ms", return_value=start + 100):
            current._handle_command(serial_module.CMD_ARRANGE_LEFT_DONE)
        self.assertEqual(current.mode, MODE_POST_ARRANGE_CHECK)

    def test_near05_blue_without_acceptable_task_target_returns24(self):
        cases = (
            (False, serial_module.CMD_SELECT_GREEN_OBJECT, ([3], [3, 3], [3, 4], [3, 6], [3] * 5)),
            (True, serial_module.CMD_SELECT_RED_OBJECT, ([3], [3, 5], [3, 6], [3, 5, 6])),
            (True, serial_module.CMD_SELECT_GREEN_OBJECT, ([3], [3, 4], [3] * 5)),
            (True, serial_module.CMD_SELECT_BLACK_OBJECT, ([3], [3, 4], [3] * 5)),
            (True, serial_module.CMD_SELECT_BLACK_OR_GREEN, ([3], [3, 4], [3] * 5)),
        )
        for after07, command, id_lists in cases:
            for ids in id_lists:
                with self.subTest(after07=after07, command=command, ids=ids):
                    current = controller(MODE_FINAL_ROI_TRACK)
                    current._post07_started = after07
                    current._handle_command(command)
                    self.assertEqual(self._confirm_near05_round(current, ids, 1000),
                                     build_event_packet(EVENT_SKIP_TO_FINAL_TRACK))
                    self.assertEqual(current.mode, MODE_FINAL_ROI_TRACK)
                    self.assertEqual(current._pending_events, [])
                    self.assertEqual(current._near_view_empty_cycles, 0)
                    self.assertIsNone(current._latest_packet)
                    self.assertFalse(current._first_green_both_sides_required)

    def test_near05_blue_with_acceptable_target_still_arranges(self):
        for after07, command, ids in (
                (False, serial_module.CMD_SELECT_GREEN_OBJECT, [3, 5]),
                (True, serial_module.CMD_SELECT_RED_OBJECT, [3, 4]),
                (True, serial_module.CMD_SELECT_GREEN_OBJECT, [3, 6]),
                (True, serial_module.CMD_SELECT_BLACK_OBJECT, [3, 5]),
                (True, serial_module.CMD_SELECT_BLACK_OR_GREEN, [3, 5, 6])):
            with self.subTest(command=command, ids=ids):
                current = controller(MODE_FINAL_ROI_TRACK)
                current._post07_started = after07
                current._handle_command(command)
                self.assertEqual(self._confirm_near05_round(current, ids, 1000),
                                 build_event_packet(EVENT_ROI_READY))
                self.assertEqual(current.mode, MODE_ARRANGE_RIGHTMOST)
                self._complete_near05_arrangement(current, 1700)

    def test_first_green_blue_no_target24_retains_second_missing05_e5(self):
        current = controller(MODE_FINAL_ROI_TRACK)
        self.assertEqual(self._confirm_near05_round(current, [3], 1000),
                         build_event_packet(EVENT_SKIP_TO_FINAL_TRACK))
        self.assertEqual(current._near_view_green_missing_cycles, 1)
        self.assertEqual(self._confirm_near05_round(current, [3, 3], 2000),
                         build_event_packet(serial_module.EVENT_FIRST_GREEN_EMPTY_RECOVERY))
        self.assertEqual(current.mode, CMD_TRACK)
        self.assertTrue(current._track_green_recovery_waiting)
        self.assertEqual(current._near_view_empty_cycles, 0)
        self.assertEqual(current._near_view_green_missing_cycles, 0)
        self.assertEqual(current._pending_events, [])

    def test_post07_blue_only_repeated05_always24_not_e5(self):
        current = controller(MODE_FINAL_ROI_TRACK)
        current._handle_command(serial_module.CMD_SELECT_BLACK_OR_GREEN)
        for cycle in range(3):
            self.assertEqual(self._confirm_near05_round(current, [3] * (cycle + 1), 1000 + cycle * 1000),
                             build_event_packet(EVENT_SKIP_TO_FINAL_TRACK))
            self.assertEqual(current._near_view_green_missing_cycles, 0)
            self.assertEqual(current._near_view_empty_cycles, 0)

    def test_near05_blue_return24_has_own_settled_three_frame_bucket(self):
        current = controller(MODE_FINAL_ROI_TRACK)
        current._handle_command(serial_module.CMD_SELECT_BLACK_OR_GREEN)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            current._handle_command(serial_module.CMD_SEARCH_BLUE_ZONE)
        write_near_view_request(current, 1000)
        blue = [block(200, 200, 3)]
        with_supply = blue + [block(250, 200)]
        for now, objects in ((1001, blue), (1200, blue), (1249, blue),
                             (1250, blue), (1260, blue), (1270, with_supply),
                             (1280, blue), (1290, blue)):
            with patch.object(serial_module, "_now_ms", return_value=now):
                current.submit(objects, LABELS, raw_objects=objects)
            self.assertEqual(current.mode, serial_module.MODE_NEAR_VIEW_CHECK)
            self.assertEqual(current._pending_events, [])
        with patch.object(serial_module, "_now_ms", return_value=1300):
            current.submit(blue, LABELS, raw_objects=blue)
        self.assertEqual(current._next_outgoing_packet(1300), build_event_packet(EVENT_SKIP_TO_FINAL_TRACK))

    def test_near05_explicit_blue_target_keeps_existing_single_and_multiple_rules(self):
        for ids, expected in (([3], serial_module.EVENT_ZONE_SEARCH_STARTED),
                              ([3, 3], EVENT_ROI_READY)):
            current = controller(MODE_FINAL_ROI_TRACK)
            current._handle_command(serial_module.CMD_SELECT_BLUE_OBJECT)
            self.assertEqual(self._confirm_near05_round(current, ids, 1000), build_event_packet(expected))

    def test_near05_blue_no_target_uses_verified_counts_not_raw_cargo_claim(self):
        current = controller(MODE_FINAL_ROI_TRACK)
        current._handle_command(serial_module.CMD_SELECT_BLACK_OR_GREEN)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            current._handle_command(serial_module.CMD_SEARCH_BLUE_ZONE)
        write_near_view_request(current, 1000)
        blue, unverified_supply = block(200, 200, 3), block(250, 200)
        for index in range(3):
            with patch.object(serial_module, "_now_ms", return_value=1500 + index * 10):
                current.submit([blue], LABELS, raw_objects=[blue, unverified_supply])
        self.assertEqual(current._next_outgoing_packet(1520), build_event_packet(EVENT_SKIP_TO_FINAL_TRACK))
        self.assertEqual(current._pending_events, [])

    def test_first_green_two_no_green05_rounds_e5_even_with_foreign_objects(self):
        for first_ids, second_ids in (([4], [3]), ([6], [4]),
                                     ([4, 6], [3, 6]), ([], [4]), ([6], [])):
            with self.subTest(first_ids=first_ids, second_ids=second_ids):
                current = controller(MODE_FINAL_ROI_TRACK)
                expected_first = EVENT_ROI_READY if first_ids else EVENT_SKIP_TO_FINAL_TRACK
                self.assertEqual(self._confirm_near05_round(current, first_ids, 1000),
                                 build_event_packet(expected_first))
                self.assertEqual(current._near_view_green_missing_cycles, 1)
                if first_ids:
                    self._complete_near05_arrangement(current, 1700)
                    self.assertEqual(current._near_view_green_missing_cycles, 1)
                self.assertEqual(self._confirm_near05_round(current, second_ids, 2000),
                                 build_event_packet(serial_module.EVENT_FIRST_GREEN_EMPTY_RECOVERY))
                self.assertEqual(current.mode, CMD_TRACK)
                self.assertTrue(current._track_green_recovery_waiting)
                self.assertEqual(current._near_view_green_missing_cycles, 0)
                # No obsolete02 or24 may accompany the second-round E5.
                self.assertEqual(current._pending_events, [])
                for now in (2700, 2800, 2900):
                    with patch.object(serial_module, "_now_ms", return_value=now):
                        current.submit([], LABELS)
                    self.assertIsNone(current._next_outgoing_packet(now))

    def test_first_green_seen_in05_breaks_missing_streak_even_with_foreign_or_multiple(self):
        for middle_ids in ([5, 4], [5, 5], [5, 6]):
            with self.subTest(middle_ids=middle_ids):
                current = controller(MODE_FINAL_ROI_TRACK)
                self.assertEqual(self._confirm_near05_round(current, [], 1000),
                                 build_event_packet(EVENT_SKIP_TO_FINAL_TRACK))
                self.assertEqual(current._near_view_green_missing_cycles, 1)
                self.assertEqual(self._confirm_near05_round(current, middle_ids, 2000),
                                 build_event_packet(EVENT_ROI_READY))
                self.assertEqual(current._near_view_green_missing_cycles, 0)
                self._complete_near05_arrangement(current, 2700)
                self.assertEqual(self._confirm_near05_round(current, [4], 3000),
                                 build_event_packet(EVENT_ROI_READY))
                self.assertEqual(current._near_view_green_missing_cycles, 1)

    def test_first_green24_recapture_preserves_no_green05_count_with_foreign_cargo(self):
        # Only settled05 cargo resets a no-green streak. A visible green in24
        # is a capture candidate, not proof that it was retained in the frame.
        cases = (([], [4]), ([], [6]), ([3], [4, 6]),
                 ([4], [6]), ([6], [4]), ([4, 6], [3]))
        for first_ids, second_ids in cases:
            with self.subTest(first_ids=first_ids, second_ids=second_ids):
                current = controller(MODE_FINAL_ROI_TRACK)
                first_event = (EVENT_SKIP_TO_FINAL_TRACK if not first_ids or 3 in first_ids
                               else EVENT_ROI_READY)
                self.assertEqual(self._confirm_near05_round(current, first_ids, 1000),
                                 build_event_packet(first_event))
                self.assertEqual(current._near_view_green_missing_cycles, 1)
                if first_event == EVENT_ROI_READY:
                    self._complete_near05_arrangement(current, 1700)

                target = block(288, 340)
                with patch.object(serial_module, "_now_ms", return_value=1900):
                    current.submit([target], LABELS, raw_objects=[target])
                self.assertEqual(current.mode, MODE_FINAL_ROI_TRACK)
                self.assertEqual(current._near_view_green_missing_cycles, 1)
                expected_packet = (build_event_packet(EVENT_SKIP_TO_FINAL_TRACK)
                                   if first_event == EVENT_ROI_READY
                                   else serial_module.build_coordinate_packet(5, 0, -44))
                self.assertEqual(current._next_outgoing_packet(1900), expected_packet)

                inside = block(288, 400)
                with patch.object(serial_module, "_now_ms", return_value=1910):
                    current.submit([inside], LABELS, raw_objects=[inside])
                self.assertEqual(current._next_outgoing_packet(1910),
                                 build_event_packet(serial_module.EVENT_FINAL_TARGET_IN_ROI))
                self.assertEqual(current._near_view_green_missing_cycles, 1)
                self.assertEqual(self._confirm_near05_round(current, second_ids, 2000),
                                 build_event_packet(serial_module.EVENT_FIRST_GREEN_EMPTY_RECOVERY))
                self.assertEqual(current.mode, CMD_TRACK)
                self.assertTrue(current._track_green_recovery_waiting)
                self.assertEqual(current._near_view_green_missing_cycles, 0)
                self.assertEqual(current._pending_events, [])

    def test_no_green05_requires_settling_three_frames_and_counts_once_per_round(self):
        current = controller(MODE_FINAL_ROI_TRACK)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            current._handle_command(serial_module.CMD_SEARCH_BLUE_ZONE)
        write_near_view_request(current, 1000)
        foreign = [near_view_block(200, 200, 4)]
        for now in (1001, 1200, 1249, 1250, 1300):
            with patch.object(serial_module, "_now_ms", return_value=now):
                current.submit(foreign, LABELS, raw_objects=foreign)
            self.assertEqual(current._near_view_green_missing_cycles, 0)
            self.assertEqual(current._pending_events, [])
        with patch.object(serial_module, "_now_ms", return_value=1350):
            current.submit(foreign, LABELS, raw_objects=foreign)
        self.assertEqual(current._near_view_green_missing_cycles, 1)
        self.assertEqual(current._next_outgoing_packet(1350), build_event_packet(EVENT_ROI_READY))
        for now in (1450, 1550, 1650):
            with patch.object(serial_module, "_now_ms", return_value=now):
                current.submit(foreign, LABELS, raw_objects=foreign)
            self.assertEqual(current._near_view_green_missing_cycles, 1)

    def test_foreign_only_and_green_foreign_frames_cannot_mix_absence_confirmation(self):
        current = controller(MODE_FINAL_ROI_TRACK)
        self._confirm_near05_round(current, [], 1000)
        with patch.object(serial_module, "_now_ms", return_value=2000):
            current._handle_command(serial_module.CMD_SEARCH_BLUE_ZONE)
        write_near_view_request(current, 2000)
        foreign = [near_view_block(200, 200, 4)]
        with_green = foreign + [block(250, 200)]
        for now, objects in ((2500, foreign), (2550, foreign), (2600, with_green),
                             (2650, foreign), (2700, foreign)):
            with patch.object(serial_module, "_now_ms", return_value=now):
                current.submit(objects, LABELS, raw_objects=objects)
            self.assertEqual(current.mode, serial_module.MODE_NEAR_VIEW_CHECK)
            self.assertEqual(current._pending_events, [])
        self.assertEqual(current._near_view_green_missing_cycles, 0)
        with patch.object(serial_module, "_now_ms", return_value=2750):
            current.submit(foreign, LABELS, raw_objects=foreign)
        self.assertEqual(current._near_view_green_missing_cycles, 1)
        self.assertEqual(current._next_outgoing_packet(2750), build_event_packet(EVENT_ROI_READY))

    def test_first_green_missing05_history_survives_same_task_but_resets_on_new_task(self):
        current = controller(MODE_FINAL_ROI_TRACK)
        self._confirm_near05_round(current, [4], 1000)
        self._complete_near05_arrangement(current, 1700)
        for command in (serial_module.CMD_SELECT_GREEN_OBJECT, serial_module.CMD_SEARCH,
                        serial_module.CMD_TRACK):
            current._handle_command(command)
            self.assertEqual(current._near_view_green_missing_cycles, 1)
        current._handle_command(serial_module.CMD_SELECT_RED_OBJECT)
        self.assertEqual(current._near_view_green_missing_cycles, 0)

    def test_post07_green_foreign05_does_not_trigger_first_green_e5(self):
        current = controller(MODE_FINAL_ROI_TRACK)
        current._post07_started = True
        for cycle in range(3):
            self.assertEqual(self._confirm_near05_round(current, [4], 1000 + cycle * 1000),
                             build_event_packet(EVENT_ROI_READY))
            self.assertEqual(current._near_view_green_missing_cycles, 0)
            self._complete_near05_arrangement(current, 1700 + cycle * 1000)

    def test_empty05_counts_once_and_success_or_target_change_resets(self):
        current = controller(MODE_FINAL_ROI_TRACK)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            current._handle_command(serial_module.CMD_SEARCH_BLUE_ZONE)
        write_near_view_request(current, 1000)
        # Settling and incomplete confirmation never increment the count.
        for now in (1100, 1500, 1550):
            with patch.object(serial_module, "_now_ms", return_value=now):
                current.submit([], LABELS)
            self.assertEqual(current._near_view_empty_cycles, 0)
        with patch.object(serial_module, "_now_ms", return_value=1600):
            current.submit([], LABELS)
        self.assertEqual(current._near_view_empty_cycles, 1)
        current._next_outgoing_packet(1600)
        for now in (1700, 1750):
            with patch.object(serial_module, "_now_ms", return_value=now):
                current.submit([], LABELS)
        self.assertEqual(current._near_view_empty_cycles, 1)
        with patch.object(serial_module, "_now_ms", return_value=2000):
            current._handle_command(serial_module.CMD_SEARCH_BLUE_ZONE)
        write_near_view_request(current, 2000)
        for now in (2500, 2550, 2600):
            with patch.object(serial_module, "_now_ms", return_value=now):
                current.submit([block(288, 310)], LABELS)
        self.assertEqual(current._near_view_empty_cycles, 1)
        delivered_at = confirm_final_load_check(current, [5], 2600)
        self.assertEqual(current._near_view_empty_cycles, 0)
        self.assertEqual(current._next_outgoing_packet(delivered_at), serial_module.build_coordinate_packet(5, 0, 0))
        self.assertEqual(current._next_outgoing_packet(delivered_at), build_event_packet(serial_module.EVENT_ZONE_SEARCH_STARTED))
        current._near_view_empty_cycles = 1
        current._near_view_green_missing_cycles = 1
        current._handle_command(serial_module.CMD_SELECT_RED_OBJECT)
        self.assertEqual(current._near_view_empty_cycles, 0)
        self.assertEqual(current._near_view_green_missing_cycles, 0)

    def test_later_tasks_repeated_empty05_never_force_first_green_arrangement(self):
        for label in ("sqareredgreen", "sqarered", "triangualrblack", serial_module.TARGET_BLACK_OR_GREEN):
            with self.subTest(label=label):
                current = controller(MODE_FINAL_ROI_TRACK)
                current.target_label = label
                current._post07_started = True
                for cycle in range(3):
                    start = 1000 + cycle * 1000
                    with patch.object(serial_module, "_now_ms", return_value=start):
                        current._handle_command(serial_module.CMD_SEARCH_BLUE_ZONE)
                    write_near_view_request(current, start)
                    for index in range(3):
                        with patch.object(serial_module, "_now_ms", return_value=start + 500 + index * 50):
                            current.submit([], LABELS)
                    self.assertEqual(current._near_view_empty_cycles, cycle + 1)
                    result_at = start + 600
                    packet = current._next_outgoing_packet(result_at)
                    self.assertEqual(packet, build_event_packet(EVENT_SKIP_TO_FINAL_TRACK))
                    serial = SimpleNamespace(write=Mock(return_value=len(packet)))
                    with patch.object(current, "_serial", serial), patch.object(serial_module, "_now_ms", return_value=result_at):
                        current._write_packet(packet, current._outgoing_generation)
                    self.assertIsNone(current._next_outgoing_packet(result_at + config.UART_SUPPLY_SEARCH_SETTLE_MS))
                    self.assertFalse(current._first_green_both_sides_required)

    def test_post07_near05_supply_counts_and_foreign_colors(self):
        cases = (([5], serial_module.EVENT_ZONE_SEARCH_STARTED),
                 ([6], serial_module.EVENT_ZONE_SEARCH_STARTED),
                 ([5, 6], serial_module.EVENT_ZONE_SEARCH_STARTED),
                 ([5, 5, 6], serial_module.EVENT_ZONE_SEARCH_STARTED),
                 ([], EVENT_SKIP_TO_FINAL_TRACK),
                 ([5, 5, 6, 6], EVENT_ROI_READY),
                 ([5, 3], EVENT_ROI_READY), ([6, 4], EVENT_ROI_READY),
                 ([3], EVENT_SKIP_TO_FINAL_TRACK), ([4], EVENT_ROI_READY))
        for command in (serial_module.CMD_SELECT_GREEN_OBJECT,
                        serial_module.CMD_SELECT_BLACK_OBJECT,
                        serial_module.CMD_SELECT_BLACK_OR_GREEN):
            for ids, expected in cases:
                with self.subTest(command=command, ids=ids):
                    if command == serial_module.CMD_SELECT_BLACK_OR_GREEN and ids in ([5], [6]):
                        expected = serial_module.EVENT_COLLECT_MORE_SUPPLIES
                    current = controller(serial_module.MODE_NEAR_VIEW_CHECK)
                    current._handle_command(serial_module.CMD_SELECT_RED_OBJECT)
                    current._handle_command(command)
                    with patch.object(serial_module, "_now_ms", return_value=100):
                        current._handle_command(serial_module.CMD_SEARCH_BLUE_ZONE)
                    self.assertEqual(write_near_view_request(current, 100),
                                     build_event_packet(serial_module.EVENT_TARGET_READY))
                    objects = [near_view_block(100 + index * 30, 200, class_id)
                               for index, class_id in enumerate(ids)]
                    # Camera settling and stable action confirmation still apply.
                    with patch.object(serial_module, "_now_ms", return_value=101):
                        current.submit(objects, LABELS, raw_objects=objects)
                    self.assertEqual(current._pending_events, [])
                    for index in range(config.UART_NEAR_VIEW_STABLE_FRAMES):
                        with patch.object(serial_module, "_now_ms", return_value=1000 + index):
                            current.submit(objects, LABELS, raw_objects=objects)
                    result_at = 1100
                    if expected == serial_module.EVENT_ZONE_SEARCH_STARTED:
                        result_at = confirm_final_load_check(current, ids, result_at)
                        self.assertEqual(current._next_outgoing_packet(result_at),
                                         serial_module.build_coordinate_packet(5, 0, 0))
                    self.assertEqual(current._next_outgoing_packet(result_at), build_event_packet(expected))
        # First green still arranges a black/green mix, and red tasks remain strict.
        for after07, command in ((False, serial_module.CMD_SELECT_GREEN_OBJECT),
                                (True, serial_module.CMD_SELECT_RED_OBJECT)):
            current = controller(serial_module.MODE_NEAR_VIEW_CHECK)
            current._handle_command(command)
            with patch.object(serial_module, "_now_ms", return_value=100):
                current._handle_command(serial_module.CMD_SEARCH_BLUE_ZONE)
            write_near_view_request(current, 100)
            objects = [near_view_block(200, 200, 5), near_view_block(230, 200, 6)]
            for index in range(config.UART_NEAR_VIEW_STABLE_FRAMES):
                with patch.object(serial_module, "_now_ms", return_value=1000 + index):
                    current.submit(objects, LABELS, raw_objects=objects)
            self.assertEqual(current._next_outgoing_packet(1100), build_event_packet(EVENT_ROI_READY))

    def test_post_22_qualified_emits_24_ineligible_visible_emits_04(self):
        current = controller(MODE_POST_ARRANGE_CHECK)
        target = block(200, 300)
        current.submit([target], LABELS, raw_objects=[target])
        self.assertEqual(current.mode, MODE_FINAL_ROI_TRACK)
        self.assertIn(
            build_event_packet(EVENT_SKIP_TO_FINAL_TRACK),
            current._pending_events,
        )

        current = controller(MODE_POST_ARRANGE_CHECK)
        current._first_green_near_view_decided = True
        nearby = block(250, 300, 3)
        current.submit([target, nearby], LABELS, raw_objects=[target, nearby])
        self.assertEqual(current.mode, serial_module.MODE_WAIT_04_HANDSHAKE)
        self.assertEqual(
            current._next_outgoing_packet(current._latest_submit_ms),
            build_event_packet(serial_module.EVENT_SEARCH_CENTER_REACHED),
        )
        self.assertEqual(current._next_outgoing_packet(current._latest_submit_ms),
                         serial_module.build_coordinate_packet(5, -88, 76))

    def test_post22_raw_target_rejected_by_verifier_approaches_before_e2(self):
        current = controller(MODE_POST_ARRANGE_CHECK)
        current._handle_command(serial_module.CMD_SELECT_RED_OBJECT)
        current.mode = MODE_POST_ARRANGE_CHECK
        with patch.object(serial_module, "_now_ms", return_value=1000):
            current.submit([], LABELS, raw_objects=[block(250, 200, 4), block(255, 210, 3)])
        self.assertEqual(current._next_outgoing_packet(1000),
                         build_event_packet(serial_module.EVENT_SEARCH_CENTER_REACHED))
        self.assertEqual(current._next_outgoing_packet(1001),
                         serial_module.build_coordinate_packet(4, -38, -24))
        self.assertEqual(current._last_arrangement_no_object_ms, 0)

    def test_first_green_post22_approaches_before_arrange_or_final_decision(self):
        for needs_arrangement in (True, False):
            with self.subTest(needs_arrangement=needs_arrangement):
                current = controller(serial_module.MODE_ARRANGE_LEFTMOST)
                current._handle_command(serial_module.CMD_ARRANGE_LEFT_DONE)
                target, neighbor = block(200, 300), block(230, 300, 3)
                with patch.object(serial_module, "_now_ms", return_value=1000):
                    current.submit([target, neighbor], LABELS, raw_objects=[target, neighbor])
                self.assertEqual(current.mode, serial_module.MODE_WAIT_04_HANDSHAKE)
                self.assertEqual(current._track_target_lock, current._final_target_identity(target))
                self.assertEqual(current._next_outgoing_packet(1000),
                                 build_event_packet(serial_module.EVENT_SEARCH_CENTER_REACHED))
                self.assertEqual(current._next_outgoing_packet(1001),
                                 serial_module.build_coordinate_packet(5, -88, 76))
                # Already centered while waiting: coordinates, but no early 14.
                centered, adjacent = block(288, 250), block(295, 250, 3)
                with patch.object(serial_module, "_now_ms", return_value=1100):
                    current.submit([], LABELS, raw_objects=[centered, adjacent])
                self.assertEqual(current.mode, serial_module.MODE_WAIT_04_HANDSHAKE)
                self.assertEqual(current._next_outgoing_packet(1100),
                                 serial_module.build_coordinate_packet(5, 0, 26))
                anchor = current._track_target_lock
                current._handle_command(CMD_TRACK)
                self.assertEqual(current._track_target_lock, anchor)
                with patch.object(serial_module, "_now_ms", return_value=1200):
                    current.submit([], LABELS, raw_objects=[centered, adjacent])
                self.assertEqual(current._next_outgoing_packet(1200),
                                 build_event_packet(EVENT_TRACK_CENTER_REACHED))
                self.assertEqual(current.mode, MODE_WAIT_14_HANDSHAKE)
                objects = [centered, adjacent] if needs_arrangement else [centered]
                with patch.object(serial_module, "_now_ms", return_value=1300):
                    current._handle_command(serial_module.CMD_TRACK_CENTER_DONE)
                    for _ in range(config.UART_FIRST_GREEN_DECISION_CONFIRM_FRAMES):
                        current.submit(objects, LABELS, raw_objects=objects)
                expected = EVENT_ROI_READY if needs_arrangement else EVENT_SKIP_TO_FINAL_TRACK
                self.assertEqual(current._next_outgoing_packet(1300), build_event_packet(expected))
                if needs_arrangement:
                    current._handle_command(serial_module.CMD_ARRANGE_FIRST_DONE)
                    self.assertEqual(current._next_outgoing_packet(1400),
                                     build_event_packet(serial_module.EVENT_ARRANGE_FIRST_ACK))
                    current._handle_command(serial_module.CMD_ARRANGE_LEFT_DONE)
                    with patch.object(serial_module, "_now_ms", return_value=1500):
                        current.submit(objects, LABELS, raw_objects=objects)
                    self.assertEqual(current._next_outgoing_packet(1500),
                                     build_event_packet(serial_module.EVENT_SEARCH_CENTER_REACHED))
                else:
                    self.assertEqual(current.mode, MODE_FINAL_ROI_TRACK)

    def test_first_green_post22_qualified_goes_directly_to_24(self):
        current = controller(serial_module.MODE_ARRANGE_LEFTMOST)
        current._handle_command(serial_module.CMD_ARRANGE_LEFT_DONE)
        target = block(200, 300)
        with patch.object(serial_module, "_now_ms", return_value=3000):
            current.submit([target], LABELS, raw_objects=[target])
        self.assertEqual(current.mode, MODE_FINAL_ROI_TRACK)
        self.assertEqual(current._next_outgoing_packet(3000), build_event_packet(EVENT_SKIP_TO_FINAL_TRACK))

    def test_first_green_post22_raw_visible_requests_04_and_fresh_coordinates(self):
        current = controller(MODE_POST_ARRANGE_CHECK)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            current.submit([], LABELS, raw_objects=[block(200, 300)])
        self.assertEqual(current._next_outgoing_packet(1000),
                         build_event_packet(serial_module.EVENT_SEARCH_CENTER_REACHED))
        self.assertEqual(current.mode, serial_module.MODE_WAIT_04_HANDSHAKE)
        self.assertEqual(current._track_target_lock, (5, 200.0, 300.0, 400.0))
        self.assertEqual(current._next_outgoing_packet(1001),
                         serial_module.build_coordinate_packet(5, -88, 76))

    def test_after_post22_e2_raw_detection_requests_04_and_inherits_lock(self):
        for command, actual_id in ((serial_module.CMD_SELECT_RED_OBJECT, 4),
                                   (serial_module.CMD_SELECT_BLACK_OBJECT, 6),
                                   (serial_module.CMD_SELECT_BLACK_OR_GREEN, 5)):
            with self.subTest(command=command):
                current = controller(serial_module.MODE_ARRANGE_LEFTMOST)
                current._handle_command(command)
                current.mode = serial_module.MODE_ARRANGE_LEFTMOST
                current._handle_command(serial_module.CMD_ARRANGE_LEFT_DONE)
                with patch.object(serial_module, "_now_ms", return_value=1000):
                    current.submit([], LABELS, raw_objects=[])
                self.assertEqual(current._next_outgoing_packet(1000),
                                 build_event_packet(EVENT_ARRANGEMENT_NO_OBJECT))
                # Mere unrelated YOLO detections cannot change the selected task.
                with patch.object(serial_module, "_now_ms", return_value=1500):
                    current.submit([], LABELS, raw_objects=[block(288, 250, 3)])
                self.assertEqual(current._next_outgoing_packet(1500),
                                 build_event_packet(EVENT_ARRANGEMENT_NO_OBJECT))
                target, blocker = block(210, 300, actual_id), block(215, 310, 3)
                with patch.object(serial_module, "_now_ms", return_value=1600), \
                        patch.object(current, "_select_final_roi_target",
                                     side_effect=AssertionError("E2 recovery must bypass merged screening")):
                    current.submit([], LABELS, raw_objects=[target, blocker])
                self.assertEqual(current.mode, serial_module.MODE_WAIT_04_HANDSHAKE)
                self.assertEqual(current._next_outgoing_packet(1600),
                                 build_event_packet(serial_module.EVENT_SEARCH_CENTER_REACHED))
                self.assertEqual(current._next_outgoing_packet(1601),
                                 serial_module.build_coordinate_packet(actual_id, -78, 76))
                self.assertIsNone(current._next_outgoing_packet(1602))
                anchor = current._track_target_lock
                current._handle_command(CMD_TRACK)
                self.assertEqual(current.mode, CMD_TRACK)
                self.assertEqual(current._track_target_lock, anchor)
                centered = block(288, 250, actual_id)
                with patch.object(serial_module, "_now_ms", return_value=1700):
                    current.submit([], LABELS, raw_objects=[centered])
                self.assertEqual(current._next_outgoing_packet(1700),
                                 build_event_packet(EVENT_TRACK_CENTER_REACHED))
                self.assertEqual(current.mode, MODE_WAIT_14_HANDSHAKE)

    def test_e2_recovery_never_tracks_safe_zone_contents_or_stale_coordinates(self):
        current = controller(MODE_POST_ARRANGE_CHECK)
        current._handle_command(serial_module.CMD_SELECT_RED_OBJECT)
        current.mode = MODE_POST_ARRANGE_CHECK
        with patch.object(serial_module, "_now_ms", return_value=1000):
            current.submit([], LABELS, raw_objects=[])
        self.assertEqual(current._next_outgoing_packet(1000), build_event_packet(EVENT_ARRANGEMENT_NO_OBJECT))
        target = block(200, 300, 4)
        zone = SimpleNamespace(x=150, y=250, w=100, h=100, class_id=2, score=0.9)
        with patch.object(serial_module, "_now_ms", return_value=1100):
            current.submit([], LABELS, raw_objects=[zone, target])
        self.assertEqual(current.mode, MODE_POST_ARRANGE_CHECK)
        self.assertIsNone(current._next_outgoing_packet(1100))
        with patch.object(serial_module, "_now_ms", return_value=1200):
            current.submit([], LABELS, raw_objects=[target])
        self.assertEqual(current._next_outgoing_packet(1200),
                         build_event_packet(serial_module.EVENT_SEARCH_CENTER_REACHED))
        with patch.object(serial_module, "_now_ms", return_value=1250):
            current.submit([], LABELS, raw_objects=[])
        self.assertIsNone(current._next_outgoing_packet(1250))

    def test_first_green_post22_missing_uses_one_e4_then_44_search03(self):
        current = controller(serial_module.MODE_ARRANGE_LEFTMOST)
        current._handle_command(serial_module.CMD_ARRANGE_LEFT_DONE)
        for index in range(config.UART_TRACK_LOST_CONFIRM_FRAMES):
            now = 1000 + index * 100
            with patch.object(serial_module, "_now_ms", return_value=now):
                current.submit([], LABELS, raw_objects=[block(200, 300, 4)])
            packet = current._next_outgoing_packet(now)
            if index + 1 < config.UART_TRACK_LOST_CONFIRM_FRAMES:
                self.assertIsNone(packet)
            else:
                self.assertEqual(packet, build_event_packet(serial_module.EVENT_TRACK_NO_TARGET))
        self.assertEqual(current.mode, CMD_TRACK)
        with patch.object(serial_module, "_now_ms", return_value=2000):
            current.submit([], LABELS, raw_objects=[])
        self.assertIsNone(current._next_outgoing_packet(2000))
        target = block(288, 250)
        with patch.object(serial_module, "_now_ms", return_value=2100):
            current.submit([target], LABELS, raw_objects=[target])
        self.assertEqual(current.mode, CMD_TRACK)
        self.assertEqual(current._pending_events, [])
        self.assertIsNotNone(current._next_outgoing_packet(2100))
        with patch.object(serial_module, "_now_ms", return_value=2200):
            current._handle_command(serial_module.CMD_TRACK_RECOVERY_DONE)
            for _ in range(config.UART_FIRST_GREEN_DECISION_CONFIRM_FRAMES):
                current.submit([target], LABELS, raw_objects=[target])
        self.assertEqual(current._next_outgoing_packet(2200), serial_module.build_coordinate_packet(5, 0, 26))
        self.assertEqual(current.mode, serial_module.CMD_SEARCH)

    def test_later_red_post22_still_uses_e2(self):
        current = controller(MODE_POST_ARRANGE_CHECK)
        current.target_label = "sqarered"
        with patch.object(serial_module, "_now_ms", return_value=1000):
            current.submit([], LABELS, raw_objects=[])
        self.assertEqual(current._next_outgoing_packet(1000), build_event_packet(EVENT_ARRANGEMENT_NO_OBJECT))

    def test_04_uses_frame_center_and_y_can_pass_below_it(self):
        current = controller(CMD_TRACK)
        self.assertEqual(current._reference_point(CMD_TRACK), (288.0, 224.0))
        self.assertEqual(
            current._reference_point(MODE_WAIT_14_HANDSHAKE),
            (288.0, 403.5),
        )
        target = block(288, 300)  # below image center, still outside tracking ROI
        current.submit([target], LABELS, raw_objects=[target])
        self.assertEqual(current.mode, MODE_WAIT_14_HANDSHAKE)
        self.assertIn(
            build_event_packet(EVENT_TRACK_CENTER_REACHED),
            current._pending_events,
        )

        too_high = controller(CMD_TRACK)
        target = block(288, 203)
        too_high.submit([target], LABELS, raw_objects=[target])
        self.assertEqual(too_high.mode, CMD_TRACK)

    def test_debug_shows_all_close_neighbors(self):
        preview = DebugTargetPreview()
        target = block(200, 300)
        neighbors = [
            block(100, 300, 3),
            block(150, 300, 4),
            block(350, 300, 6),
        ]
        for _ in range(3):
            _, white_boxes, yellow_box, _ = preview.select(
                [target], LABELS, [target] + neighbors, "sqareredgreen"
            )
            self.assertEqual(white_boxes, neighbors)
            self.assertIsNone(yellow_box)

    def test_debug_single_green_is_yellow_on_every_detected_frame(self):
        preview = DebugTargetPreview()
        target = block(200, 300)
        for frame in range(1, 4):
            candidates, white_boxes, yellow_box, _ = preview.select(
                [target], LABELS, [target], "sqareredgreen"
            )
            self.assertEqual(candidates, [target])
            self.assertEqual(white_boxes, [])
            self.assertIs(yellow_box, target)

    def test_debug_single_green_remains_yellow_after_jumps_and_reacquisition(self):
        preview = DebugTargetPreview(tracking_roi=[0, 359, 576, 89])
        for x, y in ((200, 300), (500, 80), (40, 400)):
            target = block(x, y)
            _, neighbors, selected, _ = preview.select(
                [target], LABELS, [target], "sqareredgreen"
            )
            self.assertEqual(neighbors, [])
            self.assertIs(selected, target)
        _, _, selected, _ = preview.select([], LABELS, [], "sqareredgreen")
        self.assertIsNone(selected)
        target = block(280, 330)
        _, _, selected, _ = preview.select([target], LABELS, [target], "sqareredgreen")
        self.assertIs(selected, target)

    def test_debug_raw_single_green_visible_before_verification(self):
        preview = DebugTargetPreview()
        for x, y in ((100, 200), (450, 400), (200, 300)):
            target = block(x, y)
            candidates, neighbors, selected, _ = preview.select(
                [], LABELS, [target], "sqareredgreen"
            )
            self.assertEqual(candidates, [target])
            self.assertEqual(neighbors, [])
            self.assertIs(selected, target)

    def test_debug_does_not_invent_boxes_on_yolo_miss_or_select_safety_zone_contents(self):
        preview = DebugTargetPreview()
        target = block(100, 300)
        candidates, _, selected, _ = preview.select(
            [target], LABELS, [], "sqareredgreen"
        )
        self.assertEqual(candidates, [])
        self.assertIsNone(selected)
        zone = SimpleNamespace(x=0, y=200, w=300, h=200, class_id=1, score=0.9)
        candidates, _, selected, _ = preview.select(
            [], LABELS, [target, zone], "sqareredgreen"
        )
        self.assertEqual(candidates, [])
        self.assertIsNone(selected)

    def test_debug_single_display_does_not_bypass_raw_neighbor_rejection(self):
        preview = DebugTargetPreview()
        target = block(200, 300)
        unverified_neighbor = block(240, 320, 3)
        for _ in range(4):
            _, neighbors, selected, _ = preview.select(
                [target], LABELS, [target, unverified_neighbor], "sqareredgreen"
            )
            self.assertEqual(neighbors, [unverified_neighbor])
            self.assertIsNone(selected)

    def test_distant_block_with_large_y_gap_is_not_a_neighbor(self):
        target = block(100, 310)
        distant_red = block(350, 100, 4)
        self.assertTrue(
            VisionSerialController._matches_final_primary_spacing(
                target, [target, distant_red]
            )
        )
        preview = DebugTargetPreview()
        for frame in range(1, 4):
            candidates, white_boxes, yellow_box, _ = preview.select(
                [target], LABELS, [target, distant_red], "sqareredgreen"
            )
            self.assertEqual(candidates, [target])
            self.assertEqual(white_boxes, [])
            self.assertIs(yellow_box, target if frame == 3 else None)


class FirstGreenRecovery44SearchTests(unittest.TestCase):
    """RX44 restarts real-default03 only for pre07 green, including after05."""

    def setUp(self):
        for helper in (patch("builtins.print"), patch.object(config, "VISION_DIAGNOSTICS_ENABLED", False)):
            helper.start()
            self.addCleanup(helper.stop)

    def make(self, mode=CMD_TRACK, after05=False):
        with patch.object(config, "UART_COORDINATE_ENABLED", False):
            current = VisionSerialController(576, 448, make_bottom_center_roi(576, 448))
        current.mode = mode
        current.target_label = "sqareredgreen"
        current._first_green_near_view_decided = after05
        current._serial = SimpleNamespace(write=Mock(side_effect=lambda data: len(data)))
        return current

    def command(self, current, command=0x44, now=1000):
        with patch.object(serial_module, "_now_ms", return_value=now):
            current.process_received_bytes(serial_module.build_command_packet(command))

    def observe(self, current, objects=(), now=1100, generation=None, raw=None):
        if generation is None:
            generation = current.frame_generation()
        with patch.object(serial_module, "_now_ms", return_value=now):
            return current.submit(list(objects), LABELS,
                                  raw_objects=list(objects) if raw is None else list(raw),
                                  frame_generation=generation)

    def write_next(self, current, now):
        with patch.object(serial_module, "_now_ms", return_value=now):
            packet = current._next_outgoing_packet(now)
            if packet is not None:
                current._write_packet(packet, current._outgoing_generation)
            return packet

    def test44_clears_old_output_and_locks_but_preserves_first_mission_history(self):
        for mode in (CMD_TRACK, MODE_WAIT_14_HANDSHAKE):
            for after05 in (False, True):
                with self.subTest(mode=mode, after05=after05):
                    current = self.make(mode, after05)
                    old = current._final_target_identity(block(100, 150))
                    current._search_target_lock = current._track_target_lock = current._final_target_lock = old
                    current._track_green_recovery_waiting = current._track_no_target_sent = True
                    current._initial_arrangement_decision_pending = True
                    current._near_view_empty_cycles = current._near_view_green_missing_cycles = 1
                    current._micro_rounds, current._micro42_rounds = 2, 1
                    current._first_green_both_sides_required = True
                    current._latest_packet = serial_module.build_coordinate_packet(5, 30, 90)
                    current._coordinate_in_flight = current._latest_packet
                    current._pending_events = [build_event_packet(event) for event in (0xE4, 0x14, 0x02, 0x24, 0x34)]
                    generation, epoch = current.frame_context()
                    self.command(current)
                    self.assertEqual(current.frame_context(), (generation + 1, epoch))
                    self.assertEqual(current.mode, serial_module.CMD_SEARCH)
                    self.assertEqual(current.target_label, "sqareredgreen")
                    self.assertFalse(current._post07_started)
                    self.assertEqual(current._first_green_near_view_decided, after05)
                    self.assertEqual((current._near_view_empty_cycles, current._near_view_green_missing_cycles), (1, 1))
                    self.assertFalse(current._track_green_recovery_waiting)
                    self.assertFalse(current._track_no_target_sent)
                    self.assertFalse(current._initial_arrangement_decision_pending)
                    self.assertFalse(current._first_green_both_sides_required)
                    self.assertEqual((current._micro_rounds, current._micro42_rounds), (0, 0))
                    for name in ("_search_target_lock", "_track_target_lock", "_final_target_lock",
                                 "_latest_packet", "_coordinate_in_flight", "_initial_confirmation_anchor"):
                        self.assertIsNone(getattr(current, name))
                    self.assertEqual(current._pending_events, [])
                    self.assertIsNone(self.write_next(current, 1010))

    def test44_reacquires_green_with_normal03_rules_then_follows_raw_yolo(self):
        for after05 in (False, True):
            with self.subTest(after05=after05):
                current = self.make(after05=after05)
                old, near = block(100, 150), block(330, 250)
                current._track_target_lock = current._final_target_identity(old)
                self.command(current)
                neighbor = block(335, 250, 3)
                for now in (1100, 1150):
                    self.assertIsNone(self.observe(current, [old, near, neighbor], now))
                    self.assertIsNone(current._latest_packet)
                self.assertIs(self.observe(current, [old, near, neighbor], 1200), near)
                self.assertEqual(self.write_next(current, 1200), serial_module.build_coordinate_packet(5, 17, -89))
                self.assertEqual(current.mode, serial_module.CMD_SEARCH)
                moved = block(500, 250)
                moved.w, moved.h = 100, 60
                self.assertIs(self.observe(current, [], 1250, raw=[moved, block(330, 250, 4)]), moved)
                self.assertEqual(current._latest_packet[2], 5)

    def test44_rejects_old_inflight_images_and_prepared_events(self):
        current = self.make()
        generation = current.frame_generation()
        packet = serial_module.build_coordinate_packet(5, 10, 20)
        self.command(current)
        target = block(330, 250)
        for now in (1100, 1150, 1200):
            self.assertIsNone(self.observe(current, [target], now, generation))
        with patch.object(serial_module, "_now_ms", return_value=1200):
            for stale in (packet, build_event_packet(0x14), build_event_packet(0x02), build_event_packet(0x24)):
                current._write_packet(stale, generation)
        current._serial.write.assert_not_called()
        for now in (1250, 1300):
            self.assertIsNone(self.observe(current, [target], now))
        self.assertIs(self.observe(current, [target], 1350), target)

    def test_complete44_03_04_14_path_requests_direct_capture_even_with_neighbors(self):
        for after05 in (False, True):
            with self.subTest(after05=after05):
                current = self.make(after05=after05)
                self.command(current)
                search_target = block(313, 340)
                for now in (1100, 1150, 1200):
                    self.observe(current, [search_target], now)
                self.assertEqual(current.mode, serial_module.MODE_WAIT_04_HANDSHAKE)
                self.assertEqual(self.write_next(current, 1200), build_event_packet(0x04))
                self.observe(current, [search_target], 1250)
                self.assertNotIn(build_event_packet(0x14), current._pending_events)
                self.command(current, 0x04, 1300)
                target = block(313, 294)
                self.observe(current, [target], 1350)
                self.assertEqual(self.write_next(current, 1350), build_event_packet(0x14))
                self.command(current, 0x14, 1450)
                self.assertEqual(self.write_next(current, 1450), build_event_packet(0x24))
                self.assertEqual(current._final_target_lock, current._final_target_identity(target))
                crowded = [target, block(360, 294, 3), block(313, 248, 3)]
                self.assertIs(self.observe(current, crowded, 1451), target)
                self.assertFalse(current._final_roi_sent)
                self.assertEqual(self.write_next(current, 1451), serial_module.build_coordinate_packet(5, 0, -65))
                crowded[0] = block(313, 360)
                self.assertIs(self.observe(current, crowded, 1700), crowded[0])
                self.assertTrue(current._final_roi_sent)
                self.assertIsNone(current._latest_packet)
                self.assertIsNone(self.write_next(current, 1749))
                self.assertEqual(self.write_next(current, 1750), build_event_packet(0x34))

    def test_no_green_after44_uses_search_e3_instead_of_repeating_e4_or_arranging(self):
        current = self.make()
        current._track_green_recovery_waiting = True
        self.command(current)
        for index in range(config.UART_SEARCH_NO_TARGET_FRAMES):
            now = 1100 + index * 50
            self.observe(current, [], now)
            packet = self.write_next(current, now)
            self.assertEqual(packet, build_event_packet(0xE3) if index == 4 else None)
        self.assertEqual(current.mode, serial_module.CMD_SEARCH)
        self.assertIsNone(self.write_next(current, 2300))
        self.command(current, 0x13, 2400)
        self.observe(current, [], 2900)
        self.assertEqual(self.write_next(current, 2900), build_event_packet(0xE3))
        target = block(330, 250)
        for now in (2950, 3000, 3050):
            self.observe(current, [target], now)
        self.assertEqual(self.write_next(current, 3050), serial_module.build_coordinate_packet(5, 17, -89))

    def test_search_after44_still_excludes_roi_and_both_safety_zones(self):
        for zone_id in (1, 2):
            with self.subTest(zone_id=zone_id):
                current = self.make()
                self.command(current)
                zone = SimpleNamespace(x=100, y=100, w=80, h=80, class_id=zone_id, score=.9)
                objects = [block(140, 150), block(288, 400), block(330, 250, 4), zone]
                for now in (1100, 1150, 1200, 1250, 1300):
                    self.assertIsNone(self.observe(current, objects, now))
                    self.assertIsNone(current._latest_packet)
                self.assertIsNone(current._search_target_lock)
                self.assertEqual(self.write_next(current, 1300), build_event_packet(0xE3))

    def test_later_tasks44_still_judge_and_duplicate44_in03_does_not_reset_tracking(self):
        for label, post07 in (("sqareredgreen", True), ("sqarered", True),
                              ("triangualrblack", True), (serial_module.TARGET_BLACK_OR_GREEN, True),
                              ("sqareblue", False)):
            with self.subTest(label=label):
                current = self.make()
                current.target_label, current._post07_started = label, post07
                self.command(current)
                self.assertEqual(current.mode, MODE_ARRANGE_RIGHTMOST)
                self.assertTrue(current._initial_arrangement_decision_pending)
        current = self.make()
        self.command(current)
        target = block(330, 250)
        for now in (1100, 1150, 1200):
            self.observe(current, [target], now)
        generation, lock = current.frame_generation(), current._search_target_lock
        self.command(current, now=1250)
        self.assertEqual(current.mode, serial_module.CMD_SEARCH)
        self.assertEqual(current.frame_generation(), generation)
        self.assertEqual(current._search_target_lock, lock)

    def test_first_green_no_green05_history_survives44_to03_and_can_still_trigger_e5(self):
        current = self.make(after05=True)
        current._near_view_green_missing_cycles = 1
        self.command(current)
        self.assertEqual(current._near_view_green_missing_cycles, 1)
        self.command(current, 0x25, 2000)
        self.assertEqual(self.write_next(current, 2000), build_event_packet(0x05))
        for now in (2250, 2260, 2270):
            self.observe(current, [], now)
        self.assertEqual(current._pending_events, [build_event_packet(0xE5)])
        self.assertEqual(current.mode, CMD_TRACK)
        self.command(current, now=2300)
        self.assertEqual(current.mode, serial_module.CMD_SEARCH)
        self.assertTrue(current._first_green_near_view_decided)


class Arrange02CenterCoordinateTests(unittest.TestCase):
    """02 follows the original target to IMAGE center;12 stays open-loop."""

    def setUp(self):
        for helper in (patch("builtins.print"), patch.object(config, "VISION_DIAGNOSTICS_ENABLED", False)):
            helper.start()
            self.addCleanup(helper.stop)

    def make(self, class_id=5, target=None, sent=True, label=None):
        with patch.object(config, "UART_COORDINATE_ENABLED", False):
            current = VisionSerialController(576, 448, make_bottom_center_roi(576, 448))
        current.mode = MODE_WAIT_14_HANDSHAKE
        current.target_label = label or LABELS[class_id]
        current._post07_started = class_id != 5 or label == serial_module.TARGET_BLACK_OR_GREEN
        target = target or block(330, 250, class_id)
        current._track_target_lock = current._final_target_identity(target)
        current._initial_confirmation_anchor = current._track_target_lock
        current._serial = SimpleNamespace(write=Mock(side_effect=lambda data: len(data)))
        with current._lock:
            current._finish_initial_arrangement_decision(EVENT_ROI_READY, target, "test_crowded", 1000)
        if sent:
            self.assertEqual(self.write_next(current, 1000), build_event_packet(0x02))
        return current, target

    def command(self, current, command, now):
        with patch.object(serial_module, "_now_ms", return_value=now):
            current.process_received_bytes(serial_module.build_command_packet(command))

    def observe(self, current, raw=(), now=1050, verified=None, generation=None):
        if generation is None:
            generation = current.frame_generation()
        with patch.object(serial_module, "_now_ms", return_value=now):
            return current.submit(list(raw) if verified is None else list(verified), LABELS,
                                  raw_objects=list(raw), frame_generation=generation)

    def write_next(self, current, now):
        with patch.object(serial_module, "_now_ms", return_value=now):
            packet = current._next_outgoing_packet(now)
            if packet is not None:
                current._write_packet(packet, current._outgoing_generation)
            return packet

    def test02_sends_original_class_to_image_center_not_roi_or_side_target(self):
        for class_id in (3, 4, 5, 6):
            with self.subTest(class_id=class_id):
                current, target = self.make(class_id)
                nearer_same = block(288, 224, class_id)
                foreign = block(332, 250, 3 if class_id != 3 else 4)
                self.assertEqual(current._reference_point(MODE_ARRANGE_RIGHTMOST), (288., 224.))
                self.assertEqual(current._reference_point(serial_module.MODE_MICRO_SIDE), (288., 383.5))
                self.assertIs(self.observe(current, [nearer_same, foreign, target], verified=[]), target)
                self.assertEqual(self.write_next(current, 1050), serial_module.build_coordinate_packet(class_id, 42, 26))
                self.assertIsNone(self.write_next(current, 1051))
                self.assertEqual(current.mode, MODE_ARRANGE_RIGHTMOST)
                self.assertEqual(current._track_target_lock, current._final_target_identity(target))

    def test02_real14_crowding_seeds_approached_green_and_waits_for_actual_request_write(self):
        current, target = self.make(sent=False)
        self.command(current, 0x03, 1010)
        current.mode = MODE_WAIT_14_HANDSHAKE
        target = block(288, 224)
        current._track_target_lock = current._final_target_identity(target)
        with patch.object(config, "UART_DIRECT_CAPTURE_AFTER_14", False):
            self.command(current, 0x14, 1100)
        crowded = [target, block(335, 224, 3), block(288, 270, 3)]
        for now in (1150, 1200, 1250):
            self.observe(current, crowded, now)
        self.assertEqual(current._pending_events, [build_event_packet(0x02)])
        self.assertFalse(current._arrangement_02_request_sent)
        self.assertEqual(current._track_target_lock, current._final_target_identity(target))
        self.assertIsNone(self.observe(current, crowded, 1270))
        self.assertIsNone(current._latest_packet)
        self.assertEqual(self.write_next(current, 1300), build_event_packet(0x02))
        self.assertIs(self.observe(current, crowded, 1350, verified=[]), target)
        self.assertEqual(self.write_next(current, 1350), serial_module.build_coordinate_packet(5, 0, 0))
        self.assertEqual(current.mode, MODE_ARRANGE_RIGHTMOST)
        self.assertEqual(current._pending_events, [])

    def test02_short_writes_retry_once_without_coordinates_or_accepting_old_frame(self):
        for sent in (-1, 0, 2, 4):
            with self.subTest(sent=sent):
                current, target = self.make(sent=False)
                generation = current.frame_generation()
                packet = current._next_outgoing_packet(1100)
                current._serial.write.side_effect = None
                current._serial.write.return_value = sent
                with patch.object(serial_module, "_now_ms", return_value=1100):
                    current._write_packet(packet, current._outgoing_generation)
                    current._write_packet(packet, current._outgoing_generation)
                self.assertFalse(current._arrangement_02_request_sent)
                self.assertEqual(current._pending_events, [build_event_packet(0x02)])
                self.assertIsNone(self.observe(current, [target], 1200))
                self.assertIsNone(current._latest_packet)
                current._serial.write.return_value = 5
                self.assertEqual(self.write_next(current, 1300), build_event_packet(0x02))
                self.assertEqual(current.frame_generation(), generation + 1)
                self.assertIsNone(self.observe(current, [target], 1350, generation=generation))
                self.assertIs(self.observe(current, [target], 1400, verified=[]), target)
                writes = current._serial.write.call_count
                with patch.object(serial_module, "_now_ms", return_value=1401):
                    current._write_packet(packet)
                self.assertEqual(current._serial.write.call_count, writes)

    def test02_roi_entry_and_center_crossing_do_not_send34_14_or_change_reference(self):
        for class_id in (3, 4, 5, 6):
            with self.subTest(class_id=class_id):
                current, _ = self.make(class_id)
                inside = block(288, 400, class_id)
                self.assertTrue(current._target_center_in_tracking_roi(inside))
                self.assertIs(self.observe(current, [inside], verified=[]), inside)
                self.assertEqual(current._latest_packet, serial_module.build_coordinate_packet(class_id, 0, 176))
                self.assertEqual(current.mode, MODE_ARRANGE_RIGHTMOST)
                self.assertEqual(current._pending_events, [])
                self.assertFalse(current._final_roi_sent)

    def test02_raw_lock_ignores_spacing_jump_area_and_foreign_targets_but_never_predicts(self):
        current, target = self.make()
        changed = block(500, 250)
        changed.w, changed.h = 70, 50
        with patch.object(current, "_select_arrangement_neighbor", side_effect=AssertionError("02 must not select side")), \
                patch.object(current, "_select_final_roi_target", side_effect=AssertionError("02 lock must not re-screen")):
            self.assertIs(self.observe(current, [changed, block(330, 250, 4)], verified=[]), changed)
            locked = current._track_target_lock
            for index in range(8):
                self.assertIsNone(self.observe(current, [block(300, 250, 4)], 1100 + index * 50, verified=[]))
                self.assertIsNone(self.write_next(current, 1100 + index * 50))
                self.assertEqual(current._track_target_lock, locked)
            self.assertIs(self.observe(current, [changed], 1550, verified=[]), changed)
        self.assertEqual(current.mode, MODE_ARRANGE_RIGHTMOST)
        self.assertNotIn(build_event_packet(0xE4), current._pending_events)

    def test02_prepared_coordinates_are_canceled_by_miss_safety_zone_and_expiry(self):
        for zone_id in (1, 2):
            with self.subTest(zone_id=zone_id):
                current, target = self.make()
                self.observe(current, [target])
                packet = current._next_outgoing_packet(1050)
                zone = SimpleNamespace(x=320, y=240, w=30, h=30, class_id=zone_id, score=.9)
                self.assertIsNone(self.observe(current, [target, zone], 1100))
                writes = current._serial.write.call_count
                with patch.object(serial_module, "_now_ms", return_value=1101):
                    current._write_packet(packet)
                self.assertEqual(current._serial.write.call_count, writes)
                self.assertIs(self.observe(current, [target], 1700, verified=[]), target)
                packet = current._next_outgoing_packet(1700)
                with patch.object(serial_module, "_now_ms", return_value=1901):
                    current._write_packet(packet)
                self.assertEqual(current._serial.write.call_count, writes)

    def test05_to02_waits_for_actual02_plus_wide300_then_confirms_new_task_target(self):
        current, _ = self.make()
        self.command(current, 0x25, 2000)
        self.assertEqual(self.write_next(current, 2000), build_event_packet(0x05))
        near = [near_view_block(300, 200), near_view_block(350, 200, 3)]
        for now in (2250, 2260, 2270):
            self.observe(current, near, now)
        self.assertEqual(current.mode, MODE_ARRANGE_RIGHTMOST)
        self.assertIsNone(current._track_target_lock)
        packet = current._next_outgoing_packet(2300)
        self.assertEqual(packet, build_event_packet(0x02))
        with patch.object(serial_module, "_now_ms", return_value=2500):
            current._write_packet(packet, current._outgoing_generation)
        self.assertEqual(current._arrangement_ready_ms, 2800)
        target = block(330, 250)
        for now in (2600, 2700, 2799):
            self.assertIsNone(self.observe(current, [target], now))
        for now in (2800, 2850):
            self.assertIsNone(self.observe(current, [target, block(288, 224, 3)], now))
        self.assertIs(self.observe(current, [target, block(288, 224, 3)], 2900), target)
        self.assertEqual(current._latest_packet, serial_module.build_coordinate_packet(5, 42, 26))

    def test02_12_22_handshake_still_works_and12_never_writes_old02_coordinates(self):
        current, target = self.make()
        self.observe(current, [target])
        packet = current._next_outgoing_packet(1050)
        generation = current._outgoing_generation
        self.command(current, 0x12, 1100)
        self.assertEqual(current.mode, serial_module.MODE_ARRANGE_LEFTMOST)
        self.assertFalse(current._arrangement_02_request_sent)
        self.assertEqual(self.write_next(current, 1100), build_event_packet(0x12))
        writes = current._serial.write.call_count
        with patch.object(serial_module, "_now_ms", return_value=1101):
            current._write_packet(packet, generation)
            current._write_packet(packet)
        self.assertEqual(current._serial.write.call_count, writes)
        self.observe(current, [target, block(200, 250, 3)], 1150)
        self.assertIsNone(current._latest_packet)
        self.command(current, 0x22, 1200)
        self.assertEqual(current.mode, MODE_POST_ARRANGE_CHECK)

    def test02_supply51_can_follow_new_target_inside_loaded_roi_without_tracking_old_cargo(self):
        current, target = self.make(6, label=serial_module.TARGET_BLACK_OR_GREEN)
        current._supply_collect_started_ms, current._supply_collect_attempts = 100, 1
        carried = block(205, 420, 5)
        current._supply_loaded_roi_anchors = [current._final_target_identity(carried)]
        current._supply_roi_observed = True
        entered = block(288, 400, 6)
        self.assertIs(self.observe(current, [carried, entered], verified=[]), entered)
        self.assertEqual(current._latest_packet, serial_module.build_coordinate_packet(6, 0, 176))
        self.assertEqual(current._pending_events, [])
        self.assertEqual(current._supply_collect_started_ms, 100)
        deadline = 100 + config.UART_SUPPLY_COLLECT_TIMEOUT_MS
        self.assertEqual(self.write_next(current, deadline), build_event_packet(0x05))
        self.assertEqual(current._zone_load_recheck_source, "before06")


class NearViewRedAreaTests(SingleFrameTrackingScenario):
    def setUp(self):
        for setting in (patch.object(config, "VISION_DIAGNOSTICS_ENABLED", False),
                        patch("builtins.print")):
            setting.start()
            self.addCleanup(setting.stop)

    @staticmethod
    def red(ratio=.25):
        return SimpleNamespace(x=0, y=100, w=576, h=448 * ratio,
                               class_id=4, score=.9)

    def begin(self, target="sqarered", phase="ordinary", post07=True, now=1000):
        current = controller(MODE_FINAL_ROI_TRACK)
        current.target_label = target
        current._post07_started = post07
        with patch.object(serial_module, "_now_ms", return_value=now):
            if phase == "ordinary":
                current._handle_command(serial_module.CMD_SEARCH_BLUE_ZONE)
            elif phase == "before06":
                current._begin_zone_load_recheck("before06")
            else:
                current.mode = serial_module.MODE_SEARCH_ZONE
                current._handle_command(0x05)
        packet = current._next_outgoing_packet(now)
        self.assertEqual(packet, build_event_packet(0x05))
        serial = SimpleNamespace(write=lambda data: len(data))
        with patch.object(current, "_serial", serial), \
                patch.object(serial_module, "_now_ms", return_value=now):
            current._write_packet(packet, current._outgoing_generation)
        return current

    def observe(self, current, objects, now, raw=None):
        with patch.object(serial_module, "_now_ms", return_value=now):
            return current.submit(objects, LABELS, raw_objects=objects if raw is None else raw,
                                  frame_generation=current.frame_generation())

    def confirm(self, current, objects, start=1500, raw=None):
        for offset in (0, 10, 20):
            self.observe(current, objects, start + offset, raw=raw)

    def test_config_and_exact15percent_boundary_use_full_frame_not_roi(self):
        self.assertEqual(config.UART_NEAR_VIEW_RED_MIN_AREA_RATIO, .15)
        current = controller(serial_module.MODE_NEAR_VIEW_CHECK)
        current.frame_width, current.frame_height = 100, 100
        for height, expected in ((14.999, []), (15, [4]), (15.001, [4])):
            with self.subTest(height=height):
                obj = SimpleNamespace(x=0, y=0, w=100, h=height, class_id=4)
                self.assertEqual(current._near_view_target_ids([obj], LABELS), expected)

    def test_missing_config_also_defaults15_and_custom_value_is_respected(self):
        current = controller(serial_module.MODE_NEAR_VIEW_CHECK)
        old_config = SimpleNamespace(**{name: value for name, value in vars(config).items()
                                        if name != "UART_NEAR_VIEW_RED_MIN_AREA_RATIO"})
        with patch.object(serial_module, "config", old_config):
            self.assertEqual(current._near_view_target_ids([self.red(.14)], LABELS), [])
            self.assertEqual(current._near_view_target_ids([self.red(.16)], LABELS), [4])
        with patch.object(config, "UART_NEAR_VIEW_RED_MIN_AREA_RATIO", .2):
            self.assertEqual(current._near_view_target_ids([self.red(.16)], LABELS), [])

    def test_area_applies_per_red_box_not_sum_and_other_cargo_uses_own_rules(self):
        current = controller(serial_module.MODE_NEAR_VIEW_CHECK)
        small_reds = [self.red(.1), self.red(.1)]
        others = [near_view_block(200, 200, class_id) for class_id in (3, 5, 6)]
        self.assertEqual(current._near_view_target_ids(small_reds + others, LABELS), [3, 5, 6])
        self.assertEqual(current._movable_target_ids(small_reds, LABELS), [4, 4])

    def test_clipping_invalid_boxes_and_fractional_geometry(self):
        current = controller(serial_module.MODE_NEAR_VIEW_CHECK)
        current.frame_width, current.frame_height = 100, 100
        for geometry, expected in (((-90, 0, 100, 100), []),  # visible10%, not100%
                                   ((-50, 0, 100, 100), [4]),
                                   ((0, 0, 29.99, 50), []),
                                   ((0, 0, 30, 50), [4]),
                                   ((0, 0, 0, 100), []), ((0, 0, -30, 100), []),
                                   ((100, 0, 100, 100), []),
                                   ((float("nan"), 0, 100, 100), [])):
            with self.subTest(geometry=geometry):
                obj = SimpleNamespace(**dict(zip(("x", "y", "w", "h"), geometry)), class_id=4)
                before = vars(obj).copy()
                self.assertEqual(current._near_view_target_ids([obj], LABELS), expected)
                self.assertEqual(vars(obj), before)

    def test_ordinary_final_and_evacuation05_ignore_lone_small_red_and_return24(self):
        for phase in ("ordinary", "before06", "evacuation"):
            with self.subTest(phase=phase):
                current = self.begin(phase=phase)
                self.confirm(current, [self.red(.14)])
                self.assertEqual(current.mode, MODE_FINAL_ROI_TRACK)
                self.assertEqual(current._pending_events, [build_event_packet(0x24)])
                self.assertIsNone(current._latest_packet)
                self.assertEqual(current._near_view_empty_cycles, int(phase == "ordinary"))

    def test_small_red_does_not_arrange_otherwise_eligible_green_black_or_joint_load(self):
        for target, post07, phase, expected in (
                ("sqareredgreen", False, "ordinary", 0x05),
                ("sqareredgreen", True, "ordinary", 0x05),
                ("triangualrblack", True, "ordinary", 0x05),
                (serial_module.TARGET_BLACK_OR_GREEN, True, "ordinary", 0x03),
                (serial_module.TARGET_BLACK_OR_GREEN, True, "before06", None)):
            with self.subTest(target=target, post07=post07, phase=phase):
                current = self.begin(target, phase, post07)
                cargo_id = 6 if target == "triangualrblack" else 5
                self.confirm(current, [near_view_block(250, 200, cargo_id), self.red(.14)])
                if expected is None:
                    self.assertEqual(current._pending_events, [
                        serial_module.build_coordinate_packet(5, 0, 0), build_event_packet(0x06)])
                else:
                    self.assertEqual(current._pending_events, [build_event_packet(expected)])

    def test_qualifying_red_plus_any_other_counted_cargo_still_arranges(self):
        for target, cargo_id in (("sqarered", 3), ("sqarered", 5),
                                 ("sqareredgreen", 5), (serial_module.TARGET_BLACK_OR_GREEN, 6)):
            with self.subTest(target=target, cargo_id=cargo_id):
                current = self.begin(target)
                self.confirm(current, [self.red(.16), near_view_block(250, 200, cargo_id)])
                self.assertEqual(current._pending_events, [build_event_packet(0x02)])

    def test_final05_wide_red_waits_for_completed45_before_publishing_load_and06(self):
        current = self.begin()
        objects = [self.red(.14), self.red(.25)]
        self.confirm(current, objects)
        self.assertEqual(current._pending_events, [build_event_packet(0x05)])
        # The mandatory fresh final check must also ignore the small red.
        packet = current._next_outgoing_packet(1600)
        self.assertEqual(packet, build_event_packet(0x05))
        with patch.object(current, "_serial", SimpleNamespace(write=lambda data: len(data))), \
                patch.object(serial_module, "_now_ms", return_value=1600):
            current._write_packet(packet, current._outgoing_generation)
        self.confirm(current, objects, start=2100)
        request = build_event_packet(0x45)
        ack = serial_module.build_command_packet(0x45)
        report = serial_module.build_coordinate_packet(4, 0, 0)
        self.assertEqual(current._pending_events, [request])
        serial = SimpleNamespace(write=Mock(side_effect=[2, 5, 11, 5]))
        with patch.object(current, "_serial", serial):
            for now in (2120, 2121):
                with patch.object(serial_module, "_now_ms", return_value=now):
                    current.process_received_bytes(ack)
                    self.assertEqual(current._next_outgoing_packet(now), request)
                    current._write_packet(request, current._outgoing_generation)
            self.confirm(current, [], start=2200)
            self.assertEqual(current.mode, serial_module.MODE_NEAR_VIEW_CHECK)
            self.assertIsNone(current._next_outgoing_packet(2300))
            with patch.object(serial_module, "_now_ms", return_value=2500):
                current.process_received_bytes(ack + ack)
                self.assertEqual(current._next_outgoing_packet(2500), report)
                current._write_packet(report, current._outgoing_generation)
            self.assertIsNone(current._next_outgoing_packet(2549))
            result = current._next_outgoing_packet(2550)
            self.assertEqual(result, build_event_packet(0x06))
            with patch.object(serial_module, "_now_ms", return_value=2550):
                current._write_packet(result, current._outgoing_generation)
        self.assertEqual([call.args[0] for call in serial.write.call_args_list],
                         [request, request, report, result])

    def test_final05_wide_casualty_threshold_is_strict(self):
        for width in (299, 300, 301):
            with self.subTest(width=width):
                current = self.begin(phase="before06")
                box = SimpleNamespace(x=100, y=100, w=width, h=200, class_id=4, score=.9)
                self.confirm(current, [box])
                expected = ([build_event_packet(0x45)] if width > 300 else
                            [serial_module.build_coordinate_packet(4, 0, 0), build_event_packet(0x06)])
                self.assertEqual(current._pending_events, expected)

    def test_final_cannot_use_old_large_red_when_only_small_red_remains(self):
        current = self.begin()
        self.confirm(current, [self.red(.25)])
        packet = current._next_outgoing_packet(1600)
        self.assertEqual(packet, build_event_packet(0x05))
        with patch.object(current, "_serial", SimpleNamespace(write=lambda data: len(data))), \
                patch.object(serial_module, "_now_ms", return_value=1600):
            current._write_packet(packet, current._outgoing_generation)
        self.confirm(current, [self.red(.14)], start=2100)
        self.assertEqual(current._pending_events, [build_event_packet(0x24)])

    def test_small_red_empty05_counts_toward_first_green_second_no_green_e5(self):
        current = self.begin("sqareredgreen", post07=False)
        self.confirm(current, [self.red(.14)])
        self.assertEqual(current._next_outgoing_packet(1520), build_event_packet(0x24))
        self.assertEqual(current._near_view_green_missing_cycles, 1)
        with patch.object(serial_module, "_now_ms", return_value=2000):
            current._handle_command(serial_module.CMD_SEARCH_BLUE_ZONE)
        self.assertEqual(write_near_view_request(current, 2000), build_event_packet(0x05))
        self.confirm(current, [self.red(.14)], start=2500)
        self.assertEqual(current._pending_events, [build_event_packet(0xE5)])
        self.assertTrue(current._track_green_recovery_waiting)

    def test_filtering_requires250ms_settle_and_three_same_action_frames(self):
        current = self.begin()
        for now in (1100, 1200, 1249):
            self.observe(current, [self.red(.14)], now)
        self.assertEqual(current._near_view_stable_frames, 0)
        for now, ratio in ((1250, .25), (1260, .25), (1270, .14),
                            (1280, .25), (1290, .25)):
            self.observe(current, [self.red(ratio)], now)
            self.assertEqual(current._pending_events, [])
        self.observe(current, [self.red(.25)], 1300)
        self.assertEqual(current._pending_events, [build_event_packet(0x05)])

    def test_raw_unverified_large_red_is_not_counted(self):
        current = self.begin()
        self.confirm(current, [], raw=[self.red(.25)])
        self.assertEqual(current._pending_events, [build_event_packet(0x24)])

    def test_small_red_remains_trackable_outside05(self):
        for mode in (serial_module.CMD_SEARCH, CMD_TRACK, MODE_FINAL_ROI_TRACK):
            with self.subTest(mode=mode):
                current = controller(mode)
                current.target_label = "sqarered"
                current._post07_started = True
                target = block(100, 150, 4)
                self.assertIs(self.observe(current, [target], 1000), target)
                self.assertEqual(current._latest_packet[2], 4)

    def test_verbose_near_detail_ids_match_filtered_count_without_default_extra_logs(self):
        current = self.begin()
        with patch.object(config, "VISION_DIAGNOSTICS_ENABLED", True), \
                patch.object(config, "VISION_DIAGNOSTICS_VERBOSE", True), \
                patch("builtins.print") as output:
            self.observe(current, [self.red(.14)], 1500)
        lines = [str(call) for call in output.call_args_list if "[NEAR-DETAIL]" in str(call)]
        self.assertEqual(len(lines), 1)
        self.assertIn("ids=[]", lines[0])


class NearViewSettleTimingTests(unittest.TestCase):
    """Real250ms default applies to every05 inventory; three frames remain."""

    setUp = NearViewRedAreaTests.setUp
    begin = NearViewRedAreaTests.begin
    observe = NearViewRedAreaTests.observe

    def test_default_settle250_and_three_frame_confirmation(self):
        self.assertEqual(config.UART_NEAR_VIEW_SETTLE_MS, 250)
        self.assertEqual(config.UART_NEAR_VIEW_STABLE_FRAMES, 3)
        self.assertEqual(config.UART_WIDE_VIEW_SETTLE_MS, 300)

    def test_all05_inventory_entries_wait250_then_three_fresh_frames(self):
        for phase in ("ordinary", "before06", "evacuation"):
            with self.subTest(phase=phase):
                current = self.begin(phase=phase)
                for now in (1100, 1200, 1249):
                    self.observe(current, [], now)
                    self.assertEqual(current._near_view_stable_frames, 0)
                    self.assertEqual(current._pending_events, [])
                for now, count in ((1250, 1), (1260, 2)):
                    self.observe(current, [], now)
                    self.assertEqual(current._near_view_stable_frames, count)
                    self.assertEqual(current.mode, serial_module.MODE_NEAR_VIEW_CHECK)
                    self.assertEqual(current._pending_events, [])
                self.observe(current, [], 1270)
                self.assertEqual(current._pending_events, [build_event_packet(0x24)])

    def test_recheck_settling_starts_at_successful05_write_not_queue_or_short_write(self):
        for phase in ("before06", "evacuation"):
            with self.subTest(phase=phase):
                current = controller(serial_module.MODE_SEARCH_ZONE)
                with patch.object(serial_module, "_now_ms", return_value=1000):
                    current._begin_zone_load_recheck(phase)
                old_generation = current.frame_generation()
                packet = current._next_outgoing_packet(1100)
                self.assertEqual(packet, build_event_packet(0x05))
                serial = SimpleNamespace(write=Mock(return_value=2))
                with patch.object(current, "_serial", serial), \
                        patch.object(serial_module, "_now_ms", return_value=1100):
                    current._write_packet(packet, current._outgoing_generation)
                self.observe(current, [], 1500)
                self.assertEqual(current._zone_load_recheck, "wait05tx")
                self.assertEqual(current._near_view_stable_frames, 0)
                packet = current._next_outgoing_packet(2000)
                self.assertEqual(packet, build_event_packet(0x05))
                serial.write.return_value = len(packet)
                with patch.object(current, "_serial", serial), \
                        patch.object(serial_module, "_now_ms", return_value=2000):
                    current._write_packet(packet, current._outgoing_generation)
                self.assertEqual(current._near_view_started_ms, 2000)
                self.observe(current, [], 2249)
                with patch.object(serial_module, "_now_ms", return_value=2250):
                    current.submit([], LABELS, raw_objects=[], frame_generation=old_generation)
                self.assertEqual(current._near_view_stable_frames, 0)
                for now in (2250, 2260):
                    self.observe(current, [], now)
                    self.assertEqual(current._pending_events, [])
                self.observe(current, [], 2270)
                self.assertEqual(current._pending_events, [build_event_packet(0x24)])

    def test_ordinary05_all_tasks_and_zone_commands_wait_for_actual_write_then250(self):
        tasks = (("sqareredgreen", False, [5]), ("sqarered", True, [4]),
                 ("triangualrblack", True, [6]),
                 (serial_module.TARGET_BLACK_OR_GREEN, True, [5, 6]))
        for command in (serial_module.CMD_SEARCH_RED_ZONE, serial_module.CMD_SEARCH_BLUE_ZONE):
            for target, post07, ids in tasks:
                with self.subTest(command=command, target=target):
                    current = controller(MODE_FINAL_ROI_TRACK)
                    current.target_label, current._post07_started = target, post07
                    with patch.object(serial_module, "_now_ms", return_value=1000):
                        current._handle_command(command)
                    objects = [near_view_block(120 + i * 100, 200, k)
                               for i, k in enumerate(ids)]
                    for now in (1250, 1260, 1270):
                        self.observe(current, objects, now)
                    self.assertIsNone(current._near_view_started_ms)
                    self.assertEqual(current._near_view_stable_frames, 0)
                    packet = current._next_outgoing_packet(1500)
                    self.assertEqual(packet, build_event_packet(0x05))
                    for now in (1750, 1760, 1770):
                        self.observe(current, objects, now)
                    self.assertEqual(current._near_view_stable_frames, 0)
                    generation = current.frame_generation()
                    serial = SimpleNamespace(write=Mock(return_value=len(packet)))
                    with patch.object(current, "_serial", serial), \
                            patch.object(serial_module, "_now_ms", return_value=2000):
                        current._write_packet(packet, current._outgoing_generation)
                    self.assertEqual(current._near_view_started_ms, 2000)
                    self.assertEqual(current.frame_generation(), generation + 1)
                    self.observe(current, objects, 2249)
                    with patch.object(serial_module, "_now_ms", return_value=2250):
                        current.submit(objects, LABELS, raw_objects=objects, frame_generation=generation)
                    self.assertEqual(current._near_view_stable_frames, 0)
                    for now, count in ((2250, 1), (2260, 2)):
                        self.observe(current, objects, now)
                        self.assertEqual(current._near_view_stable_frames, count)
                        self.assertEqual(current._pending_events, [])
                    self.observe(current, objects, 2270)
                    self.assertEqual(current._zone_load_recheck_source, "before06")
                    self.assertEqual(current._pending_events, [build_event_packet(0x05)])

    def test_ordinary05_short_or_zero_writes_retry_without_starting_or_counting(self):
        for sent in (-1, 0, 2, 4):
            with self.subTest(sent=sent):
                current = controller(MODE_FINAL_ROI_TRACK)
                with patch.object(serial_module, "_now_ms", return_value=1000):
                    current._handle_command(serial_module.CMD_SEARCH_BLUE_ZONE)
                packet = current._next_outgoing_packet(1100)
                serial = SimpleNamespace(write=Mock(return_value=sent))
                with patch.object(current, "_serial", serial), \
                        patch.object(serial_module, "_now_ms", return_value=1100):
                    current._write_packet(packet, current._outgoing_generation)
                    # Even a duplicate failed attempt keeps at most one retry.
                    current._write_packet(packet, current._outgoing_generation)
                self.assertIsNone(current._near_view_started_ms)
                self.assertEqual(current._pending_events, [build_event_packet(0x05)])
                self.assertIsNone(current._last_event_write_ms)
                for now in (1500, 1600, 1700):
                    self.observe(current, [], now)
                self.assertEqual(current._near_view_stable_frames, 0)
                write_near_view_request(current, 2000)
                self.observe(current, [], 2249)
                self.assertEqual(current._near_view_stable_frames, 0)
                for now in (2250, 2260):
                    self.observe(current, [], now)
                    self.assertEqual(current._pending_events, [])
                self.observe(current, [], 2270)
                self.assertEqual(current._pending_events, [build_event_packet(0x24)])

    def test_ordinary05_waits_through_event_gap_and_accepts_success_at_tick_zero(self):
        current = controller(MODE_FINAL_ROI_TRACK)
        current._last_event_write_ms = 990
        with patch.object(serial_module, "_now_ms", return_value=1000):
            current._handle_command(serial_module.CMD_SEARCH_RED_ZONE)
        self.assertIsNone(current._next_outgoing_packet(1039))
        self.assertIsNone(current._near_view_started_ms)
        write_near_view_request(current, 1040)
        self.assertEqual(current._near_view_started_ms, 1040)
        self.observe(current, [], 1289)
        self.assertEqual(current._near_view_stable_frames, 0)
        self.observe(current, [], 1290)
        self.assertEqual(current._near_view_stable_frames, 1)
        zero = controller(MODE_FINAL_ROI_TRACK)
        with patch.object(serial_module, "_now_ms", return_value=0):
            zero._handle_command(serial_module.CMD_SEARCH_BLUE_ZONE)
        write_near_view_request(zero, 0)
        self.assertEqual(zero._near_view_started_ms, 0)
        self.observe(zero, [], 249)
        self.assertEqual(zero._near_view_stable_frames, 0)
        self.observe(zero, [], 250)
        self.assertEqual(zero._near_view_stable_frames, 1)

    def test_duplicate05_writes_preserve_started_time_streak_and_generation(self):
        for phase in ("ordinary", "before06", "evacuation"):
            with self.subTest(phase=phase):
                current = self.begin(phase=phase)
                self.observe(current, [], 1250)
                self.assertEqual(current._near_view_stable_frames, 1)
                generation = current.frame_generation()
                current.send_event(0x05)
                write_near_view_request(current, 1260)
                self.assertEqual(current._near_view_started_ms, 1000)
                self.assertEqual(current._near_view_stable_frames, 1)
                self.assertEqual(current.frame_generation(), generation)
                self.observe(current, [], 1270)
                self.assertEqual(current._pending_events, [])
                self.observe(current, [], 1280)
                self.assertEqual(current._pending_events, [build_event_packet(0x24)])

    def test_old_prepared05_cannot_start_new_task_or_later05_window(self):
        commands = (serial_module.CMD_RESET_TASK, serial_module.CMD_SELECT_RED_OBJECT,
                    serial_module.CMD_SEARCH, serial_module.CMD_SEARCH_RED_ZONE)
        for command in commands:
            with self.subTest(command=command):
                current = controller(MODE_FINAL_ROI_TRACK)
                with patch.object(serial_module, "_now_ms", return_value=1000):
                    current._handle_command(serial_module.CMD_SEARCH_BLUE_ZONE)
                old = current._next_outgoing_packet(1100)
                generation = current._outgoing_generation
                with patch.object(serial_module, "_now_ms", return_value=1200):
                    current._handle_command(command)
                serial = SimpleNamespace(write=Mock())
                with patch.object(current, "_serial", serial), \
                        patch.object(serial_module, "_now_ms", return_value=1300):
                    current._write_packet(old, generation)
                    if current.mode != serial_module.MODE_NEAR_VIEW_CHECK:
                        current._write_packet(old)
                serial.write.assert_not_called()
                self.assertIsNone(current._near_view_started_ms)
                self.assertEqual(current._near_view_stable_frames, 0)

    def test_verbose_waiting05_diagnostics_do_not_treat_unsent_request_as_elapsed(self):
        current = controller(MODE_FINAL_ROI_TRACK)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            current._handle_command(serial_module.CMD_SEARCH_BLUE_ZONE)
        with patch.object(config, "VISION_DIAGNOSTICS_ENABLED", True), \
                patch.object(config, "VISION_DIAGNOSTICS_VERBOSE", True), \
                patch.object(config, "VISION_DIAGNOSTICS_INTERVAL_MS", 0), \
                patch("builtins.print") as output:
            self.observe(current, [], 2000)
            write_near_view_request(current, 2100)
            self.observe(current, [], 2101)
        lines = [str(call) for call in output.call_args_list if "[NEAR-DETAIL]" in str(call)]
        self.assertEqual(len(lines), 2)
        self.assertIn("settle_remaining_ms=250 waiting05tx=True", lines[0])
        self.assertIn("settle_remaining_ms=249 waiting05tx=False", lines[1])


class NearViewBlackAreaTests(SingleFrameTrackingScenario):
    """Black3% gate is inventory-only, matching the visible DEBUG box ratio."""

    setUp = NearViewRedAreaTests.setUp
    begin = NearViewRedAreaTests.begin
    observe = NearViewRedAreaTests.observe
    confirm = NearViewRedAreaTests.confirm

    @staticmethod
    def black(ratio=.03):
        return SimpleNamespace(x=0, y=100, w=576, h=448 * ratio,
                               class_id=6, score=.9)

    def test_exact3percent_boundary_and_full_frame_denominator(self):
        self.assertEqual(config.UART_NEAR_VIEW_BLACK_MIN_AREA_RATIO, .03)
        current = controller(serial_module.MODE_NEAR_VIEW_CHECK)
        current.frame_width, current.frame_height = 100, 100
        for height, expected in ((2, []), (2.9999, []), (3, [6]), (3.0001, [6])):
            with self.subTest(height=height):
                obj = SimpleNamespace(x=0, y=0, w=100, h=height, class_id=6)
                self.assertEqual(current._near_view_target_ids([obj], LABELS), expected)

    def test_missing_configuration_defaults3percent_and_custom_limit(self):
        current = controller(serial_module.MODE_NEAR_VIEW_CHECK)
        old_config = SimpleNamespace(**{name: value for name, value in vars(config).items()
                                        if name != "UART_NEAR_VIEW_BLACK_MIN_AREA_RATIO"})
        with patch.object(serial_module, "config", old_config):
            self.assertEqual(current._near_view_target_ids([self.black(.029)], LABELS), [])
            self.assertEqual(current._near_view_target_ids([self.black(.031)], LABELS), [6])
        with patch.object(config, "UART_NEAR_VIEW_BLACK_MIN_AREA_RATIO", .04):
            self.assertEqual(current._near_view_target_ids([self.black(.03)], LABELS), [])

    def test_each_black_box_filtered_individually_not_sum(self):
        current = controller(serial_module.MODE_NEAR_VIEW_CHECK)
        small = [self.black(.025), self.black(.025)]
        self.assertEqual(current._near_view_target_ids(small, LABELS), [])
        self.assertEqual(current._movable_target_ids(small, LABELS), [6, 6])
        self.assertEqual(current._near_view_target_ids(small + [self.black(.03)], LABELS), [6])

    def test_clipped_visible_area_fractional_invalid_and_offscreen_boxes(self):
        current = controller(serial_module.MODE_NEAR_VIEW_CHECK)
        current.frame_width, current.frame_height = 100, 100
        for geometry, expected in (((-99, 0, 100, 100), []),
                                   ((-98, 0, 100, 100), []), ((-97, 0, 100, 100), [6]),
                                   ((0, 0, 5.999, 50), []), ((0, 0, 6, 50), [6]),
                                   ((0, 0, 0, 100), []), ((0, 0, -20, 100), []),
                                   ((100, 0, 100, 100), []),
                                   ((float("nan"), 0, 100, 100), [])):
            with self.subTest(geometry=geometry):
                obj = SimpleNamespace(**dict(zip(("x", "y", "w", "h"), geometry)), class_id=6)
                before = vars(obj).copy()
                self.assertEqual(current._near_view_target_ids([obj], LABELS), expected)
                self.assertEqual(vars(obj), before)

    def test_green_blue_unchanged_and_red15percent_rule_is_preserved(self):
        current = controller(serial_module.MODE_NEAR_VIEW_CHECK)
        objects = [self.black(.029), NearViewRedAreaTests.red(.149),
                   block(200, 200, 3), block(250, 200, 5),
                   NearViewRedAreaTests.red(.15), self.black(.03)]
        self.assertEqual(current._near_view_target_ids(objects, LABELS), [3, 5, 4, 6])

    def test_ordinary_final_and_evacuation05_ignore_small_black_and_return24(self):
        for phase in ("ordinary", "before06", "evacuation"):
            with self.subTest(phase=phase):
                current = self.begin("triangualrblack", phase)
                self.confirm(current, [self.black(.029)])
                self.assertEqual(current.mode, MODE_FINAL_ROI_TRACK)
                self.assertEqual(current._pending_events, [build_event_packet(0x24)])
                self.assertIsNone(current._latest_packet)

    def test_exact3percent_black_is_accepted_and_final_reports_supply_id5(self):
        current = self.begin("triangualrblack")
        self.confirm(current, [self.black(.03)])
        self.assertEqual(current._pending_events, [build_event_packet(0x05)])
        packet = current._next_outgoing_packet(1600)
        with patch.object(current, "_serial", SimpleNamespace(write=lambda data: len(data))), \
                patch.object(serial_module, "_now_ms", return_value=1600):
            current._write_packet(packet, current._outgoing_generation)
        self.confirm(current, [self.black(.03)], start=2100)
        self.assertEqual(current._pending_events, [
            serial_module.build_coordinate_packet(5, 0, 0), build_event_packet(0x06)])

    def test_small_black_no_longer_forces_first_green_arrangement(self):
        current = self.begin("sqareredgreen", post07=False)
        self.confirm(current, [block(250, 200, 5), self.black(.029)])
        self.assertEqual(current._pending_events, [build_event_packet(0x05)])

    def test_small_black_no_longer_counts_as_foreign_cargo_for_red_task(self):
        current = self.begin("sqarered")
        self.confirm(current, [NearViewRedAreaTests.red(.16), self.black(.029)])
        self.assertEqual(current._pending_events, [build_event_packet(0x05)])

    def test_settle_time_and_three_stable_actions_are_preserved_after_area_filter(self):
        current = self.begin("triangualrblack")
        for now in (1100, 1200, 1249):
            self.observe(current, [self.black(.03)], now)
        self.assertEqual(current._near_view_stable_frames, 0)
        for now, ratio in ((1250, .03), (1260, .03), (1270, .029),
                           (1280, .03), (1290, .03)):
            self.observe(current, [self.black(ratio)], now)
            self.assertEqual(current._pending_events, [])
        self.observe(current, [self.black(.03)], 1300)
        self.assertEqual(current._pending_events, [build_event_packet(0x05)])

    def test_joint05_single_real_supply_plus_small_black_still_requests_supplement(self):
        current = self.begin(serial_module.TARGET_BLACK_OR_GREEN)
        self.confirm(current, [block(250, 200, 5), self.black(.029)])
        self.assertEqual(current.mode, serial_module.MODE_WAIT_SUPPLY_SEARCH)
        self.assertEqual(current._pending_events, [build_event_packet(0x03)])

    def test_final_inventory_cannot_reuse_old_large_black(self):
        current = self.begin("triangualrblack")
        self.confirm(current, [self.black(.03)])
        packet = current._next_outgoing_packet(1600)
        self.assertEqual(packet, build_event_packet(0x05))
        with patch.object(current, "_serial", SimpleNamespace(write=lambda data: len(data))), \
                patch.object(serial_module, "_now_ms", return_value=1600):
            current._write_packet(packet, current._outgoing_generation)
        self.confirm(current, [self.black(.029)], start=2100)
        self.assertEqual(current._pending_events, [build_event_packet(0x24)])

    def test_small_black_no_green05_still_counts_toward_first_green_e5(self):
        current = self.begin("sqareredgreen", post07=False)
        self.confirm(current, [self.black(.029)])
        self.assertEqual(current._next_outgoing_packet(1520), build_event_packet(0x24))
        self.assertEqual(current._near_view_green_missing_cycles, 1)
        with patch.object(serial_module, "_now_ms", return_value=2000):
            current._handle_command(serial_module.CMD_SEARCH_BLUE_ZONE)
        self.assertEqual(write_near_view_request(current, 2000), build_event_packet(0x05))
        self.confirm(current, [self.black(.029)], start=2500)
        self.assertEqual(current._pending_events, [build_event_packet(0xE5)])

    def test_raw_unverified_black_is_not_inventory_evidence(self):
        current = self.begin("triangualrblack")
        self.confirm(current, [], raw=[self.black(.03)])
        self.assertEqual(current._pending_events, [build_event_packet(0x24)])

    def test_small_black_remains_trackable_in03_04_24(self):
        for mode in (serial_module.CMD_SEARCH, CMD_TRACK, MODE_FINAL_ROI_TRACK):
            with self.subTest(mode=mode):
                current = controller(mode)
                current.target_label = "triangualrblack"
                current._post07_started = True
                target = block(100, 150, 6)
                self.assertIs(self.observe(current, [target], 1000), target)
                self.assertEqual(current._latest_packet[2], 6)

    def test_filtered_inventory_ids_are_used_in_verbose_logs(self):
        current = self.begin("triangualrblack")
        with patch.object(config, "VISION_DIAGNOSTICS_ENABLED", True), \
                patch.object(config, "VISION_DIAGNOSTICS_VERBOSE", True), \
                patch("builtins.print") as output:
            self.observe(current, [self.black(.029)], 1500)
        lines = [str(call) for call in output.call_args_list if "[NEAR-DETAIL]" in str(call)]
        self.assertEqual(len(lines), 1)
        self.assertIn("ids=[]", lines[0])


class WideViewSettle300Tests(unittest.TestCase):
    """Real default300ms after successful05-result06/24, not custom800."""

    setUp = NearViewRedAreaTests.setUp
    begin = NearViewRedAreaTests.begin
    observe = NearViewRedAreaTests.observe

    def test06_and24_results_start300ms_only_on_success_then_require_fresh_generation(self):
        self.assertEqual(config.UART_WIDE_VIEW_SETTLE_MS, 300)
        for accepted in (False, True):
            for retry in (False, True):
                with self.subTest(accepted=accepted, retry=retry):
                    current = self.begin(phase="before06")
                    objects = [near_view_block(288, 224, 4)] if accepted else []
                    for now in (1250, 1260, 1270):
                        self.observe(current, objects, now)
                    serial = SimpleNamespace(write=Mock(side_effect=lambda data: len(data)))
                    if accepted:
                        report = current._next_outgoing_packet(1300)
                        self.assertEqual(report, serial_module.build_coordinate_packet(4, 0, 0))
                        with patch.object(current, "_serial", serial), \
                                patch.object(serial_module, "_now_ms", return_value=1300):
                            current._write_packet(report, current._outgoing_generation)
                    result = current._next_outgoing_packet(1350)
                    self.assertEqual(result, build_event_packet(0x06 if accepted else 0x24))
                    if retry:
                        serial.write.side_effect = None
                        serial.write.return_value = 2
                        with patch.object(current, "_serial", serial), \
                                patch.object(serial_module, "_now_ms", return_value=1350):
                            current._write_packet(result, current._outgoing_generation)
                        self.assertEqual(current._zone_load_resume_ready_ms, 0)
                        self.assertIn(result, current._pending_events)
                        result = current._next_outgoing_packet(2000)
                        serial.write.return_value = 5
                    sent_at = 2000 if retry else 1350
                    with patch.object(current, "_serial", serial), \
                            patch.object(serial_module, "_now_ms", return_value=sent_at):
                        current._write_packet(result, current._outgoing_generation)
                    deadline = sent_at + 300
                    self.assertEqual(current._zone_load_recheck, "wide_settle")
                    self.assertEqual(current._zone_load_resume_ready_ms, deadline)
                    generation = current.frame_generation()
                    self.observe(current, objects, deadline - 1)
                    self.assertEqual(current._zone_load_recheck, "wide_settle")
                    self.assertIsNone(current._latest_packet)
                    self.assertIsNone(current._next_outgoing_packet(deadline - 1))
                    current._next_outgoing_packet(deadline)
                    self.assertIsNone(current._zone_load_recheck)
                    self.assertEqual(current.frame_generation(), generation + 1)
                    with patch.object(serial_module, "_now_ms", return_value=deadline + 1):
                        self.assertIsNone(current.submit(objects, LABELS, raw_objects=objects,
                                                         frame_generation=generation))
                    self.assertIsNone(current._latest_packet)


class ZoneLoadRecheckTests(SingleFrameTrackingScenario):
    """RX05 after06 corner evacuation uses an isolated near-view round."""

    def setUp(self):
        for setting in (patch.object(config, "VISION_DIAGNOSTICS_ENABLED", False),
                        patch.object(config, "UART_NEAR_VIEW_SETTLE_MS", 500),
                        patch.object(config, "UART_NEAR_VIEW_STABLE_FRAMES", 3),
                        patch.object(config, "UART_WIDE_VIEW_SETTLE_MS", 800),
                        patch("builtins.print")):
            setting.start()
            self.addCleanup(setting.stop)

    def zone_controller(self, target="sqareredgreen", post07=False, zone="bluesafety"):
        current = controller(serial_module.MODE_SEARCH_ZONE)
        current.target_label = target
        current._post07_started = post07
        current.zone_label = zone
        current._zone_found_sent = True
        current._zone_close_sent = True
        current._zone_close_active = True
        current._last_zone_close_event_ms = 900
        return current

    def command(self, current, command=0x05, now=1000):
        with patch.object(serial_module, "_now_ms", return_value=now):
            current.process_received_bytes(serial_module.build_command_packet(command))

    def observe(self, current, ids=(), now=1600, generation=None, raw=None):
        objects = [near_view_block(100 + index * 50, 400, class_id)
                   for index, class_id in enumerate(ids)]
        if generation is None:
            generation = current.frame_generation()
        with patch.object(serial_module, "_now_ms", return_value=now):
            return current.submit(objects, LABELS, raw_objects=objects if raw is None else raw,
                                  frame_generation=generation)

    def write(self, current, now, expected, sent=None):
        packet = current._next_outgoing_packet(now)
        self.assertEqual(packet, expected)
        serial = Mock()
        serial.write.return_value = len(packet) if sent is None else sent
        current._serial = serial
        with patch.object(serial_module, "_now_ms", return_value=now):
            current._write_packet(packet, current._outgoing_generation)
        serial.write.assert_called_once_with(packet)
        return serial

    def begin(self, current, now=1000):
        self.command(current, now=now)
        self.write(current, now + 100, build_event_packet(serial_module.EVENT_TARGET_READY))
        self.assertEqual(current._zone_load_recheck, "count05")
        return now + 600

    def confirm(self, current, ids, now=1600):
        for offset in (0, 10, 20):
            self.observe(current, ids, now + offset)
        if current._zone_load_recheck_source == "before06" and current._zone_load_recheck == "wait05tx":
            confirm_final_load_check(current, ids, now + 20, settle_ms=0)

    def test_rx05_preserves_task_and_zone_but_cancels_old_outputs(self):
        current = self.zone_controller("sqarered", True, "redsafety")
        current._zone_obstacle_waiting = True
        current._zone_obstacle_frames = 2
        current._supply_collect_delivery_pending = True
        current._latest_packet = serial_module.build_coordinate_packet(2, 90, -40)
        current._pending_events = [build_event_packet(serial_module.EVENT_ZONE_CLOSE),
                                   build_event_packet(serial_module.EVENT_ZONE_OBSTACLE)]
        old_packet = current._next_outgoing_packet(1000)
        old_generation = current._outgoing_generation
        self.command(current)
        self.assertEqual(current.mode, serial_module.MODE_NEAR_VIEW_CHECK)
        self.assertEqual((current.target_label, current.zone_label), ("sqarered", "redsafety"))
        self.assertTrue(current._post07_started)
        self.assertEqual(current._pending_events, [build_event_packet(serial_module.EVENT_TARGET_READY)])
        self.assertIsNone(current._latest_packet)
        self.assertFalse(current._zone_close_sent)
        self.assertFalse(current._zone_obstacle_waiting)
        self.assertEqual(current._zone_obstacle_frames, 0)
        self.assertFalse(current._supply_collect_delivery_pending)
        current._serial = Mock()
        with patch.object(serial_module, "_now_ms", return_value=1000):
            current._write_packet(old_packet, old_generation)
        current._serial.write.assert_not_called()
        self.observe(current, [4], 1800, generation=old_generation)
        self.assertEqual(current._near_view_stable_frames, 0)

    def test_rx05_ignored_outside06_and_ordinary05_is_not_restarted(self):
        for mode in (serial_module.CMD_SEARCH, CMD_TRACK, MODE_FINAL_ROI_TRACK,
                     MODE_ARRANGE_RIGHTMOST, serial_module.MODE_ARRANGE_LEFTMOST,
                     serial_module.MODE_NEAR_VIEW_CHECK):
            with self.subTest(mode=mode):
                current = controller(mode)
                old_generation = current.frame_generation()
                self.command(current)
                self.assertEqual(current.mode, mode)
                self.assertEqual(current.frame_generation(), old_generation)
                self.assertEqual(current._pending_events, [])

    def test_count_requires_actual05_write_then500ms_and_three_action_frames(self):
        current = self.zone_controller()
        self.command(current)
        for now in (1600, 1610, 1620):
            self.observe(current, [5], now)
        self.assertEqual(current._near_view_stable_frames, 0)
        old_generation = current.frame_generation()
        self.write(current, 2000, build_event_packet(serial_module.EVENT_TARGET_READY))
        self.assertEqual(current._near_view_started_ms, 2000)
        self.observe(current, [5], 2500, generation=old_generation)
        self.observe(current, [5], 2499)
        self.assertEqual(current._near_view_stable_frames, 0)
        self.observe(current, [5], 2500)
        self.observe(current, [5], 2510)
        self.assertEqual(current.mode, serial_module.MODE_NEAR_VIEW_CHECK)
        self.observe(current, [5], 2520)
        self.assertEqual(current.mode, serial_module.MODE_NEAR_VIEW_CHECK)
        self.assertEqual(current._zone_load_recheck_source, "before06")
        confirm_final_load_check(current, [5], 2520)
        self.assertEqual(current.mode, serial_module.MODE_SEARCH_ZONE)
        self.assertEqual(current._zone_load_recheck, "wait06tx")

    def test_short05_write_retries_without_starting_camera_count(self):
        current = self.zone_controller()
        self.command(current)
        self.write(current, 1100, build_event_packet(serial_module.EVENT_TARGET_READY), sent=2)
        self.assertEqual(current._zone_load_recheck, "wait05tx")
        self.observe(current, [5], 1800)
        self.assertEqual(current._near_view_stable_frames, 0)
        self.write(current, 2000, build_event_packet(serial_module.EVENT_TARGET_READY))
        self.assertEqual(current._near_view_started_ms, 2000)

    def test_duplicate05_replies_but_does_not_restart_settling_or_streak(self):
        current = self.zone_controller()
        self.command(current)
        self.command(current, now=1010)
        self.assertEqual(len(current._pending_events), 1)
        self.write(current, 1100, build_event_packet(serial_module.EVENT_TARGET_READY))
        self.observe(current, [5], 1600)
        old_generation = current.frame_generation()
        self.command(current, now=1610)
        self.write(current, 1610, build_event_packet(serial_module.EVENT_TARGET_READY))
        self.assertEqual(current._near_view_started_ms, 1100)
        self.assertEqual(current._near_view_stable_frames, 1)
        self.assertEqual(current.frame_generation(), old_generation)
        self.observe(current, [5], 1620)
        self.observe(current, [5], 1630)
        self.assertEqual(current._zone_load_recheck, "wait05tx")
        self.assertEqual(current._zone_load_recheck_source, "before06")

    def test_single51_supply_resumes_delivery_without_collection_or_budget_timeout(self):
        for ids in ([5], [6], [5, 6], [5, 6, 5]):
            with self.subTest(ids=ids):
                current = self.zone_controller(serial_module.TARGET_BLACK_OR_GREEN, True)
                current._supply_collect_attempts = 2
                current._supply_collect_started_ms = 1
                self.begin(current)
                self.confirm(current, ids)
                self.assertEqual(current.mode, serial_module.MODE_SEARCH_ZONE)
                self.assertEqual(current._zone_load_recheck, "wait06tx")
                self.assertEqual(current._supply_collect_attempts, 0)
                self.assertIsNone(current._supply_collect_started_ms)
                self.assertEqual(current._pending_events, [
                    serial_module.build_coordinate_packet(5, 0, 0),
                    build_event_packet(serial_module.EVENT_ZONE_SEARCH_STARTED)])
                self.assertNotIn(build_event_packet(serial_module.EVENT_COLLECT_MORE_SUPPLIES),
                                 current._pending_events)

    def test_valid_final_load_clears_cycles_and_reports_correct_metadata(self):
        for target, post07, ids, cargo_id in (("sqareredgreen", False, [5], 5),
                                             ("sqarered", True, [4], 4),
                                             ("triangualrblack", True, [6], 5)):
            with self.subTest(target=target):
                current = self.zone_controller(target, post07)
                current._near_view_empty_cycles = 1
                current._near_view_green_missing_cycles = 1
                self.begin(current)
                self.confirm(current, ids)
                self.assertEqual(current._near_view_empty_cycles, 0)
                self.assertEqual(current._near_view_green_missing_cycles, 0)
                self.assertEqual(current._pending_events[0],
                                 serial_module.build_coordinate_packet(cargo_id, 0, 0))

    def test_metadata_gap_and_wide_settle_gate_fresh_zone_resume(self):
        current = self.zone_controller("sqarered", True, "redsafety")
        self.begin(current)
        self.confirm(current, [4])
        zone = SimpleNamespace(x=0, y=0, w=576, h=448, class_id=2, score=.8)
        self.observe(current, now=1700, raw=[zone])
        self.assertIsNone(current._latest_packet)
        self.write(current, 1700, serial_module.build_coordinate_packet(4, 0, 0))
        self.assertIsNone(current._next_outgoing_packet(1749))
        self.write(current, 1750, build_event_packet(serial_module.EVENT_ZONE_SEARCH_STARTED))
        self.assertEqual(current._zone_load_resume_ready_ms, 2550)
        old_generation = current.frame_generation()
        self.observe(current, now=2549, raw=[zone])
        self.assertIsNone(current._latest_packet)
        self.assertFalse(current._zone_close_sent)
        self.assertIsNone(current._next_outgoing_packet(2549))
        self.observe(current, now=2550, generation=old_generation, raw=[zone])
        self.assertIsNone(current._zone_load_recheck)
        self.assertIsNone(current._latest_packet)
        self.observe(current, now=2551, raw=[zone])
        self.assertEqual(current._latest_packet[2], 2)
        self.assertEqual(current._next_outgoing_packet(2551),
                         build_event_packet(serial_module.EVENT_ZONE_FOUND))
        self.assertEqual(current._next_outgoing_packet(2601),
                         build_event_packet(serial_module.EVENT_ZONE_CLOSE))

    def test_late_duplicate05_cannot_cancel_delivery_result(self):
        current = self.zone_controller()
        self.begin(current)
        self.confirm(current, [5])
        result_packets = list(current._pending_events)
        generation = current.frame_generation()
        self.command(current, now=1700)
        self.assertEqual(current._pending_events, result_packets)
        self.assertEqual(current.frame_generation(), generation)
        self.assertEqual(current._zone_load_recheck, "wait06tx")

    def test_failed06_does_not_resume_zone_frames_and_can_retry(self):
        current = self.zone_controller()
        self.begin(current)
        self.confirm(current, [5])
        self.write(current, 1700, serial_module.build_coordinate_packet(5, 0, 0))
        self.write(current, 1750, build_event_packet(serial_module.EVENT_ZONE_SEARCH_STARTED), sent=-1)
        self.assertEqual(current._zone_load_recheck, "wait06tx")
        self.assertEqual(current._zone_load_resume_ready_ms, 0)
        self.observe(current, [5], now=9000)
        self.assertIsNone(current._latest_packet)
        self.write(current, 9001, build_event_packet(serial_module.EVENT_ZONE_SEARCH_STARTED))
        self.assertEqual(current._zone_load_resume_ready_ms, 9801)

    def test_empty_recheck_returns24_without_e5_or_empty_cycle_increment(self):
        current = self.zone_controller()
        current._near_view_empty_cycles = 1
        current._near_view_green_missing_cycles = 1
        self.begin(current)
        self.confirm(current, [])
        self.assertEqual(current.mode, MODE_FINAL_ROI_TRACK)
        self.assertEqual(current._zone_load_recheck, "wait24tx")
        self.assertEqual(current._pending_events, [build_event_packet(EVENT_SKIP_TO_FINAL_TRACK)])
        self.assertEqual(current._near_view_empty_cycles, 1)
        self.assertEqual(current._near_view_green_missing_cycles, 1)
        self.observe(current, [5], 1700)
        self.assertFalse(current._final_roi_sent)
        self.write(current, 1700, build_event_packet(EVENT_SKIP_TO_FINAL_TRACK))
        self.assertEqual(current._zone_load_resume_ready_ms, 2500)
        self.observe(current, [5], 2499)
        self.assertFalse(current._final_roi_sent)
        current._next_outgoing_packet(2500)
        self.observe(current, [5], 2501)
        self.assertTrue(current._final_roi_sent)
        self.assertEqual(current._next_outgoing_packet(2501),
                         build_event_packet(serial_module.EVENT_FINAL_TARGET_IN_ROI))

    def test_foreign_or_excess_load_uses_original_arrange_without_e5(self):
        for target, post07, ids in (("sqareredgreen", False, [5, 4]),
                                   ("sqareredgreen", False, [5, 5]),
                                   ("sqarered", True, [4, 5]),
                                   (serial_module.TARGET_BLACK_OR_GREEN, True, [5, 5, 6, 6]),
                                   (serial_module.TARGET_BLACK_OR_GREEN, True, [5, 4]),
                                   (serial_module.TARGET_BLACK_OR_GREEN, True, [5, 3])):
            with self.subTest(ids=ids, target=target):
                current = self.zone_controller(target, post07)
                current._near_view_green_missing_cycles = 1
                self.begin(current)
                self.confirm(current, ids)
                self.assertEqual(current.mode, MODE_ARRANGE_RIGHTMOST)
                self.assertIsNone(current._zone_load_recheck)
                self.assertEqual(current._pending_events, [build_event_packet(EVENT_ROI_READY)])
                self.assertEqual(current._near_view_green_missing_cycles, 1)
                self.command(current, serial_module.CMD_ARRANGE_FIRST_DONE, 3000)
                self.assertEqual(current.mode, serial_module.MODE_ARRANGE_LEFTMOST)

    def test_blue_only_returns24_for_same_task_not_delivery_or_arrange(self):
        current = self.zone_controller(serial_module.TARGET_BLACK_OR_GREEN, True)
        self.begin(current)
        self.confirm(current, [3, 3])
        self.assertEqual(current.mode, MODE_FINAL_ROI_TRACK)
        self.assertEqual(current.target_label, serial_module.TARGET_BLACK_OR_GREEN)
        self.assertEqual(current._pending_events, [build_event_packet(EVENT_SKIP_TO_FINAL_TRACK)])

    def test_wrong_color_only_recheck_returns24_for_current_cargo_task(self):
        for target, post07, ids in (("sqareredgreen", False, [4]),
                                   ("sqareredgreen", False, [6, 6]),
                                   ("sqarered", True, [5, 6]),
                                   (serial_module.TARGET_BLACK_OR_GREEN, True, [4])):
            with self.subTest(target=target, ids=ids):
                current = self.zone_controller(target, post07)
                self.begin(current)
                self.confirm(current, ids)
                self.assertEqual(current.mode, MODE_FINAL_ROI_TRACK)
                self.assertEqual(current.target_label, target)
                self.assertEqual(current._pending_events, [build_event_packet(EVENT_SKIP_TO_FINAL_TRACK)])

    def test_wrong_stage_commands_do_not_end_inventory_but_explicit_task_can_cancel(self):
        current = self.zone_controller(serial_module.TARGET_BLACK_OR_GREEN, True)
        self.begin(current)
        for command in (serial_module.CMD_SEARCH, CMD_TRACK, 0x14, 0x22, 0x24, 0x25, 0x36):
            self.command(current, command, 1200)
            self.assertEqual(current._zone_load_recheck, "count05")
            self.assertEqual(current.mode, serial_module.MODE_NEAR_VIEW_CHECK)
        self.command(current, serial_module.CMD_SELECT_RED_OBJECT, 1300)
        self.command(current, serial_module.CMD_SEARCH, 1310)
        self.assertIsNone(current._zone_load_recheck)
        self.assertEqual(current.mode, serial_module.CMD_SEARCH)
        self.assertEqual(current.target_label, "sqarered")
        self.assertEqual(current._pending_events, [])

    def test_count_uses_verified_full_image_not_raw_only_or_supply_roi_mask(self):
        current = self.zone_controller(serial_module.TARGET_BLACK_OR_GREEN, True)
        self.begin(current)
        # Helpers put every supply center inside the cargo-masked inner half.
        for offset in (0, 10):
            self.observe(current, [5, 6], 1600 + offset)
        self.assertEqual(current._near_view_stable_frames, 2)
        self.observe(current, [], 1620, raw=[block(100, 400, 5), block(150, 400, 6)])
        self.assertEqual(current._near_view_stable_frames, 1)
        self.assertEqual(current.mode, serial_module.MODE_NEAR_VIEW_CHECK)
        self.confirm(current, [5, 6], now=1630)
        self.assertEqual(current._zone_load_recheck, "wait06tx")


class BeforeZoneFinalInventoryTests(unittest.TestCase):
    """Every06 decision now needs its own fresh final05 inventory."""

    def setUp(self):
        for setting in (patch.object(config, "VISION_DIAGNOSTICS_ENABLED", False),
                        patch.object(config, "UART_NEAR_VIEW_SETTLE_MS", 500),
                        patch.object(config, "UART_NEAR_VIEW_STABLE_FRAMES", 3),
                        patch.object(config, "UART_WIDE_VIEW_SETTLE_MS", 800),
                        patch("builtins.print")):
            setting.start()
            self.addCleanup(setting.stop)

    def command(self, current, command, now):
        with patch.object(serial_module, "_now_ms", return_value=now):
            current.process_received_bytes(serial_module.build_command_packet(command))

    def observe(self, current, ids, now, generation=None):
        objects = [near_view_block(100 + index * 40, 200, class_id) for index, class_id in enumerate(ids)]
        if generation is None:
            generation = current.frame_generation()
        with patch.object(serial_module, "_now_ms", return_value=now):
            current.submit(objects, LABELS, raw_objects=objects, frame_generation=generation)

    def write_next(self, current, event, now, sent=None):
        packet = current._next_outgoing_packet(now)
        self.assertEqual(packet, build_event_packet(event))
        serial = SimpleNamespace(write=Mock(return_value=len(packet) if sent is None else sent))
        with patch.object(current, "_serial", serial), \
                patch.object(serial_module, "_now_ms", return_value=now):
            current._write_packet(packet, current._outgoing_generation)
        return packet

    def start_final(self, target="sqareredgreen", ids=(5,), zone_id=1):
        current = controller(MODE_FINAL_ROI_TRACK)
        current.target_label = target
        current._post07_started = target != "sqareredgreen"
        self.command(current, 0x25 if zone_id == 1 else 0x15, 1000)
        self.write_next(current, 0x05, 1000)
        for now in (1500, 1510, 1520):
            self.observe(current, ids, now)
        self.assertEqual(current.mode, serial_module.MODE_NEAR_VIEW_CHECK)
        self.assertEqual(current._zone_load_recheck_source, "before06")
        self.assertEqual(current._zone_load_recheck, "wait05tx")
        self.assertEqual(current._pending_events, [build_event_packet(0x05)])
        return current

    def test_normal_accepted_load_requests05_before_any_id_or06_for_all_tasks(self):
        for zone_id in (1, 2):
            for target, ids, cargo_id in (("sqareredgreen", [5], 5), ("sqarered", [4], 4),
                                           ("triangualrblack", [6], 5),
                                           (serial_module.TARGET_BLACK_OR_GREEN, [5, 6], 5)):
                with self.subTest(target=target, zone_id=zone_id):
                    current = self.start_final(target, ids, zone_id)
                    self.assertEqual(current.zone_label, LABELS[zone_id])
                    self.assertIsNone(current._latest_packet)
                    ready = confirm_final_load_check(current, ids, 1600)
                    self.assertEqual(current.mode, serial_module.MODE_SEARCH_ZONE)
                    self.assertEqual(current._pending_events, [
                        serial_module.build_coordinate_packet(cargo_id, 0, 0),
                        build_event_packet(0x06)])
                    self.observe(current, ids, ready + 100)
                    self.assertNotIn(build_event_packet(0x05), current._pending_events)
                    self.assertEqual(current._zone_load_recheck_source, "before06")

    def test_final_waits_for_successful05_write_and_three_fresh_frames_after500ms(self):
        current = self.start_final()
        generation = current.frame_generation()
        for now in (1900, 2200, 2500):
            self.observe(current, [5], now)
        self.assertEqual(current._near_view_stable_frames, 0)
        self.write_next(current, 0x05, 2600)
        self.observe(current, [5], 3100, generation)
        self.assertEqual(current._near_view_stable_frames, 0)
        self.observe(current, [5], 3099)
        self.assertEqual(current._near_view_stable_frames, 0)
        for now in (3100, 3110):
            self.observe(current, [5], now)
            self.assertEqual(current.mode, serial_module.MODE_NEAR_VIEW_CHECK)
        self.observe(current, [5], 3120)
        self.assertEqual(current.mode, serial_module.MODE_SEARCH_ZONE)
        self.assertEqual(current._zone_load_recheck, "wait06tx")

    def test_lost_or_wrong_color_final_load_returns24_without_stale_id_or_e5(self):
        for target, initial, missing in (("sqareredgreen", [5], []),
                                          ("sqareredgreen", [5], [4]),
                                          ("sqareredgreen", [5], [6]),
                                          ("sqarered", [4], [5, 6]),
                                          (serial_module.TARGET_BLACK_OR_GREEN, [5, 6], [3, 4])):
            with self.subTest(target=target, final_ids=missing):
                current = self.start_final(target, initial)
                current._near_view_green_missing_cycles = 1
                current._near_view_empty_cycles = 1
                confirm_final_load_check(current, missing, 1600)
                self.assertEqual(current.mode, MODE_FINAL_ROI_TRACK)
                self.assertEqual(current.target_label, target)
                self.assertEqual(current._pending_events, [build_event_packet(0x24)])
                self.assertEqual(current._near_view_green_missing_cycles, 1)
                self.assertEqual(current._near_view_empty_cycles, 1)

    def test_new_foreign_or_excess_final_load_arranges_using02_12_22(self):
        for target, initial, crowded in (("sqareredgreen", [5], [5, 5]),
                                          ("sqareredgreen", [5], [5, 6]),
                                          ("sqarered", [4], [4, 5]),
                                          ("sqarered", [4], [4, 4]),
                                          (serial_module.TARGET_BLACK_OR_GREEN, [5, 6], [5, 4]),
                                          (serial_module.TARGET_BLACK_OR_GREEN, [5, 6], [5, 3]),
                                          (serial_module.TARGET_BLACK_OR_GREEN, [5, 6], [5, 5, 6, 6])):
            with self.subTest(target=target, final_ids=crowded):
                current = self.start_final(target, initial)
                ready = confirm_final_load_check(current, crowded, 1600)
                self.assertEqual(current.mode, MODE_ARRANGE_RIGHTMOST)
                self.assertEqual(current._pending_events, [build_event_packet(0x02)])
                self.write_next(current, 0x02, ready)
                self.command(current, 0x12, ready + 1000)
                self.assertEqual(current.mode, serial_module.MODE_ARRANGE_LEFTMOST)
                self.write_next(current, 0x12, ready + 1000)
                self.command(current, 0x22, ready + 2000)
                self.assertEqual(current.mode, MODE_POST_ARRANGE_CHECK)

    def test_final_supply_count_one_to_three_does_not_restart_collection(self):
        for final_ids in ([5], [6], [5, 6], [5, 5, 6]):
            with self.subTest(final_ids=final_ids):
                current = self.start_final(serial_module.TARGET_BLACK_OR_GREEN, [5, 6])
                confirm_final_load_check(current, final_ids, 1600)
                self.assertEqual(current.mode, serial_module.MODE_SEARCH_ZONE)
                self.assertEqual(current._supply_collect_attempts, 0)
                self.assertIsNone(current._supply_collect_started_ms)
                self.assertNotIn(build_event_packet(0x03), current._pending_events)

    def test_timer_stops_collection_but_cannot_emit06_without_final_images(self):
        for mode in (serial_module.MODE_WAIT_SUPPLY_SEARCH, serial_module.CMD_SEARCH,
                     CMD_TRACK, MODE_FINAL_ROI_TRACK, MODE_ARRANGE_RIGHTMOST,
                     serial_module.MODE_ARRANGE_LEFTMOST, MODE_POST_ARRANGE_CHECK):
            with self.subTest(mode=mode):
                current = controller(mode)
                current.target_label = serial_module.TARGET_BLACK_OR_GREEN
                current._post07_started = True
                current._supply_collect_started_ms = 1000
                current._supply_collect_attempts = 2
                current._pending_events = [build_event_packet(0x34)]
                current._latest_packet = serial_module.build_coordinate_packet(6, 9, 9)
                self.write_next(current, 0x05, 16000)
                self.assertIsNone(current._latest_packet)
                self.assertIsNone(current._supply_collect_started_ms)
                self.assertEqual(current._zone_load_recheck_source, "before06")
                for now in (16500, 31000, 60000):
                    self.assertIsNone(current._next_outgoing_packet(now))
                    self.assertEqual(current.mode, serial_module.MODE_NEAR_VIEW_CHECK)
                    self.assertEqual(current._pending_events, [])

    def test_timeout_final_empty_cannot_deliver_old_cached_supply(self):
        current = controller(serial_module.CMD_SEARCH)
        current.target_label = serial_module.TARGET_BLACK_OR_GREEN
        current._post07_started = True
        current._supply_collect_started_ms = 1000
        current._supply_collect_attempts = 2
        current._latest_packet = serial_module.build_coordinate_packet(6, 10, 10)
        request = current._next_outgoing_packet(16000)
        self.assertEqual(request, build_event_packet(0x05))
        current._pending_events.insert(0, request)
        confirm_final_load_check(current, [], 16000)
        self.assertEqual(current.mode, MODE_FINAL_ROI_TRACK)
        self.assertEqual(current._pending_events, [build_event_packet(0x24)])
        self.assertIsNone(current._latest_packet)

    def test_final_confirmation_cannot_mix_clear_and_foreign_frames(self):
        current = self.start_final("sqarered", [4])
        self.write_next(current, 0x05, 1600)
        self.observe(current, [4], 2100)
        self.observe(current, [4, 5], 2110)
        for now in (2120, 2130):
            self.observe(current, [4], now)
            self.assertEqual(current.mode, serial_module.MODE_NEAR_VIEW_CHECK)
            self.assertEqual(current._pending_events, [])
        self.observe(current, [4], 2140)
        self.assertEqual(current._pending_events, [
            serial_module.build_coordinate_packet(4, 0, 0), build_event_packet(0x06)])

    def test_short_or_failed_final05_retries_and_never_uses_original_load(self):
        for sent in (-1, 2):
            with self.subTest(sent=sent):
                current = self.start_final()
                self.write_next(current, 0x05, 1600, sent)
                self.assertEqual(current._zone_load_recheck, "wait05tx")
                self.assertEqual(current._pending_events, [build_event_packet(0x05)])
                for now in (2200, 2300, 2400):
                    self.observe(current, [5], now)
                self.assertEqual(current._near_view_stable_frames, 0)
                confirm_final_load_check(current, [], 2500)
                self.assertEqual(current._pending_events, [build_event_packet(0x24)])

    def test_late_commands_and_duplicates_cannot_reset_final_check(self):
        current = self.start_final(serial_module.TARGET_BLACK_OR_GREEN, [5, 6])
        self.write_next(current, 0x05, 1600)
        self.observe(current, [5, 6], 2100)
        generation = current.frame_generation()
        for command in (0x51, 0x03, 0x04, 0x14, 0x22, 0x24, 0x25, 0x36, 0x13):
            self.command(current, command, 2101)
            self.assertEqual(current.frame_generation(), generation)
            self.assertEqual(current._near_view_stable_frames, 1)
        self.command(current, 0x05, 2102)
        self.write_next(current, 0x05, 2102)
        self.assertEqual(current._near_view_started_ms, 1600)
        self.assertEqual(current._near_view_stable_frames, 1)
        self.observe(current, [5, 6], 2110)
        self.observe(current, [5, 6], 2120)
        self.assertEqual(current._zone_load_recheck, "wait06tx")

    def test_explicit_new_task_cancels_final_and_rejects_prepared_old06(self):
        current = self.start_final()
        ready = confirm_final_load_check(current, [5], 1600)
        current._next_outgoing_packet(ready)  # Inspect metadata without UART.
        old06 = current._next_outgoing_packet(ready)
        generation = current._outgoing_generation
        self.command(current, 0x11, ready + 1)
        self.command(current, 0x03, ready + 2)
        serial = SimpleNamespace(write=Mock(return_value=5))
        with patch.object(current, "_serial", serial), \
                patch.object(serial_module, "_now_ms", return_value=ready + 3):
            current._write_packet(old06, generation)
        serial.write.assert_not_called()
        self.assertEqual(current.target_label, "sqarered")
        self.assertEqual(current.mode, serial_module.CMD_SEARCH)
        self.assertIsNone(current._zone_load_recheck_source)

    def test_unconfirmed06_is_rejected_at_uart_write_even_without_generation(self):
        current = controller(serial_module.MODE_SEARCH_ZONE)
        serial = SimpleNamespace(write=Mock(return_value=5))
        with patch.object(current, "_serial", serial), \
                patch.object(serial_module, "_now_ms", return_value=1000):
            current._write_packet(build_event_packet(0x06))
        serial.write.assert_not_called()


class LargestSafetyZoneTests(SingleFrameTrackingScenario):
    def setUp(self):
        for setting in (patch.object(config, "VISION_DIAGNOSTICS_ENABLED", False),
                        patch("builtins.print")):
            setting.start()
            self.addCleanup(setting.stop)

    @staticmethod
    def zone(x, y, w, h, class_id=1, score=.9):
        return SimpleNamespace(x=x, y=y, w=w, h=h, class_id=class_id, score=score)

    def submit(self, current, raw, now=1000, verified=()):
        with patch.object(serial_module, "_now_ms", return_value=now):
            return current.submit(list(verified), LABELS, raw_objects=list(raw),
                                  frame_generation=current.frame_generation())

    def test_largest_same_color_wins_over_center_confidence_and_detection_order(self):
        for class_id in (1, 2):
            for target_label in ("sqarered", "sqareredgreen", "triangualrblack"):
                for reverse in (False, True):
                    with self.subTest(class_id=class_id, task=target_label, reverse=reverse):
                        current = controller(serial_module.MODE_SEARCH_ZONE)
                        current.zone_label = LABELS[class_id]
                        current.target_label = target_label
                        small = self.zone(278, 200, 20, 20, class_id, .99)
                        large = self.zone(400, 50, 160, 200, class_id, .41)
                        raw = [small, large] if not reverse else [large, small]
                        self.assertIs(self.submit(current, raw), large)
                        reference_x, reference_y = current._reference_point(current.mode)
                        x_ratio = (config.UART_ZONE_CASUALTY_X_RATIO
                                   if target_label == config.UART_ZONE_CASUALTY_LABEL
                                   else config.UART_ZONE_SUPPLY_X_RATIO)
                        self.assertEqual(current._latest_packet,
                                         serial_module.build_coordinate_packet(
                                             class_id,
                                             int(round(large.x + large.w * x_ratio - reference_x)),
                                             int(round(large.y + large.h * .5 - reference_y))))
                        self.assertEqual(current._pending_events,
                                         [build_event_packet(serial_module.EVENT_ZONE_FOUND)])

    def test_wrong_color_even_when_larger_cannot_replace_commanded_zone(self):
        for class_id in (1, 2):
            current = controller(serial_module.MODE_SEARCH_ZONE)
            current.zone_label = LABELS[class_id]
            right = self.zone(400, 100, 100, 100, class_id)
            wrong = self.zone(0, 0, 576, 448, 3 - class_id)
            self.assertIs(self.submit(current, [wrong, right]), right)
            self.assertFalse(current._zone_close_active)
            self.assertNotIn(build_event_packet(serial_module.EVENT_ZONE_CLOSE), current._pending_events)
            self.assertIsNone(self.submit(current, [wrong], 1100))
            self.assertIsNone(current._latest_packet)

    def test_visible_clipped_area_not_outside_box_area_determines_largest(self):
        current = controller(serial_module.MODE_SEARCH_ZONE)
        outside = self.zone(-1000, 0, 1050, 448)
        visible = self.zone(400, 50, 80, 350)
        self.assertGreater(outside.w * outside.h, visible.w * visible.h)
        self.assertLess(current._visible_box_area_ratio(outside),
                        current._visible_box_area_ratio(visible))
        self.assertIs(self.submit(current, [outside, visible]), visible)

    def test_zero_visible_invalid_and_deprecated_boxes_are_not_zones(self):
        current = controller(serial_module.MODE_SEARCH_ZONE)
        raw = [self.zone(600, 10, 300, 300), self.zone(0, 500, 576, 100),
               self.zone(0, 0, 0, 100), self.zone(0, 0, -20, -20),
               self.zone(0, 0, 576, 448, 0), self.zone(0, 0, 576, 448, 5)]
        self.assertIsNone(self.submit(current, raw))
        self.assertIsNone(current._latest_packet)
        self.assertEqual(current._pending_events, [build_event_packet(serial_module.EVENT_ZONE_NO_TARGET)])

    def test_equal_area_ties_use_center_distance_then_confidence(self):
        current = controller(serial_module.MODE_SEARCH_ZONE)
        distant = self.zone(400, 100, 100, 100, score=.99)
        near = self.zone(238, 100, 100, 100, score=.4)
        self.assertIs(self.submit(current, [distant, near]), near)
        stronger = self.zone(238, 200, 100, 100, score=.8)
        self.assertIs(self.submit(current, [near, stronger], 1100), stronger)

    def test_26_uses_largest_zone_area_and_continues_its_coordinates(self):
        for class_id in (1, 2):
            current = controller(serial_module.MODE_SEARCH_ZONE)
            current.zone_label = LABELS[class_id]
            small = self.zone(278, 200, 20, 20, class_id)
            large = self.zone(0, 0, 576, 224, class_id, .4)
            self.assertIs(self.submit(current, [small, large]), large)
            self.assertTrue(current._zone_close_active)
            self.assertEqual(current._next_outgoing_packet(1000),
                             build_event_packet(serial_module.EVENT_ZONE_FOUND))
            self.assertEqual(current._next_outgoing_packet(1000),
                             build_event_packet(serial_module.EVENT_ZONE_CLOSE))
            self.assertEqual(current._next_outgoing_packet(1000), current._latest_packet)
            self.assertIs(self.submit(current, [small, large], 1500), large)
            self.assertEqual(current._next_outgoing_packet(1500),
                             build_event_packet(serial_module.EVENT_ZONE_CLOSE))
            self.assertEqual(current._next_outgoing_packet(1500), current._latest_packet)

    def test_every_frame_reselects_current_largest_without_old_lock_or_stale_coords(self):
        current = controller(serial_module.MODE_SEARCH_ZONE)
        first = self.zone(0, 0, 576, 224)
        second = self.zone(278, 200, 20, 20)
        self.assertIs(self.submit(current, [first, second]), first)
        self.assertTrue(current._zone_close_active)
        self.assertEqual(current._next_outgoing_packet(1000),
                         build_event_packet(serial_module.EVENT_ZONE_FOUND))
        first.w, first.h = 10, 10
        second.w, second.h = 200, 200
        self.assertIs(self.submit(current, [first, second], 1100), second)
        self.assertFalse(current._zone_close_active)
        self.assertNotIn(build_event_packet(serial_module.EVENT_ZONE_CLOSE), current._pending_events)
        self.assertIsNone(self.submit(current, [], 1200))
        self.assertIsNone(current._latest_packet)
        self.assertEqual(current._next_outgoing_packet(1200),
                         build_event_packet(serial_module.EVENT_ZONE_NO_TARGET))

    def test_new_eight_class_startzone_is_ignored_even_when_largest(self):
        from model_class_mapping import ModelClassMapper
        labels = LABELS[:-1] + ["startzone", "triangualrblack"]
        mapper = ModelClassMapper(labels)
        current = controller(serial_module.MODE_SEARCH_ZONE)
        start = self.zone(0, 0, 576, 448, 6)
        red = self.zone(0, 0, 576, 448, 2)
        small = self.zone(278, 200, 20, 20)
        large = self.zone(400, 50, 160, 200)
        raw = mapper.normalize([start, red, small, large])
        with patch.object(serial_module, "_now_ms", return_value=1000):
            self.assertIs(current.submit([], mapper.labels, raw_objects=raw), large)
        self.assertEqual(current._latest_packet[2], 1)


class LabColorDisabledTests(unittest.TestCase):
    """No LAB pixel work, without bypassing confidence/history/geometry."""

    def setUp(self):
        self.helpers = [
            patch.object(config, "VISION_DIAGNOSTICS_VERBOSE", False),
            patch.object(config, "VISION_DIAGNOSTICS_ENABLED", False),
        ]
        for helper in self.helpers:
            helper.start()
            self.addCleanup(helper.stop)
        self.img = SimpleNamespace(
            find_blobs=Mock(side_effect=AssertionError("LAB must stay off")),
            get_statistics=Mock(side_effect=AssertionError("no pixel statistics")),
        )
        self.verifier = DetectionVerifier(576, 448)

    def confirm(self, objects, **kwargs):
        for _ in range(config.VERIFY_REQUIRED_HITS - 1):
            self.assertEqual(self.verifier.process(self.img, objects, LABELS, **kwargs), [])
        return self.verifier.process(self.img, objects, LABELS, **kwargs)

    def test_default_off_keeps_exact_class_confidence_boundaries(self):
        from detection_confidence import DetectionConfidenceFilter
        self.assertTrue(config.VERIFY_ENABLED)
        self.assertFalse(config.VERIFY_COLOR_ENABLED)
        self.assertEqual((config.DETECT_CONFIDENCE, config.DETECT_BLACK_CONFIDENCE,
                          config.DETECT_SAFETY_ZONE_CONFIDENCE), (.5, .7, .4))
        gate = DetectionConfidenceFilter(config.DETECT_CONFIDENCE,
                                         config.DETECT_BLACK_CONFIDENCE,
                                         config.DETECT_SAFETY_ZONE_CONFIDENCE)
        targets, weak = [], []
        for class_id in range(1, 7):
            threshold = .4 if class_id in (1, 2) else .7 if class_id == 6 else .5
            target = block(60 * class_id, 150, class_id)
            target.score = threshold
            below = block(60 * class_id, 200, class_id)
            below.score = threshold - .0001
            targets.append(target)
            weak.append(below)
        filtered = gate.apply(targets + weak)
        self.assertEqual(filtered, targets)
        self.assertEqual(self.confirm(filtered), targets[2:])
        self.img.find_blobs.assert_not_called()
        self.img.get_statistics.assert_not_called()
        self.assertTrue(self.verifier._color_available)

    def test_debug_and_verbose_cannot_reenable_lab(self):
        for debug in (False, True):
            for verbose in (False, True):
                with self.subTest(debug=debug, verbose=verbose), \
                        patch.object(config, "VISION_DIAGNOSTICS_VERBOSE", verbose), \
                        patch.object(config, "VISION_DIAGNOSTICS_ENABLED", True), \
                        patch.object(config, "VERIFY_SKIP_REDUNDANT_COLOR_ON_HIGH_CONFIDENCE", False):
                    self.verifier = DetectionVerifier(576, 448)
                    target = block(200, 150)
                    target.score = .5
                    self.assertEqual(self.confirm([target], debug_enabled=debug), [target])
                    if verbose:
                        record = self.verifier.frame_diagnostics()[0]
                        self.assertEqual(record["reason"], "lab_disabled_yolo_evidence")
                        self.assertFalse(record["high_conf_bypass"])
                        self.assertTrue(record["confirmed"])
        self.img.find_blobs.assert_not_called()

    def test_valid_geometry_is_still_required_for_low_confidence_boxes(self):
        tiny, flat = block(150, 150), block(250, 150)
        tiny.w = tiny.h = 4
        flat.w, flat.h = 200, 10
        tiny.score = flat.score = .5
        for _ in range(config.VERIFY_REQUIRED_HITS_WITHOUT_COLOR + 1):
            self.assertEqual(self.verifier.process(self.img, [tiny, flat], LABELS), [])
        self.img.find_blobs.assert_not_called()

    def test_per_class_candidate_cap_is_preserved(self):
        targets = [block(80 * index + 40, 150) for index in range(4)]
        for index, target in enumerate(targets):
            target.score = .5 + index * .05
        expected = sorted(targets, key=lambda obj: obj.score, reverse=True)[:config.VERIFY_MAX_CANDIDATES_PER_CLASS]
        self.assertEqual(self.confirm(targets), expected)
        self.img.find_blobs.assert_not_called()

    def test_empty_frame_does_not_publish_cached_confirmed_objects(self):
        target = block(200, 150)
        target.score = .5
        self.assertEqual(self.confirm([target]), [target])
        self.assertEqual(self.verifier.process(self.img, [], LABELS), [])
        self.assertEqual(self.verifier.process(self.img, [target], LABELS), [target])
        self.img.find_blobs.assert_not_called()

    def test_task_reset_still_requires_fresh_temporal_confirmation(self):
        target = block(200, 150)
        target.score = .5
        self.assertEqual(self.confirm([target]), [target])
        self.verifier.reset_task_history()
        self.assertEqual(self.confirm([target]), [target])
        self.img.find_blobs.assert_not_called()

    def test_lab_can_be_reenabled_for_comparison_without_disabling_history(self):
        target = block(200, 150)
        target.score = .5
        self.assertEqual(self.confirm([target]), [target])
        self.img.find_blobs.return_value = []
        self.img.find_blobs.side_effect = None
        with patch.object(config, "VERIFY_COLOR_ENABLED", True):
            self.assertEqual(self.verifier.process(self.img, [target], LABELS), [])
        self.img.find_blobs.assert_called_once()
        self.img.find_blobs.reset_mock()
        self.assertEqual(self.verifier.process(self.img, [target], LABELS), [target])
        self.img.find_blobs.assert_not_called()

    def test_missing_color_flag_also_defaults_to_no_lab(self):
        import detection_verifier as verifier_module
        old_config = SimpleNamespace(**{
            name: value for name, value in vars(config).items()
            if name != "VERIFY_COLOR_ENABLED"
        })
        target = block(200, 150)
        target.score = .5
        with patch.object(verifier_module, "config", old_config):
            self.assertEqual(self.confirm([target]), [target])
        self.img.find_blobs.assert_not_called()


if __name__ == "__main__":
    unittest.main()
