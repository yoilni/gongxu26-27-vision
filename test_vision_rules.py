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
    )
    sys.modules["maix"] = maix_stub

import app_config as config
import serial_controller as serial_module
from debug_target_preview import DebugTargetPreview
from detection_verifier import DetectionVerifier
from vision_overlay import draw_debug_yolo_detections, draw_debug_zone_obstacle_region
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


def block(center_x, center_y, class_id=5):
    return SimpleNamespace(
        x=center_x - 10,
        y=center_y - 10,
        w=20,
        h=20,
        class_id=class_id,
        score=0.9,
    )


def controller(mode):
    with patch.object(config, "UART_COORDINATE_ENABLED", False):
        result = VisionSerialController(
            576, 448, tracking_roi=[0, 359, 576, 89]
        )
    result.mode = mode
    result.target_label = "sqareredgreen"
    return result


class VisionRuleRegressionTests(unittest.TestCase):
    def setUp(self):
        # Existing behavioral tests need not print diagnostic snapshots.
        quiet = patch.object(config, "VISION_DIAGNOSTICS_ENABLED", False)
        quiet.start()
        self.addCleanup(quiet.stop)

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
            self.assertTrue(any("[ZONE-DETAIL]" in str(call) and "(238.0, 100.0)" in str(call)
                                and "id3" in str(call) for call in log.call_args_list))
            self.assertTrue(any("[NEAR-DETAIL]" in str(call) and "ids=[3]" in str(call)
                                for call in log.call_args_list))

    def test_debug_raw_yolo_green_outer_boxes_include_all_classes(self):
        img = SimpleNamespace(draw_rect=Mock())
        objects = [block(200, 300, class_id) for class_id in range(7)]
        draw_debug_yolo_detections(img, objects, 576, 448)
        self.assertEqual(img.draw_rect.call_count, 7)
        for call in img.draw_rect.call_args_list:
            self.assertEqual(call.args, (186, 286, 28, 28))
            self.assertEqual(call.kwargs, {"color": "green", "thickness": 1})

    def test_debug_raw_yolo_boxes_clip_and_do_not_draw_missing_detections(self):
        img = SimpleNamespace(draw_rect=Mock())
        draw_debug_yolo_detections(img, [block(0, 0)], 576, 448)
        img.draw_rect.assert_called_once_with(
            0, 0, 14, 14, color="green", thickness=1
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
        vertices = ((238, 100), (338, 100), (575, 379), (0, 379))
        self.assertEqual(zone_obstacle_polygon(roi, 576, 448), vertices)
        self.assertEqual(img.draw_line.call_count, 4)
        for index, call in enumerate(img.draw_line.call_args_list):
            self.assertEqual(call.args, vertices[index] + vertices[(index + 1) % 4])
            self.assertEqual(call.kwargs, {"color": "red", "thickness": 2})
        current = controller(serial_module.MODE_SEARCH_ZONE)
        for x, y in vertices + ((288, 100), (288, 200), (200, 279), (200, 359)):
            target = block(x, y)
            self.assertIs(current._find_roi_edge_blocker([target], LABELS), target)
        for x, y in ((288, 99), (288, 380), (237, 100), (339, 100), (0, 279), (575, 279)):
            self.assertIsNone(current._find_roi_edge_blocker([block(x, y)], LABELS))

    def test_debug_zone_obstacle_trapezoid_uses_config_and_clips_to_image(self):
        img = SimpleNamespace(draw_line=Mock())
        with patch.object(config, "UART_ZONE_OBSTACLE_FRONT_Y", -10), \
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

    def test_fast_uart_worker_does_not_wait_50ms_for_updated_coordinates(self):
        current = controller(CMD_TRACK)
        clock = {"ms": 1000, "waits": 0}
        writes = []

        def write(packet):
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
            self.assertIn(build_event_packet(EVENT_ROI_READY), current._pending_events)

    def test_04_requires_initial_screening_and_rescreens_after_loss(self):
        current = controller(CMD_TRACK)
        current.target_label = "sqarered"
        first = block(100, 150, 4)
        nearby = block(130, 160, 3)
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
        for x, y in ((0, 359), (575, 359), (351, 402), (575, 447)):
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

    def test_24_34_repeats_500ms_and_stops_when_target_leaves_roi(self):
        current = controller(MODE_FINAL_ROI_TRACK)
        event = build_event_packet(serial_module.EVENT_FINAL_TARGET_IN_ROI)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            current.submit([block(351, 359)], LABELS)
        self.assertEqual(current._next_outgoing_packet(1000), event)
        self.assertIsNone(current._next_outgoing_packet(1499))
        self.assertEqual(current._next_outgoing_packet(1500), event)
        with patch.object(serial_module, "_now_ms", return_value=1600):
            current.submit([block(351, 358)], LABELS)
        self.assertFalse(current._final_roi_sent)
        self.assertIsNotNone(current._latest_packet)
        self.assertNotEqual(current._next_outgoing_packet(1600), event)
        with patch.object(serial_module, "_now_ms", return_value=1650):
            current.submit([block(351, 359)], LABELS)
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

    def test_24_loss_with_no_qualified_replacement_keeps_ee_flow(self):
        current = controller(MODE_FINAL_ROI_TRACK)
        current.target_label = serial_module.TARGET_BLACK_OR_GREEN
        initial = block(100, 300)
        distant = block(450, 310, 6)
        nearby = block(480, 310, 3)
        with patch.object(serial_module, "_now_ms", return_value=1000):
            current.submit([initial], LABELS)
            for _ in range(config.UART_FINAL_NO_TARGET_FRAMES):
                self.assertIsNone(current.submit([distant, nearby], LABELS))
        self.assertIsNone(current._final_target_lock)
        ee = build_event_packet(serial_module.EVENT_NO_TARGET)
        self.assertEqual(current._next_outgoing_packet(1000), ee)
        self.assertIsNone(current._next_outgoing_packet(2000))
        current._handle_command(serial_module.CMD_FINAL_SEARCH_DONE)
        with patch.object(serial_module, "_now_ms", return_value=2100):
            self.assertIs(current.submit([distant], LABELS), distant)

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
            self.assertIsNone(current.submit([target, blocker], LABELS))
            self.assertIsNone(current._final_target_lock)
            self.assertIs(current.submit([target], LABELS), target)

    def test_24_transition_locks_decision_target_before_next_frame(self):
        for source_mode in (MODE_POST_ARRANGE_CHECK, MODE_WAIT_14_HANDSHAKE):
            with self.subTest(source_mode=source_mode):
                current = controller(source_mode)
                first = block(100, 300)
                with patch.object(serial_module, "_now_ms", return_value=1000):
                    if source_mode == MODE_WAIT_14_HANDSHAKE:
                        current._handle_command(serial_module.CMD_TRACK_CENTER_DONE)
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
            self.assertEqual(current.mode, MODE_ARRANGE_RIGHTMOST)
            self.assertTrue(current._initial_arrangement_decision_pending)
            self.assertEqual(current._middle_target_anchor, (288.0, None, None))
            self.assertIsNone(current._next_outgoing_packet(6000))
            middle = block(288, 300)
            side = block(390, 310, 3)
            current.submit([middle, side], LABELS, raw_objects=[middle, side])
            self.assertNotIn(
                build_event_packet(EVENT_TRACK_CENTER_REACHED),
                current._pending_events,
            )
            self.assertEqual(
                current._next_outgoing_packet(6001),
                build_event_packet(EVENT_ROI_READY),
            )
            self.assertIsNone(current._next_outgoing_packet(6501))
            # A fresh search/04 round may report its own first loss.
            current._handle_command(serial_module.CMD_SEARCH)
            current._handle_command(CMD_TRACK)
            for _ in range(config.UART_TRACK_LOST_CONFIRM_FRAMES):
                current.submit([], LABELS, raw_objects=[])
            self.assertEqual(current._next_outgoing_packet(7000), e4)

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
        with patch.object(serial_module, "_now_ms", return_value=1000), \
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

    def test_search_lock_missing5_no_stale_coordinates_then_reacquires(self):
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
            self.assertIsNone(current._search_target_lock)
            self.assertEqual(current._next_outgoing_packet(1000),
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

    def test_raw_yolo_follow_requires_acquisition_and_still_excludes_safety_zones(self):
        for mode in (serial_module.CMD_SEARCH, CMD_TRACK, MODE_FINAL_ROI_TRACK):
            with self.subTest(mode=mode):
                current = controller(mode)
                target, zone = block(100, 150), block(100, 150, 1)
                with patch.object(serial_module, "_now_ms", return_value=1000):
                    self.assertIsNone(current.submit([], LABELS, raw_objects=[target]))
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

    def test_14_and_44_use_merged_selector_for_24_or_02(self):
        lone = block(200, 310)
        high = block(30, 300)
        low = block(450, 310)
        close_blue = block(300, 310, 3)
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

    def test_confirmed_red_thresholds(self):
        self.assertEqual(config.UART_FINAL_RED_MIN_DX_Y_GT_290, 240)
        self.assertEqual(config.UART_FINAL_RED_MIN_DX_Y_GT_100, 120)
        self.assertEqual(config.UART_FINAL_RED_MIN_DX_Y_GT_90, 70)
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
                target = block(30, 400)
                farther = block(450, 430)
                with patch.object(serial_module, "_now_ms", return_value=1000):
                    self.assertIs(current.submit([farther, target], LABELS), target)
                    now_closer = block(450, 403)
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

    def test_first_green_exception_does_not_change_24_or_later_green_04(self):
        target, neighbor = block(360, 250), block(380, 260, 3)
        for mode, decided in ((MODE_FINAL_ROI_TRACK, False), (CMD_TRACK, True)):
            with self.subTest(mode=mode, decided=decided):
                current = controller(mode)
                current._first_green_near_view_decided = decided
                self.assertIsNone(current.submit([target, neighbor], LABELS))
                self.assertEqual(current._selection_diagnostic_policy, "merged")

    def test_post_22_emits_24_or_repeats_e2(self):
        current = controller(MODE_POST_ARRANGE_CHECK)
        target = block(200, 300)
        current.submit([target], LABELS, raw_objects=[target])
        self.assertEqual(current.mode, MODE_FINAL_ROI_TRACK)
        self.assertIn(
            build_event_packet(EVENT_SKIP_TO_FINAL_TRACK),
            current._pending_events,
        )

        current = controller(MODE_POST_ARRANGE_CHECK)
        nearby = block(250, 300, 3)
        current.submit([target, nearby], LABELS, raw_objects=[target, nearby])
        self.assertEqual(current.mode, MODE_POST_ARRANGE_CHECK)
        self.assertEqual(
            current._next_outgoing_packet(current._latest_submit_ms),
            build_event_packet(EVENT_ARRANGEMENT_NO_OBJECT),
        )

    def test_04_uses_frame_center_and_y_can_pass_below_it(self):
        current = controller(CMD_TRACK)
        self.assertEqual(current._reference_point(CMD_TRACK), (288.0, 224.0))
        self.assertEqual(
            current._reference_point(MODE_WAIT_14_HANDSHAKE),
            (288.0, 403.5),
        )
        target = block(288, 420)
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


if __name__ == "__main__":
    unittest.main()
