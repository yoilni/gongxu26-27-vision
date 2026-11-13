"""ROI geometry, tracking handoffs and 05 inventory boundaries."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from test_vision_rules import LABELS, block, controller, near_view_block
import app_config as config
import serial_controller as s
import vision_overlay as overlay
from tracking_roi import (
    near_view_roi_vertices, point_in_near_view_roi, point_in_tracking_roi, roi_triangle_vertices,
)
from zone_obstacle_region import zone_obstacle_polygon


class TrackingTriangleTests(unittest.TestCase):
    def setUp(self):
        for helper in (patch("builtins.print"), patch.object(config, "VISION_DIAGNOSTICS_ENABLED", False)):
            helper.start()
            self.addCleanup(helper.stop)
        self.roi = overlay.make_bottom_center_roi(576, 448)

    def make(self, mode=s.MODE_FINAL_ROI_TRACK, class_id=5, post07=False):
        current = controller(mode)
        current.tracking_roi = list(self.roi)
        current.target_label = LABELS[class_id]
        current._post07_started = post07
        return current

    def submit(self, current, objects, now=1000, verified=None):
        with patch.object(s, "_now_ms", return_value=now):
            return current.submit(objects if verified is None else verified, LABELS,
                                  raw_objects=objects, frame_generation=current.frame_generation())

    def warm(self, current, objects, start=1000, verified=None):
        return [self.submit(current, objects, start + index * 50, verified) for index in range(3)]

    def test_default_apex_and_base_match_current_pixel_border(self):
        self.assertEqual(config.UART_TRACKING_ROI_SHAPE, "triangle")
        self.assertEqual(self.roi, [0, 319, 576, 129])
        self.assertEqual(roi_triangle_vertices(self.roi), ((313, 344), (150, 447), (476, 447)))
        original = list(self.roi)
        img = Mock()
        overlay.draw_roi_border(img, self.roi, near_view_roi=near_view_roi_vertices(576, 448))
        self.assertEqual([call.args for call in img.draw_line.call_args_list],
                         [(313, 344, 150, 447), (150, 447, 476, 447), (476, 447, 313, 344)])
        self.assertTrue(all(call.kwargs == dict(color=overlay.ROI_COLOR, thickness=2)
                            for call in img.draw_line.call_args_list))
        img.draw_rect.assert_not_called()
        self.assertEqual(self.roi, original)

    def test_inclusive_apex_sloping_edges_base_and_outside_corners(self):
        for x, y, inside in ((313, 344, True), (313, 343.999, False),
                              (312.99, 344, False), (313.01, 344, False),
                              (231.5, 395.5, True), (231.49, 395.5, False),
                              (394.5, 395.5, True), (394.51, 395.5, False),
                              (288, 319, False), (313, 350, True),
                              (288, 360, True), (20, 360, False), (550, 360, False),
                              (150, 447, True), (476, 447, True),
                              (149.99, 447, False), (476.01, 447, False),
                              (288, 448, False), (288, 447.01, False)):
            with self.subTest(x=x, y=y):
                self.assertEqual(point_in_tracking_roi(x, y, self.roi), inside)

    def test_polygon_draw_and_membership_share_custom_geometry(self):
        roi = [100, 200, 400, 100]
        self.assertEqual(roi_triangle_vertices(roi), ((325, 225), (250, 299), (400, 299)))
        for point, inside in (((325, 224), False), ((325, 225), True),
                               ((250, 299), True), ((400, 299), True),
                               ((249, 299), False), ((401, 299), False),
                               ((270, 250), False), ((360, 250), False), ((310, 250), True)):
            self.assertEqual(point_in_tracking_roi(*point, roi), inside)
        img = Mock()
        overlay.draw_roi_border(img, roi)
        self.assertEqual(img.draw_line.call_args_list[0].args, (325, 225, 250, 299))

    def test_apex_offsets_change_entry_geometry_and_keep_uart_reference_and_left_point(self):
        current = self.make(s.MODE_SEARCH_ZONE)
        self.assertEqual(current._reference_point(s.MODE_SEARCH_ZONE), (288, 383.5))
        apex, left, right = roi_triangle_vertices(self.roi)
        self.assertEqual((apex, left, right), ((313, 344), (150, 447), (476, 447)))
        target = block(313, 350)
        earlier_point = block(298, 330)
        self.assertTrue(current._target_center_in_tracking_roi(target))
        self.assertFalse(current._target_center_in_tracking_roi(earlier_point))
        self.assertEqual(current._reference_point(s.MODE_FINAL_ROI_TRACK), (313, 359))
        with patch.object(config, "UART_TRACKING_ROI_APEX_OFFSET_X", 0), \
                patch.object(config, "UART_TRACKING_ROI_APEX_OFFSET_Y", 0):
            self.assertEqual(roi_triangle_vertices(self.roi), ((288, 319), left, (426, 447)))
            self.assertTrue(current._target_center_in_tracking_roi(earlier_point))
            self.assertEqual(current._reference_point(s.MODE_FINAL_ROI_TRACK), (288, 334))

    def test_missing_vertex_offsets_use_current_defaults(self):
        settings = SimpleNamespace(**{key: value for key, value in vars(config).items()
                                     if key not in ("UART_TRACKING_ROI_APEX_OFFSET_X",
                                                    "UART_TRACKING_ROI_APEX_OFFSET_Y",
                                                    "UART_TRACKING_ROI_BASE_INSET_X")})
        with patch("tracking_roi.config", settings):
            self.assertEqual(roi_triangle_vertices(self.roi), ((313, 344), (150, 447), (476, 447)))
            self.assertTrue(point_in_tracking_roi(313, 350, self.roi))

    def test_invalid_or_degenerate_roi_never_draws_or_accepts(self):
        for roi in (None, [], [0, 0, 0, 10], [0, 0, 10, -1], [0, 0, 1, 10],
                    [0, 0, 10, 1], [0, float("nan"), 10, 10], [0, 0, float("inf"), 10]):
            self.assertIsNone(roi_triangle_vertices(roi))
            self.assertFalse(point_in_tracking_roi(0, 0, roi))
            img = Mock()
            overlay.draw_roi_border(img, roi)
            self.assertEqual(img.mock_calls, [])
        self.assertFalse(point_in_tracking_roi(float("nan"), 400, self.roi))
        self.assertFalse(point_in_tracking_roi(288, float("inf"), self.roi))

    def test_missing_shape_config_defaults_to_triangle(self):
        with patch.dict(vars(config), {}, clear=False):
            delattr(config, "UART_TRACKING_ROI_SHAPE")
            self.assertFalse(point_in_tracking_roi(20, 340, self.roi))
            img = Mock()
            overlay.draw_roi_border(img, self.roi)
            self.assertEqual(img.draw_line.call_count, 3)

    def test_explicit_legacy_rectangle_setting_remains_available(self):
        with patch.object(config, "UART_TRACKING_ROI_SHAPE", "rectangle"):
            self.assertTrue(point_in_tracking_roi(20, 340, self.roi))
            self.assertFalse(point_in_tracking_roi(576, 400, self.roi))
            self.assertFalse(point_in_tracking_roi(288, 448, self.roi))
            img = Mock()
            overlay.draw_roi_border(img, self.roi)
            img.draw_rect.assert_called_once_with(*self.roi, color=overlay.ROI_COLOR, thickness=2)
            img.draw_line.assert_not_called()

    def test_controller_and_overlay_use_same_float_center_not_box_overlap(self):
        current = self.make()
        for x, y in ((288, 319), (144, 383), (143.99, 383), (20, 340), (288, 400)):
            obj = block(x, y)
            self.assertEqual(current._target_center_in_tracking_roi(obj),
                             overlay.object_center_in_roi(obj, self.roi))
        tiny = SimpleNamespace(x=287.5, y=359.5, w=1, h=1, class_id=5)
        self.assertTrue(overlay.object_center_in_roi(tiny, self.roi))
        overlapping = block(270, 360)
        self.assertFalse(current._target_center_in_tracking_roi(overlapping))

    def test03_corners_remain_eligible_but_triangle_entry_is_excluded(self):
        for class_id in (3, 4, 5, 6):
            current = self.make(s.CMD_SEARCH, class_id, class_id != 5)
            outside = block(30, 340, class_id)
            self.assertEqual(self.warm(current, [outside]), [None, outside if class_id == 6 else None, outside])
            inside = block(288, 360, class_id)
            for now in (1150, 1200, 1250, 1300, 1350):
                self.assertIsNone(self.submit(current, [inside], now))
                self.assertIsNone(current._latest_packet)
            if class_id != 6:
                self.assertEqual(current._next_outgoing_packet(1350), s.build_event_packet(0xE3))

    def test_wait04_and_actual04_exclusion_use_same_triangle(self):
        for mode in (s.MODE_WAIT_04_HANDSHAKE, s.CMD_TRACK):
            current = self.make(mode)
            outside, inside = block(30, 340), block(288, 360)
            current._track_target_lock = current._final_target_identity(outside)
            self.assertIs(self.submit(current, [outside], verified=[]), outside)
            selected = self.submit(current, [inside], 1050, verified=[])
            self.assertIsNone(selected)
            self.assertIsNone(current._latest_packet)

    def test04_and_both_waits_exclude_roi_cargo_for_every_task_without_cached_coords(self):
        for mode in (s.CMD_TRACK, s.MODE_WAIT_04_HANDSHAKE, s.MODE_WAIT_14_HANDSHAKE):
            for class_id in (3, 4, 5, 6):
                with self.subTest(mode=mode, class_id=class_id):
                    current = self.make(mode, class_id, class_id != 5)
                    outside, inside = block(30, 340, class_id), block(288, 360, class_id)
                    current._track_target_lock = current._final_target_identity(outside)
                    self.assertIs(self.submit(current, [outside], verified=[]), outside)
                    self.assertIsNotNone(current._latest_packet)
                    for index in range(5):
                        self.assertIsNone(self.submit(current, [inside], 1050 + index*50, verified=[]))
                        self.assertIsNone(current._latest_packet)
                    self.assertNotIn(s.build_event_packet(0x14), current._pending_events)
                    self.assertFalse(current._final_roi_sent)

    def test04_initial_roi_only_does_not_acquire_even_after_three_confirmations(self):
        for class_id in (3, 4, 5, 6):
            current = self.make(s.CMD_TRACK, class_id, class_id != 5)
            self.assertEqual(self.warm(current, [block(288, 360, class_id)]), [None]*3)
            self.assertIsNone(current._track_target_lock)
            self.assertIsNone(current._latest_packet)
            self.assertEqual(current._pending_events, [])

    def test04_verbose_roi_exclusion_is_not_mislabeled_as_safety_zone_rejection(self):
        current = self.make(s.CMD_TRACK)
        with patch.object(config, "VISION_DIAGNOSTICS_ENABLED", True), \
                patch.object(config, "VISION_DIAGNOSTICS_VERBOSE", True), patch("builtins.print") as log:
            self.submit(current, [block(288, 360)])
        messages = [str(call.args[0]) for call in log.call_args_list if call.args]
        self.assertTrue(any("[TRACK-ROI-EXCLUDE]" in message and "id5" in message for message in messages))
        self.assertTrue(any("[SAFE-EXCLUDE]" in message and "none" in message for message in messages))

    def test14_roi_neighbors_are_ignored_for_all_tasks_but_outside_corners_are_not(self):
        for class_id in (4, 5, 6):
            current = self.make(class_id=class_id, post07=class_id != 5)
            target = block(270, 360, class_id)  # outside shifted left slope
            # Same-Y neighbor is horizontally adjacent but inside triangle.
            inside = block(288, 360, 3)
            outside = block(230, 360, 3)
            self.assertTrue(current._target_center_in_tracking_roi(inside))
            self.assertFalse(current._target_center_in_tracking_roi(target))
            self.assertEqual(current._directional_arrangement_event(target, [target, inside]), s.EVENT_SKIP_TO_FINAL_TRACK)
            self.assertEqual(current._directional_arrangement_event(target, [target, inside, outside]), s.EVENT_MICRO_FORWARD)
            # Triangle apex/edges count as inside; raw box overlap alone does not.
            target = block(313, 330, class_id)
            self.assertEqual(current._directional_arrangement_event(target, [target, block(313, 344, 4)]), s.EVENT_SKIP_TO_FINAL_TRACK)
            self.assertEqual(current._directional_arrangement_event(target, [target, block(313, 343, 4)]), s.EVENT_ROI_READY)

    def test24_crossing_capture_line_sends34_without_coordinates(self):
        for class_id in (3, 4, 5, 6):
            current = self.make(class_id=class_id, post07=class_id != 5)
            inside = block(313 if class_id == 4 else 288, 420, class_id)
            self.warm(current, [inside])
            self.assertTrue(current._final_roi_sent)
            self.assertIsNone(current._latest_packet)
            self.assertEqual(current._next_outgoing_packet(1100), s.build_event_packet(0x34))
            current._serial = SimpleNamespace(write=Mock(return_value=11))
            # Even a manually prepared packet with no generation argument is
            # barred while34 owns the handoff;05 cargo metadata is a different mode.
            current._write_packet(s.build_coordinate_packet(class_id, 0, 0))
            current._serial.write.assert_not_called()

    def test24_purple_roi_entry_above_capture_line_sends34_without_coordinates(self):
        for label in (LABELS[5], LABELS[6], s.TARGET_BLACK_OR_GREEN):
            with self.subTest(label=label):
                current = self.make(class_id=6 if label != LABELS[5] else 5,
                                    post07=label != LABELS[5])
                current.target_label = label
                class_id = 5 if label == LABELS[5] else 6
                outside, inside = block(313, 340, class_id), block(313, 344, class_id)
                self.warm(current, [outside])
                self.assertIsNotNone(current._latest_packet)
                self.assertLess(344, current._reference_point(s.MODE_FINAL_ROI_TRACK)[1])
                self.assertTrue(current._target_center_in_tracking_roi(inside))
                self.submit(current, [inside], 1150)
                self.assertEqual(current._next_outgoing_packet(1150), s.build_event_packet(0x34))
                self.assertIsNone(current._latest_packet)

    def test14_and44_roi_neighbor_exclusion_preserves_target_and_applies_to_first_green(self):
        for command in (s.CMD_TRACK_CENTER_DONE, s.CMD_TRACK_RECOVERY_DONE):
            for class_id, post07 in ((5, False), (3, True), (4, True), (6, True)):
                current = self.make(s.MODE_WAIT_14_HANDSHAKE if command == 0x14 else s.CMD_TRACK,
                                    class_id, post07)
                target, inside = block(288, 320, class_id), block(288, 360, 3)
                current._track_target_lock = current._final_target_identity(target)
                with patch.object(config, "UART_DIRECT_CAPTURE_AFTER_14", False), \
                        patch.object(s, "_now_ms", return_value=900):
                    current.process_received_bytes(s.build_command_packet(command))
                for now in (1000, 1050, 1100):
                    self.submit(current, [target, inside], now)
                if command == s.CMD_TRACK_RECOVERY_DONE and class_id == 5 and not post07:
                    self.assertEqual(current.mode, s.CMD_SEARCH)
                    self.assertEqual(current._search_target_lock, current._final_target_identity(target))
                    self.assertEqual(current._latest_packet, s.build_coordinate_packet(5, -25, -19))
                    self.assertEqual(current._pending_events, [])
                    continue
                self.assertEqual(current.mode, s.MODE_FINAL_ROI_TRACK)
                self.assertEqual(current._final_target_lock, current._final_target_identity(target))
                self.assertNotIn(s.build_event_packet(0x42), current._pending_events)

    def test24_capture_line_applies_outside_the_triangle(self):
        for class_id in (3, 5, 6):
            current = self.make(class_id=class_id, post07=class_id != 5)
            outside = block(149, 447, class_id)
            self.assertFalse(current._target_center_in_tracking_roi(outside))
            self.assertEqual(self.warm(current, [outside]), [None, outside if class_id == 6 else None, outside])
            self.assertTrue(current._final_roi_sent)
            self.assertEqual(current._next_outgoing_packet(1100), s.build_event_packet(0x34))
            self.assertIsNone(current._latest_packet)

    def test24_capture_line_uses_center_y_strictly(self):
        current = self.make()
        target = block(270, 350)
        self.warm(current, [target])
        self.assertFalse(current._final_roi_sent)
        self.assertEqual(current._latest_packet, s.build_coordinate_packet(5, -43, -9))
        _, line_y = current._reference_point(s.MODE_FINAL_ROI_TRACK)
        self.assertEqual(line_y, 359)
        self.submit(current, [block(288, line_y)], 1150)
        self.assertFalse(current._final_roi_sent)
        self.assertEqual(current._latest_packet, s.build_coordinate_packet(5, -25, 0))
        self.submit(current, [block(288, line_y + .01)], 1200)
        self.assertTrue(current._final_roi_sent)
        self.assertEqual(current._next_outgoing_packet(1200), s.build_event_packet(0x34))

    def test_casualty24_aims_deeper_and_continues_steering_until_confirmed(self):
        current = self.make(class_id=4, post07=True)
        self.warm(current, [block(270, 300, 4)])
        self.assertFalse(current._final_roi_sent)
        self.assertEqual(current._latest_packet, s.build_coordinate_packet(4, -43, -95))
        self.submit(current, [block(313, 359, 4)], 1150)
        self.assertFalse(current._final_roi_sent)
        self.assertEqual(current._latest_packet, s.build_coordinate_packet(4, 0, -36))
        for now in (1200, 1250):
            self.submit(current, [block(313, 365, 4)], now)
            self.assertFalse(current._final_roi_sent)
            self.assertEqual(current._latest_packet, s.build_coordinate_packet(4, 0, -30))
        self.submit(current, [block(313, 365, 4)], 1300)
        self.assertIsNone(current._latest_packet)
        self.assertEqual(current._next_outgoing_packet(1300), s.build_event_packet(0x34))

    def test_post07_red_can_capture_another_deep_box_without_switching_the_lock(self):
        current = self.make(class_id=4, post07=True)
        outside, corner = block(100, 180, 4), block(20, 405, 4)
        current._final_target_lock = current._final_target_identity(outside)
        self.warm(current, [outside, corner], verified=[])
        self.assertFalse(current._final_roi_sent)
        self.assertEqual(current._final_target_lock, current._final_target_identity(outside))
        inside = block(313, 395, 4)
        self.warm(current, [outside, inside], 1150, verified=[])
        self.assertTrue(current._final_roi_sent)
        self.assertEqual(current._final_target_lock, current._final_target_identity(outside))
        self.assertEqual(current._next_outgoing_packet(1250), s.build_event_packet(0x34))

    def test51_loaded_mask_is_inner_half_of_triangle_not_old_rectangle(self):
        current = self.make(post07=True)
        current.target_label = s.TARGET_BLACK_OR_GREEN
        for x, y, loaded in ((288, 383.49, False), (288, 383.5, True),
                             (20, 390, False), (550, 390, False), (288, 400, True),
                             (149, 447, False), (150, 447, True),
                             (476, 447, True), (477, 447, False), (288, 448, False)):
            self.assertEqual(current._target_center_in_supply_loaded_roi(block(x, y)), loaded)
        with patch.object(config, "UART_SUPPLY_COLLECT_LOADED_ROI", [0, 370, 80, 60]):
            self.assertTrue(current._target_center_in_supply_loaded_roi(block(20, 390)))

    def test51_new_target_crosses_capture_line_and_keeps_handoff_into_inner_roi(self):
        for class_id in (5, 6):
            current = self.make(post07=True)
            current.target_label = s.TARGET_BLACK_OR_GREEN
            cargo, new = block(288, 425), block(30, 290, class_id)
            self.assertEqual(self.warm(current, [cargo, new]), [None, new, new])
            self.assertFalse(current._final_roi_sent)
            crossing = block(30, 360, class_id)
            self.assertFalse(current._target_center_in_tracking_roi(crossing))
            self.submit(current, [cargo, crossing], 1150)
            self.assertEqual(current._next_outgoing_packet(1150), s.build_event_packet(0x34))
            self.assertIsNone(current._latest_packet)
            self.submit(current, [cargo, block(288, 420, class_id)], 1200)
            self.assertTrue(current._final_roi_sent)
            self.assertIsNone(current._latest_packet)
            self.assertEqual(current._supply_roi_handoff[0], class_id)
            self.assertEqual(len(current._supply_loaded_roi_anchors), 1)
            self.submit(current, [cargo], 1250)
            self.assertFalse(current._final_roi_sent)

    def test_state_specific_roi_reference_points(self):
        current = self.make()
        for label in (LABELS[3], LABELS[4], LABELS[5], LABELS[6], s.TARGET_BLACK_OR_GREEN):
            current.target_label = label
            for mode in (s.CMD_SEARCH, s.MODE_WAIT_04_HANDSHAKE):
                self.assertEqual(current._reference_point(mode), (313, 339))
            self.assertEqual(current._reference_point(s.MODE_FINAL_ROI_TRACK),
                             (313, 395 if label == LABELS[4] else 359))
        for mode in (s.MODE_SEARCH_ZONE, s.MODE_WAIT_14_HANDSHAKE,
                     s.MODE_MICRO_FORWARD, s.MODE_MICRO_SIDE, s.MODE_POST_ARRANGE_CHECK):
            self.assertEqual(current._reference_point(mode), (288, 383.5))
        self.assertEqual(current._reference_point(s.CMD_TRACK), (313, 294))
        self.assertEqual(zone_obstacle_polygon(self.roi, 576, 448)[-2:], ((575.0, 339.0), (0.0, 339.0)))

    def test03_uses_red_reticle_until_rx04_switches_to_purple_reticle(self):
        current = self.make(s.CMD_SEARCH)
        self.warm(current, [block(313, 294)])
        self.assertEqual(current.mode, s.CMD_SEARCH)
        self.assertEqual(current._latest_packet, s.build_coordinate_packet(5, 0, -45))

        self.submit(current, [block(313, 339)], 1150)
        self.assertEqual(current.mode, s.MODE_WAIT_04_HANDSHAKE)
        self.assertIn(s.build_event_packet(0x04), current._pending_events)
        self.assertEqual(current._latest_packet, s.build_coordinate_packet(5, 0, 0))
        self.submit(current, [block(317, 340)], 1200)
        self.assertEqual(current._latest_packet, s.build_coordinate_packet(5, 4, 1))
        self.assertNotIn(s.build_event_packet(0x14), current._pending_events)

        with patch.object(s, "_now_ms", return_value=1210):
            current.process_received_bytes(s.build_command_packet(0x04))
        self.submit(current, [block(313, 273)], 1250)
        self.assertEqual(current.mode, s.CMD_TRACK)
        self.assertEqual(current._latest_packet, s.build_coordinate_packet(5, 0, -21))
        self.submit(current, [block(313, 274)], 1300)
        self.assertEqual(current.mode, s.MODE_WAIT_14_HANDSHAKE)
        self.assertIn(s.build_event_packet(0x14), current._pending_events)

    def test_mcu03_restarts_final24_with_fresh_reticle_coordinates(self):
        for class_id in (4, 5, 6):
            with self.subTest(class_id=class_id):
                current = self.make(class_id=class_id, post07=class_id != 5)
                old_target = block(285, 325, class_id)
                self.warm(current, [old_target])
                old_coordinate = current._next_outgoing_packet(1100)
                self.assertIsNotNone(old_coordinate)
                old_generation = current.frame_generation()
                old_events = [s.build_event_packet(code) for code in (0x34, 0xEE)]
                current._pending_events.extend(old_events)
                current._serial = SimpleNamespace(write=Mock(side_effect=len))

                with patch.object(s, "_now_ms", return_value=6000):
                    current.process_received_bytes(s.build_command_packet(0x03))
                    for old_packet in [old_coordinate] + old_events:
                        current._write_packet(old_packet, old_generation)
                current._serial.write.assert_not_called()
                self.assertEqual(current.mode, s.CMD_SEARCH)
                self.assertEqual(current.target_label, LABELS[class_id])
                self.assertIsNone(current._final_target_lock)
                self.assertEqual(current._pending_events, [])
                self.assertIsNone(current.submit([old_target], LABELS, frame_generation=old_generation))

                new_target = block(330, 250, class_id)
                expected = [None, new_target if class_id == 6 else None, new_target]
                self.assertEqual(self.warm(current, [new_target], 6100), expected)
                self.assertEqual(current._next_outgoing_packet(6200),
                                 s.build_coordinate_packet(class_id, 17, -89))

    def test_direct_capture_retries_short24_and_fences_early34(self):
        current = self.make(s.MODE_WAIT_14_HANDSHAKE, 4, post07=True)
        current._track_target_lock = current._final_target_identity(block(313, 314, 4))
        current._serial = SimpleNamespace(write=Mock(side_effect=[2, 5, 5]))
        with patch.object(s, "_now_ms", return_value=1000):
            current.process_received_bytes(s.build_command_packet(0x14))
        generation = current.frame_generation()
        target = block(313, 400, 4)
        self.assertIsNone(self.submit(current, [target], 1000))
        self.assertFalse(current._final_roi_sent)
        request = s.build_event_packet(0x24)
        self.assertEqual(current._next_outgoing_packet(1000), request)
        with patch.object(s, "_now_ms", return_value=1000):
            current._write_packet(request, generation)
        self.assertEqual(current.frame_generation(), generation)
        self.assertIsNone(self.submit(current, [target], 1001))
        self.assertEqual(current._next_outgoing_packet(1001), request)
        with patch.object(s, "_now_ms", return_value=1001):
            current._write_packet(request, generation)
            current._write_packet(request, generation)
        self.assertEqual(current._serial.write.call_count, 2)
        self.assertIsNone(current.submit([target], LABELS, frame_generation=generation))
        self.assertFalse(current._final_roi_sent)
        self.submit(current, [target], 1002)
        self.assertFalse(current._final_roi_sent)
        self.submit(current, [target], 1250)
        self.assertEqual(current._next_outgoing_packet(1300), s.build_coordinate_packet(4, 0, 5))
        with patch.object(s, "_now_ms", return_value=1300):
            current._write_packet(s.build_event_packet(0x34), current.frame_generation())
        self.assertEqual(current._serial.write.call_count, 2)
        for now in (1301, 1351):
            self.submit(current, [target], now)
            self.assertFalse(current._final_roi_sent)
        self.submit(current, [target], 1401)
        handoff = current._next_outgoing_packet(1401)
        self.assertEqual(handoff, s.build_event_packet(0x34))
        with patch.object(s, "_now_ms", return_value=1401):
            current._write_packet(handoff, current._outgoing_generation)
        self.assertEqual([call.args[0] for call in current._serial.write.call_args_list],
                         [request, request, handoff])

    def test_queued24_retries_short_write_and_discards_stale_capture(self):
        current = self.make()
        request, handoff = s.build_event_packet(0x24), s.build_event_packet(0x34)
        current._pending_events.append(request)
        current._serial = SimpleNamespace(write=Mock(side_effect=[2, 5, 5]))
        for now in (1000, 1050):
            self.assertEqual(current._next_outgoing_packet(now), request)
            with patch.object(s, "_now_ms", return_value=now):
                current._write_packet(request, current._outgoing_generation)

        target = block(313, 360)
        for now in (1051, 1052, 1053):
            self.submit(current, [target], now)
        self.assertTrue(current._final_roi_sent)
        self.assertIsNone(current._next_outgoing_packet(1350))
        self.assertFalse(current._final_roi_sent)
        self.submit(current, [target], 1351)
        self.assertEqual(current._next_outgoing_packet(1351), handoff)
        with patch.object(s, "_now_ms", return_value=1351):
            current._write_packet(handoff, current._outgoing_generation)
        self.assertEqual([call.args[0] for call in current._serial.write.call_args_list],
                         [request, request, handoff])

    def test05_roi_has_trapezoid_above_full_width_rectangle_and_inclusive_edges(self):
        vertices = near_view_roi_vertices(576, 448)
        self.assertEqual(vertices, ((208, 102), (368, 102), (575, 307), (575, 407), (0, 407), (0, 307)))
        for x, y in (*vertices, (288, 112), (104, 204.5), (471.5, 204.5), (0, 350), (575, 350)):
            self.assertTrue(point_in_near_view_roi(x, y, vertices))
        for x, y in ((288, 101), (288, 408), (103.9, 204.5), (471.6, 204.5),
                     (0, 306), (575, 306), (-0.1, 350), (575.1, 350)):
            self.assertFalse(point_in_near_view_roi(x, y, vertices))

    def test05_excludes_targets_outside_red_roi_for_every_task(self):
        for class_id, post07 in ((5, False), (4, True), (6, True)):
            current = self.make(s.MODE_NEAR_VIEW_CHECK, class_id, post07)
            current.near_view_roi = near_view_roi_vertices(576, 448)
            current._near_view_started_ms = 1000
            target = near_view_block(100, 150, class_id)
            for now in (1500, 1550, 1600):
                self.submit(current, [target], now)
            self.assertEqual(current.mode, s.MODE_FINAL_ROI_TRACK)
            self.assertEqual(current._pending_events, [s.build_event_packet(0x24)])

    def test_ordinary_final_and_evacuation05_use_the_same_red_roi(self):
        green = block(10, 350, 5)
        overlapping_blue = block(505, 230, 3)
        outside_black = near_view_block(100, 180, 6)
        below_red_roi = block(313, 430, 5)
        objects = [green, overlapping_blue, outside_black, below_red_roi]
        for source in (None, "before06", "evacuation"):
            with self.subTest(source=source):
                current = self.make(s.MODE_NEAR_VIEW_CHECK)
                current.near_view_roi = near_view_roi_vertices(576, 448)
                current._near_view_started_ms = 1000
                current._zone_load_recheck_source = source
                current._zone_load_recheck = "count05" if source else None
                self.assertFalse(current._target_center_in_tracking_roi(green))
                self.assertEqual(current._near_view_target_ids(objects, LABELS), [5])
                for now in (1500, 1550, 1600):
                    self.submit(current, objects, now)
                self.assertEqual(current._zone_load_recheck_source, "before06")
                self.assertEqual(current._zone_load_recheck,
                                 "wait06tx" if source == "before06" else "wait05tx")

    def test_runtime_package_includes_shared_geometry_and_regressions(self):
        from pathlib import Path
        manifest = Path(__file__).with_name("app.yaml").read_text(encoding="utf8")
        self.assertIn("  - tracking_roi.py", manifest)
        self.assertIn("  - test_tracking_roi.py", manifest)


if __name__ == "__main__":
    unittest.main()
