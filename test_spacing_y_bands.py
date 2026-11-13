"""Nearby red checks all Y; distant red/blue use bands, green/black all Y."""

import unittest
from unittest.mock import patch

from test_vision_rules import LABELS, block, controller, SingleFrameTrackingScenario
import app_config as config
import serial_controller as s
from debug_target_preview import DebugTargetPreview
from target_selector import spacing_neighbor_in_y_scope, spacing_y_band


class SpacingYBandTests(SingleFrameTrackingScenario):
    def setUp(self):
        quiet = patch.object(config, "VISION_DIAGNOSTICS_ENABLED", False)
        quiet.start()
        self.addCleanup(quiet.stop)

    def submit_at(self, current, now, objects):
        with patch.object(s, "_now_ms", return_value=now):
            return current.submit(objects, LABELS, raw_objects=objects)

    def command_at(self, current, now, command):
        with patch.object(s, "_now_ms", return_value=now):
            current.process_received_bytes(s.build_command_packet(command))

    def test_exact_band_boundaries_use_box_center(self):
        for y, band in ((0, 0), (90, 0), (90.01, 1), (100, 1),
                        (100.01, 2), (150, 2), (150.01, 3),
                        (220, 3), (220.01, 4), (290, 4),
                        (290.01, 5), (448, 5)):
            with self.subTest(y=y):
                self.assertEqual(spacing_y_band(y), band)

    def test_red250_green200_dx100_are_neighbors_in_both_directions(self):
        red, green = block(200, 250, 4), block(300, 200, 5)
        self.assertFalse(s.VisionSerialController._matches_final_primary_spacing(
            red, [red, green]))
        self.assertFalse(s.VisionSerialController._matches_final_primary_spacing(
            green, [red, green]))

    def test_all_class_pairs_crossing_each_boundary_use_target_class_scope(self):
        for boundary in (90, 100, 150, 220, 290):
            for target_id in (3, 4, 5, 6):
                for other_id in (3, 4, 5, 6):
                    with self.subTest(boundary=boundary, target=target_id, other=other_id):
                        target = block(200, boundary, target_id)
                        other = block(200, boundary + .01, other_id)
                        for candidate in (target, other):
                            center_y = float(candidate.y) + float(candidate.h) * .5
                            ignores_band = (int(candidate.class_id) in (5, 6)
                                            or (int(candidate.class_id) == 4 and center_y > 224))
                            self.assertEqual(s.VisionSerialController._matches_final_primary_spacing(
                                candidate, [target, other]), not ignores_band)

    def test_red_image_midpoint_is_strict_and_uses_center_not_box_edge(self):
        for target_y, neighbor_y, expected in (
            (223.99, 200, True), (224, 200, True), (224.01, 200, False),
            (224, 230, False), (224.01, 230, False),
        ):
            for neighbor_id in (3, 4, 5, 6):
                with self.subTest(y=target_y, neighbor_y=neighbor_y, neighbor=neighbor_id):
                    target = block(200, target_y, 4)
                    neighbor = block(300, neighbor_y, neighbor_id)
                    self.assertEqual(current_spacing(target, [target, neighbor]), expected)

    def test_distant_red_only_checks_same_band_but_checks_every_neighbor(self):
        target = block(200, 200, 4)
        cross = block(201, 250, 5)
        far = block(400, 180, 6)
        close = block(300, 160, 3)
        self.assertTrue(current_spacing(target, [target, cross, far]))
        for neighbors in ([cross, far, close], [close, far, cross]):
            self.assertFalse(current_spacing(target, [target] + neighbors))
        self.assertTrue(current_spacing(target, [target]))

    def test_red_uses_actual_image_midpoint_not_roi_midpoint(self):
        target, neighbor = block(200, 250, 4), block(300, 200, 5)
        self.assertFalse(spacing_neighbor_in_y_scope(4, 250, 200, frame_center_y=300))
        self.assertTrue(spacing_neighbor_in_y_scope(4, 250, 200, frame_center_y=224))
        self.assertTrue(s.VisionSerialController._matches_final_primary_spacing(
            target, [target, neighbor], frame_center_y=300))
        for height, expected in ((448, None), (600, target)):
            with self.subTest(height=height):
                current = controller(s.CMD_TRACK)
                current.frame_height = height
                current.target_label = LABELS[4]
                self.assertIs(current._select_final_roi_target(
                    [target, neighbor], LABELS, LABELS[4], 1000), expected)
                _, neighbors, _, _ = current.final_track_debug_objects([target, neighbor], LABELS)
                self.assertEqual(neighbors, [neighbor] if height == 448 else [])
                preview = DebugTargetPreview(height, tracking_roi=current.tracking_roi)
                for _ in range(3):
                    _, neighbors, selected, _ = preview.select(
                        [], LABELS, [target, neighbor], LABELS[4])
                self.assertEqual(neighbors, [neighbor] if height == 448 else [])
                self.assertIs(selected, expected)

    def test_same_band_keeps_normal_and_red_dx_thresholds_inclusive(self):
        for y in (80, 95, 125, 180, 250, 330):
            for target_id in (3, 4, 5, 6):
                for other_id in (3, 4, 5, 6):
                    red_pair = target_id == 4 or other_id == 4
                    threshold, _ = s.VisionSerialController._final_spacing_limits(y, red_pair)
                    target = block(100, y, target_id)
                    for dx in (0, threshold, threshold + .01):
                        with self.subTest(y=y, target=target_id, other=other_id, dx=dx):
                            other = block(100 + dx, y + 1, other_id)
                            self.assertEqual(s.VisionSerialController._matches_final_primary_spacing(
                                target, [target, other]), dx > threshold)

    def test_blue_same_band_is_not_a_fixed_absolute_dy_limit(self):
        target = block(100, 151, 3)
        same_band = block(200, 220, 5)  # dy69 still in (150,220]
        cross_band = block(200, 221, 5)
        self.assertFalse(s.VisionSerialController._matches_final_primary_spacing(
            target, [target, same_band]))
        self.assertTrue(s.VisionSerialController._matches_final_primary_spacing(
            target, [target, cross_band]))

    def test_blue_every_same_band_neighbor_is_checked_among_cross_band_objects(self):
        target = block(200, 250, 3)
        cross = block(201, 200, 5)
        far = block(450, 240, 6)
        near = block(300, 260, 3)
        self.assertTrue(s.VisionSerialController._matches_final_primary_spacing(
            target, [target, cross, far]))
        self.assertFalse(s.VisionSerialController._matches_final_primary_spacing(
            target, [target, cross, far, near]))
        self.assertTrue(s.VisionSerialController._matches_final_primary_spacing(target, [target]))

    def test_shared_selector_acquires_blue_with_only_cross_band_neighbor(self):
        target, other = block(200, 250, 3), block(300, 200, 5)
        for mode in (s.CMD_TRACK, s.MODE_FINAL_ROI_TRACK):
            with self.subTest(mode=mode):
                current = controller(mode)
                current.target_label = LABELS[3]
                self.assertIs(self.submit_at(current, 1000, [target, other]), target)
                self.assertEqual(current._latest_packet[2], 3)

    def test_blue14_and22_decisions_keep_same_band_rule(self):
        target, other = block(200, 250, 3), block(300, 200, 5)
        for command in (s.CMD_TRACK_CENTER_DONE, s.CMD_ARRANGE_LEFT_DONE):
            with self.subTest(command=command):
                current = controller(s.MODE_WAIT_14_HANDSHAKE
                                     if command == s.CMD_TRACK_CENTER_DONE
                                     else s.MODE_ARRANGE_LEFTMOST)
                current.target_label = LABELS[3]
                current._post07_started = True
                self.command_at(current, 1000, command)
                self.submit_at(current, 1010, [target, other])
                self.assertEqual(current.mode, s.MODE_FINAL_ROI_TRACK)
                self.assertIn(s.build_event_packet(s.EVENT_SKIP_TO_FINAL_TRACK), current._pending_events)

    def test_first_green14_uses_box_overlap_instead_of_cross_band_dx(self):
        cases = ((command, neighbor_y, event)
                 for command in (s.CMD_TRACK_CENTER_DONE,)
                 for neighbor_y, event in ((150, s.EVENT_SKIP_TO_FINAL_TRACK), (200, s.EVENT_MICRO_SIDE)))
        for command, neighbor_y, event in cases:
            with self.subTest(command=command, neighbor_y=neighbor_y):
                current = controller(s.MODE_WAIT_14_HANDSHAKE
                                     if command == s.CMD_TRACK_CENTER_DONE
                                     else s.CMD_TRACK)
                target, other = block(200, 250, 5), block(200, neighbor_y, 4)
                # SameX is always rejected by oldDX; the new XY expansion
                # passes the distant box but catches the nearerY200 box.
                self.assertFalse(current._matches_final_primary_spacing(target, [target, other]))
                current._track_target_lock = current._final_target_identity(target)
                self.command_at(current, 1000, command)
                for index in range(3):
                    self.submit_at(current, 1010 + 10 * index, [target, other])
                self.assertEqual(current.mode, s.MODE_FINAL_ROI_TRACK if event == s.EVENT_SKIP_TO_FINAL_TRACK
                                 else s.MODE_MICRO_SIDE)
                self.assertEqual(current._pending_events, [s.build_event_packet(event)])

    def test_class_specific_y_scope_at_every_inclusive_dx_threshold(self):
        for target_id in (4, 5, 6):
            for target_y in (80, 95, 125, 180, 250, 330):
                for other_id in (3, 4, 5, 6):
                    limit, _ = s.VisionSerialController._final_spacing_limits(
                        target_y, target_id == 4 or other_id == 4)
                    for other_y in (20, 95, 130, 200, 270, 420):
                        for dx in (limit, limit + .01):
                            with self.subTest(target=target_id, y=target_y,
                                              other=other_id, other_y=other_y, dx=dx):
                                target = block(100, target_y, target_id)
                                neighbor = block(100 + dx, other_y, other_id)
                                skip_distant_red = (target_id == 4 and target_y <= 224
                                                    and spacing_y_band(target_y) != spacing_y_band(other_y))
                                self.assertEqual(current_spacing(target, [target, neighbor]),
                                                 skip_distant_red or dx > limit)

    def test_distant_red14_44_and22_skip_cross_band_neighbor_then_send24(self):
        target, neighbor = block(200, 200, 4), block(300, 250, 5)
        for command in (s.CMD_TRACK_CENTER_DONE, s.CMD_TRACK_RECOVERY_DONE,
                        s.CMD_ARRANGE_LEFT_DONE):
            with self.subTest(command=command):
                current = controller(s.MODE_ARRANGE_LEFTMOST
                                     if command == s.CMD_ARRANGE_LEFT_DONE else
                                     s.MODE_WAIT_14_HANDSHAKE if command == s.CMD_TRACK_CENTER_DONE
                                     else s.CMD_TRACK)
                current.target_label = LABELS[4]
                current._post07_started = True
                self.command_at(current, 1000, command)
                self.submit_at(current, 1010, [target, neighbor])
                self.assertEqual(current.mode, s.MODE_FINAL_ROI_TRACK)
                self.assertIn(s.build_event_packet(s.EVENT_SKIP_TO_FINAL_TRACK), current._pending_events)
                self.assertNotIn(s.build_event_packet(s.EVENT_ROI_READY), current._pending_events)

    def test_distant_red04_and24_use_spacing_priority_with_cross_band_neighbor(self):
        target, neighbor = block(200, 200, 4), block(300, 250, 5)
        for mode in (s.CMD_TRACK, s.MODE_FINAL_ROI_TRACK):
            with self.subTest(mode=mode):
                current = controller(mode)
                current.target_label = LABELS[4]
                current._post07_started = True
                self.assertIs(self.submit_at(current, 1000, [target, neighbor]), target)
                self.assertEqual(current._latest_packet[2], 4)
                self.assertEqual(current._selection_diagnostic_policy,
                                 "merged" if mode == s.CMD_TRACK else "final24_spacing")

    def test_red_green_black_check_every_cross_band_neighbor_and_accept_singletons(self):
        for target_id in (4, 5, 6):
            with self.subTest(target=target_id):
                target = block(100, 250, target_id)
                far = block(400, 80, 4)
                close = block(180, 200, 3)
                self.assertTrue(current_spacing(target, [target, far]))
                self.assertFalse(current_spacing(target, [target, far, close]))
                self.assertFalse(current_spacing(target, [target, close, far]))
                self.assertTrue(current_spacing(target, [target]))

    def test_only_red24_shared_selector_rejects_cross_band_spacing_neighbor(self):
        for target_id in (4, 5, 6):
            with self.subTest(target=target_id):
                target, neighbor = block(200, 250, target_id), block(300, 200, 3)
                current = controller(s.MODE_FINAL_ROI_TRACK)
                current.target_label = LABELS[target_id]
                expected = None if target_id == 4 else target
                self.assertIs(current._select_final_roi_target(
                    [target, neighbor], LABELS, LABELS[target_id], 1000), expected)
                self.assertIs(self.submit_at(current, 1000, [target, neighbor]), target)
                self.assertEqual(current._selection_diagnostic_policy,
                                 "final24_yolo_fallback" if target_id == 4 else "supply_no_spacing24")

    def test_post22_green_cross_band_neighbor_reapproaches04_instead_of24(self):
        current = controller(s.MODE_ARRANGE_LEFTMOST)
        target, neighbor = block(200, 250, 5), block(300, 200, 3)
        self.command_at(current, 1000, s.CMD_ARRANGE_LEFT_DONE)
        self.submit_at(current, 1010, [target, neighbor])
        self.assertEqual(current.mode, s.MODE_WAIT_04_HANDSHAKE)
        self.assertIn(s.build_event_packet(s.EVENT_SEARCH_CENTER_REACHED), current._pending_events)
        self.assertNotIn(s.build_event_packet(s.EVENT_SKIP_TO_FINAL_TRACK), current._pending_events)

    def test_red14_and44_outside_directional_strips_clears_despite_old_dx(self):
        for command in (s.CMD_TRACK_CENTER_DONE, s.CMD_TRACK_RECOVERY_DONE):
            with self.subTest(command=command):
                current = controller(s.MODE_WAIT_14_HANDSHAKE
                                     if command == s.CMD_TRACK_CENTER_DONE else s.CMD_TRACK)
                current.target_label = LABELS[4]
                current._post07_started = True
                target, neighbor = block(200, 250, 4), block(300, 200, 5)
                self.command_at(current, 1000, command)
                self.submit_at(current, 1010, [target, neighbor])
                self.assertFalse(current._matches_final_primary_spacing(target, [target, neighbor]))
                self.assertEqual(current.mode, s.MODE_FINAL_ROI_TRACK)
                self.assertIn(s.build_event_packet(s.EVENT_SKIP_TO_FINAL_TRACK), current._pending_events)
                self.assertNotIn(s.build_event_packet(s.EVENT_ROI_READY), current._pending_events)

    def test_post22_red_cross_band_crowding_reapproaches04_instead_of24(self):
        current = controller(s.MODE_ARRANGE_LEFTMOST)
        current.target_label = LABELS[4]
        current._post07_started = True
        target, neighbor = block(200, 250, 4), block(300, 200, 5)
        self.command_at(current, 1000, s.CMD_ARRANGE_LEFT_DONE)
        self.assertIs(self.submit_at(current, 1010, [target, neighbor]), target)
        self.assertEqual(current.mode, s.MODE_WAIT_04_HANDSHAKE)
        self.assertIn(s.build_event_packet(s.EVENT_SEARCH_CENTER_REACHED), current._pending_events)
        self.assertNotIn(s.build_event_packet(s.EVENT_SKIP_TO_FINAL_TRACK), current._pending_events)
        self.assertEqual(current._latest_packet, s.build_coordinate_packet(4, -88, 26))

    def test_unlocked_red04_cross_band_crowding_requires_merged_acquisition(self):
        current = controller(s.CMD_TRACK)
        current.target_label = LABELS[4]
        current._post07_started = True
        target, neighbor = block(200, 250, 4), block(300, 200, 5)
        self.assertIsNone(self.submit_at(current, 1000, [target, neighbor]))
        self.assertIsNone(current._latest_packet)
        self.assertIsNone(current._track_target_lock)
        self.assertIs(self.submit_at(current, 1010, [target]), target)
        self.assertEqual(current._latest_packet, s.build_coordinate_packet(4, -88, 26))

    def test_locked04_24_keep_following_if_same_band_neighbor_appears(self):
        target, cross = block(200, 250, 4), block(300, 200, 5)
        neighbor = block(300, 240, 5)
        for mode in (s.CMD_TRACK, s.MODE_FINAL_ROI_TRACK):
            with self.subTest(mode=mode):
                current = controller(mode)
                current.target_label = LABELS[4]
                self.assertIs(self.submit_at(current, 1000, [target]), target)
                self.assertFalse(current._matches_final_primary_spacing(target, [target, neighbor]))
                with patch.object(current, "_select_final_roi_target",
                                  side_effect=AssertionError("locked target must not rescreen")):
                    self.assertIs(self.submit_at(current, 1010, [target, neighbor]), target)
                self.assertIsNotNone(current._latest_packet)

    def test_locked_red_green_black03_04_24_do_not_rescreen_new_cross_band_neighbors(self):
        for mode in (s.CMD_SEARCH, s.CMD_TRACK, s.MODE_FINAL_ROI_TRACK):
            for target_id in (4, 5, 6):
                with self.subTest(mode=mode, target=target_id):
                    target = block(200, 250, target_id)
                    neighbor = block(300, 200, 3)
                    current = controller(mode)
                    current.target_label = LABELS[target_id]
                    self.assertIs(self.submit_at(current, 1000, [target]), target)
                    self.assertFalse(current_spacing(target, [target, neighbor]))
                    with patch.object(current, "_select_final_roi_target",
                                      side_effect=AssertionError("locked target must not rescreen")):
                        self.assertIs(self.submit_at(current, 1010, [target, neighbor]), target)
                    self.assertEqual(current._latest_packet[2], target_id)

    def test_red_verbose_diagnostics_report_cross_band_close_neighbors(self):
        target = block(200, 250, 4)
        for other_y, expected in ((200, "dx=100.0<=170"), (230, "dx=100.0<=170")):
            with self.subTest(other_y=other_y):
                current = controller(s.CMD_TRACK)
                current.target_label = LABELS[4]
                with patch.object(config, "VISION_DIAGNOSTICS_ENABLED", True), \
                        patch.object(config, "VISION_DIAGNOSTICS_VERBOSE", True), \
                        patch("builtins.print") as output:
                    self.submit_at(current, 1000, [target, block(300, other_y, 5)])
                self.assertTrue(any("[CANDIDATE]" in str(call) and expected in str(call)
                                    for call in output.call_args_list))

    def test_distant_red_diagnostics_and_preview_only_include_same_band_neighbors(self):
        target = block(200, 200, 4)
        cross, same = block(300, 250, 5), block(310, 160, 6)
        current = controller(s.CMD_TRACK)
        current.target_label = LABELS[4]
        _, neighbors, _, annotations = current.final_track_debug_objects([target, cross, same], LABELS)
        self.assertEqual(neighbors, [same])
        self.assertEqual(annotations, [(same, 110)])
        _, neighbors, _, annotations = DebugTargetPreview().select(
            [], LABELS, [target, cross, same], LABELS[4])
        self.assertEqual(neighbors, [same])
        self.assertEqual(annotations, [(same, 110)])
        for height, expected_close in ((448, False), (380, True)):
            with self.subTest(height=height), \
                    patch.object(config, "VISION_DIAGNOSTICS_ENABLED", True), \
                    patch.object(config, "VISION_DIAGNOSTICS_VERBOSE", True), \
                    patch("builtins.print") as output:
                current = controller(s.CMD_TRACK)
                current.frame_height = height
                current.target_label = LABELS[4]
                self.submit_at(current, 1000, [target, cross])
            candidate_lines = [str(call) for call in output.call_args_list if "[CANDIDATE]" in str(call)]
            self.assertTrue(candidate_lines)
            self.assertEqual(any("dx=100.0<=150" in line for line in candidate_lines), expected_close)

    def test_red_preview_helpers_report_every_close_neighbor_across_y_bands(self):
        target = block(200, 250, 4)
        cross = block(300, 200, 5)
        same = block(350, 230, 6)
        current = controller(s.CMD_TRACK)
        current.target_label = LABELS[4]
        candidates, neighbors, selected, annotations = current.final_track_debug_objects(
            [target, cross], LABELS)
        self.assertEqual(candidates, [target])
        self.assertEqual(neighbors, [cross])
        self.assertEqual(annotations, [(cross, 100)])
        self.assertIsNone(selected)
        _, neighbors, selected, annotations = current.final_track_debug_objects(
            [target, cross, same], LABELS)
        self.assertEqual(neighbors, [cross, same])
        self.assertEqual(annotations, [(cross, 100), (same, 150)])
        self.assertIsNone(selected)
        preview = DebugTargetPreview()
        for index in range(3):
            _, neighbors, selected, _ = preview.select([], LABELS, [target, cross], LABELS[4])
            self.assertEqual(neighbors, [cross])
        self.assertIsNone(selected)
        _, neighbors, selected, annotations = preview.select([], LABELS, [target, cross, same], LABELS[4])
        self.assertEqual(neighbors, [cross, same])
        self.assertEqual(annotations, [(cross, 100), (same, 150)])
        self.assertIsNone(selected)

    def test_supply_verbose_diagnostics_report_cross_band_close_neighbor(self):
        for target_id in (5, 6):
            with self.subTest(target=target_id):
                current = controller(s.CMD_TRACK)
                current.target_label = LABELS[target_id]
                with patch.object(config, "VISION_DIAGNOSTICS_ENABLED", True), \
                        patch.object(config, "VISION_DIAGNOSTICS_VERBOSE", True), \
                        patch("builtins.print") as output:
                    self.submit_at(current, 1000, [block(200, 250, target_id), block(300, 200, 4)])
                self.assertTrue(any("[CANDIDATE]" in str(call) and "dx=100.0<=170" in str(call)
                                    for call in output.call_args_list))

    def test_supply_preview_helpers_include_all_cross_band_neighbors(self):
        for target_id in (5, 6):
            with self.subTest(target=target_id):
                target = block(200, 250, target_id)
                other_supply_id = 5 if target_id == 6 else 6
                neighbors_expected = [block(270, 80, 3), block(300, 200, 4), block(320, 420, other_supply_id)]
                far = block(500, 80, 3)
                objects = [target] + neighbors_expected + [far]
                current = controller(s.CMD_TRACK)
                current.target_label = LABELS[target_id]
                _, neighbors, selected, annotations = current.final_track_debug_objects(objects, LABELS)
                self.assertEqual(neighbors, neighbors_expected)
                self.assertEqual(annotations, list(zip(neighbors_expected, (70, 100, 120))))
                self.assertIsNone(selected)
                _, neighbors, selected, annotations = DebugTargetPreview().select(
                    [], LABELS, objects, LABELS[target_id])
                self.assertEqual(neighbors, neighbors_expected)
                self.assertEqual(annotations, list(zip(neighbors_expected, (70, 100, 120))))
                self.assertIsNone(selected)


def current_spacing(target, objects):
    return s.VisionSerialController._matches_final_primary_spacing(target, objects)


if __name__ == "__main__":
    unittest.main()
