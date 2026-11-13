"""Green/supply14/44 uses YOLO box overlap with the displayed X range."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from test_vision_rules import LABELS, block, controller, SingleFrameTrackingScenario
import app_config as config
import serial_controller as serial
import target_selector as geometry
import vision_overlay as overlay


def box(x, y, w, h, class_id=3):
    return SimpleNamespace(x=x, y=y, w=w, h=h, class_id=class_id, score=.9)


class ExpandedTargetGeometryTests(unittest.TestCase):
    def test_display_and_decision_share_exact_rectangles_and_padding(self):
        self.assertIs(overlay.selected_target_x_padding, geometry.selected_target_x_padding)
        for y in (0, 90, 150, 159.5, 200, 224, 290, 319, 448):
            target = block(288, y)
            original = vars(target).copy()
            native, expanded = geometry.selected_target_rectangles(target, 576, 448)
            img = Mock()
            overlay.draw_selected_target(img, target, LABELS, 576, 448)
            self.assertEqual([list(call.args) for call in img.draw_rect.call_args_list], [native, expanded])
            self.assertEqual(vars(target), original)

    def test_target_expands_but_neighbor_is_not_expanded(self):
        target = block(288, 310)
        _, expanded = geometry.selected_target_rectangles(target, 576, 448)
        self.assertEqual(expanded, [205, 227, 166, 166])
        # Neighbor expansion would overlap, but its native box is outside.
        other = box(400, 300, 20, 20)
        self.assertTrue(geometry.rects_have_area_overlap(
            expanded, geometry.selected_target_rectangles(other, 576, 448)[1]))
        self.assertEqual(geometry.expanded_target_neighbors(target, [target, other], 576, 448), [])

    def test_positive_intersection_is_required_in_both_axes(self):
        target = block(288, 310)
        cases = [
            (box(371, 300, 20, 20), False),  # right edge touches
            (box(185, 300, 20, 20), False),  # left edge touches
            (box(270, 207, 20, 20), False),  # expanded top edge touches
            (box(270, 393, 20, 20), False),  # expanded bottom edge touches
            (box(370, 300, 20, 20), True),   # one-pixel overlap
            (box(186, 300, 20, 20), True),
            (box(270, 208, 20, 20), True),
            (box(270, 392, 20, 20), True),
            (box(278, 100, 20, 20), False), # same X, distant Y
        ]
        for other, expected in cases:
            with self.subTest(other=other):
                self.assertEqual(bool(geometry.expanded_target_neighbors(
                    target, [target, other], 576, 448)), expected)

    def test_all_other_cargo_classes_can_overlap_but_zones_and_self_cannot(self):
        target = block(288, 310)
        others = [box(300, 300, 20, 20, class_id) for class_id in (0, 1, 2, 3, 4, 5, 6, 99)]
        overlaps = geometry.expanded_target_neighbors(target, [target] + others, 576, 448)
        self.assertEqual([obj.class_id for obj in overlaps], [3, 4, 5, 6])

    def test_invalid_and_fully_offscreen_boxes_do_not_create_neighbors(self):
        target = block(10, 310)
        others = [box(-30, 300, 20, 20), box(580, 300, 20, 20),
                  box(0, 300, 0, 20), box(0, 300, -10, 20),
                  box(float("nan"), 300, 20, 20)]
        self.assertEqual(geometry.expanded_target_neighbors(target, [target] + others, 576, 448), [])
        others.append(box(-10, 300, 20, 20))
        self.assertEqual(geometry.expanded_target_neighbors(target, [target] + others, 576, 448), [others[-1]])

    def test_both_axes_use_original_center_y_margin_not_recomputed_expanded_y(self):
        for center_y, padding in ((0, 5), (90, 25), (150, 38), (159.5, 40),
                                   (224, 54), (290, 69), (319, 75), (440, 75)):
            target = block(288, center_y)
            native, expanded = geometry.selected_target_rectangles(target, 576, 448)
            left, top, width, height = native
            expected = [max(0, left - padding), max(0, top - padding),
                        min(576, left + width + padding) - max(0, left - padding),
                        min(448, top + height + padding) - max(0, top - padding)]
            self.assertEqual(expanded, expected)

    def test_y_only_neighbors_need_positive_overlap_with_expanded_not_native_box(self):
        for center_y in (90, 150, 224, 290, 319):
            target = block(288, center_y)
            native, expanded = geometry.selected_target_rectangles(target, 576, 448)
            top, bottom = expanded[1], expanded[1] + expanded[3]
            for other, hit in ((box(278, top - 20, 20, 20), False),
                               (box(278, top - 19, 20, 20), True),
                               (box(278, bottom, 20, 20), False),
                               (box(278, bottom - 1, 20, 20), True)):
                self.assertFalse(geometry.rects_have_area_overlap(native, geometry.clipped_yolo_rect(other, 576, 448)))
                self.assertEqual(bool(geometry.expanded_target_neighbors(target, [target, other], 576, 448)), hit)

    def test_expanding_both_targets_instead_of_native_neighbor_would_be_wrong(self):
        target = block(288, 224)
        other = block(288, 318)
        native, expanded = geometry.selected_target_rectangles(target, 576, 448)
        other_native, other_expanded = geometry.selected_target_rectangles(other, 576, 448)
        self.assertFalse(geometry.rects_have_area_overlap(expanded, other_native))
        self.assertTrue(geometry.rects_have_area_overlap(expanded, other_expanded))
        self.assertEqual(geometry.expanded_target_neighbors(target, [target, other], 576, 448), [])


class FirstGreenOverlapDecisionTests(SingleFrameTrackingScenario):
    def setUp(self):
        for helper in (patch.object(config, "VISION_DIAGNOSTICS_ENABLED", False), patch("builtins.print")):
            helper.start()
            self.addCleanup(helper.stop)

    def begin(self, target, command=serial.CMD_TRACK_CENTER_DONE, after05=False):
        mode = serial.MODE_WAIT_14_HANDSHAKE if command == serial.CMD_TRACK_CENTER_DONE else serial.CMD_TRACK
        current = controller(mode)
        current._first_green_near_view_decided = after05
        current._track_target_lock = current._final_target_identity(target)
        current._last_target_center_x = float(target.x) + float(target.w) * .5
        with patch.object(serial, "_now_ms", return_value=1000):
            current._handle_command(command)
        return current

    def submit(self, current, raw, now=1050, verified=()):
        with patch.object(serial, "_now_ms", return_value=now):
            return current.submit(list(verified), LABELS, raw_objects=list(raw),
                                  frame_generation=current.frame_generation())

    def confirm(self, current, raw, event):
        times = [1050 + index * 50 for index in range(config.UART_FIRST_GREEN_DECISION_CONFIRM_FRAMES)]
        for now in times:
            self.submit(current, raw, now)
            self.assertIsNone(current._latest_packet)
            if now != times[-1]:
                self.assertIsNone(current._next_outgoing_packet(now))
        self.assertEqual(current._next_outgoing_packet(times[-1]), serial.build_event_packet(event))

    def test_14_uses_overlap_not_old_spacing_or_front_blocking(self):
        target = block(288, 310)
        disjoint_y = block(288, 200, 4)
        disjoint_x = block(450, 310, 3)
        # Both were rejected by the old per-band center-dx table.
        self.assertFalse(serial.VisionSerialController._matches_final_primary_spacing(target, [target, disjoint_y]))
        self.assertFalse(serial.VisionSerialController._matches_final_primary_spacing(target, [target, disjoint_x]))
        for command in (serial.CMD_TRACK_CENTER_DONE,):
            current = self.begin(target, command)
            with patch.object(current, "_select_final_roi_target", side_effect=AssertionError("no old merged selector")), \
                    patch.object(current, "_matches_final_primary_spacing", side_effect=AssertionError("no dx table")), \
                    patch.object(current, "_final_candidate_is_blocked", side_effect=AssertionError("no old front gate")):
                self.confirm(current, [target, disjoint_y, disjoint_x], serial.EVENT_SKIP_TO_FINAL_TRACK)
            self.assertEqual(current.mode, serial.MODE_FINAL_ROI_TRACK)
            self.assertEqual(current._final_target_lock, current._final_target_identity(target))

    def test_native_wide_neighbor_overlap_rejects_even_when_centers_pass_old_dx(self):
        target = block(288, 100)
        other = box(324, 90, 200, 20, 4)
        self.assertTrue(serial.VisionSerialController._matches_final_primary_spacing(target, [target, other]))
        for command in (serial.CMD_TRACK_CENTER_DONE,):
            current = self.begin(target, command)
            self.confirm(current, [target, other], serial.EVENT_MICRO_FORWARD)
            self.assertEqual(current.mode, serial.MODE_MICRO_FORWARD)

    def test_every_other_cargo_color_including_extra_green_horizontal_requests32(self):
        target = block(288, 310)
        for class_id in (3, 4, 5, 6):
            with self.subTest(class_id=class_id):
                current = self.begin(target)
                self.confirm(current, [target, box(370, 300, 20, 20, class_id)], serial.EVENT_MICRO_FORWARD)

    def test_horizontal_overlap_across_old_y_band_boundary_requests32(self):
        target, other = block(200, 220), block(230, 230, 4)
        self.assertNotEqual(geometry.spacing_y_band(220), geometry.spacing_y_band(230))
        current = self.begin(target)
        self.confirm(current, [target, other], serial.EVENT_MICRO_FORWARD)

    def test_single_green_always_qualifies_after_existing_confirmation(self):
        for y in (0, 90, 150, 224, 290, 319, 400):
            with self.subTest(y=y):
                target = block(288, y)
                current = self.begin(target)
                self.confirm(current, [target], serial.EVENT_SKIP_TO_FINAL_TRACK)

    def test_overlapped_approached_green_is_not_replaced_by_another_clear_green(self):
        target, other_green = block(100, 310), block(288, 200)
        blocker = box(105, 300, 20, 20, 3)
        current = self.begin(target)
        self.confirm(current, [other_green, target, blocker], serial.EVENT_ROI_READY)
        self.assertIsNone(current._final_target_lock)

    def test_transient_overlap_resets_streak_instead_of_immediate02(self):
        target, blocker = block(288, 310), box(300, 300, 20, 20)
        current = self.begin(target)
        self.submit(current, [target, blocker], 1020)
        self.assertIsNone(current._next_outgoing_packet(1020))
        self.confirm(current, [target], serial.EVENT_SKIP_TO_FINAL_TRACK)

    def test_new_y_only_neighbor_requires_existing_two_frame_confirm_for42(self):
        target = block(288, 224)
        neighbor = box(278, 287, 20, 20, 4)  # Native gap53px; expanded bottom288.
        self.assertFalse(geometry.rects_have_area_overlap(
            geometry.clipped_yolo_rect(target, 576, 448), geometry.clipped_yolo_rect(neighbor, 576, 448)))
        for command in (serial.CMD_TRACK_CENTER_DONE,):
            current = self.begin(target, command)
            self.confirm(current, [target, neighbor], serial.EVENT_MICRO_SIDE)

    def test_shrunken_margin_drops_old_only_horizontal_and_vertical_neighbors(self):
        target = block(288, 224)
        # AtY224: old padding65, new54. Both neighbors intersected the old
        # strips but are beyond the new ones, while still outside tracking ROI.
        neighbors = [box(358, 214, 20, 20, 3), box(278, 294, 20, 20, 4)]
        for command in (serial.CMD_TRACK_CENTER_DONE,):
            current = self.begin(target, command)
            self.confirm(current, [target] + neighbors, serial.EVENT_SKIP_TO_FINAL_TRACK)
            self.assertEqual(current._final_target_lock, current._final_target_identity(target))

    def test_true_raw_absence_still_sends_one_e4_after_existing_timeout(self):
        target = block(288, 310)
        current = self.begin(target)
        self.submit(current, [], 1100)
        self.assertIsNone(current._next_outgoing_packet(1499))
        self.assertEqual(current._next_outgoing_packet(1500), serial.build_event_packet(serial.EVENT_TRACK_NO_TARGET))
        self.assertTrue(current._track_green_recovery_waiting)
        self.assertIsNone(current._next_outgoing_packet(4000))

    def test_overlapping_neighbor_inside_safety_zone_is_still_excluded(self):
        target = block(180, 310)
        other = box(260, 300, 20, 20)
        safety = box(300, 300, 100, 20, 1)  # exclusion starts atX200
        current = self.begin(target)
        self.confirm(current, [target, other, safety], serial.EVENT_SKIP_TO_FINAL_TRACK)

    def test_same_first_green_task_after05_uses_new_rule_without_changing_timing(self):
        target = block(288, 310)
        other = block(288, 200, 3)
        for command in (serial.CMD_TRACK_CENTER_DONE,):
            current = self.begin(target, command, after05=True)
            self.assertIsNone(current._initial_confirmation_started_ms)
            self.submit(current, [target, other])
            self.assertEqual(current._next_outgoing_packet(1050), serial.build_event_packet(serial.EVENT_SKIP_TO_FINAL_TRACK))

    def test_overlap02_retains_original12_then22_handshakes(self):
        target, blocker = block(288, 310), box(290, 300, 20, 20)
        current = self.begin(target)
        self.confirm(current, [target, blocker], serial.EVENT_ROI_READY)
        current._handle_command(serial.CMD_ARRANGE_FIRST_DONE)
        self.assertEqual(current.mode, serial.MODE_ARRANGE_LEFTMOST)
        self.assertEqual(current._next_outgoing_packet(1200), serial.build_event_packet(serial.EVENT_ARRANGE_FIRST_ACK))
        current._handle_command(serial.CMD_ARRANGE_LEFT_DONE)
        self.assertEqual(current.mode, serial.MODE_POST_ARRANGE_CHECK)

    def test03_04_24_and_post22_do_not_start_using_the_new_overlap_filter(self):
        target, blocker = block(288, 200), block(300, 200, 3)
        for mode in (serial.CMD_SEARCH, serial.CMD_TRACK, serial.MODE_FINAL_ROI_TRACK, serial.MODE_POST_ARRANGE_CHECK):
            current = controller(mode)
            with patch.object(serial, "expanded_target_neighbors", side_effect=AssertionError("not a14 decision")):
                self.submit(current, [target, blocker], verified=[target, blocker])
            self.assertNotEqual(current._selection_diagnostic_policy, "first_green_overlap14")

    def test_post07_red_and_supplies_use_directional_overlap_not_old_dx_table(self):
        target, blocker = block(288, 310, 4), block(288, 200, 3)
        current = controller(serial.MODE_WAIT_14_HANDSHAKE)
        current.target_label, current._post07_started = "sqarered", True
        with patch.object(current, "_select_final_roi_target", side_effect=AssertionError("red14 no old dx table")):
            current._handle_command(serial.CMD_TRACK_CENTER_DONE)
            self.submit(current, [target, blocker], verified=[target, blocker])
        self.assertEqual(current._next_outgoing_packet(1050), serial.build_event_packet(serial.EVENT_SKIP_TO_FINAL_TRACK))
        for target_label in ("sqareredgreen", "triangualrblack", serial.TARGET_BLACK_OR_GREEN):
            current = controller(serial.MODE_WAIT_14_HANDSHAKE)
            current.target_label, current._post07_started = target_label, True
            with patch.object(serial, "directional_target_neighbors", wraps=geometry.directional_target_neighbors) as expand:
                current._handle_command(serial.CMD_TRACK_CENTER_DONE)
                self.submit(current, [block(288, 310), blocker])
            expand.assert_called_once()
            self.assertEqual(current._next_outgoing_packet(1050), serial.build_event_packet(serial.EVENT_SKIP_TO_FINAL_TRACK))


    def test_verbose_diagnostics_report_shared_box_and_raw_neighbors_not_old_table(self):
        target, blocker = block(288, 310), box(370, 300, 20, 20)
        current = self.begin(target)
        with patch.object(config, "VISION_DIAGNOSTICS_ENABLED", True), \
                patch.object(config, "VISION_DIAGNOSTICS_VERBOSE", True), patch("builtins.print") as log:
            self.submit(current, [target, blocker])
        messages = [str(call) for call in log.call_args_list]
        self.assertTrue(any("rule=expanded_box_overlap" in msg and "x_padding=73" in msg
                            and "y_padding=73" in msg and "id3" in msg for msg in messages))
        self.assertTrue(any("merged_applied=no" in msg for msg in messages))


class Post07SupplyOverlapDecisionTests(unittest.TestCase):
    def setUp(self):
        for helper in (patch.object(config, "UART_DIRECT_CAPTURE_AFTER_14", False),
                       patch.object(config, "VISION_DIAGNOSTICS_ENABLED", False),
                       patch("builtins.print")):
            helper.start()
            self.addCleanup(helper.stop)

    def begin(self, target_label, target, command=serial.CMD_TRACK_CENTER_DONE, collecting=False):
        current = controller(serial.MODE_WAIT_14_HANDSHAKE if command == serial.CMD_TRACK_CENTER_DONE else serial.CMD_TRACK)
        current.target_label, current._post07_started = target_label, True
        #44 normally follows a task that already passed14. Test both inputs
        #with the same retained target, without changing ordinary lock rules.
        current._post07_supply_tracking = command == serial.CMD_TRACK_RECOVERY_DONE
        current._track_target_lock = current._final_target_identity(target)
        current._last_target_center_x = float(target.x) + float(target.w) * .5
        if collecting:
            current._supply_collect_started_ms = 900
            current._supply_collect_attempts = 1
        with patch.object(serial, "_now_ms", return_value=1000):
            current.process_received_bytes(serial.build_command_packet(command))
        self.assertIsNone(current._initial_confirmation_started_ms)
        self.assertEqual(current._initial_confirmation_anchor, current._final_target_identity(target))
        return current

    def decide(self, current, raw, event, now=1050):
        with patch.object(serial, "_now_ms", return_value=now), \
                patch.object(current, "_select_final_roi_target", side_effect=AssertionError("no old dx table")), \
                patch.object(current, "_final_candidate_is_blocked", side_effect=AssertionError("no old front gate")):
            current.submit([], LABELS, raw_objects=raw, frame_generation=current.frame_generation())
        self.assertEqual(current._next_outgoing_packet(now), serial.build_event_packet(event))
        self.assertIsNone(current._latest_packet)
        self.assertEqual(current._selection_diagnostic_policy, "supply_overlap14")

    def test14_and44_supply51_ignores_green_black_neighbors_only(self):
        for label in ("sqareredgreen", "triangualrblack", serial.TARGET_BLACK_OR_GREEN):
            for class_id in (5, 6):
                for neighbor_id in (3, 4, 5, 6):
                    for command in (serial.CMD_TRACK_CENTER_DONE, serial.CMD_TRACK_RECOVERY_DONE):
                        with self.subTest(label=label, target=class_id, neighbor=neighbor_id, command=command):
                            target = block(288, 310, class_id)
                            current = self.begin(label, target, command)
                            exempt = label == serial.TARGET_BLACK_OR_GREEN and neighbor_id in (5, 6)
                            self.decide(current, [target, box(370, 300, 20, 20, neighbor_id)],
                                        serial.EVENT_SKIP_TO_FINAL_TRACK if exempt else serial.EVENT_MICRO_FORWARD)
                            self.assertEqual(current.mode, serial.MODE_FINAL_ROI_TRACK if exempt else serial.MODE_MICRO_FORWARD)
                            if exempt:
                                self.assertEqual(current._final_target_lock, current._final_target_identity(target))
                            else:
                                self.assertIsNone(current._final_target_lock)

    def test51_all_green_black_neighbor_directions_grab_without02_32_42(self):
        for class_id in (5, 6):
            for command in (serial.CMD_TRACK_CENTER_DONE, serial.CMD_TRACK_RECOVERY_DONE):
                for collecting in (False, True):
                    target = block(288, 224, class_id)
                    horizontal, vertical = block(350, 224, 5), block(288, 280, 6)
                    for neighbors in ([horizontal], [vertical], [horizontal, vertical],
                                      [block(290, 225, 5), block(286, 223, 6)]):
                        with self.subTest(target=class_id, command=command, collecting=collecting,
                                          neighbors=len(neighbors)):
                            current = self.begin(serial.TARGET_BLACK_OR_GREEN, target, command, collecting)
                            self.decide(current, [target] + neighbors, serial.EVENT_SKIP_TO_FINAL_TRACK)
                            self.assertEqual(current._final_target_lock, current._final_target_identity(target))
                            self.assertEqual(current._micro_rounds, 0)
                            if collecting:
                                self.assertEqual(current._supply_collect_started_ms, 900)
                                self.assertEqual(current._supply_collect_attempts, 1)

    def test51_mixed_neighbors_classify_only_foreign_red_or_blue_directions(self):
        target = block(288, 224, 5)
        cases = (([block(350, 224, 6), block(288, 280, 3)], serial.EVENT_MICRO_SIDE),
                 ([block(350, 224, 4), block(288, 280, 6)], serial.EVENT_MICRO_FORWARD),
                 ([block(350, 224, 4), block(288, 280, 3)], serial.EVENT_ROI_READY))
        for command in (serial.CMD_TRACK_CENTER_DONE, serial.CMD_TRACK_RECOVERY_DONE):
            for neighbors, event in cases:
                current = self.begin(serial.TARGET_BLACK_OR_GREEN, target, command)
                self.decide(current, [target] + neighbors, event)

    def test51_ignored_supply_remains_visible_and_cannot_replace_approached_target(self):
        target, neighbor = block(330, 224, 6), block(288, 224, 5)
        current = self.begin(serial.TARGET_BLACK_OR_GREEN, target)
        raw = [neighbor, target]
        geometry_before = [(obj.x, obj.y, obj.w, obj.h, obj.class_id) for obj in raw]
        self.decide(current, raw, serial.EVENT_SKIP_TO_FINAL_TRACK)
        self.assertEqual(current._final_target_lock, current._final_target_identity(target))
        self.assertEqual([(obj.x, obj.y, obj.w, obj.h, obj.class_id) for obj in raw], geometry_before)
        self.assertEqual(current._directional_neighbor_objects(target, raw), [target])

    def test51_scope_requires_post07_and_supply_candidate(self):
        target, neighbor = block(288, 224, 5), block(350, 224, 6)
        current = controller(serial.MODE_WAIT_14_HANDSHAKE)
        current.target_label = serial.TARGET_BLACK_OR_GREEN
        self.assertFalse(current._post07_started)
        self.assertEqual(current._directional_arrangement_event(target, [target, neighbor]), serial.EVENT_MICRO_FORWARD)
        current._post07_started = True
        target.class_id = 4
        self.assertEqual(current._directional_arrangement_event(target, [target, neighbor]), serial.EVENT_MICRO_FORWARD)

    def test51_micro_completion_recheck_uses_the_same_supply_exemption(self):
        target, foreign = block(288, 224, 5), block(350, 224, 3)
        current = self.begin(serial.TARGET_BLACK_OR_GREEN, target)
        current._serial = SimpleNamespace(write=Mock(side_effect=lambda packet: len(packet)))
        self.decide(current, [target, foreign], serial.EVENT_MICRO_FORWARD)
        with patch.object(serial, "_now_ms", return_value=1050):
            current._write_packet(serial.build_event_packet(0x32), current.frame_generation())
        with patch.object(serial, "_now_ms", return_value=1450):
            current.process_received_bytes(serial.build_command_packet(0x32))
        self.decide(current, [target, block(350, 224, 6), block(288, 280, 5)],
                    serial.EVENT_SKIP_TO_FINAL_TRACK, now=1500)
        self.assertEqual(current._final_target_lock, current._final_target_identity(target))

    def test51_verbose_diagnostics_match_exempt_neighbors_not_fake_crowding(self):
        target, supply, foreign = block(288, 224, 5), block(350, 224, 6), block(288, 280, 3)
        current = self.begin(serial.TARGET_BLACK_OR_GREEN, target)
        with patch.object(config, "VISION_DIAGNOSTICS_ENABLED", True), \
                patch.object(config, "VISION_DIAGNOSTICS_VERBOSE", True), patch("builtins.print") as log:
            self.decide(current, [target, supply, foreign], serial.EVENT_MICRO_SIDE)
        messages = [call.args[0] for call in log.call_args_list if call.args and "[CANDIDATE]" in str(call.args[0])]
        target_log = next(message for message in messages if "r0:id5@" in message)
        self.assertIn("horizontal=[none]", target_log)
        self.assertIn("vertical=[r2:id3@", target_log)
        self.assertIn("ignored_supplies=[r1:id6@", target_log)

    def test_eligible_singletons_use_same_padding_at_all_y_and_seed_actual24_lock(self):
        for label in ("sqareredgreen", "triangualrblack", serial.TARGET_BLACK_OR_GREEN):
            for class_id in (5, 6):
                for y in (0, 90, 100, 150, 224, 290, 319, 440):
                    with self.subTest(label=label, class_id=class_id, y=y):
                        target = block(288, y, class_id)
                        current = self.begin(label, target)
                        if label == serial.TARGET_BLACK_OR_GREEN and y >= 403.5:
                            #51 cargo already in the inner half is not a new
                            #target, even when it is the only detection.
                            self.decide(current, [target], serial.EVENT_ROI_READY)
                            self.assertIsNone(current._final_target_lock)
                        else:
                            self.decide(current, [target], serial.EVENT_SKIP_TO_FINAL_TRACK)
                            self.assertEqual(current._final_target_lock, current._final_target_identity(target))

    def test_padding_boundary_y_edges_and_native_neighbor_width(self):
        target = block(288, 310, 6)
        for other, event in ((box(371, 300, 20, 20), serial.EVENT_SKIP_TO_FINAL_TRACK),
                             (box(370, 300, 20, 20), serial.EVENT_MICRO_FORWARD),
                             (box(270, 393, 20, 20), serial.EVENT_SKIP_TO_FINAL_TRACK),
                             (box(270, 392, 20, 20), serial.EVENT_SKIP_TO_FINAL_TRACK),
                             (box(278, 100, 20, 20), serial.EVENT_SKIP_TO_FINAL_TRACK)):
            current = self.begin("triangualrblack", target)
            self.decide(current, [target, other], event)
        target = block(288, 100, 6)
        wide = box(324, 90, 200, 20, 4)
        self.assertTrue(serial.VisionSerialController._matches_final_primary_spacing(target, [target, wide]))
        self.decide(self.begin("triangualrblack", target), [target, wide], serial.EVENT_MICRO_FORWARD)

    def test_approached_supply_is_checked_not_replaced_by_another_clear_supply(self):
        for class_id, other_id in ((5, 5), (6, 6), (5, 6), (6, 5)):
            target, other = block(100, 310, class_id), block(288, 200, other_id)
            neighbor = box(105, 300, 20, 20, 3)
            current = self.begin(serial.TARGET_BLACK_OR_GREEN, target)
            self.decide(current, [other, target, neighbor], serial.EVENT_ROI_READY)

    def test_changed_y_band_does_not_bypass_actual_overlap(self):
        target, neighbor = block(200, 220, 6), block(230, 230, 4)
        current = self.begin("triangualrblack", target)
        self.decide(current, [target, neighbor], serial.EVENT_MICRO_FORWARD)

    def test_new_y_only_neighbor_affects14_and44_for_all_supply_tasks(self):
        for label in ("sqareredgreen", "triangualrblack", serial.TARGET_BLACK_OR_GREEN):
            for class_id in (5, 6):
                for command in (serial.CMD_TRACK_CENTER_DONE, serial.CMD_TRACK_RECOVERY_DONE):
                    for neighbor_y, event in ((287, serial.EVENT_MICRO_SIDE),
                                              (288, serial.EVENT_SKIP_TO_FINAL_TRACK),
                                              (141, serial.EVENT_MICRO_SIDE),
                                              (140, serial.EVENT_SKIP_TO_FINAL_TRACK)):
                        with self.subTest(label=label, class_id=class_id, command=command, neighbor_y=neighbor_y):
                            target = block(288, 224, class_id)
                            neighbor = box(278, neighbor_y, 20, 20, 3)
                            self.decide(self.begin(label, target, command), [target, neighbor], event)

    def test_safety_zone_neighbor_exclusion_and_no_target_action_preserved(self):
        target, neighbor = block(180, 310, 6), box(260, 300, 20, 20, 4)
        safety = box(300, 300, 100, 20, 1)
        self.decide(self.begin("triangualrblack", target), [target, neighbor, safety], serial.EVENT_SKIP_TO_FINAL_TRACK)
        self.decide(self.begin("triangualrblack", target), [], serial.EVENT_ROI_READY)
        self.decide(self.begin("triangualrblack", target), [block(300, 200, 4)], serial.EVENT_ROI_READY)

    def test51_held_inner_roi_cargo_cannot_be_candidate_or_overlap_neighbor(self):
        for class_id in (5, 6):
            target = block(280, 395, class_id)
            cargo = block(285, 405, 11 - class_id)  # loaded ROI startsY403.5.
            current = self.begin(serial.TARGET_BLACK_OR_GREEN, target, collecting=True)
            self.decide(current, [cargo, target], serial.EVENT_SKIP_TO_FINAL_TRACK)
            self.assertEqual(current._final_target_lock, current._final_target_identity(target))
            self.assertEqual(current._supply_collect_started_ms, 900)
            self.assertEqual(current._supply_collect_attempts, 1)
            current = self.begin(serial.TARGET_BLACK_OR_GREEN, cargo, collecting=True)
            self.decide(current, [cargo], serial.EVENT_ROI_READY)

    def test51_collecting_also_arranges_if_new_target_is_blocked(self):
        target, neighbor = block(288, 310, 6), box(370, 300, 20, 20, 3)
        current = self.begin(serial.TARGET_BLACK_OR_GREEN, target, collecting=True)
        self.decide(current, [target, neighbor, block(288, 260, 4), block(288, 430)], serial.EVENT_ROI_READY)
        current._handle_command(serial.CMD_ARRANGE_FIRST_DONE)
        self.assertEqual(current._next_outgoing_packet(1100), serial.build_event_packet(0x12))
        current._handle_command(serial.CMD_ARRANGE_LEFT_DONE)
        self.assertEqual(current.mode, serial.MODE_POST_ARRANGE_CHECK)
        self.assertEqual(current._supply_collect_started_ms, 900)

    def test_unrelated_tracking_phases_never_call_overlap_rule_for_post07_supplies(self):
        for mode in (serial.CMD_SEARCH, serial.CMD_TRACK, serial.MODE_FINAL_ROI_TRACK, serial.MODE_POST_ARRANGE_CHECK):
            for label in ("sqareredgreen", "triangualrblack", serial.TARGET_BLACK_OR_GREEN):
                current = controller(mode)
                current.target_label, current._post07_started = label, True
                current._post07_supply_tracking = True
                target, neighbor = block(200, 250, 6), block(220, 250, 3)
                with patch.object(serial, "expanded_target_neighbors", side_effect=AssertionError("only14/44 decision")), \
                        patch.object(serial, "_now_ms", return_value=1050):
                    current.submit([target, neighbor], LABELS, raw_objects=[target, neighbor])
                self.assertNotEqual(current._selection_diagnostic_policy, "supply_overlap14")

    def test_old_pre14_frame_cannot_make_a_decision_after_handshake(self):
        target = block(288, 310, 6)
        current = self.begin("triangualrblack", target)
        generation = current.frame_generation()
        with patch.object(serial, "_now_ms", return_value=1050):
            self.assertIsNone(current.submit([], LABELS, raw_objects=[target], frame_generation=generation - 1))
        self.assertEqual(current._pending_events, [])
        self.decide(current, [target], serial.EVENT_SKIP_TO_FINAL_TRACK)

    def test32_and42_ignore_unsolicited_completion_without_matching_action(self):
        current = controller(serial.CMD_TRACK)
        generation = current.frame_generation()
        for command in (0x32, 0x42):
            self.assertIn(command, serial.EVENT_NAMES)
            current.process_received_bytes(serial.build_command_packet(command))
            self.assertEqual(current.mode, serial.CMD_TRACK)
            self.assertEqual(current.frame_generation(), generation)
            self.assertEqual(current._pending_events, [])

    def test_verbose_diagnostics_match_expanded_rule_and_skip_old_dx_table(self):
        target, neighbor = block(288, 310, 6), box(370, 300, 20, 20, 4)
        current = self.begin("triangualrblack", target)
        with patch.object(config, "VISION_DIAGNOSTICS_ENABLED", True), \
                patch.object(config, "VISION_DIAGNOSTICS_VERBOSE", True), patch("builtins.print") as log:
            self.decide(current, [target, neighbor], serial.EVENT_MICRO_FORWARD)
        messages = [str(call) for call in log.call_args_list]
        self.assertTrue(any("expanded_box_overlap" in message and "id6" in message for message in messages))
        self.assertTrue(any("merged_applied=no" in message for message in messages))

class FirstGreenTwoFrameDecisionTests(unittest.TestCase):
    """Only initial green14 conclusions change2; target acquisition stays3."""

    def setUp(self):
        for helper in (patch.object(config, "UART_DIRECT_CAPTURE_AFTER_14", False),
                       patch("builtins.print"),
                       patch.object(config, "VISION_DIAGNOSTICS_ENABLED", False)):
            helper.start()
            self.addCleanup(helper.stop)

    def make(self, inherited=True, after05=False, post07=False, class_id=5):
        with patch.object(config, "UART_COORDINATE_ENABLED", False):
            current = serial.VisionSerialController(576, 448, [0, 319, 576, 129])
        current.mode = serial.MODE_WAIT_14_HANDSHAKE
        current.target_label, current._post07_started = LABELS[class_id], post07
        current._first_green_near_view_decided = after05
        target = block(288, 224, class_id)
        if inherited:
            current._track_target_lock = current._final_target_identity(target)
        with patch.object(serial, "_now_ms", return_value=1000):
            current._handle_command(0x14)
        return current, target

    def observe(self, current, objects, now):
        with patch.object(serial, "_now_ms", return_value=now):
            return current.submit(objects, LABELS, raw_objects=objects,
                                  frame_generation=current.frame_generation())

    def test_default2_decides_each_direction_on_second_frame_not_first(self):
        self.assertEqual(config.UART_FIRST_GREEN_DECISION_CONFIRM_FRAMES, 2)
        self.assertEqual(config.UART_NEW_TARGET_CONFIRM_FRAMES, 3)
        self.assertEqual(config.UART_NEAR_VIEW_STABLE_FRAMES, 3)
        for neighbors, event in (([], 0x24), ([block(335, 224, 3)], 0x32),
                                  ([block(288, 270, 3)], 0x42),
                                  ([block(335, 224, 3), block(288, 270, 3)], 0x02)):
            with self.subTest(event=event):
                current, target = self.make()
                self.observe(current, [target] + neighbors, 1050)
                self.assertEqual(current._initial_confirmation_frames, 1)
                self.assertIsNone(current._next_outgoing_packet(1050))
                self.observe(current, [target] + neighbors, 1100)
                self.assertEqual(current._pending_events, [serial.build_event_packet(event)])

    def test_new_unlocked_green_still_requires_three_acquisition_frames_before_two_conclusions(self):
        current, target = self.make(inherited=False)
        for now in (1050, 1100):
            self.observe(current, [target], now)
            self.assertEqual(current._initial_confirmation_frames, 0)
            self.assertEqual(current._pending_events, [])
        self.observe(current, [target], 1150)
        self.assertEqual(current._initial_confirmation_frames, 1)
        self.observe(current, [target], 1200)
        self.assertEqual(current._pending_events, [serial.build_event_packet(0x24)])

    def test_changed_outcome_or_target_and_missing_frame_restart_two_frame_streak(self):
        current, target = self.make()
        self.observe(current, [target, block(335, 224, 3)], 1050)
        self.observe(current, [target], 1100)
        self.assertEqual(current._initial_confirmation_frames, 1)
        self.observe(current, [], 1150)
        self.assertEqual(current._initial_confirmation_frames, 0)
        moved = block(100, 200)
        self.observe(current, [moved], 1200)
        self.assertEqual(current._initial_confirmation_frames, 1)
        self.assertEqual(current._pending_events, [])
        self.observe(current, [moved], 1250)
        self.assertEqual(current._pending_events, [serial.build_event_packet(0x24)])

    def test_after05_and_later_red_keep_single_frame_locked_decision(self):
        for after05, post07, class_id in ((True, False, 5), (False, True, 4)):
            current, target = self.make(after05=after05, post07=post07, class_id=class_id)
            self.observe(current, [target], 1050)
            self.assertEqual(current._pending_events, [serial.build_event_packet(0x24)])


if __name__ == "__main__":
    unittest.main()
