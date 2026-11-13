"""Casualty24 confirms one red in the deep claw area before sending34."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from test_vision_rules import LABELS, block, controller, SingleFrameTrackingScenario
import app_config as config
import serial_controller as s


class RedRoiCaptureTests(SingleFrameTrackingScenario):
    def setUp(self):
        quiet = patch.object(config, "VISION_DIAGNOSTICS_ENABLED", False)
        quiet.start()
        self.addCleanup(quiet.stop)

    def make(self, post07=True, target_label=LABELS[4], mode=s.MODE_FINAL_ROI_TRACK):
        current = controller(mode)
        current.tracking_roi = [0, 299, 576, 149]
        current._post07_started = post07
        current.target_label = target_label
        return current

    def submit_at(self, current, now, objects=(), raw=None, generation=None):
        with patch.object(s, "_now_ms", return_value=now):
            return current.submit(list(objects), LABELS,
                                  raw_objects=list(objects) if raw is None else list(raw),
                                  frame_generation=generation)

    def track_outside(self):
        current = self.make()
        outside = block(100, 270, 4)
        self.assertIs(self.submit_at(current, 1000, [outside]), outside)
        return current, outside

    def write_at(self, current, now, packet, generation=None):
        with patch.object(s, "_now_ms", return_value=now):
            current._write_packet(packet, current.frame_generation() if generation is None else generation)

    def confirm(self, current, target, start=1010):
        for now in (start, start + 10, start + 20):
            self.submit_at(current, now, [target])
        return start + 20

    def test_distinct_raw_red_can_capture_without_replacing_outside_lock(self):
        current, outside = self.track_outside()
        inside = block(288, 400, 4)
        inside.score = .41
        for now in (1010, 1020):
            self.assertIs(self.submit_at(current, now, [outside], raw=[outside, inside]), outside)
            self.assertFalse(current._final_roi_sent)
            self.assertEqual(current._next_outgoing_packet(now), s.build_coordinate_packet(4, -188, -125))
        self.submit_at(current, 1030, [outside], raw=[outside, inside])
        self.assertEqual(current._final_target_lock[:3], (4, 100, 270))
        self.assertTrue(current._final_roi_sent)
        self.assertIsNone(current._latest_packet)
        self.assertEqual(current._next_outgoing_packet(1030), s.build_event_packet(0x34))
        self.assertEqual(current._final_no_target_frames, 0)

    def test_distinct_red_does_not_mix_candidates_or_use_ordinary_roi_entry(self):
        current, outside = self.track_outside()
        shallow = block(288, 360, 4)
        left, right = block(268, 400, 4), block(308, 400, 4)
        for now, cargo in ((1010, shallow), (1020, shallow), (1030, shallow),
                           (1040, left), (1050, right), (1060, left), (1070, right),
                           (1080, right)):
            self.submit_at(current, now, [outside], raw=[outside, cargo])
            self.assertFalse(current._final_roi_sent)
        self.submit_at(current, 1090, [outside], raw=[outside, left, right])
        self.assertTrue(current._final_roi_sent)
        self.assertEqual(current._next_outgoing_packet(1090), s.build_event_packet(0x34))

    def test54_uses_capture_box_ratio_and_only_applies_to_casualties(self):
        handoff, report = s.build_event_packet(0x34), s.build_event_packet(0x54)
        for class_id, height, expected in ((4, 31, report), (4, 30, None), (4, 29, None), (5, 31, None)):
            with self.subTest(class_id=class_id, height=height):
                current = self.make(target_label=LABELS[class_id])
                captured = block(288, 395, class_id)
                captured.y, captured.h = 395 - height * .5, height
                current._serial = SimpleNamespace(write=Mock(side_effect=len))
                if class_id == 4:
                    captured_at = self.confirm(current, captured)
                else:
                    captured_at = 1010
                    self.submit_at(current, captured_at, [captured])
                self.assertEqual(current._next_outgoing_packet(captured_at), handoff)
                self.write_at(current, captured_at, handoff)
                self.assertIsNone(current._next_outgoing_packet(captured_at + 49))
                self.assertEqual(current._next_outgoing_packet(captured_at + 50), expected)
                if expected is not None:
                    self.write_at(current, captured_at + 50, report)

    def test54_waits_for_complete34_and_retries_short_write_without_duplicates(self):
        current = self.make()
        captured = block(288, 395, 4)
        captured.y, captured.h = 375, 40
        handoff, report = s.build_event_packet(0x34), s.build_event_packet(0x54)
        current._serial = SimpleNamespace(write=Mock(side_effect=[2, 5, 2, 5, 5]))
        self.confirm(current, captured, 990)
        self.assertEqual(current._next_outgoing_packet(1010), handoff)
        self.write_at(current, 1010, handoff)
        self.assertIsNone(current._next_outgoing_packet(1060))

        for now in (1200, 1350, 1500):
            self.submit_at(current, now, [captured])
        self.assertEqual(current._next_outgoing_packet(1510), handoff)
        self.write_at(current, 1510, handoff)
        self.submit_at(current, 1520, [])
        for now in (1560, 1561):
            self.assertEqual(current._next_outgoing_packet(now), report)
            self.write_at(current, now, report)
        self.confirm(current, captured, 1980)
        self.assertEqual(current._next_outgoing_packet(2000), handoff)
        self.write_at(current, 2000, handoff)
        self.assertIsNone(current._next_outgoing_packet(2050))
        self.assertEqual([call.args[0] for call in current._serial.write.call_args_list],
                         [handoff, handoff, report, report, handoff])

    def test_single_raw_red_counts_without_reporting_ee_before_capture(self):
        current = self.make()
        current._final_no_target_frames = 10
        inside = block(288, 365, 4)
        self.submit_at(current, 1000, [], raw=[inside])
        self.assertFalse(current._final_roi_sent)
        self.assertEqual(current._next_outgoing_packet(1000), s.build_coordinate_packet(4, 0, -30))
        self.submit_at(current, 1010, [], raw=[inside])
        self.assertEqual(current._next_outgoing_packet(1010), s.build_coordinate_packet(4, 0, -30))
        self.submit_at(current, 1020, [], raw=[inside])
        self.assertTrue(current._final_roi_sent)
        self.assertEqual(current._final_no_target_frames, 0)
        self.assertEqual(current._last_tracked_class_id, 4)
        self.assertIsNone(current._tracking_missing_since_ms)
        self.assertEqual(current._next_outgoing_packet(1020), s.build_event_packet(0x34))
        self.assertIsNone(current._next_outgoing_packet(1021))

    def test_red_roi_center_boundaries_not_detection_box_overlap(self):
        for x, y, inside in ((288, 364.99, False), (288, 365, True),
                             (268, 365, True), (308, 365, True), (308.01, 365, False),
                             (267.99, 405, False), (288, 447, True), (288, 448, False)):
            with self.subTest(x=x, y=y):
                current = self.make()
                cargo = block(x, y, 4)
                now = self.confirm(current, cargo)
                self.assertEqual(current._final_roi_sent, inside)
                packet = current._next_outgoing_packet(now)
                if inside:
                    self.assertEqual(packet, s.build_event_packet(0x34))
                else:
                    self.assertEqual(len(packet), 11)

    def test_custom_or_absent_roi_is_respected(self):
        for roi, x, y, inside in (([300, 300, 100, 100], 350, 390, True),
                                  ([300, 300, 100, 100], 350, 401, False),
                                  (None, 288, 400, False)):
            current = self.make()
            current.tracking_roi = roi
            self.confirm(current, block(x, y, 4))
            self.assertEqual(current._final_roi_sent, inside)

    def test_miss_exit_stale_gap_or_candidate_jump_restarts_three_frame_count(self):
        for gap, break_sample, new_x in ((0, None, 288), (0, block(288, 364, 4), 288),
                                        (201, None, 288), (0, block(268, 395, 4), 308)):
            with self.subTest(gap=gap, sample=break_sample):
                current = self.make()
                initial = break_sample if new_x == 308 else block(288, 395, 4)
                self.submit_at(current, 1000, [initial])
                self.submit_at(current, 1010, [initial])
                if gap == 0 and new_x == 288:
                    self.submit_at(current, 1020, [break_sample] if break_sample is not None else [])
                start = 1030 if not gap else 1010 + gap
                target = block(new_x, 395, 4)
                for now in (start, start + 10):
                    self.submit_at(current, now, [target])
                    self.assertFalse(current._final_roi_sent)
                self.submit_at(current, start + 20, [target])
                self.assertTrue(current._final_roi_sent)

    def test_scope_excludes_first_green_other_tasks_and03_04(self):
        for post07, target_label, class_id, mode in (
            (False, LABELS[5], 5, s.MODE_FINAL_ROI_TRACK),
            (False, LABELS[4], 4, s.MODE_FINAL_ROI_TRACK),
            (True, LABELS[3], 3, s.MODE_FINAL_ROI_TRACK),
            (True, LABELS[5], 5, s.MODE_FINAL_ROI_TRACK),
            (True, LABELS[6], 6, s.MODE_FINAL_ROI_TRACK),
            (True, s.TARGET_BLACK_OR_GREEN, 5, s.MODE_FINAL_ROI_TRACK),
            (True, LABELS[4], 4, s.CMD_SEARCH),
            (True, LABELS[4], 4, s.CMD_TRACK),
        ):
            with self.subTest(post07=post07, label=target_label, mode=mode):
                current = self.make(post07, target_label, mode)
                outside = block(100, 270, class_id)
                self.submit_at(current, 1000, [outside])
                for now in (1010, 1020, 1030):
                    self.submit_at(current, now, [outside], raw=[outside, block(288, 400, 4)])
                self.assertFalse(current._final_roi_sent)
                self.assertNotIn(s.build_event_packet(0x34), current._pending_events)
                self.assertIsNotNone(current._latest_packet)

    def test_wrong_color_roi_objects_never_trigger_red_capture(self):
        for class_id in (3, 5, 6):
            with self.subTest(class_id=class_id):
                current, outside = self.track_outside()
                self.submit_at(current, 1010, [outside], raw=[outside, block(450, 375, class_id)])
                self.assertFalse(current._final_roi_sent)
                self.assertIsNotNone(current._latest_packet)

    def test_safety_zone_side_exclusion_still_applies_to_red_roi_cargo(self):
        current = self.make()
        inside = block(288, 395, 4)
        zone = SimpleNamespace(x=300, y=370, w=60, h=45, class_id=1, score=.9)
        for now in (1010, 1020, 1030):
            self.submit_at(current, now, [inside], raw=[inside, zone])
        self.assertFalse(current._final_roi_sent)
        self.assertIsNone(current._latest_packet)

    def test_repeat34_requires_fresh_cargo_and_resumes_on_exit_or_absence(self):
        for exit_y in (None, 364):
            with self.subTest(exit_y=exit_y):
                current = self.make()
                cargo = block(288, 395, 4)
                self.confirm(current, cargo, 990)
                self.assertEqual(current._next_outgoing_packet(1010), s.build_event_packet(0x34))
                for now in (1150, 1300, 1450):
                    self.submit_at(current, now, [cargo])
                self.assertIsNone(current._next_outgoing_packet(1509))
                self.assertEqual(current._next_outgoing_packet(1510), s.build_event_packet(0x34))
                visible = [] if exit_y is None else [block(288, exit_y, 4)]
                self.submit_at(current, 1520, visible)
                self.assertFalse(current._final_roi_sent)
                packet = current._next_outgoing_packet(1520)
                if visible:
                    self.assertEqual(len(packet), 11)
                else:
                    self.assertIsNone(packet)
                self.assertNotIn(s.build_event_packet(0x34), current._pending_events)

    def test_stale_or_prepared34_cannot_be_written_after_exit_or_stale_frames(self):
        for expired in (False, True):
            with self.subTest(expired=expired):
                current = self.make()
                current._serial = SimpleNamespace(write=Mock(side_effect=lambda packet: len(packet)))
                self.confirm(current, block(288, 395, 4), 990)
                prepared = current._next_outgoing_packet(1010)
                if not expired:
                    self.submit_at(current, 1020, [block(288, 364, 4)])
                self.write_at(current, 1211 if expired else 1020, prepared)
                current._serial.write.assert_not_called()
                self.assertIsNone(current._next_outgoing_packet(1600))
                self.assertFalse(current._final_roi_sent)

    def test_old_outside_coordinate_and_ee_cannot_cross_the34_handoff(self):
        current, outside = self.track_outside()
        old_coordinate = current._next_outgoing_packet(1000)
        current._serial = SimpleNamespace(write=Mock(side_effect=lambda packet: len(packet)))
        now = self.confirm(current, block(288, 395, 4))
        self.write_at(current, now, old_coordinate)
        self.write_at(current, now, s.build_event_packet(s.EVENT_NO_TARGET))
        current._serial.write.assert_not_called()
        self.write_at(current, now, current._next_outgoing_packet(now))
        current._serial.write.assert_called_once_with(s.build_event_packet(0x34))

    def test_old_frame_generation_cannot_capture_after_task_change(self):
        current, outside = self.track_outside()
        old_generation = current.frame_generation()
        with patch.object(s, "_now_ms", return_value=1010):
            current.process_received_bytes(s.build_command_packet(s.CMD_SELECT_GREEN_OBJECT))
        self.assertIsNone(self.submit_at(current, 1020, [outside],
                                        raw=[outside, block(450, 375, 4)], generation=old_generation))
        self.assertFalse(current._final_roi_sent)
        self.assertNotIn(s.build_event_packet(0x34), current._pending_events)


if __name__ == "__main__":
    unittest.main()
