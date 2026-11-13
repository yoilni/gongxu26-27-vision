"""03/04 ignore already-framed cargo;24 entry34 and full-image05 remain."""

import unittest
from unittest.mock import patch

from test_vision_rules import LABELS, block, controller, SingleFrameTrackingScenario
import app_config as config
import serial_controller as s


class SearchRoiExclusionTests(SingleFrameTrackingScenario):
    def setUp(self):
        quiet = patch.object(config, "VISION_DIAGNOSTICS_ENABLED", False)
        quiet.start()
        self.addCleanup(quiet.stop)

    def make(self, class_id=5, mode=s.CMD_SEARCH):
        current = controller(mode)
        current.tracking_roi = [0, 299, 576, 149]
        current.target_label = LABELS[class_id]
        return current

    def submit(self, current, objects, now=1000, raw=None):
        with patch.object(s, "_now_ms", return_value=now):
            return current.submit(objects, LABELS, raw_objects=objects if raw is None else raw)

    def test_initial_acquisition_excludes_every_target_class(self):
        for class_id in (3, 4, 5, 6):
            with self.subTest(class_id=class_id):
                current = self.make(class_id)
                inside = block(288, 400, class_id)
                self.assertIsNone(self.submit(current, [inside]))
                self.assertIsNone(current._latest_packet)
                self.assertIsNone(current._search_target_lock)
                outside = block(100, 270, class_id)
                self.assertIs(self.submit(current, [inside, outside], 1010), outside)
                self.assertEqual(current._latest_packet[2], class_id)

    def test_center_boundary_not_box_overlap_and_custom_roi(self):
        for roi, x, y, excluded in (
            ([0, 299, 576, 149], 100, 299, True),
            ([0, 299, 576, 149], 100, 298.9, False),
            ([0, 299, 576, 149], 0, 350, True),
            ([0, 299, 576, 149], 576, 350, False),
            ([0, 299, 576, 149], 100, 448, False),
            ([100, 300, 200, 100], 100, 300, True),
            ([100, 300, 200, 100], 99.9, 350, False),
            ([100, 300, 200, 100], 300, 350, False),
            (None, 288, 400, False),
        ):
            with self.subTest(roi=roi, x=x, y=y):
                current = self.make()
                current.tracking_roi = roi
                target = block(x, y)
                selected = self.submit(current, [target])
                self.assertIs(selected, None if excluded else target)

    def test_raw_locked_target_inside_roi_cannot_publish_and_outside_can_follow(self):
        current = self.make()
        original = block(100, 270)
        self.assertIs(self.submit(current, [original]), original)
        inside = block(100, 320)
        self.assertIsNone(self.submit(current, [], 1010, raw=[inside]))
        self.assertIsNone(current._latest_packet)
        outside = block(150, 270)
        self.assertIs(self.submit(current, [], 1020, raw=[inside, outside]), outside)
        self.assertIsNotNone(current._latest_packet)

    def test_inside_only_confirms_e3_and_recovery_does_not_bypass_roi(self):
        current = self.make()
        inside = block(288, 350)
        for index in range(config.UART_SEARCH_NO_TARGET_FRAMES):
            self.assertIsNone(self.submit(current, [inside], 1000 + index * 20))
        self.assertEqual(current._next_outgoing_packet(1100),
                         s.build_event_packet(s.EVENT_SEARCH_NO_TARGET))
        with patch.object(s, "_now_ms", return_value=1200):
            current.process_received_bytes(s.build_command_packet(s.CMD_SEARCH_SWEEP_DONE))
        self.assertIsNone(self.submit(current, [inside], 1210))
        self.assertIsNone(current._latest_packet)
        outside = block(120, 270)
        self.assertIs(self.submit(current, [inside, outside], 1220), outside)
        self.assertIsNotNone(current._latest_packet)

    def test_wait04_and_actual_mcu04_both_exclude_roi_cargo(self):
        current = self.make(mode=s.MODE_WAIT_04_HANDSHAKE)
        inside = block(288, 350)
        current._track_target_lock = current._final_target_identity(inside)
        self.assertIsNone(self.submit(current, [], raw=[inside]))
        self.assertIsNone(current._latest_packet)
        with patch.object(s, "_now_ms", return_value=1100):
            current.process_received_bytes(s.build_command_packet(s.CMD_TRACK))
        self.assertIsNone(self.submit(current, [], 1110, raw=[inside]))
        self.assertIsNone(current._latest_packet)

    def test_mixed51_search_excludes_front_half_as_well_as_loaded_inner_half(self):
        current = self.make()
        current._post07_started = True
        current.target_label = s.TARGET_BLACK_OR_GREEN
        front = block(200, 320, 5)
        inner = block(450, 400, 6)
        outside = block(20, 270, 5)
        self.assertIs(self.submit(current, [front, inner, outside]), outside)
        self.assertIsNone(self.submit(current, [], 1010, raw=[front, inner]))
        self.assertIsNone(current._latest_packet)

    def test04_excludes_roi_while05_still_counts_full_image(self):
        inside = block(100, 380)
        for mode in (s.CMD_TRACK, s.MODE_FINAL_ROI_TRACK):
            with self.subTest(mode=mode):
                current = self.make(mode=mode)
                selected = self.submit(current, [inside])
                if mode == s.MODE_FINAL_ROI_TRACK:
                    self.assertIs(selected, inside)
                    self.assertTrue(current._final_roi_sent)
                    self.assertIsNone(current._latest_packet)
                else:
                    self.assertIsNone(selected)
                    self.assertIsNone(current._latest_packet)
        current = self.make(mode=s.MODE_NEAR_VIEW_CHECK)
        current._near_view_started_ms = 0
        self.submit(current, [inside])
        self.assertEqual(current._near_view_stable_frames, 1)


if __name__ == "__main__":
    unittest.main()
