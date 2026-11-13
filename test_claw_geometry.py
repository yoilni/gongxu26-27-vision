"""Production vision-to-MCU geometry gating for a front-facing claw."""
from types import SimpleNamespace
from unittest.mock import Mock, patch

from test_vision_rules import LABELS, SingleFrameTrackingScenario
import app_config as config
import serial_controller as s
from vision_overlay import make_bottom_center_roi


def target(x, y, w, h, class_id=5):
    return SimpleNamespace(x=x - w / 2, y=y - h / 2, w=w, h=h,
                           class_id=class_id, score=0.95)


class ClawGeometryTests(SingleFrameTrackingScenario):
    def make(self, mode, class_id=5):
        with patch.object(config, "UART_COORDINATE_ENABLED", False), \
             patch.object(config, "UART_TRACKING_ROI_SHAPE", "triangle"):
            current = s.VisionSerialController(
                576, 448, make_bottom_center_roi(576, 448), claw_geometry=True
            )
        current.mode, current.target_label = mode, LABELS[class_id]
        return current

    def submit(self, current, now, obj):
        with patch.object(s, "_now_ms", return_value=now), \
             patch.object(config, "UART_TRACKING_ROI_SHAPE", "triangle"):
            return current.submit([obj], LABELS, raw_objects=[obj])

    def ref(self, current):
        with patch.object(config, "UART_TRACKING_ROI_SHAPE", "triangle"):
            return current._reference_point(current.mode)

    def test_only_same_frame_large_and_centered_box_can_signal_capture(self):
        for dx, width, height, expected_id in ((0, 40, 40, 0x85),
                                                (0, 80, 80, 0xC5),
                                                (45, 110, 100, 0xE5),
                                                (0, 110, 100, None)):
            with self.subTest(dx=dx, width=width):
                current = self.make(s.MODE_FINAL_ROI_TRACK)
                x, y = self.ref(current)
                self.submit(current, 1000, target(x + dx, y + 8, width, height))
                packet = current._next_outgoing_packet(1000)
                if expected_id is None:
                    self.assertEqual(packet, s.build_event_packet(0x64))
                else:
                    self.assertEqual(packet, s.build_coordinate_packet(expected_id, dx, 8))
                    self.assertIsNone(current._next_outgoing_packet(1050))

    def test_close_and_centered_target_advances_search_with_large_y_error(self):
        for mode, event in ((s.CMD_SEARCH, 0x04), (s.CMD_TRACK, 0x14)):
            with self.subTest(mode=mode):
                current = self.make(mode)
                x, y = self.ref(current)
                self.submit(current, 1000, target(x, y - 110, 110, 100))
                self.assertEqual(current._next_outgoing_packet(1000), s.build_event_packet(event))

    def test_qualified_red_keeps_three_frames_and_tall_casualty_handshake(self):
        current = self.make(s.MODE_FINAL_ROI_TRACK, 4)
        current._serial = SimpleNamespace(write=Mock(side_effect=len))
        x, _ = self.ref(current)
        red = target(x, 385, 90, 140, 4)
        for now in (1000, 1010):
            self.submit(current, now, red)
            self.assertEqual(current._next_outgoing_packet(now)[2], 0xE4)
        self.submit(current, 1020, red)
        capture = current._next_outgoing_packet(1020)
        self.assertEqual(capture, s.build_event_packet(0x64))
        with patch.object(s, "_now_ms", return_value=1020):
            current._write_packet(capture, current._outgoing_generation)
        self.assertEqual(current._next_outgoing_packet(1070), s.build_event_packet(0x54))
