"""Filtered YOLO coordinates sent by a non-blocking UART worker.

Packet layout, little-endian and fixed at 49 bytes::

    AB BA                         header
    01                            protocol version
    SS                            sequence, rolls over at 255
    04                            record count
    00                            reserved
    four x <BBhhhh> records       code, flags, x, y, vx, vy
    CC                            sum(payload) & 0xff
    CD DC                         tail

Record codes are 0=blue, 1=red, 2=green, 3=black triangle. ``flags`` bit 0
means the coordinate is valid and bit 1 means it is currently predicted.
Coordinates are pixels and velocities are pixels per second.
"""

import math
import struct
import threading
import time as python_time

from maix import comm, uart

import app_config as config


TARGET_CODES = {
    "sqareblue": 0,
    "sqarered": 1,
    "sqareredgreen": 2,
    "triangualrblack": 3,
}
PACKET_HEADER = b"\xAB\xBA"
PACKET_TAIL = b"\xCD\xDC"


def _now_ms():
    return int(python_time.monotonic() * 1000.0)


def _clamp_int16(value):
    return max(-32768, min(32767, int(round(value))))


class AdaptiveAlphaBetaFilter:
    """Low-cost constant-velocity filter with fast response to large motion."""

    def __init__(self):
        self.x = None
        self.y = None
        self.vx = 0.0
        self.vy = 0.0
        self.last_update_ms = None
        self.last_measurement_ms = None

    def update(self, measured_x, measured_y, now_ms):
        if (
            self.x is None
            or self.last_measurement_ms is None
            or now_ms - self.last_measurement_ms >= config.UART_FILTER_RESET_MS
        ):
            self.x = float(measured_x)
            self.y = float(measured_y)
            self.vx = 0.0
            self.vy = 0.0
            self.last_update_ms = now_ms
            self.last_measurement_ms = now_ms
            return

        dt = (now_ms - self.last_update_ms) * 0.001
        dt = max(0.005, min(0.100, dt))
        predicted_x = self.x + self.vx * dt
        predicted_y = self.y + self.vy * dt
        error_x = float(measured_x) - predicted_x
        error_y = float(measured_y) - predicted_y
        error = math.hypot(error_x, error_y)

        motion = min(1.0, error / config.UART_FILTER_FAST_ERROR_PX) ** 2
        alpha = config.UART_FILTER_ALPHA_SLOW + (
            config.UART_FILTER_ALPHA_FAST - config.UART_FILTER_ALPHA_SLOW
        ) * motion
        beta = config.UART_FILTER_BETA_SLOW + (
            config.UART_FILTER_BETA_FAST - config.UART_FILTER_BETA_SLOW
        ) * motion

        self.x = predicted_x + alpha * error_x
        self.y = predicted_y + alpha * error_y
        self.vx += beta * error_x / dt
        self.vy += beta * error_y / dt
        self.last_update_ms = now_ms
        self.last_measurement_ms = now_ms

    def snapshot(self, now_ms, frame_width, frame_height):
        if self.x is None or self.last_measurement_ms is None:
            return False, False, 0, 0, 0, 0

        age_ms = max(0, now_ms - self.last_measurement_ms)
        if age_ms > config.UART_FILTER_RESET_MS:
            return False, False, 0, 0, 0, 0

        prediction_ms = min(age_ms, config.UART_FILTER_PREDICTION_MS)
        dt = prediction_ms * 0.001
        x = self.x + self.vx * dt
        y = self.y + self.vy * dt
        x = max(0.0, min(float(frame_width - 1), x))
        y = max(0.0, min(float(frame_height - 1), y))
        return (
            True,
            age_ms > 0,
            _clamp_int16(x),
            _clamp_int16(y),
            _clamp_int16(self.vx),
            _clamp_int16(self.vy),
        )


def build_coordinate_packet(records, sequence):
    """Encode four ordered target records into one self-delimiting packet."""
    payload = bytearray(
        struct.pack(
            "<BBBB",
            config.UART_PROTOCOL_VERSION,
            sequence & 0xFF,
            len(records),
            0,
        )
    )
    for code, valid, predicted, x, y, vx, vy in records:
        flags = (1 if valid else 0) | (2 if predicted else 0)
        payload.extend(
            struct.pack("<BBhhhh", code, flags, x, y, vx, vy)
        )
    checksum = sum(payload) & 0xFF
    return PACKET_HEADER + bytes(payload) + bytes([checksum]) + PACKET_TAIL


class CoordinateSender:
    """Accept latest detections quickly and send filtered snapshots at 100 Hz."""

    def __init__(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self._filters = {
            label: AdaptiveAlphaBetaFilter() for label in TARGET_CODES
        }
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._serial = None
        self._worker = None
        self._sequence = 0
        self.enabled = False

        if config.UART_COORDINATE_ENABLED:
            self._open_and_start()

    def _open_and_start(self):
        listener_removed = False
        try:
            listener_removed = bool(comm.rm_default_comm_listener())
            self._serial = uart.UART(
                config.UART_PORT,
                config.UART_BAUDRATE,
            )
            self._worker = threading.Thread(
                target=self._send_loop,
                name="coordinate-uart",
                daemon=True,
            )
            self.enabled = True
            self._worker.start()
            print(
                "Coordinate UART started:",
                config.UART_PORT,
                "{} baud, {} Hz".format(
                    config.UART_BAUDRATE,
                    config.UART_SEND_RATE_HZ,
                ),
            )
        except Exception as exc:
            self.enabled = False
            self._serial = None
            print("Coordinate UART unavailable:", exc)
            if listener_removed:
                try:
                    comm.add_default_comm_listener()
                except Exception:
                    pass

    def submit(self, objects, labels):
        """Update filters from the best current detection of each class."""
        if not self.enabled:
            return

        selected = {}
        for obj in objects:
            class_id = int(obj.class_id)
            if class_id < 0 or class_id >= len(labels):
                continue
            label = labels[class_id]
            if label not in TARGET_CODES:
                continue
            previous = selected.get(label)
            if previous is None or float(obj.score) > float(previous.score):
                selected[label] = obj

        now_ms = _now_ms()
        with self._lock:
            for label, obj in selected.items():
                center_x = float(obj.x) + float(obj.w) * 0.5
                center_y = float(obj.y) + float(obj.h) * 0.5
                self._filters[label].update(center_x, center_y, now_ms)

    def _make_records(self, now_ms):
        records = []
        with self._lock:
            for label, code in sorted(
                TARGET_CODES.items(), key=lambda item: item[1]
            ):
                snapshot = self._filters[label].snapshot(
                    now_ms,
                    self.frame_width,
                    self.frame_height,
                )
                records.append((code,) + snapshot)
        return records

    def _send_loop(self):
        interval = 1.0 / max(1, config.UART_SEND_RATE_HZ)
        next_send = python_time.monotonic()
        while not self._stop_event.is_set():
            now_ms = _now_ms()
            packet = build_coordinate_packet(
                self._make_records(now_ms),
                self._sequence,
            )
            self._sequence = (self._sequence + 1) & 0xFF
            try:
                sent = self._serial.write(packet)
                if sent < 0:
                    print("Coordinate UART write failed:", sent)
            except Exception as exc:
                print("Coordinate UART write exception:", exc)
                self._stop_event.set()
                break

            next_send += interval
            now = python_time.monotonic()
            if next_send < now:
                next_send = now
            self._stop_event.wait(max(0.0, next_send - now))

    def close(self):
        self.enabled = False
        self._stop_event.set()
        if self._worker is not None:
            self._worker.join(timeout=1.0)
            self._worker = None
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception as exc:
                print("Coordinate UART close failed:", exc)
            self._serial = None
        print("Coordinate UART stopped")
