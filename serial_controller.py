"""MaixCAM UART state machine and compact command/result protocol.

Frames have no checksum:

MCU -> CAM command (5 bytes):
    E1 E2 CMD 1E 2E

CAM -> MCU event (5 bytes):
    F1 F2 EVENT 1F 2F

CAM -> MCU coordinate (11 bytes):
    F1 F2 CLASS XH XL XS YH YL YS 1F 2F

CLASS is the actual non-zero model class ID. X/Y are unsigned big-endian
error magnitudes; XS/YS are 01 for zero/positive and 02 for negative.
"""

import struct
import threading
import time as python_time

from maix import comm, uart

import app_config as config
from zone_obstacle_region import zone_obstacle_polygon, point_in_zone_obstacle_polygon
from vision_diagnostics import VisionDiagnostics, coordinate_details
from target_selector import (
    select_nearest_horizontal_target,
    split_error_sign,
    target_center_error,
)


MCU_HEADER = b"\xE1\xE2"
MCU_TAIL = b"\x1E\x2E"
CAM_HEADER = b"\xF1\xF2"
CAM_TAIL = b"\x1F\x2F"
COMMAND_FRAME_LENGTH = 5

# State/mode commands from the overall control scheme.
CMD_SORT = 0x02
CMD_SEARCH = 0x03
CMD_TRACK = 0x04
CMD_SEARCH_SWEEP_DONE = 0x13
CMD_TRACK_RECOVERY_DONE = 0x44
CMD_FINAL_SEARCH_DONE = 0xF1
CMD_ZONE_OBSTACLE_DONE = 0x36
CMD_TRACK_CENTER_DONE = 0x14
CMD_ARRANGE_FIRST_DONE = 0x12
CMD_ARRANGE_LEFT_DONE = 0x22
CMD_ARRANGE_SECOND_DONE = 0x24
CMD_RECHECK_SORT = 0xA5

# Commands 15/25 first enter a full-frame near-view count state. After one
# object remains, event 06 transitions internally to safety-zone searching.
MODE_NEAR_VIEW_CHECK = 0x05
MODE_SEARCH_ZONE = 0x06
MODE_WAIT_04_HANDSHAKE = 0x104
MODE_WAIT_14_HANDSHAKE = 0x114
MODE_ARRANGE_RIGHTMOST = 0x102
MODE_ARRANGE_LEFTMOST = 0x112
MODE_POST_ARRANGE_CHECK = 0x122
MODE_FINAL_ROI_TRACK = 0x24
CMD_SEARCH_RED_ZONE = 0x15
CMD_SEARCH_BLUE_ZONE = 0x25

MODE_COMMANDS = {
    CMD_SORT: "SORT",
    CMD_SEARCH: "SEARCH",
    CMD_TRACK: "TRACK",
    CMD_RECHECK_SORT: "RECHECK_SORT",
}

# Persistent class-selection commands. Black selection starts state 24 directly.
CMD_SELECT_BLUE_OBJECT = 0x01
CMD_SELECT_RED_OBJECT = 0x11
CMD_SELECT_GREEN_OBJECT = 0x21
CMD_SELECT_BLACK_OBJECT = 0x31
CMD_SELECT_BLACK_OR_GREEN = 0x51
TARGET_BLACK_OR_GREEN = "black_or_green"

TARGET_SELECTION_COMMANDS = {
    CMD_SELECT_BLUE_OBJECT: "sqareblue",       # model ID 3
    CMD_SELECT_RED_OBJECT: "sqarered",         # model ID 4
    CMD_SELECT_GREEN_OBJECT: "sqareredgreen",  # model ID 5
    CMD_SELECT_BLACK_OBJECT: "triangualrblack",# model ID 6
    CMD_SELECT_BLACK_OR_GREEN: TARGET_BLACK_OR_GREEN,
}
ZONE_SEARCH_COMMANDS = {
    CMD_SEARCH_RED_ZONE: "redsafety",          # model ID 2
    CMD_SEARCH_BLUE_ZONE: "bluesafety",        # model ID 1
}

# Outgoing events reserved by the scheme.
EVENT_ROI_READY = 0x02
EVENT_SEARCH_CENTER_REACHED = 0x04
EVENT_TRACK_CENTER_REACHED = 0x14
EVENT_ARRANGE_FIRST_ACK = 0x12
EVENT_SKIP_TO_FINAL_TRACK = 0x24
EVENT_FINAL_TARGET_IN_ROI = 0x34
EVENT_TARGET_READY = 0x05
EVENT_ZONE_SEARCH_STARTED = 0x06
EVENT_ZONE_FOUND = 0x16
EVENT_ZONE_CLOSE = 0x26
EVENT_ZONE_OBSTACLE = 0x36
EVENT_SWITCH_TO_RED_TARGET = 0x11
EVENT_ZONE_APPROACH = 0x07
EVENT_SINGLE_GREEN = 0xB5
EVENT_MULTI_GREEN = 0xC5
EVENT_NO_TASK_TARGET = 0xD5
EVENT_ARRANGEMENT_NO_OBJECT = 0xE2
EVENT_SEARCH_NO_TARGET = 0xE3
EVENT_TRACK_NO_TARGET = 0xE4
EVENT_UNCERTAIN = 0xE5
EVENT_NO_TARGET = 0xEE

EVENT_NAMES = {
    EVENT_SWITCH_TO_RED_TARGET: "SWITCH_TO_RED_CASUALTY",
    EVENT_ROI_READY: "ROI_READY_OR_ADJUST",
    EVENT_SEARCH_CENTER_REACHED: "SEARCH_CENTER_REACHED",
    EVENT_TARGET_READY: "NEAR_VIEW_READY",
    EVENT_ZONE_SEARCH_STARTED: "ZONE_SEARCH_STARTED",
    EVENT_ARRANGE_FIRST_ACK: "ARRANGE_FIRST_ACK",
    EVENT_TRACK_CENTER_REACHED: "TRACK_CENTER_REACHED",
    EVENT_ZONE_FOUND: "ZONE_FOUND",
    EVENT_SKIP_TO_FINAL_TRACK: "SKIP_TO_FINAL_TRACK",
    EVENT_ZONE_CLOSE: "ZONE_CLOSE",
    EVENT_ZONE_OBSTACLE: "ZONE_OBSTACLE_WAIT_ACK",
    EVENT_FINAL_TARGET_IN_ROI: "FINAL_TARGET_IN_ROI",
    EVENT_ZONE_APPROACH: "ZONE_APPROACH",
    EVENT_SINGLE_GREEN: "SINGLE_GREEN",
    EVENT_MULTI_GREEN: "MULTI_GREEN",
    EVENT_NO_TASK_TARGET: "NO_TASK_TARGET",
    EVENT_ARRANGEMENT_NO_OBJECT: "POST_22_NO_TARGET",
    EVENT_SEARCH_NO_TARGET: "SEARCH_NO_TARGET",
    EVENT_TRACK_NO_TARGET: "TRACK_NO_TARGET",
    EVENT_UNCERTAIN: "UNCERTAIN",
    EVENT_NO_TARGET: "NO_TARGET",
}


def _now_ms():
    return int(python_time.monotonic() * 1000.0)


def build_command_packet(command):
    """Build a command packet for protocol tests."""
    return MCU_HEADER + bytes([int(command) & 0xFF]) + MCU_TAIL


def build_event_packet(event):
    """Build a camera event packet."""
    return CAM_HEADER + bytes([int(event) & 0xFF]) + CAM_TAIL


def _encode_axis_error(error):
    magnitude, sign = split_error_sign(error)
    magnitude = max(0, min(0xFFFF, int(magnitude)))
    return struct.pack(">H", magnitude) + bytes([sign])


def build_coordinate_packet(class_id, error_x, error_y):
    """Build one 11-byte coordinate packet using the real model class ID."""
    class_id = int(class_id)
    if class_id <= 0 or class_id > 0xFF:
        raise ValueError("class_id must be in range 1..255")
    return (
        CAM_HEADER
        + bytes([class_id])
        + _encode_axis_error(error_x)
        + _encode_axis_error(error_y)
        + CAM_TAIL
    )


class CommandFrameParser:
    """Recover fixed command frames from fragmentation, noise, and sticking."""

    def __init__(self):
        self._buffer = bytearray()

    def feed(self, data):
        if data:
            self._buffer.extend(bytes(data))

        commands = []
        while True:
            start = self._buffer.find(MCU_HEADER)
            if start < 0:
                # Keep E1 if it could be the first byte of a split header.
                if self._buffer and self._buffer[-1] == MCU_HEADER[0]:
                    self._buffer[:] = self._buffer[-1:]
                else:
                    self._buffer.clear()
                break

            if start > 0:
                del self._buffer[:start]
            if len(self._buffer) < COMMAND_FRAME_LENGTH:
                break

            frame = bytes(self._buffer[:COMMAND_FRAME_LENGTH])
            if frame[3:5] == MCU_TAIL:
                commands.append(frame[2])
                del self._buffer[:COMMAND_FRAME_LENGTH]
            else:
                del self._buffer[0]

        return commands


class VisionSerialController:
    """Run the MCU/CAM state machine and publish events or target errors."""

    def __init__(self, frame_width, frame_height, tracking_roi=None):
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.tracking_roi = tracking_roi
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._tx_wakeup = threading.Event()
        self._parser = CommandFrameParser()
        self._serial = None
        self._worker = None
        self._pending_events = []

        self.mode = int(config.UART_DEFAULT_MODE)
        self.target_label = None
        self.zone_label = "bluesafety"
        self._latest_packet = None
        self._latest_submit_ms = 0
        self._latest_submit_sequence = 0
        self._last_coordinate_sequence_sent = -1
        self._last_event_write_ms = None
        self._diagnostics = VisionDiagnostics()
        self._selection_diagnostic_policy = "none"
        self._tx_coordinate_count = 0
        self._last_coordinate_write_ms = None
        self._last_coordinate_log_ms = None
        self._last_coordinate_log_class = None
        self._coordinate_in_flight = None
        self._last_frame_seen_ms = 0
        self._last_io_health_ms = 0
        self._last_no_target_ms = 0
        self._final_search_waiting_ack = False
        self._final_no_target_frames = 0
        self._final_target_lock = None
        self._track_target_lock = None
        self._search_target_lock = None
        self._track_no_target_frames = 0
        self._track_no_target_sent = False
        self._track_green_recovery_waiting = False
        self._last_search_no_target_ms = 0
        self._search_no_target_frames = 0
        self._search_no_target_initial_sent = False
        self._search_no_target_acknowledged = False
        self._last_arrangement_no_object_ms = 0
        self._roi_entry_sent = False
        self._track_center_sent = False
        self._final_roi_sent = False
        self._last_final_roi_event_ms = 0
        self._arrangement_object_count = 0
        self._arrangement_no_neighbor_frames = 0
        self._arrangement_ready_ms = 0
        self._initial_arrangement_decision_pending = False
        self._arrangement_02_ack_pending = False
        self._arrangement_02_without_side = False
        self._arrangement_side_target_available = False
        self._last_arrangement_02_event_ms = 0
        self._zone_found_sent = False
        self._zone_close_sent = False
        self._zone_obstacle_waiting = False
        self._zone_obstacle_armed = True
        self._zone_obstacle_frames = 0
        self._zone_obstacle_clear_frames = 0
        self._last_zone_obstacle_event_ms = 0
        self._near_view_started_ms = 0
        self._near_view_adjust_sent = False
        self._near_view_result_sent = False
        self._near_view_stable_bucket = None
        self._near_view_stable_frames = 0
        self._first_green_near_view_decided = False
        self._first_green_both_sides_required = False
        self._last_target_center_x = None
        self._last_target_center_y = None
        self._last_target_area = None
        self._last_target_accept_ms = 0
        self._middle_target_anchor = None
        self._arrangement_target_anchor = None
        self._last_arrangement_debug_ms = 0
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
                target=self._io_loop,
                name="vision-uart",
                daemon=True,
            )
            self.enabled = True
            self._worker.start()
            print(
                "Vision UART started:",
                config.UART_PORT,
                "{} baud, send={}, mode={}".format(
                    config.UART_BAUDRATE,
                    "on_new_frame" if config.UART_SEND_RATE_HZ <= 0
                    else "{} Hz".format(config.UART_SEND_RATE_HZ),
                    MODE_COMMANDS.get(self.mode, "UNKNOWN"),
                ),
            )
            print(
                "Vision UART tracker=middle_side_lock_v3 "
                "state05_v2 arrange02_x200 ee24only_500ms "
                "pre02_count_v1 txlog_v1 rxlog_v1 redboxes_v1 "
                "middle_reacquire14_x_v1 arrange_empty_e2_v1 "
                "near05_ids_v1 final24_front_block_v1 "
                "near05_empty24_v1 final24_center20_v1 "
                "middle_reacquire12_x_v1 arrange_open_loop_v1 "
                "e2_rx24_v1 repeat34_500ms_v1 "
                "target_cmds_01_11_21_31_v1 repeat34_while_centered_v2 "
                "send02_once_v1 pre02_merged24_or02_v1 "
                "final24_first_target_lock_v1 "
                "final24_lock_no_jump_area_gate_v1 "
                "track04_lock_no_jump_area_gate_v1 "
                "search03_to04_inherit_lock_v1 track04_missing5_e4_episode_v1 "
                "vision_diagnostics_v1 "
                "first_green03_04_frame_center_v1 "
                "search03_lock_raw_yolo_follow_v1 "
                "uart_frame_wakeup_fast_v1 "
                "final24_roi_entry34_v1 "
                "final_spacing_all_dx_no_dy_v2 search03_e3_ack13_v2 "
                "search03_missing5_v2 search03_roi_yx_v1 zone16_obstacle36_v1 "
                "exclude_safe_zone_targets_v1 zone36_trapezoid_front100_y100_v5 "
                "zone_red_casualty_right625_v2 final_target_roi_y_nearest_v1 "
                "near05_red_foreign_arrange_v1 near05_black_green_rules_v2 "
                "zone36_accumulated3_v1 black_track_red_priority_v1"
            )
            if config.VISION_DIAGNOSTICS_ENABLED:
                print("[DIAG-CONFIG] interval_ms={} max_objects={} verify_hits={}/{} high_conf={} freshness_ms={} lost04={} lost24={} ROI={} obstacle={}".format(
                    config.VISION_DIAGNOSTICS_INTERVAL_MS, config.VISION_DIAGNOSTICS_MAX_OBJECTS,
                    config.VERIFY_REQUIRED_HITS, config.VERIFY_REQUIRED_HITS_WITHOUT_COLOR,
                    config.VERIFY_HIGH_CONFIDENCE_BYPASS, config.UART_TARGET_FRESHNESS_MS,
                    config.UART_TRACK_LOST_CONFIRM_FRAMES, config.UART_FINAL_NO_TARGET_FRAMES,
                    self.tracking_roi, zone_obstacle_polygon(self.tracking_roi, self.frame_width, self.frame_height),
                ))
        except Exception as exc:
            self.enabled = False
            self._serial = None
            print("Vision UART unavailable:", exc)
            if listener_removed:
                try:
                    comm.add_default_comm_listener()
                except Exception:
                    pass

    def process_received_bytes(self, data):
        """Parse and apply every complete command contained in ``data``."""
        commands = self._parser.feed(data)
        for command in commands:
            frame = build_command_packet(command)
            frame_hex = " ".join("{:02X}".format(value) for value in frame)
            print(
                "Vision UART RX command 0x{:02X} [{}]".format(
                    command,
                    frame_hex,
                )
            )
            if config.VISION_DIAGNOSTICS_ENABLED:
                with self._lock:
                    old_mode, old_target = self.mode, self.target_label
                    track_lock, final_lock = self._track_target_lock, self._final_target_lock
                    search_lock = self._search_target_lock
                print("[RX-CONTEXT] t={} cmd={:02X} previous_state={:X} target={} search_lock={} track_lock={} final_lock={}".format(
                    _now_ms(), command, old_mode, old_target, search_lock, track_lock, final_lock
                ))
            self._handle_command(command)
        return commands

    def _handle_command(self, command):
        changed = False
        description = None
        search_no_target_ack = False
        with self._lock:
            inherit_track_lock = (
                command == CMD_TRACK
                and self.mode == MODE_WAIT_04_HANDSHAKE
                and self._track_target_lock is not None
            )
            if (
                command == CMD_FINAL_SEARCH_DONE
                and self.mode == MODE_FINAL_ROI_TRACK
                and self._final_search_waiting_ack
            ):
                self._final_search_waiting_ack = False
                self._final_no_target_frames = 0
                self._latest_packet = None
                self._latest_submit_ms = 0
                self._last_no_target_ms = 0
                self._last_target_center_x = None
                self._last_target_center_y = None
                self._last_target_area = None
                self._last_target_accept_ms = 0
                self._final_roi_sent = False
                self._last_final_roi_event_ms = 0
                self._pending_events.clear()
                print("Vision UART command 0xF1: search_done=recheck_FINAL_ROI_fresh_image")
                return
            if (
                command == CMD_ARRANGE_SECOND_DONE
                and self._first_green_both_sides_required
                and self.mode in (MODE_ARRANGE_RIGHTMOST, MODE_ARRANGE_LEFTMOST)
            ):
                # This first green load must complete both MCU-timed pushes.
                # A recovery/legacy 24 must not bypass either handshake.
                expected = "12" if self.mode == MODE_ARRANGE_RIGHTMOST else "22"
                print("Vision UART command 0x24: first_green_both_sides wait_RX" + expected)
                return
            if (
                command == CMD_ZONE_OBSTACLE_DONE
                and self.mode == MODE_SEARCH_ZONE
                and self._zone_obstacle_waiting
            ):
                self._zone_obstacle_waiting = False
                self._zone_obstacle_armed = False
                self._zone_obstacle_frames = 0
                self._zone_obstacle_clear_frames = 0
                pending_36 = build_event_packet(EVENT_ZONE_OBSTACLE)
                self._pending_events = [
                    event for event in self._pending_events
                    if event != pending_36
                ]
                # Resume only with a new image after the MCU's maneuver.
                self._latest_packet = None
                self._latest_submit_ms = 0
                self._last_target_center_x = None
                self._last_target_center_y = None
                self._last_target_area = None
                self._last_target_accept_ms = 0
                print("Vision UART command 0x36: bypass_done=resume_zone_coordinates")
                return
            if (
                (
                    command == CMD_TRACK_CENTER_DONE
                    and self.mode == MODE_WAIT_14_HANDSHAKE
                )
                or (
                    command == CMD_TRACK_RECOVERY_DONE
                    and self.mode in (CMD_TRACK, MODE_WAIT_14_HANDSHAKE)
                )
            ):
                # RX14 and recovery completion RX44 both start the same
                # arrangement decision. Preserve X identity and reacquire
                # Y/area after the MCU's movement.
                if self._last_target_center_x is not None:
                    self._middle_target_anchor = (
                        self._last_target_center_x,
                        None,
                        None,
                    )
                else:
                    self._middle_target_anchor = None
                self.mode = MODE_ARRANGE_RIGHTMOST
                description = "merged_check=TX24_if_valid_else_TX02"
                if command == CMD_TRACK_RECOVERY_DONE:
                    description = "recovery_done=" + description
                self._arrangement_object_count = 0
                self._arrangement_no_neighbor_frames = 0
                self._arrangement_ready_ms = 0
                self._arrangement_target_anchor = None
                self._pending_events.clear()
                self._initial_arrangement_decision_pending = True
                self._arrangement_02_ack_pending = False
                self._arrangement_side_target_available = False
                self._last_arrangement_02_event_ms = 0
                changed = True
            elif (
                command == CMD_ARRANGE_FIRST_DONE
                and self.mode == MODE_ARRANGE_RIGHTMOST
            ):
                # Command 12 is the MCU's first arrangement completion
                # handshake. Green target selection now uses command 21.
                self._pending_events.clear()
                # The first push/reverse changes the middle object's apparent
                # Y position and area. Keep only its X identity while the new
                # state reacquires a full anchor.
                if self._middle_target_anchor is not None:
                    middle_x = self._middle_target_anchor[0]
                    self._middle_target_anchor = (middle_x, None, None)
                # Do not skip merely because two boxes are visible: the second
                # box can be another green object that still needs pushing.
                # State 12 waits for MCU command 22 before checking whether
                # a target satisfies the state-24 selection rules.
                self.mode = MODE_ARRANGE_LEFTMOST
                description = "arrange=nearest_left_to_roi_center"
                self._arrangement_target_anchor = None
                self._pending_events.append(
                    build_event_packet(EVENT_ARRANGE_FIRST_ACK)
                )
                changed = True
            elif (
                command == CMD_ARRANGE_LEFT_DONE
                and self.mode in (MODE_ARRANGE_LEFTMOST, MODE_POST_ARRANGE_CHECK)
            ):
                self.mode = MODE_POST_ARRANGE_CHECK
                self._first_green_both_sides_required = False
                self._pending_events.clear()
                description = "mode=POST_ARRANGE_CHECK wait_valid_target_before_TX24"
                changed = True
            elif (
                command == CMD_ARRANGE_SECOND_DONE
                and self.mode
                in (MODE_ARRANGE_RIGHTMOST, MODE_ARRANGE_LEFTMOST)
            ):
                # After CAM reports E2 in either open-loop arrangement stage,
                # MCU replies with 24 to abandon arrangement and start the
                # final ROI coordinate-tracking flow. The normal state-12
                # completion handshake uses the same command.
                self.mode = MODE_FINAL_ROI_TRACK
                description = "mode=FINAL_ROI_TRACK"
                self._pending_events.clear()
                changed = True
            elif command in ZONE_SEARCH_COMMANDS:
                self.mode = MODE_NEAR_VIEW_CHECK
                self.zone_label = ZONE_SEARCH_COMMANDS[command]
                description = "mode=NEAR_VIEW_05 zone=" + self.zone_label
                # A new request starts a fresh handshake. Do not let an event
                # queued by the previous state appear before acknowledgement.
                self._pending_events.clear()
                self._pending_events.append(
                    build_event_packet(EVENT_TARGET_READY)
                )
                changed = True
            elif (
                command == CMD_SEARCH_SWEEP_DONE
                and self.mode == CMD_SEARCH
            ):
                description = "search_sweep_done=repeat_E3_if_missing"
                search_no_target_ack = True
                changed = True
            elif command in MODE_COMMANDS:
                if (
                    self.target_label == "triangualrblack"
                    and command in (CMD_SEARCH, CMD_TRACK)
                ):
                    self.mode = MODE_FINAL_ROI_TRACK
                    self._pending_events.clear()
                    description = "black_target=direct_FINAL_ROI_TRACK"
                else:
                    self.mode = command
                    description = "mode=" + MODE_COMMANDS[command]
                    if inherit_track_lock:
                        description += " inherit_03_lock=id{}".format(
                            self._track_target_lock[0]
                        )
                changed = True
            elif command in TARGET_SELECTION_COMMANDS:
                self._first_green_both_sides_required = False
                self._pending_events.clear()
                self.target_label = TARGET_SELECTION_COMMANDS[command]
                description = "target=" + self.target_label
                if command == CMD_SELECT_BLACK_OBJECT:
                    self.mode = MODE_FINAL_ROI_TRACK
                    self._pending_events.clear()
                    description += " mode=FINAL_ROI_TRACK"
                changed = True
            if changed:
                # New commands start a new selection round. RXF1 returns
                # above and deliberately preserves the locked object.
                self._final_target_lock = None
                if not search_no_target_ack:
                    self._search_target_lock = None
                if not inherit_track_lock:
                    self._track_target_lock = None
                self._track_no_target_frames = 0
                self._final_search_waiting_ack = False
                self._final_no_target_frames = 0
                self._arrangement_02_without_side = False
                if command == CMD_TRACK and self.mode == CMD_TRACK:
                    self._track_no_target_sent = False
                self._track_green_recovery_waiting = False
                self._last_arrangement_no_object_ms = 0
                # Do not reuse data collected for the previous state/class.
                self._latest_packet = None
                self._latest_submit_ms = 0
                if search_no_target_ack:
                    self._search_no_target_initial_sent = True
                    self._search_no_target_acknowledged = True
                    self._last_search_no_target_ms = _now_ms()
                else:
                    self._last_search_no_target_ms = 0
                    self._search_no_target_frames = 0
                    self._search_no_target_initial_sent = False
                    self._search_no_target_acknowledged = False
                self._roi_entry_sent = False
                self._track_center_sent = False
                self._final_roi_sent = False
                self._last_final_roi_event_ms = 0
                if self.mode != MODE_ARRANGE_RIGHTMOST:
                    self._arrangement_object_count = 0
                    self._initial_arrangement_decision_pending = False
                    self._arrangement_02_ack_pending = False
                    self._arrangement_side_target_available = False
                    self._last_arrangement_02_event_ms = 0
                self._arrangement_no_neighbor_frames = 0
                if self.mode not in (
                    MODE_ARRANGE_RIGHTMOST,
                    MODE_ARRANGE_LEFTMOST,
                ):
                    self._arrangement_ready_ms = 0
                    self._middle_target_anchor = None
                    self._arrangement_target_anchor = None
                self._zone_found_sent = False
                self._zone_close_sent = False
                self._zone_obstacle_waiting = False
                self._zone_obstacle_armed = True
                self._zone_obstacle_frames = 0
                self._zone_obstacle_clear_frames = 0
                self._last_zone_obstacle_event_ms = 0
                self._near_view_started_ms = (
                    _now_ms() if self.mode == MODE_NEAR_VIEW_CHECK else 0
                )
                self._near_view_adjust_sent = False
                self._near_view_result_sent = False
                self._near_view_stable_bucket = None
                self._near_view_stable_frames = 0
                if not inherit_track_lock:
                    self._last_target_center_x = None
                    self._last_target_center_y = None
                    self._last_target_area = None
                    self._last_target_accept_ms = 0
                self._last_arrangement_debug_ms = 0

        if changed:
            self._tx_wakeup.set()
            print("Vision UART command 0x{:02X}: {}".format(command, description))
        else:
            print("Vision UART ignored unknown command 0x{:02X}".format(command))

    def submit(self, objects, labels, raw_objects=None, verification_diagnostics=None):
        """Publish the newest selected result and return it for display.

        ``objects`` contains verified movable targets. ``raw_objects`` also
        contains safety-zone detections and is used after command 15/25.
        """
        with self._lock:
            mode = self.mode
            target_label = self.target_label
            zone_label = self.zone_label
            zone_found_sent = self._zone_found_sent
            zone_close_sent = self._zone_close_sent
            final_roi_sent = self._final_roi_sent
            arrangement_ready_ms = self._arrangement_ready_ms
            initial_arrangement_decision_pending = (
                self._initial_arrangement_decision_pending
            )
            final_target_locked = self._final_target_lock is not None
            track_target_locked = self._track_target_lock is not None
            search_target_locked = self._search_target_lock is not None
            diagnostic_anchor_attribute = (
                "_track_target_lock" if mode in (CMD_TRACK, MODE_WAIT_04_HANDSHAKE, MODE_WAIT_14_HANDSHAKE)
                else "_search_target_lock" if mode == CMD_SEARCH
                else "_final_target_lock"
            )
            diagnostic_anchor = getattr(self, diagnostic_anchor_attribute)
            diagnostic_previous = (
                self._last_target_center_x, self._last_target_center_y,
                self._last_target_area, self._last_target_accept_ms,
            )

        verified_input_objects = objects
        # Near-view load counting remains full-frame. For tracking/arrangement,
        # material already inside a detected safety zone is not a target.
        if mode != MODE_NEAR_VIEW_CHECK:
            zone_detections = raw_objects if raw_objects is not None else objects
            objects = self._exclude_objects_in_safety_zones(
                objects, labels, zone_detections
            )
        # Initial acquisition still requires verification. Once locked, use
        # current YOLO boxes so per-class caps/color checks cannot interrupt a
        # valid track. Safety-zone exclusion remains active in both paths.
        raw_tracking_objects = self._exclude_objects_in_safety_zones(
            raw_objects if raw_objects is not None else verified_input_objects,
            labels, raw_objects if raw_objects is not None else verified_input_objects,
        )

        if (
            target_label == "triangualrblack"
            and mode in (CMD_SEARCH, CMD_TRACK, MODE_FINAL_ROI_TRACK)
            and not (mode == MODE_FINAL_ROI_TRACK and final_target_locked)
            and not (mode == CMD_TRACK and track_target_locked)
            and not (mode == CMD_SEARCH and search_target_locked)
            and any(
                0 < int(obj.class_id) < len(labels)
                and labels[int(obj.class_id)] == "sqarered"
                for obj in objects
            )
        ):
            with self._lock:
                if self.mode == mode and self.target_label == "triangualrblack":
                    self.target_label = "sqarered"
                    target_label = self.target_label
                    self._latest_packet = None
                    self._latest_submit_ms = 0
                    self._last_target_center_x = None
                    self._last_target_center_y = None
                    self._last_target_area = None
                    self._last_target_accept_ms = 0
                    self._search_no_target_frames = 0
                    self._search_no_target_initial_sent = False
                    self._search_no_target_acknowledged = False
                    self._last_search_no_target_ms = 0
                    self._roi_entry_sent = False
                    self._track_center_sent = False
                    self._final_roi_sent = False
                    final_roi_sent = False
                    self._last_final_roi_event_ms = 0
                    # Completion events computed for the former black target
                    # must not precede this new target notification.
                    self._pending_events.clear()
                    self._pending_events.append(
                        build_event_packet(EVENT_SWITCH_TO_RED_TARGET)
                    )
                    print("[TARGET] black->red casualty action=TX11")

        reference_x, reference_y = self._reference_point(mode)
        arrangement_object_count = None
        arrangement_neighbor_missing = False
        near_view_count = None
        near_view_ids = []
        near_view_needs_arrangement = False
        near_view_log = None
        now_ms = _now_ms()
        continuity_removed = []
        self._selection_diagnostic_policy = "arrange_open_loop"
        arrangement_waiting = (
            mode in (MODE_ARRANGE_RIGHTMOST, MODE_ARRANGE_LEFTMOST)
            and now_ms < arrangement_ready_ms
        )

        if mode == MODE_NEAR_VIEW_CHECK:
            self._selection_diagnostic_policy = "near05"
            # State 05 uses every verified movable object in the full image.
            # Safety-zone boxes and deprecated class 0 are not counted.
            near_view_ids = self._movable_target_ids(objects, labels)
            near_view_count = len(near_view_ids)
            if target_label == "triangualrblack":
                near_view_needs_arrangement = (
                    near_view_count > 3
                    or any(
                        labels[class_id] in ("sqarered", "sqareblue")
                        for class_id in near_view_ids
                    )
                )
            else:
                near_view_needs_arrangement = (
                    near_view_count > 1
                    or (
                        target_label in (
                            config.UART_ZONE_CASUALTY_LABEL, "sqareredgreen"
                        )
                        and any(
                            labels[class_id] != target_label
                            for class_id in near_view_ids
                        )
                    )
                )
            selected = None
        elif mode == MODE_SEARCH_ZONE:
            self._selection_diagnostic_policy = "zone"
            candidates = raw_objects if raw_objects is not None else objects
            selected = self._select_nearest_label(
                candidates, labels, zone_label, reference_x
            )
        elif mode == MODE_FINAL_ROI_TRACK:
            selected = self._select_locked_target(
                objects, labels, target_label, now_ms,
                "_final_target_lock", "FINAL-24", raw_tracking_objects,
            )
        elif mode in (CMD_TRACK, MODE_WAIT_04_HANDSHAKE, MODE_WAIT_14_HANDSHAKE):
            selected = self._select_locked_target(
                objects, labels, target_label, now_ms,
                "_track_target_lock", "TRACK-04",
                raw_tracking_objects,
            )
        elif mode == CMD_SEARCH and search_target_locked:
            selected = self._select_locked_target(
                objects, labels, target_label, now_ms,
                "_search_target_lock", "SEARCH-03", raw_tracking_objects,
            )
        elif (
            mode in (
                MODE_POST_ARRANGE_CHECK,
            )
            or (
                target_label == TARGET_BLACK_OR_GREEN
                and mode == CMD_SEARCH
            )
        ):
            # Unlocked selection phases still use the merged spacing rules.
            self._selection_diagnostic_policy = "merged"
            selected = self._select_final_roi_target(
                objects,
                labels,
                target_label,
                now_ms,
            )
        elif mode == CMD_SEARCH and self._uses_first_green_center_selection(target_label):
            self._selection_diagnostic_policy = "first_green_frame_center"
            selected = self._select_frame_center_target(objects, labels, target_label)
        elif mode == CMD_SEARCH:
            self._selection_diagnostic_policy = "search_distance"
            tracking_objects = objects
            if target_label == "sqareredgreen":
                tracking_objects = self._filter_continuous_candidates(
                    objects, now_ms
                )
                continuity_removed = [
                    obj for obj in objects if all(obj is not kept for kept in tracking_objects)
                ]
            selection_reference_x = reference_x
            selection_reference_y = reference_y
            if mode == CMD_SEARCH:
                (
                    selection_reference_x,
                    selection_reference_y,
                ) = self._search_selection_reference()
            selected = select_nearest_horizontal_target(
                tracking_objects,
                labels,
                self.frame_width,
                target_label=target_label,
                reference_x=selection_reference_x,
                reference_y=selection_reference_y,
                vertical_first=(
                    mode == CMD_SEARCH
                    and not (
                        target_label == "sqareredgreen"
                        and not self._first_green_near_view_decided
                    )
                ),
                horizontal_first=(
                    mode == CMD_SEARCH
                    and target_label == "sqareredgreen"
                    and not self._first_green_near_view_decided
                ),
            )
        elif (
            mode == MODE_ARRANGE_RIGHTMOST
            and initial_arrangement_decision_pending
        ):
            # RX14/RX44: evaluate the target with the shared 04/24/22 rules
            # before choosing whether timed side arrangement is needed.
            self._selection_diagnostic_policy = "merged"
            arrangement_object_count = self._count_movable_targets(objects, labels)
            selected = self._select_final_roi_target(
                objects, labels, target_label, now_ms
            )
        elif arrangement_waiting:
            self._selection_diagnostic_policy = "arrange_camera_settle"
            # Event 02 is moving MG90 back to wide view. Ignore transitional
            # frames until the configured mechanical settling time expires.
            selected = None
        elif mode == MODE_ARRANGE_RIGHTMOST:
            arrangement_object_count = self._count_movable_targets(
                objects, labels
            )
            selected = self._select_arrangement_neighbor(
                objects,
                labels,
                target_label,
                reference_x,
                reference_y,
                preferred_right=True,
                max_middle_x_difference=(
                    config.UART_ARRANGE_02_MAX_X_DIFFERENCE
                ),
            )
            arrangement_neighbor_missing = (
                selected is None
                and self._middle_target_visible(
                    objects,
                    labels,
                    target_label,
                    reference_x,
                    reference_y,
                )
            )
        elif mode == MODE_ARRANGE_LEFTMOST:
            arrangement_object_count = self._count_movable_targets(
                objects, labels
            )
            selected = self._select_arrangement_neighbor(
                objects,
                labels,
                target_label,
                reference_x,
                reference_y,
                preferred_right=False,
            )
            arrangement_neighbor_missing = (
                selected is None
                and self._middle_target_visible(
                    objects,
                    labels,
                    target_label,
                    reference_x,
                    reference_y,
                )
            )
        else:
            # Analysis/sorting states return events rather than coordinates.
            selected = None

        if (
            mode in (MODE_ARRANGE_RIGHTMOST, MODE_ARRANGE_LEFTMOST)
            and not arrangement_waiting
        ):
            with self._lock:
                if arrangement_neighbor_missing:
                    self._arrangement_no_neighbor_frames += 1
                else:
                    self._arrangement_no_neighbor_frames = 0

        before_jump_target = selected
        if mode in (
            CMD_SEARCH,
            MODE_POST_ARRANGE_CHECK,
            MODE_SEARCH_ZONE,
        ) and not (
            mode == CMD_SEARCH and (
                search_target_locked or self._uses_first_green_center_selection(target_label)
            )
        ):
            selected = self._reject_large_target_jump(
                selected,
                now_ms,
                check_y_and_area=(
                    target_label == "sqareredgreen"
                    and mode
                    in (
                        CMD_SEARCH,
                        MODE_POST_ARRANGE_CHECK,
                    )
                ),
            )

        if (
            mode in (MODE_ARRANGE_RIGHTMOST, MODE_ARRANGE_LEFTMOST)
            and not arrangement_waiting
            and not initial_arrangement_decision_pending
        ):
            self._report_arrangement_debug(
                mode,
                objects,
                raw_objects,
                labels,
                target_label,
                selected,
                False,
                now_ms,
            )

        packet = None
        target_in_roi = False
        final_target_in_roi = False
        track_center_reached = False
        zone_is_close = False
        zone_blocker = None
        if selected is not None:
            class_id = int(selected.class_id)
            # Model class 0 is deprecated and must never be transmitted.
            if class_id > 0:
                if mode == MODE_SEARCH_ZONE:
                    zone_area_ratio = self._visible_box_area_ratio(selected)
                    zone_is_close = (
                        zone_area_ratio >= config.UART_ZONE_CLOSE_AREA_RATIO
                    )
                    target_x_ratio = (
                        config.UART_ZONE_CASUALTY_X_RATIO
                        if target_label == config.UART_ZONE_CASUALTY_LABEL
                        else config.UART_ZONE_SUPPLY_X_RATIO
                    )
                    if (
                        zone_found_sent
                        and zone_area_ratio > config.UART_ZONE_CENTER_AREA_RATIO
                    ):
                        target_x_ratio = 0.50
                    if zone_found_sent and not zone_close_sent:
                        zone_blocker = self._find_roi_edge_blocker(objects, labels)
                    if not zone_close_sent and not zone_is_close:
                        # Red casualties aim 3/8 inward from the right edge;
                        # supplies use the left quarter. Keep the >25% center switch.
                        error_x, error_y = self._zone_supply_point_error(
                            selected,
                            reference_x,
                            reference_y,
                            target_x_ratio,
                        )
                        packet = build_coordinate_packet(
                            class_id, error_x, error_y
                        )
                elif mode in (
                    MODE_ARRANGE_RIGHTMOST,
                    MODE_ARRANGE_LEFTMOST,
                    MODE_POST_ARRANGE_CHECK,
                ):
                    # Arrangement is MCU-timed/open-loop. Detection remains
                    # active for state decisions and overlays, but no 11-byte
                    # coordinate packet is transmitted in states 02 or 12.
                    packet = None
                else:
                    error_x, error_y = target_center_error(
                        selected,
                        self.frame_width,
                        self.frame_height,
                        reference_x=reference_x,
                        reference_y=reference_y,
                    )
                    final_target_below_roi_center = (
                        mode in (
                            MODE_FINAL_ROI_TRACK,
                            MODE_WAIT_14_HANDSHAKE,
                        )
                        and self.tracking_roi is not None
                        and float(selected.y) + float(selected.h) * 0.5
                        > reference_y
                    )
                    if final_target_below_roi_center:
                        error_y = 0
                    packet = build_coordinate_packet(
                        class_id, error_x, error_y
                    )
                    final_target_in_roi = (
                        mode == MODE_FINAL_ROI_TRACK
                        and self._target_center_in_tracking_roi(selected)
                    )
                    track_center_reached = (
                        mode == CMD_TRACK
                        and abs(error_x)
                        <= config.UART_FINAL_ROI_CENTER_TOLERANCE_X
                        and float(selected.y) + float(selected.h) * 0.5
                        >= self.frame_height * 0.5
                        - config.UART_FINAL_ROI_CENTER_TOLERANCE_Y
                    )
                target_in_roi = (
                    mode == CMD_SEARCH
                    and abs(error_x)
                    <= config.UART_FRAME_CENTER_TOLERANCE_X
                    and abs(error_y)
                    <= config.UART_FRAME_CENTER_TOLERANCE_Y
                )

        with self._lock:
            if arrangement_object_count is not None:
                self._arrangement_object_count = arrangement_object_count
            if (
                mode == MODE_ARRANGE_RIGHTMOST
                and self.mode == MODE_ARRANGE_RIGHTMOST
            ):
                self._arrangement_side_target_available = (
                    selected is not None and not arrangement_waiting
                )
            self._latest_packet = packet
            self._latest_submit_ms = now_ms
            self._last_frame_seen_ms = now_ms
            # A sequence counter also distinguishes frames submitted within
            # the same millisecond or containing identical coordinates.
            self._latest_submit_sequence += 1
            if (
                mode in (CMD_TRACK, MODE_WAIT_04_HANDSHAKE, MODE_WAIT_14_HANDSHAKE)
                and self.mode == mode
            ):
                if selected is None:
                    self._track_no_target_frames += 1
                    if self._track_no_target_frames in (1, config.UART_TRACK_LOST_CONFIRM_FRAMES):
                        eligible_labels = self._eligible_target_labels(target_label)
                        def count_targets(items):
                            return sum(
                                0 < int(obj.class_id) < len(labels)
                                and labels[int(obj.class_id)] in eligible_labels
                                for obj in items
                            )
                        print("[TRACK-04] no_target={}/{} raw={} verified={} outside_zone={}".format(
                            self._track_no_target_frames, config.UART_TRACK_LOST_CONFIRM_FRAMES,
                            count_targets(raw_objects if raw_objects is not None else objects),
                            count_targets(verified_input_objects), count_targets(objects),
                        ))
                else:
                    identity = self._final_target_identity(selected)
                    if self._track_no_target_frames:
                        print("[TRACK-04] recovered=id{} after_missing={} fresh_coordinates".format(
                            identity[0], self._track_no_target_frames
                        ))
                    self._track_no_target_frames = 0
                    if self._track_target_lock is None:
                        print("[TRACK-04] lock=id{}@({:.0f},{:.0f})".format(
                            identity[0], identity[1], identity[2]
                        ))
                    self._track_target_lock = identity
                    # A fresh coordinate interrupts the MCU's red/other search.
                    # Its next confirmed loss therefore needs its own E4.
                    # First green recovery is deliberately not interruptible:
                    # retain its one-shot E4/14 gating until MCU sends 44.
                    if mode == CMD_TRACK and not self._track_green_recovery_waiting:
                        if self._track_no_target_sent:
                            print("[TRACK-04] fresh_target=rearm_E4_for_next_loss")
                        self._track_no_target_sent = False
                    # RX14/RX44 use this position as the middle-object anchor.
                    self._last_target_center_x = identity[1]
                    self._last_target_center_y = identity[2]
                    self._last_target_area = identity[3]
                    self._last_target_accept_ms = now_ms
            if mode == MODE_FINAL_ROI_TRACK and self.mode == MODE_FINAL_ROI_TRACK:
                if selected is None:
                    self._final_no_target_frames += 1
                else:
                    self._final_no_target_frames = 0
                    if self._final_target_lock is None:
                        print("[FINAL-24] lock=id{}@({:.0f},{:.0f})".format(
                            int(selected.class_id),
                            float(selected.x) + float(selected.w) * 0.5,
                            float(selected.y) + float(selected.h) * 0.5,
                        ))
                    self._final_target_lock = self._final_target_identity(selected)
            if (
                mode == MODE_POST_ARRANGE_CHECK
                and self.mode == MODE_POST_ARRANGE_CHECK
                and selected is not None
            ):
                self._pending_events.clear()
                self._pending_events.append(
                    build_event_packet(EVENT_SKIP_TO_FINAL_TRACK)
                )
                self.mode = MODE_FINAL_ROI_TRACK
                self._final_target_lock = self._final_target_identity(selected)
                # Coordinates start with the next fresh image, after event 24.
                self._latest_packet = None
                self._latest_submit_ms = 0
                print("[POST-22] target=id{} action=TX24".format(
                    int(selected.class_id)
                ))
            if mode == CMD_SEARCH and self.mode == CMD_SEARCH:
                if selected is None:
                    self._search_no_target_frames += 1
                else:
                    if self._search_target_lock is None:
                        identity = self._final_target_identity(selected)
                        print("[SEARCH-03] lock=id{}@({:.0f},{:.0f}) source=acquisition".format(
                            identity[0], identity[1], identity[2]
                        ))
                    self._search_target_lock = self._final_target_identity(selected)
                    self._last_target_center_x = self._search_target_lock[1]
                    self._last_target_center_y = self._search_target_lock[2]
                    self._last_target_area = self._search_target_lock[3]
                    self._last_target_accept_ms = now_ms
                    self._last_search_no_target_ms = 0
                    self._search_no_target_frames = 0
                    self._search_no_target_initial_sent = False
                    self._search_no_target_acknowledged = False
            if arrangement_waiting:
                self._latest_packet = None
                self._latest_submit_ms = 0
            if (
                mode == MODE_NEAR_VIEW_CHECK
                and self.mode == MODE_NEAR_VIEW_CHECK
            ):
                # Do not mix generic no-target events into the 05 handshake.
                self._latest_packet = None
                self._latest_submit_ms = 0
                if (
                    now_ms - self._near_view_started_ms
                    >= config.UART_NEAR_VIEW_SETTLE_MS
                ):
                    # Confirm the resulting action (02/06/24). Black tasks can
                    # accept one to three black/green objects without arranging.
                    if near_view_needs_arrangement:
                        count_bucket = 2
                    else:
                        count_bucket = 1 if near_view_count > 0 else 0
                    if count_bucket == self._near_view_stable_bucket:
                        self._near_view_stable_frames += 1
                    else:
                        self._near_view_stable_bucket = count_bucket
                        self._near_view_stable_frames = 1

                    count_is_stable = (
                        self._near_view_stable_frames
                        >= config.UART_NEAR_VIEW_STABLE_FRAMES
                    )
                    first_green_decision = (
                        count_is_stable
                        and target_label == "sqareredgreen"
                        and not self._first_green_near_view_decided
                    )
                    if first_green_decision:
                        self._first_green_near_view_decided = True
                    if (
                        count_is_stable
                        and near_view_needs_arrangement
                        and not self._near_view_adjust_sent
                    ):
                        self._pending_events.append(
                            build_event_packet(EVENT_ROI_READY)
                        )
                        self._near_view_adjust_sent = True
                        if first_green_decision:
                            self._first_green_both_sides_required = True
                            # MCU movement is open-loop; keep requesting 02
                            # even if changing camera view hides the side box.
                            self._arrangement_02_without_side = True
                            print("[NEAR-05] first_green=push_right_then_left wait_RX12_then_RX22")
                        # MCU raises MG90 to the wide view. Resume the existing
                        # arrangement-02/12 state machine after it settles.
                        self.mode = MODE_ARRANGE_RIGHTMOST
                        self._initial_arrangement_decision_pending = False
                        self._arrangement_02_ack_pending = True
                        self._arrangement_side_target_available = False
                        self._last_arrangement_02_event_ms = 0
                        self._arrangement_ready_ms = (
                            now_ms + config.UART_WIDE_VIEW_SETTLE_MS
                        )
                        self._arrangement_object_count = 0
                        self._arrangement_no_neighbor_frames = 0
                        self._middle_target_anchor = None
                        self._arrangement_target_anchor = None
                        self._last_target_center_x = None
                        self._last_target_center_y = None
                        self._last_target_area = None
                        self._last_target_accept_ms = 0
                        near_view_log = "TX02 arrange_02_wide"
                    elif (
                        count_is_stable
                        and near_view_count > 0
                        and not near_view_needs_arrangement
                        and not self._near_view_result_sent
                    ):
                        self._pending_events.append(
                            build_event_packet(EVENT_ZONE_SEARCH_STARTED)
                        )
                        self._near_view_result_sent = True
                        self.mode = MODE_SEARCH_ZONE
                        near_view_log = "TX06 search_zone"
                    elif (
                        count_is_stable
                        and near_view_count == 0
                        and not self._near_view_result_sent
                    ):
                        self._pending_events.append(
                            build_event_packet(EVENT_SKIP_TO_FINAL_TRACK)
                        )
                        self._near_view_result_sent = True
                        self.mode = MODE_FINAL_ROI_TRACK
                        self._final_target_lock = None
                        self._latest_packet = None
                        self._latest_submit_ms = 0
                        self._last_target_center_x = None
                        self._last_target_center_y = None
                        self._last_target_area = None
                        self._last_target_accept_ms = 0
                        near_view_log = "TX24 no_object_return_final"
            if (
                mode == MODE_ARRANGE_RIGHTMOST
                and self.mode == MODE_ARRANGE_RIGHTMOST
                and initial_arrangement_decision_pending
            ):
                # A fresh merged result decides immediately; each event is
                # queued once. State 24 starts coordinates on the next frame.
                event = (
                    EVENT_SKIP_TO_FINAL_TRACK if selected is not None
                    else EVENT_ROI_READY
                )
                self._pending_events.clear()
                self._pending_events.append(build_event_packet(event))
                self.mode = (
                    MODE_FINAL_ROI_TRACK if selected is not None
                    else MODE_ARRANGE_RIGHTMOST
                )
                self._final_target_lock = (
                    self._final_target_identity(selected)
                    if selected is not None else None
                )
                self._initial_arrangement_decision_pending = False
                self._arrangement_02_ack_pending = selected is None
                self._arrangement_02_without_side = selected is None
                self._arrangement_side_target_available = False
                self._last_arrangement_02_event_ms = 0
                self._latest_packet = None
                self._latest_submit_ms = 0
                self._last_target_center_x = None
                self._last_target_center_y = None
                self._last_target_area = None
                self._last_target_accept_ms = 0
                self._arrangement_no_neighbor_frames = 0
                print("[PRE-02] merged_target={} action=TX{:02X}".format(
                    "id{}".format(int(selected.class_id))
                    if selected is not None else "none", event
                ))
            if target_in_roi and not self._roi_entry_sent:
                self._pending_events.append(
                    build_event_packet(EVENT_SEARCH_CENTER_REACHED)
                )
                self._roi_entry_sent = True
                self._track_target_lock = self._final_target_identity(selected)
                self._last_target_center_x = self._track_target_lock[1]
                self._last_target_center_y = self._track_target_lock[2]
                self._last_target_area = self._track_target_lock[3]
                self._last_target_accept_ms = now_ms
                self._track_no_target_frames = 0
                print("[SEARCH-03] center_reached=id{} retain_lock_until_RX04".format(
                    int(selected.class_id)
                ))
                # Keep publishing image-center errors, but do not emit 14
                # until MCU returns command 04 to enter CMD_TRACK.
                self.mode = MODE_WAIT_04_HANDSHAKE
                self._track_center_sent = False
            elif selected is not None and not target_in_roi:
                self._roi_entry_sent = False
            if (
                track_center_reached and not self._track_center_sent
                and not self._track_green_recovery_waiting
            ):
                self._pending_events.append(
                    build_event_packet(EVENT_TRACK_CENTER_REACHED)
                )
                self._track_center_sent = True
                self.mode = MODE_WAIT_14_HANDSHAKE
                # Do not transmit the last image-center packet after event 14.
                # The next frame will be rebuilt against the ROI center.
                self._latest_packet = None
                self._latest_submit_ms = 0
            if mode == MODE_FINAL_ROI_TRACK and final_target_in_roi:
                if not self._final_roi_sent:
                    self._pending_events.append(
                        build_event_packet(EVENT_FINAL_TARGET_IN_ROI)
                    )
                    self._final_roi_sent = True
                # While the current target center remains in ROI, send only event
                # 34 (immediately, then every configured repeat interval).
                self._latest_packet = None
                self._latest_submit_ms = 0
            elif mode == MODE_FINAL_ROI_TRACK and self._final_roi_sent:
                # The target center left the ROI or disappeared. Stop
                # repeating 34; coordinates/EE can resume, and re-entry will
                # trigger an immediate new 34.
                self._final_roi_sent = False
                self._last_final_roi_event_ms = 0
                pending_34 = build_event_packet(EVENT_FINAL_TARGET_IN_ROI)
                self._pending_events = [
                    event for event in self._pending_events
                    if event != pending_34
                ]
            if (
                mode == MODE_SEARCH_ZONE
                and selected is not None
                and int(selected.class_id) > 0
                and not self._zone_found_sent
            ):
                self._pending_events.append(build_event_packet(EVENT_ZONE_FOUND))
                self._zone_found_sent = True
            if mode == MODE_SEARCH_ZONE and not self._zone_close_sent:
                if self._zone_found_sent and selected is not None:
                    if zone_blocker is not None:
                        self._zone_obstacle_clear_frames = 0
                        if self._zone_obstacle_armed:
                            self._zone_obstacle_frames += 1
                    else:
                        # Keep accumulated blocker hits across missed frames.
                        self._zone_obstacle_clear_frames += 1
                        if (
                            not self._zone_obstacle_waiting
                            and self._zone_obstacle_clear_frames
                            >= config.UART_ZONE_OBSTACLE_CLEAR_FRAMES
                        ):
                            self._zone_obstacle_armed = True
                else:
                    self._zone_obstacle_clear_frames = 0
                if (
                    self._zone_obstacle_armed
                    and not self._zone_obstacle_waiting
                    and self._zone_obstacle_frames
                    >= config.UART_ZONE_OBSTACLE_CONFIRM_FRAMES
                ):
                    self._zone_obstacle_waiting = True
                    self._pending_events.append(
                        build_event_packet(EVENT_ZONE_OBSTACLE)
                    )
                    print("[ZONE-16] blocker=id{} action=TX36_WAIT_ACK".format(
                        int(zone_blocker.class_id)
                    ))
                if self._zone_obstacle_waiting:
                    self._latest_packet = None
                    self._latest_submit_ms = 0
            if (
                mode == MODE_SEARCH_ZONE and zone_is_close
                and not self._zone_obstacle_waiting
                and (zone_blocker is None or not self._zone_obstacle_armed)
            ):
                if not self._zone_close_sent:
                    self._pending_events.append(
                        build_event_packet(EVENT_ZONE_CLOSE)
                    )
                    self._zone_close_sent = True
                # After event 26, do not send coordinates or no-target events
                # until a new command starts another safety-zone search.
                self._latest_packet = None
                self._latest_submit_ms = 0
            elif mode == MODE_SEARCH_ZONE and self._zone_close_sent:
                self._latest_packet = None
                self._latest_submit_ms = 0
        self._tx_wakeup.set()
        self._diagnostics.report(
            self, labels, raw_objects if raw_objects is not None else verified_input_objects,
            verified_input_objects, objects, selected, verification_diagnostics,
            mode, self._selection_diagnostic_policy, diagnostic_anchor, diagnostic_previous,
            before_jump_target is not None and selected is None, continuity_removed, now_ms,
            "_track_target_lock" if target_in_roi else diagnostic_anchor_attribute,
        )
        if near_view_log is not None:
            print(
                "[NEAR-05] objects={} ids={} stable={}/{} action={}".format(
                    near_view_count,
                    near_view_ids,
                    config.UART_NEAR_VIEW_STABLE_FRAMES,
                    config.UART_NEAR_VIEW_STABLE_FRAMES,
                    near_view_log,
                )
            )
        if (
            packet is not None
            or zone_is_close
            or zone_close_sent
            or final_target_in_roi
            or final_roi_sent
        ):
            return selected
        return None

    def _find_roi_edge_blocker(self, objects, labels):
        """Find a movable center inside the shared approach trapezoid."""
        polygon = zone_obstacle_polygon(
            self.tracking_roi, self.frame_width, self.frame_height
        )
        if polygon is None:
            return None
        movable_labels = set(TARGET_SELECTION_COMMANDS.values())
        for obj in objects:
            class_id = int(obj.class_id)
            if class_id <= 0 or class_id >= len(labels):
                continue
            if labels[class_id] not in movable_labels:
                continue
            center_x = float(obj.x) + float(obj.w) * 0.5
            center_y = float(obj.y) + float(obj.h) * 0.5
            if point_in_zone_obstacle_polygon(center_x, center_y, polygon):
                return obj
        return None

    @staticmethod
    def _zone_supply_point_error(
        obj, reference_x, reference_y, target_x_ratio=0.25
    ):
        """Return ROI error to the configured horizontal point in a zone."""
        supply_x = float(obj.x) + float(obj.w) * float(target_x_ratio)
        supply_y = float(obj.y) + float(obj.h) * 0.5
        return (
            int(round(supply_x - float(reference_x))),
            int(round(supply_y - float(reference_y))),
        )

    def _visible_box_area_ratio(self, obj):
        """Return the detected box area visible inside the camera frame."""
        left = max(0, int(obj.x))
        top = max(0, int(obj.y))
        right = min(self.frame_width, int(obj.x + obj.w))
        bottom = min(self.frame_height, int(obj.y + obj.h))
        width = max(0, right - left)
        height = max(0, bottom - top)
        frame_area = max(1, self.frame_width * self.frame_height)
        return (width * height) / float(frame_area)

    def _select_arrangement_neighbor(
        self,
        objects,
        labels,
        middle_label,
        reference_x,
        reference_y,
        preferred_right,
        max_middle_x_difference=None,
    ):
        """Select one side target, releasing a stale side lock if needed."""
        selected = self._select_side_neighbor(
            objects,
            labels,
            middle_label,
            reference_x,
            reference_y,
            right_side=preferred_right,
            max_middle_x_difference=max_middle_x_difference,
            max_middle_y_difference=config.UART_ARRANGE_MAX_Y_DIFFERENCE,
            ignore_x=True,
        )
        if selected is None:
            selected = self._select_side_neighbor(
                objects,
                labels,
                middle_label,
                reference_x,
                reference_y,
                right_side=not preferred_right,
                max_middle_x_difference=max_middle_x_difference,
                max_middle_y_difference=config.UART_ARRANGE_MAX_Y_DIFFERENCE,
                ignore_x=True,
            )
        if selected is not None:
            return selected

        # The locked side object may already have been pushed out of view while
        # another valid object remains. Drop only the side lock and retry both
        # sides; the middle-object anchor remains untouched.
        with self._lock:
            had_side_lock = self._arrangement_target_anchor is not None
            if had_side_lock:
                self._arrangement_target_anchor = None
        if not had_side_lock:
            return None

        selected = self._select_side_neighbor(
            objects,
            labels,
            middle_label,
            reference_x,
            reference_y,
            right_side=preferred_right,
            max_middle_x_difference=max_middle_x_difference,
            max_middle_y_difference=config.UART_ARRANGE_MAX_Y_DIFFERENCE,
            ignore_x=True,
        )
        if selected is None:
            selected = self._select_side_neighbor(
                objects,
                labels,
                middle_label,
                reference_x,
                reference_y,
                right_side=not preferred_right,
                max_middle_x_difference=max_middle_x_difference,
                max_middle_y_difference=config.UART_ARRANGE_MAX_Y_DIFFERENCE,
                ignore_x=True,
            )
        return selected

    def _select_side_neighbor(
        self,
        objects,
        labels,
        middle_label,
        reference_x,
        reference_y,
        right_side,
        max_middle_x_difference=None,
        max_middle_y_difference=None,
        ignore_x=False,
    ):
        """Select a movable neighbor using the active arrangement rule."""
        movable_labels = set(TARGET_SELECTION_COMMANDS.values())
        valid_objects = []
        for obj in objects:
            class_id = int(obj.class_id)
            if class_id <= 0 or class_id >= len(labels):
                continue
            label = labels[class_id]
            if label not in movable_labels:
                continue
            valid_objects.append(obj)
        middle = self._find_middle_target(
            valid_objects,
            labels,
            middle_label,
            reference_x,
            reference_y,
        )
        if middle is None:
            return None
        self._update_middle_target_anchor(middle)
        middle_x = float(middle.x) + float(middle.w) * 0.5
        middle_y = float(middle.y) + float(middle.h) * 0.5

        with self._lock:
            target_anchor = self._arrangement_target_anchor

        selected = None
        selected_key = None
        for obj in valid_objects:
            if obj is middle:
                continue
            center_x = float(obj.x) + float(obj.w) * 0.5
            delta_x = center_x - middle_x
            if (
                max_middle_x_difference is not None
                and abs(delta_x) > float(max_middle_x_difference)
            ):
                continue
            if right_side is True and delta_x <= 0:
                continue
            if right_side is False and delta_x >= 0:
                continue
            center_y = float(obj.y) + float(obj.h) * 0.5
            middle_delta_y = center_y - middle_y
            if max_middle_y_difference is not None:
                if abs(middle_delta_y) > float(max_middle_y_difference):
                    continue
            delta_y = center_y - float(reference_y)
            score = float(getattr(obj, "score", 0.0))
            if target_anchor is not None:
                anchor_x, anchor_y, anchor_area, anchor_class_id = (
                    target_anchor
                )
                if int(obj.class_id) != anchor_class_id:
                    continue
                area = max(1.0, float(obj.w) * float(obj.h))
                if abs(center_x - anchor_x) > config.UART_MAX_X_JUMP_PX:
                    continue
                if abs(center_y - anchor_y) > config.UART_MAX_Y_JUMP_PX:
                    continue
                if (
                    abs(area - anchor_area) / max(1.0, anchor_area)
                    > config.UART_MAX_AREA_CHANGE_RATIO
                ):
                    continue
                key = (
                    (center_x - anchor_x) ** 2
                    + (center_y - anchor_y) ** 2,
                    abs(area - anchor_area),
                    -score,
                )
            elif ignore_x:
                # X only decides which side the object is on. Among objects
                # on that side, vertical alignment alone decides selection.
                key = (abs(middle_delta_y), -score)
            else:
                # Second stage remains constrained to the requested side and
                # nearest to (middle object's X, ROI center Y).
                key = (delta_x * delta_x + delta_y * delta_y, -score)
            if selected_key is None or key < selected_key:
                selected = obj
                selected_key = key
        if selected is not None:
            self._update_arrangement_target_anchor(selected)
        return selected

    def arrangement_side_objects(self, objects, labels, raw_objects=None):
        """Return accepted left/right neighbors for the state-02 overlay."""
        objects = self._exclude_objects_in_safety_zones(
            objects, labels,
            raw_objects if raw_objects is not None else objects,
        )
        with self._lock:
            if self.mode != MODE_ARRANGE_RIGHTMOST:
                return []
            middle_label = self.target_label

        reference_x, reference_y = self._reference_point(
            MODE_ARRANGE_RIGHTMOST
        )
        movable_labels = set(TARGET_SELECTION_COMMANDS.values())
        valid_objects = [
            obj
            for obj in objects
            if int(obj.class_id) > 0
            and int(obj.class_id) < len(labels)
            and labels[int(obj.class_id)] in movable_labels
        ]
        middle = self._find_middle_target(
            valid_objects,
            labels,
            middle_label,
            reference_x,
            reference_y,
        )
        if middle is None:
            return []

        middle_x = float(middle.x) + float(middle.w) * 0.5
        middle_y = float(middle.y) + float(middle.h) * 0.5
        accepted = []
        for obj in valid_objects:
            if obj is middle:
                continue
            center_x = float(obj.x) + float(obj.w) * 0.5
            center_y = float(obj.y) + float(obj.h) * 0.5
            if center_x == middle_x:
                continue
            if (
                abs(center_x - middle_x)
                <= config.UART_ARRANGE_02_MAX_X_DIFFERENCE
                and
                abs(center_y - middle_y)
                <= config.UART_ARRANGE_MAX_Y_DIFFERENCE
            ):
                accepted.append(obj)
        return accepted

    def _find_middle_target(
        self, objects, labels, middle_label, reference_x, reference_y
    ):
        middle_candidates = [
            obj
            for obj in objects
            if int(obj.class_id) > 0
            and int(obj.class_id) < len(labels)
            and labels[int(obj.class_id)] in self._eligible_target_labels(middle_label)
        ]
        if not middle_candidates:
            return None
        with self._lock:
            anchor = self._middle_target_anchor
        if anchor is not None:
            anchor_x, anchor_y, anchor_area = anchor
            continuous_candidates = []
            for obj in middle_candidates:
                center_x = float(obj.x) + float(obj.w) * 0.5
                center_y = float(obj.y) + float(obj.h) * 0.5
                area = max(1.0, float(obj.w) * float(obj.h))
                if abs(center_x - anchor_x) > config.UART_MAX_X_JUMP_PX:
                    continue
                if (
                    anchor_y is not None
                    and abs(center_y - anchor_y) > config.UART_MAX_Y_JUMP_PX
                ):
                    continue
                if (
                    anchor_area is not None
                    and abs(area - anchor_area) / max(1.0, anchor_area)
                    > config.UART_MAX_AREA_CHANGE_RATIO
                ):
                    continue
                continuous_candidates.append(obj)
            if not continuous_candidates:
                return None
            if anchor_y is None:
                # First frame after MCU's 14 handshake: preserve the tracked
                # object's X identity while accepting the expected Y/area
                # change caused by the forward follow-through.
                return min(
                    continuous_candidates,
                    key=lambda obj: (
                        abs(
                            float(obj.x)
                            + float(obj.w) * 0.5
                            - anchor_x
                        ),
                        abs(
                            float(obj.x)
                            + float(obj.w) * 0.5
                            - float(reference_x)
                        ),
                        -float(getattr(obj, "score", 0.0)),
                    ),
                )
            return min(
                continuous_candidates,
                key=lambda obj: (
                    (float(obj.x) + float(obj.w) * 0.5 - anchor_x) ** 2
                    + (float(obj.y) + float(obj.h) * 0.5 - anchor_y) ** 2,
                    abs(float(obj.w) * float(obj.h) - anchor_area),
                    -float(getattr(obj, "score", 0.0)),
                ),
            )
        return min(
            middle_candidates,
            key=lambda obj: (
                (float(obj.x) + float(obj.w) * 0.5 - float(reference_x)) ** 2
                + (float(obj.y) + float(obj.h) * 0.5 - float(reference_y)) ** 2,
                -float(getattr(obj, "score", 0.0)),
            ),
        )

    def _update_middle_target_anchor(self, middle):
        anchor = (
            float(middle.x) + float(middle.w) * 0.5,
            float(middle.y) + float(middle.h) * 0.5,
            max(1.0, float(middle.w) * float(middle.h)),
        )
        with self._lock:
            self._middle_target_anchor = anchor

    def _update_arrangement_target_anchor(self, selected):
        anchor = (
            float(selected.x) + float(selected.w) * 0.5,
            float(selected.y) + float(selected.h) * 0.5,
            max(1.0, float(selected.w) * float(selected.h)),
            int(selected.class_id),
        )
        with self._lock:
            self._arrangement_target_anchor = anchor

    def _filter_continuous_candidates(self, objects, now_ms):
        """Keep green candidates consistent with the last accepted target."""
        with self._lock:
            previous_x = self._last_target_center_x
            previous_y = self._last_target_center_y
            previous_area = self._last_target_area
            previous_ms = self._last_target_accept_ms
        if (
            previous_x is None
            or previous_y is None
            or previous_area is None
            or now_ms - previous_ms >= config.UART_X_JUMP_RESET_MS
        ):
            return objects

        filtered = []
        for obj in objects:
            center_x = float(obj.x) + float(obj.w) * 0.5
            center_y = float(obj.y) + float(obj.h) * 0.5
            area = max(1.0, float(obj.w) * float(obj.h))
            if abs(center_x - previous_x) > config.UART_MAX_X_JUMP_PX:
                continue
            if abs(center_y - previous_y) > config.UART_MAX_Y_JUMP_PX:
                continue
            if (
                abs(area - previous_area) / max(1.0, previous_area)
                > config.UART_MAX_AREA_CHANGE_RATIO
            ):
                continue
            filtered.append(obj)
        return filtered

    @staticmethod
    def _exclude_objects_in_safety_zones(objects, labels, zone_detections):
        """Exclude movable objects whose centers are in red/blue safety boxes."""
        zone_labels = set(ZONE_SEARCH_COMMANDS.values())
        zones = [
            zone for zone in zone_detections
            if 0 < int(zone.class_id) < len(labels)
            and labels[int(zone.class_id)] in zone_labels
            and float(zone.w) > 0 and float(zone.h) > 0
        ]
        movable_labels = set(TARGET_SELECTION_COMMANDS.values())
        eligible = []
        for obj in objects:
            class_id = int(obj.class_id)
            if (
                0 < class_id < len(labels)
                and labels[class_id] in movable_labels
            ):
                center_x = float(obj.x) + float(obj.w) * 0.5
                center_y = float(obj.y) + float(obj.h) * 0.5
                if any(
                    float(zone.x) <= center_x <= float(zone.x) + float(zone.w)
                    and float(zone.y) <= center_y <= float(zone.y) + float(zone.h)
                    for zone in zones
                ):
                    continue
            eligible.append(obj)
        return eligible

    @staticmethod
    def _movable_target_ids(objects, labels):
        movable_labels = set(TARGET_SELECTION_COMMANDS.values())
        class_ids = []
        for obj in objects:
            class_id = int(obj.class_id)
            if (
                class_id > 0
                and class_id < len(labels)
                and labels[class_id] in movable_labels
            ):
                class_ids.append(class_id)
        return class_ids

    @staticmethod
    def _count_movable_targets(objects, labels):
        return len(VisionSerialController._movable_target_ids(objects, labels))

    def _middle_target_visible(
        self,
        objects,
        labels,
        middle_label,
        reference_x,
        reference_y,
    ):
        movable_labels = set(TARGET_SELECTION_COMMANDS.values())
        valid_objects = [
            obj
            for obj in objects
            if int(obj.class_id) > 0
            and int(obj.class_id) < len(labels)
            and labels[int(obj.class_id)] in movable_labels
        ]
        return (
            self._find_middle_target(
                valid_objects,
                labels,
                middle_label,
                reference_x,
                reference_y,
            )
            is not None
        )

    def _report_arrangement_debug(
        self,
        mode,
        verified_objects,
        raw_objects,
        labels,
        middle_label,
        selected,
        skip_to_final,
        now_ms,
    ):
        """Print compact filtering evidence for arrangement diagnosis."""
        if (
            not skip_to_final
            and now_ms - self._last_arrangement_debug_ms
            < config.UART_ARRANGE_DEBUG_PRINT_MS
        ):
            return
        self._last_arrangement_debug_ms = now_ms

        raw_candidates = (
            raw_objects if raw_objects is not None else verified_objects
        )
        raw_count = self._count_movable_targets(raw_candidates, labels)
        verified_count = self._count_movable_targets(
            verified_objects, labels
        )
        reference_x, reference_y = self._reference_point(mode)
        movable_labels = set(TARGET_SELECTION_COMMANDS.values())
        valid_objects = [
            obj
            for obj in verified_objects
            if int(obj.class_id) > 0
            and int(obj.class_id) < len(labels)
            and labels[int(obj.class_id)] in movable_labels
        ]
        middle = self._find_middle_target(
            valid_objects,
            labels,
            middle_label,
            reference_x,
            reference_y,
        )

        state_name = "02" if mode == MODE_ARRANGE_RIGHTMOST else "12"
        if middle is None:
            same_label_visible = any(
                labels[int(obj.class_id)] in self._eligible_target_labels(middle_label)
                for obj in valid_objects
            )
            middle_text = (
                "missing_locked" if same_label_visible else "missing"
            )
            detail_text = "none"
        else:
            middle_x = float(middle.x) + float(middle.w) * 0.5
            middle_y = float(middle.y) + float(middle.h) * 0.5
            middle_text = "id{}@({:.0f},{:.0f})".format(
                int(middle.class_id), middle_x, middle_y
            )
            details = []
            has_valid_left = any(
                obj is not middle
                and float(obj.x) + float(obj.w) * 0.5 < middle_x
                and abs(
                    float(obj.y) + float(obj.h) * 0.5 - middle_y
                )
                <= config.UART_ARRANGE_MAX_Y_DIFFERENCE
                for obj in valid_objects
            )
            for obj in valid_objects:
                if obj is middle:
                    continue
                center_x = float(obj.x) + float(obj.w) * 0.5
                center_y = float(obj.y) + float(obj.h) * 0.5
                side = "R" if center_x > middle_x else "L"
                x_difference = abs(center_x - middle_x)
                y_difference = abs(center_y - middle_y)
                if (
                    mode == MODE_ARRANGE_RIGHTMOST
                    and x_difference
                    > config.UART_ARRANGE_02_MAX_X_DIFFERENCE
                ):
                    result = "reject_x"
                elif y_difference > config.UART_ARRANGE_MAX_Y_DIFFERENCE:
                    result = "reject_y"
                elif mode == MODE_ARRANGE_LEFTMOST and side != "L":
                    result = (
                        "reject_side" if has_valid_left else "ok_fallback"
                    )
                else:
                    result = "ok"
                details.append(
                    "id{}:{}:dx={:.0f}:dy={:.0f}:{}".format(
                        int(obj.class_id),
                        side,
                        x_difference,
                        y_difference,
                        result,
                    )
                )
            detail_text = ",".join(details) if details else "none"

        if selected is None:
            selected_text = "none"
        else:
            selected_x = float(selected.x) + float(selected.w) * 0.5
            selected_y = float(selected.y) + float(selected.h) * 0.5
            selected_text = "id{}@({:.0f},{:.0f})".format(
                int(selected.class_id), selected_x, selected_y
            )
        if skip_to_final:
            action = "TX02"
        elif selected is not None:
            action = "open_loop"
        else:
            action = "wait"
        miss_frames = self._arrangement_no_neighbor_frames
        print(
            "[ARRANGE-{}] raw={} verified={} middle={} objects=[{}] "
            "selected={} miss_frames={} action={}".format(
                state_name,
                raw_count,
                verified_count,
                middle_text,
                detail_text,
                selected_text,
                miss_frames,
                action,
            )
        )

    def _reject_large_target_jump(
        self, selected, now_ms, check_y_and_area=False
    ):
        """Drop sudden position or area changes caused by target switching."""
        if selected is None:
            return None

        center_x = float(selected.x) + float(selected.w) * 0.5
        center_y = float(selected.y) + float(selected.h) * 0.5
        area = max(1.0, float(selected.w) * float(selected.h))
        with self._lock:
            previous_x = self._last_target_center_x
            previous_y = self._last_target_center_y
            previous_area = self._last_target_area
            previous_ms = self._last_target_accept_ms
            may_reset = (
                previous_x is None
                or now_ms - previous_ms >= config.UART_X_JUMP_RESET_MS
            )
            if (
                not may_reset
                and abs(center_x - previous_x) > config.UART_MAX_X_JUMP_PX
            ):
                return None
            if (
                check_y_and_area
                and not may_reset
                and previous_y is not None
                and abs(center_y - previous_y) > config.UART_MAX_Y_JUMP_PX
            ):
                return None
            if (
                check_y_and_area
                and not may_reset
                and previous_area is not None
                and abs(area - previous_area) / max(1.0, previous_area)
                > config.UART_MAX_AREA_CHANGE_RATIO
            ):
                return None

            self._last_target_center_x = center_x
            self._last_target_center_y = center_y
            self._last_target_area = area
            self._last_target_accept_ms = now_ms
        return selected

    def _target_center_in_tracking_roi(self, obj):
        """Event 34 needs ROI entry, not alignment with its center point."""
        if self.tracking_roi is None:
            return False
        left, top, width, height = self.tracking_roi
        center_x = float(obj.x) + float(obj.w) * 0.5
        center_y = float(obj.y) + float(obj.h) * 0.5
        return (
            float(left) <= center_x < float(left) + float(width)
            and float(top) <= center_y < float(top) + float(height)
        )

    def _reference_point(self, mode):
        """Return the state-specific coordinate error reference point."""
        if mode in (CMD_SEARCH, CMD_TRACK, MODE_WAIT_04_HANDSHAKE):
            return self.frame_width * 0.5, self.frame_height * 0.5
        if mode in (MODE_ARRANGE_RIGHTMOST, MODE_ARRANGE_LEFTMOST):
            if self.tracking_roi is not None:
                left, top, width, height = self.tracking_roi
                return (
                    float(left) + float(width) * 0.5,
                    float(top) + float(height) * 0.5,
                )
            return self.frame_width * 0.5, self.frame_height * 0.5
        if (
            mode in (
                CMD_TRACK,
                MODE_WAIT_04_HANDSHAKE,
                MODE_WAIT_14_HANDSHAKE,
                MODE_FINAL_ROI_TRACK,
                MODE_POST_ARRANGE_CHECK,
                MODE_SEARCH_ZONE,
            )
            and self.tracking_roi is not None
        ):
            left, top, width, height = self.tracking_roi
            return (
                float(left) + float(width) * 0.5,
                float(top) + float(height) * 0.5,
            )
        return self.frame_width * 0.5, self.frame_height * 0.5

    def _search_selection_reference(self):
        """Return the ROI center used only for state-03 target selection."""
        if self.tracking_roi is not None:
            left, top, width, height = self.tracking_roi
            return (
                float(left) + float(width) * 0.5,
                float(top) + float(height) * 0.5,
            )
        return self.frame_width * 0.5, self.frame_height * 0.5

    @staticmethod
    def _eligible_target_labels(target_label):
        if target_label == TARGET_BLACK_OR_GREEN:
            return ("triangualrblack", "sqareredgreen")
        return (target_label,)

    @staticmethod
    def _final_candidate_is_blocked(candidate, movable_objects):
        candidate_left = float(candidate.x)
        candidate_right = candidate_left + float(candidate.w)
        candidate_center_y = float(candidate.y) + float(candidate.h) * 0.5
        for other in movable_objects:
            if other is candidate:
                continue
            other_left = float(other.x)
            other_right = other_left + float(other.w)
            horizontal_overlap = (
                min(candidate_right, other_right)
                - max(candidate_left, other_left)
            )
            other_center_y = float(other.y) + float(other.h) * 0.5
            if horizontal_overlap > 0 and other_center_y > candidate_center_y:
                return True
        return False

    def final_track_debug_objects(
        self, objects, labels, raw_objects=None, actual_selected=None
    ):
        """Preview the 24 rules without changing tracking or serial state."""
        with self._lock:
            target_label = self.target_label or "sqareredgreen"
            mode = self.mode
            actual_target_selected = self.target_label is not None
        if target_label not in TARGET_SELECTION_COMMANDS.values():
            return [], [], None, []

        zone_detections = raw_objects if raw_objects is not None else objects
        eligible = self._exclude_objects_in_safety_zones(
            objects, labels, zone_detections
        )
        movable_labels = set(TARGET_SELECTION_COMMANDS.values())
        movable = []
        candidates = []
        for obj in eligible:
            class_id = int(obj.class_id)
            if class_id <= 0 or class_id >= len(labels):
                continue
            label = labels[class_id]
            if label not in movable_labels:
                continue
            movable.append(obj)
            if label in self._eligible_target_labels(target_label):
                candidates.append(obj)

        unblocked = [
            candidate for candidate in candidates
            if not self._final_candidate_is_blocked(candidate, movable)
        ]
        if unblocked:
            max_y = max(float(obj.y) + float(obj.h) * 0.5 for obj in unblocked)
            primary = [
                obj for obj in unblocked
                if float(obj.y) + float(obj.h) * 0.5 == max_y
            ]
            preview = min(
                primary,
                key=lambda obj: (
                    float(obj.x) + float(obj.w) * 0.5,
                    -float(getattr(obj, "score", 0.0)),
                ),
            )
            selected = (
                actual_selected
                if mode == MODE_FINAL_ROI_TRACK and actual_target_selected
                else preview
            )
            return [], [], selected, []

        # Highlight every object inside this candidate's perspective dx band,
        # not just the nearest-X object used for the actual decision.
        neighbors = []
        neighbor_dx = {}
        for candidate in candidates:
            center_x = float(candidate.x) + float(candidate.w) * 0.5
            center_y = float(candidate.y) + float(candidate.h) * 0.5
            for other in movable:
                if other is candidate:
                    continue
                red_pair = (
                    int(candidate.class_id) == 4
                    or int(other.class_id) == 4
                )
                min_dx, _ = self._final_spacing_limits(center_y, red_pair)
                other_x = float(other.x) + float(other.w) * 0.5
                dx = abs(other_x - center_x)
                if dx >= min_dx:
                    continue
                if all(obj is not other for obj in neighbors):
                    neighbors.append(other)
                previous = neighbor_dx.get(id(other))
                if previous is None or dx < previous:
                    neighbor_dx[id(other)] = dx
        secondary = [
            candidate for candidate in candidates
            if self._matches_final_primary_spacing(candidate, movable)
        ]
        preview = min(
            secondary,
            key=lambda obj: (
                float(obj.x) + float(obj.w) * 0.5,
                -float(getattr(obj, "score", 0.0)),
            ),
        ) if secondary else None
        selected = (
            actual_selected
            if mode == MODE_FINAL_ROI_TRACK and actual_target_selected
            else preview
        )
        dx_annotations = [
            (neighbor, int(round(neighbor_dx[id(neighbor)])))
            for neighbor in neighbors
        ]
        return candidates, neighbors, selected, dx_annotations

    @staticmethod
    def _final_target_identity(obj):
        return (
            int(obj.class_id),
            float(obj.x) + float(obj.w) * 0.5,
            float(obj.y) + float(obj.h) * 0.5,
            max(1.0, float(obj.w) * float(obj.h)),
        )

    def _select_locked_final_target(self, objects, labels, target_label, now_ms):
        return self._select_locked_target(
            objects, labels, target_label, now_ms,
            "_final_target_lock", "FINAL-24",
        )

    def _uses_first_green_center_selection(self, target_label):
        return target_label == "sqareredgreen" and not self._first_green_near_view_decided

    def _select_frame_center_target(self, objects, labels, target_label):
        """First green: nearest 2D image center, without merged/jump rules."""
        return select_nearest_horizontal_target(
            objects, labels, self.frame_width, target_label=target_label,
            reference_x=self.frame_width * .5, reference_y=self.frame_height * .5,
        )

    def _select_unlocked_target(self, objects, labels, target_label, now_ms, anchor_attribute):
        if anchor_attribute == "_track_target_lock" and self._uses_first_green_center_selection(target_label):
            self._selection_diagnostic_policy = "first_green_frame_center"
            return self._select_frame_center_target(objects, labels, target_label)
        self._selection_diagnostic_policy = "merged"
        return self._select_final_roi_target(objects, labels, target_label, now_ms)

    def _select_locked_target(
        self, objects, labels, target_label, now_ms, anchor_attribute, log_tag,
        association_objects=None,
    ):
        """Keep a matched lock; after confirmed loss, screen targets again.

        Do not rerun spacing, obstruction or maximum-Y ranking while the
        locked class is detected. Use nearest previous position to associate
        multiple same-class boxes, without any position/area change cutoff.
        """
        with self._lock:
            anchor = getattr(self, anchor_attribute)
        if anchor is None:
            return self._select_unlocked_target(
                objects, labels, target_label, now_ms, anchor_attribute
            )

        class_id, previous_x, previous_y, _ = anchor
        self._selection_diagnostic_policy = {
            "_track_target_lock": "track_lock",
            "_final_target_lock": "final_lock",
            "_search_target_lock": "search_lock",
        }[anchor_attribute]
        matches = []
        for obj in association_objects if association_objects is not None else objects:
            obj_id, center_x, center_y, _ = self._final_target_identity(obj)
            if obj_id != class_id:
                continue
            matches.append((
                (center_x - previous_x) ** 2 + (center_y - previous_y) ** 2,
                -float(getattr(obj, "score", 0.0)),
                obj,
            ))
        if not matches:
            if anchor_attribute in ("_track_target_lock", "_search_target_lock"):
                with self._lock:
                    missing_frames = (
                        self._track_no_target_frames if anchor_attribute == "_track_target_lock"
                        else self._search_no_target_frames
                    ) + 1
                confirm_frames = (
                    config.UART_TRACK_LOST_CONFIRM_FRAMES if anchor_attribute == "_track_target_lock"
                    else config.UART_SEARCH_NO_TARGET_FRAMES
                )
                if missing_frames < confirm_frames:
                    self._selection_diagnostic_policy = (
                        "track_hold" if anchor_attribute == "_track_target_lock" else "search_hold"
                    )
                    # Keep identity only. submit publishes None for this frame;
                    # never extrapolate or retransmit old coordinates.
                    return None
            with self._lock:
                setattr(self, anchor_attribute, None)
                if anchor_attribute == "_final_target_lock":
                    self._last_target_center_x = None
                    self._last_target_center_y = None
                    self._last_target_area = None
                    self._last_target_accept_ms = 0
                # For 04 loss, retain the last observed position so RX44
                # can still seed the existing arrangement middle-X anchor.
            print("[{}] lost_lock=id{} action=reselect".format(log_tag, class_id))
            if anchor_attribute == "_search_target_lock":
                # The next fresh frame follows the normal search acquisition
                # branch. This frame must not publish a stale locked box.
                return None
            return self._select_unlocked_target(
                objects, labels, target_label, now_ms, anchor_attribute
            )
        return min(matches, key=lambda item: item[:2])[2]

    def _select_final_roi_target(self, objects, labels, target_label, now_ms):
        """Select closest center Y to ROI among unblocked, spaced targets."""
        if target_label not in TARGET_SELECTION_COMMANDS.values():
            return None

        movable_labels = set(TARGET_SELECTION_COMMANDS.values())
        movable_objects = []
        candidates = []
        for obj in objects:
            class_id = int(obj.class_id)
            if class_id <= 0 or class_id >= len(labels):
                continue
            label = labels[class_id]
            if label not in movable_labels:
                continue
            movable_objects.append(obj)
            if label in self._eligible_target_labels(target_label):
                candidates.append(obj)

        unblocked = [
            candidate for candidate in candidates
            if not self._final_candidate_is_blocked(candidate, movable_objects)
        ]

        eligible_candidates = [
            candidate
            for candidate in unblocked
            if self._matches_final_primary_spacing(
                candidate,
                movable_objects,
            )
        ]
        if not eligible_candidates:
            return None
        _, selection_reference_y = self._search_selection_reference()
        min_y_error = min(
            abs(float(obj.y) + float(obj.h) * 0.5 - selection_reference_y)
            for obj in eligible_candidates
        )
        eligible_candidates = [
            obj for obj in eligible_candidates
            if abs(float(obj.y) + float(obj.h) * 0.5 - selection_reference_y)
            == min_y_error
        ]

        with self._lock:
            previous_x = self._last_target_center_x
            previous_y = self._last_target_center_y
            previous_area = self._last_target_area
            previous_ms = self._last_target_accept_ms

        # Keep a still-valid tracked target to avoid frame-to-frame hopping.
        # If it is no longer eligible, choose from the new eligible set and
        # clear the old continuity anchor before accepting the deliberate switch.
        continuous = []
        if (
            previous_x is not None
            and previous_y is not None
            and previous_area is not None
            and now_ms - previous_ms < config.UART_X_JUMP_RESET_MS
        ):
            for candidate in eligible_candidates:
                center_x = float(candidate.x) + float(candidate.w) * 0.5
                center_y = float(candidate.y) + float(candidate.h) * 0.5
                area = max(1.0, float(candidate.w) * float(candidate.h))
                if abs(center_x - previous_x) > config.UART_MAX_X_JUMP_PX:
                    continue
                if abs(center_y - previous_y) > config.UART_MAX_Y_JUMP_PX:
                    continue
                if (
                    abs(area - previous_area) / max(1.0, previous_area)
                    > config.UART_MAX_AREA_CHANGE_RATIO
                ):
                    continue
                continuous.append(candidate)

        if continuous:
            return min(
                continuous,
                key=lambda obj: (
                    (
                        float(obj.x) + float(obj.w) * 0.5 - previous_x
                    ) ** 2
                    + (
                        float(obj.y) + float(obj.h) * 0.5 - previous_y
                    ) ** 2,
                    -float(getattr(obj, "score", 0.0)),
                ),
            )

        selected = min(
            eligible_candidates,
            key=lambda obj: (
                float(obj.x) + float(obj.w) * 0.5,
                -float(getattr(obj, "score", 0.0)),
            ),
        )
        with self._lock:
            self._last_target_center_x = None
            self._last_target_center_y = None
            self._last_target_area = None
            self._last_target_accept_ms = 0
        return selected

    @staticmethod
    def _final_spacing_limits(center_y, red_pair):
        """Return the X threshold for this Y band, with no Y-gap restriction."""
        max_delta_y = None
        if center_y > 290:
            min_delta_x = (
                config.UART_FINAL_RED_MIN_DX_Y_GT_290 if red_pair
                else config.UART_FINAL_PRIMARY_MIN_DX_Y_GT_290
            )
        elif center_y > 220:
            min_delta_x = (
                config.UART_FINAL_RED_MIN_DX_Y_GT_220 if red_pair
                else config.UART_FINAL_PRIMARY_MIN_DX_Y_GT_220
            )
        elif center_y > 150:
            # The user's later rule overrides the earlier dy<30 rule for this
            # duplicated Y band, so only dx is checked here.
            min_delta_x = (
                config.UART_FINAL_RED_MIN_DX_Y_GT_150 if red_pair
                else config.UART_FINAL_PRIMARY_MIN_DX_Y_GT_150
            )
        elif center_y > 100:
            min_delta_x = (
                config.UART_FINAL_RED_MIN_DX_Y_GT_100 if red_pair
                else config.UART_FINAL_PRIMARY_MIN_DX_Y_GT_100
            )
        elif center_y > 90:
            min_delta_x = (
                config.UART_FINAL_RED_MIN_DX_Y_GT_90 if red_pair
                else config.UART_FINAL_PRIMARY_MIN_DX_Y_GT_90
            )
        else:
            min_delta_x = (
                config.UART_FINAL_RED_MIN_DX_Y_LE_90 if red_pair
                else config.UART_FINAL_PRIMARY_MIN_DX_Y_LE_90
            )
        return min_delta_x, max_delta_y

    @staticmethod
    def _matches_final_primary_spacing(candidate, movable_objects):
        """Accept when every other movable box clears this target's X band."""
        center_x = float(candidate.x) + float(candidate.w) * 0.5
        center_y = float(candidate.y) + float(candidate.h) * 0.5
        for other in movable_objects:
            if other is candidate:
                continue
            other_x = float(other.x) + float(other.w) * 0.5
            red_pair = int(candidate.class_id) == 4 or int(other.class_id) == 4
            min_delta_x, _ = VisionSerialController._final_spacing_limits(
                center_y, red_pair
            )
            if abs(other_x - center_x) <= min_delta_x:
                return False
        return True

    def _select_nearest_label(
        self, objects, labels, required_label, reference_x=None
    ):
        if reference_x is None:
            reference_x = float(self.frame_width) * 0.5
        selected = None
        selected_key = None
        for obj in objects:
            class_id = int(obj.class_id)
            if class_id <= 0 or class_id >= len(labels):
                continue
            if labels[class_id] != required_label:
                continue
            center_x = float(obj.x) + float(obj.w) * 0.5
            score = float(getattr(obj, "score", 0.0))
            key = (abs(center_x - float(reference_x)), -score)
            if selected_key is None or key < selected_key:
                selected = obj
                selected_key = key
        return selected

    def send_event(self, event):
        """Queue one 5-byte event packet."""
        with self._lock:
            self._pending_events.append(build_event_packet(event))

    def _read_available(self):
        available = self._serial.available(0)
        if available <= 0:
            return
        try:
            data = self._serial.read(available, 0)
        except TypeError:
            data = self._serial.read(available)
        if data:
            self.process_received_bytes(data)

    def _next_outgoing_packet(self, now_ms):
        with self._lock:
            if self._pending_events:
                if (
                    self._last_event_write_ms is not None
                    and now_ms - self._last_event_write_ms
                    < config.UART_EVENT_MIN_INTERVAL_MS
                ):
                    # Do not transmit coordinates ahead of a pending handshake.
                    return None
                packet = self._pending_events.pop(0)
                if packet == build_event_packet(EVENT_FINAL_TARGET_IN_ROI):
                    self._last_final_roi_event_ms = now_ms
                elif packet == build_event_packet(EVENT_ZONE_OBSTACLE):
                    self._last_zone_obstacle_event_ms = now_ms
                elif (
                    packet == build_event_packet(EVENT_ROI_READY)
                    and self.mode == MODE_ARRANGE_RIGHTMOST
                    and self._arrangement_02_ack_pending
                ):
                    self._last_arrangement_02_event_ms = now_ms
                return packet

            if self.mode == MODE_SEARCH_ZONE and self._zone_obstacle_waiting:
                if (
                    now_ms - self._last_zone_obstacle_event_ms
                    >= config.UART_ZONE_OBSTACLE_REPEAT_MS
                ):
                    self._last_zone_obstacle_event_ms = now_ms
                    return build_event_packet(EVENT_ZONE_OBSTACLE)
                return None

            if (
                self.mode == MODE_FINAL_ROI_TRACK
                and self._final_roi_sent
                and now_ms - self._last_final_roi_event_ms
                >= config.UART_FINAL_ROI_EVENT_REPEAT_MS
            ):
                self._last_final_roi_event_ms = now_ms
                return build_event_packet(EVENT_FINAL_TARGET_IN_ROI)

            if (
                self._latest_packet is not None
                and now_ms - self._latest_submit_ms
                <= config.UART_TARGET_FRESHNESS_MS
            ):
                if config.UART_SEND_RATE_HZ <= 0:
                    if self._last_coordinate_sequence_sent == self._latest_submit_sequence:
                        return None
                    self._last_coordinate_sequence_sent = self._latest_submit_sequence
                self._coordinate_in_flight = (
                    self._latest_packet, self._latest_submit_sequence, self._latest_submit_ms, self.mode
                )
                return self._latest_packet

            if (
                self.mode == CMD_SEARCH
                and self._latest_submit_ms > 0
                and self._latest_packet is None
                and self._search_no_target_frames
                >= config.UART_SEARCH_NO_TARGET_FRAMES
            ):
                if not self._search_no_target_initial_sent:
                    self._search_no_target_initial_sent = True
                    self._search_no_target_acknowledged = False
                    self._last_search_no_target_ms = now_ms
                    return build_event_packet(EVENT_SEARCH_NO_TARGET)
                if (
                    self._search_no_target_acknowledged
                    and now_ms - self._last_search_no_target_ms
                    >= config.UART_NO_TARGET_REPEAT_MS
                ):
                    self._last_search_no_target_ms = now_ms
                    return build_event_packet(EVENT_SEARCH_NO_TARGET)

            if (
                self.mode == CMD_TRACK
                and self._latest_submit_ms > 0
                and self._latest_packet is None
                and self._track_no_target_frames >= config.UART_TRACK_LOST_CONFIRM_FRAMES
                and not self._track_no_target_sent
            ):
                # One E4 per confirmed loss, not a periodic retry. Fresh
                # reacquisition re-arms it except during first-green RX44 wait.
                self._track_no_target_sent = True
                self._last_coordinate_log_ms = None
                if (
                    self.target_label == "sqareredgreen"
                    and not self._first_green_near_view_decided
                ):
                    # Coordinates may reappear while MCU finishes its first
                    # green push; do not advance to 14 until its 44 handshake.
                    self._track_green_recovery_waiting = True
                return build_event_packet(EVENT_TRACK_NO_TARGET)

            if (
                self.mode == MODE_POST_ARRANGE_CHECK
                and self._latest_submit_ms > 0
                and self._latest_packet is None
                and (
                    self._last_arrangement_no_object_ms == 0
                    or now_ms - self._last_arrangement_no_object_ms
                    >= config.UART_NO_TARGET_REPEAT_MS
                )
            ):
                # MCU's interruptible recovery changes the camera viewpoint.
                # Keep checking fresh frames; a valid target emits 24 before
                # any coordinate packet and stops these E2 reports.
                self._last_arrangement_no_object_ms = now_ms
                return build_event_packet(EVENT_ARRANGEMENT_NO_OBJECT)

            if (
                self.mode == MODE_FINAL_ROI_TRACK
                and not self._final_roi_sent
                and self._latest_submit_ms > 0
                and self._latest_packet is None
                and not self._final_search_waiting_ack
                and self._final_no_target_frames
                >= config.UART_FINAL_NO_TARGET_FRAMES
            ):
                self._last_no_target_ms = now_ms
                self._final_search_waiting_ack = True
                return build_event_packet(EVENT_NO_TARGET)
        return None

    def _write_packet(self, packet):
        sent = self._serial.write(packet)
        if sent < 0:
            print("Vision UART write failed:", sent)
        elif sent != len(packet):
            print("Vision UART short write: {}/{}".format(sent, len(packet)))
        elif (
            len(packet) == COMMAND_FRAME_LENGTH
            and packet[:2] == CAM_HEADER
            and packet[-2:] == CAM_TAIL
        ):
            with self._lock:
                self._last_event_write_ms = _now_ms()
            event = int(packet[2])
            packet_hex = " ".join("{:02X}".format(value) for value in packet)
            print(
                "Vision UART TX command 0x{:02X}: {} [{}]".format(
                    event,
                    EVENT_NAMES.get(event, "UNKNOWN_EVENT"),
                    packet_hex,
                )
            )
            if config.VISION_DIAGNOSTICS_ENABLED:
                with self._lock:
                    state, frame = self.mode, self._latest_submit_sequence
                print("[TX-EVENT] t={} f={} state={:X} cmd={:02X} bytes={}/{}".format(
                    _now_ms(), frame, state, event, sent, len(packet)
                ))
        elif len(packet) == 11 and packet[:2] == CAM_HEADER and packet[-2:] == CAM_TAIL:
            now_ms = _now_ms()
            with self._lock:
                self._tx_coordinate_count += 1
                self._last_coordinate_write_ms = now_ms
                metadata = self._coordinate_in_flight
                count, state = self._tx_coordinate_count, self.mode
            if config.VISION_DIAGNOSTICS_ENABLED and (
                self._last_coordinate_log_ms is None
                or self._last_coordinate_log_class != packet[2]
                or now_ms - self._last_coordinate_log_ms >= config.VISION_DIAGNOSTICS_INTERVAL_MS
            ):
                self._last_coordinate_log_ms = now_ms
                self._last_coordinate_log_class = packet[2]
                frame, age_ms = "?", "?"
                if metadata is not None and metadata[0] == packet:
                    frame, age_ms, state = metadata[1], now_ms - metadata[2], metadata[3]
                print("[TX-COORD] t={} f={} state={:X} {} age_ms={} successful_count={} bytes={}/{} [{}]".format(
                    now_ms, frame, state, coordinate_details(packet), age_ms, count, sent, len(packet),
                    " ".join("{:02X}".format(value) for value in packet),
                ))

    def _report_io_health(self, now_ms):
        if not config.VISION_DIAGNOSTICS_ENABLED:
            return
        if now_ms - self._last_io_health_ms < 1000:
            return
        self._last_io_health_ms = now_ms
        with self._lock:
            frame_age = now_ms - self._last_frame_seen_ms if self._last_frame_seen_ms else None
            state, frame = self.mode, self._latest_submit_sequence
        if frame_age is not None and frame_age >= 1000:
            print("[IO-HEALTH] t={} f={} state={:X} no_new_frame_ms={} UART_alive=1 check_camera_yolo_main_loop".format(
                now_ms, frame, state, frame_age
            ))

    def _io_loop(self):
        interval = (
            1.0 / config.UART_SEND_RATE_HZ
            if config.UART_SEND_RATE_HZ > 0 else 0.0
        )
        next_send = python_time.monotonic()
        while not self._stop_event.is_set():
            # Clear before reading/publishing; a concurrent submit after this
            # point sets the event and cannot be lost before the wait.
            self._tx_wakeup.clear()
            try:
                self._read_available()
                self._report_io_health(_now_ms())
                now = python_time.monotonic()
                if now >= next_send:
                    packet = self._next_outgoing_packet(_now_ms())
                    if packet is not None:
                        self._write_packet(packet)
                    next_send = now + interval
            except Exception as exc:
                print("Vision UART I/O exception:", exc)
                self._stop_event.set()
                break

            self._tx_wakeup.wait(config.UART_READ_POLL_MS * 0.001)

    def close(self):
        self.enabled = False
        self._stop_event.set()
        self._tx_wakeup.set()
        if self._worker is not None:
            self._worker.join(timeout=1.0)
            self._worker = None
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception as exc:
                print("Vision UART close failed:", exc)
            self._serial = None
        print("Vision UART stopped")
