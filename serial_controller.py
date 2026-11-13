"""MaixCAM UART state machine and compact command/result protocol.

Frames have no checksum:

MCU -> CAM command (5 bytes):
    E1 E2 CMD 1E 2E

CAM -> MCU event (5 bytes):
    F1 F2 EVENT 1F 2F

CAM -> MCU coordinate (11 bytes):
    F1 F2 CLASS XH XL XS YH YL YS 1F 2F

CLASS is the stable non-zero vision/UART category ID (black is6 even when
the model's black label is at index7; see model_class_mapping). X/Y are unsigned big-endian
error magnitudes; XS/YS are 01 for zero/positive and 02 for negative.
"""

import struct
import math
import threading
import time as python_time

from maix import comm, uart

import app_config as config
from zone_obstacle_region import zone_obstacle_polygon, point_in_zone_obstacle_polygon
from zone_blackbar import select_zone_blackbar, blackbar_requires_realign
from tracking_roi import (
    final_track_reticle_point, point_in_near_view_roi, point_in_tracking_roi,
    roi_reticle_point, roi_triangle_vertices, search_reticle_point,
)
from vision_diagnostics import VisionDiagnostics, coordinate_details
from target_selector import (
    clipped_yolo_rect,
    expanded_target_neighbors,
    directional_target_neighbors,
    casualty_vertical_grab_allowed,
    select_nearest_horizontal_target,
    spacing_neighbor_in_y_scope,
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
CMD_ZONE_LOAD_RECHECK = 0x05
CMD_SEARCH_SWEEP_DONE = 0x13
CMD_TRACK_RECOVERY_DONE = 0x44
CMD_MICRO_FORWARD_DONE = 0x32
CMD_MICRO_SIDE_DONE = 0x42
CMD_FINAL_SEARCH_DONE = 0xF1
CMD_ZONE_OBSTACLE_DONE = 0x36
CMD_CASUALTY_LOAD_ADJUST_DONE = 0x45
CMD_ZONE_REALIGN_DONE = 0x46
CMD_TRACK_CENTER_DONE = 0x14
CMD_ARRANGE_FIRST_DONE = 0x12
CMD_ARRANGE_LEFT_DONE = 0x22
CMD_ARRANGE_SECOND_DONE = 0x24
CMD_FIRST_SUPPLY_CAPTURE_DONE = 0x34
CMD_RECHECK_SORT = 0xA5
CMD_RESET_TASK = 0xFE
CMD_STARTZONE_ESCAPE_DONE = 0xAE

# Commands 15/25 first enter a near-view count state. After one
# object remains, event 06 transitions internally to safety-zone searching.
MODE_NEAR_VIEW_CHECK = 0x05
MODE_SEARCH_ZONE = 0x06
MODE_WAIT_04_HANDSHAKE = 0x104
MODE_WAIT_14_HANDSHAKE = 0x114
MODE_WAIT_RED_SWITCH = 0x111
MODE_WAIT_SUPPLY_SEARCH = 0x103
MODE_WAIT_TASK_START = 0x1FE
MODE_ARRANGE_RIGHTMOST = 0x102
MODE_ARRANGE_LEFTMOST = 0x112
MODE_POST_ARRANGE_CHECK = 0x122
MODE_FINAL_ROI_TRACK = 0x24
MODE_MICRO_FORWARD = 0x32
MODE_MICRO_SIDE = 0x42
MODE_WAIT_STARTZONE_ESCAPE = 0xAE
MICRO_MODES = (MODE_MICRO_FORWARD, MODE_MICRO_SIDE)
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
    CMD_SELECT_BLACK_OBJECT: "triangualrblack",# UART ID 6 (new model index 7)
    CMD_SELECT_BLACK_OR_GREEN: TARGET_BLACK_OR_GREEN,
}
ZONE_SEARCH_COMMANDS = {
    CMD_SEARCH_RED_ZONE: "redsafety",          # model ID 2
    CMD_SEARCH_BLUE_ZONE: "bluesafety",        # model ID 1
}

# Outgoing events reserved by the scheme.
EVENT_ROI_READY = 0x02
EVENT_COLLECT_MORE_SUPPLIES = 0x03
EVENT_SEARCH_CENTER_REACHED = 0x04
EVENT_TRACK_CENTER_REACHED = 0x14
EVENT_MICRO_FORWARD = 0x32
EVENT_MICRO_SIDE = 0x42
EVENT_ARRANGE_FIRST_ACK = 0x12
EVENT_SKIP_TO_FINAL_TRACK = 0x24
EVENT_FINAL_TARGET_IN_ROI = 0x34
EVENT_GEOMETRY_CAPTURE = 0x64  # Qualified RX34 on the MCU; TX34 ACK unchanged.
EVENT_TALL_CASUALTY_CAPTURED = 0x54
EVENT_WIDE_CASUALTY_LOAD_ADJUST = 0x45
EVENT_TARGET_READY = 0x05
EVENT_ZONE_SEARCH_STARTED = 0x06
EVENT_ZONE_FOUND = 0x16
EVENT_ZONE_CLOSE = 0x26
EVENT_ZONE_OBSTACLE = 0x36
EVENT_ZONE_REALIGN = 0x46
EVENT_ZONE_NO_TARGET = 0xE6
EVENT_SWITCH_TO_RED_TARGET = 0x11
EVENT_ZONE_APPROACH = 0x07
EVENT_SINGLE_GREEN = 0xB5
EVENT_MULTI_GREEN = 0xC5
EVENT_NO_TASK_TARGET = 0xD5
EVENT_ARRANGEMENT_NO_OBJECT = 0xE2
EVENT_SEARCH_NO_TARGET = 0xE3
EVENT_TRACK_NO_TARGET = 0xE4
EVENT_FIRST_GREEN_EMPTY_RECOVERY = 0xE5  # First-green05 lacks ID5 for two rounds.
EVENT_UNCERTAIN = EVENT_FIRST_GREEN_EMPTY_RECOVERY  # Legacy reserved-event name.
EVENT_NO_TARGET = 0xEE
EVENT_STARTZONE_ESCAPE = 0xAE

OBJECT_GEOMETRY_ID_MASK = 0x0F
OBJECT_GEOMETRY_PRESENT = 0x80
OBJECT_GEOMETRY_NEAR = 0x40
OBJECT_GEOMETRY_OPEN_AREA = 0x20

EVENT_NAMES = {
    EVENT_STARTZONE_ESCAPE: "STARTZONE_TURN180_WAIT_ACK",
    EVENT_COLLECT_MORE_SUPPLIES: "COLLECT_MORE_SUPPLIES",
    EVENT_SWITCH_TO_RED_TARGET: "SWITCH_TO_RED_CASUALTY",
    EVENT_ROI_READY: "ROI_READY_OR_ADJUST",
    EVENT_SEARCH_CENTER_REACHED: "SEARCH_CENTER_REACHED",
    EVENT_TARGET_READY: "NEAR_VIEW_READY",
    EVENT_ZONE_SEARCH_STARTED: "ZONE_SEARCH_STARTED",
    EVENT_ARRANGE_FIRST_ACK: "ARRANGE_FIRST_ACK",
    EVENT_TRACK_CENTER_REACHED: "TRACK_CENTER_REACHED",
    EVENT_MICRO_FORWARD: "MICRO_FORWARD_WAIT32",
    EVENT_MICRO_SIDE: "MICRO_SIDE_WAIT42",
    EVENT_ZONE_FOUND: "ZONE_FOUND",
    EVENT_SKIP_TO_FINAL_TRACK: "SKIP_TO_FINAL_TRACK",
    EVENT_ZONE_CLOSE: "ZONE_CLOSE",
    EVENT_ZONE_OBSTACLE: "ZONE_OBSTACLE_WAIT_ACK",
    EVENT_ZONE_REALIGN: "ZONE_BLACKBAR_REALIGN_WAIT_ACK",
    EVENT_ZONE_NO_TARGET: "SAFE_ZONE_MISSING_OR_TOO_SMALL",
    EVENT_FINAL_TARGET_IN_ROI: "FINAL_TARGET_IN_ROI",
    EVENT_GEOMETRY_CAPTURE: "FINAL_TARGET_CENTER_AND_AREA_OK",
    EVENT_TALL_CASUALTY_CAPTURED: "TALL_CASUALTY_CAPTURED",
    EVENT_WIDE_CASUALTY_LOAD_ADJUST: "WIDE_CASUALTY_LOAD_ADJUST",
    EVENT_ZONE_APPROACH: "ZONE_APPROACH",
    EVENT_SINGLE_GREEN: "SINGLE_GREEN",
    EVENT_MULTI_GREEN: "MULTI_GREEN",
    EVENT_NO_TASK_TARGET: "NO_TASK_TARGET",
    EVENT_ARRANGEMENT_NO_OBJECT: "POST_22_NO_TARGET",
    EVENT_SEARCH_NO_TARGET: "SEARCH_NO_TARGET",
    EVENT_TRACK_NO_TARGET: "TRACK_NO_TARGET",
    EVENT_FIRST_GREEN_EMPTY_RECOVERY: "FIRST_GREEN_NO_GREEN05_RECOVERY",
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
    """Build one 11-byte coordinate packet using a stable vision/UART ID."""
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

    def __init__(self, frame_width, frame_height, tracking_roi=None, near_view_roi=None,
                 *, claw_geometry=False):
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.tracking_roi = tracking_roi
        self.near_view_roi = near_view_roi
        self._claw_geometry = claw_geometry
        self._capture_event_packet = build_event_packet(
            EVENT_GEOMETRY_CAPTURE if claw_geometry else EVENT_FINAL_TARGET_IN_ROI
        )
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._tx_wakeup = threading.Event()
        self._parser = CommandFrameParser()
        self._serial = None
        self._worker = None
        self._pending_events = []

        # Generations are transport fences: FE must advance, never reset them.
        self._command_generation = 0
        self._outgoing_generation = 0
        self._task_reset_epoch = 0
        self._last_io_health_ms = 0
        self._reset_task_state(int(config.UART_DEFAULT_MODE))
        self.enabled = False

        if config.UART_COORDINATE_ENABLED:
            self._open_and_start()

    def _reset_task_state(self, mode):
        """Initialize task history only; keep live UART/camera resources."""
        self._pending_events.clear()
        self.mode = int(mode)
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
        self._startzone_escape_phase = None
        self._startzone_escape_armed = True
        # Keep at most the latest15/25 received during AE. It must not run
        # until a fully sent AE request gets its matching MCU completion.
        self._startzone_deferred_zone_command = None
        self._red_switch_stage = None
        self._red_switch_candidate = None
        self._red_switch_request_sent = False
        self._red_switch_retry_at_ms = None
        self._red_priority_tracking = False
        self._last_tracked_class_id = None
        self._last_frame_seen_ms = 0
        self._last_no_target_ms = 0
        self._final_search_waiting_ack = False
        self._final_no_target_frames = 0
        self._final_target_lock = None
        self._direct_capture_pending = False
        self._track_target_lock = None
        self._search_target_lock = None
        self._track_no_target_frames = 0
        self._track_no_target_sent = False
        self._track_green_recovery_waiting = False
        self._last_search_no_target_ms = 0
        self._search_no_target_frames = 0
        self._search_no_target_initial_sent = False
        self._search_no_target_acknowledged = False
        self._search_e3_active = False
        self._tracking_missing_since_ms = None
        self._tracking_missing_class_id = None
        self._last_arrangement_no_object_ms = 0
        self._roi_entry_sent = False
        self._track_center_sent = False
        self._final_roi_sent = False
        self._casualty_capture_streak = None
        self._final_capture_ready_ms = 0
        self._final_capture_tall_casualty = False
        self._final_capture_followup = None
        self._last_final_roi_event_ms = 0
        self._arrangement_object_count = 0
        self._arrangement_no_neighbor_frames = 0
        self._arrangement_ready_ms = 0
        self._initial_arrangement_decision_pending = False
        self._reset_micro_adjust()
        self._reset_initial_arrangement_confirmation()
        self._arrangement_02_ack_pending = False
        self._arrangement_02_request_sent = False
        self._arrangement_02_without_side = False
        self._arrangement_side_target_available = False
        self._last_arrangement_02_event_ms = 0
        self._zone_found_sent = False
        self._zone_area_ratio = 0.0
        self._zone_no_target_phase = None
        self._zone_close_sent = False
        self._zone_close_active = False
        self._last_zone_close_event_ms = None
        self._zone_close_confirm_frames = 0
        self._zone_close_confirm_seen_ms = None
        self._zone_realign_phase = None
        self._last_zone_realign_write_ms = None
        self._zone_obstacle_waiting = False
        self._zone_obstacle_request_sent = False
        self._zone_obstacle_area_allowed = False
        self._zone_obstacle_armed = True
        self._zone_obstacle_frames = 0
        self._zone_obstacle_clear_frames = 0
        self._last_zone_obstacle_event_ms = 0
        # None means05 has not yet been written successfully. Never count
        # queueing time (or a partial UART write) as camera settling time.
        self._near_view_started_ms = None
        self._near_view_adjust_sent = False
        self._near_view_result_sent = False
        self._near_view_stable_bucket = None
        self._near_view_stable_frames = 0
        # RX05 during06 is an evacuation load recheck, not another capture
        # or supplemental attempt. Camera settling starts on actual TX05.
        self._zone_load_recheck = None
        self._zone_load_recheck_source = None
        self._zone_load_resume_ready_ms = 0
        # Count completed empty05 decisions across24/re-capture handshakes.
        self._near_view_empty_cycles = 0
        # First-green recovery counts consecutive05 rounds without ID5,
        # including rounds containing only foreign-color objects.
        self._near_view_green_missing_cycles = 0
        self._supply_collect_attempts = 0
        self._supply_collect_started_ms = None
        self._supply_search_ready_ms = None
        self._supply_first_capture_bypassed = False
        self._supply_collect_delivery_pending = False
        # Delivery-level latch, independent of the active timer/cargo mask.
        # Final05 may reject cargo and send02/24; that does not buy more attempts.
        self._supply_collect_finished = False
        self._supply_collect_finish_reason = None
        # Wide-view cargo identities are only used to distinguish a NEW ROI
        # entry from already loaded supplies, never to publish coordinates.
        self._supply_loaded_roi_anchors = []
        self._supply_roi_observed = False
        self._supply_roi_handoff = None
        self._supply_roi_handoff_seen_ms = 0
        self._reset_supply_acquisition_guard()
        self._first_green_near_view_decided = False
        self._first_green_both_sides_required = False
        # MCU signals the post-07 phase through red/black/mixed selection.
        self._post07_started = False
        self._post07_supply_tracking = False
        self._last_target_center_x = None
        self._last_target_center_y = None
        self._last_target_area = None
        self._last_target_accept_ms = 0
        self._middle_target_anchor = None
        self._arrangement_target_anchor = None
        self._last_arrangement_debug_ms = 0

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
                "near05_ids_v1 near05_actual_tx_settle250_confirm3_v3 final24_front_block_v1 "
                "near05_empty24_v1 final24_center20_v1 "
                "near05_blue_without_target24_v1 "
                "first_green_near05_no_green2_E5_wait44_v3 "
                "post07_51_collect2_total12s_v1 "
                "post07_first_supply_rx34_skip05_v1 "
                "supply51_finished_latch_across_arrange_v1 "
                "post07_51_direct03_no_precheck_v2 "
                "post07_51_roi_exclude_all_tracks34_handoff_v2 "
                "post07_51_inner_half_cargo_mask_v3 "
                "zone06_rx05_load_recheck_v1 "
                "before06_final_inventory05_all_paths_v1 supply51_mcu05_final_inventory_v1 "
                "middle_reacquire12_x_v1 arrange02_original_lock_frame_center_coords_v2 wide_after05_settle300_v1 arrange12_open_loop_v1 "
                "e2_rx24_v1 repeat34_500ms_v1 "
                "target_cmds_01_11_21_31_v1 repeat34_while_centered_v2 "
                "send02_once_v1 pre02_merged24_or02_v1 "
                "mcu_FE_task_reset_wait03_v1 "
                "startzone_AE_disabled_by_default_v2 ae_defer15_25_until_ack_v2 "
                "first_green_pre14_confirm2_500ms_v2 first_green44_resume03_fresh_v1 first_green14_xbox_overlap5to75_v2 "
                "post07_supply14_xbox_overlap5to75_v2 "
                "supply14_xybox_overlap5to75_v3 "
                "direction14_xy_padding5to75_v1 "
                "direction14_H32_V42_HV02_corner24_red_v3 "
                "red14_vertical_front_or_w_over_h_gt1p5_grab24_v3 "
                "red14_upper_diagonals_vertical_v1 "
                "micro42_roi_center_alternate_sweep_mcu_timed_v3 "
                "micro32_42_mcu_completion_no_min_wait42_11s_v2 "
                "supply51_neighbor_green_black_no_scatter_v1 "
                "roi03_04_no_coords14_no_neighbors24_keep34_v1 "
                "all_tasks_new_target3_zone_hold500_v2 "
                "first_green_e4_resume04_keep_lock_v1 "
                "black_loss800ms_v1 auto_red11_disabled_by_default_v1 "
                "final24_first_target_lock_v1 "
                "final24_lock_no_jump_area_gate_v1 "
                "track04_lock_no_jump_area_gate_v1 "
                "search03_to04_inherit_lock_v1 track04_missing5_e4_episode_v1 "
                "vision_diagnostics_v1 "
                "first_green03_04_frame_center_v1 "
                "search03_lock_raw_yolo_follow_v1 "
                "uart_frame_wakeup_fast_v1 "
                "final24_roi_entry34_v1 red24_deep_roi_any_after07_confirm3_v2 "
                "final_spacing_red_near_all_y_far_same_band_v6 search03_e3_ack13_timeout8s_v3 "
                "search03_missing5_v2 search03_roi_yx_v1 zone36_disabled_v3 "
                "exclude_safe_zone_targets_x100_v2 search03_outside_roi_v1 zone36_trapezoid_clip_y200_v6 zone36_area_le25_v1 "
                "zone_supply_left30_casualty_right30_near_unchanged_v5 final_target_roi_y_nearest_v1 "
                "zone16_supply_near_center_half_v1 "
                "zone_min2pct_E6_v2 "
                "near05_red_foreign_arrange_v1 near05_black_green_rules_v2 "
                "zone36_accumulated3_v1 mcu_red11_selection_preserved_v1"
                " zone36_rearm_after_ack_v2 zone_min_area_gt3_E6_once_v1"
                " zone46_blackbar_zone_cx_once_ack_fresh_v1"
                " zone26_confirm5_any_wrong_bar46_v2"
                " tracking_roi_triangle_top_mid_base_unchanged_v1"
            )
            if config.VISION_DIAGNOSTICS_ENABLED and config.VISION_DIAGNOSTICS_VERBOSE:
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
            if config.VISION_DIAGNOSTICS_ENABLED and config.VISION_DIAGNOSTICS_VERBOSE:
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
        received_command = command
        changed = False
        description = None
        search_no_target_ack = False
        direct_capture = False
        with self._lock:
            if command == CMD_RESET_TASK:
                # FE is authoritative in EVERY stage, before inventory/switch
                # guards. Do not reopen UART or rewind an old frame's fence.
                previous_mode = self.mode
                previous_target = self.target_label
                previous_post07 = self._post07_started
                queued_events = len(self._pending_events)
                self._command_generation += 1
                self._task_reset_epoch += 1
                self._reset_task_state(MODE_WAIT_TASK_START)
                self.target_label = "sqareredgreen"
                self._tx_wakeup.set()
                print("[TASK-RESET] RXFE previous_state={:X} target={} post07={} cleared_events={} epoch={} wait_RX21_RX03 no_reply".format(
                    previous_mode, previous_target, previous_post07,
                    queued_events, self._task_reset_epoch,
                ))
                return
            if command == CMD_STARTZONE_ESCAPE_DONE:
                if self._startzone_escape_phase == "wait_ack":
                    deferred_zone_command = self._startzone_deferred_zone_command
                    self._startzone_deferred_zone_command = None
                    self._clear_supply_tracking_output()
                    self._reset_initial_arrangement_confirmation()
                    self._pending_events.clear()
                    self._coordinate_in_flight = None
                    self._last_tracked_class_id = None
                    self._post07_supply_tracking = False
                    self._middle_target_anchor = None
                    self._arrangement_target_anchor = None
                    self._last_frame_seen_ms = 0
                    self.mode = CMD_SEARCH
                    self._startzone_escape_phase = None
                    self._command_generation += 1
                    self._tx_wakeup.set()
                    if deferred_zone_command in ZONE_SEARCH_COMMANDS:
                        # Reuse the normal15/25 handler BELOW, under this same
                        # lock. Do not recursively acquire the non-reentrant
                        # lock or allow an intermediate03 steering packet.
                        command = deferred_zone_command
                        received_command = command
                        print("[STARTZONE-AE] RXAE turn180_done apply_deferred_RX{:02X} enter05 target={} coordinates=paused".format(
                            command, self.target_label))
                    else:
                        print("[STARTZONE-AE] RXAE turn180_done resume03 target={} fresh_frame_required".format(
                            self.target_label))
                        return
                else:
                    print("[STARTZONE-AE] RXAE ignored no_sent_request")
                    return
            if self._startzone_escape_phase is not None:
                # FE above is authoritative.15/25 is remembered, not executed;
                # no handshake or target change interrupts the180deg turn.
                if command in ZONE_SEARCH_COMMANDS:
                    self._startzone_deferred_zone_command = command
                    print("[STARTZONE-AE] RX{:02X} deferred wait_RXAE latest_zone={} coordinates=paused".format(
                        command, ZONE_SEARCH_COMMANDS[command]))
                else:
                    print("[STARTZONE-AE] RX{:02X} ignored wait_RXAE coordinates=paused".format(command))
                return
            if (self.mode == MODE_WAIT_TASK_START
                    and command not in TARGET_SELECTION_COMMANDS
                    and command != CMD_SEARCH):
                print("Vision UART command 0x{:02X}: ignored wait_new_task_after_FE".format(command))
                return
            self._clear_disabled_zone_obstacle()
            if command in (CMD_MICRO_FORWARD_DONE, CMD_MICRO_SIDE_DONE):
                if self.mode != command or not self._micro_request_sent:
                    print("[MICRO] RX{:02X} ignored no_matching_sent_action".format(command))
                    return
                # MCU owns motion duration and replies only after completion.
                #Neither32 nor42 enforces a historical minimum motion time.
                #Judge again using only a NEW frame after the matching ACK.
                self.mode = MODE_WAIT_14_HANDSHAKE
                self._micro_request_sent = False
                self._micro_started_ms = None
                self._micro_decision_ms = _now_ms()
                self._micro_loss_pending = False
                command = CMD_TRACK_CENTER_DONE
            elif (self.mode in MICRO_MODES and command not in TARGET_SELECTION_COMMANDS
                  and command != CMD_ZONE_LOAD_RECHECK):
                print("[MICRO] RX{:02X} ignored wait_RX{:02X}".format(command, self.mode))
                return
            if command == CMD_ZONE_REALIGN_DONE:
                if self.mode == MODE_SEARCH_ZONE and self._zone_realign_phase == "wait_ack":
                    self._reset_zone_approach()
                    self._zone_realign_phase = "done"
                    self._pending_events.clear()
                    self._latest_packet = None
                    self._latest_submit_ms = 0
                    self._command_generation += 1
                    self._tx_wakeup.set()
                    print("[ZONE-46] RX46 maneuver_done resume_fresh_zone_frame skip_blackbar_this_delivery=1")
                else:
                    print("[ZONE-46] RX46 ignored no_sent_request")
                return
            if (self.mode == MODE_SEARCH_ZONE and self._zone_realign_phase == "wait_ack"
                    and command not in TARGET_SELECTION_COMMANDS
                    and command != CMD_ZONE_LOAD_RECHECK):
                # A late14/25/36 is not completion of the46 maneuver.
                print("[ZONE-46] RX{:02X} ignored wait_RX46".format(command))
                return
            if command == CMD_ZONE_OBSTACLE_DONE and not self._zone_obstacle_enabled():
                print("Vision UART command 0x36: ignored obstacle_bypass_disabled")
                return
            if command == CMD_CASUALTY_LOAD_ADJUST_DONE:
                if self._zone_load_recheck == "wait45ack":
                    # Resume with the confirmed casualty from the final05.
                    self._zone_load_recheck = "count05"
                    result = self._finish_zone_load_recheck([4])
                    print("[LOAD-CHECK] RX45 adjustment_done {}".format(result))
                else:
                    print("[LOAD-CHECK] RX45 ignored no_sent_request")
                return
            if command == CMD_ZONE_LOAD_RECHECK:
                if self._zone_load_recheck in ("wait05tx", "count05"):
                    # An MCU retry may need another acknowledgment, but must
                    # not restart the same inventory or camera settling time.
                    reply = build_event_packet(EVENT_TARGET_READY)
                    if reply not in self._pending_events:
                        self._pending_events.append(reply)
                    self._tx_wakeup.set()
                    print("[ZONE-RECHECK] duplicate_RX05 reply05 keep_inventory")
                elif ((self._zone_load_recheck is None or self._zone_load_recheck_source == "supply_empty")
                      and self._can_request_supply_final_inventory()):
                    # MCU's collection deadline is authoritative even if a
                    # previous empty05 already cancelled our local budget.
                    # This is the FINAL inventory, not an ordinary capture
                    # check or evacuation's two-check delivery sequence.
                    print("[COLLECT-51] RX05 final_inventory_requested state={:X}".format(self.mode))
                    self._begin_zone_load_recheck("before06")
                elif self.mode == MODE_SEARCH_ZONE and self._zone_load_recheck is None:
                    self._begin_zone_load_recheck()
                else:
                    print("[ZONE-RECHECK] RX05 ignored state={:X} phase={}".format(
                        self.mode, self._zone_load_recheck))
                return
            if command == CMD_FIRST_SUPPLY_CAPTURE_DONE:
                if (self.mode == MODE_FINAL_ROI_TRACK
                        and self._zone_load_recheck is None
                        and self._post07_started
                        and self.target_label in ("sqareredgreen", "triangualrblack", TARGET_BLACK_OR_GREEN)
                        and self._final_capture_followup == "done"
                        and self._supply_collect_started_ms is None
                        and not self._supply_collect_finished
                        and not self._supply_first_capture_bypassed):
                    self.target_label = TARGET_BLACK_OR_GREEN
                    self._supply_first_capture_bypassed = True
                    self._coordinate_in_flight = None
                    self._request_supply_collection(_now_ms(), mcu_started=True)
                    self._tx_wakeup.set()
                else:
                    print("[COLLECT-51] RX34 ignored no_matching_first_supply_capture")
                return
            if (self._zone_load_recheck is not None
                    and (command not in TARGET_SELECTION_COMMANDS
                         or TARGET_SELECTION_COMMANDS[command] == self.target_label)):
                print("[ZONE-RECHECK] RX{:02X} ignored phase={} coordinates=paused".format(
                    command, self._zone_load_recheck))
                return
            obsolete_collect_commands = (
                CMD_SELECT_BLACK_OR_GREEN, CMD_SORT, CMD_RECHECK_SORT, CMD_SEARCH, CMD_TRACK,
                CMD_TRACK_CENTER_DONE, CMD_TRACK_RECOVERY_DONE,
                CMD_ARRANGE_FIRST_DONE, CMD_ARRANGE_LEFT_DONE,
                CMD_ARRANGE_SECOND_DONE, CMD_SEARCH_SWEEP_DONE,
                CMD_FINAL_SEARCH_DONE, CMD_SEARCH_RED_ZONE, CMD_SEARCH_BLUE_ZONE,
            )
            if (self._supply_collect_delivery_pending and self.mode == MODE_SEARCH_ZONE
                    and not self._zone_close_sent and command in obsolete_collect_commands):
                print("[COLLECT-51] late_RX{:02X} ignored already_TX06".format(command))
                return
            if (self.mode == MODE_WAIT_SUPPLY_SEARCH
                    and command not in TARGET_SELECTION_COMMANDS and command != CMD_SEARCH):
                print("[COLLECT-51] RX{:02X} ignored wait_RX51_RX03".format(command))
                return
            supplement_resume = self.mode == MODE_WAIT_SUPPLY_SEARCH and command == CMD_SEARCH
            if (self._supply_search_ready_ms is not None
                    and command in (CMD_SELECT_BLACK_OR_GREEN, CMD_SEARCH)):
                # Retransmitted51/03 must not restart the same settling window.
                return
            if self._supply_collect_delivery_pending and (
                    command in TARGET_SELECTION_COMMANDS or command in ZONE_SEARCH_COMMANDS):
                self._supply_collect_delivery_pending = False
            if self._red_switch_stage is not None:
                if command == CMD_SELECT_RED_OBJECT:
                    self._red_switch_stage = "wait03"
                    self._red_switch_retry_at_ms = None
                    self._pending_events.clear()
                    self._latest_packet = None
                    self._latest_submit_ms = 0
                    self._command_generation += 1
                    self._tx_wakeup.set()
                    print("[RED-SWITCH] RX11 accepted keep_candidate={} wait_RX03 coordinates=paused".format(
                        self._red_switch_candidate,
                    ))
                    return
                if command == CMD_SEARCH:
                    if self._red_switch_stage != "wait03":
                        print("[RED-SWITCH] RX03 ignored before_RX11 coordinates=paused")
                        return
                    self.mode = CMD_SEARCH
                    self.target_label = "sqarered"
                    self._post07_started = True
                    self._post07_supply_tracking = False
                    self._search_target_lock = self._red_switch_candidate
                    self._red_switch_stage = None
                    self._red_switch_candidate = None
                    self._red_switch_request_sent = False
                    self._search_no_target_frames = 0
                    self._search_no_target_initial_sent = False
                    self._search_no_target_acknowledged = False
                    self._last_search_no_target_ms = 0
                    self._tracking_missing_since_ms = None
                    self._tracking_missing_class_id = None
                    self._pending_events.clear()
                    self._latest_packet = None
                    self._latest_submit_ms = 0
                    self._command_generation += 1
                    self._tx_wakeup.set()
                    print("[RED-SWITCH] RX03 resume=id4 reference={} keep_candidate={} fresh_frame_required".format(
                        self._reference_point(CMD_SEARCH), self._search_target_lock,
                    ))
                    return
                if command not in TARGET_SELECTION_COMMANDS:
                    print("[RED-SWITCH] RX{:02X} ignored stage={} coordinates=paused".format(
                        command, self._red_switch_stage,
                    ))
                    return
                # An explicit new class/task command cancels this handshake.
                if (self._red_switch_stage == "wait11" and self._red_switch_request_sent
                        and command in (CMD_SELECT_BLACK_OBJECT, CMD_SELECT_BLACK_OR_GREEN)):
                    print("[RED-SWITCH] RX{:02X} declined_TX11 keep_material rearm_on_new_E3".format(command))
                self._red_switch_stage = None
                self._red_switch_candidate = None
                self._red_switch_request_sent = False
                self._red_switch_retry_at_ms = None
                self._red_priority_tracking = False
                self._pending_events.clear()
                self.mode = CMD_SEARCH
            elif (command == CMD_SELECT_RED_OBJECT and self._red_priority_tracking
                  and self.target_label == "sqarered"):
                print("[RED-SWITCH] duplicate_RX11 ignored keep_red_task_and_lock")
                return
            final_track_restart = command == CMD_SEARCH and self.mode == MODE_FINAL_ROI_TRACK
            first_green_recovery_search = (
                command == CMD_TRACK_RECOVERY_DONE
                and self.target_label == "sqareredgreen"
                and not self._post07_started
                and self.mode in (CMD_TRACK, MODE_WAIT_14_HANDSHAKE)
            )
            if first_green_recovery_search:
                # First-green E4/E5 recovery now restarts the normal03 flow,
                # not14's arrangement decision. Reuse the ordinary search
                # reset/fresh-frame fence below; this is NOT a new mission.
                self._pending_events.clear()
                self._coordinate_in_flight = None
                self._track_no_target_sent = False
                self._first_green_both_sides_required = False
                command = CMD_SEARCH
            red_track_resume = (
                command == CMD_TRACK and self.target_label == "sqarered"
                and self.mode in (CMD_SEARCH, CMD_TRACK, MODE_WAIT_14_HANDSHAKE)
            )
            if red_track_resume and self.mode == CMD_SEARCH:
                self._track_target_lock = self._search_target_lock
            recovery_track_resume = (
                command == CMD_TRACK and self.mode == CMD_TRACK
                and self._track_green_recovery_waiting
            ) or red_track_resume
            inherit_track_lock = (
                command == CMD_TRACK
                and (self.mode == MODE_WAIT_04_HANDSHAKE or recovery_track_resume)
                and self._track_target_lock is not None
            )
            inherit_collect_lock = (
                self._supply_collect_started_ms is not None
                and command in (CMD_TRACK_CENTER_DONE, CMD_TRACK_RECOVERY_DONE)
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
                self._command_generation += 1
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
                self._zone_obstacle_request_sent = False
                self._zone_obstacle_area_allowed = False
                # A completed maneuver may leave the same/new obstruction.
                # Start a fresh three-hit round without requiring clear frames.
                self._zone_obstacle_armed = True
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
                self._zone_close_active = False
                self._last_target_center_x = None
                self._last_target_center_y = None
                self._last_target_area = None
                self._last_target_accept_ms = 0
                self._command_generation += 1
                print("Vision UART command 0x36: bypass_done=resume_zone_coordinates obstacle_rearmed=1 hits=0")
                return
            if (received_command == CMD_TRACK_CENTER_DONE
                    and self.mode == MODE_WAIT_14_HANDSHAKE
                    and getattr(config, "UART_DIRECT_CAPTURE_AFTER_14", True)):
                self.mode = MODE_FINAL_ROI_TRACK
                self._post07_supply_tracking = self._post07_started and self.target_label in (
                    "sqareredgreen", "triangualrblack", TARGET_BLACK_OR_GREEN,
                )
                self._pending_events.clear()
                self._reset_micro_adjust()
                self._reset_initial_arrangement_confirmation()
                direct_capture = True
                description = "direct_capture=TX24"
                changed = True
            elif (
                (
                    command == CMD_TRACK_CENTER_DONE
                    and self.mode == MODE_WAIT_14_HANDSHAKE
                )
                or (
                    command == CMD_TRACK_RECOVERY_DONE
                    and self.mode in (CMD_TRACK, MODE_WAIT_14_HANDSHAKE)
                )
            ):
                # RX14 and later-task RX44 start the same arrangement
                # decision (pre07 green44 was remapped to03 above). Preserve
                # X identity and reacquire
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
                if command == CMD_TRACK_CENTER_DONE and self._post07_started:
                    self._post07_supply_tracking = self.target_label in (
                        "sqareredgreen", "triangualrblack", TARGET_BLACK_OR_GREEN
                    )
                description = "merged_check=TX24_if_valid_else_TX02"
                if command == CMD_TRACK_RECOVERY_DONE:
                    description = "recovery_done=" + description
                self._arrangement_object_count = 0
                self._arrangement_no_neighbor_frames = 0
                self._arrangement_ready_ms = 0
                self._arrangement_target_anchor = None
                self._pending_events.clear()
                self._initial_arrangement_decision_pending = True
                self._reset_initial_arrangement_confirmation()
                if self._uses_expanded_overlap_decision(self.target_label):
                    self._initial_confirmation_anchor = self._track_target_lock
                    description = "directional_check=HV02_H32_V42_corner_or_clear24"
                    if command == CMD_TRACK_RECOVERY_DONE:
                        description = "recovery_done=" + description
                if self._uses_first_green_center_selection(self.target_label):
                    self._initial_confirmation_started_ms = _now_ms()
                    self._initial_confirmation_anchor = self._track_target_lock
                    description = "first_green_directional_confirm2_missing500ms=02_32_42_24_orE4"
                    if command == CMD_TRACK_RECOVERY_DONE:
                        description = "recovery_done=" + description
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
                if final_track_restart:
                    # MCU has stopped the 24 approach and switched references.
                    # Drop queued34/EE and preserve the current task/budget.
                    self._pending_events.clear()
                    self._coordinate_in_flight = None
                if recovery_track_resume:
                    # MCU has explicitly ended recovery and resumed center
                    # tracking (e.g. a green360 search interrupted by a target).
                    # Drop the prior recovery event but keep any reacquired
                    # target identity. Only a new image can publish coordinates
                    # or emit14; coordinates alone never end timed recovery.
                    self._pending_events.clear()
                if command == CMD_SEARCH:
                    self._post07_supply_tracking = False
                    self._red_priority_tracking = False
                if (
                    self.target_label == "triangualrblack"
                    and command in (CMD_SEARCH, CMD_TRACK)
                    and not inherit_track_lock
                    and not final_track_restart
                ):
                    self.mode = MODE_FINAL_ROI_TRACK
                    self._pending_events.clear()
                    description = "black_target=direct_FINAL_ROI_TRACK"
                else:
                    self.mode = command
                    description = "mode=" + MODE_COMMANDS[command]
                    if inherit_track_lock:
                        description += " {}=id{}".format(
                            "retain_recovery_lock" if recovery_track_resume else "inherit_03_lock",
                            self._track_target_lock[0]
                        )
                    if recovery_track_resume:
                        description += " recovery_resume=RX04 wait44=False fresh_frame_required"
                    if first_green_recovery_search:
                        description += " first_green_recovery_done=RX44 reference={} fresh_frame_required".format(
                            self._reference_point(CMD_SEARCH))
                    if final_track_restart:
                        description += " final24_restart=RX03 fresh_frame_required"
                changed = True
            elif command in TARGET_SELECTION_COMMANDS:
                new_task = (
                    TARGET_SELECTION_COMMANDS[command] != self.target_label
                    or (command == CMD_SELECT_BLACK_OR_GREEN
                        and self.mode == MODE_SEARCH_ZONE and self._zone_close_sent)
                )
                if command != CMD_SELECT_BLACK_OR_GREEN or new_task:
                    self._reset_supply_collection(new_task=new_task)
                if new_task:
                    self._startzone_escape_armed = True
                if self.target_label != TARGET_SELECTION_COMMANDS[command]:
                    self._near_view_empty_cycles = 0
                    self._near_view_green_missing_cycles = 0
                self._red_priority_tracking = False
                self._first_green_both_sides_required = False
                if command in (CMD_SELECT_RED_OBJECT, CMD_SELECT_BLACK_OBJECT,
                               CMD_SELECT_BLACK_OR_GREEN):
                    self._post07_started = True
                self._post07_supply_tracking = False
                self._pending_events.clear()
                self.target_label = TARGET_SELECTION_COMMANDS[command]
                description = "target=" + self.target_label
                if command == CMD_SELECT_BLACK_OBJECT and self.mode != MODE_WAIT_TASK_START:
                    self.mode = MODE_FINAL_ROI_TRACK
                    self._pending_events.clear()
                    description += " mode=FINAL_ROI_TRACK"
                changed = True
            if changed:
                self._supply_search_ready_ms = None
                if supplement_resume and self._uses_supply_collection():
                    self._pending_events.clear()
                    settle_ms = getattr(config, "UART_SUPPLY_SEARCH_SETTLE_MS", 250)
                    if settle_ms > 0:
                        self._supply_search_ready_ms = _now_ms() + settle_ms
                        description += " wide_settle={}ms".format(settle_ms)
                # Confirmation evidence must not span command/frame fences.
                # Inherited locks remain exempt; keep zone memory across
                # same-view recovery ACKs, never across tasks/camera changes.
                self._supply_outside_history = []
                self._new_target_confirmation_pending = False
                if (command in TARGET_SELECTION_COMMANDS or command == CMD_SEARCH
                        or command in ZONE_SEARCH_COMMANDS):
                    self._supply_recent_zones = []
                if (command in TARGET_SELECTION_COMMANDS or command == CMD_SEARCH
                        or command in ZONE_SEARCH_COMMANDS):
                    self._reset_micro_adjust()
                self._zone_load_recheck = None
                self._zone_load_recheck_source = None
                self._zone_load_resume_ready_ms = 0
                self._command_generation += 1
                self._supply_roi_handoff = None
                self._supply_roi_handoff_seen_ms = 0
                if command in TARGET_SELECTION_COMMANDS or command == CMD_SEARCH or command in ZONE_SEARCH_COMMANDS:
                    # New task or camera reframe: the next wide-view image
                    # establishes which supplies are already inside the ROI.
                    self._supply_loaded_roi_anchors = []
                    self._supply_roi_observed = False
                if command in TARGET_SELECTION_COMMANDS or command == CMD_SEARCH:
                    self._last_tracked_class_id = None
                if command not in (CMD_TRACK_CENTER_DONE, CMD_TRACK_RECOVERY_DONE):
                    self._reset_initial_arrangement_confirmation()
                # New commands start a new selection round. RXF1 returns
                # above and deliberately preserves the locked object.
                self._direct_capture_pending = direct_capture
                self._final_target_lock = self._track_target_lock if direct_capture else None
                self._final_capture_tall_casualty = False
                self._final_capture_followup = None
                if not search_no_target_ack:
                    self._search_target_lock = None
                    self._tracking_missing_since_ms = None
                    self._tracking_missing_class_id = None
                if not (inherit_track_lock or inherit_collect_lock):
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
                self._search_e3_active = False
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
                    self._arrangement_02_request_sent = False
                    self._arrangement_side_target_available = False
                    self._last_arrangement_02_event_ms = 0
                self._arrangement_no_neighbor_frames = 0
                if self._initial_arrangement_decision_pending:
                    self._arrangement_02_request_sent = False
                if self.mode not in (
                    MODE_ARRANGE_RIGHTMOST,
                    MODE_ARRANGE_LEFTMOST,
                ):
                    self._arrangement_ready_ms = 0
                    self._middle_target_anchor = None
                    self._arrangement_target_anchor = None
                self._zone_found_sent = False
                self._zone_area_ratio = 0.0
                self._zone_no_target_phase = None
                self._zone_close_sent = False
                self._zone_close_active = False
                self._last_zone_close_event_ms = None
                self._zone_close_confirm_frames = 0
                self._zone_close_confirm_seen_ms = None
                self._zone_realign_phase = None
                self._last_zone_realign_write_ms = None
                self._zone_obstacle_waiting = False
                self._zone_obstacle_request_sent = False
                self._zone_obstacle_area_allowed = False
                self._zone_obstacle_armed = True
                self._zone_obstacle_frames = 0
                self._zone_obstacle_clear_frames = 0
                self._last_zone_obstacle_event_ms = 0
                self._near_view_started_ms = None
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
            print("Vision UART command 0x{:02X}: {}".format(received_command, description))
        else:
            print("Vision UART ignored unknown command 0x{:02X}".format(command))

    def frame_generation(self):
        """Snapshot before camera.read(), not after detection completes."""
        with self._lock:
            return self._command_generation

    def frame_context(self):
        """Atomically snapshot the frame fence and FE verifier-reset epoch."""
        with self._lock:
            if self._supply_search_ready_ms is not None:
                self._supply_search_settling(_now_ms())
            return self._command_generation, self._task_reset_epoch

    def submit(self, objects, labels, raw_objects=None, verification_diagnostics=None,
               frame_generation=None, blackbar_objects=None, startzone_objects=None):
        """Publish the newest selected result and return it for display.

        ``objects`` contains verified movable targets. ``raw_objects`` also
        contains safety-zone detections and is used after command 15/25.
        ``blackbar_objects`` is this frame's separately mapped auxiliary YOLO
        class7, never part of cargo verification, counting or UART categories.
        ``startzone_objects`` contains separately mapped auxiliary class8 for
        AE escape only; it must not join movable targets or05 inventories.
        """
        verbose_diagnostics = (
            config.VISION_DIAGNOSTICS_ENABLED and config.VISION_DIAGNOSTICS_VERBOSE
        )
        with self._lock:
            if self.mode == MODE_WAIT_TASK_START:
                return None
            if self._startzone_escape_phase is not None:
                if frame_generation is None or frame_generation == self._command_generation:
                    # Observations do not warm locks/counters while turning,
                    # but live camera frames are not a camera-stall fault.
                    self._last_frame_seen_ms = _now_ms()
                return None
            if frame_generation is not None and frame_generation != self._command_generation:
                return None
            if (getattr(config, "UART_STARTZONE_ESCAPE_ENABLED", False)
                    and self.mode in (CMD_SEARCH, CMD_TRACK, MODE_WAIT_04_HANDSHAKE,
                              MODE_WAIT_14_HANDSHAKE, MODE_FINAL_ROI_TRACK)
                    and self._zone_load_recheck is None
                    and self._supply_search_ready_ms is None
                    and startzone_objects is not None):
                area_ratio = self._startzone_area_ratio(startzone_objects)
                if area_ratio <= getattr(config, "UART_STARTZONE_ESCAPE_MIN_AREA_RATIO", .04):
                    self._startzone_escape_armed = True
                elif self._startzone_escape_armed:
                    self._begin_startzone_escape(area_ratio)
                    return None
            self._expire_supply_collection(_now_ms())
            self._expire_zone_load_resume(_now_ms())
            self._clear_disabled_zone_obstacle()
            if self._supply_search_ready_ms is not None and self._supply_search_settling(_now_ms()):
                self._last_frame_seen_ms = _now_ms()
                return None
            generation = self._command_generation
            if frame_generation is not None and frame_generation != generation:
                return None
            if self._direct_capture_pending:
                self._last_frame_seen_ms = _now_ms()
                return None
            if self.mode in MICRO_MODES and (not self._micro_request_sent or self._micro_loss_pending):
                self._last_frame_seen_ms = _now_ms()
                return None
            if (self.mode == MODE_ARRANGE_RIGHTMOST
                    and not self._initial_arrangement_decision_pending
                    and (not self._arrangement_02_request_sent
                         or _now_ms() < self._arrangement_ready_ms)):
                # Do not steer/count acquisition before actual TX02 or while
                # the camera returns from05's near view to the wide position.
                self._last_frame_seen_ms = _now_ms()
                return None
            if self.mode == MODE_SEARCH_ZONE and self._zone_realign_phase == "wait_ack":
                # The MCU owns motion until46 ACK; neither lost-zoneE6 nor
                # fresh coordinates/26 may interrupt the timed maneuver.
                self._last_frame_seen_ms = _now_ms()
                return None
            if self._zone_load_recheck in ("wait45tx", "wait45ack", "wait06tx", "wait24tx", "wide_settle"):
                # Camera frames are arriving, but near view cannot be used
                # for wide-view24/06 steering, entry34 or zone area26.
                self._last_frame_seen_ms = _now_ms()
                return None
            mode = self.mode
            target_label = self.target_label
            post07_started = self._post07_started
            zone_label = self.zone_label
            zone_found_sent = self._zone_found_sent
            zone_close_sent = self._zone_close_sent
            final_roi_sent = self._final_roi_sent
            arrangement_ready_ms = self._arrangement_ready_ms
            post22_e2_recovery = (
                mode == MODE_POST_ARRANGE_CHECK
                and self._last_arrangement_no_object_ms > 0
            )
            initial_arrangement_decision_pending = (
                self._initial_arrangement_decision_pending
            )
            initial_confirmation_started_ms = self._initial_confirmation_started_ms
            final_target_locked = self._final_target_lock is not None
            track_target_locked = self._track_target_lock is not None
            search_target_locked = self._search_target_lock is not None
            arrangement_center_tracking = (mode == MODE_ARRANGE_RIGHTMOST
                                           and not initial_arrangement_decision_pending)
            diagnostic_anchor_attribute = (
                "_track_target_lock" if mode in (CMD_TRACK, MODE_WAIT_04_HANDSHAKE, MODE_WAIT_14_HANDSHAKE) + MICRO_MODES
                or arrangement_center_tracking
                else "_search_target_lock" if mode == CMD_SEARCH
                else "_final_target_lock"
            )
            diagnostic_anchor = getattr(self, diagnostic_anchor_attribute)
            diagnostic_previous = (
                self._last_target_center_x, self._last_target_center_y,
                self._last_target_area, self._last_target_accept_ms,
            )
            following_micro = (mode in MICRO_MODES or arrangement_center_tracking or (
                mode == MODE_ARRANGE_RIGHTMOST and self._initial_arrangement_decision_pending
                and self._micro_rounds > 0))
            micro_association_anchor = (self._track_target_lock if mode in MICRO_MODES or arrangement_center_tracking
                                        else self._initial_confirmation_anchor)
            exclude_loaded_supplies = self._uses_supply_collection() and (
                mode in (CMD_SEARCH, CMD_TRACK, MODE_WAIT_04_HANDSHAKE,
                         MODE_WAIT_14_HANDSHAKE, MODE_FINAL_ROI_TRACK,
                         MODE_POST_ARRANGE_CHECK) + MICRO_MODES
                or mode == MODE_ARRANGE_RIGHTMOST
            )
            loaded_roi_anchors = list(self._supply_loaded_roi_anchors)
            previous_roi_handoff = self._supply_roi_handoff
            supply_roi_observed = self._supply_roi_observed
            zone_load_rechecking = self._zone_load_recheck in ("wait05tx", "count05")
            supply_guard_active = mode in (
                CMD_SEARCH, CMD_TRACK, MODE_WAIT_04_HANDSHAKE,
                MODE_WAIT_14_HANDSHAKE, MODE_FINAL_ROI_TRACK,
                MODE_POST_ARRANGE_CHECK, MODE_ARRANGE_RIGHTMOST,
                MODE_ARRANGE_LEFTMOST, MODE_WAIT_RED_SWITCH,
            ) + MICRO_MODES
            supply_recent_zones = list(self._supply_recent_zones)
            supply_outside_history = list(self._supply_outside_history)
            supply_guard_anchors = [anchor for anchor in (
                diagnostic_anchor, previous_roi_handoff,
                self._initial_confirmation_anchor if initial_arrangement_decision_pending else None,
            ) if anchor is not None]

        now_ms = _now_ms()
        verified_input_objects = objects
        # Near-view load counting uses its own ROI. For tracking/arrangement,
        # material inside a safety zone or its100px side margins is not a target.
        # Zone06 obstacle detection keeps the unexpanded safety-zone exclusion.
        safety_margin_x = (
            0 if mode == MODE_SEARCH_ZONE else config.UART_SAFE_ZONE_TARGET_MARGIN_X
        )
        if mode != MODE_NEAR_VIEW_CHECK:
            zone_detections = raw_objects if raw_objects is not None else objects
            objects = self._exclude_objects_in_safety_zones(
                objects, labels, zone_detections, horizontal_padding=safety_margin_x
            )
        # Current YOLO boxes are used for locked association and for the
        # post07 supply acquisition below. Both paths exclude safety zones.
        raw_tracking_objects = self._exclude_objects_in_safety_zones(
            raw_objects if raw_objects is not None else verified_input_objects,
            labels, raw_objects if raw_objects is not None else verified_input_objects,
            horizontal_padding=safety_margin_x,
        )
        if supply_guard_active:
            # Prepare locally; commit only after the frame-generation check.
            # A short zone miss cannot expose delivered cargo as a new target.
            supply_recent_zones = self._remember_supply_zones(
                raw_objects if raw_objects is not None else verified_input_objects,
                labels, supply_recent_zones, now_ms,
            )
            objects = self._exclude_recent_supply_zones(objects, supply_recent_zones)
            raw_tracking_objects = self._exclude_recent_supply_zones(
                raw_tracking_objects, supply_recent_zones)
        if mode in (CMD_SEARCH, CMD_TRACK, MODE_WAIT_04_HANDSHAKE, MODE_WAIT_14_HANDSHAKE):
            #03/04 and their coordinate-producing handshake waits never steer
            # toward cargo whose CENTER is already inside the tracking ROI.
            # Apply to acquisition AND current raw locked association/recovery.
            objects = [obj for obj in objects
                       if not self._target_center_in_tracking_roi(obj)]
            raw_tracking_objects = [obj for obj in raw_tracking_objects
                                    if not self._target_center_in_tracking_roi(obj)]
        reference_x, reference_y = self._reference_point(mode, target_label)
        casualty_final_track = (mode == MODE_FINAL_ROI_TRACK
                                and target_label == config.UART_ZONE_CASUALTY_LABEL)
        supply_roi_handoff = None
        continuing_supply_target = None
        if exclude_loaded_supplies:
            # Apply to EVERY acquisition and raw lock association, not only
            # the first supplemental search. ROI cargo cannot steal a lock.
            unmasked_raw = raw_tracking_objects
            objects = self._unloaded_supply_candidates(objects)
            raw_tracking_objects = self._unloaded_supply_candidates(raw_tracking_objects)
            supply_roi_handoff, continuing_supply_target, loaded_roi_anchors = self._observe_supply_roi_entry(
                unmasked_raw, raw_tracking_objects, loaded_roi_anchors,
                (previous_roi_handoff or (micro_association_anchor if following_micro else diagnostic_anchor))
                if (mode == MODE_FINAL_ROI_TRACK or following_micro) and supply_roi_observed else None,
                previous_roi_handoff is not None and (mode == MODE_FINAL_ROI_TRACK or following_micro),
                target_label,
            )
            if (following_micro and supply_roi_handoff is not None
                    and all(obj is not supply_roi_handoff for obj in raw_tracking_objects)):
                #32/42 is pushing the already locked NEW target, not grabbing
                #cargo. Follow its inward entry without34; remembered old
                #cargo remains excluded by the one-to-one association above.
                raw_tracking_objects.append(supply_roi_handoff)

        post07_supply_acquisition = post07_started and self._is_supply_acquisition(mode, target_label)
        if post07_supply_acquisition:
            # The per-object confirmation below is sufficient for new supply
            # targets. Keep verified non-supply boxes for the front-block test.
            objects = [obj for obj in objects if int(obj.class_id) not in (5, 6)]
            objects.extend(obj for obj in raw_tracking_objects if int(obj.class_id) in (5, 6)
                           and clipped_yolo_rect(obj, self.frame_width, self.frame_height) is not None)

        supply_confirmed = set()
        if supply_guard_active:
            supply_confirmed, supply_outside_history = self._confirm_supply_outside_targets(
                raw_tracking_objects, supply_outside_history, now_ms,
                post07_supply_acquisition=post07_supply_acquisition)

        arrangement_object_count = None
        arrangement_neighbor_missing = False
        near_view_count = None
        near_view_ids = []
        near_view_needs_arrangement = False
        near_view_blue_without_target = False
        near_view_recheck_without_target = False
        near_view_log = None
        zone_area_ratio = 0.0
        if mode == MODE_WAIT_SUPPLY_SEARCH:
            return None
        if mode == MODE_WAIT_RED_SWITCH:
            with self._lock:
                if generation != self._command_generation:
                    return None
                self._supply_recent_zones = supply_recent_zones
                self._supply_outside_history = supply_outside_history
            return self._associate_pending_red(raw_tracking_objects, now_ms, generation)
        continuity_removed = []
        self._selection_diagnostic_policy = "arrange_open_loop"
        arrangement_waiting = (
            mode in (MODE_ARRANGE_RIGHTMOST, MODE_ARRANGE_LEFTMOST)
            and now_ms < arrangement_ready_ms
        )

        if mode == MODE_NEAR_VIEW_CHECK:
            self._selection_diagnostic_policy = "near05"
            near_view_boxes = self._near_view_qualified_objects(objects)
            near_view_ids = self._movable_target_ids(near_view_boxes, labels)
            near_view_count = len(near_view_ids)
            near_view_supply_task = target_label == "triangualrblack" or (
                self._post07_started
                and target_label in ("sqareredgreen", TARGET_BLACK_OR_GREEN)
            )
            accepted_labels = (
                ("sqareredgreen", "triangualrblack") if near_view_supply_task
                else self._eligible_target_labels(target_label)
            )
            near_view_blue_without_target = (
                3 in near_view_ids
                and not any(labels[class_id] in accepted_labels for class_id in near_view_ids)
            )
            near_view_recheck_without_target = (
                zone_load_rechecking
                and not any(labels[class_id] in accepted_labels for class_id in near_view_ids)
            )
            if near_view_blue_without_target or near_view_recheck_without_target:
                # Normal05 rejects blue without a task target. Evacuation
                # rechecks return24 for ANY load lacking acceptable cargo.
                near_view_needs_arrangement = False
            elif near_view_supply_task:
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
            selected = self._select_largest_zone(
                candidates, labels, zone_label, reference_x
            )
            if selected is not None:
                zone_area_ratio = self._visible_box_area_ratio(selected)
                if zone_area_ratio <= getattr(config, "UART_ZONE_MIN_AREA_RATIO", 0.02):
                    # Keep raw detections for DEBUG, but this is not yet a
                    # usable zone: no steering coordinates or first16.
                    selected = None
        elif mode == MODE_FINAL_ROI_TRACK:
            if supply_roi_handoff is not None:
                # A separately associated new entry is a34 handoff, NOT a
                # tracking target. Never build coordinates for this ROI box.
                selected = None
                self._selection_diagnostic_policy = "collect51_roi_handoff34"
            elif continuing_supply_target is not None:
                selected = continuing_supply_target
                self._selection_diagnostic_policy = "collect51_continue_track"
            else:
                selected = self._select_locked_target(
                    objects, labels, target_label, now_ms,
                    "_final_target_lock", "FINAL-24", raw_tracking_objects,
                    generation=generation,
                )
            if (exclude_loaded_supplies and selected is not None
                    and self._final_capture_reached(selected, reference_y)):
                # Only eligible new cargo can cross the line and emit34.
                supply_roi_handoff = selected
                selected = None
                self._selection_diagnostic_policy = "collect51_roi_handoff34"
        elif mode in MICRO_MODES:
            selected = self._select_micro_target(raw_tracking_objects, now_ms, generation)
        elif mode in (CMD_TRACK, MODE_WAIT_04_HANDSHAKE, MODE_WAIT_14_HANDSHAKE):
            selected = self._select_locked_target(
                objects, labels, target_label, now_ms,
                "_track_target_lock", "TRACK-04",
                raw_tracking_objects,
                generation=generation,
            )
        elif mode == CMD_SEARCH and search_target_locked:
            selected = self._select_locked_target(
                objects, labels, target_label, now_ms,
                "_search_target_lock", "SEARCH-03", raw_tracking_objects,
                generation=generation,
            )
        elif mode == CMD_SEARCH and self._supply_collect_started_ms is not None:
            selected = self._select_supply_collection_target(
                objects, raw_tracking_objects, labels, now_ms)
        elif (
            mode in (
                MODE_POST_ARRANGE_CHECK,
            )
            or (
                target_label == TARGET_BLACK_OR_GREEN
                and mode == CMD_SEARCH
            )
        ):
            # Share ROI ranking with the post-arrangement check.
            self._selection_diagnostic_policy = "merged"
            if post22_e2_recovery:
                selected = self._select_post22_yolo_target(raw_tracking_objects, labels, target_label)
            elif self._uses_relaxed_supply_tracking(target_label):
                selected = self._select_relaxed_supply_target(raw_tracking_objects, labels)
            else:
                selected = self._select_final_roi_target(
                    objects, labels, target_label, now_ms,
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
                    and not self._post07_started
                ),
            )
        elif (
            mode == MODE_ARRANGE_RIGHTMOST
            and initial_arrangement_decision_pending
        ):
            # First-green AND post07 supplies14/44 use the SAME expanded box
            # as the yellow preview. Red/blue retain their merged decision.
            # Also keep blue's legacy merged-neighbor decision free of ROI
            # cargo. Expanded decisions retain the approached raw identity;
            # their shared directional helper masks neighbors separately.
            objects = [obj for obj in objects if not self._target_center_in_tracking_roi(obj)]
            arrangement_object_count = self._count_movable_targets(objects, labels)
            if self._uses_expanded_overlap_decision(target_label):
                self._selection_diagnostic_policy = (
                    "red_overlap14" if target_label == "sqarered" else
                    "supply_overlap14" if post07_started else "first_green_overlap14")
                selected = self._select_initial_overlap_candidate(raw_tracking_objects, labels, target_label)
            else:
                self._selection_diagnostic_policy = "merged"
                selected = self._select_final_roi_target(
                    objects, labels, target_label, now_ms
                )
        elif arrangement_waiting:
            self._selection_diagnostic_policy = "arrange_camera_settle"
            # Event 02 is moving MG90 back to wide view. Ignore transitional
            # frames until the configured mechanical settling time expires.
            selected = None
        elif mode == MODE_ARRANGE_RIGHTMOST:
            #02 now follows the approached center target, never a side box.
            # After05 there is no cross-camera position lock: acquire a fresh
            # commanded-color target, then keep raw nearest-previous identity.
            arrangement_object_count = self._count_movable_targets(objects, labels)
            if track_target_locked:
                selected = self._select_micro_target(raw_tracking_objects, now_ms, generation)
            else:
                selected = self._select_frame_center_target(objects, labels, target_label)
            self._selection_diagnostic_policy = "arrange02_center_lock" if track_target_locked else "arrange02_center_acquire"
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

        supply_guard_pending = False
        post22_yolo_objects = raw_tracking_objects
        if supply_guard_active:
            def already_following(obj):
                return any(
                    int(obj.class_id) == anchor[0]
                    or self._uses_relaxed_supply_tracking(target_label)
                    and int(obj.class_id) in (5, 6) and anchor[0] in (5, 6)
                    for anchor in supply_guard_anchors
                )

            def acquisition_allowed(obj):
                return (already_following(obj)
                        or self._final_target_identity(obj) in supply_confirmed)

            confirmed_raw = [obj for obj in raw_tracking_objects
                             if int(obj.class_id) in (3, 4, 5, 6) and acquisition_allowed(obj)]
            post22_yolo_objects = confirmed_raw
            if (selected is not None and int(selected.class_id) in (3, 4, 5, 6)
                    and not acquisition_allowed(selected)):
                # Keep the original task selector and all its neighbors.
                # Unconfirmed boxes cannot acquire a lock or emit completion.
                selected = None
                supply_guard_pending = True
            if supply_roi_handoff is not None and not acquisition_allowed(supply_roi_handoff):
                supply_roi_handoff = None
                supply_guard_pending = True
            eligible_labels = self._eligible_target_labels(target_label)
            if self._uses_relaxed_supply_tracking(target_label):
                eligible_labels = ("sqareredgreen", "triangualrblack")
            if (selected is None and supply_roi_handoff is None
                    and any(0 < int(obj.class_id) < len(labels)
                            and labels[int(obj.class_id)] in eligible_labels
                            and not acquisition_allowed(obj) for obj in raw_tracking_objects)):
                supply_guard_pending = True
            if supply_guard_pending:
                self._selection_diagnostic_policy = "new_target_wait_outside"

        # Commit evidence even if this frame initiates the black-to-red pause.
        # An in-flight command must never warm counters in the next generation.
        with self._lock:
            if generation != self._command_generation:
                return None
            if supply_guard_active:
                self._supply_recent_zones = supply_recent_zones
                self._supply_outside_history = supply_outside_history

        # Switching tasks is a command handshake, never a coordinate update
        # in the old 03/04/24 reference system.
        if supply_roi_handoff is None and self._request_red_after_black_loss(
            mode, raw_tracking_objects, labels, selected, now_ms, generation, supply_confirmed,
        ):
            return None

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
        ) and not (
            (mode == MODE_POST_ARRANGE_CHECK
             and (post22_e2_recovery or self._uses_relaxed_supply_tracking(target_label)))
            or
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
            mode == MODE_ARRANGE_LEFTMOST
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
        ready_at_center = False
        capture_target = supply_roi_handoff
        if (self._claw_geometry and capture_target is not None
                and not self._claw_capture_gate(capture_target, reference_x)):
            # Keep steering toward the eligible new supply until its own
            # geometry qualifies; do not switch back to existing cargo.
            selected = capture_target
            capture_target = None
        track_center_reached = False
        zone_is_close = False
        zone_blocker = None
        zone_obstacle_area_allowed = False
        if selected is not None:
            class_id = int(selected.class_id)
            # Model class 0 is deprecated and must never be transmitted.
            if class_id > 0:
                if mode == MODE_SEARCH_ZONE:
                    zone_obstacle_area_allowed = (
                        self._zone_obstacle_enabled()
                        and zone_area_ratio <= config.UART_ZONE_OBSTACLE_MAX_AREA_RATIO
                    )
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
                        target_x_ratio = (
                            config.UART_ZONE_CASUALTY_NEAR_X_RATIO
                            if target_label == config.UART_ZONE_CASUALTY_LABEL
                            else config.UART_ZONE_SUPPLY_NEAR_X_RATIO
                        )
                    if zone_found_sent and not zone_close_sent and zone_obstacle_area_allowed:
                        zone_blocker = self._find_roi_edge_blocker(objects, labels)
                    # Event 26 repeats while the current visible area qualifies;
                    # continue publishing fresh zone coordinates between events.
                    # Red casualties aim 2/5 inward from the right edge;
                    # supplies start 3/10 from the left. After16 and >25%
                    # area, BOTH supplies and casualties use horizontal center.
                    error_x, error_y = self._zone_supply_point_error(
                        selected,
                        reference_x,
                        reference_y,
                        target_x_ratio,
                    )
                    packet = build_coordinate_packet(
                        class_id, error_x, error_y
                    )
                elif (mode == MODE_ARRANGE_RIGHTMOST and initial_arrangement_decision_pending) or mode in (
                    MODE_ARRANGE_LEFTMOST,
                    MODE_POST_ARRANGE_CHECK,
                ):
                    #14's decision stage is not an active02 push.12 remains
                    #open-loop;only actually requested02 sends center errors.
                    packet = None
                elif (mode == MODE_FINAL_ROI_TRACK and not casualty_final_track
                      and self._final_capture_reached(selected, reference_y)
                      and self._claw_capture_gate(selected, reference_x)):
                    if capture_target is None:
                        capture_target = selected
                else:
                    error_x, error_y = target_center_error(
                        selected,
                        self.frame_width,
                        self.frame_height,
                        reference_x=reference_x,
                        reference_y=reference_y,
                    )
                    final_target_below_reference = (
                        mode == MODE_WAIT_14_HANDSHAKE
                        and self.tracking_roi is not None
                        and float(selected.y) + float(selected.h) * 0.5 > reference_y
                    )
                    if final_target_below_reference:
                        error_y = 0
                    object_mode = mode in (CMD_SEARCH, CMD_TRACK, MODE_WAIT_04_HANDSHAKE,
                                           MODE_WAIT_14_HANDSHAKE, MODE_FINAL_ROI_TRACK)
                    packet = (
                        self._build_claw_coordinate(selected, error_x, error_y)
                        if object_mode else build_coordinate_packet(class_id, error_x, error_y)
                    )
                    ready_at_center = (self._claw_geometry
                                       and self._claw_capture_gate(selected, reference_x))
                    track_center_reached = (
                        mode == CMD_TRACK
                        and (ready_at_center or (
                            abs(error_x) <= config.UART_FINAL_ROI_CENTER_TOLERANCE_X
                            and float(selected.y) + float(selected.h) * 0.5
                            >= reference_y - config.UART_FINAL_ROI_CENTER_TOLERANCE_Y
                        ))
                    )
                target_in_roi = (
                    mode == CMD_SEARCH
                    and (ready_at_center or (
                        abs(error_x) <= config.UART_FRAME_CENTER_TOLERANCE_X
                        and abs(error_y) <= config.UART_FRAME_CENTER_TOLERANCE_Y
                    ))
                )

        with self._lock:
            # A command may have arrived during filtering/packet preparation.
            # Do not overwrite its pause/new reference with this older frame.
            if generation != self._command_generation:
                return None
            if casualty_final_track and self.mode == MODE_FINAL_ROI_TRACK:
                candidate = self._casualty_capture_candidate(
                    selected, raw_tracking_objects if post07_started else (),
                    reference_x, now_ms, generation)
                if (self._confirm_casualty_capture(candidate, now_ms, generation)
                        and (candidate is selected
                             or self._final_target_identity(candidate) in supply_confirmed)):
                    capture_target = candidate
                    packet = None
            else:
                self._casualty_capture_streak = None
            if supply_guard_active:
                self._supply_recent_zones = supply_recent_zones
                self._supply_outside_history = supply_outside_history
            self._new_target_confirmation_pending = supply_guard_pending
            if exclude_loaded_supplies:
                self._supply_loaded_roi_anchors = loaded_roi_anchors
                self._supply_roi_observed = True
                self._supply_roi_handoff = (
                    self._final_target_identity(supply_roi_handoff)
                    if supply_roi_handoff is not None else None
                )
                self._supply_roi_handoff_seen_ms = now_ms if supply_roi_handoff is not None else 0
            if arrangement_object_count is not None:
                self._arrangement_object_count = arrangement_object_count
            if (
                mode == MODE_ARRANGE_RIGHTMOST
                and self.mode == MODE_ARRANGE_RIGHTMOST
            ):
                self._arrangement_side_target_available = (
                    selected is not None and not arrangement_waiting
                )
                if arrangement_center_tracking and selected is not None:
                    identity = self._final_target_identity(selected)
                    self._track_target_lock = identity
                    self._last_target_center_x, self._last_target_center_y = identity[1:3]
                    self._last_target_area = identity[3]
                    self._last_target_accept_ms = now_ms
                    self._last_tracked_class_id = identity[0]
                    self._tracking_missing_since_ms = None
                    self._tracking_missing_class_id = None
            self._latest_packet = packet
            self._latest_submit_ms = now_ms
            self._last_frame_seen_ms = now_ms
            # A sequence counter also distinguishes frames submitted within
            # the same millisecond or containing identical coordinates.
            self._latest_submit_sequence += 1
            if mode == MODE_SEARCH_ZONE:
                self._zone_area_ratio = zone_area_ratio
                no_zone_event = build_event_packet(EVENT_ZONE_NO_TARGET)
                if selected is None:
                    if self._zone_no_target_phase is None:
                        self._zone_no_target_phase = "pending"
                        self._pending_events.append(no_zone_event)
                else:
                    # Re-arm only on a fresh valid zone, not an MCU ACK or a
                    # timer. A preparedE6 is also fenced at the actual write.
                    self._zone_no_target_phase = None
                    self._pending_events = [event for event in self._pending_events
                                            if event != no_zone_event]
            if mode == self.mode and mode in (
                CMD_SEARCH, CMD_TRACK, MODE_WAIT_04_HANDSHAKE,
                MODE_WAIT_14_HANDSHAKE, MODE_FINAL_ROI_TRACK,
            ) + MICRO_MODES:
                observed_target = selected if selected is not None else supply_roi_handoff
                if observed_target is None:
                    if not supply_guard_pending:
                        self._record_tracking_missing(now_ms)
                else:
                    self._last_tracked_class_id = int(observed_target.class_id)
                    self._tracking_missing_since_ms = None
                    self._tracking_missing_class_id = None
            if (
                mode in (CMD_TRACK, MODE_WAIT_04_HANDSHAKE, MODE_WAIT_14_HANDSHAKE) + MICRO_MODES
                and self.mode == mode
            ):
                if selected is None:
                    if not supply_guard_pending:
                        self._track_no_target_frames += 1
                    if verbose_diagnostics and self._track_no_target_frames in (1, config.UART_TRACK_LOST_CONFIRM_FRAMES):
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
                    if verbose_diagnostics and self._track_no_target_frames:
                        print("[TRACK-04] recovered=id{} after_missing={} fresh_coordinates".format(
                            identity[0], self._track_no_target_frames
                        ))
                    self._track_no_target_frames = 0
                    if verbose_diagnostics and self._track_target_lock is None:
                        print("[TRACK-04] lock=id{}@({:.0f},{:.0f})".format(
                            identity[0], identity[1], identity[2]
                        ))
                    self._track_target_lock = identity
                    # A fresh coordinate interrupts the MCU's red/other search.
                    # Its next confirmed loss therefore needs its own E4.
                    # First green recovery is deliberately not interruptible:
                    # retain its one-shot E4/14 gating until MCU sends 44.
                    if mode == CMD_TRACK and not self._track_green_recovery_waiting:
                        if verbose_diagnostics and self._track_no_target_sent:
                            print("[TRACK-04] fresh_target=rearm_E4_for_next_loss")
                        self._track_no_target_sent = False
                    # RX14/RX44 use this position as the middle-object anchor.
                    self._last_target_center_x = identity[1]
                    self._last_target_center_y = identity[2]
                    self._last_target_area = identity[3]
                    self._last_target_accept_ms = now_ms
            if mode == MODE_FINAL_ROI_TRACK and self.mode == MODE_FINAL_ROI_TRACK:
                if selected is None and supply_roi_handoff is None:
                    if not supply_guard_pending:
                        self._final_no_target_frames += 1
                else:
                    self._final_no_target_frames = 0
                    if verbose_diagnostics and selected is not None and self._final_target_lock is None:
                        print("[FINAL-24] lock=id{}@({:.0f},{:.0f})".format(
                            int(selected.class_id),
                            float(selected.x) + float(selected.w) * 0.5,
                            float(selected.y) + float(selected.h) * 0.5,
                        ))
                    if selected is not None:
                        self._final_target_lock = self._final_target_identity(selected)
            if (
                mode == MODE_POST_ARRANGE_CHECK
                and self.mode == MODE_POST_ARRANGE_CHECK
                and selected is not None
            ):
                if post22_e2_recovery:
                    self._start_post22_center_tracking(selected, now_ms)
                    packet = self._latest_packet
                    print("[POST-22] after_E2=raw_yolo id{} action=TX04 wait_RX04".format(
                        int(selected.class_id)
                    ))
                else:
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
            elif (
                mode == MODE_POST_ARRANGE_CHECK
                and self.mode == MODE_POST_ARRANGE_CHECK
                and not supply_guard_pending
            ):
                # A visible target that fails merged screening should be
                # approached with 04 immediately, before any E2 is needed.
                approach_target = self._select_post22_yolo_target(
                    post22_yolo_objects, labels, target_label
                )
                self._track_no_target_frames = (
                    0 if approach_target is not None else self._track_no_target_frames + 1
                )
                if approach_target is not None:
                    selected = approach_target
                    self._start_post22_center_tracking(selected, now_ms)
                    packet = self._latest_packet
                    print("[POST-22] raw_target=id{} merged=reject action=TX04 wait_RX04".format(
                        int(selected.class_id)
                    ))
                elif (self._uses_first_green_center_selection(target_label)
                      and self._track_no_target_frames >= config.UART_TRACK_LOST_CONFIRM_FRAMES):
                    # Use the existing first-green E4 recovery/44 handshake.
                    # Do not repeat E4 or emit 14 while MCU completes recovery.
                    self._pending_events.clear()
                    self._pending_events.append(build_event_packet(EVENT_TRACK_NO_TARGET))
                    self.mode = CMD_TRACK
                    self._track_target_lock = None
                    self._track_no_target_sent = True
                    self._track_green_recovery_waiting = True
                    self._latest_packet = None
                    self._latest_submit_ms = 0
                    print("[POST-22] first_green=missing frames={} action=TXE4 wait_RX44".format(
                        self._track_no_target_frames
                    ))
            if mode == CMD_SEARCH and self.mode == CMD_SEARCH:
                if selected is None:
                    if not supply_guard_pending:
                        self._search_no_target_frames += 1
                else:
                    if verbose_diagnostics and self._search_target_lock is None:
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
                    self._search_e3_active = False
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
                    self._near_view_started_ms is not None
                    and now_ms - self._near_view_started_ms
                    >= config.UART_NEAR_VIEW_SETTLE_MS
                    and self._zone_load_recheck != "wait05tx"
                ):
                    # Confirm the resulting action (02/06/24). Black tasks can
                    # accept one to three black/green objects without arranging.
                    if near_view_blue_without_target or (
                            near_view_recheck_without_target and near_view_count > 0):
                        # Separate from delivery, collection and arrangement
                        # so their earlier frames cannot confirm this action.
                        count_bucket = 4
                    elif near_view_needs_arrangement:
                        count_bucket = 2
                    elif (self._uses_supply_collection() and not zone_load_rechecking
                          and near_view_count == 1):
                        count_bucket = 3  # Continue collecting, distinct from delivery.
                    else:
                        count_bucket = 1 if near_view_count > 0 else 0
                    first_green_task = (target_label == "sqareredgreen" and not self._post07_started
                                        and not zone_load_rechecking)
                    green_visible = 5 in near_view_ids
                    if first_green_task:
                        if green_visible:
                            self._near_view_green_missing_cycles = 0
                        # Foreign-only and green+foreign both normally emit02.
                        # Do not use green-present frames to confirm absence.
                        count_bucket = (count_bucket, green_visible)
                    if count_bucket == self._near_view_stable_bucket:
                        self._near_view_stable_frames += 1
                    else:
                        self._near_view_stable_bucket = count_bucket
                        self._near_view_stable_frames = 1

                    count_is_stable = (
                        self._near_view_stable_frames
                        >= config.UART_NEAR_VIEW_STABLE_FRAMES
                    )
                    empty_cycle_decision = (
                        count_is_stable and near_view_count == 0
                        and not self._near_view_result_sent
                        and not zone_load_rechecking
                    )
                    if empty_cycle_decision:
                        self._near_view_empty_cycles += 1
                    green_missing_decision = (
                        count_is_stable and first_green_task and not green_visible
                        and not self._near_view_result_sent and not self._near_view_adjust_sent
                    )
                    if green_missing_decision:
                        self._near_view_green_missing_cycles += 1
                    force_first_green_empty_recovery = (
                        green_missing_decision and self._near_view_green_missing_cycles >= 2
                    )
                    if green_missing_decision:
                        print("[NEAR-05-NO-GREEN] cycles={} ids={} action={}".format(
                            self._near_view_green_missing_cycles, near_view_ids,
                            "TXE5_WAIT44" if force_first_green_empty_recovery
                            else "TX02" if near_view_needs_arrangement else "TX24",
                        ))
                    if empty_cycle_decision:
                        print("[NEAR-05-EMPTY] cycles={} first_green={} action={}".format(
                            self._near_view_empty_cycles,
                            target_label == "sqareredgreen" and not self._post07_started,
                            "TXE5_WAIT44" if force_first_green_empty_recovery else "TX24",
                        ))
                    first_green_decision = (
                        count_is_stable
                        and target_label == "sqareredgreen"
                        and not self._first_green_near_view_decided
                        and not self._post07_started
                        and not zone_load_rechecking
                    )
                    if first_green_decision:
                        self._first_green_near_view_decided = True
                    if (count_is_stable and self._uses_supply_collection() and not zone_load_rechecking
                            and near_view_count == 1 and not near_view_needs_arrangement
                            and not near_view_blue_without_target
                            and not self._near_view_result_sent):
                        self._request_supply_collection(now_ms)
                        if self._supply_collect_finished:
                            near_view_log = "collection_finished=1 reason={} action=TX05 final_inventory".format(
                                self._supply_collect_finish_reason)
                        else:
                            near_view_log = "collect_attempt={} action={}".format(
                                self._supply_collect_attempts,
                                "TX03 wait_RX03" if self.mode == MODE_WAIT_SUPPLY_SEARCH else "TX05 final_inventory",
                            )
                    elif force_first_green_empty_recovery:
                        # E5 reuses the uninterruptible first-green E4 pushes.
                        # The MCU completes those actions with44, not12/22.
                        self._pending_events.clear()
                        self._pending_events.append(build_event_packet(EVENT_FIRST_GREEN_EMPTY_RECOVERY))
                        self._near_view_result_sent = True
                        self._near_view_empty_cycles = 0
                        self._near_view_green_missing_cycles = 0
                        self.mode = CMD_TRACK
                        self._track_target_lock = None
                        self._final_target_lock = None
                        self._search_target_lock = None
                        self._track_no_target_frames = 0
                        self._track_no_target_sent = True
                        self._track_green_recovery_waiting = True
                        self._track_center_sent = False
                        self._first_green_both_sides_required = False
                        self._initial_arrangement_decision_pending = False
                        self._arrangement_02_ack_pending = False
                        self._arrangement_02_without_side = False
                        self._arrangement_side_target_available = False
                        self._final_roi_sent = False
                        self._final_search_waiting_ack = False
                        self._last_target_center_x = None
                        self._last_target_center_y = None
                        self._last_target_area = None
                        self._last_target_accept_ms = 0
                        near_view_log = "TXE5 two_no_green05_recovery wait_RX44"
                    elif (
                        count_is_stable
                        and near_view_needs_arrangement
                        and not self._near_view_adjust_sent
                    ):
                        if zone_load_rechecking:
                            self._zone_load_recheck = None
                            self._zone_load_recheck_source = None
                            self._pending_events.clear()
                            self._command_generation += 1
                        self._pending_events.append(
                            build_event_packet(EVENT_ROI_READY)
                        )
                        self._near_view_adjust_sent = True
                        # Arrangement starts a new capture round, but must
                        # preserve consecutive no-green05 history for E5.
                        if not zone_load_rechecking:
                            self._near_view_empty_cycles = 0
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
                        self._arrangement_02_request_sent = False
                        self._track_target_lock = None
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
                        and not near_view_blue_without_target
                        and not near_view_recheck_without_target
                        and not self._near_view_result_sent
                    ):
                        if zone_load_rechecking:
                            near_view_log = self._finish_zone_load_recheck(near_view_ids, near_view_boxes)
                        elif self._uses_supply_collection():
                            self._finish_supply_collection("enough_supplies", now_ms)
                            near_view_log = "TX05 final_inventory enough_supplies"
                        else:
                            near_view_log = self._accept_near_view_load(near_view_ids)
                    elif (
                        count_is_stable
                        and (near_view_count == 0 or near_view_blue_without_target
                             or near_view_recheck_without_target)
                        and not self._near_view_result_sent
                    ):
                        if zone_load_rechecking:
                            self._zone_load_recheck = "wait24tx"
                            self._pending_events.clear()
                            self._command_generation += 1
                        elif (self._uses_supply_collection()
                              and getattr(config, "UART_SUPPLY_SEARCH_SETTLE_MS", 250) > 0):
                            self._zone_load_recheck = "wait24tx"
                            self._zone_load_recheck_source = "supply_empty"
                            self._pending_events.clear()
                            self._command_generation += 1
                        self._reset_supply_collection()
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
                        near_view_log = (
                            "TX24 no_eligible_load_return_final" if near_view_recheck_without_target
                            and near_view_count > 0 else "TX24 blue_without_eligible_target_return_final"
                            if near_view_blue_without_target else "TX24 no_object_return_final"
                        )
            if (
                mode == MODE_ARRANGE_RIGHTMOST
                and self.mode == MODE_ARRANGE_RIGHTMOST
                and initial_arrangement_decision_pending
                and self._initial_arrangement_decision_pending
                and self._initial_confirmation_started_ms == initial_confirmation_started_ms
                and not supply_guard_pending
            ):
                if selected is None and self._micro_rounds:
                    # A completed push/turn can put the target out of view.
                    # Missing is E4 recovery, not evidence of two-axis crowding.
                    self._track_no_target_frames += 1
                    anchor = self._initial_confirmation_anchor
                    self._record_tracking_missing(now_ms, anchor[0] if anchor is not None else None)
                    if self._tracking_loss_confirmed(CMD_TRACK, self._track_no_target_frames, now_ms):
                        self._finish_initial_arrangement_decision(
                            EVENT_TRACK_NO_TARGET, None, "micro_recheck_missing", now_ms)
                    elif initial_confirmation_started_ms is not None:
                        self._observe_initial_green_decision(None, raw_tracking_objects, labels, now_ms)
                elif initial_confirmation_started_ms is not None:
                    self._observe_initial_green_decision(selected, raw_tracking_objects, labels, now_ms)
                else:
                    # Later tasks retain their existing single-frame decision.
                    self._finish_initial_arrangement_decision(
                        (self._directional_arrangement_event(selected, raw_tracking_objects)
                         if self._uses_expanded_overlap_decision(target_label)
                         else EVENT_SKIP_TO_FINAL_TRACK if selected is not None else EVENT_ROI_READY),
                        selected, "merged", now_ms,
                    )
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
                if verbose_diagnostics:
                    print("[SEARCH-03] center_reached=id{} retain_lock_until_RX04".format(
                        int(selected.class_id)
                    ))
                # Keep publishing the approach errors, but do not emit 14
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
                # Discard the last approach packet after event 14.
                # The next frame will be rebuilt against the ROI center.
                self._latest_packet = None
                self._latest_submit_ms = 0
            if mode == MODE_FINAL_ROI_TRACK:
                self._final_capture_tall_casualty = False
                if (capture_target is not None and target_label == config.UART_ZONE_CASUALTY_LABEL
                        and int(capture_target.class_id) == 4):
                    width, height = float(capture_target.w), float(capture_target.h)
                    self._final_capture_tall_casualty = width > 0 and height > 1.5 * width
            if mode == MODE_FINAL_ROI_TRACK and capture_target is not None:
                if not self._final_roi_sent:
                    if supply_roi_handoff is not None:
                        print("[COLLECT-51] new_target_ready_for_capture=id{} action=TX34 coordinates=paused".format(
                            int(supply_roi_handoff.class_id)))
                    elif casualty_final_track:
                        _, x, y, _ = self._final_target_identity(capture_target)
                        print("[FINAL-24] red_deep_roi=id4 x={:.1f} y={:.1f} confirm={} action=TX34".format(
                            x, y, self._casualty_capture_streak[4]))
                    self._pending_events.append(self._capture_event_packet)
                    self._final_roi_sent = True
                # While the target center still qualifies for capture, send only event
                # 34 (immediately, then every configured repeat interval).
                self._latest_packet = None
                self._latest_submit_ms = 0
            elif mode == MODE_FINAL_ROI_TRACK and self._final_roi_sent:
                # The target left the capture region or disappeared. Stop
                # repeating 34; coordinates/EE can resume, and re-entry will
                # trigger an immediate new 34.
                self._final_roi_sent = False
                self._last_final_roi_event_ms = 0
                pending_34 = self._capture_event_packet
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
            if mode == MODE_SEARCH_ZONE:
                self._zone_obstacle_area_allowed = zone_obstacle_area_allowed
                if selected is not None and not zone_obstacle_area_allowed:
                    # Near a zone, do not carry old far-view blocker hits into
                    # a new bypass. Cancel only requests not actually written;
                    # an MCU maneuver already started still needs its36 ACK.
                    self._zone_obstacle_frames = 0
                    self._zone_obstacle_clear_frames = 0
                    if self._zone_obstacle_waiting and not self._zone_obstacle_request_sent:
                        self._zone_obstacle_waiting = False
                        pending_36 = build_event_packet(EVENT_ZONE_OBSTACLE)
                        self._pending_events = [
                            event for event in self._pending_events if event != pending_36
                        ]
                        self._last_zone_obstacle_event_ms = 0
            if mode == MODE_SEARCH_ZONE and not self._zone_close_sent:
                if self._zone_found_sent and selected is not None and zone_obstacle_area_allowed:
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
                    zone_obstacle_area_allowed
                    and zone_blocker is not None
                    and self._zone_obstacle_armed
                    and not self._zone_obstacle_waiting
                    and self._zone_obstacle_frames
                    >= config.UART_ZONE_OBSTACLE_CONFIRM_FRAMES
                ):
                    self._zone_obstacle_waiting = True
                    self._zone_obstacle_request_sent = False
                    self._pending_events.append(
                        build_event_packet(EVENT_ZONE_OBSTACLE)
                    )
                    print("[ZONE-16] blocker=id{} action=TX36_WAIT_ACK".format(
                        int(zone_blocker.class_id)
                    ))
                if self._zone_obstacle_waiting:
                    self._latest_packet = None
                    self._latest_submit_ms = 0
            zone_close_ready = False
            if mode == MODE_SEARCH_ZONE:
                zone_close_ready = self._update_zone_realign(
                    selected, blackbar_objects or (), zone_is_close, now_ms)
            realigning_zone = self._zone_realign_phase in ("pending", "wait_ack")
            self._zone_close_active = (
                mode == MODE_SEARCH_ZONE and zone_close_ready
                and not realigning_zone
                and not self._zone_obstacle_waiting
                and (zone_blocker is None or not self._zone_obstacle_armed)
            )
            if self._zone_close_active:
                if not self._zone_close_sent:
                    self._pending_events.append(
                        build_event_packet(EVENT_ZONE_CLOSE)
                    )
                    self._zone_close_sent = True
                # Keep the current frame's coordinates available after 26;
                # event priority still sends queued commands before coordinates.
            else:
                # A newer miss or smaller box cancels even a queued first 26.
                pending_26 = build_event_packet(EVENT_ZONE_CLOSE)
                self._pending_events = [
                    event for event in self._pending_events
                    if event != pending_26
                ]
        self._tx_wakeup.set()
        self._diagnostics.report(
            self, labels, raw_objects if raw_objects is not None else verified_input_objects,
            verified_input_objects, objects, selected, verification_diagnostics,
            mode, self._selection_diagnostic_policy, diagnostic_anchor, diagnostic_previous,
            before_jump_target is not None and selected is None, continuity_removed, now_ms,
            "_track_target_lock" if target_in_roi else diagnostic_anchor_attribute,
            overlap_objects=raw_tracking_objects,
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
            or capture_target is not None
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
        obj, reference_x, reference_y, target_x_ratio=None
    ):
        """Return ROI error to the configured horizontal point in a zone."""
        if target_x_ratio is None:
            target_x_ratio = config.UART_ZONE_SUPPLY_X_RATIO
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

    def _build_claw_coordinate(self, obj, error_x, error_y):
        wire_id = int(obj.class_id)
        if self._claw_geometry and wire_id in (3, 4, 5, 6):
            ratio = self._visible_box_area_ratio(obj)
            wire_id |= OBJECT_GEOMETRY_PRESENT
            if ratio >= config.UART_CLAW_APPROACH_AREA_RATIO:
                wire_id |= OBJECT_GEOMETRY_NEAR
            if ratio >= config.UART_CLAW_OPEN_AREA_RATIO:
                wire_id |= OBJECT_GEOMETRY_OPEN_AREA
        return build_coordinate_packet(wire_id, error_x, error_y)

    def _claw_capture_gate(self, obj, reference_x):
        if not self._claw_geometry:
            return True
        center_x = float(obj.x) + float(obj.w) * 0.5
        return (abs(round(center_x - reference_x)) <= config.UART_CLAW_CENTER_TOLERANCE_X
                and self._visible_box_area_ratio(obj) >= config.UART_CLAW_OPEN_AREA_RATIO)

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
        with self._lock:
            if self.mode != MODE_ARRANGE_RIGHTMOST:
                return []
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
    def _safety_zone_exclusion_bounds(zone, horizontal_padding=None):
        """Expand only X; callers use zero padding for zone06 obstacles."""
        if horizontal_padding is None:
            horizontal_padding = config.UART_SAFE_ZONE_TARGET_MARGIN_X
        padding = max(0.0, float(horizontal_padding))
        return (
            float(zone.x) - padding,
            float(zone.y),
            float(zone.x) + float(zone.w) + padding,
            float(zone.y) + float(zone.h),
        )

    @staticmethod
    def _exclude_objects_in_safety_zones(
        objects, labels, zone_detections, horizontal_padding=None,
    ):
        """Exclude cargo in BOTH zone colors, independent of the destination.

        The same bounds cover own/opponent zones and their X side margins.
        Keep zone detections themselves available for navigation/DEBUG.
        """
        zone_labels = set(ZONE_SEARCH_COMMANDS.values())
        zones = [
            VisionSerialController._safety_zone_exclusion_bounds(zone, horizontal_padding)
            for zone in zone_detections
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
                    left <= center_x <= right and top <= center_y <= bottom
                    for left, top, right, bottom in zones
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

    def _near_view_qualified_objects(self, objects):
        """Keep cargo centers in the 05 ROI and apply the full-frame area limits."""
        minima = {
            4: getattr(config, "UART_NEAR_VIEW_RED_MIN_AREA_RATIO", 0.15),
            6: getattr(config, "UART_NEAR_VIEW_BLACK_MIN_AREA_RATIO", 0.03),
        }
        frame_area = float(max(1, self.frame_width * self.frame_height))
        eligible = []
        for obj in objects:
            if self.near_view_roi is not None:
                center_x = float(obj.x) + float(obj.w) * 0.5
                center_y = float(obj.y) + float(obj.h) * 0.5
                if not point_in_near_view_roi(center_x, center_y, self.near_view_roi):
                    continue
            minimum = minima.get(int(obj.class_id))
            if minimum is not None:
                if clipped_yolo_rect(obj, self.frame_width, self.frame_height) is None:
                    continue
                # Keep fractional YOLO geometry, matching DEBUG's area label.
                left = max(0.0, float(obj.x))
                top = max(0.0, float(obj.y))
                right = min(float(self.frame_width), float(obj.x) + float(obj.w))
                bottom = min(float(self.frame_height), float(obj.y) + float(obj.h))
                # Offset fractional boxes can lose a few floating-point bits
                # during clipping; an exact threshold must still count.
                if (right - left) * (bottom - top) + 1e-9 < frame_area * minimum:
                    continue
            eligible.append(obj)
        return eligible

    def _near_view_target_ids(self, objects, labels):
        return self._movable_target_ids(self._near_view_qualified_objects(objects), labels)

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
        if not config.VISION_DIAGNOSTICS_ENABLED or not config.VISION_DIAGNOSTICS_VERBOSE:
            return
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
        """Use the same triangle as the visible ROI border."""
        center_x = float(obj.x) + float(obj.w) * 0.5
        center_y = float(obj.y) + float(obj.h) * 0.5
        return point_in_tracking_roi(center_x, center_y, self.tracking_roi)

    def _final_capture_reached(self, obj, capture_line_y):
        """Accept an ROI entry or a visible center below the capture line."""
        center_x = float(obj.x) + float(obj.w) * 0.5
        center_y = float(obj.y) + float(obj.h) * 0.5
        return (self.tracking_roi is not None
                and 0 <= center_x < self.frame_width and 0 <= center_y < self.frame_height
                and (center_y > capture_line_y
                     or point_in_tracking_roi(center_x, center_y, self.tracking_roi)))

    def _casualty_in_capture_area(self, obj, reference_x):
        if (obj is None or int(obj.class_id) != 4
                or not self._target_center_in_tracking_roi(obj)):
            return False
        _, x, y, _ = self._final_target_identity(obj)
        min_y = config.UART_CASUALTY_CAPTURE_MIN_Y_PX * self.frame_height / 448
        max_x_error = config.UART_CASUALTY_CAPTURE_X_TOLERANCE_PX * self.frame_width / 576
        return (0 <= x < self.frame_width and min_y <= y < self.frame_height
                and abs(x - reference_x) <= max_x_error)

    def _casualty_capture_candidate(self, selected, raw_candidates, reference_x, now_ms, generation):
        """Keep the post07 raw-ROI handoff without replacing the steering lock."""
        if now_ms < self._final_capture_ready_ms:
            return None
        def eligible(obj):
            return self._casualty_in_capture_area(obj, reference_x) and self._claw_capture_gate(obj, reference_x)

        candidates = [obj for obj in raw_candidates if eligible(obj)]
        if all(obj is not selected for obj in candidates) and eligible(selected):
            candidates.append(selected)
        if not candidates:
            return None
        previous = self._casualty_capture_streak
        if (previous is not None and previous[0] == generation
                and 0 <= now_ms - previous[1] <= config.UART_TARGET_FRESHNESS_MS):
            _, _, old_x, old_y, _ = previous

            def distance(obj):
                _, x, y, _ = self._final_target_identity(obj)
                return (x - old_x) ** 2 + (y - old_y) ** 2

            return min(candidates, key=distance)
        return candidates[0]

    def _confirm_casualty_capture(self, candidate, now_ms, generation):
        """Confirm consecutive observations of one deep candidate, under the lock."""
        if candidate is None:
            self._casualty_capture_streak = None
            return False
        _, x, y, _ = self._final_target_identity(candidate)
        required = max(1, config.UART_CASUALTY_CAPTURE_CONFIRM_FRAMES)
        previous = self._casualty_capture_streak
        count = 1
        if previous is not None:
            old_generation, old_ms, old_x, old_y, old_count = previous
            max_step = 25 * self.frame_height / 448
            if (old_generation == generation
                    and 0 <= now_ms - old_ms <= config.UART_TARGET_FRESHNESS_MS
                    and (x - old_x) ** 2 + (y - old_y) ** 2 <= max_step ** 2):
                count = min(required, old_count + 1)
        self._casualty_capture_streak = (generation, now_ms, x, y, count)
        return count >= required

    def _reference_point(self, mode, target_label=None):
        """Return the steering reference, also used for state 24 capture."""
        if (mode == MODE_FINAL_ROI_TRACK
                and (self.target_label if target_label is None else target_label)
                == config.UART_ZONE_CASUALTY_LABEL):
            vertices = (roi_triangle_vertices(self.tracking_roi)
                        if getattr(config, "UART_TRACKING_ROI_SHAPE", "triangle") != "rectangle"
                        else None)
            x = self.frame_width * .5
            if vertices is not None:
                x = final_track_reticle_point(vertices[0])[0]
            elif self.tracking_roi is not None:
                x = self.tracking_roi[0] + self.tracking_roi[2] * .5
            return x, round(config.UART_CASUALTY_TRACK_AIM_Y_PX * self.frame_height / 448)
        if (mode in (CMD_SEARCH, CMD_TRACK, MODE_WAIT_04_HANDSHAKE, MODE_FINAL_ROI_TRACK)
                and getattr(config, "UART_TRACKING_ROI_SHAPE", "triangle") != "rectangle"):
            vertices = roi_triangle_vertices(self.tracking_roi)
            if vertices is not None:
                apex = vertices[0]
                if mode == MODE_FINAL_ROI_TRACK:
                    return final_track_reticle_point(apex)
                if mode == CMD_TRACK:
                    return roi_reticle_point(apex)
                return search_reticle_point(apex)
        if mode in (CMD_SEARCH, CMD_TRACK, MODE_WAIT_04_HANDSHAKE, MODE_ARRANGE_RIGHTMOST):
            return self.frame_width * 0.5, self.frame_height * 0.5
        if mode in MICRO_MODES:
            if self.tracking_roi is None:
                return self.frame_width * .5, self.frame_height * .5
            left, top, width, height = map(float, self.tracking_roi)
            #Both32 and42 track the ROI center;42 alternation changes ONLY
            #the MCU's first turn direction, never the coordinate reference.
            return left + width * .5, top + height * .5
        if mode == MODE_ARRANGE_LEFTMOST:
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
            roi_y_offset = -20 if mode == MODE_FINAL_ROI_TRACK else 0
            return (
                float(left) + float(width) * 0.5,
                float(top) + float(height) * 0.5 + roi_y_offset,
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

        # Red below image center and green/black check every Y; distant red
        # and blue retain their absolute-Y-band scope.
        # All applicable neighbors use this candidate's dx threshold.
        neighbors = []
        neighbor_dx = {}
        for candidate in candidates:
            center_x = float(candidate.x) + float(candidate.w) * 0.5
            center_y = float(candidate.y) + float(candidate.h) * 0.5
            for other in movable:
                if other is candidate:
                    continue
                other_y = float(other.y) + float(other.h) * 0.5
                if not spacing_neighbor_in_y_scope(
                    candidate.class_id, center_y, other_y, self.frame_height * 0.5
                ):
                    continue
                red_pair = (
                    int(candidate.class_id) == 4
                    or int(other.class_id) == 4
                )
                min_dx, _ = self._final_spacing_limits(center_y, red_pair)
                other_x = float(other.x) + float(other.w) * 0.5
                dx = abs(other_x - center_x)
                if dx > min_dx:
                    continue
                if all(obj is not other for obj in neighbors):
                    neighbors.append(other)
                previous = neighbor_dx.get(id(other))
                if previous is None or dx < previous:
                    neighbor_dx[id(other)] = dx
        secondary = [
            candidate for candidate in candidates
            if self._matches_final_primary_spacing(
                candidate, movable, self.frame_height * 0.5
            )
        ]
        preview = min(
            secondary,
            key=lambda obj: (
                abs(float(obj.y) + float(obj.h) * 0.5 - self._search_selection_reference()[1]),
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
        return (target_label == "sqareredgreen"
                and not self._first_green_near_view_decided
                and not self._post07_started)

    def _uses_expanded_overlap_decision(self, target_label):
        return (target_label in ("sqareredgreen", "sqarered")
                or self._post07_started and target_label in (
                    "triangualrblack", TARGET_BLACK_OR_GREEN))

    def _uses_supply_collection(self):
        return self._post07_started and self.target_label == TARGET_BLACK_OR_GREEN

    @staticmethod
    def _is_supply_acquisition(mode, target_label):
        return (mode in (CMD_SEARCH, CMD_TRACK, MODE_WAIT_04_HANDSHAKE,
                         MODE_WAIT_14_HANDSHAKE, MODE_FINAL_ROI_TRACK)
                and target_label in ("sqareredgreen", "triangualrblack", TARGET_BLACK_OR_GREEN))

    def _can_request_supply_final_inventory(self):
        """Accept MCU05 during a post07/51 capture or recovery round."""
        return (
            self._uses_supply_collection()
            and self._red_switch_stage is None
            and self.mode in (
                CMD_SEARCH, CMD_TRACK, MODE_WAIT_04_HANDSHAKE,
                MODE_WAIT_14_HANDSHAKE, MODE_FINAL_ROI_TRACK,
                MODE_WAIT_SUPPLY_SEARCH, MODE_NEAR_VIEW_CHECK,
                MODE_ARRANGE_RIGHTMOST, MODE_ARRANGE_LEFTMOST,
                MODE_POST_ARRANGE_CHECK,
            ) + MICRO_MODES
        )

    def _reset_supply_collection(self, new_task=False):
        # Stopping a timer or discarding near-view cargo is NOT a new mission.
        # MCU retains its exhausted budget through02/12/22/24 and same51/03.
        if new_task:
            self._supply_collect_finished = False
            self._supply_collect_finish_reason = None
            self._supply_first_capture_bypassed = False
        self._supply_collect_attempts = 0
        self._supply_collect_started_ms = None
        self._supply_search_ready_ms = None
        self._supply_loaded_roi_anchors = []
        self._supply_roi_observed = False
        self._supply_roi_handoff = None
        self._supply_roi_handoff_seen_ms = 0
        self._reset_supply_acquisition_guard()

    def _clear_supply_tracking_output(self):
        self._reset_supply_acquisition_guard()
        self._reset_micro_adjust()
        self._direct_capture_pending = False
        self._latest_packet = None
        self._latest_submit_ms = 0
        self._search_target_lock = None
        self._track_target_lock = None
        self._final_target_lock = None
        self._supply_loaded_roi_anchors = []
        self._supply_roi_observed = False
        self._supply_roi_handoff = None
        self._supply_roi_handoff_seen_ms = 0
        self._last_target_center_x = None
        self._last_target_center_y = None
        self._last_target_area = None
        self._last_target_accept_ms = 0
        self._tracking_missing_since_ms = None
        self._tracking_missing_class_id = None
        self._search_no_target_frames = 0
        self._search_no_target_initial_sent = False
        self._search_no_target_acknowledged = False
        self._search_e3_active = False
        self._track_no_target_frames = 0
        self._track_no_target_sent = False
        self._track_green_recovery_waiting = False
        self._final_no_target_frames = 0
        self._roi_entry_sent = False
        self._track_center_sent = False
        self._final_roi_sent = False
        self._final_search_waiting_ack = False
        self._initial_arrangement_decision_pending = False
        self._arrangement_02_ack_pending = False
        self._arrangement_02_request_sent = False

    def _startzone_area_ratio(self, objects):
        """Largest visible FLOAT bbox / whole frame, never a sum of boxes."""
        maximum = 0.0
        for obj in objects:
            try:
                if int(obj.class_id) != 8:
                    continue
                x, y, width, height = map(float, (obj.x, obj.y, obj.w, obj.h))
                if not all(math.isfinite(value) for value in (x, y, width, height)):
                    continue
                if width <= 0 or height <= 0:
                    continue
                visible_width = max(0.0, min(float(self.frame_width), x + width) - max(0.0, x))
                visible_height = max(0.0, min(float(self.frame_height), y + height) - max(0.0, y))
                maximum = max(maximum, visible_width * visible_height)
            except (AttributeError, TypeError, ValueError, OverflowError):
                continue
        return maximum / max(1, self.frame_width * self.frame_height)

    def _begin_startzone_escape(self, area_ratio):
        if not getattr(config, "UART_STARTZONE_ESCAPE_ENABLED", False):
            return
        # Fence old inference AND prepared UART writes before queuing AE.
        # Keep mission/post07 history and the original51 attempts/deadline.
        self._clear_supply_tracking_output()
        self._reset_initial_arrangement_confirmation()
        self._pending_events.clear()
        self._coordinate_in_flight = None
        self.mode = MODE_WAIT_STARTZONE_ESCAPE
        self._startzone_escape_phase = "pending"
        self._startzone_escape_armed = False
        self._startzone_deferred_zone_command = None
        self._command_generation += 1
        self._tx_wakeup.set()
        print("[STARTZONE-AE] area={:.2f}% action=TXAE wait_turn180 coordinates=paused".format(
            area_ratio * 100.0))

    @staticmethod
    def _zone_obstacle_enabled():
        return bool(getattr(config, "UART_ZONE_OBSTACLE_ENABLED", False))

    def _clear_disabled_zone_obstacle(self):
        """A disabled36 cannot leave a queued request or coordinate pause."""
        if self._zone_obstacle_enabled():
            return
        self._zone_obstacle_waiting = False
        self._zone_obstacle_request_sent = False
        self._zone_obstacle_area_allowed = False
        self._zone_obstacle_armed = False
        self._zone_obstacle_frames = 0
        self._zone_obstacle_clear_frames = 0
        self._last_zone_obstacle_event_ms = 0
        request = build_event_packet(EVENT_ZONE_OBSTACLE)
        if request in self._pending_events:
            self._pending_events = [packet for packet in self._pending_events if packet != request]

    def _reset_zone_approach(self):
        # A same-delivery05 recheck must not repeat an already completed46
        # correction (or revisit the decision after26 has actually been sent).
        if self._zone_realign_phase != "done":
            self._zone_realign_phase = None
        self._last_zone_realign_write_ms = None
        self._zone_found_sent = False
        self._zone_area_ratio = 0.0
        self._zone_no_target_phase = None
        self._zone_close_sent = False
        self._zone_close_active = False
        self._last_zone_close_event_ms = None
        self._zone_close_confirm_frames = 0
        self._zone_close_confirm_seen_ms = None
        self._zone_obstacle_waiting = False
        self._zone_obstacle_request_sent = False
        self._zone_obstacle_area_allowed = False
        self._zone_obstacle_armed = True
        self._zone_obstacle_frames = 0
        self._zone_obstacle_clear_frames = 0
        self._last_zone_obstacle_event_ms = 0

    def _update_zone_realign(self, zone, bars, zone_is_close, now_ms):
        """First26 needs five close frames; ONE wrong-side bar latches46.

        Continue steering during confirmation. A completed46 or successfully
        sent26 keeps the existing per-delivery bypass, without another wait.
        """
        if self._zone_realign_phase == "done":
            return zone_is_close
        if self._zone_realign_phase in ("pending", "wait_ack"):
            # A one-frame veto must survive the following missing/correct
            # bar. Only task changes, stale-camera safety or MCU46 release it.
            self._latest_packet = None
            self._latest_submit_ms = 0
            return False
        if not zone_is_close:
            self._zone_close_confirm_frames = 0
            self._zone_close_confirm_seen_ms = None
            return False
        bar = select_zone_blackbar(bars, zone, self.frame_width, self.frame_height)
        casualty = self.target_label == config.UART_ZONE_CASUALTY_LABEL
        wrong_side = blackbar_requires_realign(bar, zone, casualty)
        request = build_event_packet(EVENT_ZONE_REALIGN)
        if zone_is_close and wrong_side:
            self._zone_close_confirm_frames = 0
            self._zone_close_confirm_seen_ms = None
            if self._zone_realign_phase != "pending":
                print("[ZONE-46] zone_cx={:.1f} bar_cx={:.1f} cargo={} action=TX46_WAIT_ACK".format(
                    float(zone.x) + float(zone.w) * .5,
                    float(bar.x) + float(bar.w) * .5,
                    "casualty_left90" if casualty else "supplies_right90"))
            self._zone_realign_phase = "pending"
            # Retain16 ahead of46 if this is the first sighting; discard old
            # close/missing/obstacle events and all steering immediately.
            self._pending_events = [event for event in self._pending_events
                                    if event == build_event_packet(EVENT_ZONE_FOUND)]
            self._pending_events.append(request)
            self._latest_packet = None
            self._latest_submit_ms = 0
            self._zone_close_sent = False
            return False
        required = max(1, int(getattr(config, "UART_ZONE_CLOSE_CONFIRM_FRAMES", 5)))
        if (self._zone_close_confirm_seen_ms is None
                or not 0 <= now_ms - self._zone_close_confirm_seen_ms <= config.UART_TARGET_FRESHNESS_MS):
            self._zone_close_confirm_frames = 0
        self._zone_close_confirm_frames = min(required, self._zone_close_confirm_frames + 1)
        self._zone_close_confirm_seen_ms = now_ms
        return self._zone_close_confirm_frames >= required

    def debug_zone_realign_bar(self, objects, labels, bars, frame_generation=None):
        """Read-only current-frame46 highlight, never queue events/change locks.

        Use the same largest commanded zone, visible50% area and selected
        blackbar as UART. A sent26/completed46 disables red for this delivery.
        Pending/sent46 stays red only while its CURRENT geometry qualifies.
        """
        with self._lock:
            if (frame_generation is not None and frame_generation != self._command_generation
                    or self.mode != MODE_SEARCH_ZONE or self._zone_realign_phase == "done"
                    or self._zone_load_recheck is not None):
                return None
            reference_x, _ = self._reference_point(MODE_SEARCH_ZONE)
            zone = self._select_largest_zone(objects, labels, self.zone_label, reference_x)
            area = self._visible_box_area_ratio(zone) if zone is not None else 0.0
            if (zone is None or area <= getattr(config, "UART_ZONE_MIN_AREA_RATIO", .02)
                    or area < config.UART_ZONE_CLOSE_AREA_RATIO):
                return None
            bar = select_zone_blackbar(bars, zone, self.frame_width, self.frame_height)
            return bar if blackbar_requires_realign(
                bar, zone, self.target_label == config.UART_ZONE_CASUALTY_LABEL) else None

    def _begin_zone_load_recheck(self, source="evacuation"):
        """Request an isolated05 inventory without a collection/E5 cycle."""
        if self._uses_supply_collection() and not self._supply_collect_finished:
            self._supply_collect_finished = True
            self._supply_collect_finish_reason = "final_inventory"
        self._reset_supply_collection()
        self._supply_collect_delivery_pending = (source == "before06" and self._uses_supply_collection())
        self._clear_supply_tracking_output()
        self._pending_events.clear()
        self._pending_events.append(build_event_packet(EVENT_TARGET_READY))
        self._reset_zone_approach()
        self.mode = MODE_NEAR_VIEW_CHECK
        self._zone_load_recheck = "wait05tx"
        self._zone_load_recheck_source = source
        self._zone_load_resume_ready_ms = 0
        self._near_view_started_ms = None
        self._near_view_adjust_sent = False
        self._near_view_result_sent = False
        self._near_view_stable_bucket = None
        self._near_view_stable_frames = 0
        self._command_generation += 1
        self._tx_wakeup.set()
        print("[LOAD-CHECK] source={} action=TX05 task={} zone={} wait_near_view coordinates=paused".format(
            source, self.target_label, self.zone_label))

    def _finish_zone_load_recheck(self, ids, boxes=()):
        """Keep this delivery task; never request another51 capture here."""
        if self._zone_load_recheck_source != "before06":
            # Even evacuation's first successful inventory must make the
            # separately requested final05 before allowing any06/cargo ID.
            return self._accept_near_view_load(ids)
        self._clear_supply_tracking_output()
        self._pending_events.clear()
        if (self.target_label == config.UART_ZONE_CASUALTY_LABEL and ids == [4]
                and any(int(obj.class_id) == 4 and 0 < float(obj.h)
                        and float(obj.w) > 1.5 * float(obj.h) for obj in boxes)):
            self._zone_load_recheck = "wait45tx"
            self._pending_events.append(build_event_packet(EVENT_WIDE_CASUALTY_LOAD_ADJUST))
            self._command_generation += 1
            self._tx_wakeup.set()
            return "TX45 wide_casualty wait_RX45 before06"
        result = self._publish_near_view_load(ids)
        self._reset_zone_approach()
        self._zone_load_recheck = "wait06tx"
        self._zone_load_resume_ready_ms = 0
        self._command_generation += 1
        self._tx_wakeup.set()
        return "zone_recheck {} wait_wide_view".format(result)

    def _expire_zone_load_resume(self, now_ms):
        if (self._zone_load_recheck == "wide_settle"
                and now_ms >= self._zone_load_resume_ready_ms):
            self._zone_load_recheck = None
            self._zone_load_recheck_source = None
            self._zone_load_resume_ready_ms = 0
            # Also drop an inference captured before the settling deadline.
            self._command_generation += 1
            self._tx_wakeup.set()
            print("[ZONE-RECHECK] resume state={:X} zone={} fresh_wide_frame_required".format(
                self.mode, self.zone_label))

    def _accept_near_view_load(self, ids):
        """Every intent to deliver first requests a NEW final05 inventory."""
        self._begin_zone_load_recheck("before06")
        return "TX05 final_inventory_before06 previous_ids={}".format(ids)

    def _publish_near_view_load(self, ids):
        """Only the final inventory's fresh confirmed result may emit06."""
        if (self._zone_load_recheck_source != "before06"
                or self._zone_load_recheck != "count05"):
            raise RuntimeError("06 requires a confirmed final05 inventory")
        cargo_id = (4 if ids == [4] else
                    5 if ids and all(value in (5, 6) for value in ids) else None)
        if cargo_id is not None:
            self._pending_events.append(build_coordinate_packet(cargo_id, 0, 0))
        self._pending_events.append(build_event_packet(EVENT_ZONE_SEARCH_STARTED))
        self._near_view_result_sent = True
        self._near_view_empty_cycles = 0
        self._near_view_green_missing_cycles = 0
        self.mode = MODE_SEARCH_ZONE
        return "cargo_id={} then TX06 search_zone".format(cargo_id)

    def _request_supply_collection(self, now_ms, mcu_started=False):
        if self._supply_collect_finished:
            print("[COLLECT-51] finished=1 reason={} action=TX05_FINAL_INVENTORY no_new_TX03".format(
                self._supply_collect_finish_reason))
            self._accept_near_view_load([5])
            return
        if self._supply_collect_started_ms is None:
            self._supply_collect_started_ms = now_ms
        if self._supply_collect_attempts >= getattr(config, "UART_SUPPLY_COLLECT_MAX_ATTEMPTS", 2):
            self._finish_supply_collection("attempt_limit", now_ms)
            return
        self._supply_collect_attempts += 1
        self._clear_supply_tracking_output()
        self._pending_events.clear()
        if not mcu_started:
            self._pending_events.append(build_event_packet(EVENT_COLLECT_MORE_SUPPLIES))
        self.mode = MODE_WAIT_SUPPLY_SEARCH
        self._near_view_result_sent = True
        self._command_generation += 1
        print("[COLLECT-51] attempt={}/{} elapsed_ms={} action={}".format(
            self._supply_collect_attempts, getattr(config, "UART_SUPPLY_COLLECT_MAX_ATTEMPTS", 2),
            now_ms - self._supply_collect_started_ms,
            "RX34_SKIP05_WAIT_RX51_RX03" if mcu_started else "TX03_WAIT_RX03"))

    def _finish_supply_collection(self, reason, now_ms):
        elapsed = now_ms - self._supply_collect_started_ms if self._supply_collect_started_ms is not None else 0
        if not self._supply_collect_finished:
            self._supply_collect_finished = True
            self._supply_collect_finish_reason = reason
        print("[COLLECT-51] attempt={} elapsed_ms={} reason={} action=TX05_FINAL_INVENTORY".format(
            self._supply_collect_attempts, elapsed, reason))
        self._accept_near_view_load([5])
        self._supply_collect_delivery_pending = True

    def _expire_supply_collection(self, now_ms):
        if self._startzone_escape_phase is not None:
            # Do not emit05/06 during uninterruptible motion. The original
            # deadline is NOT extended; check it immediately after RXAE.
            return
        timeout_ms = getattr(config, "UART_SUPPLY_COLLECT_TIMEOUT_MS", 12000)
        if (not self._supply_collect_finished
                and self._supply_collect_started_ms is not None and self._uses_supply_collection()
                and now_ms - self._supply_collect_started_ms >= timeout_ms):
            self._finish_supply_collection("total_timeout", now_ms)

    def _supply_search_settling(self, now_ms):
        if self._supply_search_ready_ms is None:
            return False
        if now_ms < self._supply_search_ready_ms:
            return True
        self._supply_search_ready_ms = None
        # Discard inferences captured while the camera was switching views.
        self._command_generation += 1
        return False

    def _target_center_in_supply_loaded_roi(self, obj):
        """Command51 cargo mask: bottom/inner half, not the entire ROI."""
        center_x = float(obj.x) + float(obj.w) * .5
        center_y = float(obj.y) + float(obj.h) * .5
        if self.tracking_roi is not None:
            left, top, width, height = self.tracking_roi
            if (self._target_center_in_tracking_roi(obj)
                    and float(top) + float(height) * .5 <= center_y):
                return True
        extra_roi = getattr(config, "UART_SUPPLY_COLLECT_LOADED_ROI", None)
        return (extra_roi is not None
                and float(extra_roi[0]) <= center_x < float(extra_roi[0] + extra_roi[2])
                and float(extra_roi[1]) <= center_y < float(extra_roi[1] + extra_roi[3]))

    def _unloaded_supply_candidates(self, objects):
        # A custom cargo mask extends, but cannot replace, the inner half.
        return [obj for obj in objects if not (
            int(obj.class_id) in (5, 6) and self._target_center_in_supply_loaded_roi(obj))]

    def _observe_supply_roi_entry(self, raw, outside, loaded_anchors,
                                  target_anchor, continuing_handoff, target_label):
        """Separate existing cargo from a locked24 target entering the ROI.

        Pure frame preparation; identities are committed only after the
        generation check. Associate loaded boxes one-to-one, preferring the
        same class, and retain missing cargo identities until a camera/task
        reset. Thus a flickering carried box cannot become a new34 entry.
        No jump/area gates are added to normal outside-ROI tracking.
        """
        inside = [obj for obj in raw if int(obj.class_id) in (5, 6)
                  and self._target_center_in_supply_loaded_roi(obj)]
        updated = list(loaded_anchors)
        pairs = []
        for index, anchor in enumerate(loaded_anchors):
            for obj_index, obj in enumerate(inside):
                identity = self._final_target_identity(obj)
                pairs.append((identity[0] != anchor[0],
                              (identity[1] - anchor[1]) ** 2 + (identity[2] - anchor[2]) ** 2,
                              index, obj_index))
        matched_anchors, carried_indices = set(), set()
        for _, _, index, obj_index in sorted(pairs):
            if index not in matched_anchors and obj_index not in carried_indices:
                matched_anchors.add(index)
                carried_indices.add(obj_index)
                updated[index] = self._final_target_identity(inside[obj_index])
        new_inside = [obj for index, obj in enumerate(inside) if index not in carried_indices]
        associated = None
        if target_anchor is not None:
            candidates = [obj for obj in list(outside) + new_inside
                          if int(obj.class_id) == target_anchor[0]
                          or self._uses_relaxed_supply_tracking(target_label) and int(obj.class_id) in (5, 6)]
            if candidates:
                associated = min(candidates, key=lambda obj: (
                    (float(obj.x) + float(obj.w) * .5 - target_anchor[1]) ** 2
                    + (float(obj.y) + float(obj.h) * .5 - target_anchor[2]) ** 2,
                    -float(getattr(obj, "score", 0.0)),
                ))
        handoff = associated if associated is not None and self._target_center_in_tracking_roi(associated) else None
        # Only this known outside-INNER-half target gets an entry handoff.
        # Unmatched INNER-half detections become cargo; front-half boxes remain
        # eligible candidates. The caller decides when the entry qualifies for34.
        updated.extend(self._final_target_identity(obj) for obj in new_inside if obj is not handoff)
        returned_outside = associated if continuing_handoff and associated is not handoff else None
        return handoff, returned_outside, updated

    def _reset_supply_acquisition_guard(self):
        # Shared by ALL cargo tasks; legacy supply prefix is internal only.
        self._supply_recent_zones = []
        self._supply_outside_history = []
        self._new_target_confirmation_pending = False

    def _remember_supply_zones(self, detections, labels, previous, now_ms):
        """Bounded per-color current boxes plus at most500ms missing-color memory.

        A fresh color replaces that color's old rectangles, avoiding motion
        trails. Store geometry only, never reusable cargo/steering detections.
        """
        current = []
        seen_colors = set()
        for zone in detections:
            class_id = int(zone.class_id)
            if (class_id not in (1, 2) or class_id >= len(labels)
                    or labels[class_id] not in ZONE_SEARCH_COMMANDS.values()
                    or clipped_yolo_rect(zone, self.frame_width, self.frame_height) is None):
                continue
            current.append((class_id, self._safety_zone_exclusion_bounds(zone), now_ms))
            seen_colors.add(class_id)
        hold_ms = max(0, getattr(config, "UART_TARGET_ZONE_HOLD_MS", 500))
        current.extend(entry for entry in previous
                       if hold_ms > 0 and entry[0] not in seen_colors
                       and 0 <= now_ms - entry[2] <= hold_ms)
        return current

    @staticmethod
    def _exclude_recent_supply_zones(objects, zones):
        eligible = []
        for obj in objects:
            if int(obj.class_id) in (3, 4, 5, 6):
                center_x = float(obj.x) + float(obj.w) * .5
                center_y = float(obj.y) + float(obj.h) * .5
                if any(left <= center_x <= right and top <= center_y <= bottom
                       for _, (left, top, right, bottom), _ in zones):
                    continue
            eligible.append(obj)
        return eligible

    def _confirm_supply_outside_targets(self, objects, previous, now_ms,
                                        post07_supply_acquisition=False):
        """One-to-one same-class evidence for NEW cargo in EVERY task.

        Keep at most one record per current raw box; a miss, large motion,
        class change or stale frame gap restarts confirmation.
        This distance gate never constrains an already acquired lock.
        """
        identities = [self._final_target_identity(obj) for obj in objects
                      if int(obj.class_id) in (3, 4, 5, 6)
                      and clipped_yolo_rect(obj, self.frame_width, self.frame_height) is not None]
        max_delta = max(0, getattr(config, "UART_NEW_TARGET_ASSOC_MAX_PX", 100))
        required = max(1, int(getattr(config, "UART_NEW_TARGET_CONFIRM_FRAMES", 3)))
        supply_required = (max(1, int(getattr(config, "UART_POST07_SUPPLY_CONFIRM_FRAMES", 2)))
                           if post07_supply_acquisition else required)
        pairs = []
        for index, identity in enumerate(identities):
            for old_index, (old, hits, last_ms) in enumerate(previous):
                if (identity[0] == old[0] and 0 <= now_ms - last_ms <= config.UART_TARGET_FRESHNESS_MS
                        and abs(identity[1] - old[1]) <= max_delta
                        and abs(identity[2] - old[2]) <= max_delta):
                    pairs.append(((identity[1] - old[1]) ** 2 + (identity[2] - old[2]) ** 2,
                                  index, old_index))
        matched, used_old = {}, set()
        for _, index, old_index in sorted(pairs):
            if index not in matched and old_index not in used_old:
                matched[index] = old_index
                used_old.add(old_index)
        updated = []
        confirmed = set()
        for index, identity in enumerate(identities):
            required_hits = supply_required if identity[0] in (5, 6) else required
            hits = min(required_hits, previous[matched[index]][1] + 1) if index in matched else 1
            updated.append((identity, hits, now_ms))
            if hits >= required_hits:
                confirmed.add(identity)
        return confirmed, updated

    def _select_supply_collection_target(self, objects, raw, labels, now_ms):
        self._selection_diagnostic_policy = "collect51_unloaded_yolo"
        objects = self._unloaded_supply_candidates(objects)
        raw = self._unloaded_supply_candidates(raw)
        selected = self._select_final_roi_target(objects, labels, TARGET_BLACK_OR_GREEN, now_ms)
        if selected is not None:
            return selected
        candidates = [obj for obj in raw if int(obj.class_id) in (5, 6)]
        return select_nearest_horizontal_target(
            candidates, labels, self.frame_width,
            reference_x=self.frame_width * .5, reference_y=self.frame_height * .5)

    def _reset_micro_adjust(self):
        self._micro_rounds = 0
        self._micro42_rounds = 0
        #Legacy name:42's FIRST TURN direction, not a left/right aim point.
        self._micro42_right = False
        self._micro_request_sent = False
        self._micro_started_ms = None
        self._micro_decision_ms = None
        self._micro_loss_pending = False

    def _directional_neighbor_objects(self, candidate, objects):
        """51 collection can grab green/black together; foreign cargo still blocks.

        Ignore ROI cargo for EVERY task's neighbor decision. Filter NEIGHBORS
        only, never the acquisition/association/count stream here.
        The approached identity stays the candidate even with closer supplies.
        """
        ignore_supplies = (candidate is not None and int(candidate.class_id) in (5, 6)
                           and self._uses_supply_collection())
        return [obj for obj in objects if obj is candidate or (
            not self._target_center_in_tracking_roi(obj)
            and not (ignore_supplies and int(obj.class_id) in (5, 6)))]

    def _directional_arrangement_result(self, candidate, objects):
        """Shared14 result and ALL blocking neighbors, without changing state."""
        if candidate is None:
            return EVENT_ROI_READY, []
        horizontal, vertical = directional_target_neighbors(
            candidate, self._directional_neighbor_objects(candidate, objects),
            self.frame_width, self.frame_height)
        if casualty_vertical_grab_allowed(candidate, horizontal, vertical):
            return EVENT_SKIP_TO_FINAL_TRACK, []
        neighbors, seen = [], set()
        for obj in horizontal + vertical:
            if id(obj) not in seen:
                neighbors.append(obj)
                seen.add(id(obj))
        if horizontal and vertical:
            return EVENT_ROI_READY, neighbors
        if horizontal:
            return EVENT_MICRO_FORWARD, neighbors
        if vertical:
            return EVENT_MICRO_SIDE, neighbors
        return EVENT_SKIP_TO_FINAL_TRACK, []

    def _directional_arrangement_event(self, candidate, objects):
        return self._directional_arrangement_result(candidate, objects)[0]

    def debug_arrangement_objects(self, objects, labels, actual_selected=None, frame_generation=None,
                                  target_label=None):
        """Read-only14 preview of one current target, never a new UART decision.

        An explicit preview label overrides the MCU color for the screen only.
        Ignore incompatible actual/locked identities; never change UART state.
        Legacy callers without an override still follow the current task.
        """
        with self._lock:
            if frame_generation is not None and frame_generation != self._command_generation:
                return None, None, []
            forced_label = target_label is not None
            target_label = target_label if forced_label else self.target_label or "sqareredgreen"
            target_labels = (("sqareredgreen", "triangualrblack")
                             if not forced_label and self._post07_started and target_label in (
                                 "sqareredgreen", "triangualrblack", TARGET_BLACK_OR_GREEN)
                             else self._eligible_target_labels(target_label))
            zones = self._remember_supply_zones(objects, labels, self._supply_recent_zones, _now_ms())
            eligible = self._exclude_recent_supply_zones(objects, zones)
            movable = [obj for obj in eligible
                       if 0 < int(obj.class_id) < len(labels)
                       and labels[int(obj.class_id)] in (
                           "sqareblue", "sqarered", "sqareredgreen", "triangualrblack")
                       and clipped_yolo_rect(obj, self.frame_width, self.frame_height) is not None]
            if not forced_label and self._uses_supply_collection():
                movable = self._unloaded_supply_candidates(movable)
            candidates = [obj for obj in movable
                          if labels[int(obj.class_id)] in target_labels
                          and not self._target_center_in_tracking_roi(obj)]
            anchor = (self._final_target_lock if self.mode == MODE_FINAL_ROI_TRACK
                      else self._search_target_lock if self.mode in (CMD_SEARCH, MODE_WAIT_04_HANDSHAKE)
                      else self._track_target_lock or self._initial_confirmation_anchor)
            if (forced_label and anchor is not None
                    and (not 0 < anchor[0] < len(labels) or labels[anchor[0]] not in target_labels)):
                anchor = None
            if (actual_selected is not None
                    and 0 < int(actual_selected.class_id) < len(labels)
                    and labels[int(actual_selected.class_id)] in target_labels):
                anchor = self._final_target_identity(actual_selected)
            if anchor is not None:
                associated = [obj for obj in candidates if int(obj.class_id) == anchor[0]]
                if not associated:
                    return None, None, []
                candidate = min(associated, key=lambda obj: (
                    (float(obj.x) + float(obj.w) * .5 - anchor[1]) ** 2
                    + (float(obj.y) + float(obj.h) * .5 - anchor[2]) ** 2))
            else:
                candidate = (self._select_frame_center_target(candidates, labels, target_label)
                             if forced_label else
                             self._select_initial_overlap_candidate(candidates, labels, target_label))
            if candidate is None:
                return None, None, []
            event, neighbors = self._directional_arrangement_result(candidate, movable)
            return candidate, event, neighbors

    def _start_micro_adjust(self, event, target, now_ms):
        """Keep the same target; actual UART write starts ACK timing/round count."""
        self.mode = event
        self._track_target_lock = self._final_target_identity(target)
        self._micro_request_sent = False
        self._micro_started_ms = None
        self._micro_loss_pending = False
        self._micro_decision_ms = now_ms
        self._micro42_right = self._micro42_rounds % 2 == 1
        self._pending_events.clear()
        self._pending_events.append(build_event_packet(event))
        self._latest_packet = None
        self._latest_submit_ms = 0
        self._track_no_target_frames = 0
        self._track_no_target_sent = False
        self._track_green_recovery_waiting = False
        self._tracking_missing_since_ms = None
        self._tracking_missing_class_id = None
        self._final_roi_sent = False
        self._last_final_roi_event_ms = 0
        self._initial_arrangement_decision_pending = False
        self._reset_initial_arrangement_confirmation()
        self._command_generation += 1
        self._tx_wakeup.set()
        print("[MICRO] action=TX{:02X} target=id{} next_round={}/{} reference={} first_turn={} MCU_timed no34".format(
            event, self._track_target_lock[0], self._micro_rounds + 1,
            getattr(config, "UART_MICRO_ADJUST_MAX_ROUNDS", 3), self._reference_point(event),
            ("right" if self._micro42_right else "left") if event == EVENT_MICRO_SIDE else "center"))

    def _select_micro_target(self, objects, now_ms, generation):
        with self._lock:
            if generation != self._command_generation:
                return None
            anchor = self._track_target_lock
        self._selection_diagnostic_policy = "micro{:02X}_lock".format(self.mode)
        if anchor is None:
            return None
        candidates = [obj for obj in objects if int(obj.class_id) == anchor[0]
                      and clipped_yolo_rect(obj, self.frame_width, self.frame_height) is not None]
        if not candidates:
            with self._lock:
                if generation == self._command_generation:
                    self._record_tracking_missing(now_ms, anchor[0])
            return None
        return min(candidates, key=lambda obj: (
            (float(obj.x) + float(obj.w) * .5 - anchor[1]) ** 2
            + (float(obj.y) + float(obj.h) * .5 - anchor[2]) ** 2,
            -float(getattr(obj, "score", 0.0))))

    def _reset_initial_arrangement_confirmation(self):
        """Reset a first-green RX14/RX44 observation round (caller holds lock)."""
        self._initial_confirmation_started_ms = None
        self._initial_confirmation_anchor = None
        self._initial_confirmation_bucket = None
        self._initial_confirmation_frames = 0
        self._initial_confirmation_crowded_total = 0
        self._initial_confirmation_saw_clear = False
        self._initial_confirmation_observed_ms = None
        self._initial_confirmation_seen_ms = None

    def _select_initial_overlap_candidate(self, objects, labels, target_label):
        """Check the approached raw supply, never substitute a clearer target."""
        target_labels = (("sqareredgreen", "triangualrblack") if self._post07_started and target_label in (
                         "sqareredgreen", "triangualrblack", TARGET_BLACK_OR_GREEN)
                         else self._eligible_target_labels(target_label))
        candidates = [obj for obj in objects
                      if 0 < int(obj.class_id) < len(labels)
                      and labels[int(obj.class_id)] in target_labels
                      and clipped_yolo_rect(obj, self.frame_width, self.frame_height) is not None]
        if not candidates:
            return None
        anchor = self._initial_confirmation_anchor
        fallback_x, fallback_y = (self._search_selection_reference() if self._post07_started
                                  else (self.frame_width * .5, self.frame_height * .5))
        reference_x = anchor[1] if anchor is not None else fallback_x
        reference_y = anchor[2] if anchor is not None else fallback_y
        return min(candidates, key=lambda obj: (
            (float(obj.x) + float(obj.w) * .5 - reference_x) ** 2
            + (float(obj.y) + float(obj.h) * .5 - reference_y) ** 2
        ))

    def _observe_initial_green_decision(self, selected, objects, labels, now_ms):
        """Count fresh, same-target evidence; never treat a raw miss as crowding."""
        candidate = selected
        bucket = "missing"
        if candidate is not None:
            bucket = {EVENT_ROI_READY: "crowded", EVENT_MICRO_FORWARD: "horizontal",
                      EVENT_MICRO_SIDE: "vertical", EVENT_SKIP_TO_FINAL_TRACK: "clear"}[
                          self._directional_arrangement_event(candidate, objects)]
            self._initial_confirmation_seen_ms = now_ms
        identity = self._final_target_identity(candidate) if candidate is not None else None
        previous = self._initial_confirmation_anchor
        same_target = (identity is not None and previous is not None
                       and self._initial_confirmation_observed_ms is not None
                       and 0 <= now_ms - self._initial_confirmation_observed_ms <= config.UART_TARGET_FRESHNESS_MS
                       and identity[0] == previous[0]
                       and abs(identity[1] - previous[1]) <= config.UART_MAX_X_JUMP_PX
                       and abs(identity[2] - previous[2]) <= config.UART_MAX_Y_JUMP_PX)
        self._initial_confirmation_frames = (
            self._initial_confirmation_frames + 1
            if same_target and bucket == self._initial_confirmation_bucket
            else 1 if identity is not None else 0
        )
        if identity is not None:
            self._initial_confirmation_anchor = identity
        self._initial_confirmation_bucket = bucket
        self._initial_confirmation_observed_ms = now_ms
        if bucket == "clear":
            self._initial_confirmation_saw_clear = True
        elif bucket == "crowded":
            self._initial_confirmation_crowded_total += 1
        if self._initial_confirmation_frames >= config.UART_FIRST_GREEN_DECISION_CONFIRM_FRAMES:
            self._finish_initial_arrangement_decision(
                {"clear": EVENT_SKIP_TO_FINAL_TRACK, "crowded": EVENT_ROI_READY,
                 "horizontal": EVENT_MICRO_FORWARD, "vertical": EVENT_MICRO_SIDE}[bucket],
                candidate, "confirmed_" + bucket, now_ms,
            )
        else:
            self._expire_initial_green_decision(now_ms)

    def _expire_initial_green_decision(self, now_ms):
        """UART timer enforces the deadline even if the camera stops submitting."""
        started = self._initial_confirmation_started_ms
        if (started is None or not self._initial_arrangement_decision_pending
                or self.mode != MODE_ARRANGE_RIGHTMOST
                or now_ms - started < config.UART_FIRST_GREEN_DECISION_TIMEOUT_MS):
            return
        observed = self._initial_confirmation_observed_ms
        persistent_crowding = (
            observed is not None
            and 0 <= now_ms - observed <= config.UART_TARGET_FRESHNESS_MS
            and self._initial_confirmation_bucket == "crowded"
            and self._initial_confirmation_crowded_total >= config.UART_FIRST_GREEN_DECISION_CONFIRM_FRAMES
            and not self._initial_confirmation_saw_clear
        )
        if persistent_crowding:
            self._finish_initial_arrangement_decision(
                EVENT_ROI_READY, None, "timeout_crowded", now_ms,
            )
            return
        seen = self._initial_confirmation_seen_ms
        # Insufficient confirmation is not target loss. Continue observing
        # while raw green detections return; never publish cached coordinates.
        if now_ms - (seen if seen is not None else started) < config.UART_FIRST_GREEN_DECISION_TIMEOUT_MS:
            return
        anchor = self._initial_confirmation_anchor
        self._finish_initial_arrangement_decision(
            EVENT_TRACK_NO_TARGET, None, "raw_missing_timeout", now_ms,
        )
        self._track_target_lock = anchor

    def _finish_initial_arrangement_decision(self, event, selected, reason, now_ms):
        """Commit one handshake only; called with the controller lock held."""
        recovery_anchor = self._initial_confirmation_anchor
        if event in (EVENT_MICRO_FORWARD, EVENT_MICRO_SIDE):
            if self._micro_rounds < getattr(config, "UART_MICRO_ADJUST_MAX_ROUNDS", 3):
                self._start_micro_adjust(event, selected, now_ms)
                return
            event, reason = EVENT_ROI_READY, "micro_limit3_fallback02"
        if event in (EVENT_ROI_READY, EVENT_SKIP_TO_FINAL_TRACK):
            self._reset_micro_adjust()
        started = self._initial_confirmation_started_ms
        elapsed = now_ms - started if started is not None else 0
        rule = "expanded_box_overlap" if self._uses_expanded_overlap_decision(self.target_label) else "merged"
        print("[PRE-02] rule={} target={} confirm={} reason={} elapsed_ms={} action=TX{:02X}".format(
            rule,
            "id{}".format(int(selected.class_id)) if selected is not None else "none",
            self._initial_confirmation_frames, reason, elapsed, event,
        ))
        self._pending_events.clear()
        self._pending_events.append(build_event_packet(event))
        self.mode = (MODE_FINAL_ROI_TRACK if event == EVENT_SKIP_TO_FINAL_TRACK
                     else CMD_TRACK if event == EVENT_TRACK_NO_TARGET
                     else MODE_ARRANGE_RIGHTMOST)
        self._final_target_lock = (
            self._final_target_identity(selected)
            if event == EVENT_SKIP_TO_FINAL_TRACK else None
        )
        if event == EVENT_ROI_READY:
            self._track_target_lock = (self._final_target_identity(selected)
                                       if selected is not None else recovery_anchor)
        self._initial_arrangement_decision_pending = False
        self._reset_initial_arrangement_confirmation()
        self._arrangement_02_ack_pending = event == EVENT_ROI_READY
        self._arrangement_02_request_sent = False
        self._arrangement_02_without_side = event == EVENT_ROI_READY
        self._arrangement_side_target_available = False
        self._last_arrangement_02_event_ms = 0
        self._latest_packet = None
        self._latest_submit_ms = 0
        self._last_target_center_x = None
        self._last_target_center_y = None
        self._last_target_area = None
        self._last_target_accept_ms = 0
        self._arrangement_no_neighbor_frames = 0
        if event == EVENT_TRACK_NO_TARGET:
            self._track_target_lock = recovery_anchor
            self._track_no_target_frames = 0
            self._track_no_target_sent = True
            # Reuse the recovery synchronization gate for ALL micro tasks:
            # fresh center coordinates may resume, but MCU04/44 must confirm
            # the reference change before another14 can be issued.
            self._track_green_recovery_waiting = bool(self._micro_rounds) or self._uses_first_green_center_selection(self.target_label)

    def _uses_relaxed_supply_tracking(self, target_label):
        return (self._post07_started and self._post07_supply_tracking
                and target_label in ("sqareredgreen", "triangualrblack", TARGET_BLACK_OR_GREEN))

    def _select_post22_yolo_target(self, objects, labels, target_label):
        """Post-22 approach using current YOLO, without merged/verification gates."""
        self._selection_diagnostic_policy = "post22_yolo"
        eligible_labels = (
            ("sqareredgreen", "triangualrblack")
            if self._post07_started and target_label in (
                "sqareredgreen", "triangualrblack", TARGET_BLACK_OR_GREEN
            ) else self._eligible_target_labels(target_label)
        )
        candidates = [obj for obj in objects
                      if 0 < int(obj.class_id) < len(labels)
                      and labels[int(obj.class_id)] in eligible_labels]
        return select_nearest_horizontal_target(
            candidates, labels, self.frame_width,
            reference_x=self.frame_width * .5, reference_y=self.frame_height * .5,
        )

    def _start_post22_center_tracking(self, target, now_ms):
        """Queue 04 before the fresh approach packet while holding the lock."""
        self._pending_events.clear()
        self._pending_events.append(build_event_packet(EVENT_SEARCH_CENTER_REACHED))
        self.mode = MODE_WAIT_04_HANDSHAKE
        self._track_target_lock = self._final_target_identity(target)
        self._track_no_target_frames = 0
        self._track_no_target_sent = False
        self._track_green_recovery_waiting = False
        self._track_center_sent = False
        self._roi_entry_sent = True
        self._initial_arrangement_decision_pending = False
        self._arrangement_02_ack_pending = False
        self._arrangement_02_without_side = False
        self._arrangement_side_target_available = False
        self._arrangement_ready_ms = 0
        self._arrangement_no_neighbor_frames = 0
        self._arrangement_object_count = 0
        self._arrangement_target_anchor = None
        self._middle_target_anchor = None
        self._last_arrangement_02_event_ms = 0
        self._last_arrangement_no_object_ms = 0
        _, center_x, center_y, area = self._track_target_lock
        self._last_target_center_x = center_x
        self._last_target_center_y = center_y
        self._last_target_area = area
        self._last_target_accept_ms = now_ms
        reference_x, reference_y = self._reference_point(self.mode)
        self._latest_packet = self._build_claw_coordinate(
            target, int(round(center_x - reference_x)),
            int(round(center_y - reference_y)),
        )
        self._latest_submit_ms = now_ms

    def _select_relaxed_supply_target(self, objects, labels):
        """Post-14 supplies: raw black/green, nearest Y then X to ROI center."""
        self._selection_diagnostic_policy = "post07_supply_yolo"
        candidates = [obj for obj in objects
                      if 0 < int(obj.class_id) < len(labels)
                      and labels[int(obj.class_id)] in ("sqareredgreen", "triangualrblack")]
        reference_x, reference_y = self._search_selection_reference()
        return select_nearest_horizontal_target(
            candidates, labels, self.frame_width,
            reference_x=reference_x, reference_y=reference_y, vertical_first=True,
        )

    def _select_frame_center_target(self, objects, labels, target_label):
        """First green: nearest 2D image center, without merged/jump rules."""
        return select_nearest_horizontal_target(
            objects, labels, self.frame_width, target_label=target_label,
            reference_x=self.frame_width * .5, reference_y=self.frame_height * .5,
        )

    def _select_unlocked_target(
        self, objects, labels, target_label, now_ms, anchor_attribute,
        association_objects=None,
    ):
        if anchor_attribute == "_track_target_lock" and self._uses_first_green_center_selection(target_label):
            self._selection_diagnostic_policy = "first_green_frame_center"
            return self._select_frame_center_target(objects, labels, target_label)
        if anchor_attribute == "_final_target_lock":
            self._selection_diagnostic_policy = "final24_spacing"
            selected = self._select_final_roi_target(
                objects, labels, target_label, now_ms, check_front_block=False,
            )
            if selected is not None:
                return selected
            # State 24 must not stop merely because spacing/verification failed.
            # Fall back only to this task's current raw YOLO class(es), outside
            # safety zones, then let submit establish the usual tracking lock.
            self._selection_diagnostic_policy = "final24_yolo_fallback"
            source = association_objects if association_objects is not None else objects
            target_labels = self._eligible_target_labels(target_label)
            candidates = [
                obj for obj in source
                if 0 < int(obj.class_id) < len(labels)
                and labels[int(obj.class_id)] in target_labels
            ]
            reference_x, reference_y = self._search_selection_reference()
            return select_nearest_horizontal_target(
                candidates, labels, self.frame_width,
                reference_x=reference_x, reference_y=reference_y,
            )
        self._selection_diagnostic_policy = "merged"
        return self._select_final_roi_target(objects, labels, target_label, now_ms)

    def _select_locked_target(
        self, objects, labels, target_label, now_ms, anchor_attribute, log_tag,
        association_objects=None, generation=None,
    ):
        """Keep a matched lock; after confirmed loss, screen targets again.

        Do not rerun spacing, obstruction or maximum-Y ranking while the
        locked class is detected. Use nearest previous position to associate
        multiple same-class boxes, without any position/area change cutoff.
        """
        with self._lock:
            if generation is not None and generation != self._command_generation:
                return None
            anchor = getattr(self, anchor_attribute)
        if anchor is None:
            if self._supply_collect_started_ms is not None:
                return self._select_supply_collection_target(
                    objects, association_objects if association_objects is not None else objects, labels, now_ms)
            if self._uses_relaxed_supply_tracking(target_label):
                return self._select_relaxed_supply_target(
                    association_objects if association_objects is not None else objects, labels
                )
            return self._select_unlocked_target(
                objects, labels, target_label, now_ms, anchor_attribute,
                association_objects,
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
            if self._uses_relaxed_supply_tracking(target_label):
                if obj_id not in (5, 6):
                    continue
            elif obj_id != class_id:
                continue
            matches.append((
                (center_x - previous_x) ** 2 + (center_y - previous_y) ** 2,
                -float(getattr(obj, "score", 0.0)),
                obj,
            ))
        if not matches:
            with self._lock:
                if generation is not None and generation != self._command_generation:
                    return None
                self._record_tracking_missing(now_ms, class_id)
                missing_frames = {
                    "_track_target_lock": self._track_no_target_frames,
                    "_search_target_lock": self._search_no_target_frames,
                    "_final_target_lock": self._final_no_target_frames,
                }[anchor_attribute] + 1
                confirmed = self._tracking_loss_confirmed(self.mode, missing_frames, now_ms)
            if (
                anchor_attribute in ("_track_target_lock", "_search_target_lock")
                or class_id == 6
            ) and not confirmed:
                self._selection_diagnostic_policy = {
                    "_track_target_lock": "track_hold",
                    "_search_target_lock": "search_hold",
                    "_final_target_lock": "final_black_hold",
                }[anchor_attribute]
                # Retain identity only, never old/predicted coordinates.
                return None
            if self._red_priority_tracking and class_id == 4:
                # A confirmed red miss still emits E3/E4 normally. Keep this
                # task's association identity for recovery instead of reranking
                # or revalidating a different red box after five missed frames.
                self._selection_diagnostic_policy = "red_recovery_hold"
                return None
            with self._lock:
                if generation is not None and generation != self._command_generation:
                    return None
                setattr(self, anchor_attribute, None)
                if anchor_attribute == "_final_target_lock":
                    self._last_target_center_x = None
                    self._last_target_center_y = None
                    self._last_target_area = None
                    self._last_target_accept_ms = 0
                # For 04 loss, retain the last observed position so RX44
                # can still seed the existing arrangement middle-X anchor.
            if config.VISION_DIAGNOSTICS_ENABLED and config.VISION_DIAGNOSTICS_VERBOSE:
                print("[{}] lost_lock=id{} action=reselect".format(log_tag, class_id))
            if anchor_attribute == "_search_target_lock":
                # The next fresh frame follows the normal search acquisition
                # branch. This frame must not publish a stale locked box.
                return None
            if self._supply_collect_started_ms is not None:
                return self._select_supply_collection_target(
                    objects, association_objects if association_objects is not None else objects,
                    labels, now_ms)
            if self._uses_relaxed_supply_tracking(target_label):
                return self._select_relaxed_supply_target(
                    association_objects if association_objects is not None else objects, labels
                )
            return self._select_unlocked_target(
                objects, labels, target_label, now_ms, anchor_attribute,
                association_objects,
            )
        return min(matches, key=lambda item: item[:2])[2]

    def _record_tracking_missing(self, now_ms, class_id=None):
        """Caller holds lock. Remember actual lost class through reselection."""
        if self._tracking_missing_since_ms is None:
            self._tracking_missing_since_ms = now_ms
        if class_id is not None:
            self._tracking_missing_class_id = int(class_id)

    def _tracking_loss_confirmed(self, mode, missing_frames, now_ms):
        """Frame minimum plus an elapsed-time gate for black, not stale boxes."""
        minimum = (
            config.UART_FINAL_NO_TARGET_FRAMES if mode == MODE_FINAL_ROI_TRACK
            else config.UART_SEARCH_NO_TARGET_FRAMES if mode == CMD_SEARCH
            else config.UART_TRACK_LOST_CONFIRM_FRAMES
        )
        if missing_frames < minimum:
            return False
        black_loss = (self._tracking_missing_class_id == 6 or (
            self._tracking_missing_class_id is None and self.target_label == "triangualrblack"
        ))
        return (not black_loss or (
            self._tracking_missing_since_ms is not None
            and now_ms - self._tracking_missing_since_ms >= config.UART_BLACK_TARGET_LOST_CONFIRM_MS
        ))

    def _request_red_after_black_loss(self, mode, objects, labels, selected,
                                      now_ms, generation, confirmed_targets):
        """Request a red task, then pause until ordered MCU11/03 confirmation."""
        black_priority_enabled = getattr(config, "UART_BLACK_TO_RED_PRIORITY_ENABLED", False)
        e3_priority_enabled = (
            mode == CMD_SEARCH and getattr(config, "UART_E3_MATERIAL_TO_RED_ENABLED", True)
        )
        if not (black_priority_enabled or e3_priority_enabled):
            return False
        anchors = {
            CMD_SEARCH: ("_search_target_lock", "_search_no_target_frames"),
            CMD_TRACK: ("_track_target_lock", "_track_no_target_frames"),
            MODE_WAIT_04_HANDSHAKE: ("_track_target_lock", "_track_no_target_frames"),
            MODE_WAIT_14_HANDSHAKE: ("_track_target_lock", "_track_no_target_frames"),
            MODE_FINAL_ROI_TRACK: ("_final_target_lock", "_final_no_target_frames"),
        }
        if mode not in anchors:
            return False
        # objects are this frame's raw YOLO detections outside safety zones.
        red_objects = [obj for obj in objects
                       if 0 < int(obj.class_id) < len(labels)
                       and labels[int(obj.class_id)] == "sqarered"
                       and clipped_yolo_rect(obj, self.frame_width, self.frame_height) is not None]
        if not red_objects:
            return False
        anchor_attribute, counter_attribute = anchors[mode]
        with self._lock:
            if (self.mode != mode or generation != self._command_generation
                    or self._supply_collect_started_ms is not None
                    or not self._post07_started):
                return False
            source_label = self.target_label
            e3_switch = e3_priority_enabled and self._search_e3_active and source_label in (
                "sqareredgreen", "triangualrblack", TARGET_BLACK_OR_GREEN,
            )
            if not e3_switch and (
                not black_priority_enabled or source_label not in ("triangualrblack", TARGET_BLACK_OR_GREEN)
            ):
                return False
            if not e3_switch:
                red_objects = [obj for obj in red_objects
                               if self._final_target_identity(obj) in confirmed_targets]
                if not red_objects:
                    return False
            anchor = getattr(self, anchor_attribute)
            # 31 retains the existing unlocked-red priority. 51 must actually
            # have tracked black; an empty task or a green lock cannot request11.
            initial_black_priority = (
                source_label == "triangualrblack" and anchor is None
                and self._last_tracked_class_id is None
            )
            if not (e3_switch or initial_black_priority):
                if (self._last_tracked_class_id != 6
                        or self._tracking_missing_class_id != 6
                        or any(int(obj.class_id) == 6 for obj in objects)
                        or (selected is not None and int(selected.class_id) == 5)
                        or not self._tracking_loss_confirmed(
                            mode, getattr(self, counter_attribute) + 1, now_ms)):
                    return False
            reference_x, reference_y = self._reference_point(mode)
            candidate = min(red_objects, key=lambda obj: (
                (float(obj.x) + float(obj.w) * .5 - reference_x) ** 2
                + (float(obj.y) + float(obj.h) * .5 - reference_y) ** 2,
                -float(getattr(obj, "score", 0.0)),
            ))
            missing_ms = (now_ms - self._tracking_missing_since_ms
                          if self._tracking_missing_since_ms is not None else 0)
            self.mode = MODE_WAIT_RED_SWITCH
            self._red_switch_stage = "wait11"
            self._red_switch_request_sent = False
            # Arm one repeat for E3 switches. Its deadline is set by the first full TX11.
            self._red_switch_retry_at_ms = now_ms if e3_switch else None
            self._red_switch_candidate = self._final_target_identity(candidate)
            self._red_priority_tracking = True
            self.target_label = "sqarered"
            self._post07_started = True
            self._post07_supply_tracking = False
            self._first_green_both_sides_required = False
            self._search_target_lock = None
            self._track_target_lock = None
            self._final_target_lock = None
            self._last_target_center_x = None
            self._last_target_center_y = None
            self._last_target_area = None
            self._last_target_accept_ms = 0
            self._search_no_target_frames = 0
            self._search_no_target_initial_sent = False
            self._search_no_target_acknowledged = False
            self._search_e3_active = False
            self._last_search_no_target_ms = 0
            self._track_no_target_frames = 0
            self._track_no_target_sent = False
            self._track_green_recovery_waiting = False
            self._final_no_target_frames = 0
            self._final_search_waiting_ack = False
            self._last_no_target_ms = 0
            self._roi_entry_sent = False
            self._track_center_sent = False
            self._final_roi_sent = False
            self._last_final_roi_event_ms = 0
            self._last_arrangement_no_object_ms = 0
            self._tracking_missing_since_ms = None
            self._tracking_missing_class_id = None
            self._latest_packet = None
            self._latest_submit_ms = 0
            self._pending_events.clear()
            self._pending_events.append(build_event_packet(EVENT_SWITCH_TO_RED_TARGET))
            self._command_generation += 1
            self._last_frame_seen_ms = now_ms
            self._selection_diagnostic_policy = "red_switch_wait11"
            self._tx_wakeup.set()
            print("[RED-SWITCH] from={} state={:X} reason={} lost={}ms candidate={} action=TX11 wait_RX11_RX03 coordinates=paused".format(
                source_label, mode, "E3_material_search" if e3_switch else "black_priority",
                missing_ms, self._red_switch_candidate,
            ))
            return True

    def _associate_pending_red(self, objects, now_ms, generation):
        """Update identity from a fresh raw red box, without preparing output."""
        with self._lock:
            if (generation != self._command_generation
                    or self.mode != MODE_WAIT_RED_SWITCH):
                return None
            anchor = self._red_switch_candidate
            matches = [obj for obj in objects if int(obj.class_id) == 4]
            candidate = None
            if anchor is not None and matches:
                candidate = min(matches, key=lambda obj: (
                    (float(obj.x) + float(obj.w) * .5 - anchor[1]) ** 2
                    + (float(obj.y) + float(obj.h) * .5 - anchor[2]) ** 2,
                    -float(getattr(obj, "score", 0.0)),
                ))
                self._red_switch_candidate = self._final_target_identity(candidate)
            self._last_frame_seen_ms = now_ms
            self._selection_diagnostic_policy = "red_switch_" + self._red_switch_stage
            # A missing frame preserves only the anchor, never a cached box.
            return candidate

    def _select_final_roi_target(
        self, objects, labels, target_label, now_ms, check_front_block=True,
    ):
        """Rank eligible targets by distance to the ROI-center Y."""
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
            if not check_front_block
            or not self._final_candidate_is_blocked(candidate, movable_objects)
        ]

        if self._is_supply_acquisition(self.mode, target_label):
            if candidates:
                self._selection_diagnostic_policy = (
                    "supply_no_spacing" if check_front_block else "supply_no_spacing24"
                )
            eligible_candidates = unblocked
        else:
            eligible_candidates = [
                candidate for candidate in unblocked
                if self._matches_final_primary_spacing(candidate, movable_objects, self.frame_height * 0.5)
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
        """Return dx for the candidate's Y band after class-specific Y scope."""
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
    def _matches_final_primary_spacing(candidate, movable_objects, frame_center_y=224.0):
        """Check every in-scope neighbor using candidate-Y dx thresholds."""
        center_x = float(candidate.x) + float(candidate.w) * 0.5
        center_y = float(candidate.y) + float(candidate.h) * 0.5
        for other in movable_objects:
            if other is candidate:
                continue
            other_y = float(other.y) + float(other.h) * 0.5
            if not spacing_neighbor_in_y_scope(
                candidate.class_id, center_y, other_y, frame_center_y
            ):
                continue
            other_x = float(other.x) + float(other.w) * 0.5
            red_pair = int(candidate.class_id) == 4 or int(other.class_id) == 4
            min_delta_x, _ = VisionSerialController._final_spacing_limits(
                center_y, red_pair
            )
            if abs(other_x - center_x) <= min_delta_x:
                return False
        return True

    def _select_largest_zone(
        self, objects, labels, required_label, reference_x=None
    ):
        """Use the largest current visible box of the commanded zone color.

        Area is measured exactly as the25%/50% rules, clipped to the image.
        Center distance/confidence only resolve equal areas, never outrank it.
        """
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
            area_ratio = self._visible_box_area_ratio(obj)
            if area_ratio <= 0:
                continue
            center_x = float(obj.x) + float(obj.w) * 0.5
            score = float(getattr(obj, "score", 0.0))
            key = (-area_ratio, abs(center_x - float(reference_x)), -score)
            if selected_key is None or key < selected_key:
                selected = obj
                selected_key = key
        return selected

    def send_event(self, event):
        """Queue one 5-byte event packet."""
        with self._lock:
            if (int(event) == EVENT_STARTZONE_ESCAPE
                    and not getattr(config, "UART_STARTZONE_ESCAPE_ENABLED", False)):
                return
            if (int(event) == EVENT_SWITCH_TO_RED_TARGET
                    and not getattr(config, "UART_BLACK_TO_RED_PRIORITY_ENABLED", False)):
                return
            if (self.mode == MODE_WAIT_TASK_START or self._direct_capture_pending
                    or self._red_switch_stage is not None
                    or self._startzone_escape_phase is not None):
                return
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
            if self.mode == MODE_WAIT_TASK_START:
                return None
            if (self._pending_events
                    and self._red_switch_stage != "wait11"
                    and not getattr(config, "UART_BLACK_TO_RED_PRIORITY_ENABLED", False)):
                self._pending_events = [event for event in self._pending_events
                                        if event != build_event_packet(EVENT_SWITCH_TO_RED_TARGET)]
            if (self._pending_events
                    and not getattr(config, "UART_STARTZONE_ESCAPE_ENABLED", False)):
                self._pending_events = [event for event in self._pending_events
                                        if event != build_event_packet(EVENT_STARTZONE_ESCAPE)]
            if self._startzone_escape_phase is not None:
                self._outgoing_generation = self._command_generation
                if (getattr(config, "UART_STARTZONE_ESCAPE_ENABLED", False)
                        and self._startzone_escape_phase == "pending"
                        and (self._last_event_write_ms is None
                             or now_ms - self._last_event_write_ms >= config.UART_EVENT_MIN_INTERVAL_MS)):
                    return build_event_packet(EVENT_STARTZONE_ESCAPE)
                return None
            self._expire_supply_collection(now_ms)
            self._expire_zone_load_resume(now_ms)
            self._clear_disabled_zone_obstacle()
            if self._supply_search_settling(now_ms):
                return None
            if self._supply_collect_finished:
                request = build_event_packet(EVENT_COLLECT_MORE_SUPPLIES)
                self._pending_events = [event for event in self._pending_events if event != request]
            self._outgoing_generation = self._command_generation
            if self._direct_capture_pending:
                if (self._last_event_write_ms is None
                        or now_ms - self._last_event_write_ms >= config.UART_EVENT_MIN_INTERVAL_MS):
                    return build_event_packet(EVENT_SKIP_TO_FINAL_TRACK)
                return None
            if (self.mode == MODE_ARRANGE_RIGHTMOST and self._initial_arrangement_decision_pending
                    and self._micro_rounds and self._micro_decision_ms is not None
                    and now_ms - self._micro_decision_ms >= getattr(config, "UART_MICRO_ADJUST_ACK_TIMEOUT_MS", 5000)):
                self._finish_initial_arrangement_decision(
                    EVENT_TRACK_NO_TARGET, None, "micro_recheck_camera_timeout", now_ms)
            if self.mode in MICRO_MODES:
                pending_stale = (not self._micro_request_sent and self._micro_decision_ms is not None
                                 and now_ms - self._micro_decision_ms > config.UART_TARGET_FRESHNESS_MS)
                expired = (self._micro_started_ms is not None and now_ms - self._micro_started_ms
                           >= (getattr(config, "UART_MICRO42_ACK_TIMEOUT_MS", 11000)
                               if self.mode == MODE_MICRO_SIDE else
                               getattr(config, "UART_MICRO_ADJUST_ACK_TIMEOUT_MS", 5000)))
                lost = (self._micro_request_sent and self._latest_packet is None
                        and self._tracking_loss_confirmed(self.mode, self._track_no_target_frames, now_ms))
                if self._micro_loss_pending or pending_stale or expired or lost:
                    self._micro_loss_pending = True
                    self._pending_events.clear()
                    self._latest_packet = None
                    self._latest_submit_ms = 0
                    return build_event_packet(EVENT_TRACK_NO_TARGET)
            if self.mode == MODE_SEARCH_ZONE and self._zone_realign_phase == "wait_ack":
                repeat_ms = getattr(config, "UART_ZONE_REALIGN_REPEAT_MS", 500)
                if (now_ms - self._last_zone_realign_write_ms >= repeat_ms
                        and (self._last_event_write_ms is None
                             or now_ms - self._last_event_write_ms >= config.UART_EVENT_MIN_INTERVAL_MS)):
                    return build_event_packet(EVENT_ZONE_REALIGN)
                return None
            if self.mode == MODE_SEARCH_ZONE and self._zone_realign_phase == "pending":
                if now_ms - self._last_frame_seen_ms > config.UART_TARGET_FRESHNESS_MS:
                    self._zone_realign_phase = None
                    self._zone_close_confirm_frames = 0
                    self._zone_close_confirm_seen_ms = None
                    self._pending_events = [event for event in self._pending_events
                                            if event != build_event_packet(EVENT_ZONE_REALIGN)]
            if self._red_switch_stage is not None:
                # Only the 11 request may be sent during switching.
                # In particular, no coordinates/34/EE/E3/E4 cross this pause.
                request = build_event_packet(EVENT_SWITCH_TO_RED_TARGET)
                retry_due = (self._red_switch_request_sent
                             and self._red_switch_retry_at_ms is not None
                             and now_ms >= self._red_switch_retry_at_ms)
                if self._red_switch_stage == "wait11" and (request in self._pending_events or retry_due):
                    if (self._last_event_write_ms is not None
                            and now_ms - self._last_event_write_ms < config.UART_EVENT_MIN_INTERVAL_MS):
                        return None
                    self._pending_events.clear()
                    return request
                self._pending_events.clear()
                return None
            self._expire_initial_green_decision(now_ms)
            capture_event = self._capture_event_packet
            capture_ready = now_ms >= self._final_capture_ready_ms
            if self.mode == MODE_FINAL_ROI_TRACK and self._final_roi_sent:
                last_capture_seen_ms = (self._supply_roi_handoff_seen_ms if self._uses_supply_collection()
                                        else self._last_frame_seen_ms)
                if now_ms - last_capture_seen_ms > config.UART_TARGET_FRESHNESS_MS:
                    # A queued34 can expire while the claw is settling.
                    self._final_roi_sent = False
                    self._supply_roi_handoff = None
                    self._last_final_roi_event_ms = 0
                    self._pending_events = [event for event in self._pending_events if event != capture_event]
            if (
                self.mode == MODE_SEARCH_ZONE and self._zone_close_active
                and now_ms - self._latest_submit_ms > config.UART_TARGET_FRESHNESS_MS
            ):
                # Do not repeat 26 from stale camera observations, including
                # a first event delayed behind another queued handshake.
                self._zone_close_active = False
                pending_26 = build_event_packet(EVENT_ZONE_CLOSE)
                self._pending_events = [
                    event for event in self._pending_events
                    if event != pending_26
                ]
            if self._pending_events:
                if not capture_ready and self._pending_events[0] == capture_event:
                    return None
                if (
                    self._last_event_write_ms is not None
                    and now_ms - self._last_event_write_ms
                    < config.UART_EVENT_MIN_INTERVAL_MS
                ):
                    # Do not transmit coordinates ahead of a pending handshake.
                    return None
                packet = self._pending_events.pop(0)
                if packet == capture_event:
                    self._last_final_roi_event_ms = now_ms
                elif packet == build_event_packet(EVENT_ZONE_OBSTACLE):
                    self._last_zone_obstacle_event_ms = now_ms
                elif packet == build_event_packet(EVENT_ZONE_CLOSE):
                    self._last_zone_close_event_ms = now_ms
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
                self.mode == MODE_SEARCH_ZONE and self._zone_close_active
                and self._latest_packet is not None
                and (
                    self._last_zone_close_event_ms is None
                    or now_ms - self._last_zone_close_event_ms
                    >= config.UART_ZONE_CLOSE_EVENT_REPEAT_MS
                )
            ):
                if (
                    self._last_event_write_ms is not None
                    and now_ms - self._last_event_write_ms
                    < config.UART_EVENT_MIN_INTERVAL_MS
                ):
                    return None
                self._last_zone_close_event_ms = now_ms
                return build_event_packet(EVENT_ZONE_CLOSE)

            if (
                self.mode == MODE_FINAL_ROI_TRACK
                and self._final_roi_sent
                and capture_ready
                and now_ms - self._last_final_roi_event_ms
                >= config.UART_FINAL_ROI_EVENT_REPEAT_MS
            ):
                self._last_final_roi_event_ms = now_ms
                return capture_event

            if (
                self._latest_packet is not None
                and now_ms - self._latest_submit_ms
                <= config.UART_TARGET_FRESHNESS_MS
            ):
                # A repeated vision frame must not refresh its age on the MCU.
                if (config.UART_SEND_RATE_HZ <= 0
                        or self._latest_packet[2] & OBJECT_GEOMETRY_PRESENT):
                    if self._last_coordinate_sequence_sent == self._latest_submit_sequence:
                        return None
                    self._last_coordinate_sequence_sent = self._latest_submit_sequence
                self._coordinate_in_flight = (
                    self._latest_packet, self._latest_submit_sequence, self._latest_submit_ms, self.mode
                )
                return self._latest_packet

            if (
                self.mode == CMD_SEARCH
                and not (self._new_target_confirmation_pending
                         and now_ms - self._last_frame_seen_ms <= config.UART_TARGET_FRESHNESS_MS)
                and self._latest_submit_ms > 0
                and self._latest_packet is None
                and self._search_no_target_frames
                >= config.UART_SEARCH_NO_TARGET_FRAMES
                and self._tracking_loss_confirmed(self.mode, self._search_no_target_frames, now_ms)
            ):
                if not self._search_no_target_initial_sent:
                    self._search_no_target_initial_sent = True
                    self._search_no_target_acknowledged = False
                    self._last_search_no_target_ms = now_ms
                    return build_event_packet(EVENT_SEARCH_NO_TARGET)
                repeat_ms = (
                    config.UART_NO_TARGET_REPEAT_MS
                    if self._search_no_target_acknowledged
                    else config.UART_SEARCH_NO_TARGET_ACK_TIMEOUT_MS
                )
                if now_ms - self._last_search_no_target_ms >= repeat_ms:
                    self._last_search_no_target_ms = now_ms
                    return build_event_packet(EVENT_SEARCH_NO_TARGET)

            if (
                self.mode == CMD_TRACK
                and not (self._new_target_confirmation_pending
                         and now_ms - self._last_frame_seen_ms <= config.UART_TARGET_FRESHNESS_MS)
                and self._latest_submit_ms > 0
                and self._latest_packet is None
                and self._track_no_target_frames >= config.UART_TRACK_LOST_CONFIRM_FRAMES
                and self._tracking_loss_confirmed(self.mode, self._track_no_target_frames, now_ms)
                and not self._track_no_target_sent
            ):
                # One E4 per confirmed loss, not a periodic retry. Fresh
                # reacquisition re-arms it except during first-green RX44 wait.
                self._track_no_target_sent = True
                self._last_coordinate_log_ms = None
                if (
                    self._uses_first_green_center_selection(self.target_label)
                ):
                    # Coordinates may reappear while MCU finishes its first
                    # green push; only explicit MCU44 (decision) or MCU04
                    # (resume center tracking) may release the recovery wait.
                    self._track_green_recovery_waiting = True
                return build_event_packet(EVENT_TRACK_NO_TARGET)

            if (
                self.mode == MODE_POST_ARRANGE_CHECK
                and not self._uses_first_green_center_selection(self.target_label)
                and not (self._new_target_confirmation_pending
                         and now_ms - self._last_frame_seen_ms <= config.UART_TARGET_FRESHNESS_MS)
                and self._latest_submit_ms > 0
                and self._latest_packet is None
                and (
                    self._last_arrangement_no_object_ms == 0
                    or now_ms - self._last_arrangement_no_object_ms
                    >= config.UART_NO_TARGET_REPEAT_MS
                )
            ):
                # Record the E2 recovery round. Its next matching raw YOLO
                # target emits 04, then fresh image-center coordinates.
                self._last_arrangement_no_object_ms = now_ms
                return build_event_packet(EVENT_ARRANGEMENT_NO_OBJECT)

            if (
                self.mode == MODE_FINAL_ROI_TRACK
                and not self._final_roi_sent
                and not (self._new_target_confirmation_pending
                         and now_ms - self._last_frame_seen_ms <= config.UART_TARGET_FRESHNESS_MS)
                and self._latest_submit_ms > 0
                and self._latest_packet is None
                and not self._final_search_waiting_ack
                and self._final_no_target_frames
                >= config.UART_FINAL_NO_TARGET_FRAMES
                and self._tracking_loss_confirmed(self.mode, self._final_no_target_frames, now_ms)
            ):
                self._last_no_target_ms = now_ms
                self._final_search_waiting_ack = True
                return build_event_packet(EVENT_NO_TARGET)
        return None

    def _write_packet(self, packet, generation=None):
        # Serial writes and a switching request share the lock: an already
        # prepared old packet either finishes BEFORE11, or is discarded. It
        # cannot slip out between the pause and its request/confirmation.
        with self._lock:
            if self.mode == MODE_WAIT_TASK_START:
                return
            red_switch_request = packet == build_event_packet(EVENT_SWITCH_TO_RED_TARGET)
            if (red_switch_request and (self._red_switch_stage == "wait03"
                    or (self._red_switch_stage != "wait11"
                        and not getattr(config, "UART_BLACK_TO_RED_PRIORITY_ENABLED", False)))):
                # Block queued/prepared old TX11 at the wire as well. MCU RX11
                # remains an ordinary target-selection command, untouched.
                return
            if (red_switch_request and self._red_switch_stage == "wait11"
                    and self._red_switch_request_sent
                    and (self._red_switch_retry_at_ms is None
                         or _now_ms() < self._red_switch_retry_at_ms)):
                return
            escape_request = packet == build_event_packet(EVENT_STARTZONE_ESCAPE)
            if escape_request and not getattr(config, "UART_STARTZONE_ESCAPE_ENABLED", False):
                return
            if self._startzone_escape_phase is not None:
                if not escape_request or self._startzone_escape_phase != "pending":
                    return
            elif escape_request:
                return
            self._expire_supply_collection(_now_ms())
            self._expire_zone_load_resume(_now_ms())
            self._clear_disabled_zone_obstacle()
            if self._supply_search_ready_ms is not None and self._supply_search_settling(_now_ms()):
                return
            if generation is not None and generation != self._command_generation:
                return
            direct_capture_request = self._direct_capture_pending
            final_track_request = packet == build_event_packet(EVENT_SKIP_TO_FINAL_TRACK)
            if direct_capture_request and not final_track_request:
                return
            if self.mode == MODE_ARRANGE_LEFTMOST and len(packet) == 11:
                return
            arrange02_request = (packet == build_event_packet(EVENT_ROI_READY)
                                 and self.mode == MODE_ARRANGE_RIGHTMOST)
            if arrange02_request and (self._initial_arrangement_decision_pending
                                      or self._arrangement_02_request_sent):
                return
            if self.mode == MODE_ARRANGE_RIGHTMOST and len(packet) == 11:
                if (self._initial_arrangement_decision_pending
                        or not self._arrangement_02_request_sent
                        or _now_ms() < self._arrangement_ready_ms
                        or packet != self._latest_packet
                        or _now_ms() - self._latest_submit_ms > config.UART_TARGET_FRESHNESS_MS):
                    return
            if packet == build_event_packet(EVENT_TARGET_READY) and self.mode != MODE_NEAR_VIEW_CHECK:
                # A prepared camera-position request from an older05 must
                # not reframe the MCU after a new task/stage has started.
                return
            if (self.mode == MODE_FINAL_ROI_TRACK and self._final_roi_sent
                    and len(packet) == 11):
                # All cargo tasks: ROI entry is34-only, not steering toward
                # already-framed cargo. Also fence a prepared outside packet.
                return
            if (len(packet) == 11 and (packet[2] & OBJECT_GEOMETRY_ID_MASK) in (3, 4, 5, 6)
                    and self.mode in (CMD_SEARCH, CMD_TRACK, MODE_WAIT_04_HANDSHAKE,
                                      MODE_WAIT_14_HANDSHAKE, MODE_FINAL_ROI_TRACK,
                                      MODE_ARRANGE_RIGHTMOST) + MICRO_MODES
                    and self._latest_submit_sequence > 0
                    and (packet != self._latest_packet
                         or _now_ms() - self._latest_submit_ms > config.UART_TARGET_FRESHNESS_MS)):
                # A newer exclusion/miss invalidates previously prepared
                # steering, even when no command changed the generation.
                return
            if (len(packet) == 5 and packet[2] in (
                        EVENT_ARRANGEMENT_NO_OBJECT, EVENT_SEARCH_NO_TARGET,
                        EVENT_TRACK_NO_TARGET, EVENT_NO_TARGET)
                    and self._new_target_confirmation_pending
                    and _now_ms() - self._last_frame_seen_ms <= config.UART_TARGET_FRESHNESS_MS):
                return
            capture_event = packet == self._capture_event_packet
            tall_casualty_event = packet == build_event_packet(EVENT_TALL_CASUALTY_CAPTURED)
            if tall_casualty_event and (self.mode != MODE_FINAL_ROI_TRACK
                                       or self._final_capture_followup != "pending54"):
                return
            if capture_event:
                now_ms = _now_ms()
                if now_ms < self._final_capture_ready_ms:
                    return
                if (self._latest_submit_sequence > 0
                        and (self.mode != MODE_FINAL_ROI_TRACK or not self._final_roi_sent
                             or now_ms - self._last_frame_seen_ms > config.UART_TARGET_FRESHNESS_MS)):
                    # Entry34 is current visual evidence, not a latched cargo fact.
                    return
            micro_request = len(packet) == 5 and packet[2] in MICRO_MODES
            if micro_request and (self.mode != packet[2] or self._micro_request_sent):
                return
            if micro_request and (self._micro_decision_ms is None
                    or _now_ms() - self._micro_decision_ms > config.UART_TARGET_FRESHNESS_MS):
                self._micro_loss_pending = True
                return
            if self.mode in MICRO_MODES:
                if len(packet) == 11:
                    if (not self._micro_request_sent or self._micro_loss_pending
                            or packet != self._latest_packet
                            or _now_ms() - self._latest_submit_ms > config.UART_TARGET_FRESHNESS_MS):
                        return
                elif packet == build_event_packet(EVENT_TRACK_NO_TARGET):
                    if not self._micro_loss_pending:
                        return
                elif not micro_request:
                    # No34,14,24/05 or other completion can interrupt32/42.
                    return
            realign_request = build_event_packet(EVENT_ZONE_REALIGN)
            if packet == realign_request:
                if (self.mode != MODE_SEARCH_ZONE
                        or self._zone_realign_phase not in ("pending", "wait_ack")):
                    return
                if (self._zone_realign_phase == "pending"
                        and _now_ms() - self._last_frame_seen_ms > config.UART_TARGET_FRESHNESS_MS):
                    return
            if self.mode == MODE_SEARCH_ZONE:
                if self._zone_realign_phase == "wait_ack" and packet != realign_request:
                    return
                if (self._zone_realign_phase == "pending"
                        and packet not in (realign_request, build_event_packet(EVENT_ZONE_FOUND))):
                    return
            if (self._supply_collect_finished
                    and packet == build_event_packet(EVENT_COLLECT_MORE_SUPPLIES)):
                return
            if packet == build_event_packet(EVENT_SEARCH_NO_TARGET) and (
                    self.mode != CMD_SEARCH
                    or self._latest_packet is not None
                    or not self._search_no_target_initial_sent
                    or self._search_no_target_frames < config.UART_SEARCH_NO_TARGET_FRAMES):
                # A fresh target may recover between preparing the E3 retry
                # and writing it. Do not restart MCU search from that old miss.
                return
            if packet == build_event_packet(EVENT_ZONE_OBSTACLE) and (
                    not self._zone_obstacle_enabled()
                    or self.mode != MODE_SEARCH_ZONE
                    or not self._zone_obstacle_waiting
                    or (not self._zone_obstacle_request_sent
                        and not self._zone_obstacle_area_allowed)):
                # An old prepared36 must not slip out after a newer >25%
                # image cancels it. Retries of an already sent round retain
                # their original handshake until MCU36 completes the action.
                return
            if packet == build_event_packet(EVENT_ZONE_NO_TARGET) and (
                    self.mode != MODE_SEARCH_ZONE
                    or self._zone_no_target_phase != "pending"):
                return
            if packet == build_event_packet(EVENT_ZONE_CLOSE) and (
                    self.mode != MODE_SEARCH_ZONE or not self._zone_close_active
                    or _now_ms() - self._last_frame_seen_ms > config.UART_TARGET_FRESHNESS_MS):
                # A prepared26 must not outlive a subsequent46 decision,
                # cancellation/smaller image, or a stopped camera.
                return
            if self.mode == MODE_SEARCH_ZONE and len(packet) == 11 and packet[2] in (1, 2):
                # A newer small/missing zone cancels any in-flight old steering
                # packet. Do not confuse05 cargo metadata (IDs4/5) with this.
                if (self._zone_area_ratio <= getattr(config, "UART_ZONE_MIN_AREA_RATIO", 0.02)
                        or packet != self._latest_packet
                        or _now_ms() - self._latest_submit_ms > config.UART_TARGET_FRESHNESS_MS):
                    return
            if packet == build_event_packet(EVENT_ZONE_SEARCH_STARTED) and (
                    self._zone_load_recheck != "wait06tx"
                    or self._zone_load_recheck_source != "before06"):
                # Protect the wire as well as the decision paths: a queued or
                # prepared old06 cannot bypass the new final inventory.
                return
            casualty_adjust_request = packet == build_event_packet(EVENT_WIDE_CASUALTY_LOAD_ADJUST)
            if casualty_adjust_request != (self._zone_load_recheck == "wait45tx"):
                return
            if self._zone_load_recheck in ("wait05tx", "count05"):
                if packet != build_event_packet(EVENT_TARGET_READY):
                    return
            elif self._zone_load_recheck == "wait06tx":
                cargo_report = (len(packet) == 11 and packet[2] in (4, 5)
                                and packet == build_coordinate_packet(packet[2], 0, 0))
                if not cargo_report and packet != build_event_packet(EVENT_ZONE_SEARCH_STARTED):
                    return
            elif self._zone_load_recheck == "wait24tx":
                if packet != build_event_packet(EVENT_SKIP_TO_FINAL_TRACK):
                    return
            elif self._zone_load_recheck in ("wait45ack", "wide_settle"):
                return
            if (self._red_switch_stage is not None
                    and packet != build_event_packet(EVENT_SWITCH_TO_RED_TARGET)):
                return
            if self.mode == MODE_FINAL_ROI_TRACK and self._uses_supply_collection():
                if len(packet) == 11 and self._final_roi_sent:
                    # Discard an outside coordinate prepared before entry.
                    return
                if capture_event and (
                        not self._final_roi_sent
                        or _now_ms() - self._supply_roi_handoff_seen_ms > config.UART_TARGET_FRESHNESS_MS):
                    return
            if (self.mode == MODE_FINAL_ROI_TRACK
                    and self.target_label == config.UART_ZONE_CASUALTY_LABEL):
                if self._final_roi_sent and (
                        len(packet) == 11 or packet == build_event_packet(EVENT_NO_TARGET)):
                    # Cancel an outside coordinate/loss event prepared before
                    # the fresh red-cargo34 handoff.
                    return
                if capture_event and (
                        not self._final_roi_sent
                        or _now_ms() - self._last_frame_seen_ms > config.UART_TARGET_FRESHNESS_MS):
                    return
            sent = self._serial.write(packet)
            if casualty_adjust_request and sent == len(packet):
                self._zone_load_recheck = "wait45ack"
            if final_track_request and sent == len(packet):
                self._final_capture_ready_ms = _now_ms() + getattr(config, "UART_FINAL_CLAW_SETTLE_MS", 300)
                self._final_capture_followup = None
            if (capture_event and sent == len(packet) and self.mode == MODE_FINAL_ROI_TRACK
                    and self._final_capture_followup is None):
                # The first successful34 commits this capture's shape decision.
                self._final_capture_followup = "done"
                if self._final_capture_tall_casualty:
                    self._final_capture_followup = "pending54"
                    self._pending_events.insert(0, build_event_packet(EVENT_TALL_CASUALTY_CAPTURED))
            if tall_casualty_event:
                if sent == len(packet):
                    self._final_capture_followup = "done"
                elif packet not in self._pending_events:
                    self._pending_events.insert(0, packet)
            if red_switch_request and self._red_switch_stage == "wait11":
                if sent == len(packet):
                    if self._red_switch_request_sent:
                        self._red_switch_retry_at_ms = None
                    elif self._red_switch_retry_at_ms is not None:
                        self._red_switch_retry_at_ms = _now_ms() + getattr(
                            config, "UART_E3_RED_SWITCH_RETRY_MS", 100)
                    self._red_switch_request_sent = True
                elif not self._red_switch_request_sent and packet not in self._pending_events:
                    self._pending_events.append(packet)
            if direct_capture_request and sent == len(packet):
                self._direct_capture_pending = False
                # The next usable image begins after the claw-open request.
                self._command_generation += 1
            if arrange02_request:
                if sent == len(packet):
                    self._arrangement_02_request_sent = True
                    self._pending_events = [event for event in self._pending_events if event != packet]
                    self._latest_packet = None
                    self._latest_submit_ms = 0
                    if self._arrangement_ready_ms > 0:
                        self._arrangement_ready_ms = _now_ms() + config.UART_WIDE_VIEW_SETTLE_MS
                    self._command_generation += 1
                elif packet not in self._pending_events:
                    self._pending_events.insert(0, packet)
            if escape_request and sent == len(packet):
                self._startzone_escape_phase = "wait_ack"
                self._pending_events.clear()
                self._latest_packet = None
                self._latest_submit_ms = 0
                self._command_generation += 1
            if micro_request:
                if sent == len(packet):
                    self._micro_request_sent = True
                    self._micro_started_ms = _now_ms()
                    self._micro_rounds += 1
                    if self.mode == MODE_MICRO_SIDE:
                        self._micro42_rounds += 1
                    self._pending_events.clear()
                    self._latest_packet = None
                    self._latest_submit_ms = 0
                    self._command_generation += 1
                elif packet not in self._pending_events:
                    self._pending_events.insert(0, packet)
            if (self.mode in MICRO_MODES and packet == build_event_packet(EVENT_TRACK_NO_TARGET)
                    and sent == len(packet)):
                # E4 hands control to the EXISTING04 recovery protocol. All
                # subsequent fresh coordinates use IMAGE center, never the
                # old32/42 ROI-center reference.
                previous_mode = self.mode
                self.mode = CMD_TRACK
                self._micro_request_sent = False
                self._micro_started_ms = None
                self._micro_decision_ms = None
                self._micro_loss_pending = False
                self._track_no_target_sent = True
                self._track_green_recovery_waiting = True
                self._latest_packet = None
                self._latest_submit_ms = 0
                self._command_generation += 1
                print("[MICRO] state={:02X} action=E4_resume04_image_center rounds={} MCU_cancel_old_micro".format(
                    previous_mode, self._micro_rounds))
            if packet == realign_request:
                if sent == len(packet):
                    first_write = self._zone_realign_phase == "pending"
                    self._zone_realign_phase = "wait_ack"
                    self._last_zone_realign_write_ms = _now_ms()
                    self._pending_events.clear()
                    self._latest_packet = None
                    self._latest_submit_ms = 0
                    self._zone_close_active = False
                    if first_write:
                        # Drop camera inferences/prepared writes that started
                        # before the MCU began its physical maneuver.
                        self._command_generation += 1
                elif self._zone_realign_phase == "pending" and packet not in self._pending_events:
                    self._pending_events.insert(0, packet)
            if (sent == len(packet) and self.mode == MODE_SEARCH_ZONE
                    and packet == build_event_packet(EVENT_ZONE_CLOSE)):
                # Once pushing has started, later rod flicker must not start46.
                self._zone_realign_phase = "done"
            if packet == build_event_packet(EVENT_ZONE_NO_TARGET):
                if sent == len(packet):
                    self._zone_no_target_phase = "sent"
                elif packet not in self._pending_events:
                    # A failed UART write is not a delivered recovery request.
                    self._pending_events.insert(0, packet)
            if sent == len(packet) and packet == build_event_packet(EVENT_SEARCH_NO_TARGET):
                # Start/restart the ACK timeout at the successful UART write,
                # not at earlier packet preparation.
                self._last_search_no_target_ms = _now_ms()
                self._search_e3_active = True
            if sent == len(packet) and packet == build_event_packet(EVENT_ZONE_OBSTACLE):
                self._zone_obstacle_request_sent = True
            if (sent == len(packet) and packet == build_event_packet(EVENT_TARGET_READY)
                    and self.mode == MODE_NEAR_VIEW_CHECK
                    and self._near_view_started_ms is None):
                # ALL05 entries start settling only after05 really left UART.
                # Also fence images captured before this camera request. A
                # duplicate successful05 must not restart an active window.
                if self._zone_load_recheck == "wait05tx":
                    self._zone_load_recheck = "count05"
                self._near_view_started_ms = _now_ms()
                self._near_view_stable_bucket = None
                self._near_view_stable_frames = 0
                self._command_generation += 1
            wide_view_result = (
                (self._zone_load_recheck == "wait06tx"
                 and packet == build_event_packet(EVENT_ZONE_SEARCH_STARTED))
                or (self._zone_load_recheck == "wait24tx"
                    and packet == build_event_packet(EVENT_SKIP_TO_FINAL_TRACK))
            )
            if sent == len(packet) and wide_view_result:
                self._zone_load_recheck = "wide_settle"
                settle_ms = (getattr(config, "UART_SUPPLY_SEARCH_SETTLE_MS", 250)
                             if self._zone_load_recheck_source == "supply_empty"
                             else config.UART_WIDE_VIEW_SETTLE_MS)
                self._zone_load_resume_ready_ms = _now_ms() + settle_ms
                self._command_generation += 1
            if (sent != len(packet) and (
                    self._zone_load_recheck in ("wait05tx", "wait45tx", "wait06tx", "wait24tx")
                    or (final_track_request and self.mode == MODE_FINAL_ROI_TRACK and not direct_capture_request)
                    or (packet == build_event_packet(EVENT_TARGET_READY)
                        and self.mode == MODE_NEAR_VIEW_CHECK
                        and self._near_view_started_ms is None))
                    and packet not in self._pending_events):
                # A failed reply/result write must not strand this handshake.
                self._pending_events.insert(0, packet)
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
            if config.VISION_DIAGNOSTICS_ENABLED and config.VISION_DIAGNOSTICS_VERBOSE:
                with self._lock:
                    state, frame = self.mode, self._latest_submit_sequence
                print("[TX-EVENT] t={} f={} state={:X} cmd={:02X} bytes={}/{}".format(
                    _now_ms(), frame, state, event, sent, len(packet)
                ))
        elif (len(packet) == 11 and packet[2] in (4, 5)
              and packet == build_coordinate_packet(packet[2], 0, 0)
              and self.mode == MODE_SEARCH_ZONE):
            # Leave the MCU time to latch the load type before receiving06.
            with self._lock:
                self._last_event_write_ms = _now_ms()
            print("Vision UART TX cargo ID{}: NEAR05_LOAD [{}]".format(
                packet[2], " ".join("{:02X}".format(value) for value in packet)))
        elif len(packet) == 11 and packet[:2] == CAM_HEADER and packet[-2:] == CAM_TAIL:
            now_ms = _now_ms()
            with self._lock:
                self._tx_coordinate_count += 1
                self._last_coordinate_write_ms = now_ms
                metadata = self._coordinate_in_flight
                count, state = self._tx_coordinate_count, self.mode
            if config.VISION_DIAGNOSTICS_ENABLED and config.VISION_COORDINATE_LOG_ENABLED and (
                self._last_coordinate_log_ms is None
                or now_ms - self._last_coordinate_log_ms >= config.VISION_COORDINATE_LOG_INTERVAL_MS
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
                        self._write_packet(packet, generation=self._outgoing_generation)
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
