"""Application settings shared by the MaixCAM YOLO modules."""

MODEL_PATH = "/root/sd/gongxun27/model_10439.mud"

# ISP image quality settings.
AUTO_AWB = True
AWB_GAIN = [0.134, 0.0625, 0.0625, 0.1139]  # R, GR, GB, B
CONTRAST = 70
CAMERA_STABILIZE_FRAMES = 30

# Detection settings. These thresholds are used by YOLO but are not drawn.
DETECT_DUAL_BUFFER = True
DETECT_CONFIDENCE = 0.5
DETECT_BLACK_CONFIDENCE = 0.7
DETECT_SAFETY_ZONE_CONFIDENCE = 0.4
DETECT_IOU = 0.45

# Secondary verification for the four useful YOLO classes. RGB888 images use
# LAB thresholds in the order [L_min, L_max, A_min, A_max, B_min, B_max].
VERIFY_ENABLED = True
VERIFY_COLOR_ENABLED = False  # Keep LAB pixel/blob checks disabled.
VERIFY_WINDOW_FRAMES = 5
VERIFY_REQUIRED_HITS = 3
VERIFY_REQUIRED_HITS_WITHOUT_COLOR = 4
VERIFY_COLOR_MEMORY_FRAMES = 8
VERIFY_MAX_MISSES = 8
VERIFY_HIGH_CONFIDENCE_BYPASS = 0.72
VERIFY_SKIP_REDUNDANT_COLOR_ON_HIGH_CONFIDENCE = True
VERIFY_ROI_INSET_RATIO = 0.00
VERIFY_MIN_BOX_SIZE = 8
VERIFY_MAX_CANDIDATES_PER_CLASS = 3
VERIFY_X_STRIDE = 2
VERIFY_Y_STRIDE = 2

# Actual target colors: crimson red, green, and light cyan-blue.
VERIFY_LAB_RED = [5, 100, 20, 127, -30, 127]
VERIFY_LAB_GREEN = [5, 100, -128, -8, -15, 127]
VERIFY_LAB_BLUE = [15, 100, -128, 30, -128, 0]
VERIFY_LAB_BLACK = [0, 42, -128, 127, -128, 127]

VERIFY_SINGLE_COLOR_MIN_RATIO = 0.04
VERIFY_LARGEST_BLOB_MIN_RATIO = 0.015
VERIFY_BLACK_MIN_RATIO = 0.04
VERIFY_BLACK_MAX_RATIO = 0.92
VERIFY_SQUARE_ASPECT_MIN = 0.20
VERIFY_SQUARE_ASPECT_MAX = 4.50
VERIFY_TRIANGLE_DENSITY_MIN = 0.08
VERIFY_TRIANGLE_DENSITY_MAX = 0.96

# MCU command input and single-target result output on MaixCAM-Pro UART0.
# Frames use fixed two-byte headers/tails and no checksum; see serial_controller.
UART_COORDINATE_ENABLED = True
UART_PORT = "/dev/ttyS0"
UART_BAUDRATE = 115200
# 0: send each newly submitted coordinate result immediately, without a
# fixed Hz cap. Positive values retain periodic coordinate publishing.
UART_SEND_RATE_HZ = 0
UART_READ_POLL_MS = 1
# Compact diagnostics keep wireless console traffic low. Verbose object and
# color/screening evidence can be re-enabled independently of screen DEBUG.
VISION_DIAGNOSTICS_ENABLED = True
VISION_DIAGNOSTICS_VERBOSE = False
VISION_DIAGNOSTICS_INTERVAL_MS = 5000
VISION_DIAGNOSTICS_MAX_OBJECTS = 24
# Coordinate UART writes continue at full speed; only console output is gated.
VISION_COORDINATE_LOG_ENABLED = False
VISION_COORDINATE_LOG_INTERVAL_MS = 3000
# Preserve spacing between queued command events so MCU's single event slot
# can consume each handshake; coordinates do not use this limit.
UART_EVENT_MIN_INTERVAL_MS = 50
UART_TARGET_FRESHNESS_MS = 200
# Temporary experiment: keep startzone previews but do not request AE turns
# or pause tracking for them. Restart the vision program after changing this.
UART_STARTZONE_ESCAPE_ENABLED = False
UART_STARTZONE_ESCAPE_MIN_AREA_RATIO = 0.04
# ROI boundary changes only; retain the existing rectangular center reference.
UART_TRACKING_ROI_SHAPE = "triangle"
UART_TRACKING_ROI_APEX_OFFSET_X = 25
UART_TRACKING_ROI_APEX_OFFSET_Y = 25
UART_TRACKING_ROI_BASE_INSET_X = 150  # Left base inset. The right point is mirrored about the apex.
UART_NO_TARGET_REPEAT_MS = 500
# State 03 must miss its selected target for this many consecutive frames
# before reporting E3, preventing one-frame detection flicker from triggering it.
UART_SEARCH_NO_TARGET_FRAMES = 5
# After E3, wait8s for MCU13 before retrying while still missing the target.
# Once13 arrives, retain the existing500ms missing-target E3 interval.
UART_SEARCH_NO_TARGET_ACK_TIMEOUT_MS = 8000
UART_FINAL_NO_TARGET_FRAMES = 10
# Allow the claw to reach its capture angle after a complete outgoing24.
UART_FINAL_CLAW_SETTLE_MS = 300
# Casualty-only state 24. Pixel calibration at 576x448; scaled to frame size.
UART_CASUALTY_CAPTURE_MIN_Y_PX = 365
UART_CASUALTY_TRACK_AIM_Y_PX = 395
UART_CASUALTY_CAPTURE_X_TOLERANCE_PX = 20
UART_CASUALTY_CAPTURE_CONFIRM_FRAMES = 3
# Visible box / full frame. Brake early before the closed claw touches a block.
UART_CLAW_APPROACH_AREA_RATIO = 0.02
UART_CLAW_OPEN_AREA_RATIO = 0.03
# Keep in sync with ROBOT_VISION_X_TOLERANCE_PX on the MCU (576x448).
UART_CLAW_CENTER_TOLERANCE_X = 15
# Repeat event 34 while the state-24 target center remains inside the ROI.
UART_FINAL_ROI_EVENT_REPEAT_MS = 500
UART_DEFAULT_MODE = 0x03
UART_MAX_X_JUMP_PX = 100
UART_MAX_Y_JUMP_PX = 100
UART_MAX_AREA_CHANGE_RATIO = 0.40
UART_X_JUMP_RESET_MS = 500
# Exclude movable target centers in either detected safety zone, including
# 100px beyond each horizontal edge. Keep the zone's original Y bounds.
# Near-view05 counts and zone06 obstacle detection retain their old geometry.
UART_SAFE_ZONE_TARGET_MARGIN_X = 100
# State 04 aligns the selected object with the image center before event 14.
UART_FRAME_CENTER_TOLERANCE_X = 20
UART_FRAME_CENTER_TOLERANCE_Y = 20
# Hold the 03->04 target identity through short misses, without reusing old
# coordinates. Confirm an actual loss/E4 only after this many fresh misses.
UART_TRACK_LOST_CONFIRM_FRAMES = 5
# Match MCU direct capture. A normal 14 ACK goes straight to 24.
UART_DIRECT_CAPTURE_AFTER_14 = True
# A tracked black object must remain missing this long as well as satisfying
# the existing frame minimum before releasing its lock/reporting E3/E4/EE.
UART_BLACK_TARGET_LOST_CONFIRM_MS = 800
# MCU owns task scheduling. Do not autonomously switch31/51 black to red11.
# This does NOT disable receiving MCU11/03 to start a casualty task.
UART_BLACK_TO_RED_PRIORITY_ENABLED = False
# After 07, an active E3 material search may request a casualty instead.
UART_E3_MATERIAL_TO_RED_ENABLED = True
UART_E3_RED_SWITCH_RETRY_MS = 100
# Legacy arrangement checks use two confirmations for first green.
UART_FIRST_GREEN_DECISION_CONFIRM_FRAMES = 2
UART_FIRST_GREEN_DECISION_TIMEOUT_MS = 500
#14/44 directional adjustment: MCU owns ALL motion timing.
# Vision only bounds completion-ACK waiting; repeated requests do not restart motion.
UART_MICRO_ADJUST_MAX_ROUNDS = 3
UART_MICRO_ADJUST_ACK_TIMEOUT_MS = 5000
# Longest current42 motion7.8s+3s margin rounded up;32 stays5s.
UART_MICRO42_ACK_TIMEOUT_MS = 11000
#32/42 completion is MCU-owned, with no minimum elapsed-time gate.
# These tolerances are used by state 04's event 14. State 24's event 34
# checks entry into the entire tracking ROI, not center-point alignment.
UART_FINAL_ROI_CENTER_TOLERANCE_X = 20
UART_FINAL_ROI_CENTER_TOLERANCE_Y = 20
# Shared tracking spacing rules use absolute box-center Y to select the X
# threshold. Red targets below the image midpoint (Y>224 at448px height)
# compare dx across all Y bands; red at/above it and blue use the SAME Y band.
# Green/black always compare dx against every movable neighbor regardless of Y.
# Every applicable neighbor must lie strictly outside the X threshold.
UART_FINAL_PRIMARY_MIN_DX_Y_GT_290 = 170
UART_FINAL_PRIMARY_MIN_DX_Y_GT_220 = 150
UART_FINAL_PRIMARY_MIN_DX_Y_GT_150 = 120
UART_FINAL_PRIMARY_MIN_DX_Y_GT_100 = 90
UART_FINAL_PRIMARY_MIN_DX_Y_GT_90 = 70
UART_FINAL_PRIMARY_MIN_DX_Y_LE_90 = 60
# Use these red-specific dx limits when either box in a pair is red.
UART_FINAL_RED_MIN_DX_Y_GT_290 = 240
UART_FINAL_RED_MIN_DX_Y_GT_220 = 170
UART_FINAL_RED_MIN_DX_Y_GT_150 = 150
UART_FINAL_RED_MIN_DX_Y_GT_100 = 120
UART_FINAL_RED_MIN_DX_Y_GT_90 = 70
UART_FINAL_RED_MIN_DX_Y_LE_90 = 50
# Multi-object DEBUG choices need confirmation; a lone valid target is
# highlighted immediately, without this delay.
DEBUG_SECONDARY_CLEAR_FRAMES = 3
# Screen-only14 preview: test casualty neighbors regardless of MCU task color.
DEBUG_ARRANGEMENT_TARGET_LABEL = "sqarered"
#02 follows the original center target and sends IMAGE-center errors after
# actual TX02.12 remains MCU-timed/open-loop without coordinate publishing.
UART_ARRANGE_MAX_Y_DIFFERENCE = 80
# Legacy side-preview/helper range only. Active02 follows the original
# center lock and does NOT gate its coordinates by this side-object range.
UART_ARRANGE_02_MAX_X_DIFFERENCE = 200
UART_ARRANGE_DEBUG_PRINT_MS = 2000
# After a successful actual05 UART write (not RX15/25 or queue preparation),
# allow the camera mechanism to settle, then confirm three full-frame actions.
UART_NEAR_VIEW_SETTLE_MS = 250
UART_NEAR_VIEW_STABLE_FRAMES = 3
# Only in05 inventories: ignore red below15% and black below3% visible
# box/image area. Exact thresholds count; green/blue and tracking are unchanged.
UART_NEAR_VIEW_RED_MIN_AREA_RATIO = 0.15
UART_NEAR_VIEW_BLACK_MIN_AREA_RATIO = 0.03
# Post07 command51 may collect for two extra rounds with a 12s visual budget. At the limit
# stop collection and request final05; NEVER assume retained cargo for06.
# MCU may also request final inventory with05 in any active51 capture/recovery
# phase. The visual timer is a backup and MCU05 can still end collection.
UART_SUPPLY_COLLECT_MAX_ATTEMPTS = 2
UART_SUPPLY_COLLECT_TIMEOUT_MS = 12000
UART_SUPPLY_SEARCH_SETTLE_MS = 250  # Wide-view resume after a supplemental05.
# All cargo tasks, including first green: confirm NEW outside targets only.
# Existing locks keep raw-YOLO association; remembered zones never steer.
UART_NEW_TARGET_CONFIRM_FRAMES = 3
UART_POST07_SUPPLY_CONFIRM_FRAMES = 2  # New green/black tracking targets after07.
UART_TARGET_ZONE_HOLD_MS = 500
UART_NEW_TARGET_ASSOC_MAX_PX = 100
# Post07 command51: exclude supplies in the bottom/inner half of the ROI
# in acquisition AND raw tracking; the front half remains eligible.
# Optional [x, y, width, height] extends this mask; None adds no extra mask.
UART_SUPPLY_COLLECT_LOADED_ROI = None
# Event02 restores wide view for arrangement. After a final/evacuation05
# recheck, successful result06/24 also waits this long before fresh steering.
UART_WIDE_VIEW_SETTLE_MS = 300
UART_ZONE_MIN_AREA_RATIO = 0.02  # <=2% is unrecognized; report E6.
UART_ZONE_CENTER_AREA_RATIO = 0.25
UART_ZONE_CLOSE_AREA_RATIO = 0.50
UART_ZONE_CLOSE_CONFIRM_FRAMES = 5  # Any wrong-side bar vetoes26 immediately.
UART_ZONE_CLOSE_EVENT_REPEAT_MS = 500
# 46 is a one-maneuver-per-delivery blackbar correction, ACKed by MCU46.
UART_ZONE_REALIGN_REPEAT_MS = 500
# Red objects are casualties; green/black/light-blue objects are supplies.
UART_ZONE_CASUALTY_LABEL = "sqarered"
# Far aim points: supplies 3/10 inward from the left; casualties 2/5
# inward from the right (3/5 measured from the left).
UART_ZONE_CASUALTY_X_RATIO = 0.6
UART_ZONE_SUPPLY_X_RATIO = 0.3
# After event16 and visible zone area >25%: both supplies and casualties
# aim at the horizontal center (1/2 from either edge).
UART_ZONE_SUPPLY_NEAR_X_RATIO = 0.5
UART_ZONE_CASUALTY_NEAR_X_RATIO = 0.5
# Safety-zone approach reference: 100px edge at Y100, rear at ROI top+20.
# Keep the geometry/DEBUG outline, but do not request36, pause coordinates or
# defer26 for objects in this range. Missing config also defaults to disabled.
UART_ZONE_OBSTACLE_ENABLED = False
# Clip away the part above Y200 without changing the existing sloped sides;
# the actual new upper edge keeps the original width at Y200, not100px.
UART_ZONE_OBSTACLE_FRONT_WIDTH_PX = 100
UART_ZONE_OBSTACLE_FRONT_Y = 100
UART_ZONE_OBSTACLE_CLIP_TOP_Y = 200
UART_ZONE_OBSTACLE_EDGE_INSIDE_PX = 20
# Request a new bypass only while the visible safety zone occupies <=25%.
UART_ZONE_OBSTACLE_MAX_AREA_RATIO = 0.25
UART_ZONE_OBSTACLE_CONFIRM_FRAMES = 3
UART_ZONE_OBSTACLE_CLEAR_FRAMES = 3
UART_ZONE_OBSTACLE_REPEAT_MS = 500

# Streaming settings.
STREAM_WIDTH = 576
STREAM_HEIGHT = 448
STREAM_FPS = 30
STREAM_BITRATE = 3_000_000
STREAM_BUFFER_NUM = 3

# Native Rtsp.stop() crashes on the currently tested firmware.
RTSP_DYNAMIC_ENABLED = False
PROTOCOL_RTSP = "RTSP"
PROTOCOL_WEBRTC = "WEBRTC"

# Recording settings.
RECORD_DIR = "/root/videos"
RECORD_WIDTH = 576
RECORD_HEIGHT = 448
RECORD_FPS = 30
RECORD_BITRATE = 4_000_000
RECORD_BUFFER_NUM = 4
RECORD_AUDIO_SAMPLE_RATE = 48_000
RECORD_AUDIO_CHANNELS = 1
RECORD_AUDIO_VOLUME = 50

# UI and debug settings.
CHINESE_FONT_NAME = "sourcehansans"
CHINESE_FONT_PATH = "/maixapp/share/font/SourceHanSansCN-Regular.otf"
DEBUG_REPORT_INTERVAL_US = 5_000_000
DEBUG_SCREEN_FPS_INTERVAL_US = 500_000
DEBUG_SCREEN_FPS_RIGHT_MARGIN_PX = 30
# Display-only46 reference preview, relative to the current tracking ROI center.
# Positive X points right; negative Y points up. UART reference is unchanged.
DEBUG_ZONE_REALIGN_POINT_OFFSET_X = 165
DEBUG_ZONE_REALIGN_POINT_OFFSET_Y = -90

# Photos go directly into SD-root/YYYY-MM-DD, using the camera's system date.
DATASET_CAPTURE_ROOT = "/root/sd"

# QR code and one-dimensional barcode scanning settings. Decoding uses a
# smaller image and alternates decoder types to protect the main-loop FPS.
CODE_SCAN_WIDTH = 288
CODE_SCAN_HEIGHT = 224
CODE_SCAN_INTERVAL_FRAMES = 12
CODE_SCAN_SUCCESS_COOLDOWN_FRAMES = 60
CODE_SCAN_RESULT_HOLD_FRAMES = 60
CODE_SCAN_MAX_TEXT_CHARS = 28
CODE_SCAN_TEXT_CACHE_MAX = 16
