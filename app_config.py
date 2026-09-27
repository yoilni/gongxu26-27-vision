"""Application settings shared by the MaixCAM YOLO modules."""

MODEL_PATH = "/root/sd/gongxun27/model_9320.mud"

# ISP image quality settings.
AUTO_AWB = True
AWB_GAIN = [0.134, 0.0625, 0.0625, 0.1139]  # R, GR, GB, B
CONTRAST = 70
CAMERA_STABILIZE_FRAMES = 30

# Detection settings. These thresholds are used by YOLO but are not drawn.
DETECT_CONFIDENCE = 0.5
DETECT_IOU = 0.45

# Secondary verification for the four useful YOLO classes. RGB888 images use
# LAB thresholds in the order [L_min, L_max, A_min, A_max, B_min, B_max].
VERIFY_ENABLED = True
VERIFY_WINDOW_FRAMES = 5
VERIFY_REQUIRED_HITS = 3
VERIFY_REQUIRED_HITS_WITHOUT_COLOR = 4
VERIFY_COLOR_MEMORY_FRAMES = 8
VERIFY_MAX_MISSES = 8
VERIFY_HIGH_CONFIDENCE_BYPASS = 0.72
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
UART_SEND_RATE_HZ = 20
UART_READ_POLL_MS = 5
UART_TARGET_FRESHNESS_MS = 200
UART_NO_TARGET_REPEAT_MS = 500
# State 03 must miss its selected target for this many consecutive frames
# before reporting E3, preventing one-frame detection flicker from triggering it.
UART_SEARCH_NO_TARGET_FRAMES = 3
# Repeat event 34 while state 24 is waiting for the MCU's 15/25 command.
UART_FINAL_ROI_EVENT_REPEAT_MS = 500
# Repeat arrangement event 02 until MCU acknowledges it with command 12.
UART_ARRANGE_02_REPEAT_MS = 500
# After MCU command 14, skip arrangement if no valid 02 decision is possible.
UART_PRE_02_DECISION_TIMEOUT_MS = 2500
UART_DEFAULT_MODE = 0x03
UART_MAX_X_JUMP_PX = 100
UART_MAX_Y_JUMP_PX = 100
UART_MAX_AREA_CHANGE_RATIO = 0.40
UART_X_JUMP_RESET_MS = 500
# State 04 aligns the selected object with the image center before event 14.
UART_FRAME_CENTER_TOLERANCE_X = 20
UART_FRAME_CENTER_TOLERANCE_Y = 20
# State 24 must align the selected object's center with the ROI center before
# event 34; merely entering the large full-width ROI is not sufficient.
UART_FINAL_ROI_CENTER_TOLERANCE_X = 20
UART_FINAL_ROI_CENTER_TOLERANCE_Y = 20
# State-24 primary target spacing rules use absolute box-center Y and the
# nearest movable object's box-center dx/dy. Threshold comparisons are strict.
UART_FINAL_PRIMARY_MAX_DY = 30
UART_FINAL_PRIMARY_MIN_DX_Y_GT_290 = 200
UART_FINAL_PRIMARY_MIN_DX_Y_GT_220 = 150
UART_FINAL_PRIMARY_MIN_DX_Y_GT_150 = 120
UART_FINAL_PRIMARY_MIN_DX_Y_GT_100 = 90
UART_FINAL_PRIMARY_MIN_DX_Y_GT_90 = 70
UART_FINAL_PRIMARY_MIN_DX_Y_LE_90 = 60
# Arrangement states 02/12 only identify side objects and drive event/state
# decisions. Their MCU movement is timed/open-loop; no coordinate packet is sent.
UART_ARRANGE_MAX_Y_DIFFERENCE = 80
# In arrangement state 02, ignore side objects whose center is horizontally
# more than 200 pixels from the locked middle object.
UART_ARRANGE_02_MAX_X_DIFFERENCE = 200
UART_ARRANGE_NO_NEIGHBOR_FRAMES = 5
UART_ARRANGE_DEBUG_PRINT_MS = 500
# After event 05, allow the MCU-operated camera mechanism to reach its near
# position, then require a stable full-frame object count before deciding.
UART_NEAR_VIEW_SETTLE_MS = 500
UART_NEAR_VIEW_STABLE_FRAMES = 3
# Event 02 makes the MCU return MG90 to the wide-angle position before the
# existing arrangement-02/12 vision logic resumes.
UART_WIDE_VIEW_SETTLE_MS = 800
UART_ZONE_CENTER_AREA_RATIO = 0.25
UART_ZONE_CLOSE_AREA_RATIO = 0.50
# Red objects are casualties; green/black/light-blue objects are supplies.
UART_ZONE_CASUALTY_LABEL = "sqarered"
UART_ZONE_CASUALTY_X_RATIO = 0.75
UART_ZONE_SUPPLY_X_RATIO = 0.25
# Safety-zone approach: 20 px outside / 10 px inside the ROI top edge.
UART_ZONE_OBSTACLE_EDGE_OUTSIDE_PX = 20
UART_ZONE_OBSTACLE_EDGE_INSIDE_PX = 10
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
DEBUG_REPORT_INTERVAL_US = 1_000_000

# QR code and one-dimensional barcode scanning settings. Decoding uses a
# smaller image and alternates decoder types to protect the main-loop FPS.
CODE_SCAN_WIDTH = 288
CODE_SCAN_HEIGHT = 224
CODE_SCAN_INTERVAL_FRAMES = 12
CODE_SCAN_SUCCESS_COOLDOWN_FRAMES = 60
CODE_SCAN_RESULT_HOLD_FRAMES = 60
CODE_SCAN_MAX_TEXT_CHARS = 28
CODE_SCAN_TEXT_CACHE_MAX = 16
