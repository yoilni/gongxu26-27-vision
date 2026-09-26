"""On-screen menu rendering and touch routing."""

from maix import image

import app_config as config


ACTION_RECORD = "record"
ACTION_AUDIO = "audio"
ACTION_ROI = "roi"
ACTION_SCAN = "scan"
ACTION_DEBUG = "debug"
ACTION_RTSP = "rtsp"
ACTION_WEBRTC = "webrtc"

PAGE_MAIN = 0
PAGE_MENU = 1
PAGE_STREAM = 2

BUTTON_MARGIN = 6
BUTTON_COLOR_START = image.Color.from_rgb(20, 170, 80)
BUTTON_COLOR_STOP = image.Color.from_rgb(220, 40, 40)
BUTTON_COLOR_BORDER = image.Color.from_rgb(255, 255, 255)
BUTTON_COLOR_MENU = image.Color.from_rgb(40, 100, 190)
BUTTON_COLOR_BACK = image.Color.from_rgb(80, 80, 80)
BUTTON_COLOR_DISABLED = image.Color.from_rgb(95, 95, 95)


class MenuUI:
    def __init__(self, frame_width, frame_height, display_device):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.display = display_device
        self.page = PAGE_MAIN
        self.chinese_font_ready = self._load_chinese_font()
        self._button_cache = {}
        self._create_layout()

    def _load_chinese_font(self):
        try:
            image.load_font(config.CHINESE_FONT_NAME, config.CHINESE_FONT_PATH)
            image.set_default_font(config.CHINESE_FONT_NAME)
            return True
        except Exception as exc:
            print("Chinese font unavailable, use English UI:", exc)
            return False

    def _create_layout(self):
        width = self.frame_width
        height = self.frame_height
        top_width = min(96, max(72, width // 3))
        top_height = min(38, max(30, height // 7))
        option_width = min(150, max(100, width // 2))
        option_height = min(40, max(32, height // 10))
        option_x = (width - option_width) // 2

        self.menu_rect = [
            width - top_width - BUTTON_MARGIN,
            BUTTON_MARGIN,
            top_width,
            top_height,
        ]
        self.back_rect = [
            BUTTON_MARGIN,
            BUTTON_MARGIN,
            top_width,
            top_height,
        ]
        option_count = 6
        option_area_height = (
            option_count * option_height
            + (option_count - 1) * BUTTON_MARGIN
        )
        option_y = max(
            top_height + BUTTON_MARGIN * 2,
            (height - option_area_height) // 2,
        )
        self.stream_rect = [
            option_x,
            option_y,
            option_width,
            option_height,
        ]
        self.record_rect = self._next_option(self.stream_rect, option_height)
        self.audio_rect = self._next_option(self.record_rect, option_height)
        self.roi_rect = self._next_option(self.audio_rect, option_height)
        self.scan_rect = self._next_option(self.roi_rect, option_height)
        self.debug_rect = self._next_option(self.scan_rect, option_height)
        self.rtsp_rect = [
            option_x,
            height // 2 - option_height - BUTTON_MARGIN,
            option_width,
            option_height,
        ]
        self.webrtc_rect = self._next_option(self.rtsp_rect, option_height)

        self.menu_touch_rect = self._map_to_touch(self.menu_rect)
        self.back_touch_rect = self._map_to_touch(self.back_rect)
        self.stream_touch_rect = self._map_to_touch(self.stream_rect)
        self.record_touch_rect = self._map_to_touch(self.record_rect)
        self.audio_touch_rect = self._map_to_touch(self.audio_rect)
        self.roi_touch_rect = self._map_to_touch(self.roi_rect)
        self.scan_touch_rect = self._map_to_touch(self.scan_rect)
        self.debug_touch_rect = self._map_to_touch(self.debug_rect)
        self.rtsp_touch_rect = self._map_to_touch(self.rtsp_rect)
        self.webrtc_touch_rect = self._map_to_touch(self.webrtc_rect)

    @staticmethod
    def _next_option(previous_rect, option_height):
        return [
            previous_rect[0],
            previous_rect[1] + option_height + BUTTON_MARGIN,
            previous_rect[2],
            option_height,
        ]

    def _map_to_touch(self, rect):
        return image.resize_map_pos(
            self.frame_width,
            self.frame_height,
            self.display.width(),
            self.display.height(),
            image.Fit.FIT_CONTAIN,
            rect[0],
            rect[1],
            rect[2],
            rect[3],
        )

    @staticmethod
    def _point_in_rect(x, y, rect):
        left, top, width, height = rect
        return left <= x < left + width and top <= y < top + height

    def handle_release(self, touch_x, touch_y):
        """Update the current page and return a requested action, if any."""
        if self.page == PAGE_MAIN:
            if self._point_in_rect(touch_x, touch_y, self.menu_touch_rect):
                self.page = PAGE_MENU
            return None

        if self.page == PAGE_MENU:
            if self._point_in_rect(touch_x, touch_y, self.back_touch_rect):
                self.page = PAGE_MAIN
            elif self._point_in_rect(touch_x, touch_y, self.stream_touch_rect):
                self.page = PAGE_STREAM
            elif self._point_in_rect(touch_x, touch_y, self.record_touch_rect):
                return ACTION_RECORD
            elif self._point_in_rect(touch_x, touch_y, self.audio_touch_rect):
                return ACTION_AUDIO
            elif self._point_in_rect(touch_x, touch_y, self.roi_touch_rect):
                return ACTION_ROI
            elif self._point_in_rect(touch_x, touch_y, self.scan_touch_rect):
                return ACTION_SCAN
            elif self._point_in_rect(touch_x, touch_y, self.debug_touch_rect):
                return ACTION_DEBUG
            return None

        if self._point_in_rect(touch_x, touch_y, self.back_touch_rect):
            self.page = PAGE_MENU
        elif self._point_in_rect(touch_x, touch_y, self.rtsp_touch_rect):
            return ACTION_RTSP
        elif self._point_in_rect(touch_x, touch_y, self.webrtc_touch_rect):
            return ACTION_WEBRTC
        return None

    def _draw_button(self, img, rect, label, color):
        """Copy a pre-rendered button instead of rasterizing text every frame."""
        left, top, width, height = rect
        cache_key = (width, height, label, id(color))
        button_img = self._button_cache.get(cache_key)
        if button_img is None:
            button_img = image.Image(width, height, image.Format.FMT_RGB888)
            button_img.draw_rect(
                0,
                0,
                width,
                height,
                color=color,
                thickness=-1,
            )
            button_img.draw_rect(
                0,
                0,
                width,
                height,
                color=BUTTON_COLOR_BORDER,
                thickness=2,
            )
            text_size = image.string_size(label)
            text_x = (width - text_size.width()) // 2
            text_y = (height - text_size.height()) // 2
            button_img.draw_string(
                text_x,
                text_y,
                label,
                color=BUTTON_COLOR_BORDER,
            )
            self._button_cache[cache_key] = button_img
        img.draw_image(left, top, button_img)

    def _label(self, chinese, english):
        return chinese if self.chinese_font_ready else english

    def draw(
        self,
        img,
        active_protocol,
        recording,
        record_audio_enabled,
        roi_enabled,
        scan_enabled,
        debug_enabled,
    ):
        if self.page == PAGE_MAIN:
            menu_color = (
                BUTTON_COLOR_STOP
                if active_protocol is not None or recording
                else BUTTON_COLOR_MENU
            )
            self._draw_button(
                img,
                self.menu_rect,
                self._label("菜单", "MENU"),
                menu_color,
            )
        elif self.page == PAGE_MENU:
            self._draw_button(
                img,
                self.back_rect,
                self._label("返回", "BACK"),
                BUTTON_COLOR_BACK,
            )
            self._draw_button(
                img,
                self.stream_rect,
                self._label("图传", "STREAM"),
                BUTTON_COLOR_STOP if active_protocol else BUTTON_COLOR_MENU,
            )
            self._draw_button(
                img,
                self.record_rect,
                self._label("停止录像", "STOP REC")
                if recording
                else self._label("录像", "RECORD"),
                BUTTON_COLOR_STOP if recording else BUTTON_COLOR_MENU,
            )
            self._draw_button(
                img,
                self.audio_rect,
                self._label("音频：开", "AUDIO ON")
                if record_audio_enabled
                else self._label("音频：关", "AUDIO OFF"),
                BUTTON_COLOR_DISABLED
                if recording
                else (
                    BUTTON_COLOR_START
                    if record_audio_enabled
                    else BUTTON_COLOR_BACK
                ),
            )
            self._draw_button(
                img,
                self.roi_rect,
                self._label("关闭ROI", "ROI OFF")
                if roi_enabled
                else self._label("启动ROI", "ROI ON"),
                BUTTON_COLOR_STOP if roi_enabled else BUTTON_COLOR_START,
            )
            self._draw_button(
                img,
                self.scan_rect,
                self._label("关闭扫码", "SCAN OFF")
                if scan_enabled
                else self._label("启动扫码", "SCAN ON"),
                BUTTON_COLOR_STOP if scan_enabled else BUTTON_COLOR_START,
            )
            self._draw_button(
                img,
                self.debug_rect,
                self._label("关闭Debug", "DEBUG OFF")
                if debug_enabled
                else self._label("启动Debug", "DEBUG ON"),
                BUTTON_COLOR_STOP if debug_enabled else BUTTON_COLOR_START,
            )
        else:
            self._draw_button(
                img,
                self.back_rect,
                self._label("返回", "BACK"),
                BUTTON_COLOR_BACK,
            )
            rtsp_color = (
                BUTTON_COLOR_DISABLED
                if not config.RTSP_DYNAMIC_ENABLED
                else (
                    BUTTON_COLOR_STOP
                    if active_protocol == config.PROTOCOL_RTSP
                    else BUTTON_COLOR_START
                )
            )
            rtsp_label = (
                self._label("RTSP禁用", "RTSP OFF")
                if not config.RTSP_DYNAMIC_ENABLED
                else "RTSP"
            )
            self._draw_button(img, self.rtsp_rect, rtsp_label, rtsp_color)
            self._draw_button(
                img,
                self.webrtc_rect,
                "WebRTC",
                BUTTON_COLOR_STOP
                if active_protocol == config.PROTOCOL_WEBRTC
                else BUTTON_COLOR_START,
            )

        if recording:
            indicator_y = self.frame_height - 14
            img.draw_circle(
                12,
                indicator_y,
                6,
                BUTTON_COLOR_STOP,
                thickness=-1,
            )
