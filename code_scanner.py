"""Optional QR-code and one-dimensional barcode recognition."""

from maix import image

import app_config as config


KIND_QR = "QR"
KIND_BARCODE = "BAR"
QR_COLOR = image.Color.from_rgb(0, 220, 255)
BARCODE_COLOR = image.Color.from_rgb(255, 210, 0)
LABEL_BACKGROUND = image.Color.from_rgb(20, 20, 20)
LABEL_TEXT = image.Color.from_rgb(255, 255, 255)
BOX_THICKNESS = 2


class CodeScanner:
    """Run the two decoders intermittently and draw cached results cheaply."""

    def __init__(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.enabled = False
        self._frame_index = 0
        self._next_scan_frame = 1
        self._next_kind = KIND_QR
        self._results = {KIND_QR: [], KIND_BARCODE: []}
        self._expires_at = {KIND_QR: 0, KIND_BARCODE: 0}
        self._announced = set()
        self._label_cache = {}

    def toggle(self):
        self.set_enabled(not self.enabled)

    def set_enabled(self, enabled):
        enabled = bool(enabled)
        if self.enabled == enabled:
            return
        self.enabled = enabled
        self._frame_index = 0
        self._next_scan_frame = 1
        self._next_kind = KIND_QR
        if not enabled:
            self._clear_results()
        print("Code scanner:", "enabled" if enabled else "disabled")

    def process(self, img):
        """Scan when scheduled, then draw recent detections on every frame."""
        if not self.enabled:
            return

        self._frame_index += 1
        if self._frame_index >= self._next_scan_frame:
            found = self._scan_kind(img, self._next_kind)
            self._next_kind = (
                KIND_BARCODE if self._next_kind == KIND_QR else KIND_QR
            )
            wait_frames = (
                config.CODE_SCAN_SUCCESS_COOLDOWN_FRAMES
                if found
                else config.CODE_SCAN_INTERVAL_FRAMES
            )
            self._next_scan_frame = self._frame_index + max(1, wait_frames)

        self._expire_old_results()
        self._draw_results(img)

    def _scan_kind(self, img, kind):
        scan_img = img
        scale_x = 1.0
        scale_y = 1.0
        try:
            scan_width = min(self.frame_width, config.CODE_SCAN_WIDTH)
            scan_height = min(self.frame_height, config.CODE_SCAN_HEIGHT)
            if (
                scan_width != self.frame_width
                or scan_height != self.frame_height
            ):
                scan_img = img.resize(scan_width, scan_height)
                scale_x = self.frame_width / float(scan_width)
                scale_y = self.frame_height / float(scan_height)

            if kind == KIND_QR:
                raw_results = scan_img.find_qrcodes()
            else:
                raw_results = scan_img.find_barcodes()
        except Exception as exc:
            print(kind, "scan failed:", exc)
            return False
        finally:
            if scan_img is not img:
                del scan_img

        results = [
            self._copy_result(item, kind, scale_x, scale_y)
            for item in raw_results
        ]
        if not results:
            return False

        self._results[kind] = results
        self._expires_at[kind] = (
            self._frame_index + config.CODE_SCAN_RESULT_HOLD_FRAMES
        )
        for result in results:
            key = (kind, result["payload"])
            if key not in self._announced:
                print("{} detected: {}".format(kind, result["payload"]))
                self._announced.add(key)
        return True

    @staticmethod
    def _copy_result(item, kind, scale_x, scale_y):
        try:
            raw_rect = [int(value) for value in item.rect()]
        except Exception:
            raw_rect = [
                int(item.x()),
                int(item.y()),
                int(item.w()),
                int(item.h()),
            ]
        rect = [
            int(raw_rect[0] * scale_x + 0.5),
            int(raw_rect[1] * scale_y + 0.5),
            int(raw_rect[2] * scale_x + 0.5),
            int(raw_rect[3] * scale_y + 0.5),
        ]

        corners = []
        try:
            corners = [
                (
                    int(point[0] * scale_x + 0.5),
                    int(point[1] * scale_y + 0.5),
                )
                for point in item.corners()
            ]
        except Exception:
            pass

        try:
            payload = str(item.payload())
        except Exception:
            payload = ""
        return {
            "kind": kind,
            "rect": rect,
            "corners": corners,
            "payload": payload,
        }

    def _expire_old_results(self):
        changed = False
        for kind in (KIND_QR, KIND_BARCODE):
            if (
                self._results[kind]
                and self._frame_index > self._expires_at[kind]
            ):
                self._results[kind] = []
                changed = True
        if changed:
            self._announced = {
                (kind, result["payload"])
                for kind in (KIND_QR, KIND_BARCODE)
                for result in self._results[kind]
            }

    def _draw_results(self, img):
        for kind, color in (
            (KIND_QR, QR_COLOR),
            (KIND_BARCODE, BARCODE_COLOR),
        ):
            for result in self._results[kind]:
                corners = result["corners"]
                rect = result["rect"]
                if len(corners) >= 4:
                    for index in range(len(corners)):
                        start = corners[index]
                        end = corners[(index + 1) % len(corners)]
                        img.draw_line(
                            start[0],
                            start[1],
                            end[0],
                            end[1],
                            color=color,
                            thickness=BOX_THICKNESS,
                        )
                else:
                    img.draw_rect(
                        rect[0],
                        rect[1],
                        rect[2],
                        rect[3],
                        color=color,
                        thickness=BOX_THICKNESS,
                    )
                self._draw_label(img, kind, result["payload"], rect, color)

    def _draw_label(self, img, kind, payload, rect, color):
        text = "{}: {}".format(kind, payload)
        text = text[: config.CODE_SCAN_MAX_TEXT_CHARS]
        cache_key = (kind, text)
        label_img = self._label_cache.get(cache_key)
        if label_img is None:
            if len(self._label_cache) >= config.CODE_SCAN_TEXT_CACHE_MAX:
                self._label_cache.clear()
            text_size = image.string_size(text)
            label_width = min(
                self.frame_width,
                max(8, text_size.width() + 8),
            )
            label_height = max(18, text_size.height() + 4)
            label_img = image.Image(
                label_width,
                label_height,
                image.Format.FMT_RGB888,
            )
            label_img.draw_rect(
                0,
                0,
                label_width,
                label_height,
                color=LABEL_BACKGROUND,
                thickness=-1,
            )
            label_img.draw_rect(
                0,
                0,
                label_width,
                label_height,
                color=color,
                thickness=1,
            )
            label_img.draw_string(4, 2, text, color=LABEL_TEXT)
            self._label_cache[cache_key] = label_img

        label_x = max(0, min(rect[0], self.frame_width - label_img.width()))
        label_y = rect[1] - label_img.height()
        if label_y < 0:
            label_y = min(
                self.frame_height - label_img.height(),
                rect[1] + rect[3],
            )
        img.draw_image(label_x, label_y, label_img)

    def _clear_results(self):
        self._results = {KIND_QR: [], KIND_BARCODE: []}
        self._expires_at = {KIND_QR: 0, KIND_BARCODE: 0}
        self._announced.clear()
