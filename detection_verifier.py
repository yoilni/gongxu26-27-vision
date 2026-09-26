"""Class-specific color checks and temporal confirmation for YOLO results."""

from maix import time

import app_config as config


LABEL_BLUE = "sqareblue"
LABEL_RED = "sqarered"
LABEL_GREEN = "sqareredgreen"  # Model label is historical; target is green.
LABEL_BLACK_TRIANGLE = "triangualrblack"

USEFUL_LABELS = {
    LABEL_BLUE,
    LABEL_RED,
    LABEL_GREEN,
    LABEL_BLACK_TRIANGLE,
}


class DetectionVerifier:
    """Reject weak color/shape candidates, then require repeated detections."""

    def __init__(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self._frame_index = 0
        # Confirmation follows the semantic class instead of screen position.
        # Camera motion can move a box a long distance between frames, so an
        # IoU-based track would incorrectly treat it as a brand-new target.
        self._class_states = {}
        self._color_available = True
        self._color_error_reported = False

        self._debug_start_us = 0
        self._debug_checked = 0
        self._debug_color_passed = 0
        self._debug_confirmed = 0
        self._debug_latest = {}

    def process(self, img, objects, labels, debug_enabled=False):
        """Return only current-frame objects that pass all confirmation stages."""
        if not config.VERIFY_ENABLED:
            return objects

        self._frame_index += 1
        candidates = self._select_candidates(objects, labels)
        frame_groups = {}
        confirmed_objects = []

        for obj, label in candidates:
            passed, metrics = self._validate_candidate(img, obj, label)
            self._debug_checked += 1
            if passed:
                self._debug_color_passed += 1

            score = float(getattr(obj, "score", 0.0))
            evidence = passed or score >= config.VERIFY_HIGH_CONFIDENCE_BYPASS
            frame_groups.setdefault(label, []).append(
                (obj, metrics, evidence)
            )

        for label, records in frame_groups.items():
            state = self._class_states.get(label)
            if state is None:
                state = {
                    "hit_frames": [],
                    "evidence_frames": [],
                    "confirmed": False,
                    "misses": 0,
                }
                self._class_states[label] = state

            state["misses"] = 0
            state["hit_frames"].append(self._frame_index)
            if any(record[2] for record in records):
                state["evidence_frames"].append(self._frame_index)
            self._trim_class_state(state)

            hits = len(state["hit_frames"])
            evidence_hits = len(state["evidence_frames"])
            if (
                hits >= config.VERIFY_REQUIRED_HITS
                and evidence_hits > 0
            ) or hits >= config.VERIFY_REQUIRED_HITS_WITHOUT_COLOR:
                state["confirmed"] = True

            for obj, metrics, current_evidence in records:
                metrics["hits"] = hits
                metrics["evidence"] = evidence_hits
                self._debug_latest[label] = metrics
                # Class history only proves that this class is stable. Each
                # individual box must still have current color/high-confidence
                # evidence; otherwise one good green box would approve every
                # false green candidate in the same frame.
                if state["confirmed"] and current_evidence:
                    confirmed_objects.append(obj)
                    self._debug_confirmed += 1

        self._age_class_states(frame_groups)
        self._report_debug(debug_enabled)
        return confirmed_objects

    @staticmethod
    def _select_candidates(objects, labels):
        grouped = {}
        for obj in objects:
            class_id = int(obj.class_id)
            if class_id < 0 or class_id >= len(labels):
                continue
            label = labels[class_id]
            if label not in USEFUL_LABELS:
                continue
            grouped.setdefault(label, []).append(obj)

        selected = []
        limit = max(1, config.VERIFY_MAX_CANDIDATES_PER_CLASS)
        for label, label_objects in grouped.items():
            label_objects.sort(
                key=lambda item: float(getattr(item, "score", 0.0)),
                reverse=True,
            )
            for obj in label_objects[:limit]:
                selected.append((obj, label))
        return selected

    def _validate_candidate(self, img, obj, label):
        roi = self._make_inner_roi(obj)
        metrics = {
            "ok": False,
            "red": 0.0,
            "green": 0.0,
            "blue": 0.0,
            "black": 0.0,
            "density": 0.0,
            "hits": 0,
            "evidence": 0,
        }
        if roi is None:
            return False, metrics

        aspect = roi[2] / float(max(1, roi[3]))
        if label != LABEL_BLACK_TRIANGLE and not (
            config.VERIFY_SQUARE_ASPECT_MIN
            <= aspect
            <= config.VERIFY_SQUARE_ASPECT_MAX
        ):
            return False, metrics

        if not self._color_available:
            metrics["ok"] = True
            return True, metrics

        try:
            passed = self._validate_color(img, roi, label, metrics)
        except Exception as exc:
            self._color_available = False
            if not self._color_error_reported:
                print(
                    "Detection color verification unavailable; "
                    "fall back to temporal confirmation:",
                    exc,
                )
                self._color_error_reported = True
            metrics["ok"] = True
            return True, metrics

        metrics["ok"] = passed
        return passed, metrics

    def _validate_color(self, img, roi, label, metrics):
        if label == LABEL_BLUE:
            color = self._blob_metrics(img, roi, [config.VERIFY_LAB_BLUE])
            metrics["blue"] = color["ratios"][0]
            metrics["density"] = color["largest_density"]
            return self._single_color_passed(color)

        if label == LABEL_RED:
            color = self._blob_metrics(img, roi, [config.VERIFY_LAB_RED])
            metrics["red"] = color["ratios"][0]
            metrics["density"] = color["largest_density"]
            return self._single_color_passed(color)

        if label == LABEL_GREEN:
            color = self._blob_metrics(img, roi, [config.VERIFY_LAB_GREEN])
            metrics["green"] = color["ratios"][0]
            metrics["density"] = color["largest_density"]
            return self._single_color_passed(color)

        color = self._blob_metrics(img, roi, [config.VERIFY_LAB_BLACK])
        black_ratio = color["ratios"][0]
        density = color["largest_density"]
        metrics["black"] = black_ratio
        metrics["density"] = density
        return (
            config.VERIFY_BLACK_MIN_RATIO
            <= black_ratio
            <= config.VERIFY_BLACK_MAX_RATIO
            and config.VERIFY_TRIANGLE_DENSITY_MIN
            <= density
            <= config.VERIFY_TRIANGLE_DENSITY_MAX
        )

    @staticmethod
    def _single_color_passed(color):
        return (
            color["ratios"][0] >= config.VERIFY_SINGLE_COLOR_MIN_RATIO
            and color["largest_ratio"]
            >= config.VERIFY_LARGEST_BLOB_MIN_RATIO
        )

    @staticmethod
    def _blob_metrics(img, roi, thresholds):
        roi_area = max(1, roi[2] * roi[3])
        sampled_area = max(
            1,
            roi_area // max(
                1,
                config.VERIFY_X_STRIDE * config.VERIFY_Y_STRIDE,
            ),
        )
        minimum_pixels = max(4, int(sampled_area * 0.01))
        minimum_area = max(4, int(roi_area * 0.01))
        blobs = img.find_blobs(
            thresholds,
            roi=roi,
            x_stride=config.VERIFY_X_STRIDE,
            y_stride=config.VERIFY_Y_STRIDE,
            area_threshold=minimum_area,
            pixels_threshold=minimum_pixels,
            merge=False,
        )

        pixels_by_threshold = [0 for _ in thresholds]
        largest_pixels = 0
        largest_density = 0.0
        for blob in blobs:
            pixels = int(blob.pixels())
            code = int(blob.code())
            for index in range(len(thresholds)):
                if code & (1 << index):
                    pixels_by_threshold[index] += pixels
            if pixels > largest_pixels:
                largest_pixels = pixels
                largest_density = float(blob.density())

        return {
            "ratios": [
                min(1.0, pixels / float(roi_area))
                for pixels in pixels_by_threshold
            ],
            "largest_ratio": min(1.0, largest_pixels / float(roi_area)),
            "largest_density": largest_density,
        }

    def _make_inner_roi(self, obj):
        x = max(0, int(obj.x))
        y = max(0, int(obj.y))
        right = min(self.frame_width, int(obj.x + obj.w))
        bottom = min(self.frame_height, int(obj.y + obj.h))
        width = right - x
        height = bottom - y
        if (
            width < config.VERIFY_MIN_BOX_SIZE
            or height < config.VERIFY_MIN_BOX_SIZE
        ):
            return None

        inset_x = int(width * config.VERIFY_ROI_INSET_RATIO)
        inset_y = int(height * config.VERIFY_ROI_INSET_RATIO)
        x += inset_x
        y += inset_y
        width -= inset_x * 2
        height -= inset_y * 2
        if width <= 0 or height <= 0:
            return None
        return [x, y, width, height]

    def _trim_class_state(self, state):
        first_valid_frame = (
            self._frame_index - config.VERIFY_WINDOW_FRAMES + 1
        )
        state["hit_frames"] = [
            frame
            for frame in state["hit_frames"]
            if frame >= first_valid_frame
        ]
        first_evidence_frame = (
            self._frame_index - config.VERIFY_COLOR_MEMORY_FRAMES + 1
        )
        state["evidence_frames"] = [
            frame
            for frame in state["evidence_frames"]
            if frame >= first_evidence_frame
        ]

    def _age_class_states(self, frame_groups):
        expired_labels = []
        for label, state in self._class_states.items():
            self._trim_class_state(state)
            if label not in frame_groups:
                state["misses"] += 1
            if state["misses"] > config.VERIFY_MAX_MISSES:
                expired_labels.append(label)
        for label in expired_labels:
            del self._class_states[label]

    def _report_debug(self, enabled):
        if not enabled:
            self._debug_start_us = 0
            self._reset_debug_counts()
            return

        now_us = time.ticks_us()
        if self._debug_start_us == 0:
            self._debug_start_us = now_us
            return
        if now_us - self._debug_start_us < config.DEBUG_REPORT_INTERVAL_US:
            return

        details = []
        for label in sorted(self._debug_latest):
            values = self._debug_latest[label]
            details.append(
                "{}:R{:.2f}/G{:.2f}/B{:.2f}/K{:.2f}/D{:.2f}/H{}/E{}".format(
                    label,
                    values["red"],
                    values["green"],
                    values["blue"],
                    values["black"],
                    values["density"],
                    values["hits"],
                    values["evidence"],
                )
            )
        print(
            "[VERIFY] checked={} color_pass={} confirmed={} active_classes={}{}".format(
                self._debug_checked,
                self._debug_color_passed,
                self._debug_confirmed,
                len(self._class_states),
                " | " + " | ".join(details) if details else "",
            )
        )
        self._debug_start_us = now_us
        self._reset_debug_counts()

    def _reset_debug_counts(self):
        self._debug_checked = 0
        self._debug_color_passed = 0
        self._debug_confirmed = 0
        self._debug_latest = {}
