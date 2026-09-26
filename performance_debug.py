"""Low-overhead, optional per-stage FPS reporting."""

from maix import time

import app_config as config


class PerformanceProfiler:
    STAGES = (
        "camera",
        "yolo",
        "verify",
        "serial",
        "scan",
        "draw",
        "ui",
        "display",
        "media",
    )

    def __init__(self):
        self.enabled = False
        self._stage_us = {}
        self._frame_count = 0
        self._report_start_us = 0
        self._reset()

    def _reset(self):
        self._stage_us = {name: 0 for name in self.STAGES}
        self._frame_count = 0
        self._report_start_us = time.ticks_us() if self.enabled else 0

    def set_enabled(self, enabled):
        enabled = bool(enabled)
        if self.enabled == enabled:
            return
        self.enabled = enabled
        self._reset()
        print("Debug profiler:", "enabled" if enabled else "disabled")

    def toggle(self):
        self.set_enabled(not self.enabled)

    def begin_frame(self):
        if not self.enabled:
            return None
        return time.ticks_us()

    def mark(self, stage, stage_start_us):
        if not self.enabled or stage_start_us is None:
            return None
        now_us = time.ticks_us()
        self._stage_us[stage] += max(0, now_us - stage_start_us)
        return now_us

    def end_frame(self, frame_start_us):
        if not self.enabled or frame_start_us is None:
            return

        now_us = time.ticks_us()
        self._frame_count += 1
        elapsed_us = now_us - self._report_start_us
        if elapsed_us < config.DEBUG_REPORT_INTERVAL_US:
            return

        total_fps = self._frame_count * 1_000_000.0 / max(1, elapsed_us)
        parts = ["total={:.1f} FPS".format(total_fps)]
        for stage in self.STAGES:
            average_us = self._stage_us[stage] / max(1, self._frame_count)
            stage_fps = 1_000_000.0 / max(1.0, average_us)
            parts.append(
                "{}={:.1f} FPS/{:.2f} ms".format(
                    stage,
                    stage_fps,
                    average_us / 1000.0,
                )
            )
        print("[DEBUG] " + " | ".join(parts))
        self._reset()
