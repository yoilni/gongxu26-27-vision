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
        self.fps = None
        self._screen_start_us = None
        self._screen_frames = 0
        self._screen_interval_us = max(1, int(getattr(config, "DEBUG_SCREEN_FPS_INTERVAL_US", 500_000)))
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
        self.fps = None
        self._screen_frames = 0
        self._screen_start_us = self._report_start_us if enabled else None
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
        if not config.VISION_DIAGNOSTICS_VERBOSE:
            # Compact DEBUG only reports overall FPS. Per-stage native clock
            # calls/accumulation would not be used by its report.
            return stage_start_us
        now_us = time.ticks_us()
        self._stage_us[stage] += max(0, now_us - stage_start_us)
        return now_us

    def end_frame(self, frame_start_us):
        if not self.enabled or frame_start_us is None:
            return

        now_us = time.ticks_us()
        self._screen_frames += 1
        if self._screen_start_us is None:
            self._screen_start_us = frame_start_us
        screen_elapsed_us = now_us - self._screen_start_us
        if screen_elapsed_us >= self._screen_interval_us:
            self.fps = self._screen_frames * 1_000_000.0 / max(1, screen_elapsed_us)
            self._screen_frames = 0
            self._screen_start_us = now_us
        self._frame_count += 1
        elapsed_us = now_us - self._report_start_us
        if elapsed_us < config.DEBUG_REPORT_INTERVAL_US:
            return

        total_fps = self._frame_count * 1_000_000.0 / max(1, elapsed_us)
        if not config.VISION_DIAGNOSTICS_VERBOSE:
            print("[DEBUG] total={:.1f} FPS".format(total_fps))
            self._reset()
            return
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
