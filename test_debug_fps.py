"""Low-rate DEBUG screen FPS, black text below the upper-right BACK button."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import test_vision_rules  # Installs the host-only maix stub.
import app_config as config
import menu_ui
import performance_debug as performance


class ScreenFpsCounterTests(unittest.TestCase):
    def setUp(self):
        self.helpers = [
            patch.object(config, "VISION_DIAGNOSTICS_VERBOSE", False),
            patch.object(config, "DEBUG_REPORT_INTERVAL_US", 5_000_000),
            patch.object(config, "DEBUG_SCREEN_FPS_INTERVAL_US", 500_000),
            patch.object(performance.time, "ticks_us", return_value=0),
            patch("builtins.print"),
        ]
        mocks = []
        for helper in self.helpers:
            mocks.append(helper.start())
            self.addCleanup(helper.stop)
        self.clock, self.log = mocks[-2:]
        self.profiler = performance.PerformanceProfiler()

    def complete_frame(self, now_us):
        self.clock.return_value = now_us
        self.profiler.end_frame(0)

    def test_disabled_has_no_timing_work_or_screen_value(self):
        self.assertIsNone(self.profiler.fps)
        self.assertIsNone(self.profiler.begin_frame())
        self.profiler.mark("camera", None)
        self.profiler.end_frame(None)
        self.clock.assert_not_called()

    def test_half_second_window_counts_completed_frames_including_start_at_zero(self):
        self.profiler.set_enabled(True)
        for now in (100_000, 200_000, 300_000, 400_000):
            self.complete_frame(now)
            self.assertIsNone(self.profiler.fps)
        self.complete_frame(500_000)
        self.assertEqual(self.profiler.fps, 10.0)

    def test_screen_value_updates_at_low_rate_and_measures_recent_window(self):
        self.profiler.set_enabled(True)
        for now in range(100_000, 500_001, 100_000):
            self.complete_frame(now)
        self.assertEqual(self.profiler.fps, 10.0)
        self.complete_frame(600_000)
        self.assertEqual(self.profiler.fps, 10.0)
        self.complete_frame(1_000_000)
        self.assertEqual(self.profiler.fps, 4.0)

    def test_missing_screen_interval_config_uses_half_second_fallback(self):
        with patch.object(performance, "config", SimpleNamespace()):
            profiler = performance.PerformanceProfiler()
        self.assertEqual(profiler._screen_interval_us, 500_000)

    def test_console_report_does_not_reset_screen_fps_or_increase_log_frequency(self):
        self.profiler.set_enabled(True)
        self.log.reset_mock()
        for now in range(100_000, 5_000_001, 100_000):
            self.complete_frame(now)
        self.assertEqual(self.profiler.fps, 10.0)
        self.log.assert_called_once()
        self.assertIn("[DEBUG] total=", self.log.call_args.args[0])
        self.complete_frame(5_100_000)
        self.assertEqual(self.profiler.fps, 10.0)

    def test_turning_debug_off_and_on_starts_a_fresh_window(self):
        self.profiler.set_enabled(True)
        self.complete_frame(500_000)
        self.assertEqual(self.profiler.fps, 2.0)
        self.profiler.set_enabled(False)
        self.assertIsNone(self.profiler.fps)
        self.clock.return_value = 10_000_000
        self.profiler.set_enabled(True)
        self.complete_frame(10_500_000)
        self.assertEqual(self.profiler.fps, 2.0)


class LegacyScreenFpsHelperTests(unittest.TestCase):
    """Retained layout helpers remain usable, but current DEBUG hides this HUD."""
    def setUp(self):
        self.helpers = [
            patch.object(menu_ui.image, "Fit", SimpleNamespace(FIT_CONTAIN=1), create=True),
            patch.object(menu_ui.image, "resize_map_pos", lambda *args: list(args[-4:]), create=True),
            patch.object(menu_ui.MenuUI, "_load_chinese_font", return_value=False),
            patch.object(menu_ui.image, "string_size", side_effect=lambda text: SimpleNamespace(width=lambda: len(text) * 8), create=True),
            patch.object(menu_ui.MenuUI, "_draw_button"),
            patch.object(menu_ui.time, "ticks_us", return_value=0),
        ]
        for helper in self.helpers:
            mock = helper.start()
            self.addCleanup(helper.stop)
        self.clock = mock
        self.menu = menu_ui.MenuUI(576, 448, SimpleNamespace(width=lambda: 576, height=lambda: 448))

    def render(self, fps=12.5, debug=True):
        img = Mock()
        value = self.menu._resolve_debug_fps(debug, fps)
        if debug:
            self.menu._draw_debug_fps(img, value)
        return img

    def test_current_menu_hides_fps_and_save_status_but_keeps_photo_controls(self):
        for page in (menu_ui.PAGE_MAIN, menu_ui.PAGE_MENU, menu_ui.PAGE_STREAM):
            self.menu.page = page
            img = Mock()
            with patch.object(self.menu, "_draw_button") as button, \
                    patch.object(self.menu, "_resolve_debug_fps") as resolve:
                self.menu.draw(img, None, False, False, False, False, True,
                               photo_status="SAVED 1", debug_fps=12.5)
                img.draw_string.assert_not_called()
                resolve.assert_not_called()
                self.assertIn("PHOTO", [call.args[2] for call in button.call_args_list])

    def test_debug_screen_fps_text_is_black_and_thin(self):
        img = self.render()
        img.draw_string.assert_called_once_with(
            438, 56, "FPS: 12.5", color="black", scale=1.5, thickness=1,
        )

    def test_none_displays_placeholder_before_first_window_finishes(self):
        img = self.render(fps=None)
        self.assertEqual(img.draw_string.call_args.args[2], "FPS: --")

    def test_older_entry_omitting_fps_still_gets_numeric_draw_cadence(self):
        for now_us in range(0, 500_001, 100_000):
            self.clock.return_value = now_us
            img = Mock()
            # Deliberately use the old main.py call with NO debug_fps argument.
            self.menu._draw_debug_fps(img, self.menu._resolve_debug_fps(True, None))
            expected = "FPS: --" if now_us < 500_000 else "FPS: 10.0"
            self.assertEqual(img.draw_string.call_args.args[2], expected)
        self.clock.return_value = 1_000_000
        self.assertEqual(self.render(fps=None).draw_string.call_args.args[2], "FPS: 2.0")

    def test_explicit_profiler_value_wins_and_does_not_read_clock(self):
        self.assertEqual(self.render(0.0).draw_string.call_args.args[2], "FPS: 0.0")
        self.render(12.5)
        self.clock.assert_not_called()
        self.render(fps=None)
        self.clock.return_value = 500_000
        self.assertEqual(self.render(fps=None).draw_string.call_args.args[2], "FPS: 2.0")
        self.assertEqual(self.render(7.5).draw_string.call_args.args[2], "FPS: 7.5")
        self.assertIsNone(self.menu._fallback_fps_start_us)
        self.assertIsNone(self.menu._fallback_fps)

    def test_fallback_resets_on_debug_exit_and_does_not_count_disabled_time(self):
        self.render(fps=None)
        self.clock.return_value = 500_000
        self.render(fps=None)
        self.render(fps=None, debug=False)
        self.assertEqual(self.clock.call_count, 2)
        self.clock.return_value = 10_000_000
        self.assertEqual(self.render(fps=None).draw_string.call_args.args[2], "FPS: --")
        self.clock.return_value = 10_500_000
        self.assertEqual(self.render(fps=None).draw_string.call_args.args[2], "FPS: 2.0")

    def test_fps_is_below_return_and_to_the_right_of_photo_on_every_page(self):
        for page in (menu_ui.PAGE_MAIN, menu_ui.PAGE_MENU, menu_ui.PAGE_STREAM):
            with self.subTest(page=page):
                self.menu.page = page
                img = self.render()
                x, y, _ = img.draw_string.call_args.args
                self.assertGreater(y, self.menu.debug_back_rect[1] + self.menu.debug_back_rect[3])
                self.assertGreater(x, self.menu.photo_rect[0] + self.menu.photo_rect[2])
                self.assertEqual(x + 108, 576 - 30)

    def test_missing_right_margin_setting_still_moves_text_inside(self):
        with patch.object(menu_ui, "config", SimpleNamespace()):
            img = self.render()
        self.assertEqual(img.draw_string.call_args.args[:2], (438, 56))

    def test_longer_numeric_fps_still_has_thirty_pixel_right_margin(self):
        img = self.render(999.9)
        x, y, text = img.draw_string.call_args.args
        self.assertEqual(x + len(text) * 8 * 1.5, 576 - 30)
        self.assertEqual(y, 56)

    def test_non_debug_never_displays_fps(self):
        self.render(debug=False).draw_string.assert_not_called()
        self.clock.assert_not_called()

    def test_latest_text_layout_is_reused_without_growing_button_cache(self):
        with patch.object(menu_ui.image, "string_size", side_effect=lambda text: SimpleNamespace(width=lambda: len(text) * 8)) as measure:
            self.render(12.5)
            self.render(12.5)
            self.render(12.5)
            measure.assert_called_once()
            self.render(12.4)
            self.assertEqual(measure.call_count, 2)
        self.assertEqual(self.menu._button_cache, {})

    def test_fps_text_adds_no_touch_action_or_button_overlap(self):
        self.assertIsNone(self.menu.handle_release(500, 60, True))
        self.assertEqual(self.menu.handle_release(10, 10, True), menu_ui.ACTION_PHOTO)
        self.menu.page = menu_ui.PAGE_MENU
        self.assertIsNone(self.menu.handle_release(500, 10, True))
        self.assertEqual(self.menu.page, menu_ui.PAGE_MAIN)


if __name__ == "__main__":
    unittest.main()
