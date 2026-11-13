"""Raw DEBUG boxes remain; padding, LAB means and purple filtering stay gone."""

import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import test_vision_rules  # Supplies the host-only maix stub.
from test_vision_rules import LABELS, block
from maix import image
import app_config as config
import debug_lab_preview
import menu_ui
import performance_debug
import serial_controller as serial_module
import vision_overlay as overlay


SAMPLES = [
    ((50, 40, -40), (160, 40, 190)),  # purple
    ((60, -30, 30), (40, 180, 50)),   # green
    ((45, 50, 25), (190, 40, 40)),   # red
    ((2, 0, 0), (5, 5, 5)),         # black
]


class PixelImage:
    """Synthetic LAB/RGB pairs; emulates API semantics, not RGB conversion."""

    def __init__(self, samples=None):
        self.samples = list(SAMPLES if samples is None else samples)
        self.binary = Mock(side_effect=AssertionError("purple filtering was removed"))
        self.get_statistics = Mock(side_effect=AssertionError("DEBUG LAB means were removed"))
        self.find_blobs = Mock(return_value=[])
        self.get_regression = Mock(return_value=[])
        self.find_lines = Mock(return_value=[])
        self.get_pixel = Mock(side_effect=AssertionError("per-pixel native scanning must stay removed"))
        self.format = Mock(return_value=image.Format.FMT_RGB888)
        self.to_bytes = Mock(side_effect=AssertionError("line-preview byte conversion was removed"))
        self.draw_image = Mock(side_effect=AssertionError("line-preview ROI conversion was removed"))
        self.draw_calls = []

    def _draw(self, name, args, kwargs):
        self.draw_calls.append((name, args, kwargs))
        # Any overlay contaminates pixel data: tests catch sampling after draw.
        self.samples[0] = ((99, 0, 0), (255, 255, 255))

    def draw_rect(self, *args, **kwargs):
        self._draw("rect", args, kwargs)

    def draw_string(self, *args, **kwargs):
        self._draw("text", args, kwargs)

    def draw_line(self, *args, **kwargs):
        self._draw("line", args, kwargs)



class PurpleFilterRemovedTests(unittest.TestCase):
    def test_filter_class_and_purple_threshold_have_been_removed(self):
        self.assertFalse(hasattr(debug_lab_preview, "DebugLabPreview"))
        self.assertFalse(hasattr(debug_lab_preview, "DEFAULT_PURPLE_LAB"))
        self.assertFalse(hasattr(config, "DEBUG_LAB_PURPLE"))
        self.assertFalse(hasattr(config, "DEBUG_LAB_PREVIEW_ENABLED"))

    def test_box_means_and_padding_implementation_have_been_removed(self):
        self.assertFalse(hasattr(debug_lab_preview, "DebugBoxLabMeans"))
        self.assertFalse(hasattr(overlay, "draw_debug_box_lab_means"))
        self.assertFalse(hasattr(overlay, "DEBUG_YOLO_PADDING"))
        self.assertFalse(hasattr(overlay, "DEBUG_YOLO_ZONE_PADDING"))


class OriginalDebugBoxTests(unittest.TestCase):
    def test_all_classes_use_original_yolo_rectangle_without_padding(self):
        for class_id in range(7):
            with self.subTest(class_id=class_id):
                raw = PixelImage()
                obj = block(200, 300, class_id)
                self.assertEqual(overlay.debug_yolo_rect(obj, 576, 448), [190, 290, 20, 20])
                overlay.draw_debug_yolo_detections(raw, [obj], 576, 448)
                self.assertEqual(raw.draw_calls[0][1], (190, 290, 20, 20))
                raw.get_statistics.assert_not_called()
                raw.binary.assert_not_called()
                self.assertEqual((obj.x, obj.y, obj.w, obj.h), (190, 290, 20, 20))

    def test_clips_original_boxes_at_both_image_corners(self):
        objects = [block(0, 0), block(576, 448)]
        raw = PixelImage()
        overlay.draw_debug_yolo_detections(raw, objects, 576, 448)
        self.assertEqual([args for name, args, _ in raw.draw_calls if name == "rect"],
                         [(0, 0, 10, 10), (566, 438, 10, 10)])

    def test_fractional_edges_round_outward_before_clipping(self):
        obj = SimpleNamespace(x=20.2, y=30.7, w=10.1, h=12.5, class_id=5)
        self.assertEqual(overlay.debug_yolo_rect(obj, 576, 448), [20, 30, 11, 14])

    def test_invalid_or_offscreen_objects_do_not_draw_or_sample(self):
        invalid = block(100, 100)
        invalid.w = 0
        nonfinite = block(100, 100)
        nonfinite.x = float("nan")
        raw = PixelImage()
        overlay.draw_debug_yolo_detections(raw, [invalid, nonfinite, block(700, 600)], 576, 448)
        self.assertEqual(raw.draw_calls, [])
        raw.get_statistics.assert_not_called()

    def test_all_overlapping_boxes_keep_class_colors_without_mean_text(self):
        raw = PixelImage()
        objects = [block(200, 200, 4), block(205, 200, 5)]
        overlay.draw_debug_yolo_detections(raw, objects, 576, 448)
        self.assertEqual([kwargs["color"] for name, _, kwargs in raw.draw_calls if name == "rect"],
                         ["red", "green"])
        self.assertFalse(any(name == "text" for name, _, _ in raw.draw_calls))
        raw.get_statistics.assert_not_called()

    def test_box_just_outside_image_is_not_made_visible_by_old_padding(self):
        raw = PixelImage()
        outside = [block(-15, 200), block(591, 200), block(200, -15), block(200, 463)]
        overlay.draw_debug_yolo_detections(raw, outside, 576, 448)
        self.assertEqual(raw.draw_calls, [])

    def test_empty_frame_never_reuses_previous_boxes(self):
        first, second = PixelImage(), PixelImage()
        overlay.draw_debug_yolo_detections(first, [block(200, 200)], 576, 448)
        overlay.draw_debug_yolo_detections(second, [], 576, 448)
        self.assertEqual(len(first.draw_calls), 1)
        self.assertEqual(second.draw_calls, [])


class LabeledDebugBoxTests(unittest.TestCase):
    def test_model_labels_and_scores_survive_semantic_id_remapping(self):
        from model_class_mapping import ModelClassMapper
        labels = ["bluesafety", "redsafety", "sqareblue", "sqaregreen",
                  "sqarered", "startzone", "triangualrblack", "blackbar"]
        mapper = ModelClassMapper(labels)
        raw = [block(200, 200, index) for index in range(len(labels))]
        for index, obj in enumerate(raw):
            obj.score = .40 + index * .07
        objects = mapper.normalize(raw) + mapper.blackbars(raw) + mapper.startzones(raw)
        img = PixelImage()
        overlay.draw_debug_yolo_detections(img, objects, 576, 448, labels=labels)
        self.assertCountEqual([args[2] for name, args, _ in img.draw_calls if name == "text"],
                              ["{} {:.2f}".format(label, obj.score) for label, obj in zip(labels, raw)])
        self.assertEqual([kwargs["color"] for name, _, kwargs in img.draw_calls if name == "rect"],
                         ["blue", "red", (0, 191, 255), "green", "red", "black", "black", (128, 128, 128)])
        self.assertFalse(any(name == "line" for name, _, _ in img.draw_calls))

    def test_labels_are_at_upper_left_and_remain_inside_image_edges(self):
        ordinary, corner = block(200, 200, 4), block(576, 0, 5)
        ordinary.score, corner.score = .82, .91
        img = PixelImage()
        with patch.object(overlay.image, "string_size", side_effect=AttributeError, create=True):
            overlay.draw_debug_yolo_detections(img, [ordinary, corner], 576, 448, labels=LABELS)
        text = [args for name, args, _ in img.draw_calls if name == "text"]
        self.assertEqual(text[0], (190, 170, "sqarered 0.82"))
        self.assertEqual(text[1][2], "sqareredgreen 0.91")
        self.assertGreaterEqual(text[1][0], 0)
        self.assertLess(text[1][0], 566)
        self.assertEqual(text[1][1], 2)


class DebugObjectAreaTests(unittest.TestCase):
    def test_each_raw_cargo_box_gets_thin_black_percentage_without_pixel_sampling(self):
        raw = PixelImage()
        objects = [block(100 + i * 80, 200, class_id)
                   for i, class_id in enumerate((3, 4, 5, 6))]
        before = [vars(obj).copy() for obj in objects]
        for obj in objects:
            obj.score = object()  # Area is not confidence.
        overlay.draw_debug_object_area(raw, objects, 576, 448)
        self.assertEqual(len(raw.draw_calls), 4)
        for name, args, kwargs in raw.draw_calls:
            self.assertEqual(name, "text")
            self.assertEqual(args[2], "AREA: 0.2%")
            self.assertEqual(kwargs, {"color": "black", "scale": 1.2, "thickness": 1})
        for obj, original in zip(objects, before):
            self.assertEqual((obj.x, obj.y, obj.w, obj.h, obj.class_id),
                             tuple(original[key] for key in ("x", "y", "w", "h", "class_id")))
        raw.get_statistics.assert_not_called()
        raw.binary.assert_not_called()

    def test_denominator_is_entire_image_and_not_roi_or_box_confidence(self):
        raw = PixelImage()
        obj = SimpleNamespace(x=0, y=224, w=288, h=224, class_id=5)
        overlay.draw_debug_object_area(raw, [obj], 576, 448)
        self.assertEqual(raw.draw_calls[0][1][2], "AREA: 25.0%")

    def test_offscreen_portion_is_not_counted_and_full_frame_never_exceeds100(self):
        for geometry, expected in (((-288, -224, 576, 448), "AREA: 25.0%"),
                                   ((-100, -100, 1000, 1000), "AREA: 100.0%")):
            with self.subTest(geometry=geometry):
                raw = PixelImage()
                obj = SimpleNamespace(**dict(zip(("x", "y", "w", "h"), geometry)), class_id=4)
                overlay.draw_debug_object_area(raw, [obj], 576, 448)
                self.assertEqual(raw.draw_calls[0][1][2], expected)

    def test_fractional_area_uses_yolo_geometry_not_rounded_draw_rectangle(self):
        raw = PixelImage()
        obj = SimpleNamespace(x=20.2, y=30.7, w=10.1, h=12.5, class_id=6)
        overlay.draw_debug_object_area(raw, [obj], 100, 100)
        self.assertEqual(raw.draw_calls[0][1][2], "AREA: 1.3%")

    def test_zones_and_ignored_classes_do_not_duplicate_existing_zone_area_labels(self):
        raw = PixelImage()
        overlay.draw_debug_object_area(raw, [block(200, 200, class_id)
                                            for class_id in (-1, 0, 1, 2, 7, 99)], 576, 448)
        self.assertEqual(raw.draw_calls, [])

    def test_empty_invalid_and_offscreen_boxes_do_not_create_text(self):
        invalid = block(200, 200)
        invalid.w = 0
        nonfinite = block(200, 200)
        nonfinite.y = float("nan")
        for objects in ([], [invalid, nonfinite, block(-100, 200), block(700, 600)]):
            with self.subTest(objects=objects):
                raw = PixelImage()
                overlay.draw_debug_object_area(raw, objects, 576, 448)
                self.assertEqual(raw.draw_calls, [])

    def test_annotations_stay_inside_image_and_use_box_top_when_possible(self):
        for x, y, expected in ((200, 200, (190, 168)), (0, 0, (0, 2)),
                               (576, 448, (440, 416))):
            with self.subTest(x=x, y=y):
                raw = PixelImage()
                overlay.draw_debug_object_area(raw, [block(x, y)], 576, 448)
                self.assertEqual(raw.draw_calls[0][1][:2], expected)

    def test_no_cached_percentage_after_empty_frame_and_zero_size_is_safe(self):
        first, second = PixelImage(), PixelImage()
        overlay.draw_debug_object_area(first, [block(200, 200)], 576, 448)
        overlay.draw_debug_object_area(second, [], 576, 448)
        self.assertEqual(len(first.draw_calls), 1)
        self.assertEqual(second.draw_calls, [])
        for dimensions in ((0, 448), (576, 0), (-1, 448)):
            overlay.draw_debug_object_area(second, [block(200, 200)], *dimensions)
        self.assertEqual(second.draw_calls, [])


class NormalSelectedTargetPaddingTests(unittest.TestCase):
    def draw(self, obj, dimensions=True):
        img = Mock()
        img.width.return_value, img.height.return_value = 576, 448
        args = (576, 448) if dimensions else ()
        overlay.draw_selected_target(img, obj, LABELS, *args)
        img.draw_string.assert_not_called()
        return img

    def test_original_and_y_scaled_xy_outer_yellow_boxes_for_each_cargo_class(self):
        for class_id in (3, 4, 5, 6):
            with self.subTest(class_id=class_id):
                obj = block(200, 200, class_id)
                before = vars(obj).copy()
                img = self.draw(obj)
                self.assertEqual([call.args for call in img.draw_rect.call_args_list],
                                 [(190, 190, 20, 20), (141, 141, 118, 118)])
                for call in img.draw_rect.call_args_list:
                    self.assertEqual(call.kwargs, {
                        "color": overlay.SELECTED_COLOR,
                        "thickness": overlay.SELECTED_BOX_THICKNESS,
                    })
                self.assertEqual(vars(obj), before)
                img.width.assert_not_called()
                img.height.assert_not_called()

    def test_outer_box_clips_at_all_edges(self):
        for x, y, inner, outer in (
                (10, 10, (0, 0, 20, 20), (0, 0, 27, 27)),
                (566, 438, (556, 428, 20, 20), (481, 353, 95, 95)),
                (0, 0, (0, 0, 10, 10), (0, 0, 15, 15)),
                (576, 448, (566, 438, 10, 10), (491, 363, 85, 85))):
            with self.subTest(x=x, y=y):
                img = self.draw(block(x, y))
                self.assertEqual([call.args for call in img.draw_rect.call_args_list], [inner, outer])

    def test_legacy_three_argument_call_uses_image_dimensions(self):
        img = self.draw(block(200, 200), dimensions=False)
        self.assertEqual(img.draw_rect.call_args.args, (141, 141, 118, 118))
        img.width.assert_called_once()
        img.height.assert_called_once()

    def test_x_margin_interpolates_by_original_center_y_and_clamps(self):
        for center_y, expected in ((-10, 5), (0, 5), (90, 25), (100, 27),
                                   (150, 38), (159.5, 40), (200, 49),
                                   (220, 53), (224, 54), (290, 69),
                                   (319, 75), (448, 75)):
            with self.subTest(center_y=center_y):
                self.assertEqual(overlay.selected_target_x_padding(center_y), expected)
        margins = [overlay.selected_target_x_padding(y) for y in range(320)]
        self.assertEqual(margins, sorted(margins))
        self.assertEqual((min(margins), max(margins)), (5, 75))

    def test_y_expansion_uses_same_original_center_y_margin_as_x(self):
        for center_y, expected in ((0, (185, 0, 30, 15)),
                                   (159.5, (150, 109, 100, 101)),
                                   (319, (115, 234, 170, 170))):
            with self.subTest(center_y=center_y):
                img = self.draw(block(200, center_y))
                inner, outer = [call.args for call in img.draw_rect.call_args_list]
                self.assertEqual(outer, expected)
                self.assertLessEqual(outer[1], inner[1])
                self.assertGreater(outer[3], inner[3])

    def test_absent_invalid_or_offscreen_target_does_not_invent_an_outer_box(self):
        invalid = block(200, 200)
        invalid.w = 0
        nonfinite = block(200, 200)
        nonfinite.x = float("nan")
        for obj in (None, block(200, 200, 0), block(200, 200, 99),
                    block(-15, 200), block(591, 200), invalid, nonfinite):
            with self.subTest(obj=obj):
                self.draw(obj).draw_rect.assert_not_called()

    def test_safety_zone_selected_border_stays_single_and_unexpanded(self):
        for class_id in (1, 2):
            img = self.draw(block(200, 200, class_id))
            img.draw_rect.assert_called_once_with(190, 190, 20, 20,
                                                  color=overlay.SELECTED_COLOR,
                                                  thickness=overlay.SELECTED_BOX_THICKNESS)


class NormalSafetyZoneDisplayTests(unittest.TestCase):
    def test_both_zone_colors_keep_borders_but_have_no_text_or_percentages(self):
        img = Mock()
        zones = [block(200, 200, 1), block(400, 200, 2)]
        # Normal rendering no longer even reads/formats confidence values.
        for zone in zones:
            zone.score = object()
        overlay.draw_safety_zones(img, zones, LABELS)
        self.assertEqual(img.draw_rect.call_count, 2)
        for call, zone in zip(img.draw_rect.call_args_list, zones):
            self.assertEqual(call.args, (zone.x, zone.y, zone.w, zone.h))
            self.assertEqual(call.kwargs, {
                "color": overlay.SAFETY_ZONE_STYLES[LABELS[zone.class_id]][1],
                "thickness": overlay.SAFETY_BOX_THICKNESS,
            })
        img.draw_string.assert_not_called()

    def test_deprecated_foreign_and_invalid_classes_have_no_zone_overlay(self):
        img = Mock()
        objects = [block(200, 200, class_id) for class_id in (-1, 0, 3, 4, 5, 6, 99)]
        overlay.draw_safety_zones(img, objects, LABELS)
        img.draw_rect.assert_not_called()
        img.draw_string.assert_not_called()


class DebugLabMenuTests(unittest.TestCase):
    def setUp(self):
        self.fit_patch = patch.object(menu_ui.image, "Fit", SimpleNamespace(FIT_CONTAIN=1), create=True)
        self.map_patch = patch.object(menu_ui.image, "resize_map_pos", lambda *args: list(args[-4:]), create=True)
        self.font_patch = patch.object(menu_ui.MenuUI, "_load_chinese_font", return_value=False)
        for helper in (self.fit_patch, self.map_patch, self.font_patch):
            helper.start()
            self.addCleanup(helper.stop)
        self.menu = menu_ui.MenuUI(576, 448, SimpleNamespace(width=lambda: 576, height=lambda: 448))

    def test_large_photo_button_remains_without_lab_button_or_touch_region(self):
        self.assertEqual(self.menu.photo_rect, [6, 6, 144, 60])
        self.assertFalse(hasattr(self.menu, "lab_rect"))
        self.assertFalse(hasattr(self.menu, "lab_touch_rect"))
        self.assertFalse(hasattr(self.menu, "lab_preview_enabled"))
        self.assertFalse(hasattr(menu_ui, "ACTION_DEBUG_LAB"))

    def test_photo_still_works_and_former_lab_region_does_nothing(self):
        for page in (menu_ui.PAGE_MAIN, menu_ui.PAGE_MENU, menu_ui.PAGE_STREAM):
            self.menu.page = page
            self.assertIsNone(self.menu.handle_release(10, 80, True))
            self.assertEqual(self.menu.handle_release(10, 10, True), menu_ui.ACTION_PHOTO)
        self.menu.page = menu_ui.PAGE_MAIN
        self.assertIsNone(self.menu.handle_release(10, 80, False))
        self.assertEqual(self.menu.page, menu_ui.PAGE_MAIN)

    def test_debug_renders_photo_but_no_lab_button(self):
        img = Mock()
        with patch.object(self.menu, "_draw_button") as button:
            self.menu.draw(img, None, False, False, False, False, True)
            labels = [call.args[2] for call in button.call_args_list]
            self.assertIn("PHOTO", labels)
            self.assertFalse(any("LAB" in label for label in labels))
            button.reset_mock()
            self.menu.draw(img, None, False, False, False, False, False)
            self.assertNotIn("PHOTO", [call.args[2] for call in button.call_args_list])


class IdeImageTransportStartupTests(unittest.TestCase):
    def test_transport_disabled_before_maix_import_with_or_without_ide_environment(self):
        source = Path(__file__).with_name("main.py").read_text(encoding="utf8")
        bootstrap, separator, _ = source.partition("from maix import")
        self.assertTrue(separator)
        for initial in (None, "1", "0"):
            with self.subTest(initial=initial), patch.dict(os.environ):
                if initial is None:
                    os.environ.pop("MAIXVISION", None)
                else:
                    os.environ["MAIXVISION"] = initial
                exec(compile(bootstrap, "main.py", "exec"), {})
                self.assertEqual(os.environ["MAIXVISION"], "0")


class DebugMainIntegrationTests(unittest.TestCase):
    def test_startzone_is_gray_in_debug_only_without_leaking_into_uart(self):
        for debug in (False, True):
            for source_id, score, expected in ((5, .5, ((400, 200),)),
                                                (5, .4999, ())):
                self._check_main(
                    model_labels=["bluesafety", "redsafety", "sqareblue", "sqaregreen",
                                  "sqarered", "startzone", "triangualrblack", "blackbar"],
                    model_ids=(3, 6, source_id), model_scores=(.5, .7, score),
                    expected_class_ids=(5, 6), debug_enabled=debug,
                    expected_startzone_centers=expected,
                )

    def test_blackbars_keep_black_class_color_without46_preview_in_actual_main(self):
        self._check_main(
            model_labels=["bluesafety", "redsafety", "sqareblue", "sqaregreen",
                          "sqarered", "startzone", "triangualrblack", "blackbar"],
            model_ids=(3, 6, 0, 7), expected_class_ids=(5, 6, 1),
            expected_blackbar_centers=((500, 200),), zone_detection_rect=(0, 0, 576, 448),
            highlight_bar=True,
        )

    def test_new_blackbar_flows_separately_without_cargo_or_cv_processing(self):
        for debug_enabled in (False, True):
            with self.subTest(debug_enabled=debug_enabled):
                self._check_main(
                    model_labels=["bluesafety", "redsafety", "sqareblue", "sqaregreen",
                                  "sqarered", "startzone", "triangualrblack", "blackbar"],
                    model_ids=(3, 6, 0, 7, 7, 5), model_scores=(.5, .7, .4, .5, .4999, .99),
                    expected_class_ids=(5, 6, 1), expected_blackbar_centers=((500, 200),),
                    zone_detection_rect=(0, 0, 576, 448), debug_enabled=debug_enabled,
                )

    def test_new_blackbar_without_zone_is_not_shown_as_cargo_in_debug(self):
        self._check_main(
            model_labels=["bluesafety", "redsafety", "sqareblue", "sqaregreen",
                          "sqarered", "startzone", "triangualrblack", "blackbar"],
            model_ids=(3, 6, 7), expected_class_ids=(5, 6),
            expected_blackbar_centers=((400, 200),),
        )

    def test_annotated_photo_keeps_inference_and_uart_pixels_raw_without_lab_work(self):
        self._check_main()

    def test_actual_main_photo_has_all_class_color_boxes_labels_and_scores(self):
        for mode in (serial_module.CMD_SEARCH, serial_module.MODE_FINAL_ROI_TRACK):
            with self.subTest(mode=mode):
                self._check_main(
                    model_ids=(4, 3, 5, 7), expected_class_ids=(4, 3, 5, 6),
                    detection_centers=((288, 224), (350, 224), (288, 280), (500, 100)),
                    serial_mode=mode,
                )

    def test_actual_main_does_not_hide_unrelated_or_casualty_relaxed_neighbors(self):
        self._check_main(
            model_ids=(4, 3, 5), expected_class_ids=(4, 3, 5),
            detection_centers=((288, 224), (250, 190), (500, 100)),
        )

    def test_debug_shows_both_safety_zones_unverified_cargo_and_roi(self):
        for shape in ("triangle", "rectangle"):
            with patch.object(config, "UART_TRACKING_ROI_SHAPE", shape):
                self._check_main(
                    model_ids=(1, 2, 4, 5, 7), expected_class_ids=(1, 2, 4, 5, 6),
                    reject_verified=True, toggle_roi=True,
                )

    def test_reordered_seven_class_model_reaches_uart_and_debug_normalized(self):
        self._check_main(model_labels=["bluesafety", "redsafety", "sqareblue", "sqaregreen",
                                      "sqarered", "startzone", "triangualrblack"],
                         model_ids=(3, 6, 5))

    def test_missing_photo_root_config_still_defaults_to_sd(self):
        self._check_main(missing_photo_root=True)

    def test_normal_mode_also_does_no_debug_lab_work(self):
        self._check_main(debug_enabled=False)

    def test_live_fps_counter_reaches_menu_as_numbers_after_warmup(self):
        self._check_main(live_fps=True)

    def test_live_fps_counter_is_wired_in_both_yolo_buffer_modes(self):
        for dual_buff in (False, True):
            with self.subTest(dual_buff=dual_buff), patch.object(config, "DETECT_DUAL_BUFFER", dual_buff):
                self._check_main(live_fps=True)

    def test_empty_yolo_results_do_not_stop_fps_in_either_buffer_mode(self):
        for dual_buff in (False, True):
            with self.subTest(dual_buff=dual_buff), patch.object(config, "DETECT_DUAL_BUFFER", dual_buff):
                self._check_main(live_fps=True, detection_empty_frames=(0, 1, 4))

    def test_class_confidence_gates_every_consumer_in_the_actual_main_loop(self):
        self._check_main(
            model_labels=["bluesafety", "redsafety", "sqareblue", "sqaregreen", "sqarered", "startzone", "triangualrblack"],
            model_ids=(3, 6, 5, 6, 2, 0),
            model_scores=(.5, .7, .99, .6999, .4999, .3999),
        )

    def test_safety_zone_at_point4_survives_lower_detector_floor(self):
        self._check_main(
            model_labels=["bluesafety", "redsafety", "sqareblue", "sqaregreen", "sqarered", "startzone", "triangualrblack"],
            model_ids=(3, 6, 0, 5), model_scores=(.5, .7, .4, .99),
            expected_class_ids=(5, 6, 1),
        )

    def test_non_debug_view_does_not_show_safety_zone_text_or_area(self):
        self._check_main(
            model_labels=["bluesafety", "redsafety", "sqareblue", "sqaregreen", "sqarered", "startzone", "triangualrblack"],
            model_ids=(3, 6, 0, 5), model_scores=(.5, .7, .4, .99),
            expected_class_ids=(5, 6, 1), debug_enabled=False,
        )

    def test_stale_purple_config_cannot_reenable_removed_work(self):
        with patch.object(config, "DEBUG_LAB_PREVIEW_ENABLED", True, create=True), \
             patch.object(config, "DEBUG_LAB_PURPLE", [0, 100, -128, 127, -128, 127], create=True):
            self._check_main()

    def test_debug_safety_zone_never_scans_black_edges_or_draws_static46_marker(self):
        self._check_main(
            model_labels=["bluesafety", "redsafety", "sqareblue", "sqaregreen", "sqarered", "startzone", "triangualrblack"],
            model_ids=(3, 6, 0), model_scores=(.5, .7, .4),
            expected_class_ids=(5, 6, 1), zone_detection_rect=(390, 170, 40, 120))

    def test_normal_safety_zone_never_runs_retired_line_detection(self):
        self._check_main(
            model_labels=["bluesafety", "redsafety", "sqareblue", "sqaregreen", "sqarered", "startzone", "triangualrblack"],
            model_ids=(3, 6, 0), model_scores=(.5, .7, .4),
            expected_class_ids=(5, 6, 1), debug_enabled=False,
            zone_detection_rect=(390, 170, 40, 120))

    def test_empty_debug_frames_never_reuse_a_retired_yellow_line(self):
        self._check_main(
            model_labels=["bluesafety", "redsafety", "sqareblue", "sqaregreen", "sqarered", "startzone", "triangualrblack"],
            model_ids=(3, 6, 0), model_scores=(.5, .7, .4),
            expected_class_ids=(5, 6, 1), detection_empty_frames=(1, 3),
            zone_detection_rect=(390, 170, 40, 120))

    def test_empty_frames_show_roi_without_stale_targets_trapezoid_or_crosses(self):
        self._check_main(detection_empty_frames=(0, 1, 2, 3, 4))

    def test_stale_line_preview_settings_cannot_reenable_cv_processing(self):
        with patch.object(config, "DEBUG_ZONE_BLACK_ENABLED", True, create=True), \
                patch.object(config, "DEBUG_ZONE_BLACK_WORK_MAX_SIDE", 192, create=True):
            self._check_main(
                model_labels=["bluesafety", "redsafety", "sqareblue", "sqaregreen", "sqarered", "startzone", "triangualrblack"],
                model_ids=(3, 6, 0), model_scores=(.5, .7, .4),
                expected_class_ids=(5, 6, 1), zone_detection_rect=(0, 0, 576, 448))

    def test_no_opencv_numpy_or_line_module_needed_for_uart_photo_and_screen(self):
        with patch.dict(sys.modules, {"cv2": None, "numpy": None, "debug_zone_black": None}):
            self._check_main(
                model_labels=["bluesafety", "redsafety", "sqareblue", "sqaregreen", "sqarered", "startzone", "triangualrblack"],
                model_ids=(3, 6, 0), model_scores=(.5, .7, .4),
                expected_class_ids=(5, 6, 1), zone_detection_rect=(390, 170, 40, 120))

    def test_original_camera_frames_reach_display_in_both_modes(self):
        for debug_enabled in (False, True):
            with self.subTest(debug=debug_enabled):
                self._check_main(debug_enabled=debug_enabled)

    def _check_main(self, missing_photo_root=False, model_labels=None, model_ids=(5, 7, 6), debug_enabled=True,
                    model_scores=None, expected_class_ids=(5, 6), live_fps=False, detection_empty_frames=(),
                    zone_detection_rect=None, expected_blackbar_centers=(), highlight_bar=False,
                    expected_startzone_centers=None, detection_centers=None,
                    reject_verified=False, toggle_roi=False, serial_mode=serial_module.CMD_SEARCH):
        frames = [PixelImage() for _ in range(5)]
        clock_state = {"us": 0}
        detector = Mock()
        detector.input_width.return_value, detector.input_height.return_value = 576, 448
        detector.labels = model_labels if model_labels is not None else [
            "blue safety", "bluesafety", "redsafety", "sqareblue", "sqarered",
            "sqareredgreen", "startzone", "triangualrblack"]
        detector.detect.return_value = [block(200 + index * 100, 200, model_id)
                                        for index, model_id in enumerate(model_ids)]
        if detection_centers is not None:
            for obj, (x, y) in zip(detector.detect.return_value, detection_centers):
                obj.x, obj.y = x - obj.w * .5, y - obj.h * .5
        if zone_detection_rect is not None:
            for obj in detector.detect.return_value:
                if detector.labels[obj.class_id] in ("bluesafety", "redsafety"):
                    obj.x, obj.y, obj.w, obj.h = zone_detection_rect
        if model_scores is not None:
            for obj, score in zip(detector.detect.return_value, model_scores):
                obj.score = score
        if expected_startzone_centers is None:
            expected_startzone_centers = [(obj.x + obj.w*.5, obj.y + obj.h*.5)
                                         for obj in detector.detect.return_value
                                         if detector.labels[obj.class_id] == "startzone"
                                         and obj.score >= config.DETECT_CONFIDENCE]
        detected_pixels, verified_pixels, serial_pixels = [], [], []
        def detect_frame(img, **kwargs):
            index = len(detected_pixels)
            detected_pixels.append(list(img.samples))
            return [] if index in detection_empty_frames else detector.detect.return_value
        detector.detect.side_effect = detect_frame
        cam = Mock()
        cam.read.side_effect = frames
        if live_fps:
            source_frames = iter(frames)
            def read_frame():
                clock_state["us"] += 200_000
                return next(source_frames)
            cam.read.side_effect = read_frame
        disp, touch = Mock(), Mock()
        touch.read.side_effect = [(10, 10, True), (10, 10, False), (10, 80, True),
                                  (10, 80, False), (0, 0, False)]
        verifier, serial, profiler, media, scanner, capture, menu = [Mock() for _ in range(7)]
        serial.mode = serial_mode
        def verify(img, objs, labels, **kwargs):
            verified_pixels.append(list(img.samples))
            return [] if reject_verified else objs
        verifier.process.side_effect = verify
        verifier.frame_diagnostics.return_value = []
        serial.frame_context.return_value = (0, 1)
        serial.submit.side_effect = lambda objs, labels, **kw: serial_pixels.append(list(frames[len(serial_pixels)].samples)) or (objs[0] if objs else None)
        serial.arrangement_side_objects.return_value = []
        serial.debug_zone_realign_bar.side_effect = AssertionError("DEBUG must not preview46")
        serial.debug_arrangement_objects.side_effect = AssertionError("DEBUG must not preview14")
        if live_fps:
            profiler = performance_debug.PerformanceProfiler()
            with patch.object(performance_debug.time, "ticks_us", return_value=0), patch("builtins.print"):
                profiler.set_enabled(True)
        else:
            profiler.enabled = debug_enabled
            profiler.fps = 12.5
            profiler.begin_frame.return_value = 0
            profiler.mark.return_value = 0
        media.active_protocol, media.recording, media.record_audio_enabled = None, False, False
        scanner.enabled = True
        scanner.process.side_effect = lambda img: img.draw_string(0, 0, "SCAN")
        capture.status = ""
        saved_pixels, saved_draw_calls = [], []
        def save_annotated(img):
            saved_pixels.append(list(img.samples))
            saved_draw_calls.append(list(img.draw_calls))
        capture.save.side_effect = save_annotated
        menu.handle_release.side_effect = [menu_ui.ACTION_PHOTO, menu_ui.ACTION_ROI if toggle_roi else None]
        displayed_samples = []
        menu.draw.side_effect = lambda img, *args, **kwargs: displayed_samples.append(list(img.samples))
        fake_maix = SimpleNamespace(
            app=SimpleNamespace(need_exit=Mock(side_effect=[False] * 5 + [True])),
            camera=SimpleNamespace(Camera=Mock(return_value=cam)),
            display=SimpleNamespace(Display=Mock(return_value=disp)),
            touchscreen=SimpleNamespace(TouchScreen=Mock(return_value=touch)),
            nn=SimpleNamespace(YOLO26=Mock(return_value=detector)),
            image=SimpleNamespace(
                Fit=SimpleNamespace(FIT_CONTAIN=1),
                image2cv=Mock(side_effect=AssertionError("retired OpenCV conversion")),
                cv2image=Mock(side_effect=AssertionError("retired Maix conversion")),
            ),
        )
        fake_menu_module = SimpleNamespace(MenuUI=Mock(return_value=menu))
        for action in ("AUDIO", "DEBUG", "PHOTO", "RECORD", "ROI", "RTSP", "SCAN", "WEBRTC"):
            setattr(fake_menu_module, "ACTION_" + action, getattr(menu_ui, "ACTION_" + action))
        replacements = {"maix": fake_maix, "menu_ui": fake_menu_module,
                        "cv2": None, "numpy": None, "debug_zone_black": None}
        if missing_photo_root:
            replacements["app_config"] = SimpleNamespace(**{
                name: value for name, value in vars(config).items()
                if name != "DATASET_CAPTURE_ROOT"
            })
        for module_name, class_name, instance in (
            ("code_scanner", "CodeScanner", scanner), ("detection_verifier", "DetectionVerifier", verifier),
            ("dataset_capture", "DatasetCapture", capture), ("media_control", "MediaController", media),
            ("performance_debug", "PerformanceProfiler", profiler),
            ("serial_controller", "VisionSerialController", serial),
        ):
            replacements[module_name] = SimpleNamespace(**{class_name: Mock(return_value=instance)})
        replacements["serial_controller"].MODE_FINAL_ROI_TRACK = serial_module.MODE_FINAL_ROI_TRACK
        spec = importlib.util.spec_from_file_location("_main_debug_test", Path(__file__).with_name("main.py"))
        loaded_main = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, replacements), patch.dict(os.environ, {"MAIXVISION": "1"}), \
                patch("os.makedirs"), patch("builtins.print"), \
                patch.object(performance_debug.time, "ticks_us", side_effect=lambda: clock_state["us"]):
            spec.loader.exec_module(loaded_main)
            self.assertEqual(os.environ["MAIXVISION"], "0")
            loaded_main.main()

        fake_maix.nn.YOLO26.assert_called_once_with(
            model=config.MODEL_PATH, dual_buff=config.DETECT_DUAL_BUFFER,
        )
        replacements["serial_controller"].VisionSerialController.assert_called_once_with(
            576, 448, tracking_roi=[0, 319, 576, 129],
            near_view_roi=((208, 102), (368, 102), (575, 307), (575, 407), (0, 407), (0, 307)),
            claw_geometry=True,
        )
        replacements["dataset_capture"].DatasetCapture.assert_called_once_with(
            "/root/sd" if missing_photo_root else config.DATASET_CAPTURE_ROOT
        )

        self.assertEqual(detected_pixels, [SAMPLES] * 5)
        self.assertEqual(verified_pixels, [SAMPLES] * 5)
        self.assertEqual(serial_pixels, [SAMPLES] * 5)
        capture.save.assert_called_once_with(frames[1])
        self.assertEqual(saved_pixels, [displayed_samples[1]])
        if frames[1].draw_calls:
            self.assertNotEqual(saved_pixels, [SAMPLES])
        self.assertEqual(saved_draw_calls, [frames[1].draw_calls])
        verifier.reset_task_history.assert_called_once()
        self.assertEqual(serial.arrangement_side_objects.call_count, 0 if debug_enabled else 5)
        serial.debug_zone_realign_bar.assert_not_called()
        serial.debug_arrangement_objects.assert_not_called()
        self.assertEqual(scanner.process.call_count, 0 if debug_enabled else 5)
        self.assertEqual(disp.show.call_count, 5)
        for index, frame in enumerate(frames):
            frame.binary.assert_not_called()
            frame.get_statistics.assert_not_called()
            frame.find_blobs.assert_not_called()
            frame.get_regression.assert_not_called()
            frame.get_pixel.assert_not_called()
            frame.find_lines.assert_not_called()
            frame.to_bytes.assert_not_called()
            frame.draw_image.assert_not_called()
            frame.format.assert_not_called()
            displayed = disp.show.call_args_list[index].args[0]
            self.assertIs(displayed, frame)
            gray_rects = [args for name, args, kwargs in displayed.draw_calls
                          if name == "rect" and kwargs.get("color") == (128, 128, 128)]
            expected_gray_rects = []
            if debug_enabled and index not in detection_empty_frames:
                for x, y in expected_startzone_centers:
                    left, top = max(0, x-10), max(0, y-10)
                    right, bottom = min(576, x+10), min(448, y+10)
                    if left < right and top < bottom:
                        expected_gray_rects.append((left, top, right-left, bottom-top))
            self.assertEqual(gray_rects, expected_gray_rects)
            self.assertEqual(displayed_samples[index][1:], SAMPLES[1:])
            text = [args[2] for name, args, _ in displayed.draw_calls if name == "text"]
            start_area_calls = [(args, kwargs) for name, args, kwargs in displayed.draw_calls
                                if name == "text" and args[2].startswith("START AREA:")]
            self.assertEqual(start_area_calls, [])
            yellow_rects = [args for name, args, kwargs in displayed.draw_calls
                            if name == "rect" and kwargs.get("color") == overlay.SELECTED_COLOR]
            yellow_lines = [args for name, args, kwargs in displayed.draw_calls
                            if name == "line" and kwargs.get("color") == overlay.SELECTED_COLOR]
            if debug_enabled:
                # Every native YOLO detection is visible, independent of the
                # task lock,14 decision,ROI entry or safety-zone membership.
                published = serial.submit.call_args_list[index]
                drawn_objects = (list(published.kwargs["raw_objects"])
                                 + published.kwargs["blackbar_objects"]
                                 + published.kwargs["startzone_objects"])
                expected_rects, expected_text = [], []
                for obj in drawn_objects:
                    rect = overlay.debug_yolo_rect(obj, 576, 448)
                    if rect is None:
                        continue
                    color = "black" if int(obj.class_id) == 7 else overlay.DEBUG_YOLO_COLORS[int(obj.class_id)]
                    expected_rects.append((tuple(rect), color))
                    source_id = int(getattr(obj, "model_class_id", obj.class_id))
                    expected_text.append("{} {:.2f}".format(detector.labels[source_id], obj.score))
                expected_lines = [((208, 102, 368, 102), overlay.image.COLOR_RED),
                                  ((368, 102, 575, 307), overlay.image.COLOR_RED),
                                  ((575, 307, 575, 407), overlay.image.COLOR_RED),
                                  ((575, 407, 0, 407), overlay.image.COLOR_RED),
                                  ((0, 407, 0, 307), overlay.image.COLOR_RED),
                                  ((0, 307, 208, 102), overlay.image.COLOR_RED)]
                if config.UART_TRACKING_ROI_SHAPE == "rectangle":
                    expected_rects.append(((0, 319, 576, 129), overlay.ROI_COLOR))
                else:
                    expected_lines += [((313, 344, 150, 447), overlay.ROI_COLOR),
                                       ((150, 447, 476, 447), overlay.ROI_COLOR),
                                       ((476, 447, 313, 344), overlay.ROI_COLOR),
                                       ((303, 294, 323, 294), overlay.ROI_COLOR),
                                       ((313, 284, 313, 304), overlay.ROI_COLOR)]
                    red_y = 359 if serial_mode == serial_module.MODE_FINAL_ROI_TRACK else 339
                    expected_lines += [((308, red_y, 318, red_y), overlay.image.COLOR_RED),
                                       ((313, red_y - 5, 313, red_y + 5), overlay.image.COLOR_RED)]
                self.assertEqual([(args, kwargs["color"]) for name, args, kwargs in displayed.draw_calls
                                  if name == "rect"], expected_rects)
                self.assertEqual(text, expected_text)
                self.assertEqual([(args, kwargs["color"]) for name, args, kwargs in displayed.draw_calls
                                  if name == "line"], expected_lines)
                self.assertEqual(yellow_lines, [])
                self.assertEqual(yellow_rects, [])
            else:
                self.assertEqual(yellow_lines, [])
                self.assertEqual(text, ["SCAN"])
                if index not in detection_empty_frames:
                    self.assertEqual(yellow_rects, [(190, 190, 20, 20), (141, 141, 118, 118)])
            expected_fps = (None if index < 3 else 5.0) if live_fps else 12.5
            self.assertEqual(menu.draw.call_args_list[index].kwargs["debug_fps"], expected_fps)
            self.assertIs(menu.draw.call_args_list[index].args[0], displayed)
            published = serial.submit.call_args_list[index]
            frame_ids = [] if index in detection_empty_frames else list(expected_class_ids)
            self.assertEqual([obj.class_id for obj in published.kwargs["raw_objects"]], frame_ids)
            self.assertEqual([obj.class_id for obj in verifier.process.call_args_list[index].args[1]], frame_ids)
            bars = published.kwargs["blackbar_objects"]
            starts = published.kwargs["startzone_objects"]
            self.assertEqual([obj.class_id for obj in starts], [8] * len(expected_startzone_centers)
                             if index not in detection_empty_frames else [])
            self.assertEqual([(obj.x + obj.w*.5, obj.y + obj.h*.5) for obj in starts],
                             list(expected_startzone_centers) if index not in detection_empty_frames else [])
            self.assertEqual([obj.class_id for obj in bars], [7] * len(expected_blackbar_centers)
                             if index not in detection_empty_frames else [])
            self.assertEqual([(obj.x + obj.w * .5, obj.y + obj.h * .5) for obj in bars],
                             list(expected_blackbar_centers) if index not in detection_empty_frames else [])
            for center_x, center_y in expected_blackbar_centers:
                bar_rect = (center_x - 10, center_y - 10, 20, 20)
                drawn = [args for name, args, kwargs in displayed.draw_calls
                         if name == "rect" and args == bar_rect and kwargs.get("color") == "black"]
                self.assertEqual(len(drawn), int(debug_enabled and index not in detection_empty_frames))
            self.assertEqual(detector.detect.call_args_list[index].kwargs["conf_th"], .4)
        fake_maix.image.image2cv.assert_not_called()
        fake_maix.image.cv2image.assert_not_called()
        for resource in (serial, media, touch, disp, cam):
            resource.close.assert_called_once()

    def test_packaging_drops_retired_lab_module_and_keeps_regression_file(self):
        root = Path(__file__).resolve().parent
        manifest = (root / "app.yaml").read_text(encoding="utf8")
        self.assertNotIn("  - debug_lab_preview.py", manifest)
        self.assertIn("  - test_debug_lab_preview.py", manifest)
        self.assertNotIn("debug_lab_preview", (root / "main.py").read_text(encoding="utf8"))


if __name__ == "__main__":
    unittest.main()
