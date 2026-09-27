"""MaixCAM YOLO26 application entry point."""

import os

from maix import app, camera, display, image, nn, touchscreen

import app_config as config
from code_scanner import CodeScanner
from detection_verifier import DetectionVerifier
from media_control import MediaController
from menu_ui import (
    ACTION_AUDIO,
    ACTION_DEBUG,
    ACTION_RECORD,
    ACTION_ROI,
    ACTION_RTSP,
    ACTION_SCAN,
    ACTION_WEBRTC,
    MenuUI,
)
from performance_debug import PerformanceProfiler
from serial_controller import VisionSerialController
from vision_overlay import (
    draw_arrangement_side_targets,
    draw_debug_adjacent_coordinate_gaps,
    draw_debug_detections,
    draw_debug_safety_zone_areas,
    draw_roi_border,
    draw_safety_zones,
    draw_selected_target,
    make_bottom_center_roi,
)


def configure_camera(detector):
    cam = camera.Camera(
        detector.input_width(),
        detector.input_height(),
        detector.input_format(),
    )
    if not config.AUTO_AWB:
        cam.awb_mode(camera.AwbMode.Manual)
        cam.set_wb_gain(config.AWB_GAIN)
    cam.constrast(config.CONTRAST)  # MaixPy API method name is constrast.
    cam.skip_frames(config.CAMERA_STABILIZE_FRAMES)
    return cam


def main():
    os.makedirs(config.RECORD_DIR, exist_ok=True)

    detector = nn.YOLO26(model=config.MODEL_PATH, dual_buff=False)
    cam = configure_camera(detector)
    disp = display.Display()
    touch = touchscreen.TouchScreen()

    menu = MenuUI(detector.input_width(), detector.input_height(), disp)
    media = MediaController(cam)
    scanner = CodeScanner(detector.input_width(), detector.input_height())
    verifier = DetectionVerifier(
        detector.input_width(),
        detector.input_height(),
    )
    profiler = PerformanceProfiler()
    roi_rect = make_bottom_center_roi(
        detector.input_width(),
        detector.input_height(),
    )
    serial_controller = VisionSerialController(
        detector.input_width(),
        detector.input_height(),
        tracking_roi=roi_rect,
    )

    roi_enabled = False
    last_pressed = False

    try:
        while not app.need_exit():
            frame_start_us = profiler.begin_frame()
            stage_start_us = frame_start_us

            img = cam.read()
            stage_start_us = profiler.mark("camera", stage_start_us)

            objects = detector.detect(
                img,
                conf_th=config.DETECT_CONFIDENCE,
                iou_th=config.DETECT_IOU,
            )
            stage_start_us = profiler.mark("yolo", stage_start_us)

            verified_objects = verifier.process(
                img,
                objects,
                detector.labels,
                debug_enabled=profiler.enabled,
            )
            stage_start_us = profiler.mark("verify", stage_start_us)

            selected_target = serial_controller.submit(
                verified_objects,
                detector.labels,
                raw_objects=objects,
            )
            arrangement_side_objects = (
                serial_controller.arrangement_side_objects(
                    verified_objects,
                    detector.labels,
                    raw_objects=objects,
                )
            )
            stage_start_us = profiler.mark("serial", stage_start_us)

            scanner.process(img)
            stage_start_us = profiler.mark("scan", stage_start_us)

            # Safety zones do not need the color/temporal verifier used by the
            # four movable targets, so draw them from the raw YOLO results.
            if profiler.enabled:
                draw_debug_detections(
                    img,
                    objects,
                    detector.labels,
                    selected_obj=selected_target,
                )
                draw_debug_safety_zone_areas(
                    img,
                    objects,
                    detector.labels,
                    detector.input_width(),
                    detector.input_height(),
                )
                draw_debug_adjacent_coordinate_gaps(
                    img,
                    verified_objects,
                    detector.labels,
                    detector.input_width(),
                    detector.input_height(),
                )
            else:
                draw_safety_zones(img, objects, detector.labels)
            draw_selected_target(img, selected_target, detector.labels)
            # Keep every accepted non-middle object red in ARRANGE-02, including
            # the currently selected side object (do not let yellow cover it).
            draw_arrangement_side_targets(img, arrangement_side_objects)
            if roi_enabled or profiler.enabled:
                draw_roi_border(img, roi_rect)
            stage_start_us = profiler.mark("draw", stage_start_us)

            touch_x, touch_y, pressed = touch.read()
            released = last_pressed != pressed and not pressed
            requested_action = None
            if released:
                requested_action = menu.handle_release(touch_x, touch_y)
                if requested_action == ACTION_AUDIO:
                    media.toggle_audio_option()
                    requested_action = None
                elif requested_action == ACTION_ROI:
                    roi_enabled = not roi_enabled
                    print("ROI:", "enabled" if roi_enabled else "disabled")
                    requested_action = None
                elif requested_action == ACTION_SCAN:
                    scanner.toggle()
                    requested_action = None
                elif requested_action == ACTION_DEBUG:
                    profiler.toggle()
                    requested_action = None
            last_pressed = pressed

            menu.draw(
                img,
                media.active_protocol,
                media.recording,
                media.record_audio_enabled,
                roi_enabled,
                scanner.enabled,
                profiler.enabled,
            )
            stage_start_us = profiler.mark("ui", stage_start_us)

            disp.show(img, fit=image.Fit.FIT_CONTAIN)
            stage_start_us = profiler.mark("display", stage_start_us)

            # Release the primary RGB frame before adding/removing camera and
            # hardware-encoder channels.
            del img

            if requested_action == ACTION_RECORD:
                media.toggle_recording()
            elif requested_action == ACTION_RTSP:
                media.toggle_stream(config.PROTOCOL_RTSP)
            elif requested_action == ACTION_WEBRTC:
                media.toggle_stream(config.PROTOCOL_WEBRTC)
            stage_start_us = profiler.mark("media", stage_start_us)
            profiler.end_frame(frame_start_us)
    finally:
        serial_controller.close()
        media.close()
        touch.close()
        disp.close()
        cam.close()


if __name__ == "__main__":
    main()
