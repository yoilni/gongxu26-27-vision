"""MaixCAM YOLO26 application entry point."""

import os

# Disable IDE image transport before Maix initializes the display channel.
# Local screen output through disp.show() remains enabled.
os.environ["MAIXVISION"] = "0"

from maix import app, camera, display, image, nn, touchscreen

import app_config as config
from code_scanner import CodeScanner
from detection_confidence import DetectionConfidenceFilter
from detection_verifier import DetectionVerifier
from dataset_capture import DatasetCapture
from media_control import MediaController
from model_class_mapping import ModelClassMapper
from menu_ui import (
    ACTION_AUDIO,
    ACTION_DEBUG,
    ACTION_PHOTO,
    ACTION_RECORD,
    ACTION_ROI,
    ACTION_RTSP,
    ACTION_SCAN,
    ACTION_WEBRTC,
    MenuUI,
)
from performance_debug import PerformanceProfiler
from serial_controller import MODE_FINAL_ROI_TRACK, VisionSerialController
from tracking_roi import near_view_roi_vertices
from vision_overlay import (
    draw_arrangement_side_targets,
    draw_debug_yolo_detections,
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

    detector = nn.YOLO26(
        model=config.MODEL_PATH,
        dual_buff=getattr(config, "DETECT_DUAL_BUFFER", True),
    )
    frame_width = detector.input_width()
    frame_height = detector.input_height()
    model_labels = detector.labels
    class_mapper = ModelClassMapper(model_labels)
    labels = class_mapper.labels
    print("Vision model_classes=" + class_mapper.summary() + " black_uart_id=6")
    confidence_filter = DetectionConfidenceFilter(
        config.DETECT_CONFIDENCE,
        getattr(config, "DETECT_BLACK_CONFIDENCE", 0.7),
        getattr(config, "DETECT_SAFETY_ZONE_CONFIDENCE", 0.4),
    )
    print("Vision confidence default={} black={} safety={} detector_floor={}".format(
        config.DETECT_CONFIDENCE,
        getattr(config, "DETECT_BLACK_CONFIDENCE", 0.7),
        getattr(config, "DETECT_SAFETY_ZONE_CONFIDENCE", 0.4),
        confidence_filter.detector_threshold,
    ))
    cam = configure_camera(detector)
    disp = display.Display()
    touch = touchscreen.TouchScreen()

    menu = MenuUI(frame_width, frame_height, disp)
    media = MediaController(cam)
    scanner = CodeScanner(frame_width, frame_height)
    verifier = DetectionVerifier(
        frame_width,
        frame_height,
    )
    profiler = PerformanceProfiler()
    roi_rect = make_bottom_center_roi(
        frame_width,
        frame_height,
    )
    near_view_roi = near_view_roi_vertices(frame_width, frame_height)
    serial_controller = VisionSerialController(
        frame_width,
        frame_height,
        tracking_roi=roi_rect,
        near_view_roi=near_view_roi,
        claw_geometry=True,
    )
    print("Vision DEBUG preview=all_class_color_boxes_labels_confidence_v1 photo_annotated_v1 roi_base_inset_v1 no_hud_v1 zone_line_preview_removed_v8")

    roi_enabled = False
    last_pressed = False
    dataset_capture = DatasetCapture(
        getattr(config, "DATASET_CAPTURE_ROOT", "/root/sd")
    )
    verifier_task_reset_epoch = 0

    try:
        while not app.need_exit():
            frame_start_us = profiler.begin_frame()
            stage_start_us = frame_start_us

            frame_generation, task_reset_epoch = serial_controller.frame_context()
            if task_reset_epoch != verifier_task_reset_epoch:
                # Only FE clears confirmation history, not normal stage ACKs.
                verifier.reset_task_history()
                verifier_task_reset_epoch = task_reset_epoch
            img = cam.read()
            stage_start_us = profiler.mark("camera", stage_start_us)

            objects = detector.detect(
                img,
                conf_th=confidence_filter.detector_threshold,
                iou_th=config.DETECT_IOU,
            )
            blackbar_objects = confidence_filter.apply_blackbars(class_mapper.blackbars(objects))
            # Independent auxiliary channel: normal03/04/24 needs startzone
            # for AE, but class8 never enters cargo verification/counting.
            startzone_objects = confidence_filter.apply_startzones(class_mapper.startzones(objects))
            # Normalize by model LABEL before every consumer. New model0 is
            # bluesafety,3 is sqaregreen,5 is ignored startzone; old layouts
            # remain supported without leaking model indices into UART IDs.
            objects = class_mapper.normalize(objects)
            objects = confidence_filter.apply(objects)
            stage_start_us = profiler.mark("yolo", stage_start_us)

            verified_objects = verifier.process(
                img,
                objects,
                labels,
                debug_enabled=profiler.enabled,
            )
            stage_start_us = profiler.mark("verify", stage_start_us)

            selected_target = serial_controller.submit(
                verified_objects,
                labels,
                raw_objects=objects,
                verification_diagnostics=verifier.frame_diagnostics(),
                frame_generation=frame_generation,
                blackbar_objects=blackbar_objects,
                startzone_objects=startzone_objects,
            )
            arrangement_side_objects = [] if profiler.enabled else (
                serial_controller.arrangement_side_objects(
                    verified_objects,
                    labels,
                    raw_objects=objects,
                )
            )
            stage_start_us = profiler.mark("serial", stage_start_us)

            if not profiler.enabled:
                scanner.process(img)
            stage_start_us = profiler.mark("scan", stage_start_us)

            if profiler.enabled:
                draw_debug_yolo_detections(
                    img, list(objects) + blackbar_objects + startzone_objects,
                    frame_width, frame_height, labels=model_labels,
                )
            elif not profiler.enabled:
                draw_safety_zones(img, objects, labels)
                draw_selected_target(
                    img, selected_target, labels, frame_width, frame_height
                )
            # Keep every accepted non-middle object red in ARRANGE-02, including
            # the currently selected side object (do not let yellow cover it).
            if not profiler.enabled:
                draw_arrangement_side_targets(img, arrangement_side_objects)
            if roi_enabled or profiler.enabled:
                draw_roi_border(img, roi_rect, show_reticle=profiler.enabled,
                                near_view_roi=near_view_roi,
                                final_track=serial_controller.mode == MODE_FINAL_ROI_TRACK)
            stage_start_us = profiler.mark("draw", stage_start_us)

            touch_x, touch_y, pressed = touch.read()
            released = last_pressed != pressed and not pressed
            requested_action = None
            if released:
                requested_action = menu.handle_release(touch_x, touch_y, profiler.enabled)
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
                elif requested_action == ACTION_PHOTO:
                    # Save this annotated frame, before buttons/status cover it.
                    dataset_capture.save(img)
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
                dataset_capture.status,
                debug_fps=profiler.fps,
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
