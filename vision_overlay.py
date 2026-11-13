"""YOLO target, safety-zone, and ROI drawing helpers."""

from math import isfinite

from maix import image

import app_config as config
from target_selector import (
    SELECTED_OUTER_X_MAX_PX,
    SELECTED_OUTER_X_MAX_Y,
    SELECTED_OUTER_X_MIN_PX,
    clipped_yolo_rect,
    selected_target_rectangles,
    selected_target_x_padding,
    target_center_error,
)
from zone_obstacle_region import zone_obstacle_polygon
from tracking_roi import (
    final_track_reticle_point, point_in_tracking_roi, roi_reticle_point, roi_triangle_vertices,
    search_reticle_point,
)


LABEL_COLORS = {
    "sqareblue": image.COLOR_BLUE,
    "sqarered": image.COLOR_RED,
    "sqareredgreen": image.COLOR_GREEN,
    "triangualrblack": image.COLOR_BLACK,
}

# New model safety-zone classes. Class ID 0 ("blue safety") is deprecated and
# intentionally ignored; only IDs 1 and 2 are displayed.
SAFETY_ZONE_STYLES = {
    "bluesafety": ("BLUE SAFE", image.Color.from_rgb(0, 160, 255)),
    "redsafety": ("RED SAFE", image.Color.from_rgb(255, 40, 40)),
}
BOX_THICKNESS = 2
SAFETY_BOX_THICKNESS = 4
SELECTED_BOX_THICKNESS = 4
ROI_COLOR = image.Color.from_rgb(180, 0, 255)
SELECTED_COLOR = image.Color.from_rgb(255, 220, 0)
ARRANGEMENT_SIDE_COLOR = image.COLOR_RED
ARRANGEMENT_SIDE_THICKNESS = 3
DEBUG_FINAL_CANDIDATE_COLOR = image.Color.from_rgb(175, 0, 235)
DEBUG_FINAL_NEIGHBOR_COLOR = image.Color.from_rgb(255, 255, 255)
DEBUG_FINAL_BOX_THICKNESS = 3
DEBUG_TARGET_Y_COLOR = image.COLOR_RED
DEBUG_TARGET_Y_TOP = 36
DEBUG_YOLO_COLORS = {
    0: image.COLOR_BLUE,
    1: image.COLOR_BLUE,
    2: image.COLOR_RED,
    3: image.Color.from_rgb(0, 191, 255),
    4: image.COLOR_RED,
    5: image.COLOR_GREEN,
    6: image.COLOR_BLACK,
    7: SELECTED_COLOR,  # Auxiliary blackbar: yellow unless46 qualifies.
    8: image.Color.from_rgb(128, 128, 128),  # DEBUG-only startzone, no UART ID.
}


def make_bottom_center_roi(frame_width, frame_height):
    """Create a full-width bottom ROI, extended upward by 40 pixels."""
    roi_width = frame_width
    roi_height = min(frame_height, frame_height // 5 + 40)
    return [
        0,
        frame_height - roi_height,
        roi_width,
        roi_height,
    ]


def point_in_rect(x, y, rect):
    left, top, width, height = rect
    return left <= x < left + width and top <= y < top + height


def object_center_in_roi(obj, roi_rect):
    center_x = float(obj.x) + float(obj.w) * .5
    center_y = float(obj.y) + float(obj.h) * .5
    return point_in_tracking_roi(center_x, center_y, roi_rect)


def debug_yolo_rect(obj, frame_width, frame_height):
    """Clip the original YOLO rectangle to the image, without outer padding."""
    return clipped_yolo_rect(obj, frame_width, frame_height)


def draw_debug_yolo_detections(img, objects, frame_width, frame_height, realign_bar=None,
                               arrangement_preview=None, labels=None):
    """Native class-color boxes and model labels/scores; legacy14 preview optional."""
    preview_target, _, neighbors = arrangement_preview if arrangement_preview is not None else (None, None, [])
    blocking_ids = {id(obj) for obj in neighbors}
    for obj in objects:
        class_id = int(obj.class_id)
        if (arrangement_preview is not None and class_id in (3, 4, 5, 6)
                and obj is not preview_target and id(obj) not in blocking_ids):
            continue
        rect = debug_yolo_rect(obj, frame_width, frame_height)
        if rect is None:
            continue
        color = (image.COLOR_BLACK if labels is not None and class_id == 7 else
                 image.COLOR_RED if class_id == 7 and obj is realign_bar
                 else DEBUG_YOLO_COLORS.get(class_id, image.COLOR_GREEN))
        thickness = 3
        if arrangement_preview is not None and class_id in (3, 4, 5, 6):
            # Red exclusively means a blocker, not merely a red cargo class.
            color = image.COLOR_RED if id(obj) in blocking_ids else SELECTED_COLOR
            if obj is preview_target or id(obj) in blocking_ids:
                thickness = SELECTED_BOX_THICKNESS
        img.draw_rect(
            *rect,
            color=color,
            thickness=thickness,
        )
        if obj is preview_target and class_id in (3, 4, 5, 6):
            _, outer = selected_target_rectangles(obj, frame_width, frame_height)
            if outer is not None:
                img.draw_rect(*outer, color=SELECTED_COLOR, thickness=SELECTED_BOX_THICKNESS)
        if labels is not None:
            model_id = int(getattr(obj, "model_class_id", class_id))
            if not 0 <= model_id < len(labels):
                continue
            score = float(getattr(obj, "score", 0.0))
            score = min(1.0, max(0.0, score)) if isfinite(score) else 0.0
            text = "{} {:.2f}".format(labels[model_id], score)
            scale = 1.2
            try:
                size = image.string_size(text)
                text_width = int(size.width() * scale + .5)
                text_height = int(size.height() * scale + .5)
            except (AttributeError, TypeError):
                text_width, text_height = int(len(text) * 8 * scale), 20
            text_x = max(0, min(rect[0], int(frame_width) - text_width))
            text_y = rect[1] - text_height if rect[1] >= text_height else rect[1] + 2
            text_y = max(0, min(text_y, int(frame_height) - text_height))
            img.draw_string(text_x, text_y, text, color=color, scale=scale, thickness=1)


def draw_debug_object_area(img, objects, frame_width, frame_height):
    """Visible box/image percentages for cargo and preview-only startzone."""
    if frame_width <= 0 or frame_height <= 0:
        return
    frame_area = float(frame_width) * float(frame_height)
    for obj in objects:
        class_id = int(obj.class_id)
        if class_id not in (3, 4, 5, 6, 8):
            continue
        rect = debug_yolo_rect(obj, frame_width, frame_height)
        if rect is None:
            continue
        left = max(0.0, float(obj.x))
        top = max(0.0, float(obj.y))
        right = min(float(frame_width), float(obj.x) + float(obj.w))
        bottom = min(float(frame_height), float(obj.y) + float(obj.h))
        percent = (right - left) * (bottom - top) * 100.0 / frame_area
        # Keep the short ASCII label inside the frame at top/right edges.
        text_x = max(0, min(rect[0], int(frame_width) - (224 if class_id == 8 else 136)))
        text_y = rect[1] - 22 if rect[1] >= 22 else rect[1] + 2
        if class_id == 8:
            # Large startzone boxes often begin atY0: stay below PHOTO/BACK
            # and the FPS row rather than hiding the percentage under them.
            text_y = max(84, text_y)
        text_y = max(0, min(text_y, int(frame_height) - 20))
        img.draw_string(
            text_x, text_y,
            ("START AREA: {:.1f}%" if class_id == 8 else "AREA: {:.1f}%").format(percent),
            color=image.COLOR_BLACK, scale=1.2, thickness=1,
        )


def draw_target_borders(img, objects, labels, roi_enabled, roi_rect):
    """Draw useful targets as borders only; do not draw labels or confidence."""
    for obj in objects:
        label = labels[obj.class_id]
        color = LABEL_COLORS.get(label)
        if color is None:
            continue

        if roi_enabled and object_center_in_roi(obj, roi_rect):
            color = ROI_COLOR

        img.draw_rect(
            obj.x,
            obj.y,
            obj.w,
            obj.h,
            color=color,
            thickness=BOX_THICKNESS,
        )


def draw_selected_target(img, obj, labels, frame_width=None, frame_height=None):
    """Normal mode: native cargo box plus a center-Y-scaled XY outer box."""
    if obj is None:
        return

    class_id = int(obj.class_id)
    if class_id < 0 or class_id >= len(labels):
        return
    label = labels[class_id]
    if label not in LABEL_COLORS and label not in SAFETY_ZONE_STYLES:
        return

    if label in SAFETY_ZONE_STYLES:
        # The padding is for comparing movable neighbors, not safety zones.
        img.draw_rect(obj.x, obj.y, obj.w, obj.h,
                      color=SELECTED_COLOR, thickness=SELECTED_BOX_THICKNESS)
        return

    # Cached detector dimensions avoid native image calls in the current main;
    # keep older three-argument callers compatible after a partial deployment.
    if frame_width is None:
        frame_width = img.width()
    if frame_height is None:
        frame_height = img.height()
    rect, outer = selected_target_rectangles(obj, frame_width, frame_height)
    if rect is None or outer is None:
        return
    img.draw_rect(*rect, color=SELECTED_COLOR, thickness=SELECTED_BOX_THICKNESS)
    img.draw_rect(*outer, color=SELECTED_COLOR, thickness=SELECTED_BOX_THICKNESS)


def draw_debug_final_candidates(img, candidates, neighbors, selected_obj):
    """Show fallback target inputs in purple and all compared boxes in white."""
    for obj in candidates:
        if obj is selected_obj:
            continue
        img.draw_rect(
            int(obj.x), int(obj.y), int(obj.w), int(obj.h),
            color=DEBUG_FINAL_CANDIDATE_COLOR,
            thickness=DEBUG_FINAL_BOX_THICKNESS,
        )
    # Draw neighbors last: a target can also be another target's neighbor,
    # and every close box must remain visibly white in DEBUG.
    for obj in neighbors:
        img.draw_rect(
            int(obj.x), int(obj.y), int(obj.w), int(obj.h),
            color=DEBUG_FINAL_NEIGHBOR_COLOR,
            thickness=DEBUG_FINAL_BOX_THICKNESS,
        )


def draw_debug_target_y(img, selected_obj, frame_width):
    """Show the highlighted object's absolute center-Y in the upper-right."""
    if selected_obj is None:
        return
    center_y = int(round(float(selected_obj.y) + float(selected_obj.h) * 0.5))
    img.draw_string(
        max(0, int(frame_width) - 112),
        DEBUG_TARGET_Y_TOP,
        "Y={}".format(center_y),
        color=DEBUG_TARGET_Y_COLOR,
        scale=1.5,
        thickness=1,
    )


def draw_debug_neighbor_dx(img, dx_annotations, frame_width, frame_height):
    """Show center-X gaps next to the white fallback-neighbor boxes."""
    for obj, dx in dx_annotations:
        text_x = max(0, min(int(frame_width) - 105, int(obj.x)))
        text_y = max(0, min(int(frame_height) - 20, int(obj.y) - 19))
        img.draw_string(
            text_x,
            text_y,
            "dx={}".format(dx),
            color=DEBUG_FINAL_NEIGHBOR_COLOR,
            scale=1.2,
            thickness=1,
        )


def draw_debug_candidate_y(
    img, candidates, selected_obj, frame_width, frame_height
):
    """Show absolute center-Y beside each purple fallback target box."""
    for obj in candidates:
        if obj is selected_obj:
            continue
        center_y = int(round(float(obj.y) + float(obj.h) * 0.5))
        text_x = max(0, min(int(frame_width) - 100, int(obj.x)))
        text_y = max(0, min(int(frame_height) - 20, int(obj.y + obj.h) + 2))
        img.draw_string(
            text_x,
            text_y,
            "Y={}".format(center_y),
            color=DEBUG_FINAL_CANDIDATE_COLOR,
            scale=1.2,
            thickness=1,
        )


def draw_arrangement_side_targets(img, objects):
    """Mark objects accepted as left/right neighbors of the middle object."""
    for obj in objects:
        img.draw_rect(
            int(obj.x),
            int(obj.y),
            int(obj.w),
            int(obj.h),
            color=ARRANGEMENT_SIDE_COLOR,
            thickness=ARRANGEMENT_SIDE_THICKNESS,
        )


def draw_safety_zones(img, objects, labels):
    """Normal mode: draw zone borders only, without names or percentages."""
    for obj in objects:
        class_id = int(obj.class_id)
        if class_id < 0 or class_id >= len(labels):
            continue

        style = SAFETY_ZONE_STYLES.get(labels[class_id])
        if style is None:
            continue

        img.draw_rect(
            obj.x,
            obj.y,
            obj.w,
            obj.h,
            color=style[1],
            thickness=SAFETY_BOX_THICKNESS,
        )


def draw_debug_safety_zone_area(img, objects, labels, frame_width, frame_height, roi_rect=None):
    """Black DEBUG area and signed zone-center Y error to the UART ROI center."""
    reference_y = None
    if (roi_rect is not None and len(roi_rect) == 4
            and all(isfinite(value) for value in roi_rect)
            and roi_rect[2] > 0 and roi_rect[3] > 0):
        reference_y = float(roi_rect[1]) + float(roi_rect[3]) * .5
    row = 0
    for obj in objects:
        class_id = int(obj.class_id)
        if not 0 < class_id < len(labels):
            continue
        style = SAFETY_ZONE_STYLES.get(labels[class_id])
        if style is None:
            continue
        left = max(0.0, min(float(frame_width), float(obj.x)))
        top = max(0.0, min(float(frame_height), float(obj.y)))
        right = max(0.0, min(float(frame_width), float(obj.x) + float(obj.w)))
        bottom = max(0.0, min(float(frame_height), float(obj.y) + float(obj.h)))
        area = max(0.0, right - left) * max(0.0, bottom - top)
        percent = area * 100.0 / max(1, frame_width * frame_height)
        text = "{} AREA: {:.1f}%".format(style[0], percent)
        if reference_y is not None:
            _, error_y = target_center_error(
                obj, frame_width, frame_height, reference_y=reference_y,
            )
            text += " dY:{:+d}px".format(error_y)
        img.draw_string(
            # Keep the combined line below PHOTO/BACK and the FPS row.
            8, (84 if reference_y is not None else 36) + row * 24,
            text,
            color=image.COLOR_BLACK, scale=1.5, thickness=1,
        )
        row += 1


def draw_debug_frame_center(img, frame_width, frame_height):
    """Mark the image center with a white cross outlined in black."""
    center_x = int(frame_width) // 2
    center_y = int(frame_height) // 2
    for color, half_length, half_width in (
        (image.COLOR_BLACK, 12, 3),
        (DEBUG_FINAL_NEIGHBOR_COLOR, 10, 1),
    ):
        img.draw_rect(
            center_x - half_length, center_y - half_width,
            half_length * 2 + 1, half_width * 2 + 1,
            color=color, thickness=half_width * 2 + 1,
        )
        img.draw_rect(
            center_x - half_width, center_y - half_length,
            half_width * 2 + 1, half_length * 2 + 1,
            color=color, thickness=half_width * 2 + 1,
        )


def _draw_roi_polygon(img, vertices, color):
    for index, start in enumerate(vertices):
        end = vertices[(index + 1) % len(vertices)]
        img.draw_line(int(round(start[0])), int(round(start[1])),
                      int(round(end[0])), int(round(end[1])),
                      color=color, thickness=2)


def draw_roi_border(img, roi_rect, show_reticle=False, near_view_roi=None, final_track=False):
    """Draw the tracking ROI, and show the 05 ROI and reticles in DEBUG."""
    if show_reticle and near_view_roi is not None:
        _draw_roi_polygon(img, near_view_roi, image.COLOR_RED)
    if getattr(config, "UART_TRACKING_ROI_SHAPE", "triangle") == "rectangle":
        if roi_rect is not None:
            img.draw_rect(*roi_rect, color=ROI_COLOR, thickness=2)
        return
    vertices = roi_triangle_vertices(roi_rect)
    if vertices is None:
        return
    _draw_roi_polygon(img, vertices, ROI_COLOR)
    if show_reticle:
        apex = vertices[0]
        red_point = final_track_reticle_point(apex) if final_track else search_reticle_point(apex)
        for (x, y), color, radius in (
            (roi_reticle_point(apex), ROI_COLOR, 10),
            (red_point, image.COLOR_RED, 5),
        ):
            img.draw_line(x - radius, y, x + radius, y, color=color, thickness=2)
            img.draw_line(x, y - radius, x, y + radius, color=color, thickness=2)


def draw_debug_zone_realign_point(img, roi_rect, frame_width, frame_height):
    """Preview46's proposed point only; never change the UART reference."""
    if (roi_rect is None or len(roi_rect) != 4
            or frame_width <= 0 or frame_height <= 0):
        return
    left, top, width, height = roi_rect
    if (not all(isfinite(value) for value in roi_rect)
            or width <= 0 or height <= 0):
        return
    point_x = left + width * 0.5 + getattr(config, "DEBUG_ZONE_REALIGN_POINT_OFFSET_X", 165)
    point_y = top + height * 0.5 + getattr(config, "DEBUG_ZONE_REALIGN_POINT_OFFSET_Y", -90)
    if (not isfinite(point_x) or not isfinite(point_y)
            or not 0 <= point_x < frame_width or not 0 <= point_y < frame_height):
        return
    # Round only for rendering; keep the half-pixel ROI center in the geometry.
    center_x = min(int(frame_width) - 1, int(round(point_x)))
    center_y = min(int(frame_height) - 1, int(round(point_y)))
    for color, half_length, thickness in (
        (image.COLOR_BLACK, 10, 5),
        (SELECTED_COLOR, 8, 3),
    ):
        img.draw_line(
            max(0, center_x - half_length), center_y,
            min(int(frame_width) - 1, center_x + half_length), center_y,
            color=color, thickness=thickness,
        )
        img.draw_line(
            center_x, max(0, center_y - half_length),
            center_x, min(int(frame_height) - 1, center_y + half_length),
            color=color, thickness=thickness,
        )
    img.draw_string(
        max(0, min(center_x + 12, int(frame_width) - 28)),
        max(0, min(center_y - 20, int(frame_height) - 20)),
        "46", color=image.COLOR_BLACK, scale=1.2, thickness=1,
    )


def draw_debug_zone_obstacle_region(img, roi_rect, frame_width, frame_height):
    """Outline the exact 06/16 obstacle-center trapezoid in red."""
    polygon = zone_obstacle_polygon(roi_rect, frame_width, frame_height)
    if polygon is None:
        return
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        img.draw_line(
            int(round(start[0])), int(round(start[1])),
            int(round(end[0])), int(round(end[1])),
            color=image.COLOR_RED, thickness=2,
        )
