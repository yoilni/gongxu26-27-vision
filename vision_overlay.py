"""YOLO target, safety-zone, and ROI drawing helpers."""

from maix import image

from zone_obstacle_region import zone_obstacle_polygon


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
DEBUG_YOLO_PADDING = 4


def make_bottom_center_roi(frame_width, frame_height):
    """Create a full-width ROI covering the bottom fifth of the image."""
    roi_width = frame_width
    roi_height = frame_height // 5
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
    center_x = obj.x + obj.w // 2
    center_y = obj.y + obj.h // 2
    return point_in_rect(center_x, center_y, roi_rect)


def draw_debug_yolo_detections(img, objects, frame_width, frame_height):
    """Draw all raw YOLO boxes as thin green outer borders, with no labels.

    Padding keeps raw detection evidence visible beside yellow/purple/white
    screening borders; no class, verification or safety-zone filter applies.
    """
    for obj in objects:
        if float(obj.w) <= 0 or float(obj.h) <= 0:
            continue
        left = max(0, int(obj.x) - DEBUG_YOLO_PADDING)
        top = max(0, int(obj.y) - DEBUG_YOLO_PADDING)
        right = min(int(frame_width) - 1, int(obj.x + obj.w) + DEBUG_YOLO_PADDING - 1)
        bottom = min(int(frame_height) - 1, int(obj.y + obj.h) + DEBUG_YOLO_PADDING - 1)
        if left > right or top > bottom:
            continue
        img.draw_rect(
            left, top, right - left + 1, bottom - top + 1,
            color=image.COLOR_GREEN,
            thickness=1,
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


def draw_selected_target(img, obj, labels):
    """Always mark the currently tracked target with a yellow border."""
    if obj is None:
        return

    class_id = int(obj.class_id)
    if class_id < 0 or class_id >= len(labels):
        return
    label = labels[class_id]
    if label not in LABEL_COLORS and label not in SAFETY_ZONE_STYLES:
        return

    img.draw_rect(
        obj.x,
        obj.y,
        obj.w,
        obj.h,
        color=SELECTED_COLOR,
        thickness=SELECTED_BOX_THICKNESS,
    )


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
    """Draw active safety-zone classes from the raw YOLO detections."""
    for obj in objects:
        class_id = int(obj.class_id)
        if class_id < 0 or class_id >= len(labels):
            continue

        style = SAFETY_ZONE_STYLES.get(labels[class_id])
        if style is None:
            continue

        title, color = style
        img.draw_rect(
            obj.x,
            obj.y,
            obj.w,
            obj.h,
            color=color,
            thickness=SAFETY_BOX_THICKNESS,
        )

        score = max(0.0, min(1.0, float(getattr(obj, "score", 0.0))))
        text = "{} {}%".format(title, int(score * 100.0 + 0.5))
        text_y = max(0, int(obj.y) - 18)
        img.draw_string(
            max(0, int(obj.x)),
            text_y,
            text,
            color=color,
            scale=1.0,
            thickness=2,
        )


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


def draw_roi_border(img, roi_rect):
    """Draw the ROI border without additional text."""
    img.draw_rect(
        roi_rect[0],
        roi_rect[1],
        roi_rect[2],
        roi_rect[3],
        color=ROI_COLOR,
        thickness=2,
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
