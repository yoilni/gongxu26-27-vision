"""YOLO target, safety-zone, and ROI drawing helpers."""

from maix import image


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
DEPRECATED_LABELS = {"blue safety"}
BOX_THICKNESS = 2
SAFETY_BOX_THICKNESS = 4
SELECTED_BOX_THICKNESS = 4
ROI_COLOR = image.Color.from_rgb(180, 0, 255)
SELECTED_COLOR = image.Color.from_rgb(255, 220, 0)
DEBUG_BOX_COLOR = image.Color.from_rgb(220, 220, 220)
DEBUG_AREA_TEXT_COLOR = image.COLOR_BLACK
DEBUG_AREA_TEXT_SCALE = 1.5
DEBUG_AREA_TEXT_THICKNESS = 1
ARRANGEMENT_SIDE_COLOR = image.COLOR_RED
ARRANGEMENT_SIDE_THICKNESS = 3


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


def draw_debug_detections(img, objects, labels, selected_obj=None):
    """Draw raw YOLO boxes without text, except the highlighted selection."""
    for obj in objects:
        if obj is selected_obj:
            continue

        class_id = int(obj.class_id)
        if class_id < 0 or class_id >= len(labels):
            label = "unknown"
            color = DEBUG_BOX_COLOR
        else:
            label = labels[class_id]
            if label in DEPRECATED_LABELS:
                continue
            color = LABEL_COLORS.get(label, DEBUG_BOX_COLOR)
            safety_style = SAFETY_ZONE_STYLES.get(label)
            if safety_style is not None:
                color = safety_style[1]

        img.draw_rect(
            obj.x,
            obj.y,
            obj.w,
            obj.h,
            color=color,
            thickness=BOX_THICKNESS,
        )


def draw_debug_safety_zone_areas(img, objects, labels, frame_width, frame_height):
    """Show each safety-zone box area and its percentage of the frame."""
    frame_width = int(frame_width)
    frame_height = int(frame_height)
    frame_area = max(1, frame_width * frame_height)

    for obj in objects:
        class_id = int(obj.class_id)
        if class_id < 0 or class_id >= len(labels):
            continue

        style = SAFETY_ZONE_STYLES.get(labels[class_id])
        if style is None:
            continue

        left = max(0, int(obj.x))
        top = max(0, int(obj.y))
        right = min(frame_width, int(obj.x + obj.w))
        bottom = min(frame_height, int(obj.y + obj.h))
        width = max(0, right - left)
        height = max(0, bottom - top)
        area = width * height
        percentage = area * 100.0 / frame_area

        text = "A={} {:.1f}%".format(area, percentage)
        text_y = max(0, min(frame_height - 28, top))
        img.draw_string(
            left,
            text_y,
            text,
            color=DEBUG_AREA_TEXT_COLOR,
            scale=DEBUG_AREA_TEXT_SCALE,
            thickness=DEBUG_AREA_TEXT_THICKNESS,
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
