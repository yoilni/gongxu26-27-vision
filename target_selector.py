"""Select the movable target closest to a requested image reference."""

import math


SELECTED_OUTER_X_MIN_PX = 5
SELECTED_OUTER_X_MAX_PX = 75
SELECTED_OUTER_X_MAX_Y = 319


def clipped_yolo_rect(obj, frame_width, frame_height):
    """Original visible pixel rectangle; shared by display and overlap checks."""
    x, y, width, height = (float(obj.x), float(obj.y), float(obj.w), float(obj.h))
    if not all(math.isfinite(value) for value in (x, y, width, height)):
        return None
    if width <= 0 or height <= 0:
        return None
    left, top = max(0, math.floor(x)), max(0, math.floor(y))
    right = min(int(frame_width), math.ceil(x + width))
    bottom = min(int(frame_height), math.ceil(y + height))
    if left >= right or top >= bottom:
        return None
    return [left, top, right - left, bottom - top]


def selected_target_x_padding(center_y):
    """Per-side XY margin (legacy name):5px atY0,75px atY319, clamped."""
    y = max(0.0, min(float(center_y), float(SELECTED_OUTER_X_MAX_Y)))
    padding = SELECTED_OUTER_X_MIN_PX + (
        SELECTED_OUTER_X_MAX_PX - SELECTED_OUTER_X_MIN_PX
    ) * y / SELECTED_OUTER_X_MAX_Y
    return int(padding + 0.5)


def selected_target_rectangles(obj, frame_width, frame_height):
    """Native and XY-expanded boxes; both axes use the original center-Y."""
    rect = clipped_yolo_rect(obj, frame_width, frame_height)
    if rect is None:
        return None, None
    left, top, width, height = rect
    padding = selected_target_x_padding(float(obj.y) + float(obj.h) * .5)
    outer_left = max(0, left - padding)
    outer_right = min(int(frame_width), left + width + padding)
    outer_top = max(0, top - padding)
    outer_bottom = min(int(frame_height), top + height + padding)
    return rect, [outer_left, outer_top, outer_right - outer_left, outer_bottom - outer_top]


def rects_have_area_overlap(first, second):
    """Positive X AND Y intersection; touching edges alone are not overlap."""
    if first is None or second is None:
        return False
    return (max(first[0], second[0]) < min(first[0] + first[2], second[0] + second[2])
            and max(first[1], second[1]) < min(first[1] + first[3], second[1] + second[3]))


def expanded_target_neighbors(candidate, objects, frame_width, frame_height):
    """All other movable native boxes intersecting the target's expanded box."""
    _, expanded = selected_target_rectangles(candidate, frame_width, frame_height)
    if expanded is None:
        return []
    return [obj for obj in objects
            if obj is not candidate and int(obj.class_id) in (3, 4, 5, 6)
            and rects_have_area_overlap(expanded, clipped_yolo_rect(obj, frame_width, frame_height))]


def directional_target_neighbors(candidate, objects, frame_width, frame_height):
    """H/V strips; casualty upper neighbors belong to V, including diagonals."""
    native, expanded = selected_target_rectangles(candidate, frame_width, frame_height)
    if native is None:
        return [], []
    horizontal = [expanded[0], native[1], expanded[2], native[3]]
    vertical = [native[0], expanded[1], native[2], expanded[3]]
    casualty_target = int(candidate.class_id) == 4
    target_y = float(candidate.y) + float(candidate.h) * .5
    horizontal_hits, vertical_hits = [], []
    for obj in objects:
        if obj is candidate or int(obj.class_id) not in (3, 4, 5, 6):
            continue
        rect = clipped_yolo_rect(obj, frame_width, frame_height)
        if (casualty_target
                and float(obj.y) + float(obj.h) * .5 < target_y
                and rects_have_area_overlap(expanded, rect)):
            # An upper-left/right neighbor is V-only even when its tall box
            # also reaches the horizontal strip. Keep the original XY range.
            vertical_hits.append(obj)
            continue
        if rects_have_area_overlap(horizontal, rect):
            horizontal_hits.append(obj)
        if rects_have_area_overlap(vertical, rect):
            vertical_hits.append(obj)
    return horizontal_hits, vertical_hits


def casualty_vertical_grab_allowed(candidate, horizontal_neighbors, vertical_neighbors):
    """RED with V-only neighbors may be grabbed if wide OR in front of all.

    Use ORIGINAL YOLO width/height, not long-side/short-side or the expanded
    rectangle. Width/height>1.5, or candidate centerY strictly larger than
    EVERY vertical neighbor centerY. Horizontal congestion always wins.
    This changes the decision only, never removes detections/neighbors.
    """
    if (candidate is None or int(candidate.class_id) != 4
            or horizontal_neighbors or not vertical_neighbors):
        return False
    width, height = float(candidate.w), float(candidate.h)
    if not (math.isfinite(width) and math.isfinite(height) and width > 0 and height > 0):
        return False
    if width > 1.5 * height:
        return True
    try:
        center_y = float(candidate.y) + height * .5
        if not math.isfinite(center_y):
            return False
        for neighbor in vertical_neighbors:
            neighbor_y, neighbor_h = float(neighbor.y), float(neighbor.h)
            other_center_y = neighbor_y + neighbor_h * .5
            if (not math.isfinite(other_center_y) or not math.isfinite(neighbor_h)
                    or neighbor_h <= 0 or center_y <= other_center_y):
                return False
    except (AttributeError, TypeError, ValueError, OverflowError):
        return False
    return True


TARGET_LABELS = {
    "sqareblue",
    "sqarered",
    "sqareredgreen",
    "triangualrblack",
}

SIGN_POSITIVE = 0x01
SIGN_NEGATIVE = 0x02


def spacing_y_band(center_y):
    """Return the shared absolute-center-Y spacing band, low to high.

    Bands are Y<=90, (90,100], (100,150], (150,220], (220,290],
    and Y>290. Boundary values belong to the lower band.
    """
    center_y = float(center_y)
    for band, upper_y in enumerate((90, 100, 150, 220, 290)):
        if center_y <= upper_y:
            return band
    return 5


def spacing_neighbor_in_y_scope(
    target_class_id, target_center_y, other_center_y, frame_center_y=224.0
):
    """Red checks all Y below image center, same band at/above it.

    Scope follows the candidate's actual class, not the neighbor's class.
    Green/black always check all Y; blue always retains same-band scope.
    The candidate's Y still selects the dx threshold, including red-pair limits.
    The default midpoint is for the current 576x448 model; callers with image
    geometry supply its actual midpoint, never the tracking ROI's midpoint.
    """
    return (
        int(target_class_id) in (5, 6)
        or (int(target_class_id) == 4
            and float(target_center_y) > float(frame_center_y))
        or spacing_y_band(target_center_y) == spacing_y_band(other_center_y)
    )


def target_center_error(
    obj,
    frame_width,
    frame_height,
    reference_x=None,
    reference_y=None,
):
    """Return signed (X, Y) pixel errors from the selected reference.

    The image center is used when no explicit reference point is supplied.
    Left and above the reference are negative; right and below are positive.
    """
    target_center_x = float(obj.x) + float(obj.w) * 0.5
    target_center_y = float(obj.y) + float(obj.h) * 0.5
    if reference_x is None:
        reference_x = float(frame_width) * 0.5
    if reference_y is None:
        reference_y = float(frame_height) * 0.5
    return (
        int(round(target_center_x - float(reference_x))),
        int(round(target_center_y - float(reference_y))),
    )


def split_error_sign(error):
    """Return (absolute pixel error, sign byte) for the UART protocol."""
    error = int(error)
    if error < 0:
        return abs(error), SIGN_NEGATIVE
    return error, SIGN_POSITIVE


def select_nearest_horizontal_target(
    objects,
    labels,
    frame_width,
    target_label=None,
    reference_x=None,
    reference_y=None,
    vertical_first=False,
    horizontal_first=False,
):
    """Return the target nearest to the supplied reference.

    ``target_label`` is reserved for the MCU color command. Passing ``None``
    accepts all four movable target classes. Invalid and safety-zone classes
    are ignored. When ``reference_y`` is supplied, squared X/Y distance is
    used; otherwise selection remains X-only. When ``vertical_first`` is true,
    absolute Y distance is the primary key and absolute X distance is the
    secondary key. ``horizontal_first`` reverses that priority. Confidence
    breaks equal-distance ties. The horizontal frame center is the default X
    reference.
    """
    if target_label is not None and target_label not in TARGET_LABELS:
        return None

    if reference_x is None:
        reference_x = float(frame_width) * 0.5
    selected = None
    selected_key = None

    for obj in objects:
        class_id = int(obj.class_id)
        if class_id < 0 or class_id >= len(labels):
            continue

        label = labels[class_id]
        if label not in TARGET_LABELS:
            continue
        if target_label is not None and label != target_label:
            continue

        object_center_x = float(obj.x) + float(obj.w) * 0.5
        horizontal_error = abs(object_center_x - float(reference_x))
        confidence = float(getattr(obj, "score", 0.0))
        if reference_y is None:
            candidate_key = (horizontal_error, -confidence)
        else:
            object_center_y = float(obj.y) + float(obj.h) * 0.5
            vertical_error = abs(object_center_y - float(reference_y))
            if vertical_first:
                candidate_key = (
                    vertical_error,
                    horizontal_error,
                    -confidence,
                )
            elif horizontal_first:
                candidate_key = (
                    horizontal_error,
                    vertical_error,
                    -confidence,
                )
            else:
                distance = horizontal_error * horizontal_error + (
                    vertical_error * vertical_error
                )
                candidate_key = (distance, -confidence)
        if selected_key is None or candidate_key < selected_key:
            selected = obj
            selected_key = candidate_key

    return selected
