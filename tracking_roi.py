"""Shared tracking ROI geometry and reference points."""

from math import isfinite
import app_config as config


def _bounds(roi):
    if roi is None or len(roi) != 4:
        return None
    try:
        values = tuple(map(float, roi))
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(isfinite(value) for value in values) or values[2] <= 0 or values[3] <= 0:
        return None
    return values


def roi_triangle_vertices(roi):
    """Build an isosceles ROI by mirroring the left base point across the apex X.

    Default [0,319,576,129] gives (313,344), (150,447), (476,447).
    """
    bounds = _bounds(roi)
    if bounds is None:
        return None
    left, top, width, height = bounds
    if width <= 1 or height <= 1:
        return None
    try:
        offset_x = float(getattr(config, "UART_TRACKING_ROI_APEX_OFFSET_X", 25))
        offset_y = float(getattr(config, "UART_TRACKING_ROI_APEX_OFFSET_Y", 25))
        base_inset = float(getattr(config, "UART_TRACKING_ROI_BASE_INSET_X", 150))
    except (TypeError, ValueError, OverflowError):
        return None
    if (not isfinite(offset_x) or not isfinite(offset_y)
            or not isfinite(base_inset) or base_inset < 0
            or top + offset_y >= top + height - 1):
        return None
    # Keep the left base point in the left half of a narrow custom ROI.
    base_inset = min(base_inset, max(0.0, (width - 2) * .5))
    apex_x = left + width * .5 + offset_x
    bottom_left_x = left + base_inset
    if bottom_left_x >= apex_x:
        return None
    bottom_y = top + height - 1
    return ((apex_x, top + offset_y),
            (bottom_left_x, bottom_y),
            (2 * apex_x - bottom_left_x, bottom_y))


def near_view_roi_vertices(frame_width, frame_height):
    """State 05 outline with a trapezoid above a full-width rectangle."""
    top_y = 102
    bottom_y = frame_height - 1 - 40
    right_x = frame_width - 1
    if right_x <= 0 or bottom_y <= top_y:
        return None
    center_x = frame_width // 2
    shoulder_y = max(top_y, bottom_y - 100)
    return ((max(0, center_x - 80), top_y),
            (min(right_x, center_x + 80), top_y),
            (right_x, shoulder_y), (right_x, bottom_y),
            (0, bottom_y), (0, shoulder_y))


def point_in_near_view_roi(x, y, vertices):
    """Test against the same six vertices drawn in DEBUG, including borders."""
    top_left, top_right, shoulder_right, bottom_right, bottom_left, shoulder_left = vertices
    if y >= shoulder_left[1]:
        return y <= bottom_left[1] and bottom_left[0] <= x <= bottom_right[0]
    return point_in_trapezoid(x, y, (top_left, top_right, shoulder_right, shoulder_left))


def roi_reticle_point(apex):
    """Purple state 04 reticle, 50 pixels above the triangle apex."""
    x, y = apex
    return int(round(x)), int(round(y - 50))


def search_reticle_point(apex):
    """Shared state 03 reticle, 20 pixels above the state 24 reticle."""
    x, y = final_track_reticle_point(apex)
    return x, y - 20


def final_track_reticle_point(apex):
    """Shared state 24 steering reticle and capture line."""
    x, y = apex
    return int(round(x)), int(round(y + 15))


def point_in_tracking_roi(x, y, roi):
    """Center-entry test shared by display/03 exclusion/24/51 handoff.

    Triangle edges/vertices are inclusive and match the drawn polygon.
    An explicit legacy rectangle setting retains old half-open semantics.
    """
    bounds = _bounds(roi)
    if bounds is None or not isfinite(x) or not isfinite(y):
        return False
    left, top, width, height = bounds
    if getattr(config, "UART_TRACKING_ROI_SHAPE", "triangle") == "rectangle":
        return left <= x < left + width and top <= y < top + height
    vertices = roi_triangle_vertices(roi)
    if vertices is None:
        return False
    apex, bottom_left, bottom_right = vertices
    return point_in_trapezoid(x, y, (apex, apex, bottom_right, bottom_left))


def point_in_trapezoid(x, y, vertices):
    """Test a polygon with horizontal top and bottom edges, including borders."""
    top_left, top_right, bottom_right, bottom_left = vertices
    if not top_left[1] <= y <= bottom_left[1]:
        return False
    progress = (y - top_left[1]) / (bottom_left[1] - top_left[1])
    left_x = top_left[0] + progress * (bottom_left[0] - top_left[0])
    right_x = top_right[0] + progress * (bottom_right[0] - top_right[0])
    return left_x - 1e-9 <= x <= right_x + 1e-9
