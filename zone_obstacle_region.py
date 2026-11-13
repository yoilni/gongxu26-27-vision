"""Shared safety-zone obstacle geometry for UART decisions and DEBUG."""

import app_config as config


def zone_obstacle_polygon(roi_rect, frame_width, frame_height):
    """Return top-left/right, bottom-right/left vertices, or None.

    Build the original centered front edge and ROI-relative rear edge,
    then clip its top at the configured Y, preserving both sloped sides.
    """
    if roi_rect is None or frame_width <= 0 or frame_height <= 0:
        return None
    left, top, width, _ = roi_rect
    if width <= 0 or config.UART_ZONE_OBSTACLE_FRONT_WIDTH_PX <= 0:
        return None
    center_x = float(frame_width) * 0.5
    half_width = float(config.UART_ZONE_OBSTACLE_FRONT_WIDTH_PX) * 0.5
    front_left = max(0.0, center_x - half_width)
    front_right = min(float(frame_width - 1), center_x + half_width)
    rear_left = max(0.0, float(left))
    rear_right = min(float(frame_width - 1), float(left + width - 1))
    front_y = max(0.0, float(config.UART_ZONE_OBSTACLE_FRONT_Y))
    rear_y = min(
        float(frame_height - 1),
        float(top) + float(config.UART_ZONE_OBSTACLE_EDGE_INSIDE_PX),
    )
    if front_left >= front_right or rear_left >= rear_right or front_y >= rear_y:
        return None
    clipped_y = max(front_y, float(config.UART_ZONE_OBSTACLE_CLIP_TOP_Y))
    if clipped_y >= rear_y:
        return None
    progress = (clipped_y - front_y) / (rear_y - front_y)
    front_left += progress * (rear_left - front_left)
    front_right += progress * (rear_right - front_right)
    front_y = clipped_y
    return (
        (front_left, front_y), (front_right, front_y),
        (rear_right, rear_y), (rear_left, rear_y),
    )


def point_in_zone_obstacle_polygon(center_x, center_y, polygon):
    """Test a center against the trapezoid, including all four edges."""
    if polygon is None:
        return False
    front_left, front_right, rear_right, rear_left = polygon
    if not front_left[1] <= center_y <= rear_left[1]:
        return False
    progress = (center_y - front_left[1]) / (rear_left[1] - front_left[1])
    left_x = front_left[0] + progress * (rear_left[0] - front_left[0])
    right_x = front_right[0] + progress * (rear_right[0] - front_right[0])
    return left_x <= center_x <= right_x
