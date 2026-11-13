"""Fresh YOLO blackbar geometry; auxiliary only, never a cargo/UART class."""

from target_selector import clipped_yolo_rect

BLACKBAR_CLASS_ID = 7


def blackbar_in_zone(bar, zone, frame_width, frame_height):
    if int(bar.class_id) != BLACKBAR_CLASS_ID:
        return False
    if (clipped_yolo_rect(bar, frame_width, frame_height) is None
            or clipped_yolo_rect(zone, frame_width, frame_height) is None):
        return False
    cx = float(bar.x) + float(bar.w) * .5
    cy = float(bar.y) + float(bar.h) * .5
    left, top = max(0.0, float(zone.x)), max(0.0, float(zone.y))
    right = min(float(frame_width), float(zone.x) + float(zone.w))
    bottom = min(float(frame_height), float(zone.y) + float(zone.h))
    return left <= cx < right and top <= cy < bottom


def select_zone_blackbar(bars, zone, frame_width, frame_height):
    """Take the central bar of this SAME selected zone, never another zone."""
    if zone is None:
        return None
    candidates = [bar for bar in bars
                  if blackbar_in_zone(bar, zone, frame_width, frame_height)]
    center_x = float(zone.x) + float(zone.w) * .5
    return min(candidates, key=lambda bar: (
        abs(float(bar.x) + float(bar.w) * .5 - center_x),
        -float(bar.w) * float(bar.h),
    )) if candidates else None


def blackbar_side(bar, zone):
    """-1=left,0=exactly centered/missing,+1=right, versus zone BOX center."""
    if bar is None:
        return 0
    delta = (float(bar.x) + float(bar.w) * .5
             - float(zone.x) - float(zone.w) * .5)
    return (delta > 0) - (delta < 0)


def blackbar_requires_realign(bar, zone, casualty):
    """Shared46 side rule: supplies LEFT, casualties RIGHT of zone center."""
    return bar is not None and zone is not None and blackbar_side(bar, zone) == (1 if casualty else -1)
