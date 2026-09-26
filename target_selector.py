"""Select the movable target closest to a requested image reference."""


TARGET_LABELS = {
    "sqareblue",
    "sqarered",
    "sqareredgreen",
    "triangualrblack",
}

SIGN_POSITIVE = 0x01
SIGN_NEGATIVE = 0x02


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
):
    """Return the target nearest to the supplied reference.

    ``target_label`` is reserved for the MCU color command. Passing ``None``
    accepts all four movable target classes. Invalid and safety-zone classes
    are ignored. When ``reference_y`` is supplied, squared X/Y distance is
    used; otherwise selection remains X-only. Confidence breaks equal-distance
    ties. The horizontal frame center is the default X reference.
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
        if reference_y is None:
            distance = horizontal_error
        else:
            object_center_y = float(obj.y) + float(obj.h) * 0.5
            vertical_error = object_center_y - float(reference_y)
            distance = horizontal_error * horizontal_error + (
                vertical_error * vertical_error
            )
        confidence = float(getattr(obj, "score", 0.0))
        candidate_key = (distance, -confidence)
        if selected_key is None or candidate_key < selected_key:
            selected = obj
            selected_key = candidate_key

    return selected
