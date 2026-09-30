"""DEBUG-only target visualization; never changes the UART tracking state."""

import app_config as config


MOVABLE_LABELS = {
    "sqareblue",
    "sqarered",
    "sqareredgreen",
    "triangualrblack",
}
SAFETY_LABELS = {"bluesafety", "redsafety"}
MIXED_LABEL = "black_or_green"


def _center(obj):
    return (
        float(obj.x) + float(obj.w) * 0.5,
        float(obj.y) + float(obj.h) * 0.5,
    )


def _dx_limit(y, red_pair):
    if y > 290:
        return (
            config.UART_FINAL_RED_MIN_DX_Y_GT_290 if red_pair
            else config.UART_FINAL_PRIMARY_MIN_DX_Y_GT_290
        )
    if y > 220:
        return (
            config.UART_FINAL_RED_MIN_DX_Y_GT_220 if red_pair
            else config.UART_FINAL_PRIMARY_MIN_DX_Y_GT_220
        )
    if y > 150:
        return (
            config.UART_FINAL_RED_MIN_DX_Y_GT_150 if red_pair
            else config.UART_FINAL_PRIMARY_MIN_DX_Y_GT_150
        )
    if y > 100:
        return (
            config.UART_FINAL_RED_MIN_DX_Y_GT_100 if red_pair
            else config.UART_FINAL_PRIMARY_MIN_DX_Y_GT_100
        )
    if y > 90:
        return (
            config.UART_FINAL_RED_MIN_DX_Y_GT_90 if red_pair
            else config.UART_FINAL_PRIMARY_MIN_DX_Y_GT_90
        )
    return (
        config.UART_FINAL_RED_MIN_DX_Y_LE_90 if red_pair
        else config.UART_FINAL_PRIMARY_MIN_DX_Y_LE_90
    )


class DebugTargetPreview:
    """Show isolated targets immediately; confirm multi-object choices."""

    def __init__(self, frame_height=448, tracking_roi=None):
        self._reference_y = (
            float(tracking_roi[1]) + float(tracking_roi[3]) * 0.5
            if tracking_roi is not None else float(frame_height) * 0.5
        )
        self.reset()

    def reset(self):
        self._anchor = None
        self._clear_frames = 0
        self._label = None

    def select(self, verified_objects, labels, raw_objects, target_label):
        target_label = target_label or "sqareredgreen"
        if target_label != self._label:
            self.reset()
            self._label = target_label
        if target_label == MIXED_LABEL:
            target_labels = {"sqareredgreen", "triangualrblack"}
        elif target_label in MOVABLE_LABELS:
            target_labels = {target_label}
        else:
            self.reset()
            return [], [], None, []

        zones = [
            obj for obj in raw_objects
            if 0 < int(obj.class_id) < len(labels)
            and labels[int(obj.class_id)] in SAFETY_LABELS
            and float(obj.w) > 0 and float(obj.h) > 0
        ]
        def movable_object(obj):
            class_id = int(obj.class_id)
            if class_id <= 0 or class_id >= len(labels):
                return False
            label = labels[class_id]
            return label in MOVABLE_LABELS

        def outside_safety_zone(obj):
            center_x, center_y = _center(obj)
            return not any(
                float(zone.x) <= center_x <= float(zone.x) + float(zone.w)
                and float(zone.y) <= center_y <= float(zone.y) + float(zone.h)
                for zone in zones
            )

        # DEBUG previews current raw YOLO geometry, independently of color/
        # temporal verification used for actual UART tracking. Otherwise an
        # isolated detected box vanishes from DEBUG until verification passes.
        # Safety-zone-contained boxes remain excluded as target candidates,
        # but are still visible as white neighbors when close to a candidate.
        movable = [obj for obj in raw_objects if movable_object(obj)]
        candidates = [
            obj for obj in movable
            if movable_object(obj)
            and outside_safety_zone(obj)
            and labels[int(obj.class_id)] in target_labels
        ]

        neighbors = []
        neighbor_dx = {}
        close_candidates = set()
        for candidate in candidates:
            center_x, center_y = _center(candidate)
            for other in movable:
                if other is candidate:
                    continue
                other_x, _ = _center(other)
                dx = abs(other_x - center_x)
                red_pair = (
                    int(candidate.class_id) == 4
                    or int(other.class_id) == 4
                )
                if dx > _dx_limit(center_y, red_pair):
                    continue
                close_candidates.add(id(candidate))
                if id(other) not in neighbor_dx:
                    neighbors.append(other)
                    neighbor_dx[id(other)] = dx
                else:
                    neighbor_dx[id(other)] = min(neighbor_dx[id(other)], dx)

        clear = [
            obj for obj in candidates
            if id(obj) not in close_candidates
        ]
        if clear:
            min_y_error = min(
                abs(_center(obj)[1] - self._reference_y) for obj in clear
            )
            clear = [
                obj for obj in clear
                if abs(_center(obj)[1] - self._reference_y) == min_y_error
            ]
        continuous = []
        if self._anchor is not None:
            class_id, last_x, last_y = self._anchor
            for obj in clear:
                center_x, center_y = _center(obj)
                if (
                    int(obj.class_id) == class_id
                    and abs(center_x - last_x) <= config.UART_MAX_X_JUMP_PX
                    and abs(center_y - last_y) <= config.UART_MAX_Y_JUMP_PX
                ):
                    continuous.append(obj)
        if continuous:
            preview = min(
                continuous,
                key=lambda obj: (
                    (_center(obj)[0] - self._anchor[1]) ** 2
                    + (_center(obj)[1] - self._anchor[2]) ** 2
                ),
            )
            self._clear_frames += 1
        elif clear:
            preview = min(
                clear,
                key=lambda obj: (
                    _center(obj)[0], -float(getattr(obj, "score", 0.0))
                ),
            )
            self._clear_frames = 1
        else:
            preview = None
            self._clear_frames = 0

        self._anchor = (
            int(preview.class_id), *_center(preview)
        ) if preview is not None else None
        isolated_target = len(candidates) == 1 and len(movable) <= 1
        selected = (
            preview if (
                isolated_target
                or self._clear_frames >= config.DEBUG_SECONDARY_CLEAR_FRAMES
            )
            else None
        )
        annotations = [
            (obj, int(round(neighbor_dx[id(obj)]))) for obj in neighbors
        ]
        return candidates, neighbors, selected, annotations
