"""Per-class score gates shared by every consumer of normalized YOLO boxes."""

import math


class DetectionConfidenceFilter:
    def __init__(self, default_threshold, black_threshold=0.7, safety_zone_threshold=0.4):
        values = tuple(float(value) for value in (
            default_threshold, black_threshold, safety_zone_threshold,
        ))
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
            raise ValueError("Detection confidence thresholds must be in [0, 1]")
        default, black, safety = values
        # Stable semantic IDs AFTER ModelClassMapper: zones1/2, blue3,
        # red4, green5, black6. Native model0 may be a valid blue zone.
        self._thresholds = (default, safety, safety, default, default, default, black)
        # A single higher YOLO threshold would discard valid .4-.5 zones
        # before we could apply their own threshold. Gate other classes below.
        self.detector_threshold = min(values)

    def apply(self, objects):
        accepted = []
        for obj in objects:
            class_id = int(obj.class_id)
            if not 0 < class_id < len(self._thresholds):
                continue
            score = float(getattr(obj, "score", 0.0))
            if math.isfinite(score) and score >= self._thresholds[class_id]:
                accepted.append(obj)
        return accepted

    def apply_blackbars(self, objects):
        """Blackbar uses the default .5 gate, not triangle cargo's .7 gate."""
        return self._apply_auxiliary(objects, 7)

    def apply_startzones(self, objects):
        """Auxiliary startzone boxes use the ordinary confidence threshold."""
        return self._apply_auxiliary(objects, 8)

    def _apply_auxiliary(self, objects, auxiliary_id):
        accepted = []
        for obj in objects:
            score = float(getattr(obj, "score", 0.0))
            if (int(obj.class_id) == auxiliary_id and math.isfinite(score)
                    and score >= self._thresholds[0]):
                accepted.append(obj)
        return accepted
