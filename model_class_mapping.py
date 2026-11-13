"""Map YOLO label positions to stable vision/UART category IDs.

Supported model layouts include the reordered seven classes:
bluesafety, redsafety, sqareblue, sqaregreen, sqarered, startzone, triangualrblack.
Resolve by label so new indices and the green rename cannot change UART IDs.
The auxiliary blackbar/startzone channels are separate, never UART cargo.
"""


VISION_LABELS = (
    "blue safety", "bluesafety", "redsafety", "sqareblue",
    "sqarered", "sqareredgreen", "triangualrblack",
)
LABEL_TO_VISION_ID = {
    label: class_id for class_id, label in enumerate(VISION_LABELS)
    if class_id > 0
}
# New spelling, same green task. Keep the canonical labels and existing MCU
# IDs (zones1/2, blue3, red4, green5, black6), including old-model support.
LABEL_TO_VISION_ID["sqaregreen"] = LABEL_TO_VISION_ID["sqareredgreen"]


class MappedDetection:
    """Current-frame box with a semantic ID; never mutate native YOLO boxes."""

    __slots__ = ("x", "y", "w", "h", "score", "class_id", "model_class_id")

    def __init__(self, obj, class_id):
        self.x = obj.x
        self.y = obj.y
        self.w = obj.w
        self.h = obj.h
        self.score = getattr(obj, "score", 0.0)
        self.class_id = class_id
        self.model_class_id = int(obj.class_id)


class ModelClassMapper:
    """Resolve labels once, then discard ignored classes before all consumers.

    Supports reordered/legacy seven-class and eight-class models without
    guessing from class count. In particular, model0=bluesafety is VALID;
    only the deprecated LABEL blue safety is ignored, never index0 itself.
    startzone and unknown labels are discarded from task consumers. An
    independent startzones() channel serves DEBUG drawing and AE escape,
    never cargo selection/counting or coordinate category IDs.
    """

    def __init__(self, model_labels):
        self.labels = VISION_LABELS
        self._blackbar_ids = frozenset(index for index, label in enumerate(model_labels)
                                      if label == "blackbar")
        self._startzone_ids = frozenset(index for index, label in enumerate(model_labels)
                                       if label == "startzone")
        self._class_ids = tuple(
            LABEL_TO_VISION_ID.get(label) for label in model_labels
        )

    def normalize(self, objects):
        normalized = []
        for obj in objects:
            model_id = int(obj.class_id)
            if not 0 <= model_id < len(self._class_ids):
                continue
            class_id = self._class_ids[model_id]
            if class_id is None:
                continue
            # Existing categories retain object identity and require no copy.
            normalized.append(
                obj if class_id == model_id else MappedDetection(obj, class_id)
            )
        return normalized

    def blackbars(self, objects):
        """Auxiliary YOLO detections, independent of legacy black-at-index7."""
        return [obj if int(obj.class_id) == 7 else MappedDetection(obj, 7)
                for obj in objects if int(obj.class_id) in self._blackbar_ids]

    def startzones(self, objects):
        """Class8 for AE/DEBUG, resolved by LABEL, never cargo UART."""
        return [obj if int(obj.class_id) == 8 else MappedDetection(obj, 8)
                for obj in objects if int(obj.class_id) in self._startzone_ids]

    def summary(self):
        """Small startup-only diagnostic; no per-frame console traffic."""
        return " ".join(
            "{}->{}".format(model_id, "blackbar(aux)" if model_id in self._blackbar_ids
                           else "ignore" if class_id is None else class_id)
            for model_id, class_id in enumerate(self._class_ids)
        )
