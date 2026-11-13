"""Save the supplied image, including DEBUG annotations, in dated SD folders."""

import os
import time


class DatasetCapture:
    def __init__(self, directory="/root/sd"):
        # The argument is the SD root, not a fixed dataset folder. Reopening
        # on the same date reuses its folder without removing previous images.
        self.root_directory = directory
        self.directory = self._date_directory(time.localtime())
        self.sequence = 0
        self.count = 0
        self.status = ""
        try:
            os.makedirs(self.directory, exist_ok=True)
        except OSError as exc:
            print("[PHOTO] directory unavailable:", exc)

    def _date_directory(self, local_now):
        return os.path.join(self.root_directory, time.strftime("%Y-%m-%d", local_now))

    def save(self, img):
        try:
            # Use one clock snapshot for both date folder and filename; also
            # roll over if this program stays open across midnight.
            local_now = time.localtime()
            self.directory = self._date_directory(local_now)
            os.makedirs(self.directory, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S", local_now)
            while True:
                self.sequence += 1
                path = os.path.join(self.directory, "miss_{}_{:06d}.jpg".format(stamp, self.sequence))
                if not os.path.exists(path):
                    break
            img.save(path)
            if not os.path.isfile(path) or os.path.getsize(path) == 0:
                raise OSError("image save produced no data")
            self.count += 1
            self.status = "SAVED {}".format(self.count)
            print("[PHOTO] saved:", path)
            return path
        except Exception as exc:
            self.status = "SAVE FAILED"
            print("[PHOTO] save failed:", exc)
            return None
