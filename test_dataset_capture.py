import os
import time
import unittest
from unittest.mock import Mock, patch

from dataset_capture import DatasetCapture


class DatasetCaptureTests(unittest.TestCase):
    def setUp(self):
        self.today = time.struct_time((2026, 10, 10, 21, 30, 45, 5, 283, -1))
        self.tomorrow = time.struct_time((2026, 10, 11, 0, 0, 1, 6, 284, -1))
        helpers = {
            "clock": patch("dataset_capture.time.localtime", return_value=self.today),
            "mkdir": patch("dataset_capture.os.makedirs"),
            "exists": patch("dataset_capture.os.path.exists", return_value=False),
            "isfile": patch("dataset_capture.os.path.isfile", return_value=True),
            "getsize": patch("dataset_capture.os.path.getsize", return_value=123),
        }
        for name, helper in helpers.items():
            setattr(self, name, helper.start())
            self.addCleanup(helper.stop)
        self.today_dir = os.path.join("/root/sd", "2026-10-10")

    def test_open_creates_today_folder_directly_under_sd(self):
        capture = DatasetCapture()
        self.assertEqual(capture.directory, self.today_dir)
        self.assertEqual(capture.root_directory, "/root/sd")
        self.mkdir.assert_called_once_with(self.today_dir, exist_ok=True)

    def test_photo_path_uses_date_folder_and_timestamp(self):
        capture = DatasetCapture("/root/sd")
        img = Mock()
        path = capture.save(img)
        self.assertEqual(path, os.path.join(self.today_dir, "miss_20261010_213045_000001.jpg"))
        img.save.assert_called_once_with(path)
        self.assertEqual(capture.status, "SAVED 1")
        self.assertEqual(capture.count, 1)

    def test_collision_does_not_overwrite_existing_sample(self):
        capture = DatasetCapture("/root/sd")
        img = Mock()
        self.exists.side_effect = [True, False]
        path = capture.save(img)
        self.assertEqual(path, os.path.join(self.today_dir, "miss_20261010_213045_000002.jpg"))
        img.save.assert_called_once_with(path)
        self.assertEqual(capture.status, "SAVED 1")

    def test_same_day_restart_reuses_folder_and_skips_previous_filename(self):
        first = DatasetCapture("/root/sd")
        first_path = first.save(Mock())
        second = DatasetCapture("/root/sd")
        self.exists.side_effect = lambda path: path == first_path
        second_path = second.save(Mock())
        self.assertEqual(first.directory, second.directory)
        self.assertNotEqual(first_path, second_path)
        self.assertTrue(second_path.endswith("_000002.jpg"))
        self.assertEqual(second.count, 1)

    def test_multiple_photos_same_second_have_distinct_sequences(self):
        capture = DatasetCapture()
        first, second = capture.save(Mock()), capture.save(Mock())
        self.assertTrue(first.endswith("_000001.jpg"))
        self.assertTrue(second.endswith("_000002.jpg"))
        self.assertEqual(capture.count, 2)
        self.assertEqual(capture.status, "SAVED 2")

    def test_staying_open_across_midnight_switches_date_folder(self):
        capture = DatasetCapture()
        first_path = capture.save(Mock())
        self.clock.return_value = self.tomorrow
        second_path = capture.save(Mock())
        tomorrow_dir = os.path.join("/root/sd", "2026-10-11")
        self.assertEqual(os.path.dirname(first_path), self.today_dir)
        self.assertEqual(os.path.dirname(second_path), tomorrow_dir)
        self.assertEqual(capture.directory, tomorrow_dir)
        self.assertTrue(second_path.endswith("miss_20261011_000001_000002.jpg"))
        self.mkdir.assert_called_with(tomorrow_dir, exist_ok=True)

    def test_folder_and_filename_share_one_clock_snapshot(self):
        capture = DatasetCapture()
        self.clock.reset_mock()
        self.clock.side_effect = [self.today, self.tomorrow]
        path = capture.save(Mock())
        self.clock.assert_called_once_with()
        self.assertEqual(path, os.path.join(self.today_dir, "miss_20261010_213045_000001.jpg"))

    def test_custom_root_also_gets_date_subfolder(self):
        capture = DatasetCapture("/root/sd/custom_dataset")
        path = capture.save(Mock())
        self.assertEqual(os.path.dirname(path), os.path.join("/root/sd/custom_dataset", "2026-10-10"))

    def test_sd_write_failure_is_reported_without_stopping_vision(self):
        capture = DatasetCapture("/root/sd")
        img = Mock()
        img.save.side_effect = OSError("SD full")
        with patch("builtins.print") as log:
            self.assertIsNone(capture.save(img))
        self.assertIn("save failed", log.call_args.args[0])
        self.assertEqual(capture.count, 0)
        self.assertEqual(capture.status, "SAVE FAILED")

    def test_unavailable_directory_at_open_can_recover_on_photo(self):
        self.mkdir.side_effect = OSError("SD unavailable")
        with patch("builtins.print") as log:
            capture = DatasetCapture()
        self.assertIn("directory unavailable", log.call_args.args[0])
        self.mkdir.side_effect = None
        self.assertIsNotNone(capture.save(Mock()))
        self.assertEqual(capture.count, 1)

    def test_directory_failure_during_save_does_not_attempt_image_write(self):
        capture = DatasetCapture()
        self.mkdir.side_effect = OSError("SD unavailable")
        img = Mock()
        with patch("builtins.print"):
            self.assertIsNone(capture.save(img))
        img.save.assert_not_called()
        self.assertEqual(capture.count, 0)
        self.assertEqual(capture.status, "SAVE FAILED")

    def test_zero_byte_image_is_not_reported_as_saved(self):
        capture = DatasetCapture()
        self.getsize.return_value = 0
        with patch("builtins.print"):
            self.assertIsNone(capture.save(Mock()))
        self.assertEqual(capture.count, 0)
        self.assertEqual(capture.status, "SAVE FAILED")

    def test_no_produced_file_is_not_reported_as_saved(self):
        capture = DatasetCapture()
        self.isfile.return_value = False
        with patch("builtins.print"):
            self.assertIsNone(capture.save(Mock()))
        self.assertEqual(capture.count, 0)
        self.assertEqual(capture.status, "SAVE FAILED")


if __name__ == "__main__":
    unittest.main()
