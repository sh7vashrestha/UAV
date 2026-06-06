from __future__ import annotations

import unittest

from pixhawk_recorder.config import Config
from pixhawk_recorder.recorder import PixhawkDroneRecorder


class CropDedupTests(unittest.TestCase):
    def build_recorder(self, **cfg_overrides) -> PixhawkDroneRecorder:
        recorder = PixhawkDroneRecorder.__new__(PixhawkDroneRecorder)
        recorder.cfg = Config(use_pixhawk=False, **cfg_overrides)
        recorder._recent_saved_crops = []
        return recorder

    def test_skips_duplicate_crop_for_same_class_within_cooldown(self) -> None:
        recorder = self.build_recorder(
            save_crop_dedup_enabled=True,
            save_crop_dedup_cooldown_sec=2.0,
            save_crop_dedup_iou_threshold=0.5,
        )

        first = {"cls": 1}
        second = {"cls": 1}

        self.assertTrue(recorder._should_save_crop(first, (10, 10, 50, 50), now=0.0))
        self.assertFalse(recorder._should_save_crop(second, (12, 12, 52, 52), now=1.0))

    def test_allows_same_class_again_after_cooldown(self) -> None:
        recorder = self.build_recorder(
            save_crop_dedup_enabled=True,
            save_crop_dedup_cooldown_sec=1.0,
            save_crop_dedup_iou_threshold=0.5,
        )

        det = {"cls": 2}

        self.assertTrue(recorder._should_save_crop(det, (10, 10, 50, 50), now=0.0))
        self.assertTrue(recorder._should_save_crop(det, (12, 12, 52, 52), now=1.1))

    def test_allows_duplicate_when_dedup_disabled(self) -> None:
        recorder = self.build_recorder(save_crop_dedup_enabled=False)
        det = {"cls": 3}

        self.assertTrue(recorder._should_save_crop(det, (10, 10, 50, 50), now=0.0))
        self.assertTrue(recorder._should_save_crop(det, (10, 10, 50, 50), now=0.1))

    def test_allows_same_box_for_different_classes(self) -> None:
        recorder = self.build_recorder(
            save_crop_dedup_enabled=True,
            save_crop_dedup_cooldown_sec=2.0,
            save_crop_dedup_iou_threshold=0.5,
        )

        self.assertTrue(recorder._should_save_crop({"cls": 0}, (10, 10, 50, 50), now=0.0))
        self.assertTrue(recorder._should_save_crop({"cls": 1}, (10, 10, 50, 50), now=0.5))


if __name__ == "__main__":
    unittest.main()
