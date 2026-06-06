from __future__ import annotations

import unittest

import numpy as np

from pixhawk_recorder.config import Config
from pixhawk_recorder.utils import class_dir_name, draw_dets


class AnnotationStyleTests(unittest.TestCase):
    def test_class_dir_name_normalizes_damage_labels(self) -> None:
        self.assertEqual(class_dir_name("light damage"), "light_damage")
        self.assertEqual(class_dir_name(" severe damage "), "severe_damage")
        self.assertEqual(class_dir_name(""), "unknown")

    def test_draw_dets_uses_class_specific_box_colors(self) -> None:
        cfg = Config(
            use_pixhawk=False,
            annot_box_color_bgr=(12, 34, 56),
            annot_class_colors_bgr=((10, 20, 30), (40, 50, 60)),
            annot_box_outline_thickness=0,
            annot_text_bg_color_bgr=(1, 2, 3),
        )
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        dets = [
            {"bbox": (10, 40, 50, 80), "name": "weed", "conf": 0.95, "cls": 0},
            {"bbox": (55, 40, 90, 80), "name": "crop", "conf": 0.85, "cls": 1},
        ]

        annotated = draw_dets(frame, dets, cfg)

        self.assertTrue(np.array_equal(annotated[40, 10], np.array([10, 20, 30], dtype=np.uint8)))
        self.assertTrue(np.array_equal(annotated[40, 55], np.array([40, 50, 60], dtype=np.uint8)))
        self.assertTrue(np.any(np.all(annotated == np.array([1, 2, 3], dtype=np.uint8), axis=2)))


if __name__ == "__main__":
    unittest.main()
