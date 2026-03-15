from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

import cv2
import numpy as np
import torch

from .compat import ensure_checkpoint_compat
from .config import Config


class OrinNanoYOLOEngine:
    def __init__(self, cfg: Config, device: torch.device, logger):
        self.cfg = cfg
        self.device = device
        self.logger = logger
        self.model = None
        self.names = {}
        self.half = False
        self.inference_count = 0
        self.error_count = 0
        self.nms_func = None
        self.scale_boxes_func = None
        self.letterbox_func = None
        self._load_model()

    def _load_model(self) -> None:
        ensure_checkpoint_compat()
        sys.path.insert(0, str(self.cfg.yolo_dir))
        cwd = os.getcwd()
        os.chdir(str(self.cfg.yolo_dir))
        try:
            from models.experimental import attempt_load
            from utils.general import non_max_suppression, scale_boxes
            from utils.augmentations import letterbox

            self.model = attempt_load(str(self.cfg.weights), device=self.device)
            self.model.eval()

            self.half = self.device.type == "cuda" and self.cfg.fp16_inference
            if self.half:
                self.model.half()

            self.names = getattr(self.model, "names", {i: str(i) for i in range(1000)})
            self.nms_func = non_max_suppression
            self.scale_boxes_func = scale_boxes
            self.letterbox_func = letterbox

            with torch.inference_mode():
                dummy = torch.zeros(
                    (1, 3, self.cfg.img_size, self.cfg.img_size),
                    device=self.device,
                    dtype=torch.float16 if self.half else torch.float32,
                )
                _ = self.model(dummy)

            self.logger.info("YOLO model loaded")
        except Exception as e:
            self.logger.exception("Failed to load YOLO: %s", e)
            raise
        finally:
            os.chdir(cwd)

    def infer(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        if self.model is None:
            return []

        try:
            inf = cv2.resize(frame, (self.cfg.inference_width, self.cfg.inference_height))
            im = self.letterbox_func(inf, new_shape=self.cfg.img_size, auto=False)[0]
            im = im[:, :, ::-1].transpose(2, 0, 1)
            im = np.ascontiguousarray(im)

            im_t = torch.from_numpy(im).to(self.device)
            im_t = im_t.half() if self.half else im_t.float()
            im_t = (im_t / 255.0).unsqueeze(0)

            with torch.inference_mode():
                pred = self.model(im_t)[0]
                pred = self.nms_func(pred, self.cfg.conf_threshold, self.cfg.iou_threshold)[0]

            dets = []
            if pred is not None and len(pred):
                p = pred.clone()
                p[:, :4] = self.scale_boxes_func(
                    (self.cfg.img_size, self.cfg.img_size),
                    p[:, :4],
                    inf.shape,
                ).round()

                h_inf, w_inf = inf.shape[:2]
                h_raw, w_raw = frame.shape[:2]
                sx, sy = w_raw / w_inf, h_raw / h_inf

                for (*xyxy, conf, cls) in p.tolist():
                    x1 = int(round(xyxy[0] * sx))
                    y1 = int(round(xyxy[1] * sy))
                    x2 = int(round(xyxy[2] * sx))
                    y2 = int(round(xyxy[3] * sy))

                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w_raw - 1, x2), min(h_raw - 1, y2)

                    dets.append(
                        {
                            "bbox": (x1, y1, x2, y2),
                            "conf": float(conf),
                            "cls": int(cls),
                            "name": self.names.get(int(cls), str(int(cls))),
                        }
                    )

            self.inference_count += 1
            return dets
        except Exception as e:
            self.error_count += 1
            self.logger.error("Inference error: %s", e)
            return []
