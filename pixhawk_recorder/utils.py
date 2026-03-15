from __future__ import annotations

from typing import Any, Dict, List, Optional

import cv2
import numpy as np


def draw_dets(frame: np.ndarray, dets: List[Dict[str, Any]]) -> np.ndarray:
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{d['name']} {d['conf']:.2f}"
        cv2.putText(
            frame,
            label,
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    return frame


def thermal_to_bgr(frame: np.ndarray) -> Optional[np.ndarray]:
    if frame is None or frame.size == 0:
        return None

    try:
        if frame.ndim == 2:
            if frame.dtype == np.uint16:
                f = frame.astype(np.float32)
                lo, hi = np.percentile(f, [2, 98])
                hi = max(hi, lo + 1)
                u8 = np.clip((f - lo) * 255 / (hi - lo), 0, 255).astype(np.uint8)
            else:
                u8 = frame.astype(np.uint8) if frame.dtype != np.uint8 else frame
            return cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)

        if frame.ndim == 3:
            gray = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2GRAY)
            return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    except Exception:
        return None

    return None


def safe_stem(text: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(text))
