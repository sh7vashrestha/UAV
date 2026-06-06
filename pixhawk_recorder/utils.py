from __future__ import annotations

from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .config import Config


def draw_dets(frame: np.ndarray, dets: List[Dict[str, Any]], cfg: Optional[Config] = None) -> np.ndarray:
    default_box_color = _bgr(cfg.annot_box_color_bgr if cfg is not None else (255, 0, 255))
    class_box_colors = tuple(_bgr(color) for color in cfg.annot_class_colors_bgr) if cfg is not None else ()
    outline_color = _bgr(cfg.annot_box_outline_color_bgr if cfg is not None else (0, 0, 0))
    text_color = _bgr(cfg.annot_text_color_bgr if cfg is not None else (255, 255, 255))
    text_bg_color = _bgr(cfg.annot_text_bg_color_bgr if cfg is not None else (0, 0, 0))
    box_thickness = cfg.annot_box_thickness if cfg is not None else 3
    outline_thickness = cfg.annot_box_outline_thickness if cfg is not None else 2
    text_scale = cfg.annot_text_scale if cfg is not None else 0.9
    text_thickness = cfg.annot_text_thickness if cfg is not None else 2
    text_padding = cfg.annot_text_padding_px if cfg is not None else 4
    font = cv2.FONT_HERSHEY_SIMPLEX

    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        box_color = _det_box_color(d, class_box_colors, default_box_color)
        if outline_thickness > 0:
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                outline_color,
                box_thickness + (2 * outline_thickness),
            )
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, box_thickness)
        label = f"{d['name']} {d['conf']:.2f}"
        (label_width, label_height), baseline = cv2.getTextSize(
            label,
            font,
            text_scale,
            text_thickness,
        )

        label_left = max(x1, 0)
        label_bottom = max(y1 - 6, label_height + baseline + text_padding)
        label_top = max(label_bottom - label_height - baseline - (2 * text_padding), 0)
        label_right = min(label_left + label_width + (2 * text_padding), frame.shape[1] - 1)

        cv2.rectangle(
            frame,
            (label_left, label_top),
            (label_right, label_bottom),
            text_bg_color,
            thickness=-1,
        )
        cv2.putText(
            frame,
            label,
            (label_left + text_padding, label_bottom - baseline - text_padding),
            font,
            text_scale,
            text_color,
            text_thickness,
            cv2.LINE_AA,
        )
    return frame


def _bgr(color: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(int(channel) for channel in color)


def _det_box_color(
    det: Dict[str, Any],
    class_box_colors: tuple[tuple[int, int, int], ...],
    default_box_color: tuple[int, int, int],
) -> tuple[int, int, int]:
    try:
        class_id = int(det.get("cls", -1))
    except (TypeError, ValueError):
        return default_box_color

    if 0 <= class_id < len(class_box_colors):
        return class_box_colors[class_id]
    return default_box_color


def class_dir_name(label: object) -> str:
    name = safe_stem(str(label).strip().lower()).strip("_")
    return name or "unknown"


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
