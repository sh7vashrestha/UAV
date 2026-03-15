from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional

from .config import Config
from .telemetry import PixhawkTelemetry


EARTH_RADIUS_M = 6378137.0
CAMERA_TO_BODY_FORWARD = (
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
)

GEOLOCATION_CSV_HEADER = [
    "estimate_method",
    "capture_wall_time",
    "clock_source",
    "flight_name",
    "infer_id",
    "t_sec",
    "detection_index",
    "crop_file",
    "crop_path",
    "class_id",
    "class_name",
    "conf",
    "x1",
    "y1",
    "x2",
    "y2",
    "bbox_cx",
    "bbox_cy",
    "image_width",
    "image_height",
    "drone_lat",
    "drone_lon",
    "drone_alt_abs_m",
    "drone_alt_rel_m",
    "drone_roll_deg",
    "drone_pitch_deg",
    "drone_yaw_deg",
    "gps_hdop",
    "gps_satellites",
    "ground_speed_mps",
    "camera_focal_mm",
    "sensor_width_mm",
    "effective_sensor_height_mm",
    "camera_roll_deg",
    "camera_pitch_deg",
    "camera_yaw_deg",
    "target_valid",
    "target_reason",
    "target_lat",
    "target_lon",
    "target_north_m",
    "target_east_m",
    "ground_distance_m",
    "slant_range_m",
    "ray_down_component",
]


def build_crop_geolocation_record(
    cfg: Config,
    telemetry: Optional[PixhawkTelemetry],
    *,
    flight_name: str,
    infer_id: int,
    t_sec: float,
    detection_index: int,
    crop_path: Path,
    bbox: tuple[int, int, int, int],
    class_id: int,
    class_name: str,
    conf: float,
    clock_source: str,
    capture_wall_time: str,
) -> Dict[str, Any]:
    x1, y1, x2, y2 = bbox
    bbox_cx = (x1 + x2) / 2.0
    bbox_cy = (y1 + y2) / 2.0

    record: Dict[str, Any] = {
        "estimate_method": "flat_ground_bbox_center_v1",
        "capture_wall_time": capture_wall_time,
        "clock_source": clock_source,
        "flight_name": flight_name,
        "infer_id": int(infer_id),
        "t_sec": round(float(t_sec), 3),
        "detection_index": int(detection_index),
        "crop_file": crop_path.name,
        "crop_path": str(crop_path),
        "class_id": int(class_id),
        "class_name": str(class_name),
        "conf": round(float(conf), 4),
        "x1": int(x1),
        "y1": int(y1),
        "x2": int(x2),
        "y2": int(y2),
        "bbox_cx": round(bbox_cx, 2),
        "bbox_cy": round(bbox_cy, 2),
        "image_width": int(cfg.rgb_width),
        "image_height": int(cfg.rgb_height),
        "camera_focal_mm": float(cfg.rgb_lens_focal_length_mm),
        "sensor_width_mm": float(cfg.rgb_sensor_width_mm),
        "effective_sensor_height_mm": round(_effective_sensor_height_mm(cfg), 4),
        "camera_roll_deg": float(cfg.camera_mount_roll_deg),
        "camera_pitch_deg": float(cfg.camera_mount_pitch_deg),
        "camera_yaw_deg": float(cfg.camera_mount_yaw_deg),
        "target_valid": False,
        "target_reason": "pixhawk_disabled_or_missing",
        "target_lat": None,
        "target_lon": None,
        "target_north_m": None,
        "target_east_m": None,
        "ground_distance_m": None,
        "slant_range_m": None,
        "ray_down_component": None,
        "drone_lat": None,
        "drone_lon": None,
        "drone_alt_abs_m": None,
        "drone_alt_rel_m": None,
        "drone_roll_deg": None,
        "drone_pitch_deg": None,
        "drone_yaw_deg": None,
        "gps_hdop": None,
        "gps_satellites": None,
        "ground_speed_mps": None,
    }

    if telemetry is None:
        return record

    record.update(
        {
            "drone_lat": telemetry.gps_latitude,
            "drone_lon": telemetry.gps_longitude,
            "drone_alt_abs_m": _round_or_none(telemetry.gps_altitude, 3),
            "drone_alt_rel_m": _round_or_none(telemetry.altitude_relative, 3),
            "drone_roll_deg": round(math.degrees(telemetry.roll), 3),
            "drone_pitch_deg": round(math.degrees(telemetry.pitch), 3),
            "drone_yaw_deg": round(math.degrees(telemetry.yaw), 3),
            "gps_hdop": _round_or_none(telemetry.gps_hdop, 3),
            "gps_satellites": int(telemetry.gps_satellites),
            "ground_speed_mps": round(float(telemetry.ground_speed), 3),
        }
    )

    estimate = estimate_ground_target(
        cfg,
        telemetry,
        pixel_x=bbox_cx,
        pixel_y=bbox_cy,
    )
    record.update(estimate)
    return record


def csv_row_from_record(record: Dict[str, Any]) -> list[Any]:
    return [record.get(column, "") for column in GEOLOCATION_CSV_HEADER]


def estimate_ground_target(
    cfg: Config,
    telemetry: PixhawkTelemetry,
    *,
    pixel_x: float,
    pixel_y: float,
) -> Dict[str, Any]:
    if telemetry.gps_latitude is None or telemetry.gps_longitude is None:
        return _invalid_estimate("missing_gps")

    if telemetry.altitude_relative <= cfg.geolocation_min_altitude_m:
        return _invalid_estimate("altitude_too_low")

    lat_rad = math.radians(telemetry.gps_latitude)
    cos_lat = math.cos(lat_rad)
    if abs(cos_lat) < 1e-6:
        return _invalid_estimate("latitude_too_close_to_pole")

    ray_body = _camera_ray_body(cfg, pixel_x, pixel_y)
    ray_ned = _mat_vec_mul(
        _rotation_ned_from_body(telemetry.roll, telemetry.pitch, telemetry.yaw),
        ray_body,
    )
    ray_ned = _normalize(ray_ned)

    if ray_ned[2] <= 1e-6:
        return _invalid_estimate("ray_not_pointing_to_ground")

    slant_range_m = telemetry.altitude_relative / ray_ned[2]
    north_m = slant_range_m * ray_ned[0]
    east_m = slant_range_m * ray_ned[1]
    ground_distance_m = math.hypot(north_m, east_m)

    target_lat = telemetry.gps_latitude + math.degrees(north_m / EARTH_RADIUS_M)
    target_lon = telemetry.gps_longitude + math.degrees(east_m / (EARTH_RADIUS_M * cos_lat))

    return {
        "target_valid": True,
        "target_reason": "",
        "target_lat": round(target_lat, 8),
        "target_lon": round(target_lon, 8),
        "target_north_m": round(north_m, 3),
        "target_east_m": round(east_m, 3),
        "ground_distance_m": round(ground_distance_m, 3),
        "slant_range_m": round(slant_range_m, 3),
        "ray_down_component": round(ray_ned[2], 6),
    }


def _camera_ray_body(cfg: Config, pixel_x: float, pixel_y: float) -> tuple[float, float, float]:
    fx, fy, cx, cy = _camera_intrinsics(cfg)
    ray_camera = _normalize(
        (
            (pixel_x - cx) / fx,
            (pixel_y - cy) / fy,
            1.0,
        )
    )
    ray_body_nominal = _mat_vec_mul(CAMERA_TO_BODY_FORWARD, ray_camera)
    ray_body = _mat_vec_mul(
        _rotation_ned_from_body(
            math.radians(cfg.camera_mount_roll_deg),
            math.radians(cfg.camera_mount_pitch_deg),
            math.radians(cfg.camera_mount_yaw_deg),
        ),
        ray_body_nominal,
    )
    return _normalize(ray_body)


def _camera_intrinsics(cfg: Config) -> tuple[float, float, float, float]:
    effective_sensor_height_mm = _effective_sensor_height_mm(cfg)
    fx = cfg.rgb_width * cfg.rgb_lens_focal_length_mm / cfg.rgb_sensor_width_mm
    fy = cfg.rgb_height * cfg.rgb_lens_focal_length_mm / effective_sensor_height_mm
    cx = cfg.rgb_width / 2.0
    cy = cfg.rgb_height / 2.0
    return fx, fy, cx, cy


def _effective_sensor_height_mm(cfg: Config) -> float:
    if cfg.rgb_effective_sensor_height_mm is not None:
        return cfg.rgb_effective_sensor_height_mm
    full_width_crop_height = cfg.rgb_sensor_width_mm * cfg.rgb_height / cfg.rgb_width
    return min(cfg.rgb_sensor_height_mm, full_width_crop_height)


def _rotation_ned_from_body(roll: float, pitch: float, yaw: float) -> tuple[tuple[float, ...], ...]:
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    return (
        (
            cp * cy,
            sr * sp * cy - cr * sy,
            cr * sp * cy + sr * sy,
        ),
        (
            cp * sy,
            sr * sp * sy + cr * cy,
            cr * sp * sy - sr * cy,
        ),
        (
            -sp,
            sr * cp,
            cr * cp,
        ),
    )


def _mat_vec_mul(matrix: tuple[tuple[float, ...], ...], vector: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    )


def _normalize(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
    if norm <= 0:
        return (0.0, 0.0, 1.0)
    return (vector[0] / norm, vector[1] / norm, vector[2] / norm)


def _invalid_estimate(reason: str) -> Dict[str, Any]:
    return {
        "target_valid": False,
        "target_reason": reason,
        "target_lat": None,
        "target_lon": None,
        "target_north_m": None,
        "target_east_m": None,
        "ground_distance_m": None,
        "slant_range_m": None,
        "ray_down_component": None,
    }


def _round_or_none(value: Optional[float], digits: int) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)
