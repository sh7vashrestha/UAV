from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional

from .config import Config
from .telemetry import PixhawkTelemetry


EARTH_RADIUS_M = 6378137.0
# Camera rays start in OpenCV camera coordinates (x=right, y=down, z=forward)
# and are remapped into the recorder's body frame convention (x=forward, y=right, z=down).
CAMERA_TO_BODY_FORWARD = (
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
)


@dataclass
class CropGeolocationRecord:
    estimate_method: str = "flat_ground_bbox_center_v1"
    capture_wall_time: str = ""
    clock_source: str = "SYSTEM"
    flight_name: str = ""
    infer_id: int = 0
    t_sec: float = 0.0
    detection_index: int = 0
    crop_file: str = ""
    crop_path: str = ""
    class_id: int = 0
    class_name: str = ""
    conf: float = 0.0
    x1: int = 0
    y1: int = 0
    x2: int = 0
    y2: int = 0
    bbox_cx: float = 0.0
    bbox_cy: float = 0.0
    image_width: int = 0
    image_height: int = 0
    drone_lat: Optional[float] = None
    drone_lon: Optional[float] = None
    drone_alt_abs_m: Optional[float] = None
    drone_alt_rel_m: Optional[float] = None
    drone_roll_deg: Optional[float] = None
    drone_pitch_deg: Optional[float] = None
    drone_yaw_deg: Optional[float] = None
    gps_hdop: Optional[float] = None
    gps_satellites: Optional[int] = None
    ground_speed_mps: Optional[float] = None
    camera_focal_mm: float = 0.0
    sensor_width_mm: float = 0.0
    effective_sensor_height_mm: float = 0.0
    camera_roll_deg: float = 0.0
    camera_pitch_deg: float = 0.0
    camera_yaw_deg: float = 0.0
    target_valid: bool = False
    target_reason: str = "pixhawk_disabled_or_missing"
    target_lat: Optional[float] = None
    target_lon: Optional[float] = None
    target_north_m: Optional[float] = None
    target_east_m: Optional[float] = None
    ground_distance_m: Optional[float] = None
    slant_range_m: Optional[float] = None
    ray_down_component: Optional[float] = None
    target_point_mode: str = "bbox_center"
    target_pixel_x: Optional[float] = None
    target_pixel_y: Optional[float] = None
    gps_quality_ok: bool = False
    gps_quality_reason: str = "unchecked"
    estimate_surface_model: str = "flat_ground"
    estimate_altitude_source: str = "unresolved"
    estimate_altitude_used_m: Optional[float] = None
    estimate_assumptions: str = (
        "flat_ground,no_lens_distortion,no_camera_lever_arm,"
        "drone_gps_used_as_camera_origin"
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def csv_row(self) -> list[Any]:
        return [getattr(self, field_name) for field_name in GEOLOCATION_CSV_HEADER]


GEOLOCATION_CSV_HEADER = [field.name for field in fields(CropGeolocationRecord)]


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
) -> CropGeolocationRecord:
    x1, y1, x2, y2 = bbox
    bbox_cx = (x1 + x2) / 2.0
    bbox_cy = (y1 + y2) / 2.0
    target_point_mode, target_pixel_x, target_pixel_y = _select_target_pixel(cfg, bbox)

    record = CropGeolocationRecord(
        estimate_method=_estimate_method_name(target_point_mode),
        capture_wall_time=capture_wall_time,
        clock_source=clock_source,
        flight_name=flight_name,
        infer_id=int(infer_id),
        t_sec=round(float(t_sec), 3),
        detection_index=int(detection_index),
        crop_file=crop_path.name,
        crop_path=str(crop_path),
        class_id=int(class_id),
        class_name=str(class_name),
        conf=round(float(conf), 4),
        x1=int(x1),
        y1=int(y1),
        x2=int(x2),
        y2=int(y2),
        bbox_cx=round(bbox_cx, 2),
        bbox_cy=round(bbox_cy, 2),
        image_width=int(cfg.rgb_width),
        image_height=int(cfg.rgb_height),
        camera_focal_mm=float(cfg.rgb_lens_focal_length_mm),
        sensor_width_mm=float(cfg.rgb_sensor_width_mm),
        effective_sensor_height_mm=round(_effective_sensor_height_mm(cfg), 4),
        camera_roll_deg=float(cfg.camera_mount_roll_deg),
        camera_pitch_deg=float(cfg.camera_mount_pitch_deg),
        camera_yaw_deg=float(cfg.camera_mount_yaw_deg),
        target_point_mode=target_point_mode,
        target_pixel_x=round(target_pixel_x, 2),
        target_pixel_y=round(target_pixel_y, 2),
    )

    if telemetry is None:
        record.gps_quality_reason = "telemetry_missing"
        return record

    record.drone_lat = telemetry.gps_latitude
    record.drone_lon = telemetry.gps_longitude
    record.drone_alt_abs_m = _round_or_none(telemetry.gps_altitude, 3)
    record.drone_alt_rel_m = _round_or_none(telemetry.altitude_relative, 3)
    record.drone_roll_deg = round(math.degrees(telemetry.roll), 3)
    record.drone_pitch_deg = round(math.degrees(telemetry.pitch), 3)
    record.drone_yaw_deg = round(math.degrees(telemetry.yaw), 3)
    record.gps_hdop = _round_or_none(telemetry.gps_hdop, 3)
    record.gps_satellites = int(telemetry.gps_satellites)
    record.ground_speed_mps = round(float(telemetry.ground_speed), 3)

    estimate = estimate_ground_target(
        cfg,
        telemetry,
        pixel_x=target_pixel_x,
        pixel_y=target_pixel_y,
    )
    for key, value in estimate.items():
        setattr(record, key, value)
    return record


def csv_row_from_record(record: CropGeolocationRecord) -> list[Any]:
    return record.csv_row()


def estimate_ground_target(
    cfg: Config,
    telemetry: PixhawkTelemetry,
    *,
    pixel_x: float,
    pixel_y: float,
) -> Dict[str, Any]:
    altitude_source, altitude_m = _estimate_altitude_m(cfg, telemetry)
    estimate_metadata = {
        "gps_quality_ok": False,
        "gps_quality_reason": "unchecked",
        "estimate_surface_model": "flat_ground",
        "estimate_altitude_source": altitude_source,
        "estimate_altitude_used_m": _round_or_none(altitude_m, 3),
        "estimate_assumptions": _estimate_assumptions(),
    }

    if telemetry.gps_latitude is None or telemetry.gps_longitude is None:
        return _invalid_estimate("missing_gps", **estimate_metadata)

    gps_quality_ok, gps_quality_reason = _gps_quality_status(cfg, telemetry)
    estimate_metadata["gps_quality_ok"] = gps_quality_ok
    estimate_metadata["gps_quality_reason"] = gps_quality_reason
    if not gps_quality_ok:
        return _invalid_estimate("gps_quality_insufficient", **estimate_metadata)

    if altitude_m is None:
        return _invalid_estimate("altitude_unavailable", **estimate_metadata)

    lat_rad = math.radians(telemetry.gps_latitude)
    cos_lat = math.cos(lat_rad)
    if abs(cos_lat) < 1e-6:
        return _invalid_estimate("latitude_too_close_to_pole", **estimate_metadata)

    pixel_x, pixel_y = _undistort_target_pixel(cfg, pixel_x, pixel_y)
    ray_body = _camera_ray_body(cfg, pixel_x, pixel_y)
    body_to_ned = _rotation_ned_from_body(telemetry.roll, telemetry.pitch, telemetry.yaw)
    ray_ned = _mat_vec_mul(body_to_ned, ray_body)
    ray_ned = _normalize(ray_ned)

    if ray_ned[2] <= 1e-6:
        return _invalid_estimate("ray_not_pointing_to_ground", **estimate_metadata)

    camera_offset_ned = _camera_origin_offset_ned_m(cfg, telemetry, body_to_ned)
    altitude_used_m = altitude_m - camera_offset_ned[2]
    estimate_metadata["estimate_altitude_used_m"] = round(altitude_used_m, 3)
    if altitude_used_m <= cfg.geolocation_min_altitude_m:
        return _invalid_estimate("altitude_too_low", **estimate_metadata)

    slant_range_m = altitude_used_m / ray_ned[2]
    north_m = camera_offset_ned[0] + (slant_range_m * ray_ned[0])
    east_m = camera_offset_ned[1] + (slant_range_m * ray_ned[1])
    ground_distance_m = math.hypot(north_m, east_m)

    target_lat = telemetry.gps_latitude + math.degrees(north_m / EARTH_RADIUS_M)
    target_lon = telemetry.gps_longitude + math.degrees(east_m / (EARTH_RADIUS_M * cos_lat))

    return {
        **estimate_metadata,
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


def _select_target_pixel(cfg: Config, bbox: tuple[int, int, int, int]) -> tuple[str, float, float]:
    x1, y1, x2, y2 = bbox
    target_x = (x1 + x2) / 2.0
    if cfg.geolocation_target_point_mode == "bbox_bottom_center":
        # YOLO boxes are used with Python slicing, so y2 is the first row outside the crop.
        target_y = float(max(y1, y2 - 1))
        return "bbox_bottom_center", target_x, target_y
    target_y = (y1 + y2) / 2.0
    return "bbox_center", target_x, target_y


def _estimate_method_name(target_point_mode: str) -> str:
    if target_point_mode == "bbox_bottom_center":
        return "flat_ground_bbox_bottom_center_v1"
    return "flat_ground_bbox_center_v1"


def _gps_quality_status(cfg: Config, telemetry: PixhawkTelemetry) -> tuple[bool, str]:
    if telemetry.gps_satellites < cfg.geolocation_min_gps_satellites:
        return False, "gps_satellites_below_min"
    if cfg.geolocation_max_hdop is not None:
        if telemetry.gps_hdop is None:
            return False, "gps_hdop_missing"
        if telemetry.gps_hdop > cfg.geolocation_max_hdop:
            return False, "gps_hdop_above_max"
    return True, ""


def _estimate_altitude_m(cfg: Config, telemetry: PixhawkTelemetry) -> tuple[str, Optional[float]]:
    if cfg.geolocation_altitude_source == "relative":
        return "relative_altitude", float(telemetry.altitude_relative)

    agl_altitude = _estimate_altitude_agl_m(cfg, telemetry)
    if agl_altitude is not None:
        return "agl", agl_altitude
    return "relative_altitude_fallback", float(telemetry.altitude_relative)


def _estimate_altitude_agl_m(cfg: Config, telemetry: PixhawkTelemetry) -> Optional[float]:
    # Extension hook: future terrain/DEM/rangefinder-based AGL can plug in here.
    _ = cfg, telemetry
    return None


def _undistort_target_pixel(cfg: Config, pixel_x: float, pixel_y: float) -> tuple[float, float]:
    # Extension hook: future lens calibration can remap image points before ray construction.
    _ = cfg
    return _clamp_pixel(cfg, pixel_x, pixel_y)


def _camera_origin_offset_ned_m(
    cfg: Config,
    telemetry: PixhawkTelemetry,
    body_to_ned: tuple[tuple[float, ...], ...],
) -> tuple[float, float, float]:
    # Extension hook: future GPS-to-camera lever-arm offsets can be rotated in here.
    lever_arm_body = _camera_lever_arm_body_m(cfg, telemetry)
    return _mat_vec_mul(body_to_ned, lever_arm_body)


def _camera_lever_arm_body_m(cfg: Config, telemetry: PixhawkTelemetry) -> tuple[float, float, float]:
    _ = cfg, telemetry
    return (0.0, 0.0, 0.0)


def _estimate_assumptions() -> str:
    return "flat_ground,no_lens_distortion,no_camera_lever_arm,drone_gps_used_as_camera_origin"


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
    # MAVLink attitude is interpreted here as body FRD -> local NED.
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


def _clamp_pixel(cfg: Config, pixel_x: float, pixel_y: float) -> tuple[float, float]:
    return (
        min(max(pixel_x, 0.0), max(cfg.rgb_width - 1, 0)),
        min(max(pixel_y, 0.0), max(cfg.rgb_height - 1, 0)),
    )


def _invalid_estimate(reason: str, **metadata: Any) -> Dict[str, Any]:
    result = {
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
    result.update(metadata)
    return result


def _round_or_none(value: Optional[float], digits: int) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)
