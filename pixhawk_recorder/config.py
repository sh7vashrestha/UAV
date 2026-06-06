from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .damage_aggregation import (
    DEFAULT_CONSENSUS_METHOD,
    DEFAULT_SPATIAL_THRESHOLD_M,
    DEFAULT_TEMPORAL_THRESHOLD_SEC,
    DEFAULT_TOP_K,
    validate_aggregation_options,
)
from .postflight_merge import available_streams


@dataclass
class Config:
    # Pixhawk / MAVLink
    use_pixhawk: bool = False
    # use_pixhawk: bool = True
    mavlink_port: str = "/dev/serial/by-id/usb-ArduPilot_Pixhawk6X_210047000451333438363734-if00"
    mavlink_baudrate: int = 115200
    mavlink_timeout: float = 3.0
    mavlink_retry_interval: float = 2.0
    mavlink_heartbeat_loss_timeout: float = 5.0
    mavlink_arm_debounce_sec: float = 1.0
    mavlink_disarm_debounce_sec: float = 5.0
    mavlink_armed_disconnect_hold_sec: float = 20.0
    prefer_pixhawk_time: bool = False

    # YOLO
    yolo_dir: Path = Path("/home/drone/yolov5")
    weights: Path = Path(
        "/home/drone/uav_mission/models/"
        "dp_finetune_clean_yolov5fmt_CLEAN_FOR_INFERENCE.pt"
    )

    # Storage
    mount_device: str = "/dev/mmcblk0p1"
    mount_point: Path = Path("/media/sdcard/uav")
    auto_mount_exfat: bool = True
    min_free_gb: float = 2.0
    save_root: Path = Path("/media/sdcard/uav/flights")
    run_name: Optional[str] = None

    # RGB camera
    rgb_sensor_id: int = 0
    rgb_width: int = 1280
    # rgb_width: int = 1920
    rgb_height: int = 720
    # rgb_height: int = 1080
    rgb_capture_fps: int = 30
    rgb_record_fps: int = 30
    rgb_annot_fps: int = 30
    rgb_flip_method: int = 2
    rgb_sensor_width_mm: float = 6.287
    rgb_sensor_height_mm: float = 4.712
    rgb_effective_sensor_height_mm: Optional[float] = None
    rgb_lens_focal_length_mm: float = 6.0
    camera_mount_roll_deg: float = 0.0
    camera_mount_pitch_deg: float = -45.0
    camera_mount_yaw_deg: float = 0.0

    # Thermal
    thermal_device: str = "/dev/v4l/by-id/usb-GroupGets_PureThermal__fw:v1.3.0__80020023-5103-3133-3332-373300000000-video-index0"
    thermal_width: int = 160
    thermal_height: int = 120
    thermal_capture_fps: int = 9
    thermal_record_fps: int = 9
    require_thermal: bool = False

    # Encoding
    rgb_raw_bitrate: int = 6000
    rgb_annot_bitrate: int = 5000
    thermal_bitrate: int = 800
    x264_preset: str = "ultrafast"
    x264_threads: int = 2
    x264_keyint: int = 30

    # Segments / checkpointing
    segment_duration_sec: int = 60
    checkpoint_interval_sec: int = 15
    csv_flush_interval_sec: float = 2.0
    storage_check_interval_sec: int = 30
    auto_merge_after_flight: bool = True
    auto_merge_streams: tuple[str, ...] = ("raw", "annotated", "thermal")
    auto_merge_ffmpeg: str = "ffmpeg"
    auto_merge_overwrite: bool = False
    auto_merge_min_segments: int = 2
    auto_aggregate_damage_records_after_flight: bool = True
    damage_record_spatial_threshold_m: float = DEFAULT_SPATIAL_THRESHOLD_M
    damage_record_temporal_threshold_sec: float = DEFAULT_TEMPORAL_THRESHOLD_SEC
    damage_record_consensus_method: str = DEFAULT_CONSENSUS_METHOD
    damage_record_top_k: int = DEFAULT_TOP_K
    damage_record_debug_raw_detections: bool = False

    # Detection
    inference_fps: int = 4
    inference_width: int = 512
    inference_height: int = 288
    img_size: int = 288
    conf_threshold: float = 0.50
    iou_threshold: float = 0.45
    fp16_inference: bool = True
    annot_box_color_bgr: tuple[int, int, int] = (255, 0, 255)
    annot_class_colors_bgr: tuple[tuple[int, int, int], ...] = (
        (255, 255, 0),
        (0, 255, 255),
        (0, 165, 255),
        (0, 0, 255),
    )
    annot_box_thickness: int = 3
    annot_box_outline_color_bgr: tuple[int, int, int] = (0, 0, 0)
    annot_box_outline_thickness: int = 2
    annot_text_color_bgr: tuple[int, int, int] = (255, 255, 255)
    annot_text_bg_color_bgr: tuple[int, int, int] = (0, 0, 0)
    annot_text_scale: float = 0.9
    annot_text_thickness: int = 2
    annot_text_padding_px: int = 4
    save_csv: bool = True
    save_crops: bool = True
    save_crop_dedup_enabled: bool = True
    save_crop_dedup_cooldown_sec: float = 2.0
    save_crop_dedup_iou_threshold: float = 0.7
    save_crop_geolocation_csv: bool = True
    save_crop_geolocation_sidecars: bool = False
    geolocation_target_point_mode: str = "bbox_center"
    geolocation_min_gps_satellites: int = 6
    geolocation_max_hdop: Optional[float] = 3.0
    geolocation_altitude_source: str = "auto"
    geolocation_min_altitude_m: float = 0.5

    # Streaming
    use_video_stream: bool = True
    # For Laptop Hotspot
    stream_host: str = "192.168.137.1"
    # For Phone Hotspot
    # stream_host: str = "172.16.137.158"
    stream_port: int = 5555
    stream_width: int = 960
    stream_height: int = 540
    stream_fps: int = 8
    stream_bitrate: int = 1500

    # Reliability
    checkpoint_retries: int = 2

    # Battery thresholds - defaults assume 6S; adjust for your actual pack.
    battery_warning_voltage: float = 22.6
    battery_critical_voltage: float = 20.4
    battery_critical_percent: float = 5.0

    # Telemetry logging
    telemetry_log_interval_sec: float = 0.05

    def __post_init__(self) -> None:
        invalid_streams = sorted(set(self.auto_merge_streams) - set(available_streams()))
        if invalid_streams:
            raise ValueError(
                f"Invalid auto_merge_streams: {', '.join(invalid_streams)}. "
                f"Expected one of: {', '.join(available_streams())}"
            )
        if self.auto_merge_min_segments < 1:
            raise ValueError("auto_merge_min_segments must be at least 1")
        validate_aggregation_options(
            spatial_threshold_m=self.damage_record_spatial_threshold_m,
            temporal_threshold_sec=self.damage_record_temporal_threshold_sec,
            consensus_method=self.damage_record_consensus_method,
            top_k=self.damage_record_top_k,
        )
        if self.mavlink_timeout <= 0:
            raise ValueError("mavlink_timeout must be positive")
        if self.mavlink_retry_interval <= 0:
            raise ValueError("mavlink_retry_interval must be positive")
        if self.mavlink_heartbeat_loss_timeout <= 0:
            raise ValueError("mavlink_heartbeat_loss_timeout must be positive")
        if self.mavlink_arm_debounce_sec < 0:
            raise ValueError("mavlink_arm_debounce_sec must be non-negative")
        if self.mavlink_disarm_debounce_sec < 0:
            raise ValueError("mavlink_disarm_debounce_sec must be non-negative")
        if self.mavlink_armed_disconnect_hold_sec < 0:
            raise ValueError("mavlink_armed_disconnect_hold_sec must be non-negative")
        if self.rgb_sensor_width_mm <= 0 or self.rgb_sensor_height_mm <= 0:
            raise ValueError("RGB sensor dimensions must be positive")
        if self.rgb_flip_method not in range(8):
            raise ValueError("rgb_flip_method must be between 0 and 7")
        if self.rgb_effective_sensor_height_mm is not None and self.rgb_effective_sensor_height_mm <= 0:
            raise ValueError("rgb_effective_sensor_height_mm must be positive when set")
        if self.rgb_lens_focal_length_mm <= 0:
            raise ValueError("rgb_lens_focal_length_mm must be positive")
        self._validate_bgr_color("annot_box_color_bgr", self.annot_box_color_bgr)
        for index, color in enumerate(self.annot_class_colors_bgr):
            self._validate_bgr_color(f"annot_class_colors_bgr[{index}]", color)
        self._validate_bgr_color("annot_box_outline_color_bgr", self.annot_box_outline_color_bgr)
        self._validate_bgr_color("annot_text_color_bgr", self.annot_text_color_bgr)
        self._validate_bgr_color("annot_text_bg_color_bgr", self.annot_text_bg_color_bgr)
        if self.annot_box_thickness < 1:
            raise ValueError("annot_box_thickness must be at least 1")
        if self.annot_box_outline_thickness < 0:
            raise ValueError("annot_box_outline_thickness must be non-negative")
        if self.annot_text_scale <= 0:
            raise ValueError("annot_text_scale must be positive")
        if self.annot_text_thickness < 1:
            raise ValueError("annot_text_thickness must be at least 1")
        if self.annot_text_padding_px < 0:
            raise ValueError("annot_text_padding_px must be non-negative")
        if self.save_crop_dedup_cooldown_sec < 0:
            raise ValueError("save_crop_dedup_cooldown_sec must be non-negative")
        if not 0.0 <= self.save_crop_dedup_iou_threshold <= 1.0:
            raise ValueError("save_crop_dedup_iou_threshold must be between 0.0 and 1.0")
        if self.geolocation_target_point_mode not in {"bbox_center", "bbox_bottom_center"}:
            raise ValueError(
                "geolocation_target_point_mode must be 'bbox_center' or 'bbox_bottom_center'"
            )
        if self.geolocation_min_gps_satellites < 0:
            raise ValueError("geolocation_min_gps_satellites must be non-negative")
        if self.geolocation_max_hdop is not None and self.geolocation_max_hdop <= 0:
            raise ValueError("geolocation_max_hdop must be positive when set")
        if self.geolocation_altitude_source not in {"auto", "relative"}:
            raise ValueError("geolocation_altitude_source must be 'auto' or 'relative'")
        if self.geolocation_min_altitude_m < 0:
            raise ValueError("geolocation_min_altitude_m must be non-negative")

    @staticmethod
    def _validate_bgr_color(name: str, value: tuple[int, int, int]) -> None:
        if len(value) != 3:
            raise ValueError(f"{name} must contain exactly 3 BGR values")
        if any(channel < 0 or channel > 255 for channel in value):
            raise ValueError(f"{name} values must be between 0 and 255")
