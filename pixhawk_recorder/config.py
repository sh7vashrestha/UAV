from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .postflight_merge import available_streams


@dataclass
class Config:
    # Pixhawk / MAVLink
    use_pixhawk: bool = False
    mavlink_port: str = "/dev/serial/by-id/usb-ArduPilot_Pixhawk6X_210047000451333438363734-if00"
    mavlink_baudrate: int = 115200
    mavlink_timeout: float = 3.0
    mavlink_retry_interval: float = 2.0
    prefer_pixhawk_time: bool = True

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
    rgb_height: int = 720
    rgb_capture_fps: int = 30
    rgb_record_fps: int = 30
    rgb_annot_fps: int = 30
    rgb_sensor_width_mm: float = 6.287
    rgb_sensor_height_mm: float = 4.712
    rgb_effective_sensor_height_mm: Optional[float] = None
    rgb_lens_focal_length_mm: float = 6.0
    camera_mount_roll_deg: float = 0.0
    camera_mount_pitch_deg: float = -45.0
    camera_mount_yaw_deg: float = 0.0

    # Thermal
    thermal_device = "/dev/v4l/by-id/usb-GroupGets_PureThermal__fw:v1.3.0__80020023-5103-3133-3332-373300000000-video-index0"
    thermal_width: int = 160
    thermal_height: int = 120
    thermal_capture_fps: int = 9
    thermal_record_fps: int = 9
    require_thermal: bool = False

    # Encoding
    rgb_raw_bitrate: int = 3500
    rgb_annot_bitrate: int = 2200
    thermal_bitrate: int = 500
    x264_preset: str = "ultrafast"
    x264_threads: int = 2
    x264_keyint: int = 30

    # Segments / checkpointing
    segment_duration_sec: int = 120
    checkpoint_interval_sec: int = 15
    csv_flush_interval_sec: float = 2.0
    storage_check_interval_sec: int = 30
    auto_merge_after_flight: bool = True
    auto_merge_streams: tuple[str, ...] = ("raw", "annotated", "thermal")
    auto_merge_ffmpeg: str = "ffmpeg"
    auto_merge_overwrite: bool = False
    auto_merge_min_segments: int = 2

    # Detection
    inference_fps: int = 8
    inference_width: int = 512
    inference_height: int = 288
    img_size: int = 288
    conf_threshold: float = 0.50
    iou_threshold: float = 0.45
    fp16_inference: bool = True
    save_csv: bool = True
    save_crops: bool = True
    save_crop_geolocation_csv: bool = True
    save_crop_geolocation_sidecars: bool = False
    geolocation_min_altitude_m: float = 0.5

    # Streaming
    use_video_stream: bool = True
    stream_host: str = "10.42.0.223"
    stream_port: int = 5555
    stream_width: int = 960
    stream_height: int = 540
    stream_fps: int = 15
    stream_bitrate: int = 1500

    # Reliability
    checkpoint_retries: int = 2

    # Battery thresholds - defaults assume 6S; adjust for your actual pack.
    battery_warning_voltage: float = 21.6
    battery_critical_voltage: float = 20.4
    battery_critical_percent: float = 5.0

    # Telemetry logging
    telemetry_log_interval_sec: float = 0.05

    def __post_init__(self) -> None:
        if self.run_name is None:
            self.run_name = time.strftime("flight_%Y-%m-%d_%H-%M-%S")
        invalid_streams = sorted(set(self.auto_merge_streams) - set(available_streams()))
        if invalid_streams:
            raise ValueError(
                f"Invalid auto_merge_streams: {', '.join(invalid_streams)}. "
                f"Expected one of: {', '.join(available_streams())}"
            )
        if self.auto_merge_min_segments < 1:
            raise ValueError("auto_merge_min_segments must be at least 1")
        if self.rgb_sensor_width_mm <= 0 or self.rgb_sensor_height_mm <= 0:
            raise ValueError("RGB sensor dimensions must be positive")
        if self.rgb_effective_sensor_height_mm is not None and self.rgb_effective_sensor_height_mm <= 0:
            raise ValueError("rgb_effective_sensor_height_mm must be positive when set")
        if self.rgb_lens_focal_length_mm <= 0:
            raise ValueError("rgb_lens_focal_length_mm must be positive")
        if self.geolocation_min_altitude_m < 0:
            raise ValueError("geolocation_min_altitude_m must be non-negative")
