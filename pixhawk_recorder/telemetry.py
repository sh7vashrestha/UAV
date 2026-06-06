from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class PixhawkTelemetry:
    timestamp: float = 0.0

    connection_status: str = "DISCONNECTED"
    vehicle_system_id: Optional[int] = None
    vehicle_component_id: Optional[int] = None

    is_armed: bool = False
    flight_mode: str = "UNKNOWN"

    battery_voltage: float = 0.0
    battery_current: float = 0.0
    battery_remaining: float = 100.0
    battery_consumed_mah: float = 0.0

    altitude_absolute: float = 0.0
    altitude_relative: float = 0.0
    gps_latitude: Optional[float] = None
    gps_longitude: Optional[float] = None
    gps_altitude: Optional[float] = None
    gps_hdop: Optional[float] = None
    gps_satellites: int = 0

    velocity_x: float = 0.0
    velocity_y: float = 0.0
    velocity_z: float = 0.0
    ground_speed: float = 0.0
    airspeed: Optional[float] = None

    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    temperature: Optional[float] = None

    frame_count: int = 0
    inference_count: int = 0
    detected_objects: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ErrorTelemetry:
    frame_dropped: int = 0
    inference_errors: int = 0
    writer_errors: int = 0
    checkpoint_errors: int = 0
    mavlink_errors: int = 0
    gps_loss_events: int = 0
    battery_warnings: int = 0
    power_cycles: int = 0
    thermal_errors: int = 0
