from __future__ import annotations

import threading
import time
from dataclasses import asdict
from typing import Optional

from .config import Config
from .telemetry import PixhawkTelemetry
from .time_sync import PixhawkTimeSync

try:
    from pymavlink import mavutil
    HAS_MAVLINK = True
except ImportError:
    HAS_MAVLINK = False


class PixhawkManager:
    def __init__(self, cfg: Config, logger):
        self.cfg = cfg
        self.logger = logger
        self.master = None
        self.running = True
        self._armed = False
        self.telemetry = PixhawkTelemetry()
        self.lock = threading.Lock()
        self.connection_status = "DISCONNECTED"
        self.last_heartbeat = 0.0
        self.error_count = 0
        self.enabled = cfg.use_pixhawk
        self.thread = None
        self.time_sync = PixhawkTimeSync()

        if not self.enabled:
            self.logger.info("Pixhawk disabled - manual mode")
            return

        self._connect()
        self.thread = threading.Thread(target=self._telemetry_loop, daemon=True, name="PixhawkManager")
        self.thread.start()

    def _connect(self) -> bool:
        if not HAS_MAVLINK:
            self.logger.warning("pymavlink not available - install with: pip install pymavlink")
            return False

        try:
            self.logger.info(
                "Connecting to Pixhawk on %s @ %d baud...",
                self.cfg.mavlink_port,
                self.cfg.mavlink_baudrate,
            )
            self.master = mavutil.mavlink_connection(
                self.cfg.mavlink_port,
                baud=self.cfg.mavlink_baudrate,
                timeout=self.cfg.mavlink_timeout,
                autoreconnect=True,
                source_system=250,
            )
            self.logger.info("Waiting for heartbeat from Pixhawk...")
            self.master.wait_heartbeat(timeout=self.cfg.mavlink_timeout)
            self.last_heartbeat = time.time()
            self.connection_status = "CONNECTED"
            self.logger.info(
                "Heartbeat received from Pixhawk (sys=%s comp=%s)",
                self.master.target_system,
                self.master.target_component,
            )
            self._request_streams()
            return True
        except Exception as e:
            self.logger.error("Failed to connect to Pixhawk: %s", e)
            self.error_count += 1
            self.master = None
            self.connection_status = "DISCONNECTED"
            return False

    def _request_streams(self) -> None:
        if self.master is None or not HAS_MAVLINK:
            return
        try:
            self.master.mav.request_data_stream_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_ALL,
                10,
                1,
            )
            self.logger.info("Requested MAVLink data streams @ 10 Hz")
        except Exception as e:
            self.logger.warning("Could not request MAVLink streams: %s", e)

    def _decode_flight_mode(self, custom_mode: int) -> str:
        modes = {
            0: "STABILIZE",
            1: "ACRO",
            2: "ALTHOLD",
            3: "AUTO",
            4: "GUIDED",
            5: "LOITER",
            6: "RTL",
            7: "CIRCLE",
            8: "POSHOLD",
            9: "BRAKE",
            10: "DRIFT",
            11: "SPORT",
            13: "AUTOTUNE",
            15: "SMART_RTL",
            16: "FLOWHOLD",
            17: "FOLLOW",
            18: "ZIGZAG",
            19: "SYSTEMID",
            20: "AUTOROTATE",
            21: "AUTO_RTL",
        }
        return modes.get(custom_mode, f"MODE_{custom_mode}")

    def _handle_disconnect(self, reason: str) -> None:
        self.logger.warning("Pixhawk connection lost: %s", reason)
        self.connection_status = reason
        with self.lock:
            self._armed = False
            self.telemetry.is_armed = False
            self.telemetry.flight_mode = "DISCONNECTED"
        try:
            if self.master is not None:
                self.master.close()
        except Exception:
            pass
        self.master = None

    def _telemetry_loop(self) -> None:
        while self.running:
            try:
                if self.master is None:
                    time.sleep(self.cfg.mavlink_retry_interval)
                    self._connect()
                    continue

                msg = self.master.recv_match(blocking=False)
                if msg is None:
                    if self.last_heartbeat > 0 and (time.time() - self.last_heartbeat > 3.0):
                        self._handle_disconnect("HEARTBEAT_TIMEOUT")
                    time.sleep(0.01)
                    continue

                msg_type = msg.get_type()
                if msg_type == "BAD_DATA":
                    continue

                sync_source = self.time_sync.update_from_message(msg)
                if sync_source is not None:
                    self.logger.info(
                        "Pixhawk time sync acquired from %s: %s",
                        sync_source,
                        self.time_sync.strftime("%Y-%m-%d %H:%M:%S"),
                    )

                if msg_type == "HEARTBEAT":
                    self.last_heartbeat = time.time()
                    with self.lock:
                        self._armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                        self.telemetry.is_armed = self._armed
                        self.telemetry.flight_mode = self._decode_flight_mode(msg.custom_mode)
                    self.connection_status = "CONNECTED"

                elif msg_type == "SYS_STATUS":
                    with self.lock:
                        if getattr(msg, "voltage_battery", -1) != -1:
                            self.telemetry.battery_voltage = msg.voltage_battery / 1000.0
                        if getattr(msg, "current_battery", -1) != -1:
                            self.telemetry.battery_current = msg.current_battery / 100.0
                        if getattr(msg, "battery_remaining", -1) != -1:
                            self.telemetry.battery_remaining = msg.battery_remaining

                elif msg_type == "GLOBAL_POSITION_INT":
                    with self.lock:
                        self.telemetry.gps_latitude = msg.lat / 1e7
                        self.telemetry.gps_longitude = msg.lon / 1e7
                        self.telemetry.gps_altitude = msg.alt / 1000.0
                        self.telemetry.altitude_absolute = msg.alt / 1000.0
                        self.telemetry.altitude_relative = getattr(msg, "relative_alt", 0) / 1000.0
                        self.telemetry.velocity_x = msg.vx / 100.0
                        self.telemetry.velocity_y = msg.vy / 100.0
                        self.telemetry.velocity_z = msg.vz / 100.0

                elif msg_type == "LOCAL_POSITION_NED":
                    with self.lock:
                        self.telemetry.altitude_relative = -msg.z
                        self.telemetry.velocity_x = msg.vx
                        self.telemetry.velocity_y = msg.vy
                        self.telemetry.velocity_z = msg.vz

                elif msg_type == "ATTITUDE":
                    with self.lock:
                        self.telemetry.roll = msg.roll
                        self.telemetry.pitch = msg.pitch
                        self.telemetry.yaw = msg.yaw

                elif msg_type == "GPS_RAW_INT":
                    with self.lock:
                        self.telemetry.gps_satellites = getattr(msg, "satellites_visible", 0)
                        eph = getattr(msg, "eph", 65535)
                        if eph != 65535:
                            self.telemetry.gps_hdop = eph / 100.0

                elif msg_type == "VFR_HUD":
                    with self.lock:
                        self.telemetry.ground_speed = msg.groundspeed
                        self.telemetry.airspeed = msg.airspeed if msg.airspeed > 0 else None
                        self.telemetry.altitude_relative = msg.alt

            except Exception as e:
                self.error_count += 1
                self.logger.error("Telemetry loop error: %s", e)
                time.sleep(0.1)

    def get_telemetry(self) -> PixhawkTelemetry:
        with self.lock:
            return PixhawkTelemetry(**asdict(self.telemetry))

    def is_connected(self) -> bool:
        return self.master is not None and self.connection_status == "CONNECTED"

    def is_armed(self) -> bool:
        with self.lock:
            return self._armed

    def has_time(self) -> bool:
        return self.time_sync.has_time()

    def now_epoch(self, fallback_to_system: bool = True) -> float:
        return self.time_sync.now_epoch(fallback_to_system)

    def strftime(self, fmt: str, fallback_to_system: bool = True) -> str:
        return self.time_sync.strftime(fmt, fallback_to_system)

    def time_source(self) -> Optional[str]:
        return self.time_sync.source()

    def stop(self) -> None:
        self.running = False
        if self.thread:
            try:
                self.thread.join(timeout=1.0)
            except Exception:
                pass
        try:
            if self.master is not None:
                self.master.close()
        except Exception:
            pass
        self.master = None
