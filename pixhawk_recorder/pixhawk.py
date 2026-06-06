from __future__ import annotations

import errno
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
        self._raw_armed = False
        self._armed = False
        self._armed_candidate: Optional[bool] = None
        self._armed_candidate_since = 0.0
        self.telemetry = PixhawkTelemetry()
        self.lock = threading.Lock()
        self.connection_status = "DISCONNECTED"
        self.vehicle_system_id: Optional[int] = None
        self.vehicle_component_id: Optional[int] = None
        self.last_heartbeat = 0.0
        self.last_vehicle_message = 0.0
        self.error_count = 0
        self.enabled = cfg.use_pixhawk
        self.thread = None
        self.time_sync = PixhawkTimeSync()
        self.ever_connected = False
        self._ignored_heartbeat_sources: set[tuple[Optional[int], Optional[int]]] = set()
        self._last_connect_issue: Optional[str] = None
        self._set_connection_status("DISCONNECTED")

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
            heartbeat = self._wait_for_initial_vehicle_heartbeat()

            self.vehicle_system_id = self._safe_src_system(heartbeat) or self.master.target_system
            self.vehicle_component_id = self._safe_src_component(heartbeat) or self.master.target_component
            if self.vehicle_system_id is not None:
                self.master.target_system = self.vehicle_system_id
            if self.vehicle_component_id is not None:
                self.master.target_component = self.vehicle_component_id

            now = time.monotonic()
            self.last_heartbeat = now
            self.last_vehicle_message = now
            self.ever_connected = True
            self._set_connection_status("CONNECTED")
            self.logger.info(
                "Heartbeat received from Pixhawk (sys=%s comp=%s)",
                self.vehicle_system_id,
                self.vehicle_component_id,
            )
            self._apply_vehicle_heartbeat(heartbeat, now)
            self._request_streams()
            self._last_connect_issue = None
            return True
        except Exception as e:
            self.error_count += 1
            try:
                if self.master is not None:
                    self.master.close()
            except Exception:
                pass
            self.master = None

            err_no = getattr(e, "errno", None)
            msg = str(e).lower()
            if err_no == errno.EBUSY or "device or resource busy" in msg:
                self._set_connection_status("PORT_BUSY")
                self._log_connect_issue_once(
                    "PORT_BUSY",
                    "warning",
                    "Pixhawk serial port is busy: %s. Another process may still be using %s; retrying in %.1fs.",
                    e,
                    self.cfg.mavlink_port,
                    self.cfg.mavlink_retry_interval,
                )
            elif isinstance(e, TimeoutError):
                self._set_connection_status("HEARTBEAT_WAIT")
                self._log_connect_issue_once(
                    "HEARTBEAT_WAIT",
                    "warning",
                    "Opened Pixhawk serial port but did not receive a heartbeat within %.1fs; retrying in %.1fs.",
                    self.cfg.mavlink_timeout,
                    self.cfg.mavlink_retry_interval,
                )
            else:
                self._set_connection_status("DISCONNECTED")
                issue_key = f"CONNECT:{type(e).__name__}:{e}"
                self._log_connect_issue_once(issue_key, "error", "Failed to connect to Pixhawk: %s", e)
            return False

    def _wait_for_initial_vehicle_heartbeat(self):
        if self.master is None:
            raise RuntimeError("MAVLink master is not connected")

        deadline = time.monotonic() + self.cfg.mavlink_timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"No heartbeat received within {self.cfg.mavlink_timeout:.1f}s")

            heartbeat = self.master.recv_match(type="HEARTBEAT", blocking=True, timeout=remaining)
            if heartbeat is None:
                raise TimeoutError(f"No heartbeat received within {self.cfg.mavlink_timeout:.1f}s")
            if not self._is_initial_vehicle_heartbeat(heartbeat):
                continue
            return heartbeat

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

    def _log_connect_issue_once(self, issue_key: str, level: str, message: str, *args) -> None:
        if self._last_connect_issue == issue_key:
            return
        getattr(self.logger, level)(message, *args)
        self._last_connect_issue = issue_key

    def _set_connection_status(self, status: str) -> None:
        with self.lock:
            self.connection_status = status
            self.telemetry.connection_status = status
            self.telemetry.vehicle_system_id = self.vehicle_system_id
            self.telemetry.vehicle_component_id = self.vehicle_component_id

    def _safe_src_system(self, msg) -> Optional[int]:
        getter = getattr(msg, "get_srcSystem", None)
        if getter is None:
            return None
        try:
            return int(getter())
        except Exception:
            return None

    def _safe_src_component(self, msg) -> Optional[int]:
        getter = getattr(msg, "get_srcComponent", None)
        if getter is None:
            return None
        try:
            return int(getter())
        except Exception:
            return None

    def _vehicle_source(self) -> tuple[Optional[int], Optional[int]]:
        target_system = self.vehicle_system_id
        target_component = self.vehicle_component_id
        if self.master is not None:
            target_system = getattr(self.master, "target_system", 0) or target_system
            target_component = getattr(self.master, "target_component", 0) or target_component
        return target_system, target_component

    def _is_initial_vehicle_heartbeat(self, msg) -> bool:
        if not self._is_probably_vehicle_heartbeat(msg):
            self._log_ignored_heartbeat(msg, "not a vehicle/autopilot heartbeat during initial discovery")
            return False

        src_system = self._safe_src_system(msg)
        src_component = self._safe_src_component(msg)
        target_system = getattr(self.master, "target_system", 0) if self.master is not None else 0
        target_component = getattr(self.master, "target_component", 0) if self.master is not None else 0

        if target_system not in (None, 0) and src_system != target_system:
            self._log_ignored_heartbeat(
                msg,
                f"source system does not match vehicle system {target_system}",
            )
            return False

        if target_component not in (None, 0):
            if src_component != target_component:
                self._log_ignored_heartbeat(
                    msg,
                    f"source component does not match vehicle component {target_component}",
                )
                return False
            return True

        if src_component not in (None, 0, mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1):
            self._log_ignored_heartbeat(
                msg,
                "source component is not the autopilot component during initial discovery",
            )
            return False

        return True

    def _is_probably_vehicle_heartbeat(self, msg) -> bool:
        if self._safe_src_component(msg) == mavutil.mavlink.MAV_COMP_ID_GIMBAL:
            return False
        if getattr(msg, "type", None) in (
            mavutil.mavlink.MAV_TYPE_GCS,
            mavutil.mavlink.MAV_TYPE_GIMBAL,
            mavutil.mavlink.MAV_TYPE_ADSB,
            mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
        ):
            return False
        if getattr(msg, "autopilot", mavutil.mavlink.MAV_AUTOPILOT_INVALID) == mavutil.mavlink.MAV_AUTOPILOT_INVALID:
            return False
        return True

    def _log_ignored_heartbeat(self, msg, reason: str) -> None:
        src = (self._safe_src_system(msg), self._safe_src_component(msg))
        if src in self._ignored_heartbeat_sources:
            return
        self._ignored_heartbeat_sources.add(src)
        self.logger.info(
            "Ignoring HEARTBEAT from sys=%s comp=%s: %s",
            src[0],
            src[1],
            reason,
        )

    def _is_vehicle_heartbeat(self, msg) -> bool:
        if not self._is_probably_vehicle_heartbeat(msg):
            self._log_ignored_heartbeat(msg, "not a vehicle/autopilot heartbeat")
            return False

        src_system = self._safe_src_system(msg)
        src_component = self._safe_src_component(msg)
        target_system, target_component = self._vehicle_source()

        if target_system not in (None, 0) and src_system != target_system:
            self._log_ignored_heartbeat(
                msg,
                f"source system does not match vehicle system {target_system}",
            )
            return False

        if target_component not in (None, 0):
            if src_component != target_component:
                self._log_ignored_heartbeat(
                    msg,
                    f"source component does not match vehicle component {target_component}",
                )
                return False
        elif src_component != mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1:
            self._log_ignored_heartbeat(
                msg,
                "source component is not the autopilot component",
            )
            return False

        return True

    def _is_vehicle_message(self, msg) -> bool:
        target_system, _ = self._vehicle_source()
        if target_system in (None, 0):
            return True
        src_system = self._safe_src_system(msg)
        return src_system == target_system

    def _update_armed_state(self, raw_armed: bool, now: float) -> None:
        log_message = None

        with self.lock:
            self._raw_armed = raw_armed

            if raw_armed == self._armed:
                self._armed_candidate = None
                self._armed_candidate_since = 0.0
                self.telemetry.is_armed = self._armed
                return

            if self._armed_candidate != raw_armed:
                self._armed_candidate = raw_armed
                self._armed_candidate_since = now
                delay = self.cfg.mavlink_arm_debounce_sec if raw_armed else self.cfg.mavlink_disarm_debounce_sec
                log_message = (
                    f"Pixhawk armed-state change detected: "
                    f"{'ARMED' if raw_armed else 'DISARMED'} candidate "
                    f"(waiting {delay:.1f}s)"
                )
            else:
                delay = self.cfg.mavlink_arm_debounce_sec if raw_armed else self.cfg.mavlink_disarm_debounce_sec
                if now - self._armed_candidate_since >= delay:
                    self._armed = raw_armed
                    self.telemetry.is_armed = raw_armed
                    self._armed_candidate = None
                    self._armed_candidate_since = 0.0
                    log_message = f"Pixhawk armed state confirmed: {'ARMED' if raw_armed else 'DISARMED'}"

        if log_message is not None:
            self.logger.info(log_message)

    def _apply_vehicle_heartbeat(self, msg, now: float) -> None:
        self.last_heartbeat = now
        self.last_vehicle_message = now
        self._set_connection_status("CONNECTED")

        with self.lock:
            self.telemetry.flight_mode = self._decode_flight_mode(msg.custom_mode)

        raw_armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
        self._update_armed_state(raw_armed, now)

    def _log_time_sync_if_available(self, msg) -> None:
        sync_source = self.time_sync.update_from_message(msg)
        if sync_source is not None:
            self.logger.info(
                "Pixhawk time sync acquired from %s: %s",
                sync_source,
                self.time_sync.strftime("%Y-%m-%d %H:%M:%S"),
            )

    def _process_message(self, msg, now: Optional[float] = None) -> bool:
        msg_type = msg.get_type()
        if msg_type == "BAD_DATA":
            return False

        if now is None:
            now = time.monotonic()

        if msg_type == "HEARTBEAT":
            if not self._is_vehicle_heartbeat(msg):
                return False
            self._log_time_sync_if_available(msg)
            self._apply_vehicle_heartbeat(msg, now)
            return True

        if not self._is_vehicle_message(msg):
            return False

        self.last_vehicle_message = now
        self._log_time_sync_if_available(msg)

        if msg_type == "SYS_STATUS":
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

        return True

    def _handle_disconnect(self, reason: str) -> None:
        self.logger.warning("Pixhawk connection lost: %s", reason)
        with self.lock:
            self._armed_candidate = None
            self._armed_candidate_since = 0.0
        self._set_connection_status(reason)
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
                    if self.last_heartbeat > 0 and (
                        time.monotonic() - self.last_heartbeat > self.cfg.mavlink_heartbeat_loss_timeout
                    ):
                        self._handle_disconnect("HEARTBEAT_TIMEOUT")
                    time.sleep(0.01)
                    continue

                self._process_message(msg, now=time.monotonic())

            except Exception as e:
                self.error_count += 1
                self.logger.error("Telemetry loop error: %s", e)
                time.sleep(0.1)

    def get_telemetry(self) -> PixhawkTelemetry:
        with self.lock:
            return PixhawkTelemetry(**asdict(self.telemetry))

    def is_connected(self) -> bool:
        with self.lock:
            status = self.connection_status
        return self.master is not None and status == "CONNECTED"

    def is_armed(self) -> bool:
        with self.lock:
            armed = self._armed
            status = self.connection_status
            last_heartbeat = self.last_heartbeat

        if not armed:
            return False
        if status == "CONNECTED":
            return True
        if last_heartbeat <= 0:
            return False
        return (time.monotonic() - last_heartbeat) <= self.cfg.mavlink_armed_disconnect_hold_sec

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
