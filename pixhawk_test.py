#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

from pymavlink import mavutil


DEFAULT_PORT = "/dev/serial/by-id/usb-ArduPilot_Pixhawk6X_210047000451333438363734-if00"
DEFAULT_BAUD = 115200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple Pixhawk MAVLink diagnostic tool."
    )
    parser.add_argument("--port", default=DEFAULT_PORT, help="Serial device path")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD, help="Serial baud rate")
    parser.add_argument(
        "--heartbeat-timeout",
        type=float,
        default=8.0,
        help="Seconds to wait for the first heartbeat before failing",
    )
    parser.add_argument(
        "--summary-period",
        type=float,
        default=1.0,
        help="Seconds between telemetry summary prints",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Optional max runtime in seconds. 0 means run until Ctrl+C",
    )
    return parser.parse_args()


def fmt_number(value: float | None, unit: str = "", digits: int = 1) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "n/a"
    return f"{value:.{digits}f}{unit}"


def fmt_int(value: int | None) -> str:
    return "n/a" if value is None else str(value)


def request_streams(master) -> None:
    try:
        master.mav.request_data_stream_send(
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL,
            10,
            1,
        )
    except Exception as exc:
        print(f"Warning: could not request MAVLink streams: {exc}", flush=True)


def connect(port: str, baud: int, heartbeat_timeout: float):
    port_path = Path(port)
    if "/dev/" in port and not port_path.exists():
        raise FileNotFoundError(f"Serial device does not exist: {port}")

    print(f"Opening MAVLink connection on {port} @ {baud} baud", flush=True)
    master = mavutil.mavlink_connection(port, baud=baud)
    print(
        f"Waiting for heartbeat for up to {heartbeat_timeout:.1f}s...",
        flush=True,
    )
    heartbeat = master.wait_heartbeat(timeout=heartbeat_timeout)
    if heartbeat is None:
        raise TimeoutError(
            f"No heartbeat received within {heartbeat_timeout:.1f}s on {port}"
        )

    print(
        "Connected: "
        f"sys={master.target_system} comp={master.target_component} "
        f"type={heartbeat.type} autopilot={heartbeat.autopilot}",
        flush=True,
    )
    request_streams(master)
    return master


def update_state(state: dict[str, object], msg) -> None:
    msg_type = msg.get_type()

    if msg_type == "HEARTBEAT":
        state["armed"] = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
        state["mode"] = mavutil.mode_string_v10(msg)
    elif msg_type == "SYS_STATUS":
        if getattr(msg, "voltage_battery", -1) != -1:
            state["battery_voltage"] = msg.voltage_battery / 1000.0
        if getattr(msg, "current_battery", -1) != -1:
            state["battery_current"] = msg.current_battery / 100.0
        if getattr(msg, "battery_remaining", -1) != -1:
            state["battery_remaining"] = int(msg.battery_remaining)
    elif msg_type == "GPS_RAW_INT":
        state["gps_satellites"] = int(getattr(msg, "satellites_visible", 0))
        eph = getattr(msg, "eph", 65535)
        state["gps_hdop"] = None if eph == 65535 else eph / 100.0
        fix_type = getattr(msg, "fix_type", 0)
        state["gps_fix"] = {
            0: "NO_GPS",
            1: "NO_FIX",
            2: "2D",
            3: "3D",
            4: "DGPS",
            5: "RTK_FLOAT",
            6: "RTK_FIXED",
        }.get(fix_type, str(fix_type))
    elif msg_type == "ATTITUDE":
        state["roll_deg"] = math.degrees(msg.roll)
        state["pitch_deg"] = math.degrees(msg.pitch)
        state["yaw_deg"] = math.degrees(msg.yaw)
    elif msg_type == "GLOBAL_POSITION_INT":
        state["lat"] = msg.lat / 1e7
        state["lon"] = msg.lon / 1e7
        state["alt_msl_m"] = msg.alt / 1000.0
        state["alt_rel_m"] = getattr(msg, "relative_alt", 0) / 1000.0
        state["vx_mps"] = msg.vx / 100.0
        state["vy_mps"] = msg.vy / 100.0
        state["vz_mps"] = msg.vz / 100.0
    elif msg_type == "VFR_HUD":
        state["groundspeed_mps"] = float(msg.groundspeed)
        state["airspeed_mps"] = float(msg.airspeed) if msg.airspeed > 0 else None
        state["alt_rel_m"] = float(msg.alt)


def print_summary(state: dict[str, object]) -> None:
    location = "n/a"
    lat = state.get("lat")
    lon = state.get("lon")
    if lat is not None and lon is not None:
        location = f"{lat:.7f},{lon:.7f}"

    print(
        " | ".join(
            [
                f"Mode={state.get('mode', 'UNKNOWN')}",
                "ARMED" if state.get("armed", False) else "DISARMED",
                f"Bat={fmt_number(state.get('battery_voltage'), 'V')} "
                f"{fmt_int(state.get('battery_remaining'))}%",
                f"Cur={fmt_number(state.get('battery_current'), 'A')}",
                f"GPS={state.get('gps_fix', 'n/a')} "
                f"sats={fmt_int(state.get('gps_satellites'))} "
                f"hdop={fmt_number(state.get('gps_hdop'), '', 2)}",
                f"AltRel={fmt_number(state.get('alt_rel_m'), 'm')}",
                f"GndSpd={fmt_number(state.get('groundspeed_mps'), 'm/s')}",
                f"Roll={fmt_number(state.get('roll_deg'), 'deg')}",
                f"Pitch={fmt_number(state.get('pitch_deg'), 'deg')}",
                f"Yaw={fmt_number(state.get('yaw_deg'), 'deg')}",
                f"Pos={location}",
            ]
        ),
        flush=True,
    )


def main() -> int:
    args = parse_args()

    try:
        master = connect(args.port, args.baud, args.heartbeat_timeout)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr, flush=True)
        return 1

    state: dict[str, object] = {
        "armed": False,
        "mode": "UNKNOWN",
    }
    last_mode = None
    last_armed = None
    last_summary = 0.0
    start = time.time()

    print("Listening for telemetry. Press Ctrl+C to stop.", flush=True)
    try:
        while True:
            if args.duration > 0 and (time.time() - start) >= args.duration:
                print("Reached requested duration. Exiting.", flush=True)
                return 0

            msg = master.recv_match(blocking=True, timeout=1.0)
            now = time.time()
            if msg is None:
                if now - last_summary >= args.summary_period:
                    print_summary(state)
                    last_summary = now
                continue

            if msg.get_type() == "BAD_DATA":
                continue

            update_state(state, msg)

            if msg.get_type() == "HEARTBEAT":
                armed = state.get("armed")
                mode = state.get("mode")
                if armed != last_armed or mode != last_mode:
                    print(
                        f"Heartbeat: mode={mode} "
                        f"state={'ARMED' if armed else 'DISARMED'}",
                        flush=True,
                    )
                    last_armed = armed
                    last_mode = mode

            if now - last_summary >= args.summary_period:
                print_summary(state)
                last_summary = now
    except KeyboardInterrupt:
        print("Stopped by user.", flush=True)
        return 0
    finally:
        try:
            master.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
