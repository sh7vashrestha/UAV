from __future__ import annotations

import logging
import time
import unittest

from pymavlink import mavutil

from pixhawk_recorder.config import Config
from pixhawk_recorder.pixhawk import PixhawkManager
from pixhawk_recorder.time_sync import PixhawkTimeSync


class FakeMaster:
    target_system = 1
    target_component = 1

    def close(self) -> None:
        return


class FakeHeartbeatStreamMaster:
    def __init__(self, messages, *, target_system: int = 0, target_component: int = 0) -> None:
        self.messages = list(messages)
        self.target_system = target_system
        self.target_component = target_component

    def recv_match(self, type=None, blocking=False, timeout=None):
        _ = type, blocking, timeout
        if self.messages:
            return self.messages.pop(0)
        return None

    def close(self) -> None:
        return


class FakeHeartbeat:
    def __init__(
        self,
        armed: bool,
        *,
        src_system: int = 1,
        src_component: int = 1,
        mav_type: int | None = None,
        autopilot: int | None = None,
        custom_mode: int = 4,
    ) -> None:
        self.base_mode = mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED if armed else 0
        self.custom_mode = custom_mode
        self.type = mav_type if mav_type is not None else mavutil.mavlink.MAV_TYPE_QUADROTOR
        self.autopilot = (
            autopilot if autopilot is not None else mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA
        )
        self._src_system = src_system
        self._src_component = src_component

    def get_type(self) -> str:
        return "HEARTBEAT"

    def get_srcSystem(self) -> int:
        return self._src_system

    def get_srcComponent(self) -> int:
        return self._src_component


class FakeSystemTime:
    def __init__(self, time_unix_usec: int) -> None:
        self.time_unix_usec = time_unix_usec

    def get_type(self) -> str:
        return "SYSTEM_TIME"


def build_manager() -> PixhawkManager:
    cfg = Config(
        use_pixhawk=False,
        mavlink_arm_debounce_sec=1.0,
        mavlink_disarm_debounce_sec=2.0,
        mavlink_armed_disconnect_hold_sec=5.0,
    )
    manager = PixhawkManager(cfg, logging.getLogger("test.pixhawk"))
    manager.master = FakeMaster()
    manager.vehicle_system_id = 1
    manager.vehicle_component_id = 1
    manager._set_connection_status("CONNECTED")
    return manager


class PixhawkManagerTests(unittest.TestCase):
    def test_initial_discovery_skips_non_vehicle_heartbeats(self) -> None:
        manager = build_manager()
        manager.master = FakeHeartbeatStreamMaster(
            [
                FakeHeartbeat(
                    False,
                    src_system=42,
                    src_component=mavutil.mavlink.MAV_COMP_ID_MISSIONPLANNER,
                    mav_type=mavutil.mavlink.MAV_TYPE_GCS,
                    autopilot=mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                ),
                FakeHeartbeat(
                    True,
                    src_system=1,
                    src_component=mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1,
                ),
            ]
        )

        heartbeat = manager._wait_for_initial_vehicle_heartbeat()

        self.assertEqual(heartbeat.get_srcSystem(), 1)
        self.assertEqual(heartbeat.get_srcComponent(), mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1)

    def test_ignores_non_vehicle_heartbeats(self) -> None:
        manager = build_manager()

        accepted = manager._process_message(FakeHeartbeat(True, src_system=42), now=1.0)

        self.assertFalse(accepted)
        self.assertFalse(manager.is_armed())

    def test_requires_debounced_arm_and_disarm(self) -> None:
        manager = build_manager()

        self.assertTrue(manager._process_message(FakeHeartbeat(True), now=0.0))
        self.assertFalse(manager.is_armed())

        self.assertTrue(manager._process_message(FakeHeartbeat(True), now=0.5))
        self.assertFalse(manager.is_armed())

        self.assertTrue(manager._process_message(FakeHeartbeat(True), now=1.1))
        self.assertTrue(manager.is_armed())

        self.assertTrue(manager._process_message(FakeHeartbeat(False), now=2.0))
        self.assertTrue(manager.is_armed())

        self.assertTrue(manager._process_message(FakeHeartbeat(False), now=3.5))
        self.assertTrue(manager.is_armed())

        self.assertTrue(manager._process_message(FakeHeartbeat(False), now=4.1))
        self.assertFalse(manager.is_armed())

    def test_holds_armed_state_briefly_across_disconnect(self) -> None:
        manager = build_manager()

        manager._process_message(FakeHeartbeat(True), now=0.0)
        manager._process_message(FakeHeartbeat(True), now=1.1)
        self.assertTrue(manager.is_armed())

        manager.last_heartbeat = time.monotonic()
        manager._handle_disconnect("HEARTBEAT_TIMEOUT")
        self.assertTrue(manager.is_armed())

        manager.last_heartbeat = time.monotonic() - 6.0
        self.assertFalse(manager.is_armed())


class ConfigAndTimeSyncTests(unittest.TestCase):
    def test_config_allows_overriding_thermal_device(self) -> None:
        cfg = Config(use_pixhawk=False, thermal_device="/tmp/fake-thermal")

        self.assertEqual(cfg.thermal_device, "/tmp/fake-thermal")

    def test_pixhawk_time_formats_in_utc(self) -> None:
        sync = PixhawkTimeSync()

        source = sync.update_from_message(FakeSystemTime(946684800_000_000))

        self.assertEqual(source, "SYSTEM_TIME")
        self.assertEqual(
            sync.strftime("%Y-%m-%d %H:%M:%S", fallback_to_system=False),
            "2000-01-01 00:00:00",
        )


if __name__ == "__main__":
    unittest.main()
