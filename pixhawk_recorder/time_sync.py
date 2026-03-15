from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Optional


VALID_UNIX_EPOCH_MIN = 946684800.0  # 2000-01-01 00:00:00 UTC
SOURCE_PRIORITY = {
    "GPS_RAW_INT": 1,
    "SYSTEM_TIME": 2,
}


class PixhawkTimeSync:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._unix_epoch_at_sync: Optional[float] = None
        self._monotonic_at_sync: float = 0.0
        self._source: Optional[str] = None

    def update_from_message(self, msg) -> Optional[str]:
        source, epoch = self._extract_epoch(msg)
        if source is None or epoch is None:
            return None

        with self._lock:
            previous_source = self._source
            if previous_source is not None:
                previous_priority = SOURCE_PRIORITY.get(previous_source, 0)
                new_priority = SOURCE_PRIORITY.get(source, 0)
                if new_priority < previous_priority:
                    return None

            self._unix_epoch_at_sync = epoch
            self._monotonic_at_sync = time.monotonic()
            self._source = source

            if previous_source != source:
                return source
            return None

    def has_time(self) -> bool:
        with self._lock:
            return self._unix_epoch_at_sync is not None

    def source(self) -> Optional[str]:
        with self._lock:
            return self._source

    def now_epoch(self, fallback_to_system: bool = True) -> float:
        with self._lock:
            current = self._current_epoch_locked()

        if current is not None:
            return current
        if fallback_to_system:
            return time.time()
        raise RuntimeError("Pixhawk time is not available yet")

    def strftime(self, fmt: str, fallback_to_system: bool = True) -> str:
        return datetime.fromtimestamp(self.now_epoch(fallback_to_system)).strftime(fmt)

    def _current_epoch_locked(self) -> Optional[float]:
        if self._unix_epoch_at_sync is None:
            return None
        return self._unix_epoch_at_sync + (time.monotonic() - self._monotonic_at_sync)

    def _extract_epoch(self, msg) -> tuple[Optional[str], Optional[float]]:
        msg_type = msg.get_type()
        if msg_type == "SYSTEM_TIME":
            return "SYSTEM_TIME", self._normalize_epoch_us(getattr(msg, "time_unix_usec", 0))
        if msg_type == "GPS_RAW_INT":
            return "GPS_RAW_INT", self._normalize_epoch_us(getattr(msg, "time_usec", 0))
        return None, None

    @staticmethod
    def _normalize_epoch_us(raw_value: object) -> Optional[float]:
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return None

        if value <= 0:
            return None

        epoch = value / 1_000_000.0
        if epoch < VALID_UNIX_EPOCH_MIN:
            return None
        return epoch
