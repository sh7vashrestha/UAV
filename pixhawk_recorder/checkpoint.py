from __future__ import annotations

import json
import os
import threading
import time
import zlib
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .telemetry import PixhawkTelemetry


class CheckpointManager:
    def __init__(self, run_dir: Path, now_fn: Callable[[], float] | None = None):
        self.run_dir = run_dir
        self.checkpoint_file = run_dir / "checkpoint.json"
        self.manifest_file = run_dir / "segments_manifest.jsonl"
        self.flight_log_file = run_dir / "flight_telemetry.jsonl"
        self.lock = threading.Lock()
        self.now_fn = now_fn or time.time

    def save_checkpoint(self, data: Dict[str, Any], retries: int = 2) -> bool:
        for _ in range(retries):
            try:
                with self.lock:
                    payload = dict(data)
                    payload["timestamp"] = self.now_fn()
                    json_str = json.dumps(payload, sort_keys=True, default=str)
                    payload["crc32"] = zlib.crc32(json_str.encode()) & 0xFFFFFFFF

                    tmp = self.checkpoint_file.with_suffix(".tmp")
                    with open(tmp, "w") as f:
                        json.dump(payload, f, indent=2)
                        f.flush()
                        os.fsync(f.fileno())
                    tmp.replace(self.checkpoint_file)
                    return True
            except Exception:
                time.sleep(0.2)
        return False

    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        try:
            if not self.checkpoint_file.exists():
                return None
            with self.lock:
                with open(self.checkpoint_file, "r") as f:
                    data = json.load(f)

            if "crc32" in data:
                saved_crc = data.pop("crc32")
                json_str = json.dumps(data, sort_keys=True, default=str)
                calc_crc = zlib.crc32(json_str.encode()) & 0xFFFFFFFF
                if saved_crc != calc_crc:
                    return None
            data.pop("timestamp", None)
            return data
        except Exception:
            return None

    def append_manifest(self, row: Dict[str, Any]) -> None:
        with self.lock:
            with open(self.manifest_file, "a") as f:
                f.write(json.dumps(row) + "\n")
                f.flush()
                os.fsync(f.fileno())

    def log_flight_telemetry(self, telemetry: PixhawkTelemetry) -> None:
        with self.lock:
            with open(self.flight_log_file, "a") as f:
                f.write(json.dumps(telemetry.to_dict()) + "\n")
