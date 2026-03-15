from __future__ import annotations

import os
import shutil
import subprocess
import time

from .config import Config


class StorageManager:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.last_check = 0.0
        self.last_free_gb = 0.0

    def is_mounted(self) -> bool:
        try:
            result = subprocess.run(
                ["mountpoint", "-q", str(self.cfg.mount_point)],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False

    def ensure_mountpoint_exists(self) -> None:
        self.cfg.mount_point.mkdir(parents=True, exist_ok=True)

    def try_auto_mount(self) -> None:
        self.ensure_mountpoint_exists()
        if self.is_mounted():
            return

        if not self.cfg.auto_mount_exfat:
            raise RuntimeError("SD card not mounted")

        if os.geteuid() != 0:
            raise RuntimeError("Requires root for auto-mount")

        for _ in range(2):
            cmd = ["/usr/sbin/mount.exfat-fuse", self.cfg.mount_device, str(self.cfg.mount_point)]
            result = subprocess.run(cmd, text=True, capture_output=True, timeout=5)
            if result.returncode == 0 and self.is_mounted():
                return
            time.sleep(0.5)

        raise RuntimeError("Failed to mount SD card")

    def ensure_writable(self) -> None:
        self.try_auto_mount()
        if not self.cfg.mount_point.exists():
            raise RuntimeError(f"Mount point {self.cfg.mount_point} does not exist")
        if not os.access(self.cfg.mount_point, os.W_OK):
            raise RuntimeError(f"Mount point {self.cfg.mount_point} is not writable")

        test_dir = self.cfg.mount_point / ".test"
        test_file = test_dir / "ok.txt"
        test_dir.mkdir(parents=True, exist_ok=True)
        with open(test_file, "w") as f:
            f.write("ok\n")
            f.flush()
            os.fsync(f.fileno())
        if test_file.exists():
            test_file.unlink()
        if test_dir.exists() and not any(test_dir.iterdir()):
            test_dir.rmdir()

    def free_gb(self) -> float:
        try:
            usage = shutil.disk_usage(self.cfg.mount_point)
            return usage.free / (1024 ** 3)
        except Exception:
            return 0.0

    def assert_space(self) -> float:
        now = time.time()
        if now - self.last_check < self.cfg.storage_check_interval_sec:
            if self.last_free_gb < self.cfg.min_free_gb:
                raise RuntimeError(
                    f"Low storage: {self.last_free_gb:.2f}GB available "
                    f"(need {self.cfg.min_free_gb:.2f}GB)"
                )
            return self.last_free_gb

        free_gb = self.free_gb()
        self.last_free_gb = free_gb
        self.last_check = now
        if free_gb < self.cfg.min_free_gb:
            raise RuntimeError(
                f"Low storage: {free_gb:.2f}GB available "
                f"(need {self.cfg.min_free_gb:.2f}GB)"
            )
        return free_gb
