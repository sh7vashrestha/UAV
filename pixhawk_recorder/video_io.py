from __future__ import annotations

import csv
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, List, Optional

import cv2
import numpy as np


class Capturer:
    def __init__(self, name: str, pipeline: str):
        self.name = name
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open {name}")

        self.latest = None
        self.lock = threading.Lock()
        self.running = True
        self.fps = 0.0
        self.frame_count = 0
        self.error_count = 0
        self._count = 0
        self._fps_window_start = time.monotonic()
        self.thread = None

        for _ in range(20):
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.latest = frame.copy()
                break
            time.sleep(0.05)
        else:
            self.cap.release()
            raise RuntimeError(f"{name} failed to read initial frame")

        self.thread = threading.Thread(target=self._loop, daemon=True, name=f"Capturer-{name}")
        self.thread.start()

    def _loop(self) -> None:
        while self.running:
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self.lock:
                        self.latest = frame
                    self._count += 1
                    self.frame_count += 1
                    dt = time.monotonic() - self._fps_window_start
                    if dt >= 1.0:
                        self.fps = self._count / dt
                        self._count = 0
                        self._fps_window_start = time.monotonic()
                else:
                    self.error_count += 1
                    time.sleep(0.01)
            except Exception:
                self.error_count += 1
                time.sleep(0.01)

    def get(self) -> Optional[np.ndarray]:
        try:
            with self.lock:
                return self.latest.copy() if self.latest is not None else None
        except Exception:
            return None

    def stop(self) -> None:
        self.running = False
        if self.thread:
            try:
                self.thread.join(timeout=1.0)
            except Exception:
                pass
        try:
            self.cap.release()
        except Exception:
            pass


class Writer:
    def __init__(self, name: str, pipeline: str, fps: int, width: int, height: int, logger):
        self.name = name
        self.logger = logger
        self.fps = fps
        self.width = width
        self.height = height
        self.period = 1.0 / max(fps, 1)

        self.writer = cv2.VideoWriter(pipeline, cv2.CAP_GSTREAMER, 0, fps, (width, height), True)
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open writer: {name}")

        self.queue = queue.Queue(maxsize=5)
        self.running = True
        self.written = 0
        self.dropped = 0
        self.error_count = 0
        self.thread = threading.Thread(target=self._loop, daemon=True, name=f"Writer-{name}")
        self.thread.start()

    def push(self, frame: Optional[np.ndarray]) -> None:
        if frame is None:
            return
        try:
            self.queue.put_nowait(frame.copy())
        except queue.Full:
            self.dropped += 1

    def _loop(self) -> None:
        last_frame = None
        next_t = time.perf_counter()

        while self.running or not self.queue.empty():
            try:
                now = time.perf_counter()
                remaining = next_t - now
                if remaining > 0:
                    time.sleep(min(remaining, 0.01))
                    continue

                try:
                    frame = self.queue.get_nowait()
                    if frame.shape[:2] != (self.height, self.width):
                        frame = cv2.resize(frame, (self.width, self.height))
                    last_frame = frame
                except queue.Empty:
                    pass

                if last_frame is not None:
                    try:
                        self.writer.write(last_frame)
                        self.written += 1
                    except Exception as e:
                        self.error_count += 1
                        self.logger.error("Writer %s write error: %s", self.name, e)

                next_t += self.period
            except Exception as e:
                self.error_count += 1
                self.logger.error("Writer %s loop error: %s", self.name, e)
                time.sleep(0.01)

    def close(self) -> None:
        self.running = False
        if self.thread:
            try:
                self.thread.join(timeout=2.0)
            except Exception:
                pass
        try:
            self.writer.release()
        except Exception:
            pass


class CSVWriter:
    def __init__(
        self,
        csv_path: Path,
        flush_interval_sec: float,
        logger,
        header: Optional[List[str]] = None,
    ):
        self.csv_path = csv_path
        self.flush_interval_sec = flush_interval_sec
        self.logger = logger
        self.header = header or [
            "infer_id",
            "t_sec",
            "class_id",
            "class_name",
            "conf",
            "x1",
            "y1",
            "x2",
            "y2",
        ]
        self.queue = queue.Queue(maxsize=500)
        self.running = True
        self.written = 0
        self.error_count = 0
        self.last_flush = time.monotonic()
        self.file = None
        self.writer = None

        self._open_file()
        self.thread = threading.Thread(target=self._loop, daemon=True, name="CSVWriter")
        self.thread.start()

    def _open_file(self) -> None:
        self.file = open(self.csv_path, "w", buffering=1, newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(self.header)
        self.file.flush()
        os.fsync(self.file.fileno())

    def write(self, row: List[Any]) -> None:
        try:
            self.queue.put_nowait(row)
        except queue.Full:
            self.logger.warning("CSV queue full")

    def _flush_rows(self, rows: List[List[Any]]) -> None:
        if not rows or self.writer is None:
            return
        for row in rows:
            self.writer.writerow(row)
            self.written += 1
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def _loop(self) -> None:
        rows = []
        while self.running or not self.queue.empty():
            try:
                try:
                    row = self.queue.get(timeout=0.2)
                    rows.append(row)
                except queue.Empty:
                    pass

                now = time.monotonic()
                if rows and (now - self.last_flush >= self.flush_interval_sec or len(rows) >= 50):
                    self._flush_rows(rows)
                    rows = []
                    self.last_flush = now
            except Exception as e:
                self.error_count += 1
                self.logger.error("CSV error: %s", e)
                time.sleep(0.1)

        if rows:
            try:
                self._flush_rows(rows)
            except Exception as e:
                self.logger.error("Final CSV flush error: %s", e)

    def close(self) -> None:
        self.running = False
        if self.thread:
            try:
                self.thread.join(timeout=2.0)
            except Exception:
                pass
        try:
            if self.file:
                self.file.flush()
                os.fsync(self.file.fileno())
                self.file.close()
        except Exception as e:
            self.logger.error("CSV close error: %s", e)
