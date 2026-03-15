from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import torch

from .checkpoint import CheckpointManager
from .config import Config
from .geolocation import GEOLOCATION_CSV_HEADER, build_crop_geolocation_record, csv_row_from_record
from .postflight_merge import merge_flight_segments
from .pipelines import (
    build_mp4_writer_pipeline,
    build_rgb_capture_pipeline,
    build_stream_writer_pipeline,
    build_thermal_capture_pipeline,
)
from .pixhawk import PixhawkManager
from .storage import StorageManager
from .telemetry import ErrorTelemetry
from .utils import draw_dets, safe_stem, thermal_to_bgr
from .video_io import CSVWriter, Capturer, Writer
from .yolo_engine import OrinNanoYOLOEngine


logger = logging.getLogger("sdrec")


class PixhawkDroneRecorder:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.storage = StorageManager(cfg)
        self.pixhawk = PixhawkManager(cfg, logger)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.telemetry_obj = ErrorTelemetry()

        self.run_dir: Optional[Path] = None
        self.rgb_dir: Optional[Path] = None
        self.thermal_dir: Optional[Path] = None
        self.crop_dir: Optional[Path] = None

        self.rgb_cap = None
        self.thermal_cap = None
        self.raw_writer = None
        self.annot_writer = None
        self.thermal_writer = None
        self.stream_writer = None
        self.csv_writer = None
        self.geo_csv_writer = None
        self.yolo = None
        self.ckpt = None
        self.file_handler = None

        self.recording = False
        self.keep_running = True
        self.shutdown_done = False
        self.initialized = False

        self.t0_monotonic = None
        self.last_checkpoint_monotonic = 0.0
        self.last_rotate_monotonic = 0.0
        self.segment_index = 0
        self.last_dets: List[Dict[str, object]] = []
        self.infer_count = 0
        self.last_logged_infer_id = -1
        self.current_segment_files: Dict[str, Path] = {}
        self.last_telemetry_log_monotonic = 0.0
        self.postflight_threads: List[threading.Thread] = []
        self.flight_clock_use_pixhawk = False
        self.flight_clock_source = "SYSTEM"

    def _attach_run_logger(self) -> None:
        self._detach_run_logger()
        if self.run_dir is None:
            return
        self.file_handler = logging.FileHandler(self.run_dir / "recorder.log")
        self.file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(self.file_handler)

    def _detach_run_logger(self) -> None:
        if self.file_handler is not None:
            try:
                logger.removeHandler(self.file_handler)
                self.file_handler.close()
            except Exception:
                pass
            self.file_handler = None

    def _lock_flight_clock(self) -> None:
        self.flight_clock_use_pixhawk = (
            self.cfg.use_pixhawk and self.cfg.prefer_pixhawk_time and self.pixhawk.has_time()
        )
        self.flight_clock_source = (
            (self.pixhawk.time_source() or "PIXHAWK") if self.flight_clock_use_pixhawk else "SYSTEM"
        )

    def _wall_time_epoch(self) -> float:
        if self.flight_clock_use_pixhawk:
            return self.pixhawk.now_epoch()
        return time.time()

    def _wall_time_strftime(self, fmt: str) -> str:
        if self.flight_clock_use_pixhawk:
            return self.pixhawk.strftime(fmt)
        return time.strftime(fmt)

    def _wall_time_source(self) -> str:
        return self.flight_clock_source

    def _elapsed_flight_sec(self, now_monotonic: Optional[float] = None) -> float:
        if self.t0_monotonic is None:
            return 0.0
        if now_monotonic is None:
            now_monotonic = time.monotonic()
        return max(now_monotonic - self.t0_monotonic, 0.0)

    def _path_relative_to_run(self, path: Path) -> Path:
        if self.run_dir is None:
            return path
        try:
            return path.relative_to(self.run_dir)
        except ValueError:
            return path

    def _write_crop_geolocation(
        self,
        crop_path: Path,
        det: Dict[str, object],
        pix_tel,
        bbox: tuple[int, int, int, int],
        detection_index: int,
        now: float,
    ) -> None:
        if not self.cfg.save_crop_geolocation_csv and not self.cfg.save_crop_geolocation_sidecars:
            return

        if self.t0_monotonic is None:
            return

        record = build_crop_geolocation_record(
            self.cfg,
            pix_tel,
            flight_name=self.cfg.run_name,
            infer_id=self.infer_count,
            t_sec=self._elapsed_flight_sec(now),
            detection_index=detection_index,
            crop_path=self._path_relative_to_run(crop_path),
            bbox=bbox,
            class_id=int(det["cls"]),
            class_name=str(det["name"]),
            conf=float(det["conf"]),
            clock_source=self._wall_time_source(),
            capture_wall_time=self._wall_time_strftime("%Y-%m-%d %H:%M:%S"),
        )

        if self.cfg.save_crop_geolocation_csv and self.geo_csv_writer is not None:
            self.geo_csv_writer.write(csv_row_from_record(record))

        if self.cfg.save_crop_geolocation_sidecars:
            sidecar_path = crop_path.with_suffix(".json")
            try:
                with open(sidecar_path, "w", encoding="utf-8") as handle:
                    json.dump(record, handle, indent=2)
            except Exception as e:
                logger.error("Crop geolocation sidecar error: %s", e)

    def _setup_flight_dir(self) -> None:
        self.run_dir = self.cfg.save_root / self.cfg.run_name
        self.rgb_dir = self.run_dir / "rgb"
        self.thermal_dir = self.run_dir / "thermal"
        self.crop_dir = self.run_dir / "crops"

        self.storage.ensure_writable()
        self.storage.assert_space()

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.rgb_dir.mkdir(parents=True, exist_ok=True)
        self.thermal_dir.mkdir(parents=True, exist_ok=True)
        if self.cfg.save_crops:
            self.crop_dir.mkdir(parents=True, exist_ok=True)

        self.ckpt = CheckpointManager(self.run_dir, now_fn=self._wall_time_epoch)
        self._attach_run_logger()

    def _segment_base(self) -> str:
        return f"seg_{self._wall_time_strftime('%Y%m%d_%H%M%S')}_{self.segment_index:04d}"

    def _open_segment_writers(self) -> None:
        self._close_segment_writers()

        base = self._segment_base()
        raw_file = self.rgb_dir / f"{base}_raw.mp4"
        ann_file = self.rgb_dir / f"{base}_annotated.mp4"
        thm_file = self.thermal_dir / f"{base}_thermal.mp4"

        self.current_segment_files = {
            "raw": raw_file,
            "annotated": ann_file,
            "thermal": thm_file,
        }

        self.raw_writer = Writer(
            "raw",
            build_mp4_writer_pipeline(
                raw_file,
                self.cfg.rgb_width,
                self.cfg.rgb_height,
                self.cfg.rgb_record_fps,
                self.cfg.rgb_raw_bitrate,
                self.cfg.x264_preset,
                self.cfg.x264_threads,
                self.cfg.x264_keyint,
            ),
            self.cfg.rgb_record_fps,
            self.cfg.rgb_width,
            self.cfg.rgb_height,
            logger,
        )

        self.annot_writer = Writer(
            "annotated",
            build_mp4_writer_pipeline(
                ann_file,
                self.cfg.rgb_width,
                self.cfg.rgb_height,
                self.cfg.rgb_annot_fps,
                self.cfg.rgb_annot_bitrate,
                self.cfg.x264_preset,
                self.cfg.x264_threads,
                self.cfg.x264_keyint,
            ),
            self.cfg.rgb_annot_fps,
            self.cfg.rgb_width,
            self.cfg.rgb_height,
            logger,
        )

        if self.thermal_cap is not None:
            self.thermal_writer = Writer(
                "thermal",
                build_mp4_writer_pipeline(
                    thm_file,
                    self.cfg.thermal_width,
                    self.cfg.thermal_height,
                    self.cfg.thermal_record_fps,
                    self.cfg.thermal_bitrate,
                    self.cfg.x264_preset,
                    1,
                    self.cfg.thermal_record_fps,
                ),
                self.cfg.thermal_record_fps,
                self.cfg.thermal_width,
                self.cfg.thermal_height,
                logger,
            )

        if self.cfg.use_video_stream:
            self.stream_writer = Writer(
                "stream",
                build_stream_writer_pipeline(self.cfg),
                self.cfg.stream_fps,
                self.cfg.stream_width,
                self.cfg.stream_height,
                logger,
            )

        self.last_rotate_monotonic = time.monotonic()
        logger.info("Opened segment: %s", base)

    def _close_segment_writers(self) -> None:
        for w in [self.raw_writer, self.annot_writer, self.thermal_writer, self.stream_writer]:
            try:
                if w is not None:
                    self.telemetry_obj.writer_errors += int(getattr(w, "error_count", 0))
                    self.telemetry_obj.frame_dropped += int(getattr(w, "dropped", 0))
                    w.close()
            except Exception as e:
                logger.error("Writer close error: %s", e)

        self.raw_writer = None
        self.annot_writer = None
        self.thermal_writer = None
        self.stream_writer = None

    def _close_capturers(self, *, log_errors: bool) -> None:
        for attr_name, label in [
            ("rgb_cap", "RGB capturer stop error"),
            ("thermal_cap", "Thermal capturer stop error"),
        ]:
            cap = getattr(self, attr_name)
            try:
                if cap is not None:
                    cap.stop()
            except Exception as e:
                if log_errors:
                    logger.error("%s: %s", label, e)
            setattr(self, attr_name, None)

    def _close_csv_writers(self, *, log_errors: bool) -> None:
        for attr_name, label in [
            ("csv_writer", "CSV close error"),
            ("geo_csv_writer", "Crop geolocation CSV close error"),
        ]:
            writer = getattr(self, attr_name)
            try:
                if writer is not None:
                    writer.close()
            except Exception as e:
                if log_errors:
                    logger.error("%s: %s", label, e)
            setattr(self, attr_name, None)

    def _finalize_segment(self) -> None:
        if not self.current_segment_files or self.ckpt is None:
            return

        row = {
            "segment_index": self.segment_index,
            "closed_at": self._wall_time_strftime("%Y-%m-%d %H:%M:%S"),
            "closed_at_source": self._wall_time_source(),
            "runtime_sec": round(self._elapsed_flight_sec(), 2),
        }
        for key, path in self.current_segment_files.items():
            row[f"{key}_path"] = str(path)
            row[f"{key}_exists"] = path.exists()
            row[f"{key}_size_bytes"] = path.stat().st_size if path.exists() else 0

        self.ckpt.append_manifest(row)
        self.current_segment_files = {}

    def _rotate_segment_if_needed(self) -> None:
        if time.monotonic() - self.last_rotate_monotonic < self.cfg.segment_duration_sec:
            return

        logger.info("Rotating segment")
        self._close_segment_writers()
        self._finalize_segment()
        self.storage.assert_space()
        self.segment_index += 1
        self._open_segment_writers()

    def _save_checkpoint(self) -> bool:
        if self.ckpt is None or self.t0_monotonic is None:
            return False

        pix_tel = self.pixhawk.get_telemetry()
        return self.ckpt.save_checkpoint(
            {
                "runtime_sec": round(self._elapsed_flight_sec(), 2),
                "segment_index": self.segment_index,
                "inferences": self.infer_count,
                "free_gb": round(self.storage.free_gb(), 2),
                "battery_voltage": round(pix_tel.battery_voltage, 2),
                "battery_remaining": round(pix_tel.battery_remaining, 1),
                "altitude_agl": round(pix_tel.altitude_relative, 1),
            },
            retries=self.cfg.checkpoint_retries,
        )

    def _prune_postflight_threads(self) -> None:
        self.postflight_threads = [thread for thread in self.postflight_threads if thread.is_alive()]

    def _run_postflight_merge(self, flight_dir: Path) -> None:
        def report(message: str) -> None:
            logger.info("Post-flight merge [%s]: %s", flight_dir.name, message)

        try:
            merged = merge_flight_segments(
                flight_dir=flight_dir,
                streams=self.cfg.auto_merge_streams,
                ffmpeg_bin=self.cfg.auto_merge_ffmpeg,
                overwrite=self.cfg.auto_merge_overwrite,
                min_segments=self.cfg.auto_merge_min_segments,
                report=report,
            )
            if merged:
                report(f"Finished {len(merged)} merged stream(s)")
            else:
                report("Nothing needed merging")
        except Exception as e:
            logger.error("Post-flight merge [%s] failed: %s", flight_dir.name, e)

    def _start_postflight_merge(self, flight_dir: Optional[Path]) -> None:
        if not self.cfg.auto_merge_after_flight or flight_dir is None:
            return

        self._prune_postflight_threads()
        thread = threading.Thread(
            target=self._run_postflight_merge,
            args=(flight_dir,),
            name=f"postflight-merge-{flight_dir.name}",
        )
        self.postflight_threads.append(thread)
        thread.start()
        logger.info("Started background post-flight merge for %s", flight_dir)

    def _wait_for_postflight_threads(self) -> None:
        self._prune_postflight_threads()
        current = threading.current_thread()
        for thread in self.postflight_threads:
            if thread is current:
                continue
            logger.info("Waiting for %s to finish", thread.name)
            thread.join()
        self._prune_postflight_threads()

    def _cleanup_partial_init(self) -> None:
        self._close_capturers(log_errors=False)
        self._close_segment_writers()
        self._close_csv_writers(log_errors=False)

        self.initialized = False
        self.recording = False

    def initialize(self) -> None:
        if self.cfg.use_pixhawk:
            logger.info("DRONE ARMED - initializing recorder")
        else:
            logger.info("MANUAL MODE - starting recording immediately")

        self.t0_monotonic = time.monotonic()
        self._lock_flight_clock()
        self.cfg.run_name = self._wall_time_strftime("flight_%Y-%m-%d_%H-%M-%S")
        self.segment_index = 0
        self.infer_count = 0
        self.last_logged_infer_id = -1
        self.last_dets = []
        self.current_segment_files = {}
        self.last_checkpoint_monotonic = 0.0
        self.last_rotate_monotonic = 0.0
        self.last_telemetry_log_monotonic = 0.0

        try:
            self._setup_flight_dir()
            logger.info("Flight clock source: %s", self._wall_time_source())
            self.yolo = OrinNanoYOLOEngine(self.cfg, self.device, logger)
            self.rgb_cap = Capturer("rgb", build_rgb_capture_pipeline(self.cfg))

            try:
                self.thermal_cap = Capturer("thermal", build_thermal_capture_pipeline(self.cfg))
            except Exception as e:
                if self.cfg.require_thermal:
                    raise
                logger.warning("Thermal disabled: %s", e)
                self.thermal_cap = None
                self.telemetry_obj.thermal_errors += 1

            if self.cfg.save_csv:
                self.csv_writer = CSVWriter(
                    self.rgb_dir / "detections.csv",
                    flush_interval_sec=self.cfg.csv_flush_interval_sec,
                    logger=logger,
                )
            if self.cfg.save_crops and self.cfg.save_crop_geolocation_csv:
                self.geo_csv_writer = CSVWriter(
                    self.rgb_dir / "crop_geolocation.csv",
                    flush_interval_sec=self.cfg.csv_flush_interval_sec,
                    logger=logger,
                    header=GEOLOCATION_CSV_HEADER,
                )

            self._open_segment_writers()
            self.recording = True
            self.initialized = True
            logger.info("Recorder initialized - recording started")
            logger.info("Flight directory: %s", self.run_dir)
        except Exception as e:
            logger.exception("Initialization failed: %s", e)
            self._cleanup_partial_init()
            raise

    def _log_detections_once(self, infer_id: int, now: float) -> None:
        if self.csv_writer is None or infer_id == self.last_logged_infer_id or self.t0_monotonic is None:
            return

        self.last_logged_infer_id = infer_id
        for det in self.last_dets:
            x1, y1, x2, y2 = det["bbox"]
            self.csv_writer.write(
                [
                    infer_id,
                    f"{self._elapsed_flight_sec(now):.3f}",
                    det["cls"],
                    det["name"],
                    f"{det['conf']:.3f}",
                    x1,
                    y1,
                    x2,
                    y2,
                ]
            )

    def run_flight(self) -> None:
        next_infer = time.monotonic()
        infer_period = 1.0 / max(self.cfg.inference_fps, 1)
        last_print = time.monotonic()
        use_pixhawk = self.cfg.use_pixhawk

        try:
            while self.keep_running and self.recording and (not use_pixhawk or self.pixhawk.is_armed()):
                now = time.monotonic()

                if use_pixhawk:
                    pix_tel = self.pixhawk.get_telemetry()
                    if (
                        pix_tel.battery_voltage > 0
                        and (
                            pix_tel.battery_voltage < self.cfg.battery_critical_voltage
                            or pix_tel.battery_remaining < self.cfg.battery_critical_percent
                        )
                    ):
                        logger.warning(
                            "BATTERY CRITICAL (%.1fV, %.0f%%) - stopping",
                            pix_tel.battery_voltage,
                            pix_tel.battery_remaining,
                        )
                        self.recording = False
                        self.telemetry_obj.battery_warnings += 1
                        break
                else:
                    pix_tel = None

                self._rotate_segment_if_needed()
                rgb = self.rgb_cap.get() if self.rgb_cap else None
                thm = self.thermal_cap.get() if self.thermal_cap else None

                did_infer = False
                if rgb is not None and now >= next_infer:
                    next_infer = now + infer_period
                    self.last_dets = self.yolo.infer(rgb)
                    self.infer_count += 1
                    self.telemetry_obj.inference_errors = getattr(self.yolo, "error_count", 0)
                    if pix_tel:
                        pix_tel.inference_count = self.infer_count
                        pix_tel.detected_objects = len(self.last_dets)
                    did_infer = True

                if rgb is not None and self.raw_writer is not None:
                    self.raw_writer.push(rgb)
                    annot = draw_dets(rgb.copy(), self.last_dets) if self.last_dets else rgb
                    if self.annot_writer is not None:
                        self.annot_writer.push(annot)
                    if self.stream_writer is not None:
                        stream_frame = cv2.resize(annot, (self.cfg.stream_width, self.cfg.stream_height))
                        self.stream_writer.push(stream_frame)

                if thm is not None and self.thermal_writer is not None:
                    thm_bgr = thermal_to_bgr(thm)
                    if thm_bgr is not None:
                        self.thermal_writer.push(thm_bgr)

                if did_infer:
                    self._log_detections_once(self.infer_count, now)
                    if self.cfg.save_crops and rgb is not None and self.crop_dir is not None:
                        for j, det in enumerate(self.last_dets):
                            try:
                                x1, y1, x2, y2 = det["bbox"]
                                if x2 <= x1 or y2 <= y1:
                                    continue
                                crop = rgb[y1:y2, x1:x2]
                                if crop.size == 0:
                                    continue
                                crop_name = (
                                    self.crop_dir
                                    / f"infer{self.infer_count:07d}_det{j:02d}_{safe_stem(det['name'])}_{det['conf']:.2f}.jpg"
                                )
                                if not cv2.imwrite(str(crop_name), crop):
                                    logger.error("Crop save error: failed to write %s", crop_name)
                                    continue
                                self._write_crop_geolocation(
                                    crop_path=crop_name,
                                    det=det,
                                    pix_tel=pix_tel,
                                    bbox=(x1, y1, x2, y2),
                                    detection_index=j,
                                    now=now,
                                )
                            except Exception as e:
                                logger.error("Crop save error: %s", e)

                if now - self.last_checkpoint_monotonic >= self.cfg.checkpoint_interval_sec:
                    self.last_checkpoint_monotonic = now
                    if not self._save_checkpoint():
                        self.telemetry_obj.checkpoint_errors += 1

                if use_pixhawk and pix_tel is not None and now - self.last_telemetry_log_monotonic >= self.cfg.telemetry_log_interval_sec:
                    self.last_telemetry_log_monotonic = now
                    pix_tel.timestamp = self._elapsed_flight_sec(now)
                    pix_tel.frame_count = self.rgb_cap.frame_count if self.rgb_cap else 0
                    if self.ckpt is not None:
                        self.ckpt.log_flight_telemetry(pix_tel)

                if now - last_print >= 5.0:
                    last_print = now
                    elapsed = max(self._elapsed_flight_sec(now), 0.1)
                    if use_pixhawk and pix_tel is not None:
                        logger.info(
                            "FLIGHT | Alt=%.1fm | Spd=%.1fm/s | Bat=%.1fV (%.0f%%) | RGB=%.0f/s | Infer=%.0f/s det=%d | Free=%.1fGB",
                            pix_tel.altitude_relative,
                            pix_tel.ground_speed,
                            pix_tel.battery_voltage,
                            pix_tel.battery_remaining,
                            self.rgb_cap.fps if self.rgb_cap else 0.0,
                            self.infer_count / elapsed,
                            len(self.last_dets),
                            self.storage.free_gb(),
                        )
                    else:
                        logger.info(
                            "RECORDING | RGB=%.0f/s | Infer=%.0f/s det=%d | Free=%.1fGB",
                            self.rgb_cap.fps if self.rgb_cap else 0.0,
                            self.infer_count / elapsed,
                            len(self.last_dets),
                            self.storage.free_gb(),
                        )

                time.sleep(0.001)
        except Exception as e:
            logger.exception("Flight loop error: %s", e)
        finally:
            self.stop_recording()

    def stop_recording(self) -> None:
        if not self.initialized and not self.recording:
            return

        flight_dir = self.run_dir
        logger.info("Finalizing recording")
        self.recording = False

        self._close_capturers(log_errors=True)
        self._close_segment_writers()
        self._finalize_segment()
        self._close_csv_writers(log_errors=True)

        try:
            if self.ckpt is not None:
                self._save_checkpoint()
        except Exception as e:
            logger.error("Final checkpoint save error: %s", e)

        try:
            if self.cfg.use_pixhawk and self.ckpt and self.ckpt.flight_log_file.exists():
                with open(self.ckpt.flight_log_file, "a") as f:
                    f.flush()
                    os.fsync(f.fileno())
        except Exception as e:
            logger.error("Flight log flush error: %s", e)

        runtime = self._elapsed_flight_sec()
        pix_tel = self.pixhawk.get_telemetry()
        metadata = {
            "flight_name": self.cfg.run_name,
            "flight_dir": str(self.run_dir) if self.run_dir else "",
            "runtime_sec": round(runtime, 2),
            "device": str(self.device),
            "total_inferences": self.infer_count,
            "total_segments": self.segment_index + 1 if self.initialized else 0,
            "pixhawk_connected": self.pixhawk.is_connected() if self.cfg.use_pixhawk else False,
            "battery_voltage_end": round(pix_tel.battery_voltage, 2),
            "battery_percent_end": round(pix_tel.battery_remaining, 1),
            "altitude_end": round(pix_tel.altitude_relative, 1),
            "gps_sats": pix_tel.gps_satellites,
            "storage_mount": str(self.cfg.mount_point),
            "free_gb_at_end": round(self.storage.free_gb(), 2),
            "clock_source": self._wall_time_source(),
            "telemetry": asdict(self.telemetry_obj),
        }

        try:
            if self.run_dir is not None:
                with open(self.run_dir / "flight_metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.error("Metadata save error: %s", e)

        if self.run_dir is not None:
            logger.info("Recording saved to: %s", self.run_dir)
        logger.info("Recording duration: %.1f seconds", runtime)

        self.initialized = False
        self._detach_run_logger()
        self._start_postflight_merge(flight_dir)

    def main_loop(self) -> None:
        if self.cfg.use_pixhawk:
            logger.info("=" * 60)
            logger.info("PIXHAWK MODE - waiting for armed signal")
            logger.info("=" * 60)

            if not self.pixhawk.is_connected():
                logger.warning("Not connected to Pixhawk yet - waiting...")

            try:
                while self.keep_running:
                    if self.pixhawk.is_armed() and not self.recording:
                        logger.info("DRONE ARMED SIGNAL RECEIVED")
                        self.telemetry_obj.power_cycles += 1
                        try:
                            self.initialize()
                            self.run_flight()
                        except Exception as e:
                            logger.exception("Flight recording failed: %s", e)
                            self.recording = False
                            try:
                                self.stop_recording()
                            except Exception:
                                pass
                    elif not self.pixhawk.is_armed() and self.recording:
                        logger.info("DRONE DISARMED SIGNAL RECEIVED")
                        self.stop_recording()
                    time.sleep(0.1)
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt")
            finally:
                self.shutdown()
        else:
            logger.info("=" * 60)
            logger.info("MANUAL MODE - recording starts immediately")
            logger.info("Press Ctrl+C to stop")
            logger.info("=" * 60)
            try:
                self.telemetry_obj.power_cycles += 1
                self.initialize()
                self.run_flight()
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt - stopping recording")
            except Exception as e:
                logger.exception("Recording error: %s", e)
            finally:
                try:
                    if self.recording or self.initialized:
                        self.stop_recording()
                except Exception:
                    pass
                self.shutdown()

    def shutdown(self) -> None:
        if self.shutdown_done:
            return
        self.shutdown_done = True
        logger.info("Shutting down...")
        self.keep_running = False

        if self.recording or self.initialized:
            try:
                self.stop_recording()
            except Exception as e:
                logger.error("Shutdown stop error: %s", e)

        if self.cfg.auto_merge_after_flight:
            self._wait_for_postflight_threads()

        try:
            self.pixhawk.stop()
        except Exception as e:
            logger.error("Pixhawk stop error: %s", e)

        self._detach_run_logger()
        logger.info("Shutdown complete")
