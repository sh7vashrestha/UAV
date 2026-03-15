#!/usr/bin/env python3
from __future__ import annotations

import atexit
import logging
import signal
import sys

import cv2

from pixhawk_recorder.compat import ensure_checkpoint_compat
from pixhawk_recorder.config import Config
from pixhawk_recorder.recorder import PixhawkDroneRecorder


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sdrec")


def main() -> None:
    ensure_checkpoint_compat()
    cfg = Config()

    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    logger.info("=" * 60)
    logger.info("Pixhawk 6X Drone Recorder Starting")
    logger.info("MAVLink port: %s @ %d baud", cfg.mavlink_port, cfg.mavlink_baudrate)
    logger.info("=" * 60)

    rec = PixhawkDroneRecorder(cfg)
    atexit.register(rec.shutdown)

    def handle_signal(sig, _frame):
        logger.info("Signal %d received", sig)
        rec.keep_running = False
        rec.recording = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        rec.main_loop()
    except Exception as e:
        logger.exception("Fatal error: %s", e)
        try:
            rec.shutdown()
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
