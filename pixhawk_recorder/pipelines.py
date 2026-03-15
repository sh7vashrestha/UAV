from __future__ import annotations

from pathlib import Path

from .config import Config


def build_rgb_capture_pipeline(cfg: Config) -> str:
    return (
        f"nvarguscamerasrc sensor-id={cfg.rgb_sensor_id} ! "
        f"video/x-raw(memory:NVMM),width={cfg.rgb_width},height={cfg.rgb_height},"
        f"format=NV12,framerate={cfg.rgb_capture_fps}/1 ! "
        "nvvidconv flip-method=0 ! "
        f"video/x-raw,width={cfg.rgb_width},height={cfg.rgb_height},format=BGRx ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false emit-signals=false"
    )


def build_thermal_capture_pipeline(cfg: Config) -> str:
    return (
        f"v4l2src device={cfg.thermal_device} io-mode=2 ! "
        f"video/x-raw,format=BGR,width={cfg.thermal_width},height={cfg.thermal_height},"
        f"framerate={cfg.thermal_capture_fps}/1 ! "
        "appsink drop=true max-buffers=1 sync=false emit-signals=false"
    )


def build_mp4_writer_pipeline(
    out_file: Path,
    width: int,
    height: int,
    fps: int,
    bitrate: int,
    preset: str,
    threads: int,
    keyint: int,
) -> str:
    return (
        "appsrc is-live=true block=false format=time do-timestamp=true ! "
        f"video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1 ! "
        "videoconvert ! video/x-raw,format=I420 ! "
        f"x264enc tune=zerolatency speed-preset={preset} bitrate={bitrate} "
        f"key-int-max={keyint} bframes=0 threads={threads} ! "
        "h264parse ! mp4mux ! "
        f"filesink location={str(out_file)} sync=false async=false"
    )


def build_stream_writer_pipeline(cfg: Config) -> str:
    return (
        "appsrc is-live=true block=false format=time do-timestamp=true ! "
        "queue leaky=downstream max-size-buffers=4 max-size-bytes=0 max-size-time=0 ! "
        f"video/x-raw,format=BGR,width={cfg.stream_width},height={cfg.stream_height},"
        f"framerate={cfg.stream_fps}/1 ! "
        "videoconvert ! video/x-raw,format=I420 ! "
        f"x264enc tune=zerolatency speed-preset=ultrafast bitrate={cfg.stream_bitrate} "
        f"key-int-max={cfg.stream_fps} bframes=0 threads=1 ! "
        "h264parse ! queue leaky=downstream max-size-buffers=8 max-size-bytes=0 max-size-time=0 ! "
        "rtph264pay config-interval=1 pt=96 ! "
        f"udpsink host={cfg.stream_host} port={cfg.stream_port} sync=false async=false"
    )
