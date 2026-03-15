# Pixhawk Drone Recorder (Modular)

A modular Jetson + Pixhawk drone recorder for:
- Pixhawk MAVLink arm/disarm driven recording
- RGB + optional thermal capture
- YOLOv5 inference on Jetson
- segmented MP4 recording for crash tolerance
- detection CSV + crops
- per-flight telemetry JSONL + checkpointing
- optional live UDP video stream

## Project layout

```text
pixhawk_modular_project/
├── run.py
├── README.md
├── laptop_postprocess/
│   ├── README.md
│   └── merge_flight_segments.py
├── systemd/
│   └── pixhawk-drone-recorder.service
└── pixhawk_recorder/
    ├── __init__.py
    ├── checkpoint.py
    ├── compat.py
    ├── config.py
    ├── geolocation.py
    ├── postflight_merge.py
    ├── pipelines.py
    ├── pixhawk.py
    ├── recorder.py
    ├── storage.py
    ├── telemetry.py
    ├── time_sync.py
    ├── utils.py
    ├── video_io.py
    └── yolo_engine.py
```

## Quick start

Edit `pixhawk_recorder/config.py` first:
- verify `use_pixhawk = True`
- verify `mavlink_port` matches your Pixhawk `/dev/serial/by-id/...` path
- verify camera dimensions / thermal device
- verify SD card mount path
- verify YOLO weights path
- adjust 6S battery thresholds for your pack
- optionally enable `auto_merge_after_flight = True` for automatic post-flight MP4 merging
- leave `prefer_pixhawk_time = True` if you want flight names and metadata to use Pixhawk GPS/MAVLink time when available
- adjust the geolocation defaults if your camera mount angle or lens setup changes

Then run:

```bash
python3 run.py
```

## Behavior

With `use_pixhawk=True`:
1. process starts and waits for MAVLink heartbeat
2. waits until drone is armed
3. when armed, creates a new flight folder and begins recording
4. when disarmed, finalizes files and stops that flight
5. keeps running and waits for the next arm event

Each arm cycle creates a new flight ID:

```text
/media/sdcard/uav/flights/flight_YYYY-MM-DD_HH-MM-SS/
```

## Notes

### Segments
Video is stored in segments so that a crash or power loss usually only risks the current segment instead of the entire flight.

Each arm/disarm cycle creates one flight folder. If the vehicle is armed again later, the recorder starts a new flight folder automatically.

### Old/custom YOLO checkpoints
If your checkpoint was saved with a custom pickle symbol like `forbid_accumulation_hook`, compatibility is handled in `pixhawk_recorder/compat.py`.

### Merging segments later
Use the laptop-side merge tool after copying a flight folder off the Jetson:

```bash
python3 laptop_postprocess/merge_flight_segments.py /path/to/flight_YYYY-MM-DD_HH-MM-SS
```

This merges `raw`, `annotated`, and `thermal` segment sets into single MP4 files without re-encoding.

### Pixhawk-derived time
When `prefer_pixhawk_time = True`, the recorder uses UTC learned from Pixhawk MAVLink messages for:
- flight folder names
- segment file names
- manifest `closed_at` values
- checkpoint timestamps

This is handled by `pixhawk_recorder/time_sync.py` and does not modify the Linux system clock. If
Pixhawk time is not available yet, the recorder falls back to the Jetson system clock.

### Crop Geolocation
When crop saving is enabled, the recorder also writes:
- `rgb/crop_geolocation.csv`
- one JSON sidecar per crop image in `crops/`

Each row/sidecar stores the crop path, detection box, drone GPS, altitude, roll/pitch/yaw, and an
estimated ground GPS point for the detection center. The first-pass estimate uses:
- IMX477 defaults
- a 6 mm lens
- a camera mount pitch of `-45` degrees
- a flat-ground assumption

This is implemented in `pixhawk_recorder/geolocation.py`. It is a practical first estimate, not
survey-grade mapping, so calibration and terrain handling would be the next accuracy upgrades.

### Optional Jetson auto-merge
To merge segments automatically on the Jetson after each flight stops, enable this in
`pixhawk_recorder/config.py`:

```python
auto_merge_after_flight = True
```

Useful related settings:
- `auto_merge_min_segments = 2` skips single-segment flights
- `auto_merge_streams = ("raw", "annotated", "thermal")` limits which outputs are generated
- `auto_merge_ffmpeg = "ffmpeg"` lets you point at a specific binary

The merge starts after `stop_recording()` finalizes the flight folder. It runs in a background
thread during normal operation, and shutdown waits for any active post-flight merge to finish.

## systemd
A sample systemd unit is provided in `systemd/pixhawk-drone-recorder.service`.
Update paths before enabling it.
