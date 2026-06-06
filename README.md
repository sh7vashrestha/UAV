# DamageScout

DamageScout is a modular Jetson + Pixhawk recording pipeline for UAV-based building
damage assessment. It records synchronized RGB video, optional thermal video, YOLO detections,
crop-level geolocation metadata, and post-flight aggregation outputs suitable for research
analysis and ICDM-style reporting.

## Capabilities

- Pixhawk MAVLink arm/disarm controlled recording
- RGB and optional thermal capture
- YOLOv5 inference on Jetson
- crash-tolerant segmented MP4 recording
- detection CSV and class-organized crop export
- crop-level geolocation metadata
- post-flight building-level damage record aggregation
- publication-ready pre/post aggregation class-count figures
- optional laptop or Jetson-side MP4 segment merging

## Damage Detection Model

The onboard detector uses a fine-tuned YOLOv5 model. The YOLO model fine-tuning framework is
available at:

[Kennesaw-IntelliTrust-Lab/Lightweight-and-Privacy-Enhanced-Damage-Detection-Framework](https://github.com/Kennesaw-IntelliTrust-Lab/Lightweight-and-Privacy-Enhanced-Damage-Detection-Framework)

## Repository Layout

```text
pixhawk_modular_project/
├── run.py
├── pixhawk_test.py
├── laptop_postprocess/
│   ├── crop_size_stats.py
│   ├── merge_flight_segments.py
│   ├── README.md
│   └── standalone_damage_aggregator.py
├── pixhawk_recorder/
│   ├── checkpoint.py
│   ├── compat.py
│   ├── config.py
│   ├── damage_aggregation.py
│   ├── geolocation.py
│   ├── pipelines.py
│   ├── pixhawk.py
│   ├── postflight_merge.py
│   ├── recorder.py
│   ├── storage.py
│   ├── telemetry.py
│   ├── time_sync.py
│   ├── utils.py
│   ├── video_io.py
│   └── yolo_engine.py
├── systemd/
│   └── pixhawk-drone-recorder.service
└── tests/
```

## Pixhawk Recording Workflow

Edit `pixhawk_recorder/config.py` before deployment:

- set `use_pixhawk = True` for arm/disarm controlled recording
- verify `mavlink_port` and `mavlink_baudrate`
- verify RGB/thermal camera settings
- verify SD-card mount and `save_root`
- verify YOLO repository and weights paths
- tune MAVLink debounce and heartbeat timeout settings for the radio/link quality
- tune camera geolocation parameters when the mount angle or lens changes

Start the recorder:

```bash
python3 run.py
```

In Pixhawk mode, the recorder waits for a valid vehicle/autopilot heartbeat, confirms a stable
armed state, creates a new flight folder, records until disarm is stable, finalizes metadata, and
then waits for the next arm event.

Each arm/disarm cycle creates a numbered flight folder:

```text
/media/sdcard/uav/flights/flight_0001/
```

## Flight Folder Outputs

A typical flight folder contains:

```text
flight_0001/
├── checkpoint.json
├── flight_metadata.json
├── flight_telemetry.jsonl
├── recorder.log
├── segments_manifest.jsonl
├── rgb/
│   ├── detections.csv
│   ├── seg_0000_raw.mp4
│   └── seg_0000_annotated.mp4
├── thermal/
│   └── seg_0000_thermal.mp4
├── crops/
│   ├── crop_geolocation.csv
│   ├── no_damage/
│   ├── light_damage/
│   ├── moderate_damage/
│   └── severe_damage/
└── damage_records/
    ├── damage_building_records.json
    ├── damage_building_records.csv
    ├── class_before_after_summary.csv
    ├── before_after_class_counts.pdf
    └── before_after_class_counts.svg
```

The recorder keeps the original flight files and adds `damage_records/` after the flight when
`auto_aggregate_damage_records_after_flight` is enabled.

The core recorded crop output is still:

```text
crops/
    ├── crop_geolocation.csv
    ├── no_damage/
    ├── light_damage/
    ├── moderate_damage/
    └── severe_damage/
```

## Crop Geolocation

When crop saving is enabled, the recorder writes `crop_geolocation.csv` under `crops/`.
Each row stores the crop path, detection box, class label, confidence, drone GPS, altitude,
roll/pitch/yaw, GPS quality, and an estimated ground point for the detection.

The current geolocation estimate uses:

- RGB camera intrinsics from `Config`
- camera mount roll/pitch/yaw
- Pixhawk GPS and attitude
- a flat-ground ray intersection model
- no lens-distortion correction
- no camera-to-GPS lever-arm offset

This produces practical field estimates for small UAV missions, not survey-grade measurements.

## ICDM-Ready Damage Record Aggregation

Building-level aggregation is integrated into the Pixhawk recorder pipeline. After a flight is
finalized, the recorder keeps the original videos, detection CSV, crops, and geolocation CSV, then
adds final building-level damage records under `damage_records/`.

The same aggregation can also be rerun later on a laptop:

```bash
python3 laptop_postprocess/standalone_damage_aggregator.py /path/to/flight_0001
```

The aggregator reads `crops/crop_geolocation.csv`, groups detections that are close in both
location and time, and collapses each group into one final building-level damage record.

Default grouping and consensus behavior:

- spatial threshold: `10 m`
- temporal threshold: `5 s`
- consensus method: `top_k_vote`
- `top_k`: `3`

For each building group, the final damage label is selected from the top 3 highest-confidence
detections. The label with the highest count wins; summed confidence is used as the tie-breaker.
The final building location is the confidence-weighted average of supporting detection locations.

Main outputs:

```text
damage_records/
├── damage_building_records.json
├── damage_building_records.csv
├── class_before_after_summary.csv
├── before_after_class_counts.pdf
└── before_after_class_counts.svg
```

The PDF/SVG figure is formatted as a clean pre-aggregation vs post-aggregation class-count plot
for paper figures. Raw normalized detections can be exported for auditing with:

```bash
python3 laptop_postprocess/standalone_damage_aggregator.py /path/to/flight_0001 --debug-raw-detections
```

Recorder-side aggregation is controlled in `pixhawk_recorder/config.py`:

```python
auto_aggregate_damage_records_after_flight = True
damage_record_spatial_threshold_m = 10.0
damage_record_temporal_threshold_sec = 5.0
damage_record_consensus_method = "top_k_vote"
damage_record_top_k = 3
damage_record_debug_raw_detections = False
```

## Crop File-Size Statistics

To report average crop file size from files on disk:

```bash
python3 laptop_postprocess/crop_size_stats.py /path/to/flight_0001 --output-csv crop_file_size_stats.csv
```

This reports per-class and overall file count, total size, average size, minimum size, and maximum
size using exact filesystem byte counts.

## MP4 Segment Merging

Video is recorded in segments to reduce data loss risk during field operation. Segments can be
merged without re-encoding:

```bash
python3 laptop_postprocess/merge_flight_segments.py /path/to/flight_0001
```

Optional Jetson-side automatic merging is controlled by:

```python
auto_merge_after_flight = True
auto_merge_min_segments = 2
auto_merge_streams = ("raw", "annotated", "thermal")
```

## Pixhawk Diagnostics

Use the MAVLink diagnostic utility to verify telemetry before deployment:

```bash
python3 pixhawk_test.py --port /dev/serial/by-id/<pixhawk-device> --baud 115200
```

The diagnostic output reports heartbeat state, flight mode, battery, GPS quality, altitude,
ground speed, attitude, and current position.

## systemd Deployment

A sample unit is provided at:

```text
systemd/pixhawk-drone-recorder.service
```

Update `User`, `WorkingDirectory`, and `ExecStart` paths for the deployment machine before enabling
the service.

## Notes

- Generated flight data, plots, local cache files, and Python bytecode are ignored by `.gitignore`.
- Older/custom YOLO checkpoint compatibility is handled by `pixhawk_recorder/compat.py`.
- Pixhawk-derived timestamps are available through `time_sync.py` and do not modify the Linux system clock.
