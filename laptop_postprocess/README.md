# Laptop Post-Processing

Use the tools in this directory on your laptop after copying a flight folder off the Jetson.

## What it does

`standalone_damage_aggregator.py` converts geo-localized frame-level detections into final
building-level damage records. It is a laptop-friendly wrapper around
`pixhawk_recorder/damage_aggregation.py`, the same aggregation code used by the Pixhawk recorder
after each flight.

`crop_size_stats.py` reports accurate crop file-size statistics from files on disk.

`merge_flight_segments.py` merges segmented MP4 files into single MP4 outputs without re-encoding.
The same merge logic is also used by the optional Jetson-side automatic post-flight merge.

It handles these streams when present:
- `raw`
- `annotated`
- `thermal`

## Requirements

- Python 3.10+
- `ffmpeg` installed on your laptop only if you want to merge MP4 segments

## Example

Generate or regenerate building-level damage records:

```bash
python3 laptop_postprocess/standalone_damage_aggregator.py /path/to/flight_0001
```

During normal Pixhawk recorder operation, these records are generated automatically after the
flight when `auto_aggregate_damage_records_after_flight` is enabled.

The main exported results are written to:

```text
/path/to/flight_0001/damage_records/damage_building_records.json
/path/to/flight_0001/damage_records/damage_building_records.csv
/path/to/flight_0001/damage_records/class_before_after_summary.csv
/path/to/flight_0001/damage_records/before_after_class_counts.pdf
/path/to/flight_0001/damage_records/before_after_class_counts.svg
```

The aggregation reads `crops/crop_geolocation.csv`, groups detections that are close in both
location and time, and collapses each group into one final record. By default, the consensus damage
label is chosen by taking the top 3 highest-confidence detections in a building group, voting by
label, and using summed confidence as the tie-breaker. The final location is the
confidence-weighted average of the grouped detection locations.

Calculate average crop file size:

```bash
python3 laptop_postprocess/crop_size_stats.py /path/to/flight_0001
```

Merge video segments:

```bash
python3 laptop_postprocess/merge_flight_segments.py /path/to/flight_0001
```

Merged files are written to:

```text
/path/to/flight_0001/merged/
```

## Flight Metadata

When a flight was recorded in Pixhawk mode, `flight_metadata.json` may now include:
- `pixhawk_connection_status`
- `pixhawk_vehicle_system_id`
- `pixhawk_vehicle_component_id`

Those fields are useful when you are reviewing a flight later and want to see which MAVLink
source the Jetson actually trusted during the recording.

## Useful options

Merge only raw and annotated video:

```bash
python3 laptop_postprocess/merge_flight_segments.py /path/to/flight_0001 --streams raw annotated
```

Overwrite old merged outputs:

```bash
python3 laptop_postprocess/merge_flight_segments.py /path/to/flight_0001 --overwrite
```

Preview what would run without calling `ffmpeg`:

```bash
python3 laptop_postprocess/merge_flight_segments.py /path/to/flight_0001 --dry-run
```

Tune building record grouping thresholds:

```bash
python3 laptop_postprocess/standalone_damage_aggregator.py /path/to/flight_0001 \
  --spatial-threshold-m 10 \
  --temporal-threshold-sec 10
```

Use a different number of high-confidence detections for consensus:

```bash
python3 laptop_postprocess/standalone_damage_aggregator.py /path/to/flight_0001 --top-k 5
```

Save the filtered frame-level detections used for debugging:

```bash
python3 laptop_postprocess/standalone_damage_aggregator.py /path/to/flight_0001 --debug-raw-detections
```
