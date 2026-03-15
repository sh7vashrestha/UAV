# Laptop Post-Processing

Use the tools in this directory on your laptop after copying a flight folder off the Jetson.

## What it does

`merge_flight_segments.py` merges segmented MP4 files into single MP4 outputs without re-encoding.
The same merge logic is also used by the optional Jetson-side automatic post-flight merge.

It handles these streams when present:
- `raw`
- `annotated`
- `thermal`

## Requirements

- Python 3.10+
- `ffmpeg` installed on your laptop

## Example

```bash
python3 laptop_postprocess/merge_flight_segments.py /path/to/flight_2026-03-12_14-33-07
```

Merged files are written to:

```text
/path/to/flight_2026-03-12_14-33-07/merged/
```

## Useful options

Merge only raw and annotated video:

```bash
python3 laptop_postprocess/merge_flight_segments.py /path/to/flight_2026-03-12_14-33-07 --streams raw annotated
```

Overwrite old merged outputs:

```bash
python3 laptop_postprocess/merge_flight_segments.py /path/to/flight_2026-03-12_14-33-07 --overwrite
```

Preview what would run without calling `ffmpeg`:

```bash
python3 laptop_postprocess/merge_flight_segments.py /path/to/flight_2026-03-12_14-33-07 --dry-run
```
