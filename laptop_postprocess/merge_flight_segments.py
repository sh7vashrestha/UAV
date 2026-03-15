#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from pixhawk_recorder.postflight_merge import STREAMS, merge_flight_segments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge segmented Pixhawk flight videos into single MP4 files."
    )
    parser.add_argument(
        "flight_dir",
        type=Path,
        help="Path to a single flight directory, for example flight_20260309_143906",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for merged outputs. Defaults to <flight_dir>/merged",
    )
    parser.add_argument(
        "--streams",
        nargs="+",
        choices=sorted(STREAMS),
        default=sorted(STREAMS),
        help="Which stream sets to merge",
    )
    parser.add_argument(
        "--ffmpeg",
        default="ffmpeg",
        help="Path to the ffmpeg binary on your laptop",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite merged outputs if they already exist",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the merge plan without calling ffmpeg",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        merged = merge_flight_segments(
            flight_dir=args.flight_dir,
            output_dir=args.output_dir,
            streams=args.streams,
            ffmpeg_bin=args.ffmpeg,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            report=print,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not merged:
        print("No matching segments were found.", file=sys.stderr)
        return 1

    if not args.dry_run:
        output_dir = (
            args.output_dir.expanduser().resolve()
            if args.output_dir is not None
            else (args.flight_dir.expanduser().resolve() / "merged")
        )
        print(f"Merged files written to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
