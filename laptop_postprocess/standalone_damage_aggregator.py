#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pixhawk_recorder.damage_aggregation import (  # noqa: E402
    DEFAULT_CONSENSUS_METHOD,
    DEFAULT_SPATIAL_THRESHOLD_M,
    DEFAULT_TEMPORAL_THRESHOLD_SEC,
    DEFAULT_TOP_K,
    aggregate_flight_damage_records,
    print_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate geo-localized frame-level damage detections into final "
            "building-level records for post-flight analysis."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Flight/crops directory containing crop_geolocation.csv, or the CSV path itself",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <input>/damage_records or <csv_parent>/damage_records",
    )
    parser.add_argument(
        "--spatial-threshold-m",
        type=float,
        default=DEFAULT_SPATIAL_THRESHOLD_M,
        help="Maximum GPS distance in meters for detections to join one group",
    )
    parser.add_argument(
        "--temporal-threshold-sec",
        type=float,
        default=DEFAULT_TEMPORAL_THRESHOLD_SEC,
        help="Maximum time gap in seconds for detections to join one group",
    )
    parser.add_argument(
        "--debug-raw-detections",
        action="store_true",
        help="Also export normalized frame-level detections for debugging",
    )
    parser.add_argument(
        "--consensus-method",
        choices=("top_k_vote", "confidence_sum"),
        default=DEFAULT_CONSENSUS_METHOD,
        help=(
            "How to choose the final damage label. top_k_vote uses the top-K "
            "highest-confidence detections and picks the most frequent label."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of highest-confidence detections used by top_k_vote",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = aggregate_flight_damage_records(
            args.input,
            output_dir=args.output_dir,
            spatial_threshold_m=args.spatial_threshold_m,
            temporal_threshold_sec=args.temporal_threshold_sec,
            consensus_method=args.consensus_method,
            top_k=args.top_k,
            debug_raw_detections=args.debug_raw_detections,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Loaded detections: {len(result.detections)}")
    print(f"Building-level records: {len(result.records)}")
    print(f"Global confidence score: {result.summary.global_confidence_score:.4f}")
    print(f"JSON: {result.json_path}")
    print(f"CSV: {result.csv_path}")
    print(f"Class summary CSV: {result.class_summary_path}")
    print(f"Before/after graph SVG: {result.graph_svg_path}")
    print(f"Before/after graph PDF: {result.graph_pdf_path}")
    if result.raw_detections_path is not None:
        print(f"Raw detections debug JSON: {result.raw_detections_path}")
    print_summary(result.summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
