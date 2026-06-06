#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass
class FileSizeStats:
    label: str
    count: int = 0
    total_bytes: int = 0
    min_bytes: int | None = None
    max_bytes: int | None = None

    def add(self, file_bytes: int) -> None:
        self.count += 1
        self.total_bytes += file_bytes
        self.min_bytes = file_bytes if self.min_bytes is None else min(self.min_bytes, file_bytes)
        self.max_bytes = file_bytes if self.max_bytes is None else max(self.max_bytes, file_bytes)

    @property
    def avg_bytes(self) -> float:
        return self.total_bytes / self.count if self.count else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate accurate crop file-size statistics from files on disk.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Flight directory, crops directory, or any directory containing crop images",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV output path for the file-size summary",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    crop_dir = resolve_crop_dir(args.input)
    if not crop_dir.exists():
        print(f"Error: crop directory not found: {crop_dir}", file=sys.stderr)
        return 1

    rows = calculate_file_size_stats(crop_dir)
    if not rows:
        print(f"No crop image files found under: {crop_dir}", file=sys.stderr)
        return 1

    print_stats(rows)
    if args.output_csv is not None:
        output_csv = args.output_csv.expanduser().resolve()
        write_stats_csv(output_csv, rows)
        print(f"CSV written to: {output_csv}")
    return 0


def resolve_crop_dir(input_path: Path) -> Path:
    input_path = input_path.expanduser().resolve()
    if (input_path / "crops").is_dir():
        return input_path / "crops"
    return input_path


def calculate_file_size_stats(crop_dir: Path) -> list[FileSizeStats]:
    by_label: dict[str, FileSizeStats] = {}
    overall = FileSizeStats("ALL_CROPS")

    for path in sorted(crop_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        file_bytes = path.stat().st_size
        label = path.parent.name
        by_label.setdefault(label, FileSizeStats(label)).add(file_bytes)
        overall.add(file_bytes)

    rows = [by_label[label] for label in sorted(by_label)]
    if overall.count:
        rows.append(overall)
    return rows


def print_stats(rows: list[FileSizeStats]) -> None:
    print(
        f"{'class':<24} {'files':>8} {'total_mib':>12} {'avg_bytes':>14} "
        f"{'avg_kib':>10} {'min_kib':>10} {'max_kib':>10}"
    )
    print("-" * 94)
    for row in rows:
        print(
            f"{row.label:<24} {row.count:>8} "
            f"{row.total_bytes / (1024.0 ** 2):>12.3f} "
            f"{row.avg_bytes:>14.1f} "
            f"{row.avg_bytes / 1024.0:>10.2f} "
            f"{(row.min_bytes or 0) / 1024.0:>10.2f} "
            f"{(row.max_bytes or 0) / 1024.0:>10.2f}"
        )

    overall = next((row for row in rows if row.label == "ALL_CROPS"), None)
    if overall is not None:
        print()
        print(
            "Overall average crop file size: "
            f"{overall.avg_bytes:.1f} bytes "
            f"({overall.avg_bytes / 1024.0:.2f} KiB, "
            f"{overall.avg_bytes / (1024.0 ** 2):.4f} MiB)"
        )


def write_stats_csv(path: Path, rows: list[FileSizeStats]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "class",
                "files",
                "total_bytes",
                "total_mib",
                "avg_bytes",
                "avg_kib",
                "avg_mib",
                "min_bytes",
                "max_bytes",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "class": row.label,
                    "files": row.count,
                    "total_bytes": row.total_bytes,
                    "total_mib": round(row.total_bytes / (1024.0 ** 2), 6),
                    "avg_bytes": round(row.avg_bytes, 3),
                    "avg_kib": round(row.avg_bytes / 1024.0, 6),
                    "avg_mib": round(row.avg_bytes / (1024.0 ** 2), 6),
                    "min_bytes": row.min_bytes or 0,
                    "max_bytes": row.max_bytes or 0,
                }
            )


if __name__ == "__main__":
    sys.exit(main())
