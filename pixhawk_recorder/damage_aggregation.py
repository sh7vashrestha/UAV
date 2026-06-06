"""DamageScout-style building-level aggregation for post-flight damage records."""

from __future__ import annotations

import ast
import csv
import html
import json
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


EARTH_RADIUS_M = 6378137.0
DEFAULT_SPATIAL_THRESHOLD_M = 10.0
DEFAULT_TEMPORAL_THRESHOLD_SEC = 5.0
DEFAULT_CONSENSUS_METHOD = "top_k_vote"
DEFAULT_TOP_K = 3
DEFAULT_OUTPUT_DIR_NAME = "damage_records"

DAMAGE_LABEL_ORDER = ["no_damage", "light_damage", "moderate_damage", "severe_damage"]
GRAPH_BEFORE_COLOR = "#4E79A7"
GRAPH_AFTER_COLOR = "#59A14F"
DAMAGE_LABEL_ALIASES = {
    "no": "no_damage",
    "none": "no_damage",
    "normal": "no_damage",
    "no_damage": "no_damage",
    "no damage": "no_damage",
    "light": "light_damage",
    "minor": "light_damage",
    "light_damage": "light_damage",
    "light damage": "light_damage",
    "moderate": "moderate_damage",
    "medium": "moderate_damage",
    "moderate_damage": "moderate_damage",
    "moderate damage": "moderate_damage",
    "severe": "severe_damage",
    "sever": "severe_damage",
    "severe_damage": "severe_damage",
    "sever_damage": "severe_damage",
    "severe damage": "severe_damage",
    "sever damage": "severe_damage",
}


@dataclass
class Detection:
    timestamp: float
    frame_id: str
    detection_id: str
    bbox: tuple[int, int, int, int]
    label: str
    confidence: float
    latitude: float
    longitude: float
    crop_path: str = ""
    source_row: dict[str, Any] = field(default_factory=dict)


@dataclass
class DamageRecord:
    record_id: str
    consensus_label: str
    aggregated_confidence: float
    global_confidence_score: float
    estimated_latitude: float
    estimated_longitude: float
    start_time: float
    end_time: float
    num_supporting_detections: int
    supporting_frame_ids: list[str]
    supporting_detection_ids: list[str]
    supporting_bounding_boxes: list[list[int]]
    supporting_labels: list[str]
    supporting_confidences: list[float]
    supporting_crop_paths: list[str]


@dataclass
class Summary:
    total_processed_frames: int
    total_frame_level_detections: int
    total_building_level_records: int
    average_detections_per_record: float
    duplicate_reduction_ratio: float
    global_confidence_score: float


@dataclass
class AggregationResult:
    detections: list[Detection]
    records: list[DamageRecord]
    summary: Summary
    output_dir: Path
    json_path: Path
    csv_path: Path
    class_summary_path: Path
    graph_svg_path: Path
    graph_pdf_path: Path
    raw_detections_path: Optional[Path] = None


class DetectionGroup:
    def __init__(self, group_id: int, detection: Detection) -> None:
        self.group_id = group_id
        self.detections = [detection]

    @property
    def start_time(self) -> float:
        return min(detection.timestamp for detection in self.detections)

    @property
    def end_time(self) -> float:
        return max(detection.timestamp for detection in self.detections)

    def add(self, detection: Detection) -> None:
        self.detections.append(detection)

    def distance_to_m(self, detection: Detection) -> float:
        lat, lon = weighted_location(self.detections)
        return haversine_m(lat, lon, detection.latitude, detection.longitude)

    def time_gap_to_sec(self, detection: Detection) -> float:
        if self.start_time <= detection.timestamp <= self.end_time:
            return 0.0
        return min(
            abs(detection.timestamp - self.start_time),
            abs(detection.timestamp - self.end_time),
        )


def aggregate_flight_damage_records(
    input_path: Path,
    *,
    output_dir: Optional[Path] = None,
    spatial_threshold_m: float = DEFAULT_SPATIAL_THRESHOLD_M,
    temporal_threshold_sec: float = DEFAULT_TEMPORAL_THRESHOLD_SEC,
    consensus_method: str = DEFAULT_CONSENSUS_METHOD,
    top_k: int = DEFAULT_TOP_K,
    debug_raw_detections: bool = False,
) -> AggregationResult:
    """Create DamageScout-style building records from geolocated frame detections.

    This post-processing stage converts repeated frame-level detections into compact
    building-level damage records. It does not run or alter the detector; it only
    consumes the crop geolocation CSV written after detections are produced.
    """
    validate_aggregation_options(
        spatial_threshold_m=spatial_threshold_m,
        temporal_threshold_sec=temporal_threshold_sec,
        consensus_method=consensus_method,
        top_k=top_k,
    )

    csv_path, crops_dir, output_dir = resolve_paths(input_path, output_dir)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    detections = load_detections(csv_path, crops_dir)
    records = aggregate_records(
        detections,
        spatial_threshold_m=spatial_threshold_m,
        temporal_threshold_sec=temporal_threshold_sec,
        consensus_method=consensus_method,
        top_k=top_k,
    )
    summary = build_summary(detections, records)

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "damage_building_records.json"
    csv_out_path = output_dir / "damage_building_records.csv"
    class_summary_path = output_dir / "class_before_after_summary.csv"
    graph_path = output_dir / "before_after_class_counts.svg"
    graph_pdf_path = output_dir / "before_after_class_counts.pdf"
    write_json(json_path, records, summary)
    write_csv(csv_out_path, records)
    write_class_summary_csv(class_summary_path, detections, records)
    write_before_after_svg(graph_path, detections, records, summary)
    write_before_after_pdf(graph_pdf_path, detections, records, summary)
    raw_detections_path = None
    if debug_raw_detections:
        raw_detections_path = output_dir / "raw_detections_debug.json"
        write_raw_detections(raw_detections_path, detections)

    return AggregationResult(
        detections=detections,
        records=records,
        summary=summary,
        output_dir=output_dir,
        json_path=json_path,
        csv_path=csv_out_path,
        class_summary_path=class_summary_path,
        graph_svg_path=graph_path,
        graph_pdf_path=graph_pdf_path,
        raw_detections_path=raw_detections_path,
    )


def validate_aggregation_options(
    *,
    spatial_threshold_m: float,
    temporal_threshold_sec: float,
    consensus_method: str,
    top_k: int,
) -> None:
    if spatial_threshold_m < 0:
        raise ValueError("spatial_threshold_m must be non-negative")
    if temporal_threshold_sec < 0:
        raise ValueError("temporal_threshold_sec must be non-negative")
    if consensus_method not in {"top_k_vote", "confidence_sum"}:
        raise ValueError("consensus_method must be 'top_k_vote' or 'confidence_sum'")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")


def resolve_paths(input_path: Path, output_dir: Optional[Path]) -> tuple[Path, Path, Path]:
    input_path = input_path.expanduser().resolve()
    if input_path.is_file():
        csv_path = input_path
        crops_dir = input_path.parent
        default_output_dir = input_path.parent / DEFAULT_OUTPUT_DIR_NAME
    else:
        candidates = [
            input_path / "crop_geolocation.csv",
            input_path / "crops" / "crop_geolocation.csv",
        ]
        csv_path = next((path for path in candidates if path.exists()), candidates[-1])
        crops_dir = csv_path.parent
        default_output_dir = input_path / DEFAULT_OUTPUT_DIR_NAME

    resolved_output = output_dir.expanduser().resolve() if output_dir else default_output_dir
    return csv_path, crops_dir, resolved_output


def load_detections(csv_path: Path, crops_dir: Path) -> list[Detection]:
    detections: list[Detection] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader, start=1):
            detection = detection_from_row(row, row_index=row_index, crops_dir=crops_dir)
            if detection is not None:
                detections.append(detection)
    return detections


def detection_from_row(row: dict[str, Any], *, row_index: int, crops_dir: Path) -> Optional[Detection]:
    if not is_target_valid(row):
        return None

    lat = parse_float(first_present(row, "target_lat", "estimated_latitude", "latitude", "lat"))
    lon = parse_float(first_present(row, "target_lon", "estimated_longitude", "longitude", "lon"))
    if lat is None or lon is None:
        return None

    timestamp = parse_float(first_present(row, "timestamp", "t_sec", "time_sec"))
    if timestamp is None:
        timestamp = float(row_index)

    frame_id = str(first_present(row, "frame_id", "frame", "frame_count", "infer_id") or row_index)
    det_index = str(first_present(row, "detection_index", "det_index", "det_id") or row_index)
    crop_path = normalize_crop_path(str(first_present(row, "crop_path", "crop_file") or ""), crops_dir)
    detection_id = str(first_present(row, "detection_id", "crop_file") or f"{frame_id}:{det_index}")
    label = normalize_label(
        str(
            first_present(row, "damage_label", "predicted_damage_label", "class_name", "label")
            or infer_label_from_path(crop_path)
            or "unknown"
        )
    )

    confidence = parse_float(first_present(row, "confidence", "conf", "score"))
    if confidence is None:
        confidence = parse_confidence_from_filename(crop_path)
    if confidence is None:
        confidence = 0.0

    bbox = parse_bbox(row)
    if bbox is None:
        return None

    return Detection(
        timestamp=timestamp,
        frame_id=frame_id,
        detection_id=detection_id,
        bbox=bbox,
        label=label,
        confidence=confidence,
        latitude=lat,
        longitude=lon,
        crop_path=crop_path,
        source_row=dict(row),
    )


def aggregate_records(
    detections: list[Detection],
    *,
    spatial_threshold_m: float,
    temporal_threshold_sec: float,
    consensus_method: str,
    top_k: int,
) -> list[DamageRecord]:
    groups: list[DetectionGroup] = []
    for detection in sorted(detections, key=lambda item: (item.timestamp, item.frame_id, item.detection_id)):
        matches: list[tuple[float, float, int, DetectionGroup]] = []
        for group in groups:
            distance_m = group.distance_to_m(detection)
            time_gap_sec = group.time_gap_to_sec(detection)
            if distance_m <= spatial_threshold_m and time_gap_sec <= temporal_threshold_sec:
                matches.append((distance_m, time_gap_sec, group.group_id, group))

        if matches:
            _, _, _, nearest_group = min(matches, key=lambda item: (item[0], item[1], item[2]))
            nearest_group.add(detection)
        else:
            groups.append(DetectionGroup(len(groups) + 1, detection))

    return [
        collapse_group(group, index, consensus_method=consensus_method, top_k=top_k)
        for index, group in enumerate(groups, start=1)
    ]


def collapse_group(
    group: DetectionGroup,
    record_index: int,
    *,
    consensus_method: str,
    top_k: int,
) -> DamageRecord:
    detections = group.detections
    consensus_label, global_confidence_score = choose_consensus_label(
        detections,
        method=consensus_method,
        top_k=top_k,
    )
    aggregated_confidence = sum(detection.confidence for detection in detections) / len(detections)
    lat, lon = weighted_location(detections)

    return DamageRecord(
        record_id=f"building_{record_index:06d}",
        consensus_label=consensus_label,
        aggregated_confidence=round(aggregated_confidence, 6),
        global_confidence_score=round(global_confidence_score, 6),
        estimated_latitude=round(lat, 8),
        estimated_longitude=round(lon, 8),
        start_time=round(group.start_time, 3),
        end_time=round(group.end_time, 3),
        num_supporting_detections=len(detections),
        supporting_frame_ids=[detection.frame_id for detection in detections],
        supporting_detection_ids=[detection.detection_id for detection in detections],
        supporting_bounding_boxes=[list(detection.bbox) for detection in detections],
        supporting_labels=[detection.label for detection in detections],
        supporting_confidences=[round(detection.confidence, 6) for detection in detections],
        supporting_crop_paths=[detection.crop_path for detection in detections],
    )


def choose_consensus_label(
    detections: list[Detection],
    *,
    method: str,
    top_k: int,
) -> tuple[str, float]:
    if method == "confidence_sum":
        return choose_consensus_by_confidence_sum(detections)
    return choose_consensus_by_top_k_vote(detections, top_k=max(top_k, 1))


def choose_consensus_by_top_k_vote(
    detections: list[Detection],
    *,
    top_k: int,
) -> tuple[str, float]:
    top_detections = sorted(
        detections,
        key=lambda detection: (detection.confidence, detection.timestamp, detection.detection_id),
        reverse=True,
    )[:top_k]
    label_counts: dict[str, int] = defaultdict(int)
    confidence_by_label: dict[str, float] = defaultdict(float)
    for detection in top_detections:
        label_counts[detection.label] += 1
        confidence_by_label[detection.label] += detection.confidence

    # This follows the requested top-k logic: look only at the highest-confidence
    # detections, then choose the label with the highest count. Confidence sum is
    # used only to break ties between labels with the same count.
    consensus_label = max(
        label_counts,
        key=lambda label: (label_counts[label], confidence_by_label[label], label),
    )
    global_confidence_score = label_counts[consensus_label] / len(top_detections)
    return consensus_label, global_confidence_score


def choose_consensus_by_confidence_sum(detections: list[Detection]) -> tuple[str, float]:
    confidence_by_label: dict[str, float] = defaultdict(float)
    for detection in detections:
        confidence_by_label[detection.label] += detection.confidence

    total_confidence = sum(confidence_by_label.values())
    consensus_label, consensus_total = max(
        confidence_by_label.items(),
        key=lambda item: (item[1], item[0]),
    )
    global_confidence_score = consensus_total / total_confidence if total_confidence > 0 else 0.0
    return consensus_label, global_confidence_score


def build_summary(detections: list[Detection], records: list[DamageRecord]) -> Summary:
    total_detections = len(detections)
    total_records = len(records)
    avg_per_record = total_detections / total_records if total_records else 0.0
    duplicate_reduction = 1.0 - (total_records / total_detections) if total_detections else 0.0
    if records:
        global_confidence = sum(
            record.global_confidence_score * record.num_supporting_detections
            for record in records
        ) / sum(record.num_supporting_detections for record in records)
    else:
        global_confidence = 0.0

    return Summary(
        total_processed_frames=infer_processed_frames(detections),
        total_frame_level_detections=total_detections,
        total_building_level_records=total_records,
        average_detections_per_record=round(avg_per_record, 6),
        duplicate_reduction_ratio=round(duplicate_reduction, 6),
        global_confidence_score=round(global_confidence, 6),
    )


def weighted_location(detections: list[Detection]) -> tuple[float, float]:
    total_weight = sum(max(detection.confidence, 0.0) for detection in detections)
    if total_weight <= 0:
        total_weight = float(len(detections))
        return (
            sum(detection.latitude for detection in detections) / total_weight,
            sum(detection.longitude for detection in detections) / total_weight,
        )

    # This directly averages latitude/longitude because small UAV missions cover
    # local areas. For larger regions, use a local projected coordinate system.
    return (
        sum(detection.latitude * max(detection.confidence, 0.0) for detection in detections) / total_weight,
        sum(detection.longitude * max(detection.confidence, 0.0) for detection in detections) / total_weight,
    )


def write_json(path: Path, records: list[DamageRecord], summary: Summary) -> None:
    payload = {
        "summary": asdict(summary),
        "records": [asdict(record) for record in records],
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def write_csv(path: Path, records: list[DamageRecord]) -> None:
    fieldnames = list(DamageRecord.__dataclass_fields__)
    list_fields = {
        "supporting_frame_ids",
        "supporting_detection_ids",
        "supporting_bounding_boxes",
        "supporting_labels",
        "supporting_confidences",
        "supporting_crop_paths",
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = asdict(record)
            for field_name in list_fields:
                row[field_name] = json.dumps(row[field_name])
            writer.writerow(row)


def write_raw_detections(path: Path, detections: list[Detection]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump([asdict(detection) for detection in detections], handle, indent=2)
        handle.write("\n")


def write_class_summary_csv(
    path: Path,
    detections: list[Detection],
    records: list[DamageRecord],
) -> None:
    before_counts = count_detection_labels(detections)
    after_counts = count_record_labels(records)
    labels = ordered_damage_labels(before_counts, after_counts)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "damage_class",
                "frame_level_detections_before",
                "building_level_records_after",
                "duplicate_reduction_count",
                "duplicate_reduction_ratio",
            ],
        )
        writer.writeheader()
        for label in labels:
            before = before_counts.get(label, 0)
            after = after_counts.get(label, 0)
            reduction = before - after
            ratio = 1.0 - (after / before) if before > 0 else 0.0
            writer.writerow(
                {
                    "damage_class": label,
                    "frame_level_detections_before": before,
                    "building_level_records_after": after,
                    "duplicate_reduction_count": reduction,
                    "duplicate_reduction_ratio": round(ratio, 6),
                }
            )


def write_before_after_svg(
    path: Path,
    detections: list[Detection],
    records: list[DamageRecord],
    summary: Summary,
) -> None:
    before_counts = count_detection_labels(detections)
    after_counts = count_record_labels(records)
    labels = ordered_damage_labels(before_counts, after_counts)
    if not labels:
        labels = DAMAGE_LABEL_ORDER[:]

    max_count = max(
        [1]
        + [before_counts.get(label, 0) for label in labels]
        + [after_counts.get(label, 0) for label in labels]
    )
    y_tick_count = 5
    tick_step = nice_tick_step(max_count, y_tick_count)
    y_max = max(tick_step * y_tick_count, max_count)

    width = max(680, 150 + (len(labels) * 115))
    height = 430
    margin_left = 74
    margin_right = 34
    margin_top = 66
    margin_bottom = 78
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom
    group_width = chart_width / max(len(labels), 1)
    bar_width = min(48, max(18, group_width * 0.34))

    def x_for_group(index: int) -> float:
        return margin_left + (index * group_width) + (group_width / 2.0)

    def y_for_count(count: int | float) -> float:
        return margin_top + chart_height - ((count / y_max) * chart_height)

    svg: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: 'Times New Roman', Times, serif; fill: #1f2933; }",
        ".axis { stroke: #263238; stroke-width: 1.2; }",
        ".grid { stroke: #d8dee5; stroke-width: 1; }",
        ".tick { font-size: 14px; fill: #52616b; }",
        ".label { font-size: 15px; fill: #263238; }",
        ".value { font-size: 17px; font-weight: 700; fill: #111827; }",
        ".legend { font-size: 15px; fill: #263238; }",
        "</style>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
    ]

    legend_x = margin_left
    legend_y = 22
    svg.extend(
        [
            f'<rect x="{legend_x}" y="{legend_y}" width="16" height="16" fill="{GRAPH_BEFORE_COLOR}"/>',
            f'<text x="{legend_x + 22}" y="{legend_y + 13}" class="legend">Pre-Aggregation</text>',
            f'<rect x="{legend_x + 186}" y="{legend_y}" width="16" height="16" fill="{GRAPH_AFTER_COLOR}"/>',
            f'<text x="{legend_x + 208}" y="{legend_y + 13}" class="legend">Post-Aggregation</text>',
        ]
    )

    for tick_index in range(y_tick_count + 1):
        tick_value = tick_index * tick_step
        y = y_for_count(tick_value)
        svg.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" class="grid"/>'
        )
        svg.append(
            f'<text x="{margin_left - 12}" y="{y + 4:.2f}" text-anchor="end" class="tick">{tick_value:g}</text>'
        )

    svg.extend(
        [
            f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + chart_height}" class="axis"/>',
            f'<line x1="{margin_left}" y1="{margin_top + chart_height}" x2="{width - margin_right}" y2="{margin_top + chart_height}" class="axis"/>',
            (
                f'<text x="24" y="{margin_top + chart_height / 2}" transform="rotate(-90 24 {margin_top + chart_height / 2})" '
                'text-anchor="middle" class="label">Count</text>'
            ),
        ]
    )

    for index, label in enumerate(labels):
        center_x = x_for_group(index)
        before = before_counts.get(label, 0)
        after = after_counts.get(label, 0)
        before_height = (before / y_max) * chart_height
        after_height = (after / y_max) * chart_height
        before_x = center_x - bar_width - 2
        after_x = center_x + 2
        before_y = margin_top + chart_height - before_height
        after_y = margin_top + chart_height - after_height

        svg.extend(
            [
                (
                    f'<rect x="{before_x:.2f}" y="{before_y:.2f}" width="{bar_width:.2f}" '
                    f'height="{before_height:.2f}" fill="{GRAPH_BEFORE_COLOR}"/>'
                ),
                (
                    f'<rect x="{after_x:.2f}" y="{after_y:.2f}" width="{bar_width:.2f}" '
                    f'height="{after_height:.2f}" fill="{GRAPH_AFTER_COLOR}"/>'
                ),
                f'<text x="{before_x + bar_width / 2:.2f}" y="{before_y - 6:.2f}" text-anchor="middle" class="value">{before}</text>',
                f'<text x="{after_x + bar_width / 2:.2f}" y="{after_y - 6:.2f}" text-anchor="middle" class="value">{after}</text>',
                (
                    f'<text x="{center_x:.2f}" y="{margin_top + chart_height + 24}" '
                    f'text-anchor="middle" class="label">{html.escape(pretty_label(label))}</text>'
                ),
            ]
        )

    svg.extend(["</svg>", ""])
    path.write_text("\n".join(svg), encoding="utf-8")


def write_before_after_pdf(
    path: Path,
    detections: list[Detection],
    records: list[DamageRecord],
    summary: Summary,
) -> None:
    before_counts = count_detection_labels(detections)
    after_counts = count_record_labels(records)
    labels = ordered_damage_labels(before_counts, after_counts)
    if not labels:
        labels = DAMAGE_LABEL_ORDER[:]

    max_count = max(
        [1]
        + [before_counts.get(label, 0) for label in labels]
        + [after_counts.get(label, 0) for label in labels]
    )
    y_tick_count = 5
    tick_step = nice_tick_step(max_count, y_tick_count)
    y_max = max(tick_step * y_tick_count, max_count)

    width = max(680, 150 + (len(labels) * 115))
    height = 430
    margin_left = 74
    margin_right = 34
    margin_top = 66
    margin_bottom = 78
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom
    group_width = chart_width / max(len(labels), 1)
    bar_width = min(48, max(18, group_width * 0.34))

    commands: list[str] = []

    def pdf_y(svg_y: float) -> float:
        return height - svg_y

    def set_fill(hex_color: str) -> None:
        r, g, b = hex_to_rgb01(hex_color)
        commands.append(f"{r:.4f} {g:.4f} {b:.4f} rg")

    def set_stroke(hex_color: str) -> None:
        r, g, b = hex_to_rgb01(hex_color)
        commands.append(f"{r:.4f} {g:.4f} {b:.4f} RG")

    def rect(x: float, y_top: float, w: float, h: float, color: str) -> None:
        set_fill(color)
        commands.append(f"{x:.2f} {pdf_y(y_top + h):.2f} {w:.2f} {h:.2f} re f")

    def line(x1: float, y1_top: float, x2: float, y2_top: float, color: str, width_pt: float = 1.0) -> None:
        set_stroke(color)
        commands.append(f"{width_pt:.2f} w")
        commands.append(f"{x1:.2f} {pdf_y(y1_top):.2f} m {x2:.2f} {pdf_y(y2_top):.2f} l S")

    def text(
        x: float,
        y_top: float,
        value: str,
        *,
        size: float = 12.0,
        color: str = "#1f2933",
        align: str = "left",
    ) -> None:
        display = str(value)
        text_width = approximate_text_width(display, size)
        if align == "center":
            x -= text_width / 2.0
        elif align == "right":
            x -= text_width
        r, g, b = hex_to_rgb01(color)
        commands.append("BT")
        commands.append(f"/F1 {size:.2f} Tf")
        commands.append(f"{r:.4f} {g:.4f} {b:.4f} rg")
        commands.append(f"1 0 0 1 {x:.2f} {pdf_y(y_top):.2f} Tm")
        commands.append(f"({pdf_escape(display)}) Tj")
        commands.append("ET")

    rect(0, 0, width, height, "#ffffff")
    legend_x = margin_left
    legend_y = 22
    rect(legend_x, legend_y, 16, 16, GRAPH_BEFORE_COLOR)
    text(legend_x + 22, legend_y + 13, "Pre-Aggregation", size=14)
    rect(legend_x + 186, legend_y, 16, 16, GRAPH_AFTER_COLOR)
    text(legend_x + 208, legend_y + 13, "Post-Aggregation", size=14)

    def x_for_group(index: int) -> float:
        return margin_left + (index * group_width) + (group_width / 2.0)

    def y_for_count(count: int | float) -> float:
        return margin_top + chart_height - ((count / y_max) * chart_height)

    for tick_index in range(y_tick_count + 1):
        tick_value = tick_index * tick_step
        y = y_for_count(tick_value)
        line(margin_left, y, width - margin_right, y, "#d8dee5", 0.8)
        text(margin_left - 12, y + 5, f"{tick_value:g}", size=13, color="#52616b", align="right")

    line(margin_left, margin_top, margin_left, margin_top + chart_height, "#263238", 1.2)
    line(
        margin_left,
        margin_top + chart_height,
        width - margin_right,
        margin_top + chart_height,
        "#263238",
        1.2,
    )
    text(26, margin_top + (chart_height / 2), "Count", size=14, color="#263238", align="center")

    for index, label in enumerate(labels):
        center_x = x_for_group(index)
        before = before_counts.get(label, 0)
        after = after_counts.get(label, 0)
        before_height = (before / y_max) * chart_height
        after_height = (after / y_max) * chart_height
        before_x = center_x - bar_width - 2
        after_x = center_x + 2
        before_y = margin_top + chart_height - before_height
        after_y = margin_top + chart_height - after_height

        rect(before_x, before_y, bar_width, before_height, GRAPH_BEFORE_COLOR)
        rect(after_x, after_y, bar_width, after_height, GRAPH_AFTER_COLOR)
        text(before_x + (bar_width / 2), before_y - 7, str(before), size=15, align="center")
        text(after_x + (bar_width / 2), after_y - 7, str(after), size=15, align="center")
        text(center_x, margin_top + chart_height + 26, pretty_label(label), size=13, align="center")

    write_pdf(path, width, height, "\n".join(commands) + "\n")


def write_pdf(path: Path, width: int, height: int, content: str) -> None:
    content_bytes = content.encode("latin-1", errors="replace")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] "
            "/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ).encode("latin-1"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Times-Roman >>",
        b"<< /Length " + str(len(content_bytes)).encode("ascii") + b" >>\nstream\n"
        + content_bytes
        + b"endstream",
    ]

    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode("ascii"))
        output.extend(obj)
        output.extend(b"\nendobj\n")

    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    path.write_bytes(bytes(output))


def hex_to_rgb01(hex_color: str) -> tuple[float, float, float]:
    value = hex_color.strip().lstrip("#")
    return (
        int(value[0:2], 16) / 255.0,
        int(value[2:4], 16) / 255.0,
        int(value[4:6], 16) / 255.0,
    )


def approximate_text_width(value: str, size: float) -> float:
    return len(value) * size * 0.52


def pdf_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def count_detection_labels(detections: list[Detection]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for detection in detections:
        counts[detection.label] += 1
    return dict(counts)


def count_record_labels(records: list[DamageRecord]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        counts[record.consensus_label] += 1
    return dict(counts)


def ordered_damage_labels(before_counts: dict[str, int], after_counts: dict[str, int]) -> list[str]:
    all_labels = set(before_counts) | set(after_counts)
    ordered = [label for label in DAMAGE_LABEL_ORDER if label in all_labels]
    ordered.extend(sorted(label for label in all_labels if label not in DAMAGE_LABEL_ORDER))
    return ordered


def nice_tick_step(max_count: int, tick_count: int) -> int:
    raw_step = max(max_count / max(tick_count, 1), 1)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    for multiplier in (1, 2, 5, 10):
        step = multiplier * magnitude
        if step >= raw_step:
            return int(max(step, 1))
    return int(max(10 * magnitude, 1))


def pretty_label(label: str) -> str:
    return label.replace("_", " ").title()


def print_summary(summary: Summary) -> None:
    print("Mission summary:")
    print(f"  total processed frames: {summary.total_processed_frames}")
    print(f"  total frame-level detections: {summary.total_frame_level_detections}")
    print(f"  total building-level records: {summary.total_building_level_records}")
    print(f"  average detections per record: {summary.average_detections_per_record:.3f}")
    print(f"  duplicate-reduction ratio: {summary.duplicate_reduction_ratio:.3f}")
    print(f"  global confidence score: {summary.global_confidence_score:.4f}")


def normalize_crop_path(value: str, crops_dir: Path) -> str:
    if not value:
        return ""
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((crops_dir / path).resolve())


def infer_label_from_path(crop_path: str) -> Optional[str]:
    if not crop_path:
        return None
    parts = [part.replace("_", " ").lower() for part in Path(crop_path).parts]
    for part in reversed(parts):
        normalized = normalize_label(part)
        if normalized != "unknown":
            return normalized
    return None


def normalize_label(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", value.strip().lower()).strip()
    if not text:
        return "unknown"
    return DAMAGE_LABEL_ALIASES.get(text, text.replace(" ", "_"))


def parse_confidence_from_filename(crop_path: str) -> Optional[float]:
    if not crop_path:
        return None
    match = re.search(r"_(0(?:\.\d+)?|1(?:\.0+)?)\.[A-Za-z0-9]+$", Path(crop_path).name)
    if match is None:
        return None
    return parse_float(match.group(1))


def parse_bbox(row: dict[str, Any]) -> Optional[tuple[int, int, int, int]]:
    bbox_text = first_present(row, "bbox", "bounding_box")
    if bbox_text:
        try:
            value = ast.literal_eval(str(bbox_text))
            if len(value) == 4:
                return tuple(int(round(float(part))) for part in value)  # type: ignore[return-value]
        except (SyntaxError, ValueError, TypeError):
            pass

    values = [
        parse_float(first_present(row, "x1", "xmin", "left")),
        parse_float(first_present(row, "y1", "ymin", "top")),
        parse_float(first_present(row, "x2", "xmax", "right")),
        parse_float(first_present(row, "y2", "ymax", "bottom")),
    ]
    if any(value is None for value in values):
        return None
    return tuple(int(round(float(value))) for value in values)  # type: ignore[return-value]


def is_target_valid(row: dict[str, Any]) -> bool:
    value = row.get("target_valid")
    if value in (None, ""):
        return True
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def infer_processed_frames(detections: list[Detection]) -> int:
    if not detections:
        return 0
    numeric_ids: list[int] = []
    for detection in detections:
        try:
            numeric_ids.append(int(detection.frame_id))
        except ValueError:
            return len({detection.frame_id for detection in detections})
    return max(numeric_ids) if numeric_ids else 0


def first_present(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def parse_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def haversine_m(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    lat_a_rad = math.radians(lat_a)
    lat_b_rad = math.radians(lat_b)
    d_lat = lat_b_rad - lat_a_rad
    d_lon = math.radians(lon_b - lon_a)
    a = (
        math.sin(d_lat / 2.0) ** 2
        + math.cos(lat_a_rad) * math.cos(lat_b_rad) * math.sin(d_lon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_M * math.atan2(math.sqrt(a), math.sqrt(max(1.0 - a, 0.0)))
