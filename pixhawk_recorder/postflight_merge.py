from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Dict, Iterable, List


STREAMS: Dict[str, Dict[str, str]] = {
    "raw": {
        "subdir": "rgb",
        "pattern": "seg_*_raw.mp4",
    },
    "annotated": {
        "subdir": "rgb",
        "pattern": "seg_*_annotated.mp4",
    },
    "thermal": {
        "subdir": "thermal",
        "pattern": "seg_*_thermal.mp4",
    },
}


StatusReporter = Callable[[str], None]


def available_streams() -> List[str]:
    return sorted(STREAMS)


def validate_streams(streams: Iterable[str] | None) -> List[str]:
    if streams is None:
        return available_streams()

    validated: List[str] = []
    seen = set()
    invalid = []

    for stream_name in streams:
        if stream_name not in STREAMS:
            invalid.append(stream_name)
            continue
        if stream_name in seen:
            continue
        validated.append(stream_name)
        seen.add(stream_name)

    if invalid:
        raise ValueError(
            f"Unknown stream name(s): {', '.join(sorted(invalid))}. "
            f"Expected one of: {', '.join(available_streams())}"
        )

    return validated


def ensure_flight_dir(flight_dir: Path) -> Path:
    flight_dir = flight_dir.expanduser().resolve()
    if not flight_dir.exists():
        raise FileNotFoundError(f"Flight directory does not exist: {flight_dir}")
    if not flight_dir.is_dir():
        raise NotADirectoryError(f"Flight path is not a directory: {flight_dir}")
    return flight_dir


def resolve_ffmpeg(binary: str) -> str:
    if "/" in binary:
        candidate = Path(binary).expanduser()
        if candidate.exists():
            return str(candidate)
        raise FileNotFoundError(f"ffmpeg was not found at: {candidate}")

    resolved = shutil.which(binary)
    if resolved is None:
        raise FileNotFoundError(
            f"Could not find '{binary}' in PATH. Install ffmpeg first."
        )
    return resolved


def load_manifest_paths(flight_dir: Path) -> Dict[str, List[Path]]:
    manifest_file = flight_dir / "segments_manifest.jsonl"
    paths: Dict[str, List[Path]] = {name: [] for name in STREAMS}
    if not manifest_file.exists():
        return paths

    key_map = {
        "raw": "raw_path",
        "annotated": "annotated_path",
        "thermal": "thermal_path",
    }

    with manifest_file.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON in {manifest_file} at line {line_number}: {exc}"
                ) from exc

            for stream_name, manifest_key in key_map.items():
                raw_path = row.get(manifest_key)
                if not raw_path:
                    continue
                path = Path(raw_path)
                if path.is_absolute():
                    try:
                        flight_dir_index = path.parts.index(flight_dir.name)
                    except ValueError:
                        flight_dir_index = -1
                    if flight_dir_index >= 0:
                        remapped = flight_dir / Path(*path.parts[flight_dir_index + 1 :])
                        if remapped.exists():
                            path = remapped
                else:
                    path = flight_dir / path
                paths[stream_name].append(path)

    return paths


def collect_segments(flight_dir: Path, stream_name: str) -> List[Path]:
    manifest_paths = load_manifest_paths(flight_dir)
    stream_cfg = STREAMS[stream_name]
    globbed = sorted((flight_dir / stream_cfg["subdir"]).glob(stream_cfg["pattern"]))

    unique: List[Path] = []
    seen = set()
    for path in manifest_paths.get(stream_name, []) + globbed:
        resolved = path.expanduser().resolve()
        if not path.exists() or resolved in seen:
            continue
        unique.append(path)
        seen.add(resolved)

    return sorted(unique, key=lambda path: path.name)


def concat_line(path: Path) -> str:
    quoted = str(path.resolve()).replace("'", r"'\''")
    return f"file '{quoted}'\n"


def _report(report: StatusReporter | None, message: str) -> None:
    if report is not None:
        report(message)


def run_ffmpeg(
    ffmpeg_bin: str,
    inputs: Iterable[Path],
    output_file: Path,
    overwrite: bool,
    dry_run: bool,
    report: StatusReporter | None = None,
) -> None:
    inputs = list(inputs)
    if not inputs:
        raise ValueError("No input segments were provided")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_file} (use overwrite to replace it)"
        )

    with tempfile.TemporaryDirectory(prefix="pixhawk_merge_") as temp_dir:
        concat_file = Path(temp_dir) / "concat.txt"
        concat_file.write_text("".join(concat_line(path) for path in inputs), encoding="utf-8")

        command = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c",
            "copy",
        ]
        command.append("-y" if overwrite else "-n")
        command.append(str(output_file))

        if dry_run:
            _report(report, " ".join(command))
            return

        result = subprocess.run(command, text=True, capture_output=True, check=False)
        if result.returncode != 0:
            stderr = result.stderr.strip() or "ffmpeg exited with a non-zero status"
            raise RuntimeError(stderr)


def merge_flight_segments(
    flight_dir: Path,
    *,
    output_dir: Path | None = None,
    streams: Iterable[str] | None = None,
    ffmpeg_bin: str = "ffmpeg",
    overwrite: bool = False,
    dry_run: bool = False,
    min_segments: int = 1,
    report: StatusReporter | None = None,
) -> Dict[str, Path]:
    if min_segments < 1:
        raise ValueError("min_segments must be at least 1")

    flight_dir = ensure_flight_dir(flight_dir)
    stream_names = validate_streams(streams)
    resolved_ffmpeg = resolve_ffmpeg(ffmpeg_bin) if not dry_run else ffmpeg_bin
    resolved_output_dir = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else (flight_dir / "merged").resolve()
    )

    merged: Dict[str, Path] = {}
    for stream_name in stream_names:
        segments = collect_segments(flight_dir, stream_name)
        if not segments:
            _report(report, f"Skipping {stream_name}: no segments found in {flight_dir}")
            continue
        if len(segments) < min_segments:
            _report(
                report,
                f"Skipping {stream_name}: only {len(segments)} segment(s) found",
            )
            continue

        output_file = resolved_output_dir / f"{flight_dir.name}_{stream_name}_merged.mp4"
        _report(
            report,
            f"{'Would merge' if dry_run else 'Merging'} "
            f"{len(segments)} {stream_name} segment(s) -> {output_file}",
        )
        run_ffmpeg(
            ffmpeg_bin=resolved_ffmpeg,
            inputs=segments,
            output_file=output_file,
            overwrite=overwrite,
            dry_run=dry_run,
            report=report,
        )
        merged[stream_name] = output_file

    return merged
