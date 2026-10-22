"""GTK-free helpers for running ffmpeg/whisper-cli and parsing their output."""

from __future__ import annotations

import re
from pathlib import Path

LINE_RE = re.compile(
    r"^\[(\d\d:\d\d:\d\d\.\d{3})\s*-->\s*(\d\d:\d\d:\d\d\.\d{3})\]\s+(.*)$"
)


def build_ffmpeg_cmd(input_path: Path, wav_path: Path) -> list[str]:
    """Build the ffmpeg argv that decodes `input_path` to 16 kHz mono WAV."""
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostats",
        "-progress",
        "pipe:1",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(wav_path),
        "-y",
    ]


def build_whisper_cmd(
    model: Path,
    wav_path: Path,
    lang: str,
    threads: int,
    translate: bool,
) -> list[str]:
    """Build the whisper-cli argv for transcribing `wav_path`."""
    cmd = [
        "whisper-cli",
        "-m",
        str(model),
        "-l",
        lang,
        "-t",
        str(threads),
        "-f",
        str(wav_path),
    ]
    if translate:
        cmd.append("--translate")
    return cmd


def parse_progress_line(line: str) -> float | None:
    """Return the decoded position in seconds from an ffmpeg `-progress` line."""
    stripped = line.strip()
    if not stripped.startswith("out_time_ms="):
        return None
    try:
        return int(stripped.split("=", 1)[1]) / 1_000_000.0
    except ValueError:
        return None


def parse_segment(line: str) -> tuple[str, str] | None:
    """Return `(end_timestamp, text)` for a whisper-cli segment line, else None."""
    match = LINE_RE.match(line)
    if match is None:
        return None
    return match.group(2), match.group(3).strip()


def ts_to_seconds(ts: str) -> float:
    """Convert a `HH:MM:SS.mmm` timestamp to seconds."""
    hours, minutes, seconds = ts.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
