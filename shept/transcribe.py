"""GTK-free helpers for running ffmpeg/whisper-cli and parsing their output.

This module also owns the transcript data model and the export formatters, so
that the parsing and every output format can be tested without PyGObject.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

LINE_RE = re.compile(
    r"^\[(\d\d:\d\d:\d\d\.\d{3})\s*-->\s*(\d\d:\d\d:\d\d\.\d{3})\]\s+(.*)$"
)

#: How long `stop_process` gives a child to honour SIGTERM before SIGKILL.
STOP_TIMEOUT = 2.0

#: After SIGKILL the child is gone almost immediately; this only bounds the
#: wait so a stuck kernel state cannot hang the UI thread.
KILL_TIMEOUT = 1.0


@dataclass(frozen=True)
class Segment:
    """One whisper-cli output segment, with millisecond timing."""

    start_ms: int
    end_ms: int
    text: str


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
    return cmd


def stop_process(
    proc: subprocess.Popen | None,
    timeout: float = STOP_TIMEOUT,
) -> None:
    """Stop `proc`: SIGTERM, then SIGKILL when it is still alive after `timeout`.

    Used by both Cancel and the window's close handler. Closing the window
    while a job runs used to leave whisper-cli (or ffmpeg) behind as an orphan
    still using the GPU; a cancelled job stopped its process only from the
    worker thread, which is a daemon and dies with the interpreter. Waiting
    here is bounded, so a close can never hang the main loop for long.

    A process that already exited, or no process at all, is a no-op.
    """
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass
    proc.kill()
    try:
        proc.wait(timeout=KILL_TIMEOUT)
    except subprocess.TimeoutExpired:
        pass


def parse_progress_line(line: str) -> float | None:
    """Return the decoded position in seconds from an ffmpeg `-progress` line."""
    stripped = line.strip()
    if not stripped.startswith("out_time_ms="):
        return None
    try:
        return int(stripped.split("=", 1)[1]) / 1_000_000.0
    except ValueError:
        return None


def parse_segment(line: str) -> Segment | None:
    """Return the `Segment` for a whisper-cli output line, else None."""
    match = LINE_RE.match(line)
    if match is None:
        return None
    return Segment(
        start_ms=ts_to_ms(match.group(1)),
        end_ms=ts_to_ms(match.group(2)),
        text=match.group(3).strip(),
    )


def ts_to_ms(ts: str) -> int:
    """Convert a `HH:MM:SS.mmm` timestamp to integer milliseconds.

    Parsed as text rather than through `float` so that values such as
    `00:00:59.999` stay exact instead of drifting to 59998 ms.
    """
    hours, minutes, rest = ts.split(":")
    seconds, _, millis = rest.partition(".")
    millis = (millis + "000")[:3]
    return (
        int(hours) * 3_600_000
        + int(minutes) * 60_000
        + int(seconds) * 1000
        + int(millis)
    )


def ts_to_seconds(ts: str) -> float:
    """Convert a `HH:MM:SS.mmm` timestamp to seconds."""
    return ts_to_ms(ts) / 1000.0


def ms_to_ts(ms: int, separator: str = ".") -> str:
    """Format milliseconds as `HH:MM:SS<sep>mmm` (srt passes a comma)."""
    hours, rest = divmod(max(0, ms), 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    seconds, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{millis:03d}"


def format_txt(segments: Sequence[Segment]) -> str:
    """Plain transcript: one line per segment, no timestamps."""
    return "".join(f"{segment.text}\n" for segment in segments)


def format_srt(segments: Sequence[Segment]) -> str:
    """SubRip subtitles, numbered from 1, with comma milliseconds."""
    blocks = [
        f"{number}\n"
        f"{ms_to_ts(segment.start_ms, ',')} --> "
        f"{ms_to_ts(segment.end_ms, ',')}\n"
        f"{segment.text}\n\n"
        for number, segment in enumerate(segments, start=1)
    ]
    return "".join(blocks)


def format_vtt(segments: Sequence[Segment]) -> str:
    """WebVTT subtitles, with dot milliseconds."""
    blocks = [
        f"{ms_to_ts(segment.start_ms)} --> {ms_to_ts(segment.end_ms)}\n"
        f"{segment.text}\n\n"
        for segment in segments
    ]
    return "WEBVTT\n\n" + "".join(blocks)


def format_lrc(segments: Sequence[Segment]) -> str:
    """LRC lyrics: `[mm:ss.xx]text`, minutes counting past 59."""
    lines = []
    for segment in segments:
        minutes, rest = divmod(max(0, segment.start_ms), 60_000)
        seconds, millis = divmod(rest, 1000)
        centis = millis // 10
        lines.append(f"[{minutes:02d}:{seconds:02d}.{centis:02d}]{segment.text}\n")
    return "".join(lines)


def format_json(segments: Sequence[Segment]) -> str:
    """Stable JSON export; the schema is versioned for future changes."""
    payload = {
        "version": 1,
        "segments": [
            {
                "start_ms": segment.start_ms,
                "end_ms": segment.end_ms,
                "text": segment.text,
            }
            for segment in segments
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


Formatters = dict[str, Callable[[Sequence[Segment]], str]]

#: Extension -> formatter. The extension is also the key the UI uses to pick
#: the format, so it never has to guess from the filename the user typed.
EXPORTERS: Formatters = {
    "txt": format_txt,
    "srt": format_srt,
    "vtt": format_vtt,
    "lrc": format_lrc,
    "json": format_json,
}

#: Extension -> human label, in the order the export menu should show them.
EXPORT_LABELS: dict[str, str] = {
    "txt": "Plain text",
    "srt": "Subtitles",
    "vtt": "WebVTT",
    "lrc": "Lyrics",
    "json": "JSON",
}
