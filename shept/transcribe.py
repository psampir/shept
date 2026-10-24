"""GTK-free helpers for running ffmpeg/whisper-cli and parsing their output.

This module also owns the transcript data model and the export formatters, so
that the parsing and every output format can be tested without PyGObject.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import subprocess
import threading
from collections import deque
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

#: Upper bound on the duration probe. A file on a hung FUSE or SMB mount can
#: make ffprobe block forever; without a timeout the job would never finish and
#: Cancel would have no handle to stop.
PROBE_TIMEOUT = 30.0

#: How much of a failed child's stderr is kept for the error message. A damaged
#: media file can make ffmpeg log hundreds of kilobytes; only the tail is ever
#: shown, and reading all of it would be pointless.
STDERR_TAIL_BYTES = 2048


@dataclass(frozen=True)
class Segment:
    """One whisper-cli output segment, with millisecond timing."""

    start_ms: int
    end_ms: int
    text: str


class ChildFailure(RuntimeError):
    """A child process failed; `details` keeps the output a row cannot show.

    `str(exc)` is the short message a queue row's subtitle has room for, while
    `details` carries the full diagnostics for its tooltip.
    """

    def __init__(self, message: str, details: str = "") -> None:
        super().__init__(message)
        self.details = details


#: How many non-segment output lines are kept to explain a failed run.
DIAGNOSTIC_LINES = 20


class DiagnosticTail:
    """Bounded ring buffer of a child's non-segment output lines.

    whisper-cli writes its progress and its real diagnostics to one stream, and
    a corrupt model, an out-of-memory kill or a bad `-vm` path all end in the
    same bare non-zero exit. Remembering the last few lines is what turns
    "whisper-cli exited with 1" into something the user can act on.
    """

    def __init__(self, max_lines: int = DIAGNOSTIC_LINES) -> None:
        self._lines: deque[str] = deque(maxlen=max_lines)

    def add(self, line: str) -> None:
        """Remember one output line unless it is blank."""
        text = line.strip()
        if text:
            self._lines.append(text)

    def summary(self, lines: int = 3) -> str:
        """The last `lines` remembered lines, oldest first, newline-joined."""
        return "\n".join(list(self._lines)[-lines:])

    @property
    def lines(self) -> list[str]:
        """Every remembered line, oldest first."""
        return list(self._lines)


def build_ffmpeg_cmd(input_path: Path, wav_path: Path) -> list[str]:
    """Build the ffmpeg argv that decodes `input_path` to 16 kHz mono WAV."""
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        # Without -nostdin a keystroke on the terminal that started Shept is
        # read as an ffmpeg command and can abort the decode.
        "-nostdin",
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
    vad_model: Path | None = None,
) -> list[str]:
    """Build the whisper-cli argv for transcribing `wav_path`.

    Passing `vad_model` turns on whisper.cpp's Silero VAD stage, which drops
    silent stretches before they reach the model, which is what stops whisper
    from inventing "thank you for watching" over trailing silence.
    """
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
    if vad_model is not None:
        cmd += ["--vad", "-vm", str(vad_model)]
    return cmd


def send_stop(proc: subprocess.Popen | None) -> bool:
    """Send SIGTERM to `proc` and return immediately; True when it was sent.

    This is the only part of stopping a child that is cheap enough for the GTK
    main thread: it asks the child to leave and returns. The grace period and
    the SIGKILL that may follow live in `escalate_stop` / `stop_in_background`,
    which must never run on that thread.

    A process that already exited, or no process at all, is a no-op.
    """
    if proc is None or proc.poll() is not None:
        return False
    try:
        proc.terminate()
    except OSError:
        # The child died between poll() and terminate(); nothing left to stop.
        return False
    return True


def escalate_stop(
    proc: subprocess.Popen | None,
    timeout: float = STOP_TIMEOUT,
) -> None:
    """Wait up to `timeout` for `proc` to exit, then SIGKILL it.

    Blocks, so it belongs on a worker thread: the UI calls `send_stop` and
    hands the escalation to `stop_in_background`. When the child honoured the
    SIGTERM this returns as soon as it is gone, so a well-behaved process
    costs a single `wait`.

    A process that already exited, or no process at all, is a no-op.
    """
    if proc is None or proc.poll() is not None:
        return
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


def stop_process(
    proc: subprocess.Popen | None,
    timeout: float = STOP_TIMEOUT,
) -> None:
    """Stop `proc` synchronously: SIGTERM, then SIGKILL after `timeout`.

    Kept for the callers that already run on a worker thread (the job worker,
    the probe timeout) and for compatibility. The GTK main thread must not use
    this: blocking it for the grace period is what made Cancel freeze the
    window, so Cancel and close use `send_stop` plus `stop_in_background`.
    """
    send_stop(proc)
    escalate_stop(proc, timeout)


#: Escalator threads that have not finished yet, so they can be joined at
#: interpreter exit. They are daemon threads: without this join an app closed
#: while a child ignores SIGTERM would exit before the SIGKILL and leave that
#: child running as an orphan, which is the bug the close handler exists for.
_PENDING_STOPS: set[threading.Thread] = set()
_PENDING_STOPS_LOCK = threading.Lock()


def stop_in_background(
    proc: subprocess.Popen | None,
    timeout: float = STOP_TIMEOUT,
) -> threading.Thread | None:
    """Escalate a SIGTERM already sent by `send_stop` on a worker thread.

    Returns the thread, or None when there was nothing to escalate. The
    caller sends the signal first (`send_stop`) so the child starts shutting
    down before the thread is even created; this function only owns the grace
    period and the SIGKILL. The thread never touches GTK.
    """
    if proc is None or proc.poll() is not None:
        return None
    thread = threading.Thread(
        target=_escalate_and_forget,
        args=(proc, timeout),
        name="shept-stop",
        daemon=True,
    )
    with _PENDING_STOPS_LOCK:
        _PENDING_STOPS.add(thread)
    thread.start()
    return thread


def _escalate_and_forget(proc: subprocess.Popen, timeout: float) -> None:
    """Body of the escalator thread: escalate, then stop being tracked."""
    try:
        escalate_stop(proc, timeout)
    finally:
        with _PENDING_STOPS_LOCK:
            _PENDING_STOPS.discard(threading.current_thread())


def join_pending_stops() -> None:
    """Wait for every escalator thread, bounded by its own grace periods."""
    with _PENDING_STOPS_LOCK:
        pending = list(_PENDING_STOPS)
    for thread in pending:
        thread.join(timeout=STOP_TIMEOUT + KILL_TIMEOUT)


atexit.register(join_pending_stops)


def read_tail(path: Path, limit: int = STDERR_TAIL_BYTES) -> str:
    """Return the last `limit` bytes of `path` as text, or "" when unreadable.

    Child diagnostics go to a temporary file rather than a pipe: an unread
    pipe fills up at 64 KiB and then blocks the child inside `write()`, which
    is exactly how a damaged file used to wedge a job forever. Only the tail
    is ever shown, so that is all that is read back.
    """
    try:
        with Path(path).open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max(0, limit)))
            data = handle.read(limit)
    except OSError:
        return ""
    return data.decode("utf-8", errors="replace").strip()


#: How much of a child's diagnostics a queue row shows. The subtitle holds one
#: line; the tooltip holds everything the error carries.
ERROR_SUMMARY_LINES = 2
ERROR_SUMMARY_CHARS = 500


def summarize_error(
    text: str,
    *,
    lines: int = ERROR_SUMMARY_LINES,
    limit: int = ERROR_SUMMARY_CHARS,
) -> str:
    """Shorten child diagnostics so they fit a queue row's subtitle.

    ffmpeg repeats one `Header missing` line per damaged frame, so the first
    lines are the useful part; the complete text stays in the row's tooltip.
    """
    kept = [line.strip() for line in text.splitlines() if line.strip()]
    if not kept:
        return ""
    summary = " ".join(kept[:lines])
    if len(summary) > limit:
        summary = summary[: max(1, limit - 1)].rstrip() + "…"
    return summary


def write_text_atomic(target: Path, text: str) -> Path:
    """Write `text` to `target` without leaving a partial file at that path.

    The text goes to `<name>.tmp` beside the target and is moved into place
    with `os.replace`, so a full disk or a vanished portal path cannot replace
    an existing transcript with half a file. Failures remove the temporary file
    and propagate the `OSError`; reporting it is the caller's job.
    """
    target = Path(target)
    tmp = target.with_name(target.name + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, target)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise
    return target


def build_ffprobe_cmd(media: Path, binary: str = "ffprobe") -> list[str]:
    """Build the ffprobe argv that prints `media`'s duration in seconds."""
    return [
        binary,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nw=1:nk=1",
        str(media),
    ]


def probe_duration(
    media: Path,
    *,
    timeout: float = PROBE_TIMEOUT,
    binary: str = "ffprobe",
    register: Callable[[subprocess.Popen[str] | None], None] | None = None,
) -> float:
    """Return `media`'s duration in seconds, or 0.0 when it cannot be known.

    The probe is bounded by `timeout`: a file on a hung network mount must not
    wedge the job. `register` receives the running `Popen` (and None once it is
    gone) so the window can stop the probe from Cancel or close, exactly like
    the ffmpeg and whisper-cli children.

    The contract is deliberately "a float, never an exception": callers use 0.0
    to mean "progress cannot be computed", and a missing or broken ffprobe must
    not fail a job that would otherwise transcribe fine.
    """
    try:
        proc = subprocess.Popen(
            build_ffprobe_cmd(media, binary),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        return 0.0
    if register is not None:
        register(proc)
    try:
        out = ""
        try:
            out, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            stop_process(proc)
            return 0.0
    finally:
        if register is not None:
            register(None)
    if proc.returncode != 0:
        return 0.0
    try:
        return float(out.strip())
    except ValueError:
        return 0.0


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
