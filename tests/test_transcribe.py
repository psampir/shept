"""Tests for shept.transcribe: argv building, parsing and export formats."""

from __future__ import annotations

import errno
import json
import signal
import subprocess
import sys
import time
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from shept.transcribe import (
    EXPORT_LABELS,
    EXPORTERS,
    ChildFailure,
    DiagnosticTail,
    Segment,
    build_ffmpeg_cmd,
    build_whisper_cmd,
    format_json,
    format_lrc,
    format_srt,
    format_txt,
    format_vtt,
    ms_to_ts,
    parse_progress_line,
    parse_segment,
    probe_duration,
    read_tail,
    stop_process,
    summarize_error,
    ts_to_ms,
    ts_to_seconds,
    write_text_atomic,
)

SEGMENTS = [
    Segment(start_ms=0, end_ms=1_500, text="Dzień dobry"),
    Segment(start_ms=1_500, end_ms=59_999, text="Zażółć gęślą jaźń"),
    Segment(start_ms=3_599_999, end_ms=3_600_500, text="over an hour"),
]

# ------------------------------------------------------------------ timestamps


@pytest.mark.parametrize(
    ("timestamp", "expected"),
    [
        ("00:00:00.000", 0.0),
        ("00:00:01.500", 1.5),
        ("01:02:03.500", 3723.5),
        ("02:00:00.000", 7200.0),
        ("10:30:45.250", 37845.25),
    ],
)
def test_ts_to_seconds(timestamp, expected):
    assert ts_to_seconds(timestamp) == expected


@pytest.mark.parametrize(
    ("timestamp", "expected"),
    [
        ("00:00:00.000", 0),
        ("00:00:01.500", 1500),
        ("01:02:03.500", 3_723_500),
        ("02:00:00.000", 7_200_000),
        ("10:30:45.250", 37_845_250),
        # 59.999 s must not drift to 59998 through float multiplication.
        ("00:00:59.999", 59_999),
        ("00:59:59.999", 3_599_999),
    ],
)
def test_ts_to_ms(timestamp, expected):
    assert ts_to_ms(timestamp) == expected


# ---------------------------------------------------------------------- parser


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        (
            "[00:00:01.000 --> 00:00:03.000] Hello there",
            Segment(start_ms=1000, end_ms=3000, text="Hello there"),
        ),
        (
            "[00:00:01.000 --> 00:00:03.000]   spaced out  ",
            Segment(start_ms=1000, end_ms=3000, text="spaced out"),
        ),
        (
            "[00:00:00.000 --> 00:00:00.500]   ",
            Segment(start_ms=0, end_ms=500, text=""),
        ),
        (
            "[01:02:03.500 --> 02:03:04.250] multi hour",
            Segment(start_ms=3_723_500, end_ms=7_384_250, text="multi hour"),
        ),
        # An empty segment (silence) keeps both timestamps and empty text.
        (
            "[00:10:00.000 --> 00:10:00.000]",
            None,
        ),
        ("[00:00:00.500 --> 00:00:01.25] nope", None),
        ("[00:00:00.500] no arrow", None),
        ("no timestamp at all", None),
        (" [00:00:00.000 --> 00:00:01.000] indented", None),
        ("", None),
    ],
)
def test_parse_segment(line, expected):
    assert parse_segment(line) == expected


def test_parse_segment_accepts_trailing_newline():
    # whisper-cli writes segments one per line; the app strips "\n" before
    # parsing, but the regex must also tolerate a newline-terminated line.
    assert parse_segment("[00:00:01.000 --> 00:00:02.000] hi\n") == Segment(
        start_ms=1000, end_ms=2000, text="hi"
    )


def test_parse_segment_returns_immutable_segment():
    segment = parse_segment("[00:00:01.000 --> 00:00:02.000] hi")

    assert segment is not None
    with pytest.raises(FrozenInstanceError):
        segment.text = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("out_time_ms=1500000\n", 1.5),
        ("out_time_ms=0", 0.0),
        ("out_time_ms=-500000", -0.5),
        ("out_time_ms=abc", None),
        ("out_time_ms=", None),
        ("frame=42\n", None),
        ("progress=continue\n", None),
        ("", None),
    ],
)
def test_parse_progress_line(line, expected):
    assert parse_progress_line(line) == expected


# ------------------------------------------------------------------- formatters


def test_format_txt_is_plain_lines_with_trailing_newline():
    assert format_txt(SEGMENTS) == ("Dzień dobry\nZażółć gęślą jaźń\nover an hour\n")


def test_format_txt_has_no_timestamps():
    output = format_txt(SEGMENTS)

    assert "-->" not in output
    assert "00:00" not in output


def test_format_srt_exact_output():
    assert format_srt(SEGMENTS) == (
        "1\n"
        "00:00:00,000 --> 00:00:01,500\n"
        "Dzień dobry\n"
        "\n"
        "2\n"
        "00:00:01,500 --> 00:00:59,999\n"
        "Zażółć gęślą jaźń\n"
        "\n"
        "3\n"
        "00:59:59,999 --> 01:00:00,500\n"
        "over an hour\n"
        "\n"
    )


def test_format_srt_numbers_from_one():
    output = format_srt(SEGMENTS)

    assert output.startswith("1\n")
    assert "\n2\n" in output
    assert "\n3\n" in output
    assert "\n4\n" not in output


def test_format_vtt_exact_output():
    assert format_vtt(SEGMENTS) == (
        "WEBVTT\n"
        "\n"
        "00:00:00.000 --> 00:00:01.500\n"
        "Dzień dobry\n"
        "\n"
        "00:00:01.500 --> 00:00:59.999\n"
        "Zażółć gęślą jaźń\n"
        "\n"
        "00:59:59.999 --> 01:00:00.500\n"
        "over an hour\n"
        "\n"
    )


def test_format_vtt_starts_with_header():
    output = format_vtt([])

    assert output == "WEBVTT\n\n"
    assert not format_vtt(SEGMENTS).startswith("WEBVTT\nWEBVTT")


def test_format_lrc_exact_output():
    assert format_lrc(SEGMENTS) == (
        "[00:00.00]Dzień dobry\n[00:01.50]Zażółć gęślą jaźń\n[59:59.99]over an hour\n"
    )


def test_format_lrc_minutes_do_not_wrap_at_sixty():
    segments = [Segment(start_ms=3_723_500, end_ms=3_724_000, text="late")]

    assert format_lrc(segments) == "[62:03.50]late\n"


def test_format_json_exact_output():
    assert format_json(SEGMENTS) == (
        "{\n"
        '  "version": 1,\n'
        '  "segments": [\n'
        "    {\n"
        '      "start_ms": 0,\n'
        '      "end_ms": 1500,\n'
        '      "text": "Dzień dobry"\n'
        "    },\n"
        "    {\n"
        '      "start_ms": 1500,\n'
        '      "end_ms": 59999,\n'
        '      "text": "Zażółć gęślą jaźń"\n'
        "    },\n"
        "    {\n"
        '      "start_ms": 3599999,\n'
        '      "end_ms": 3600500,\n'
        '      "text": "over an hour"\n'
        "    }\n"
        "  ]\n"
        "}\n"
    )


def test_format_json_is_valid_and_keeps_unicode():
    payload = json.loads(format_json(SEGMENTS))

    assert payload["version"] == 1
    assert payload["segments"][0]["text"] == "Dzień dobry"
    assert "\\u" not in format_json(SEGMENTS)


@pytest.mark.parametrize(
    ("formatter", "empty"),
    [
        (format_txt, ""),
        (format_srt, ""),
        (format_vtt, "WEBVTT\n\n"),
        (format_lrc, ""),
        (format_json, '{\n  "version": 1,\n  "segments": []\n}\n'),
    ],
)
def test_formatters_handle_no_segments(formatter, empty):
    assert formatter([]) == empty


@pytest.mark.parametrize(
    ("ms", "expected_dot", "expected_comma"),
    [
        (0, "00:00:00.000", "00:00:00,000"),
        (999, "00:00:00.999", "00:00:00,999"),
        (59_999, "00:00:59.999", "00:00:59,999"),
        (60_000, "00:01:00.000", "00:01:00,000"),
        (3_599_999, "00:59:59.999", "00:59:59,999"),
        (3_600_000, "01:00:00.000", "01:00:00,000"),
    ],
)
def test_ms_to_ts_pads_consistently(ms, expected_dot, expected_comma):
    assert ms_to_ts(ms) == expected_dot
    assert ms_to_ts(ms, ",") == expected_comma


def test_srt_and_vtt_match_padding_at_boundaries():
    segments = [
        Segment(start_ms=0, end_ms=59_999, text="start"),
        Segment(start_ms=3_600_000, end_ms=7_200_000, text="hour"),
    ]

    assert "00:00:00,000 --> 00:00:59,999" in format_srt(segments)
    assert "01:00:00,000 --> 02:00:00,000" in format_srt(segments)
    assert "00:00:00.000 --> 00:00:59.999" in format_vtt(segments)
    assert "01:00:00.000 --> 02:00:00.000" in format_vtt(segments)


# -------------------------------------------------------------------- exporters


def test_exporters_cover_exactly_five_extensions():
    assert set(EXPORTERS) == {"txt", "srt", "vtt", "lrc", "json"}
    assert len(EXPORTERS) == 5


def test_exporters_and_labels_agree():
    assert list(EXPORTERS) == list(EXPORT_LABELS)
    assert all(callable(func) for func in EXPORTERS.values())


def test_exporter_outputs_are_strings():
    for formatter in EXPORTERS.values():
        assert isinstance(formatter(SEGMENTS), str)


# ------------------------------------------------------------------------- argv


def test_build_ffmpeg_cmd_argv():
    input_path = Path("/tmp/in.mp3")
    wav_path = Path("/tmp/out.wav")

    assert build_ffmpeg_cmd(input_path, wav_path) == [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-nostats",
        "-progress",
        "pipe:1",
        "-i",
        "/tmp/in.mp3",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "/tmp/out.wav",
        "-y",
    ]


def test_build_ffmpeg_cmd_disables_the_controlling_terminal():
    """A keystroke in the launching terminal must not reach ffmpeg's stdin."""
    cmd = build_ffmpeg_cmd(Path("/tmp/in.mp3"), Path("/tmp/out.wav"))

    assert "-nostdin" in cmd


def test_build_ffmpeg_cmd_keeps_paths_with_spaces_intact():
    input_path = Path("/tmp/my recordings/lecture 1.m4a")
    wav_path = Path("/tmp/shept /input.wav")

    cmd = build_ffmpeg_cmd(input_path, wav_path)

    assert str(input_path) in cmd
    assert str(wav_path) in cmd
    assert cmd[cmd.index("-i") + 1] == "/tmp/my recordings/lecture 1.m4a"


def test_build_whisper_cmd_argv():
    model = Path("/models/ggml-large-v3-turbo.bin")
    wav_path = Path("/tmp/input.wav")

    assert build_whisper_cmd(model, wav_path, "pl", 8) == [
        "whisper-cli",
        "-m",
        "/models/ggml-large-v3-turbo.bin",
        "-l",
        "pl",
        "-t",
        "8",
        "-f",
        "/tmp/input.wav",
    ]


@pytest.mark.parametrize("threads", [1, 4, 8])
def test_build_whisper_cmd_renders_threads_as_string(threads):
    cmd = build_whisper_cmd(Path("/m/ggml-base.bin"), Path("/tmp/i.wav"), "en", threads)

    assert cmd[cmd.index("-t") + 1] == str(threads)
    assert isinstance(cmd[cmd.index("-t") + 1], str)


def test_build_whisper_cmd_keeps_paths_with_spaces_intact():
    model = Path("/models/my models/ggml-base.bin")
    wav_path = Path("/tmp/shept work/input.wav")

    cmd = build_whisper_cmd(model, wav_path, "en", 8)

    assert cmd[cmd.index("-m") + 1] == str(model)
    assert cmd[cmd.index("-f") + 1] == str(wav_path)


# ------------------------------------------------------------- process control


def test_read_tail_returns_the_last_bytes_of_a_file(tmp_path):
    log = tmp_path / "stderr.log"
    log.write_text("first problem\nlast problem\n", encoding="utf-8")

    assert read_tail(log) == "first problem\nlast problem"


def test_read_tail_caps_a_huge_log_at_the_end(tmp_path):
    log = tmp_path / "stderr.log"
    log.write_text("x" * 100_000 + "the real reason\n", encoding="utf-8")

    tail = read_tail(log, limit=64)

    assert tail.endswith("the real reason")
    assert len(tail) <= 64


def test_read_tail_of_a_missing_file_is_empty(tmp_path):
    assert read_tail(tmp_path / "gone.log") == ""


# ------------------------------------------------------------- duration probe


def _fake_ffprobe(tmp_path, body: str):
    """An executable stand-in for ffprobe, so the probe can be tested fast."""
    script = tmp_path / "fake-ffprobe"
    script.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    script.chmod(0o755)
    return script


def test_probe_duration_reads_the_printed_seconds(tmp_path):
    script = _fake_ffprobe(tmp_path, "echo 123.25")

    assert probe_duration(tmp_path / "media.mp3", binary=str(script)) == 123.25


def test_probe_duration_returns_zero_when_the_probe_hangs(tmp_path):
    """A hung mount must not wedge the job: the timeout wins and returns 0.0."""
    script = _fake_ffprobe(tmp_path, "sleep 30\necho 12.5")
    start = time.monotonic()

    duration = probe_duration(tmp_path / "media.mp3", timeout=0.2, binary=str(script))

    assert duration == 0.0
    assert time.monotonic() - start < 5


def test_probe_duration_returns_zero_for_unparseable_output(tmp_path):
    script = _fake_ffprobe(tmp_path, "echo N/A")

    assert probe_duration(tmp_path / "media.mp3", binary=str(script)) == 0.0


def test_probe_duration_returns_zero_when_ffprobe_fails(tmp_path):
    script = _fake_ffprobe(tmp_path, "exit 1")

    assert probe_duration(tmp_path / "media.mp3", binary=str(script)) == 0.0


def test_probe_duration_returns_zero_when_ffprobe_is_missing(tmp_path):
    missing = tmp_path / "no-such-ffprobe"

    assert probe_duration(tmp_path / "media.mp3", binary=str(missing)) == 0.0


def test_probe_duration_registers_and_releases_the_child(tmp_path):
    seen: list[object] = []
    script = _fake_ffprobe(tmp_path, "echo 1.5")

    probe_duration(
        tmp_path / "media.mp3",
        binary=str(script),
        register=seen.append,
    )

    assert len(seen) == 2
    assert isinstance(seen[0], subprocess.Popen)
    assert seen[1] is None


# ----------------------------------------------------------- diagnostics tail


def test_diagnostic_tail_keeps_only_the_last_lines():
    tail = DiagnosticTail(max_lines=3)

    for line in ("one", "two", "three", "four"):
        tail.add(line)

    assert tail.lines == ["two", "three", "four"]


def test_diagnostic_tail_ignores_blank_lines():
    tail = DiagnosticTail()

    tail.add("first")
    tail.add("   \n")
    tail.add("second")

    assert tail.lines == ["first", "second"]


def test_diagnostic_tail_summary_is_the_last_three_lines():
    tail = DiagnosticTail()
    for line in ("a", "b", "c", "d"):
        tail.add(line)

    assert tail.summary(3) == "b\nc\nd"


def test_diagnostic_tail_summary_of_nothing_is_empty():
    assert DiagnosticTail().summary() == ""


def test_child_failure_keeps_the_full_details():
    failure = ChildFailure("whisper-cli exited with 1", "error: failed to load model")

    assert str(failure) == "whisper-cli exited with 1"
    assert failure.details == "error: failed to load model"


# -------------------------------------------------------------- error summaries


def test_summarize_error_keeps_the_first_two_lines():
    text = "ffmpeg failed: Header missing\nSecond line\nThird line"

    assert summarize_error(text) == "ffmpeg failed: Header missing Second line"


def test_summarize_error_caps_very_long_text():
    summary = summarize_error("x" * 2_000)

    assert len(summary) <= 500
    assert summary.endswith("…")


def test_summarize_error_of_blank_text_is_empty():
    assert summarize_error("\n\n   \n") == ""


# ------------------------------------------------------------ atomic file write


def test_write_text_atomic_writes_the_file(tmp_path):
    target = tmp_path / "talk.txt"

    assert write_text_atomic(target, "Dzień dobry\n") == target
    assert target.read_text(encoding="utf-8") == "Dzień dobry\n"


def test_write_text_atomic_replaces_an_existing_file(tmp_path):
    target = tmp_path / "talk.txt"
    target.write_text("old\n", encoding="utf-8")

    write_text_atomic(target, "new\n")

    assert target.read_text(encoding="utf-8") == "new\n"


def test_write_text_atomic_leaves_no_temporary_file(tmp_path):
    write_text_atomic(tmp_path / "talk.txt", "text\n")

    assert sorted(path.name for path in tmp_path.iterdir()) == ["talk.txt"]


def test_write_text_atomic_failure_keeps_the_old_file(tmp_path, monkeypatch):
    target = tmp_path / "talk.txt"
    target.write_text("old\n", encoding="utf-8")

    def boom(_src, _dst):
        raise OSError("No space left on device")

    monkeypatch.setattr("shept.transcribe.os.replace", boom)

    with pytest.raises(OSError, match="No space left"):
        write_text_atomic(target, "new\n")

    assert target.read_text(encoding="utf-8") == "old\n"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["talk.txt"]


def test_write_text_atomic_failure_on_write_removes_the_temporary_file(
    tmp_path, monkeypatch
):
    def failing_write(self, *_args, **_kwargs):
        raise OSError(errno.ENAMETOOLONG, "File name too long")

    monkeypatch.setattr(Path, "write_text", failing_write)

    with pytest.raises(OSError, match="File name too long"):
        write_text_atomic(tmp_path / "talk.txt", "text\n")

    assert list(tmp_path.iterdir()) == []


def test_stop_process_terminates_a_running_child():
    proc = subprocess.Popen(["sleep", "30"])

    stop_process(proc)

    assert proc.poll() is not None


def _spawn_term_ignoring_child() -> subprocess.Popen:
    """A child that ignores SIGTERM and says so once the handler is installed."""
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import signal, sys, time; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "sys.stdout.write('ready'); sys.stdout.flush(); "
            "time.sleep(30)",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
    assert proc.stdout.read(5) == "ready"
    return proc


def test_stop_process_kills_a_child_that_ignores_sigterm():
    proc = _spawn_term_ignoring_child()

    stop_process(proc, timeout=0.2)

    assert proc.poll() is not None
    assert proc.returncode == -signal.SIGKILL


def test_stop_process_is_a_no_op_for_a_finished_child():
    proc = subprocess.Popen(["true"])
    proc.wait()

    stop_process(proc)

    assert proc.returncode == 0


def test_stop_process_accepts_none():
    stop_process(None)


def test_stop_process_does_not_wait_long_when_the_child_ignores_sigterm():
    proc = _spawn_term_ignoring_child()
    start = time.monotonic()

    stop_process(proc, timeout=0.2)

    assert time.monotonic() - start < 5


def test_build_whisper_cmd_without_vad_is_unchanged():
    cmd = build_whisper_cmd(Path("/m/ggml-base.bin"), Path("/tmp/i.wav"), "en", 4)

    assert "--vad" not in cmd
    assert "-vm" not in cmd


def test_build_whisper_cmd_adds_vad_flags():
    vad = Path("/models/ggml-silero-v6.2.0.bin")

    cmd = build_whisper_cmd(Path("/m/ggml-base.bin"), Path("/tmp/i.wav"), "en", 4, vad)

    assert cmd[-3:] == ["--vad", "-vm", str(vad)]
    assert cmd.count("--vad") == 1


def test_build_whisper_cmd_keeps_vad_paths_with_spaces_intact():
    vad = Path("/models/silero v6/ggml-silero-v6.2.0.bin")

    cmd = build_whisper_cmd(Path("/m/ggml-base.bin"), Path("/tmp/i.wav"), "en", 4, vad)

    assert cmd[cmd.index("-vm") + 1] == str(vad)
