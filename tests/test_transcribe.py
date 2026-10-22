"""Tests for skryba.transcribe — argv building and whisper/ffmpeg parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from skryba.transcribe import (
    build_ffmpeg_cmd,
    build_whisper_cmd,
    parse_progress_line,
    parse_segment,
    ts_to_seconds,
)


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
    ("line", "expected"),
    [
        (
            "[00:00:01.000 --> 00:00:03.000] Hello there",
            ("00:00:03.000", "Hello there"),
        ),
        (
            "[00:00:01.000 --> 00:00:03.000]   spaced out  ",
            ("00:00:03.000", "spaced out"),
        ),
        ("[00:00:00.000 --> 00:00:00.500]   ", ("00:00:00.500", "")),
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
    assert parse_segment("[00:00:01.000 --> 00:00:02.000] hi\n") == (
        "00:00:02.000",
        "hi",
    )


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


def test_build_ffmpeg_cmd_argv():
    input_path = Path("/tmp/in.mp3")
    wav_path = Path("/tmp/out.wav")

    assert build_ffmpeg_cmd(input_path, wav_path) == [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
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


def test_build_ffmpeg_cmd_keeps_paths_with_spaces_intact():
    input_path = Path("/tmp/my recordings/lecture 1.m4a")
    wav_path = Path("/tmp/skryba /input.wav")

    cmd = build_ffmpeg_cmd(input_path, wav_path)

    assert str(input_path) in cmd
    assert str(wav_path) in cmd
    assert cmd[cmd.index("-i") + 1] == "/tmp/my recordings/lecture 1.m4a"


def test_build_whisper_cmd_argv():
    model = Path("/models/ggml-large-v3-turbo.bin")
    wav_path = Path("/tmp/input.wav")

    assert build_whisper_cmd(model, wav_path, "pl", 8, False) == [
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


def test_build_whisper_cmd_translate_flag():
    model = Path("/models/ggml-base.bin")
    wav_path = Path("/tmp/input.wav")

    cmd = build_whisper_cmd(model, wav_path, "de", 4, True)

    assert cmd[-1] == "--translate"
    assert cmd.count("--translate") == 1
    assert build_whisper_cmd(model, wav_path, "de", 4, False).count("--translate") == 0


@pytest.mark.parametrize("threads", [1, 4, 8])
def test_build_whisper_cmd_renders_threads_as_string(threads):
    cmd = build_whisper_cmd(
        Path("/m/ggml-base.bin"), Path("/tmp/i.wav"), "en", threads, False
    )

    assert cmd[cmd.index("-t") + 1] == str(threads)
    assert isinstance(cmd[cmd.index("-t") + 1], str)


def test_build_whisper_cmd_keeps_paths_with_spaces_intact():
    model = Path("/models/my models/ggml-base.bin")
    wav_path = Path("/tmp/skryba work/input.wav")

    cmd = build_whisper_cmd(model, wav_path, "en", 8, False)

    assert cmd[cmd.index("-m") + 1] == str(model)
    assert cmd[cmd.index("-f") + 1] == str(wav_path)
