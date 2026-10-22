"""Tests for shept.models — catalogue, downloader and installed detection.

These must run without network access and without PyGObject: `urlopen` is
replaced by a fake that serves a small payload in chunks.
"""

from __future__ import annotations

import errno
import io
import socket
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

from shept import models
from shept.models import (
    CATALOG,
    NO_NETWORK_REASON,
    NO_RESPONSE_REASON,
    NO_SPACE_REASON,
    SIZE_TOLERANCE,
    VAD_SPEC,
    DownloadCancelled,
    DownloadError,
    ModelSpec,
    VadState,
    describe_download_error,
    download,
    download_dir,
    download_error_tooltip,
    human_size,
    is_downloaded,
    vad_state_text,
)

PAYLOAD = bytes(range(256)) * 3


class FakeResponse:
    """Minimal stand-in for the object `urllib.request.urlopen` returns."""

    def __init__(
        self,
        payload: bytes = PAYLOAD,
        content_length: bool = True,
        fail_after: int | None = None,
    ) -> None:
        self._stream = io.BytesIO(payload)
        self.headers = {"Content-Length": str(len(payload))} if content_length else {}
        self.chunks_read = 0
        self._fail_after = fail_after

    def read(self, size: int = -1) -> bytes:
        if self._fail_after is not None and self.chunks_read >= self._fail_after:
            raise OSError("connection reset")
        self.chunks_read += 1
        # Serve a small slice at a time so progress is observed incrementally.
        return self._stream.read(min(size, 100) if size >= 0 else 100)

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_exc) -> bool:
        return False


SPEC = ModelSpec(
    name="test",
    filename="ggml-test.bin",
    size_bytes=len(PAYLOAD),
    url="https://example.invalid/ggml-test.bin",
    note="test model",
)


@pytest.fixture
def dest(tmp_path):
    return tmp_path / "models"


# ------------------------------------------------------------------- catalogue


def test_catalog_has_the_expected_models():
    names = [spec.name for spec in CATALOG]

    for wanted in (
        "large-v3-turbo",
        "large-v3-turbo-q5_0",
        "large-v3",
        "medium",
        "small",
        "base",
        "tiny",
    ):
        assert wanted in names


def test_catalog_filenames_and_urls_are_consistent():
    for spec in CATALOG:
        assert spec.filename.startswith("ggml-")
        assert spec.filename.endswith(".bin")
        assert spec.filename == f"ggml-{spec.name}.bin"
        assert spec.url.startswith("https://")
        assert spec.url.endswith(spec.filename)


def test_catalog_sizes_and_notes_are_plausible():
    for spec in CATALOG:
        assert spec.size_bytes > 0
        assert spec.note.strip()


def test_catalog_has_no_duplicates():
    names = [spec.name for spec in CATALOG]
    filenames = [spec.filename for spec in CATALOG]
    urls = [spec.url for spec in CATALOG]

    assert len(names) == len(set(names))
    assert len(filenames) == len(set(filenames))
    assert len(urls) == len(set(urls))


def test_catalog_orders_large_v3_turbo_first():
    assert CATALOG[0].name == "large-v3-turbo"


def test_catalog_sizes_match_the_published_files():
    by_name = {spec.name: spec.size_bytes for spec in CATALOG}

    # ~1.62 GB and ~0.57 GB, the two sizes the README quotes.
    assert 1_500_000_000 < by_name["large-v3-turbo"] < 1_800_000_000
    assert 500_000_000 < by_name["large-v3-turbo-q5_0"] < 700_000_000
    assert by_name["tiny"] < by_name["base"] < by_name["small"] < by_name["medium"]


def test_human_size_uses_decimal_units():
    assert human_size(1_624_555_275) == "1.62 GB"
    assert human_size(574_041_195) == "574 MB"
    assert human_size(77_691_713) == "78 MB"
    assert human_size(999) == "1 kB"


# ---------------------------------------------------------------- download dir


def test_download_dir_honours_xdg_data_home(tmp_path, monkeypatch):
    data = tmp_path / "data"
    monkeypatch.setenv("XDG_DATA_HOME", str(data))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    directory = download_dir()

    assert directory == data / "shept" / "models"
    assert directory.is_dir()


def test_download_dir_falls_back_to_home_local_share(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setenv("HOME", str(home))

    directory = download_dir()

    assert directory == home / ".local" / "share" / "shept" / "models"
    assert directory.is_dir()


def test_download_dir_creates_nested_directories(tmp_path, monkeypatch):
    data = tmp_path / "deep" / "nested" / "data"
    monkeypatch.setenv("XDG_DATA_HOME", str(data))

    assert download_dir().is_dir()


def test_download_dir_removes_stale_part_files(tmp_path, monkeypatch):
    data = tmp_path / "data"
    monkeypatch.setenv("XDG_DATA_HOME", str(data))
    directory = data / "shept" / "models"
    directory.mkdir(parents=True)
    stale = directory / "ggml-small.bin.part"
    stale.write_bytes(b"half a model")
    keep = directory / "ggml-small.bin"
    keep.write_bytes(b"complete")

    download_dir()

    assert not stale.exists()
    assert keep.exists()


def test_download_dir_survives_a_part_file_it_cannot_remove(tmp_path, monkeypatch):
    data = tmp_path / "data"
    monkeypatch.setenv("XDG_DATA_HOME", str(data))
    directory = data / "shept" / "models"
    directory.mkdir(parents=True)
    (directory / "ggml-small.bin.part").write_bytes(b"half")

    def boom(self, *_args, **_kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "unlink", boom)

    assert download_dir() == directory


# -------------------------------------------------------------------- download


def test_download_writes_file_and_reports_progress(monkeypatch, dest):
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(models, "urlopen", lambda request, timeout=30: FakeResponse())

    path = download(
        SPEC,
        dest,
        progress=lambda received, total: calls.append((received, total)),
        is_cancelled=lambda: False,
    )

    assert path == dest / SPEC.filename
    assert path.read_bytes() == PAYLOAD

    received = [value for value, _total in calls]
    assert received[0] == 0
    assert received == sorted(received)
    assert received[-1] == len(PAYLOAD)
    assert all(total == len(PAYLOAD) for _value, total in calls)


def test_download_sends_a_user_agent(monkeypatch, dest):
    seen: list[str | None] = []

    def fake_urlopen(request, timeout=30):
        seen.append(request.get_header("User-agent"))
        return FakeResponse()

    monkeypatch.setattr(models, "urlopen", fake_urlopen)

    download(SPEC, dest, is_cancelled=lambda: False)

    assert seen == [models.USER_AGENT]


def test_download_without_content_length_still_reports_bytes(monkeypatch, dest):
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(
        models,
        "urlopen",
        lambda request, timeout=30: FakeResponse(content_length=False),
    )

    path = download(
        SPEC,
        dest,
        progress=lambda received, total: calls.append((received, total)),
        is_cancelled=lambda: False,
    )

    assert path.read_bytes() == PAYLOAD
    assert all(total == 0 for _received, total in calls)
    assert calls[-1][0] == len(PAYLOAD)


def test_download_leaves_no_part_file_on_success(monkeypatch, dest):
    monkeypatch.setattr(models, "urlopen", lambda request, timeout=30: FakeResponse())

    download(SPEC, dest, is_cancelled=lambda: False)

    assert sorted(p.name for p in dest.iterdir()) == [SPEC.filename]


def test_download_removes_part_file_when_the_rename_fails(monkeypatch, dest):
    monkeypatch.setattr(models, "urlopen", lambda request, timeout=30: FakeResponse())

    def failing_replace(_src, _dst):
        raise OSError("Read-only file system")

    monkeypatch.setattr(models.os, "replace", failing_replace)

    with pytest.raises(OSError, match="Read-only"):
        download(SPEC, dest, is_cancelled=lambda: False)

    assert list(dest.iterdir()) == []


def test_download_creates_destination_directory(monkeypatch, tmp_path):
    nested = tmp_path / "does" / "not" / "exist"
    monkeypatch.setattr(models, "urlopen", lambda request, timeout=30: FakeResponse())

    path = download(SPEC, nested, is_cancelled=lambda: False)

    assert path.parent == nested
    assert path.exists()


def test_download_removes_part_file_on_network_error(monkeypatch, dest):
    def failing(request, timeout=30):
        raise OSError("network is unreachable")

    monkeypatch.setattr(models, "urlopen", failing)

    with pytest.raises(DownloadError, match="network is unreachable"):
        download(SPEC, dest, is_cancelled=lambda: False)

    assert list(dest.iterdir()) == []


def test_download_removes_part_file_on_mid_stream_error(monkeypatch, dest):
    monkeypatch.setattr(
        models,
        "urlopen",
        lambda request, timeout=30: FakeResponse(fail_after=1),
    )

    with pytest.raises(DownloadError):
        download(SPEC, dest, is_cancelled=lambda: False)

    assert list(dest.iterdir()) == []


def test_download_detects_truncated_transfer(monkeypatch, dest):
    def fake(request, timeout=30):
        response = FakeResponse()
        response.headers = {"Content-Length": str(len(PAYLOAD) + 1000)}
        return response

    monkeypatch.setattr(models, "urlopen", fake)

    with pytest.raises(DownloadError, match="truncated"):
        download(SPEC, dest, is_cancelled=lambda: False)

    assert list(dest.iterdir()) == []


def test_download_cancel_stops_early_and_leaves_no_files(monkeypatch, dest):
    response = FakeResponse()
    monkeypatch.setattr(models, "urlopen", lambda request, timeout=30: response)
    checks = {"count": 0}

    def cancelled() -> bool:
        checks["count"] += 1
        # False for the pre-flight check and the first chunk, then True.
        return checks["count"] > 2

    with pytest.raises(DownloadCancelled):
        download(SPEC, dest, is_cancelled=cancelled)

    assert list(dest.iterdir()) == []
    assert response.chunks_read < 6


def test_download_cancelled_before_starting_never_touches_the_network(
    monkeypatch, dest
):
    def unexpected(*_args, **_kwargs):
        raise AssertionError("urlopen must not be called")

    monkeypatch.setattr(models, "urlopen", unexpected)

    with pytest.raises(DownloadCancelled):
        download(SPEC, dest, is_cancelled=lambda: True)

    assert list(dest.iterdir()) == []


def test_download_without_callbacks_works(monkeypatch, dest):
    monkeypatch.setattr(models, "urlopen", lambda request, timeout=30: FakeResponse())

    assert download(SPEC, dest).read_bytes() == PAYLOAD


# --------------------------------------------------------------- is_downloaded


def test_is_downloaded_true_for_exact_size(dest):
    dest.mkdir()
    (dest / SPEC.filename).write_bytes(PAYLOAD)

    assert is_downloaded(SPEC, dest) is True


def test_is_downloaded_false_when_missing(dest):
    assert is_downloaded(SPEC, dest) is False


def test_is_downloaded_tolerates_small_size_differences(dest):
    dest.mkdir()
    path = dest / SPEC.filename
    tolerance = models.size_tolerance(SPEC.size_bytes)
    path.write_bytes(b"x" * (SPEC.size_bytes + tolerance))

    assert is_downloaded(SPEC, dest) is True

    path.write_bytes(b"x" * (SPEC.size_bytes + tolerance + 1))

    assert is_downloaded(SPEC, dest) is False


def test_size_tolerance_is_capped():
    assert models.size_tolerance(1_624_555_275) == SIZE_TOLERANCE
    assert models.size_tolerance(1000) == 10
    assert models.size_tolerance(0) == 1


def test_is_downloaded_false_for_truncated_file(dest):
    dest.mkdir()
    (dest / SPEC.filename).write_bytes(b"x" * 10)

    assert is_downloaded(SPEC, dest) is False


def test_is_downloaded_ignores_part_files(dest):
    dest.mkdir()
    (dest / (SPEC.filename + ".part")).write_bytes(PAYLOAD)

    assert is_downloaded(SPEC, dest) is False


def test_is_complete_vad_accepts_the_published_size(tmp_path):
    vad = tmp_path / VAD_SPEC.filename
    vad.write_bytes(b"")
    with vad.open("rb+") as handle:
        handle.truncate(VAD_SPEC.size_bytes)

    assert models.is_complete_vad(vad) is True


def test_is_complete_vad_rejects_a_truncated_file(tmp_path):
    vad = tmp_path / VAD_SPEC.filename
    vad.write_bytes(b"not really a model")

    assert models.is_complete_vad(vad) is False


def test_is_complete_vad_rejects_a_missing_file(tmp_path):
    assert models.is_complete_vad(tmp_path / VAD_SPEC.filename) is False


# ------------------------------------------------ friendly download failures


def _dns_failure() -> socket.gaierror:
    return socket.gaierror(-3, "Temporary failure in name resolution")


def test_describe_download_error_name_resolution_keeps_the_row_short():
    assert describe_download_error(_dns_failure()) == NO_NETWORK_REASON


def test_describe_download_error_url_error_wrapping_name_resolution():
    exc = URLError(_dns_failure())

    assert describe_download_error(exc) == NO_NETWORK_REASON


def test_describe_download_error_unwraps_the_downloader_error():
    """The app sees a `DownloadError`; the useful exception is its cause."""
    exc = DownloadError(f"Could not download silero-vad: {URLError(_dns_failure())}")
    exc.__cause__ = URLError(_dns_failure())

    assert describe_download_error(exc) == NO_NETWORK_REASON


def test_describe_download_error_never_returns_the_raw_message():
    result = describe_download_error(URLError(_dns_failure()))

    assert "name resolution" not in result
    assert "Errno" not in result


def test_describe_download_error_timeout():
    assert describe_download_error(URLError(TimeoutError("timed out"))) == (
        NO_RESPONSE_REASON
    )


def test_describe_download_error_timeout_without_a_reason_object():
    assert describe_download_error(URLError("timed out")) == NO_RESPONSE_REASON


def test_describe_download_error_http_status():
    exc = HTTPError("https://example.invalid/model.bin", 404, "Not Found", {}, None)

    assert describe_download_error(exc) == "the server returned HTTP 404"


def test_describe_download_error_http_status_is_unwrapped():
    exc = DownloadError("Could not download silero-vad: HTTP Error 503")
    exc.__cause__ = HTTPError(
        "https://example.invalid/model.bin", 503, "Service Unavailable", {}, None
    )

    assert describe_download_error(exc) == "the server returned HTTP 503"


def test_describe_download_error_no_disk_space():
    exc = OSError(errno.ENOSPC, "No space left on device")

    assert describe_download_error(exc) == NO_SPACE_REASON


def test_describe_download_error_no_disk_space_is_unwrapped():
    exc = DownloadError("Could not download silero-vad: No space left on device")
    exc.__cause__ = OSError(errno.ENOSPC, "No space left on device")

    assert describe_download_error(exc) == NO_SPACE_REASON


def test_describe_download_error_falls_back_to_the_class_name():
    assert describe_download_error(ValueError("boom")) == "ValueError"
    assert describe_download_error(DownloadError("truncated")) == "DownloadError"


def test_describe_download_error_fallback_names_the_cause_class():
    exc = DownloadError("Could not download silero-vad: boom")
    exc.__cause__ = ValueError("boom")

    assert describe_download_error(exc) == "ValueError"


def test_download_error_tooltip_keeps_the_raw_message_and_the_sandbox_hint():
    tooltip = download_error_tooltip(URLError(_dns_failure()))

    assert "name resolution" in tooltip
    assert "Flatpak" in tooltip
    assert "--share=network" in tooltip


def test_download_error_tooltip_has_no_sandbox_hint_for_other_failures():
    tooltip = download_error_tooltip(OSError(errno.ENOSPC, "No space left on device"))

    assert "No space left on device" in tooltip
    assert "Flatpak" not in tooltip


# -------------------------------------------------------- VAD hint row states


def test_vad_state_text_missing():
    assert vad_state_text(VadState.MISSING) == (
        "Silero VAD is not installed",
        "Download it (885 kB) to skip silence.",
        "Download",
    )


def test_vad_state_text_missing_uses_the_given_size():
    assert vad_state_text(VadState.MISSING, size_bytes=1_000_000)[1] == (
        "Download it (1 MB) to skip silence."
    )


def test_vad_state_text_downloading_percentage():
    assert vad_state_text(VadState.DOWNLOADING, "42%") == (
        "Silero VAD is not installed",
        "Downloading Silero VAD — 42%",
        "Downloading…",
    )


def test_vad_state_text_downloading_bytes_without_a_content_length():
    assert vad_state_text(VadState.DOWNLOADING, "512 kB")[1] == (
        "Downloading Silero VAD — 512 kB"
    )


def test_vad_state_text_downloading_before_the_first_tick():
    assert vad_state_text(VadState.DOWNLOADING)[1] == (
        f"Downloading Silero VAD ({human_size(VAD_SPEC.size_bytes)})…"
    )


def test_vad_state_text_failed():
    assert vad_state_text(VadState.FAILED, NO_NETWORK_REASON) == (
        "Silero VAD is not installed",
        "Download failed — no network connection.",
        "Retry",
    )


def test_vad_state_text_failed_without_a_reason():
    assert vad_state_text(VadState.FAILED)[1] == "Download failed."


def test_vad_state_text_installed():
    assert vad_state_text(VadState.INSTALLED) == (
        "Silero VAD is installed",
        "",
        "",
    )
