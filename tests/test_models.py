"""Tests for shept.models: catalogue, downloader and installed detection.

These must run without network access and without PyGObject: `urlopen` is
replaced by a fake that serves a small payload in chunks.
"""

from __future__ import annotations

import errno
import io
import os
import socket
import threading
import time
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
        status: int = 200,
    ) -> None:
        self._stream = io.BytesIO(payload)
        self.headers = {"Content-Length": str(len(payload))} if content_length else {}
        self.status = status
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
    old = time.time() - models.STALE_PART_AGE - 60
    os.utime(stale, (old, old))
    keep = directory / "ggml-small.bin"
    keep.write_bytes(b"complete")

    download_dir()

    assert not stale.exists()
    assert keep.exists()


def test_download_dir_keeps_a_part_file_that_can_still_be_resumed(
    tmp_path, monkeypatch
):
    """A fresh `.part` is half a download, not litter: the next run resumes it."""
    data = tmp_path / "data"
    monkeypatch.setenv("XDG_DATA_HOME", str(data))
    directory = data / "shept" / "models"
    directory.mkdir(parents=True)
    fresh = directory / "ggml-large-v3-turbo.bin.part"
    fresh.write_bytes(b"half a model")

    download_dir()

    assert fresh.read_bytes() == b"half a model"


def test_download_dir_survives_a_part_file_it_cannot_remove(tmp_path, monkeypatch):
    data = tmp_path / "data"
    monkeypatch.setenv("XDG_DATA_HOME", str(data))
    directory = data / "shept" / "models"
    directory.mkdir(parents=True)
    stale = directory / "ggml-small.bin.part"
    stale.write_bytes(b"half")
    old = time.time() - models.STALE_PART_AGE - 60
    os.utime(stale, (old, old))

    def boom(self, *_args, **_kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "unlink", boom)

    assert download_dir() == directory


# -------------------------------------------------------------------- download


@pytest.fixture
def no_backoff(monkeypatch):
    """Take the real waits out of the retry loop; the delays are tested apart."""
    monkeypatch.setattr(models, "RETRY_DELAYS", (0.0, 0.0, 0.0))


class FakeServer:
    """A `urlopen` stand-in that answers `Range` requests like HuggingFace.

    `reads` records the `Range` header of every request, so a test can prove
    the transfer resumed instead of starting over.
    """

    def __init__(self, payload: bytes = PAYLOAD, honour_range: bool = True) -> None:
        self.payload = payload
        self.honour_range = honour_range
        self.ranges: list[str | None] = []

    def __call__(self, request, timeout=30) -> FakeResponse:
        header = request.get_header("Range")
        self.ranges.append(header)
        offset = 0
        if header and self.honour_range:
            offset = int(header.removeprefix("bytes=").rstrip("-"))
        body = self.payload[offset:]
        # A 206 carries the length of the remainder, a 200 the length of the
        # whole file; `download` must not confuse the two.
        response = FakeResponse(body, status=206 if offset else 200)
        response.headers = {"Content-Length": str(len(body))}
        return response


def test_download_writes_file_and_reports_progress(monkeypatch, dest):
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(models, "urlopen", FakeServer())

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
    monkeypatch.setattr(models, "urlopen", FakeServer())

    download(SPEC, dest, is_cancelled=lambda: False)

    assert sorted(p.name for p in dest.iterdir()) == [SPEC.filename]


# ------------------------------------------------------------------- resuming


def test_download_resumes_from_the_part_file_with_a_206(monkeypatch, dest):
    dest.mkdir()
    part = dest / (SPEC.filename + ".part")
    part.write_bytes(PAYLOAD[:300])
    server = FakeServer()
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(models, "urlopen", server)

    path = download(
        SPEC,
        dest,
        progress=lambda received, total: calls.append((received, total)),
        is_cancelled=lambda: False,
    )

    assert server.ranges == ["bytes=300-"]
    assert path.read_bytes() == PAYLOAD
    # The bar continues from the resumed offset instead of jumping back to 0.
    assert calls[0] == (300, len(PAYLOAD))
    assert calls[-1][0] == len(PAYLOAD)
    assert all(total == len(PAYLOAD) for _received, total in calls)


def test_download_starts_over_when_the_server_ignores_the_range(monkeypatch, dest):
    """A 200 answer means the whole body arrives: append, and the file doubles."""
    dest.mkdir()
    (dest / (SPEC.filename + ".part")).write_bytes(PAYLOAD[:300])
    server = FakeServer(honour_range=False)
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(models, "urlopen", server)

    path = download(
        SPEC,
        dest,
        progress=lambda received, total: calls.append((received, total)),
        is_cancelled=lambda: False,
    )

    assert server.ranges == ["bytes=300-"]
    assert path.read_bytes() == PAYLOAD
    assert calls[0] == (0, len(PAYLOAD))


def test_download_renames_a_complete_part_file_without_the_network(monkeypatch, dest):
    dest.mkdir()
    (dest / (SPEC.filename + ".part")).write_bytes(PAYLOAD)

    def unexpected(*_args, **_kwargs):
        raise AssertionError("a complete .part must not be downloaded again")

    monkeypatch.setattr(models, "urlopen", unexpected)

    path = download(SPEC, dest, is_cancelled=lambda: False)

    assert path.read_bytes() == PAYLOAD
    assert sorted(p.name for p in dest.iterdir()) == [SPEC.filename]


def test_download_discards_a_part_file_larger_than_the_published_size(
    monkeypatch, dest
):
    dest.mkdir()
    (dest / (SPEC.filename + ".part")).write_bytes(PAYLOAD + b"x" * 100)
    server = FakeServer()
    monkeypatch.setattr(models, "urlopen", server)

    path = download(SPEC, dest, is_cancelled=lambda: False)

    assert server.ranges == [None]
    assert path.read_bytes() == PAYLOAD


# -------------------------------------------------------------------- retrying


def test_download_retries_after_a_url_error(monkeypatch, dest, no_backoff):
    server = FakeServer()
    calls = {"count": 0}

    def flaky(request, timeout=30):
        calls["count"] += 1
        if calls["count"] == 1:
            raise URLError(socket.gaierror(-3, "Temporary failure in name resolution"))
        return server(request, timeout=timeout)

    monkeypatch.setattr(models, "urlopen", flaky)

    path = download(SPEC, dest, is_cancelled=lambda: False)

    assert calls["count"] == 2
    assert path.read_bytes() == PAYLOAD


def test_download_retries_after_a_server_error(monkeypatch, dest, no_backoff):
    server = FakeServer()
    calls = {"count": 0}

    def flaky(request, timeout=30):
        calls["count"] += 1
        if calls["count"] == 1:
            raise HTTPError(SPEC.url, 503, "Service Unavailable", {}, None)
        return server(request, timeout=timeout)

    monkeypatch.setattr(models, "urlopen", flaky)

    assert download(SPEC, dest, is_cancelled=lambda: False).read_bytes() == PAYLOAD
    assert calls["count"] == 2


def test_download_retries_a_transfer_that_stops_short(monkeypatch, dest, no_backoff):
    """A dropped connection is exactly the case resuming exists for."""
    server = FakeServer()
    truncated = {"sent": 0}

    def dropping(request, timeout=30):
        truncated["sent"] += 1
        if truncated["sent"] == 1:
            response = server(request, timeout=timeout)
            response.headers = {"Content-Length": str(len(PAYLOAD) + 500)}
            return response
        return server(request, timeout=timeout)

    monkeypatch.setattr(models, "urlopen", dropping)

    path = download(SPEC, dest, is_cancelled=lambda: False)

    assert truncated["sent"] == 2
    assert path.read_bytes() == PAYLOAD


def test_download_gives_up_after_three_retries(monkeypatch, dest, no_backoff):
    calls = {"count": 0}

    def failing(request, timeout=30):
        calls["count"] += 1
        raise URLError("connection refused")

    monkeypatch.setattr(models, "urlopen", failing)

    with pytest.raises(DownloadError, match="connection refused"):
        download(SPEC, dest, is_cancelled=lambda: False)

    assert calls["count"] == 1 + len(models.RETRY_DELAYS)


def test_retry_delays_back_off():
    assert models.RETRY_DELAYS == (1.0, 2.0, 4.0)


def test_download_never_retries_a_404(monkeypatch, dest, no_backoff):
    calls = {"count": 0}

    def missing(request, timeout=30):
        calls["count"] += 1
        raise HTTPError(SPEC.url, 404, "Not Found", {}, None)

    monkeypatch.setattr(models, "urlopen", missing)

    with pytest.raises(DownloadError, match="404"):
        download(SPEC, dest, is_cancelled=lambda: False)

    assert calls["count"] == 1


def test_download_cancel_during_the_backoff_wait_is_quick(monkeypatch, dest):
    """Cancel must not wait out a 5 s backoff before the thread notices."""
    monkeypatch.setattr(models, "RETRY_DELAYS", (5.0,))
    cancel = threading.Event()
    calls = {"count": 0}

    def failing(request, timeout=30):
        calls["count"] += 1
        # The user hits Cancel while the retry is being scheduled.
        threading.Timer(0.15, cancel.set).start()
        raise URLError("connection refused")

    monkeypatch.setattr(models, "urlopen", failing)

    start = time.monotonic()
    with pytest.raises(DownloadCancelled):
        download(SPEC, dest, is_cancelled=cancel.is_set)
    elapsed = time.monotonic() - start

    assert calls["count"] == 1
    assert elapsed < 1.5, f"cancelling waited {elapsed:.2f}s for the backoff"


# ------------------------------------------------------------ other failures


def test_download_creates_destination_directory(monkeypatch, tmp_path):
    nested = tmp_path / "does" / "not" / "exist"
    monkeypatch.setattr(models, "urlopen", FakeServer())

    path = download(SPEC, nested, is_cancelled=lambda: False)

    assert path.parent == nested
    assert path.exists()


def test_download_reports_a_local_error_without_retrying(monkeypatch, dest):
    calls = {"count": 0}

    def failing(request, timeout=30):
        calls["count"] += 1
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(models, "urlopen", failing)

    with pytest.raises(DownloadError, match="No space left on device"):
        download(SPEC, dest, is_cancelled=lambda: False)

    assert calls["count"] == 1


def test_download_keeps_the_part_file_on_a_mid_stream_error(monkeypatch, dest):
    monkeypatch.setattr(
        models,
        "urlopen",
        lambda request, timeout=30: FakeResponse(fail_after=1),
    )

    with pytest.raises(DownloadError):
        download(SPEC, dest, is_cancelled=lambda: False)

    part = dest / (SPEC.filename + ".part")
    assert part.stat().st_size > 0
    assert not (dest / SPEC.filename).exists()


def test_download_refuses_to_publish_a_file_that_is_short(monkeypatch, dest):
    """No Content-Length to compare against, so only the size check can tell."""
    short = PAYLOAD[:100]
    monkeypatch.setattr(models, "urlopen", FakeServer(payload=short))

    with pytest.raises(DownloadError, match="incomplete"):
        download(SPEC, dest, is_cancelled=lambda: False)

    assert not (dest / SPEC.filename).exists()
    assert (dest / (SPEC.filename + ".part")).read_bytes() == short


def test_download_removes_a_part_file_when_the_rename_fails(monkeypatch, dest):
    monkeypatch.setattr(models, "urlopen", FakeServer())

    def failing_replace(_src, _dst):
        raise OSError("Read-only file system")

    monkeypatch.setattr(models.os, "replace", failing_replace)

    with pytest.raises(OSError, match="Read-only"):
        download(SPEC, dest, is_cancelled=lambda: False)

    # The transfer was fine, only the rename was not: the bytes are kept, and
    # the next attempt renames them without asking the network again.
    assert (dest / (SPEC.filename + ".part")).read_bytes() == PAYLOAD


def test_download_cancel_stops_early_and_keeps_the_part_file(monkeypatch, dest):
    response = FakeResponse()
    monkeypatch.setattr(models, "urlopen", lambda request, timeout=30: response)
    checks = {"count": 0}

    def cancelled() -> bool:
        checks["count"] += 1
        # False for the pre-flight check and the first chunk, then True.
        return checks["count"] > 2

    with pytest.raises(DownloadCancelled):
        download(SPEC, dest, is_cancelled=cancelled)

    assert response.chunks_read < 6
    part = dest / (SPEC.filename + ".part")
    assert 0 < part.stat().st_size < len(PAYLOAD)
    assert not (dest / SPEC.filename).exists()


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
    monkeypatch.setattr(models, "urlopen", FakeServer())

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


# ---------------------------------------------- spotting an incomplete model


TINY = next(spec for spec in CATALOG if spec.name == "tiny")


def _sparse_model(path: Path, size: int) -> Path:
    """A file of `size` bytes that costs no disk: enough for a stat check."""
    path.write_bytes(b"")
    with path.open("rb+") as handle:
        handle.truncate(size)
    return path


def test_spec_for_path_finds_the_catalogue_entry(tmp_path):
    path = tmp_path / TINY.filename

    assert models.spec_for_path(path) is TINY


def test_spec_for_path_is_none_for_an_unpublished_model(tmp_path):
    assert models.spec_for_path(tmp_path / "ggml-borrowed.bin") is None


def test_is_complete_model_file_true_for_the_published_size(tmp_path):
    path = _sparse_model(tmp_path / TINY.filename, TINY.size_bytes)

    assert models.is_complete_model_file(path) is True


def test_is_complete_model_file_tolerates_a_small_difference(tmp_path):
    tolerance = models.size_tolerance(TINY.size_bytes)
    path = _sparse_model(tmp_path / TINY.filename, TINY.size_bytes - tolerance)

    assert models.is_complete_model_file(path) is True


def test_is_complete_model_file_rejects_a_file_one_byte_beyond_tolerance(tmp_path):
    tolerance = models.size_tolerance(TINY.size_bytes)
    path = _sparse_model(tmp_path / TINY.filename, TINY.size_bytes - tolerance - 1)

    assert models.is_complete_model_file(path) is False


def test_is_complete_model_file_is_unknown_for_an_unpublished_model(tmp_path):
    path = tmp_path / "ggml-borrowed.bin"
    path.write_bytes(b"whatever")

    assert models.is_complete_model_file(path) is None


def test_is_complete_model_file_rejects_a_zero_byte_file(tmp_path):
    path = tmp_path / TINY.filename
    path.write_bytes(b"")

    assert models.is_complete_model_file(path) is False


def test_is_complete_model_file_rejects_a_missing_file(tmp_path):
    assert models.is_complete_model_file(tmp_path / TINY.filename) is False


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


# --------------------------------------------------- deleting installed models

requires_non_root = pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root ignores directory permissions, so the check cannot fail",
)


def _touch(path: Path) -> Path:
    path.write_text("", encoding="utf-8")
    return path


def _writable(parent: Path) -> Path:
    directory = parent / "models"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _read_only(directory: Path) -> Path:
    """Make `directory` unwritable and remember to undo it for the cleanup."""
    directory.chmod(0o555)
    return directory


def test_is_deletable_accepts_a_file_in_a_writable_directory(tmp_path):
    directory = _writable(tmp_path)
    model = _touch(directory / "ggml-small.bin")

    assert models.is_deletable(model) is True


def test_is_deletable_rejects_a_missing_file(tmp_path):
    assert models.is_deletable(tmp_path / "ggml-small.bin") is False


@requires_non_root
def test_is_deletable_rejects_a_read_only_directory(tmp_path):
    directory = _writable(tmp_path)
    model = _touch(directory / "ggml-small.bin")
    _read_only(directory)
    try:
        assert models.is_deletable(model) is False
    finally:
        directory.chmod(0o755)


@requires_non_root
def test_is_deletable_accepts_a_symlink_in_a_writable_directory(tmp_path):
    """The link is what gets unlinked, so a read-only target must not matter."""
    target_dir = _writable(tmp_path / "elsewhere")
    target = _touch(target_dir / "ggml-small.bin")
    directory = _writable(tmp_path)
    link = directory / "ggml-small.bin"
    link.symlink_to(target)
    _read_only(target_dir)
    try:
        assert models.is_deletable(link) is True
    finally:
        target_dir.chmod(0o755)


def test_is_deletable_rejects_a_directory(tmp_path):
    directory = _writable(tmp_path) / "ggml-small.bin"
    directory.mkdir()

    assert models.is_deletable(directory) is False


def test_is_deletable_rejects_a_system_location(tmp_path, monkeypatch):
    """`/usr` is off limits even for a user who could technically write there."""
    model = _touch(tmp_path / "ggml-small.bin")
    monkeypatch.setattr(models, "READ_ONLY_PREFIXES", (tmp_path,))

    assert models.is_deletable(model) is False


def test_deletable_paths_keeps_only_the_deletable_ones(tmp_path):
    directory = _writable(tmp_path)
    good = _touch(directory / "ggml-small.bin")
    missing = directory / "ggml-gone.bin"

    assert models.deletable_paths([good, missing]) == {good}


def test_deletable_paths_of_nothing_is_empty():
    assert models.deletable_paths([]) == set()


def test_delete_model_removes_the_file(tmp_path):
    model = _touch(_writable(tmp_path) / "ggml-small.bin")

    models.delete_model(model)

    assert not model.exists()


def test_delete_model_unlinks_a_symlink_and_keeps_the_target(tmp_path):
    directory = _writable(tmp_path)
    target = tmp_path / "store" / "ggml-small.bin"
    target.parent.mkdir()
    _touch(target)
    link = directory / "ggml-small.bin"
    link.symlink_to(target)

    models.delete_model(link)

    assert not link.is_symlink()
    assert target.is_file()


def test_delete_model_raises_for_a_missing_file(tmp_path):
    with pytest.raises(OSError):
        models.delete_model(tmp_path / "ggml-small.bin")


def test_pick_model_after_delete_prefers_large_v3_turbo():
    remaining = ["base", "large-v3-turbo", "small"]

    assert models.pick_model_after_delete(remaining, "medium") == "large-v3-turbo"


def test_pick_model_after_delete_falls_back_to_the_first_remaining():
    remaining = ["base", "medium", "small"]

    assert models.pick_model_after_delete(remaining, "large-v3-turbo") == "base"


def test_pick_model_after_delete_when_the_first_was_selected():
    models_before = ["base", "large-v3-turbo", "small"]

    assert models.pick_model_after_delete(models_before, "base") == "large-v3-turbo"


def test_pick_model_after_delete_when_the_last_was_selected():
    models_before = ["base", "medium", "small"]

    assert models.pick_model_after_delete(models_before, "small") == "base"


def test_pick_model_after_delete_when_it_was_not_selected():
    models_before = ["small", "tiny"]

    assert models.pick_model_after_delete(models_before, "medium") == "small"


def test_pick_model_after_delete_returns_none_when_nothing_is_left():
    assert models.pick_model_after_delete(["small"], "small") is None
    assert models.pick_model_after_delete([], "small") is None


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
        "Downloading Silero VAD: 42%",
        "Downloading…",
    )


def test_vad_state_text_downloading_bytes_without_a_content_length():
    assert vad_state_text(VadState.DOWNLOADING, "512 kB")[1] == (
        "Downloading Silero VAD: 512 kB"
    )


def test_vad_state_text_downloading_before_the_first_tick():
    assert vad_state_text(VadState.DOWNLOADING)[1] == (
        f"Downloading Silero VAD ({human_size(VAD_SPEC.size_bytes)})…"
    )


def test_vad_state_text_failed():
    assert vad_state_text(VadState.FAILED, NO_NETWORK_REASON) == (
        "Silero VAD is not installed",
        "Download failed: no network connection.",
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
