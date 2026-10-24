"""Whisper model catalogue and downloader.

GTK-free on purpose: this module is imported by `shept.app` but is also the
subject of unit tests that run without PyGObject installed. Only the standard
library is used, so importing it adds no dependency to the app's start-up path.
"""

from __future__ import annotations

import errno
import http.client
import os
import socket
import stat
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
CHUNK_SIZE = 256 * 1024
USER_AGENT = "shept/0.1 (+https://github.com/psampir/shept)"

#: Waits before each retry of a transfer that failed in a recoverable way, in
#: seconds. Four attempts in total: the first one plus one per entry. Only
#: failures another attempt can get past are retried; a 404 never is.
RETRY_DELAYS: tuple[float, ...] = (1.0, 2.0, 4.0)

#: How long an untouched `.part` file is kept before `download_dir()` treats it
#: as garbage. Younger ones are a download a user can still resume, so deleting
#: them on every start (as older versions did) would throw away gigabytes.
STALE_PART_AGE = 7 * 24 * 60 * 60

#: Longest single sleep inside a backoff wait, so cancelling during one takes
#: effect within this many seconds instead of after the whole delay.
BACKOFF_SLICE = 0.1

#: Cap on the size difference tolerated when deciding whether a file is
#: complete (1% of the file, never more than this). HuggingFace publishes no
#: per-file checksum we could pin, so the size is the only integrity signal.
SIZE_TOLERANCE = 1024 * 1024


class DownloadError(RuntimeError):
    """A model download failed."""


class DownloadCancelled(DownloadError):
    """A model download was cancelled by the user."""


@dataclass(frozen=True)
class ModelSpec:
    """One downloadable `ggml-*.bin` model."""

    name: str
    filename: str
    size_bytes: int
    url: str
    note: str


def _spec(name: str, size_bytes: int, note: str) -> ModelSpec:
    filename = f"ggml-{name}.bin"
    return ModelSpec(
        name=name,
        filename=filename,
        size_bytes=size_bytes,
        url=f"{BASE_URL}/{filename}",
        note=note,
    )


#: Ordered from the most recommended model to the least. Sizes are the byte
#: counts HuggingFace serves today, so `is_downloaded` can spot truncation.
CATALOG: tuple[ModelSpec, ...] = (
    _spec(
        "large-v3-turbo",
        1_624_555_275,
        "best accuracy/speed balance",
    ),
    _spec(
        "large-v3-turbo-q5_0",
        574_041_195,
        "same model, quantized: a third of the size",
    ),
    _spec(
        "small",
        487_601_967,
        "fast, fine for clear speech",
    ),
    _spec(
        "medium",
        1_533_763_059,
        "more accurate than small, slower",
    ),
    _spec(
        "large-v3",
        3_095_033_483,
        "best accuracy, slowest",
    ),
    _spec(
        "base",
        147_951_465,
        "rough drafts only",
    ),
    _spec(
        "tiny",
        77_691_713,
        "smallest usable model",
    ),
)

#: The catalogue indexed by filename, so a file on disk can be looked up by name
#: even though `discover_models` accepts anything that looks like a model.
SPECS_BY_FILENAME: dict[str, ModelSpec] = {spec.filename: spec for spec in CATALOG}

#: The Silero voice-activity-detection model whisper.cpp loads for `--vad`.
#: It is an auxiliary asset rather than a Whisper model, so it is deliberately
#: kept out of `CATALOG`: it must never be offered as a transcription model or
#: show up in the model dropdown. Upstream publishes it in a separate Hugging
#: Face repository, hence the literal URL instead of `BASE_URL`.
VAD_SPEC = ModelSpec(
    name="silero-vad",
    filename="ggml-silero-v6.2.0.bin",
    size_bytes=885_098,
    url="https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v6.2.0.bin",
    note="voice-activity detection for skipping silence",
)


def human_size(size_bytes: int) -> str:
    """Format a byte count for the UI, using decimal units like HuggingFace."""
    if size_bytes >= 1_000_000_000:
        return f"{size_bytes / 1_000_000_000:.2f} GB"
    if size_bytes >= 1_000_000:
        return f"{size_bytes / 1_000_000:.0f} MB"
    return f"{size_bytes / 1_000:.0f} kB"


# ------------------------------------------------------- download failure text

#: The short reasons `describe_download_error` maps to. They are constants so
#: callers can compare against them without duplicating the wording.
NO_NETWORK_REASON = "no network connection"
NO_RESPONSE_REASON = "the server did not respond"
NO_SPACE_REASON = "not enough disk space"


def _root_cause(exc: BaseException) -> BaseException:
    """Follow `__cause__` to the exception that actually went wrong."""
    seen: set[int] = set()
    current = exc
    while isinstance(current.__cause__, BaseException):
        if id(current) in seen:
            break
        seen.add(id(current))
        current = current.__cause__
    return current


def _is_name_resolution(exc: BaseException) -> bool:
    """True for a DNS failure, whether it arrives bare or as a URL error."""
    reason = getattr(exc, "reason", None)
    return isinstance(exc, socket.gaierror) or isinstance(reason, socket.gaierror)


def _is_timeout(exc: BaseException | None) -> bool:
    """`socket.timeout` is `TimeoutError` since Python 3.10."""
    return isinstance(exc, TimeoutError)


def describe_download_error(exc: BaseException) -> str:
    """One short sentence explaining a failed download, for the UI row.

    `download()` wraps the useful exception in a `DownloadError`, so the cause
    chain is unwrapped first. The message is deliberately not `str(exc)`: that
    can be a mangled URL or a locale-dependent errno string. Callers keep the
    raw text for the status bar and the tooltip, where debugging detail fits.
    """
    cause = _root_cause(exc)
    text = str(cause).lower()
    if isinstance(cause, HTTPError):
        return f"the server returned HTTP {cause.code}"
    if _is_name_resolution(cause) or "name resolution" in text:
        return NO_NETWORK_REASON
    if _is_timeout(cause) or _is_timeout(getattr(cause, "reason", None)):
        return NO_RESPONSE_REASON
    if "timed out" in text:
        return NO_RESPONSE_REASON
    if isinstance(cause, OSError) and cause.errno == errno.ENOSPC:
        return NO_SPACE_REASON
    return type(cause).__name__


def download_error_tooltip(exc: BaseException) -> str:
    """Tooltip text for a failed download: the raw message plus a hint.

    The Flatpak remark belongs here rather than in the row: the row holds one
    line for every user, the tooltip is for whoever reports the bug.
    """
    tooltip = str(exc) or type(exc).__name__
    if describe_download_error(exc) == NO_NETWORK_REASON:
        tooltip += (
            "\nInside the Flatpak sandbox this usually means the app has no "
            "network access (the manifest needs --share=network)."
        )
    return tooltip


# --------------------------------------------------------- VAD hint row state


class VadState(Enum):
    """The states the Silero VAD hint row can be in.

    `MISSING` and `FAILED` both offer the download, `DOWNLOADING` shows its
    progress and `INSTALLED` hides the row. The app applies them through one
    method, so the row can never mix, say, a failure subtitle with an idle
    button.
    """

    MISSING = "missing"
    DOWNLOADING = "downloading"
    FAILED = "failed"
    INSTALLED = "installed"


def vad_state_text(
    state: VadState,
    detail: str = "",
    size_bytes: int = VAD_SPEC.size_bytes,
) -> tuple[str, str, str]:
    """Title, subtitle and button label for one state of the VAD hint row.

    `detail` is the progress text (`42%` or `512 kB`) for `DOWNLOADING` and the
    friendly reason from `describe_download_error` for `FAILED`.
    """
    if state is VadState.INSTALLED:
        return "Silero VAD is installed", "", ""
    title = "Silero VAD is not installed"
    if state is VadState.MISSING:
        return (
            title,
            f"Download it ({human_size(size_bytes)}) to skip silence.",
            "Download",
        )
    if state is VadState.DOWNLOADING:
        subtitle = (
            f"Downloading Silero VAD: {detail}"
            if detail
            else f"Downloading Silero VAD ({human_size(size_bytes)})…"
        )
        return title, subtitle, "Downloading…"
    if state is VadState.FAILED:
        subtitle = f"Download failed: {detail}." if detail else "Download failed."
        return title, subtitle, "Retry"
    raise ValueError(f"unknown VAD state: {state!r}")


def download_dir() -> Path:
    """Return the directory downloads go to, creating it if needed.

    `$XDG_DATA_HOME/shept/models` is the first place `environment` searches for
    models and the only model directory that is writable inside the Flatpak
    sandbox, so downloads always land there rather than next to an existing
    system-wide model. `.part` files older than `STALE_PART_AGE` are removed
    here, so abandoned downloads cannot pile up forever; younger ones are kept
    because `download()` resumes them.
    """
    data_home = os.environ.get("XDG_DATA_HOME", "").strip()
    base = Path(data_home) if data_home else Path.home() / ".local" / "share"
    directory = base / "shept" / "models"
    directory.mkdir(parents=True, exist_ok=True)
    _remove_stale_parts(directory)
    return directory


def _remove_stale_parts(directory: Path) -> None:
    """Delete `.part` files nothing has touched for `STALE_PART_AGE`.

    A fresh `.part` is the first half of a download `download()` will resume, so
    it must survive a restart. Only one old enough that the user has clearly
    given up on it is collected.
    """
    try:
        parts = list(directory.glob("*.part"))
    except OSError:
        return
    now = time.time()
    for part in parts:
        try:
            age = now - part.stat().st_mtime
        except OSError:
            continue
        if age > STALE_PART_AGE:
            _discard(part)


def _discard(path: Path) -> None:
    """Best-effort removal of a temporary file; never raises."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def is_downloaded(spec: ModelSpec, dest_dir: Path) -> bool:
    """True when `dest_dir` already holds a complete copy of `spec`."""
    return _matches_expected_size(spec, Path(dest_dir) / spec.filename)


def spec_for_path(path: Path) -> ModelSpec | None:
    """The catalogue entry for a model file, or None when Shept does not know it.

    A file Shept does not publish is not wrong: the user may have fetched it
    from somewhere else, so callers must treat None as "unknown", never as
    "broken".
    """
    return SPECS_BY_FILENAME.get(Path(path).name)


def is_complete_model_file(path: Path) -> bool | None:
    """Whether a model file matches the size the catalogue publishes.

    Returns None for a model Shept does not publish: `discover_models` accepts
    any `ggml-*.bin`, and a file nobody published a size for cannot be judged.
    The tolerance is the one `is_downloaded` uses, so the dropdown, the model
    manager and the downloader all agree on what a complete file is.
    """
    spec = spec_for_path(path)
    if spec is None:
        return None
    return _matches_expected_size(spec, Path(path))


def is_complete_vad(path: Path) -> bool:
    """True when `path` is a Silero VAD file of the expected size.

    A truncated or otherwise damaged download counts as missing: handing its
    path to whisper-cli as `-vm` fails the run with a message that never
    mentions the asset, so the window would offer it again anyway.
    """
    try:
        size = Path(path).stat().st_size
    except OSError:
        return False
    return abs(size - VAD_SPEC.size_bytes) <= size_tolerance(VAD_SPEC.size_bytes)


def size_tolerance(size_bytes: int) -> int:
    """Allowed size drift for a model file: 1% of its size, capped at 1 MiB."""
    return min(SIZE_TOLERANCE, max(1, size_bytes // 100))


# ---------------------------------------------------- deleting installed models

#: The model the app recommends, so it wins the fallback selection after a
#: deletion even when another installed model sorts first alphabetically.
PREFERRED_MODEL = "large-v3-turbo"

#: Directories that belong to the system rather than to the user. Shept never
#: deletes from them, even when the current user happens to be allowed to.
READ_ONLY_PREFIXES = (
    Path("/usr"),
    Path("/nix/store"),
    Path("/snap"),
)


def in_read_only_location(path: Path) -> bool:
    """True when `path` sits below a directory that is not the user's to edit."""
    path = Path(path)
    return any(
        path == prefix or prefix in path.parents for prefix in READ_ONLY_PREFIXES
    )


def is_deletable(path: Path) -> bool:
    """True when Shept can actually remove `path`.

    Both conditions are checked rather than assumed: the containing directory
    must be writable (unlinking a file needs write access to the directory,
    not to the file), and the file must not live in a system directory. Inside
    the Flatpak the manifest grants `home:ro`, so `~/.local/share/whisper-cpp/
    models` looks writable by its mode bits while `access(W_OK)` still refuses
    it; those rows get a lock icon instead of a trash button that could only
    fail.

    A symlink counts as deletable when its own directory is writable, because
    removing it writes to that directory and never touches the target.
    """
    path = Path(path)
    try:
        mode = path.lstat().st_mode
    except OSError:
        return False
    if not (stat.S_ISREG(mode) or stat.S_ISLNK(mode)):
        return False
    if in_read_only_location(path):
        return False
    return os.access(path.parent, os.W_OK)


def deletable_paths(paths: Iterable[Path]) -> set[Path]:
    """The subset of `paths` that `is_deletable` accepts."""
    return {Path(path) for path in paths if is_deletable(path)}


def delete_model(path: Path) -> None:
    """Remove one model file, raising `OSError` when that is not possible.

    `unlink` is deliberate: for a symlink it removes the link and leaves the
    file it points at alone, which is what a user who linked a model in from
    another disk expects. Nothing else in the directory is touched, so no
    `.part` file or temporary copy can be left behind by a deletion.
    """
    Path(path).unlink()


def pick_model_after_delete(models: Sequence[str], deleted: str) -> str | None:
    """The model to select once `deleted` is gone, or None when none are left.

    `large-v3-turbo` is the model the app recommends, so it is preferred over
    the alphabetically first remaining one; the first remaining model is the
    fallback, and an empty result means the window shows its "no models"
    state.
    """
    remaining = [name for name in models if name != deleted]
    if not remaining:
        return None
    if PREFERRED_MODEL in remaining:
        return PREFERRED_MODEL
    return remaining[0]


class _RecoverableFailure(Exception):
    """A transfer failed in a way another attempt could get past."""

    def __init__(self, message: str, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.cause = cause


def _status_code(response: object) -> int:
    """The HTTP status of a `urlopen` response, or 200 when it does not say."""
    status = getattr(response, "status", None)
    if isinstance(status, int):
        return status
    getcode = getattr(response, "getcode", None)
    if callable(getcode):
        try:
            return int(getcode() or 200)
        except (TypeError, ValueError):
            return 200
    return 200


def _matches_expected_size(spec: ModelSpec, path: Path) -> bool:
    """True when `path` is as big as the catalogue says `spec` is."""
    try:
        size = path.stat().st_size
    except OSError:
        return False
    return abs(size - spec.size_bytes) <= size_tolerance(spec.size_bytes)


def _backoff_sleep(seconds: float, cancelled: Callable[[], bool]) -> bool:
    """Wait `seconds`, checking `cancelled` often; False when it became true.

    The wait is sliced so Cancel takes effect immediately instead of after the
    whole delay: a 4 s backoff that ignores the button reads as a hung app.
    """
    deadline = time.monotonic() + seconds
    while True:
        if cancelled():
            return False
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return True
        time.sleep(min(BACKOFF_SLICE, remaining))


def _transfer_once(
    spec: ModelSpec,
    part_path: Path,
    progress: Callable[[int, int], None] | None,
    raise_if_cancelled: Callable[[], None],
) -> None:
    """One attempt at the transfer, appended to whatever `part_path` holds.

    A `Range` request resumes from the current size of the `.part` file. `206
    Partial Content` appends, while `200 OK` means the server ignored the range
    (or there was nothing to resume), so the file is truncated and written from
    the start rather than silently doubling its contents.
    """
    try:
        offset = part_path.stat().st_size
    except OSError:
        offset = 0
    if offset >= spec.size_bytes:
        # Not a prefix of the published file, so it cannot be resumed.
        _discard(part_path)
        offset = 0

    headers = {"User-Agent": USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = Request(spec.url, headers=headers)
    try:
        response = urlopen(request, timeout=30)  # noqa: S310  # https URL
    except HTTPError as exc:
        if exc.code >= 500:
            raise _RecoverableFailure(
                f"the server returned HTTP {exc.code}", exc
            ) from exc
        raise
    except URLError as exc:
        raise _RecoverableFailure(
            f"could not reach the server ({exc.reason})", exc
        ) from exc

    try:
        with response:
            status = _status_code(response)
            length = int(response.headers.get("Content-Length") or 0)
            if offset and status == 206:
                received = offset
                total = offset + length if length else 0
                mode = "ab"
            else:
                received = 0
                total = length
                mode = "wb"
            if progress is not None:
                progress(received, total)
            with part_path.open(mode) as handle:
                while True:
                    raise_if_cancelled()
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    handle.write(chunk)
                    received += len(chunk)
                    if progress is not None:
                        progress(received, total)
            if total and received != total:
                raise _RecoverableFailure(
                    f"the transfer stopped at {received} of {total} bytes"
                )
    except TimeoutError as exc:
        raise _RecoverableFailure("the connection timed out", exc) from exc
    except http.client.HTTPException as exc:
        # A dropped connection usually surfaces as `IncompleteRead`, which is
        # neither an `OSError` nor a `URLError`.
        raise _RecoverableFailure(
            f"the connection dropped ({type(exc).__name__})", exc
        ) from exc


def _verify_size(spec: ModelSpec, part_path: Path) -> None:
    """Refuse to publish a file that is not the size the catalogue promises."""
    try:
        size = part_path.stat().st_size
    except OSError as exc:
        raise DownloadError(f"Download of {spec.name} produced no file: {exc}") from exc
    if abs(size - spec.size_bytes) > size_tolerance(spec.size_bytes):
        raise DownloadError(
            f"Download of {spec.name} is incomplete: {size} of "
            f"{spec.size_bytes} bytes arrived. The partial file was kept, so "
            "the next attempt resumes it."
        )


def download(
    spec: ModelSpec,
    dest_dir: Path,
    progress: Callable[[int, int], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> Path:
    """Download `spec` into `dest_dir` and return the resulting path.

    The transfer goes to `<filename>.part` and is only renamed to its final
    name once it completed, so an interrupted run never leaves a truncated
    `.bin` behind for `discover_models()` to pick up. The `.part` file survives
    a failure, and the next attempt asks the server for the bytes from its size
    on, which is what makes a 1.6 GB model finish on a flaky connection.

    `progress` is called with `(received_bytes, total_bytes)`, where
    `received_bytes` counts from the resume offset and `total_bytes` is 0 when
    the server sends no `Content-Length`. Recoverable failures (`URLError`,
    timeouts, HTTP 5xx) are retried up to `len(RETRY_DELAYS)` times; a 4xx is
    final, and so is any local error such as a full disk. Cancelling keeps the
    `.part` too, so stopping and starting again continues where it stopped.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    final_path = dest_dir / spec.filename
    part_path = dest_dir / (spec.filename + ".part")

    def cancelled() -> bool:
        return is_cancelled is not None and is_cancelled()

    def raise_if_cancelled() -> None:
        if cancelled():
            raise DownloadCancelled(f"Download of {spec.name} was cancelled.")

    raise_if_cancelled()

    # An earlier run may have finished the transfer and then failed to rename
    # it; downloading those bytes a second time would be pure waste.
    if _matches_expected_size(spec, part_path):
        os.replace(part_path, final_path)
        return final_path

    attempt = 0
    while True:
        try:
            _transfer_once(spec, part_path, progress, raise_if_cancelled)
        except _RecoverableFailure as exc:
            if attempt >= len(RETRY_DELAYS):
                raise DownloadError(
                    f"Could not download {spec.name}: {exc}. The partial file "
                    "was kept, so the next attempt resumes it."
                ) from exc.cause
            if not _backoff_sleep(RETRY_DELAYS[attempt], cancelled):
                raise DownloadCancelled(
                    f"Download of {spec.name} was cancelled."
                ) from exc
            attempt += 1
            raise_if_cancelled()
            continue
        except DownloadCancelled:
            raise
        except OSError as exc:
            # A local failure (a full disk, a read-only folder) or a 4xx: no
            # number of retries can change the answer.
            raise DownloadError(f"Could not download {spec.name}: {exc}") from exc
        break

    _verify_size(spec, part_path)
    # A failed rename keeps the complete `.part`; the check at the top of the
    # next attempt turns it into the final file without a transfer.
    os.replace(part_path, final_path)
    return final_path
