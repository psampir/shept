"""Whisper model catalogue and downloader.

GTK-free on purpose: this module is imported by `shept.app` but is also the
subject of unit tests that run without PyGObject installed. Only the standard
library is used, so importing it adds no dependency to the app's start-up path.
"""

from __future__ import annotations

import errno
import os
import socket
import stat
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

BASE_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
CHUNK_SIZE = 256 * 1024
USER_AGENT = "shept/0.1 (+https://github.com/psampir/shept)"

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
        "same model, quantized — a third of the size",
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
            f"Downloading Silero VAD — {detail}"
            if detail
            else f"Downloading Silero VAD ({human_size(size_bytes)})…"
        )
        return title, subtitle, "Downloading…"
    if state is VadState.FAILED:
        subtitle = f"Download failed — {detail}." if detail else "Download failed."
        return title, subtitle, "Retry"
    raise ValueError(f"unknown VAD state: {state!r}")


def download_dir() -> Path:
    """Return the directory downloads go to, creating it if needed.

    `$XDG_DATA_HOME/shept/models` is the first place `environment` searches for
    models and the only model directory that is writable inside the Flatpak
    sandbox, so downloads always land there rather than next to an existing
    system-wide model. Leftover `.part` files from an interrupted run are
    removed here, so they cannot accumulate forever.
    """
    data_home = os.environ.get("XDG_DATA_HOME", "").strip()
    base = Path(data_home) if data_home else Path.home() / ".local" / "share"
    directory = base / "shept" / "models"
    directory.mkdir(parents=True, exist_ok=True)
    _remove_stale_parts(directory)
    return directory


def _remove_stale_parts(directory: Path) -> None:
    """Delete half-written downloads; nothing ever reads a `.part` file back."""
    try:
        parts = list(directory.glob("*.part"))
    except OSError:
        return
    for part in parts:
        _discard(part)


def _discard(path: Path) -> None:
    """Best-effort removal of a temporary file; never raises."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def is_downloaded(spec: ModelSpec, dest_dir: Path) -> bool:
    """True when `dest_dir` already holds a complete copy of `spec`."""
    path = Path(dest_dir) / spec.filename
    try:
        size = path.stat().st_size
    except OSError:
        return False
    return abs(size - spec.size_bytes) <= size_tolerance(spec.size_bytes)


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
    must be writable — unlinking a file needs write access to the directory,
    not to the file — and the file must not live in a system directory. Inside
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


def download(
    spec: ModelSpec,
    dest_dir: Path,
    progress: Callable[[int, int], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> Path:
    """Download `spec` into `dest_dir` and return the resulting path.

    The transfer goes to `<filename>.part` and is only renamed to its final
    name once it completed, so an interrupted run never leaves a truncated
    `.bin` behind for `discover_models()` to pick up. `progress` is called with
    `(received_bytes, total_bytes)`, where `total_bytes` is 0 when the server
    sends no `Content-Length`.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    final_path = dest_dir / spec.filename
    part_path = dest_dir / (spec.filename + ".part")

    def cancelled() -> bool:
        return is_cancelled is not None and is_cancelled()

    if cancelled():
        raise DownloadCancelled(f"Download of {spec.name} was cancelled.")

    request = Request(spec.url, headers={"User-Agent": USER_AGENT})
    received = 0
    try:
        with urlopen(request, timeout=30) as response:  # noqa: S310 — https URL
            total = int(response.headers.get("Content-Length") or 0)
            if progress is not None:
                progress(0, total)
            with part_path.open("wb") as handle:
                while True:
                    if cancelled():
                        raise DownloadCancelled(
                            f"Download of {spec.name} was cancelled."
                        )
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    handle.write(chunk)
                    received += len(chunk)
                    if progress is not None:
                        progress(received, total)
            if total and received != total:
                raise DownloadError(
                    f"Download of {spec.name} was truncated "
                    f"({received} of {total} bytes)."
                )
    except DownloadCancelled:
        _discard(part_path)
        raise
    except OSError as exc:
        _discard(part_path)
        raise DownloadError(f"Could not download {spec.name}: {exc}") from exc
    except BaseException:
        _discard(part_path)
        raise

    try:
        os.replace(part_path, final_path)
    except OSError:
        # A failed rename must not leave the half-written file behind either.
        _discard(part_path)
        raise
    return final_path
