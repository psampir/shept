"""Whisper model catalogue and downloader.

GTK-free on purpose: this module is imported by `shept.app` but is also the
subject of unit tests that run without PyGObject installed. Only the standard
library is used, so importing it adds no dependency to the app's start-up path.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
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


def download_dir() -> Path:
    """Return the directory downloads go to, creating it if needed.

    `$XDG_DATA_HOME/shept/models` is the first place `environment` searches for
    models and the only model directory that is writable inside the Flatpak
    sandbox, so downloads always land there rather than next to an existing
    system-wide model.
    """
    data_home = os.environ.get("XDG_DATA_HOME", "").strip()
    base = Path(data_home) if data_home else Path.home() / ".local" / "share"
    directory = base / "shept" / "models"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def is_downloaded(spec: ModelSpec, dest_dir: Path) -> bool:
    """True when `dest_dir` already holds a complete copy of `spec`."""
    path = Path(dest_dir) / spec.filename
    try:
        size = path.stat().st_size
    except OSError:
        return False
    return abs(size - spec.size_bytes) <= size_tolerance(spec.size_bytes)


def size_tolerance(size_bytes: int) -> int:
    """Allowed size drift for a model file: 1% of its size, capped at 1 MiB."""
    return min(SIZE_TOLERANCE, max(1, size_bytes // 100))


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
        part_path.unlink(missing_ok=True)
        raise
    except OSError as exc:
        part_path.unlink(missing_ok=True)
        raise DownloadError(f"Could not download {spec.name}: {exc}") from exc
    except BaseException:
        part_path.unlink(missing_ok=True)
        raise

    os.replace(part_path, final_path)
    return final_path
