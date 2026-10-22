"""Environment discovery: models, external tools, distro, locale and backend.

Everything here is GTK-free so it can be imported and tested on a machine
without PyGObject installed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

# `languages` is deliberately dependency-free, so importing it here does not
# create a cycle: nothing in that module knows about the environment.
from shept.languages import language_by_code
from shept.models import is_complete_vad

MODEL_GLOB = "ggml-*.bin"
#: whisper.cpp's VAD model is also a `ggml-*.bin` file, so it is matched
#: separately and kept out of the transcription model list.
VAD_MODEL_GLOB = "ggml-silero-*.bin"
MODELS_URL = "https://huggingface.co/ggerganov/whisper.cpp/tree/main"
README_URL = "https://github.com/psampir/shept#install"
ISSUES_URL = "https://github.com/psampir/shept/issues"

#: Tools a run cannot work without: decoding is ffmpeg, transcription is
#: whisper-cli.
RUN_TOOLS = ("whisper-cli", "ffmpeg")

#: Everything the window checks at startup. ffprobe is here so a missing probe
#: is visible instead of silently pinning every progress bar at 0 % — but it is
#: not in `RUN_TOOLS`, because transcription itself does not need it.
REQUIRED_TOOLS = (*RUN_TOOLS, "ffprobe")

_DISTRO_HINTS: dict[str, dict[str, str]] = {
    "fedora": {
        "whisper-cli": (
            "sudo dnf install -y whisper-cpp ffmpeg python3-gobject gtk4 "
            "libadwaita  (whisper-cpp: sudo dnf copr enable -y "
            "sneed/whisper-cpp-vulkan)"
        ),
        "ffmpeg": "sudo dnf install -y ffmpeg",
        "ffprobe": "sudo dnf install -y ffmpeg",
    },
    "arch": {
        "whisper-cli": (
            "sudo pacman -S whisper-cpp ffmpeg python-gobject gtk4 libadwaita"
        ),
        "ffmpeg": "sudo pacman -S ffmpeg",
        "ffprobe": "sudo pacman -S ffmpeg",
    },
}

# library name fragment -> human readable backend, best GPU backend first
BACKEND_MARKERS = (
    ("ggml-vulkan", "Vulkan"),
    ("ggml-cuda", "CUDA"),
    ("ggml-hip", "HIP"),
    ("ggml-metal", "Metal"),
    ("ggml-sycl", "SYCL"),
    ("ggml-opencl", "OpenCL"),
)


def _linked_libraries(binary: str) -> list[str]:
    """Return the ggml libraries `binary` is linked against."""
    try:
        ldd = subprocess.run(
            ["ldd", binary],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
        if ldd.returncode == 0:
            lines = [ln for ln in ldd.stdout.splitlines() if "ggml" in ln]
            if lines:
                return lines
    except Exception:
        pass
    try:
        ldconfig = subprocess.run(
            ["ldconfig", "-p"],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
        if ldconfig.returncode == 0:
            return [ln for ln in ldconfig.stdout.splitlines() if "libggml-" in ln]
    except Exception:
        pass
    return []


def detect_backend() -> str:
    """Detect the whisper.cpp backend on this machine.

    Returns one of "Vulkan", "CUDA", "HIP", "Metal", "SYCL", "OpenCL",
    "CPU" (installed but no GPU backend), "not installed" or "unknown".
    """
    try:
        binary = None
        for name in ("whisper-cli", "whisper-cpp", "main"):
            binary = shutil.which(name)
            if binary is not None:
                break
        if binary is None:
            return "not installed"

        libs = _linked_libraries(binary)
        for marker, label in BACKEND_MARKERS:
            if any(marker in lib for lib in libs):
                return label
        return "CPU"
    except Exception:
        return "unknown"


def model_search_paths() -> list[Path]:
    """Directories searched for models, highest priority first."""
    home = Path.home()
    data_home = os.environ.get("XDG_DATA_HOME", "")
    data_dir = Path(data_home) if data_home else home / ".local" / "share"
    override = os.environ.get("SHEPT_MODEL_DIR", "")

    wanted: list[Path] = []
    if override:
        wanted.append(Path(override).expanduser())
    wanted.extend(
        [
            data_dir / "shept" / "models",
            data_dir / "whisper-cpp" / "models",
            # Both the XDG data home and the real home are searched. Inside the
            # Flatpak sandbox XDG_DATA_HOME is app-private (~/.var/app/<app-id>/
            # data) while $HOME is still the user's home, because the manifest
            # grants `--filesystem=home:ro`; without this entry the models a
            # user installed in ~/.local/share would be invisible. The same
            # applies to any user with a non-default XDG_DATA_HOME.
            home / ".local" / "share" / "whisper-cpp" / "models",
            home / ".cache" / "whisper.cpp" / "models",
            Path("/usr/share/whisper-cpp/models"),
            Path("/usr/local/share/whisper-cpp/models"),
        ]
    )

    paths: list[Path] = []
    for path in wanted:
        if path not in paths:
            paths.append(path)
    return paths


def model_display_name(path: Path) -> str:
    """Turn `ggml-large-v3-turbo.bin` into `large-v3-turbo`."""
    return path.name.removeprefix("ggml-").removesuffix(".bin")


def abbreviate_home(path: Path) -> str:
    """Return `path` with the home directory written as `~`.

    Paths outside the home directory are returned unchanged, so the result is
    always a usable absolute path and never leaks what the home directory
    happens to be called.
    """
    path = Path(path)
    try:
        home = Path.home()
        relative = path.relative_to(home)
    except (RuntimeError, ValueError):
        # No home directory at all, or a path outside it.
        return str(path)
    if relative == Path("."):
        return "~"
    return f"~/{relative}"


def detect_distro(path: Path = Path("/etc/os-release")) -> str:
    """Classify the distribution as "fedora", "debian", "arch" or "other"."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "other"

    fields: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, _, value = line.partition("=")
        fields[key.strip()] = value.strip().strip("\"'")

    ids = [fields.get("ID", ""), *fields.get("ID_LIKE", "").split()]
    ids = [ident.lower() for ident in ids if ident]
    families = {
        "fedora": ("fedora", "rhel", "centos", "rocky", "almalinux"),
        "debian": ("debian", "ubuntu"),
        "arch": ("arch", "archlinux", "manjaro", "endeavouros"),
    }
    for family, names in families.items():
        if any(ident in names for ident in ids):
            return family
    return "other"


def is_flatpak(marker: Path = Path("/.flatpak-info")) -> bool:
    """True when running inside a Flatpak sandbox.

    Flatpak creates `/.flatpak-info` in the sandbox. The marker is a parameter
    so the check can be tested without actually being inside a sandbox.
    """
    return marker.exists()


def missing_tools(
    tools: Sequence[str] = REQUIRED_TOOLS,
    which: Callable[[str], str | None] = shutil.which,
) -> list[str]:
    """The tools from `tools` that are not on PATH, in the given order.

    A missing ffprobe belongs in the banner too: without it every job reports
    0 % progress for its whole length, which looks like a hung run rather than
    a missing tool. `which` is injectable so the decision can be tested without
    touching PATH.
    """
    return [name for name in tools if which(name) is None]


def install_hint(tool: str) -> str:
    """Return an install hint for `tool` that fits this distribution."""
    if is_flatpak():
        # Distributions' package managers are not available in the sandbox:
        # a missing tool means the Flatpak itself is broken.
        return (
            f"this Flatpak is missing a bundled tool — please report it at {ISSUES_URL}"
        )
    hint = _DISTRO_HINTS.get(detect_distro(), {}).get(tool)
    if hint is not None:
        return hint
    if tool == "whisper-cli":
        return f"build whisper.cpp from source, see {README_URL}#debian--ubuntu"
    return f"install {tool} with your package manager, see {README_URL}"


def default_language() -> str:
    """Pick the default transcription language from the system locale."""
    raw = ""
    for var in ("LC_ALL", "LC_MESSAGES", "LANG"):
        value = os.environ.get(var, "").strip()
        if value:
            raw = value
            break

    base = raw.split("_", 1)[0].split(".", 1)[0].split("@", 1)[0].lower()
    if not base or base in {"c", "posix"}:
        return "auto"

    # Most locales name a 2-letter language, but whisper also has 3-letter
    # codes (haw, yue), so try the whole base first and then its first two
    # letters — "haw_US" must not degrade to Hausa ("ha").
    for candidate in (base, base[:2]):
        if language_by_code(candidate) is not None:
            return candidate
    return "auto"


def discover_models() -> list[Path]:
    """Find models in every search path; first match wins on duplicates.

    Silero VAD files live in the same directories and match the same glob, but
    they are not Whisper models, so they are filtered out here.
    """
    seen: set[str] = set()
    models: list[Path] = []
    for directory in model_search_paths():
        try:
            candidates = sorted(directory.glob(MODEL_GLOB))
        except OSError:
            continue
        for candidate in candidates:
            if candidate.match(VAD_MODEL_GLOB):
                continue
            name = model_display_name(candidate)
            if name in seen:
                continue
            seen.add(name)
            models.append(candidate)
    return sorted(models, key=model_display_name)


def find_vad_model() -> Path | None:
    """Return the Silero VAD model whisper.cpp should load, or None.

    The same directories as the Whisper models are searched; the first
    `ggml-silero-*.bin` of the expected size wins, so a file the user installed
    next to their models is used before anything further down the list. A
    truncated or corrupt file counts as missing rather than being offered as
    `-vm`, which would only fail the next run.
    """
    for directory in model_search_paths():
        try:
            candidates = sorted(directory.glob(VAD_MODEL_GLOB))
        except OSError:
            continue
        for candidate in candidates:
            if is_complete_vad(candidate):
                return candidate
    return None


def default_skip_silence(vad_model: Path | None) -> bool:
    """Whether Skip silence starts on: only when the VAD asset is installed.

    Turning it on without the asset would make whisper.cpp fail on a missing
    `-vm` file, so a fresh install starts with the switch off and offers a
    download instead.
    """
    return vad_model is not None


def vad_model_for(enabled: bool, vad_model: Path | None) -> Path | None:
    """The `-vm` path a run should pass, or None when VAD is off or missing."""
    return vad_model if enabled else None
