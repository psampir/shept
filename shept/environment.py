"""Environment discovery: models, external tools, distro, locale and backend.

Everything here is GTK-free so it can be imported and tested on a machine
without PyGObject installed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path

from shept.models import CATALOG

MODEL_GLOB = "ggml-*.bin"
MODELS_URL = "https://huggingface.co/ggerganov/whisper.cpp/tree/main"
README_URL = "https://github.com/psampir/shept#install"
ISSUES_URL = "https://github.com/psampir/shept/issues"

REQUIRED_TOOLS = ("whisper-cli", "ffmpeg")

_DISTRO_HINTS: dict[str, dict[str, str]] = {
    "fedora": {
        "whisper-cli": (
            "sudo dnf install -y whisper-cpp ffmpeg python3-gobject gtk4 "
            "libadwaita  (whisper-cpp: sudo dnf copr enable -y "
            "sneed/whisper-cpp-vulkan)"
        ),
        "ffmpeg": "sudo dnf install -y ffmpeg",
    },
    "arch": {
        "whisper-cli": (
            "sudo pacman -S whisper-cpp ffmpeg python-gobject gtk4 libadwaita"
        ),
        "ffmpeg": "sudo pacman -S ffmpeg",
    },
}

LANGUAGES = [
    ("pl", "Polish"),
    ("en", "English"),
    ("de", "German"),
    ("fr", "French"),
    ("es", "Spanish"),
    ("uk", "Ukrainian"),
    ("ru", "Russian"),
    ("auto", "Auto-detect"),
]

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


def model_translates(path: Path) -> bool:
    """True when the model behind `path` can translate to English.

    The turbo models were fine-tuned without translation data, so whisper.cpp
    copies the source text verbatim when they are asked to translate. The
    check is on the file name rather than the catalogue, because models found
    on disk are not necessarily the ones Shept can download.
    """
    return "turbo" not in Path(path).name.lower()


def best_translation_model(installed: Iterable[Path]) -> Path | None:
    """Return the most recommended installed model that can translate.

    "Most recommended" means first in `CATALOG` order, so the result is the
    best model to offer as a one-click replacement when the selected one
    cannot translate. Installed models outside the catalogue are only used
    when no catalogue entry qualifies, and never for models the catalogue says
    cannot translate. Returns None when nothing suitable is installed.
    """
    by_name: dict[str, Path] = {}
    for path in installed:
        by_name.setdefault(model_display_name(path), Path(path))

    for spec in CATALOG:
        if not spec.translates:
            continue
        path = by_name.get(spec.name)
        if path is not None:
            return path

    catalogued = {spec.name for spec in CATALOG}
    for name in sorted(by_name):
        if name not in catalogued and model_translates(by_name[name]):
            return by_name[name]
    return None


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

    code = base[:2]
    if code in {lang for lang, _label in LANGUAGES}:
        return code
    return "auto"


def discover_models() -> list[Path]:
    """Find models in every search path; first match wins on duplicates."""
    seen: set[str] = set()
    models: list[Path] = []
    for directory in model_search_paths():
        try:
            candidates = sorted(directory.glob(MODEL_GLOB))
        except OSError:
            continue
        for candidate in candidates:
            name = model_display_name(candidate)
            if name in seen:
                continue
            seen.add(name)
            models.append(candidate)
    return sorted(models, key=model_display_name)
