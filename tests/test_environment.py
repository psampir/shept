"""Tests for shept.environment — model discovery, distro, locale, backend.

These must run without PyGObject/GTK installed: nothing here imports `gi`.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from shept import environment


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    """Isolate HOME and clear the variables that influence search paths."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.delenv("SHEPT_MODEL_DIR", raising=False)
    return home


@pytest.fixture
def locale_env(monkeypatch):
    """Clear the locale variables and return a helper that sets one of them."""
    for var in ("LC_ALL", "LC_MESSAGES", "LANG"):
        monkeypatch.delenv(var, raising=False)

    def set_locale(value: str, var: str = "LANG") -> None:
        monkeypatch.setenv(var, value)

    return set_locale


def _make_model(path: Path) -> Path:
    path.write_text("", encoding="utf-8")
    return path


# ---------------------------------------------------------------- model paths


def test_model_search_paths_defaults(fake_home):
    assert environment.model_search_paths() == [
        fake_home / ".local" / "share" / "shept" / "models",
        fake_home / ".local" / "share" / "whisper-cpp" / "models",
        fake_home / ".cache" / "whisper.cpp" / "models",
        Path("/usr/share/whisper-cpp/models"),
        Path("/usr/local/share/whisper-cpp/models"),
    ]


def test_model_search_paths_honours_xdg_data_home(fake_home, tmp_path, monkeypatch):
    data = tmp_path / "data"
    monkeypatch.setenv("XDG_DATA_HOME", str(data))

    paths = environment.model_search_paths()

    assert paths[0] == data / "shept" / "models"
    assert paths[1] == data / "whisper-cpp" / "models"
    assert fake_home / ".local" / "share" / "shept" / "models" not in paths


def test_model_search_paths_override_is_first_and_expanded(fake_home, monkeypatch):
    monkeypatch.setenv("SHEPT_MODEL_DIR", "~/whisper-models")

    paths = environment.model_search_paths()

    assert paths[0] == fake_home / "whisper-models"
    assert paths[1] == fake_home / ".local" / "share" / "shept" / "models"


def test_model_search_paths_priority_order(fake_home, tmp_path, monkeypatch):
    data = tmp_path / "data"
    monkeypatch.setenv("XDG_DATA_HOME", str(data))
    monkeypatch.setenv("SHEPT_MODEL_DIR", "/opt/models")

    paths = environment.model_search_paths()

    assert paths == [
        Path("/opt/models"),
        data / "shept" / "models",
        data / "whisper-cpp" / "models",
        fake_home / ".local" / "share" / "whisper-cpp" / "models",
        fake_home / ".cache" / "whisper.cpp" / "models",
        Path("/usr/share/whisper-cpp/models"),
        Path("/usr/local/share/whisper-cpp/models"),
    ]


def test_model_search_paths_deduplicates(fake_home, tmp_path, monkeypatch):
    data = tmp_path / "data"
    monkeypatch.setenv("XDG_DATA_HOME", str(data))
    monkeypatch.setenv("SHEPT_MODEL_DIR", str(data / "shept" / "models"))

    paths = environment.model_search_paths()

    assert len(paths) == len(set(paths))
    assert paths.count(data / "shept" / "models") == 1


def test_model_search_paths_searches_xdg_and_home(fake_home, monkeypatch):
    """A non-default XDG_DATA_HOME must not hide the real ~/.local/share."""
    monkeypatch.setenv("XDG_DATA_HOME", "/tmp/xdg-test")

    paths = environment.model_search_paths()

    assert Path("/tmp/xdg-test/whisper-cpp/models") in paths
    assert fake_home / ".local" / "share" / "whisper-cpp" / "models" in paths


def test_model_search_paths_default_home_appears_once(fake_home):
    """With no XDG override, the home data dir is reached twice — dedupe it."""
    paths = environment.model_search_paths()

    shared = fake_home / ".local" / "share" / "whisper-cpp" / "models"
    assert paths.count(shared) == 1
    assert len(paths) == len(set(paths))


def test_model_search_paths_xdg_equal_to_home_share_has_no_duplicates(
    fake_home, monkeypatch
):
    monkeypatch.setenv("XDG_DATA_HOME", str(fake_home / ".local" / "share"))

    paths = environment.model_search_paths()

    assert len(paths) == len(set(paths))
    assert paths.count(fake_home / ".local" / "share" / "whisper-cpp" / "models") == 1
    assert paths.count(fake_home / ".local" / "share" / "shept" / "models") == 1


def test_model_search_paths_override_still_first_with_xdg(
    fake_home, tmp_path, monkeypatch
):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("SHEPT_MODEL_DIR", "/opt/models")

    assert environment.model_search_paths()[0] == Path("/opt/models")


def test_model_display_name_strips_ggml_wrapper():
    assert (
        environment.model_display_name(Path("ggml-large-v3-turbo.bin"))
        == "large-v3-turbo"
    )
    assert environment.model_display_name(Path("/m/ggml-base.bin")) == "base"


# -------------------------------------------------------- translation capability


@pytest.mark.parametrize(
    "name",
    [
        "ggml-large-v3-turbo.bin",
        "ggml-large-v3-turbo-q5_0.bin",
        "ggml-some-future-turbo-model.bin",
    ],
)
def test_model_translates_false_for_turbo(name):
    assert environment.model_translates(Path("/models") / name) is False


@pytest.mark.parametrize(
    "name",
    [
        "ggml-large-v3.bin",
        "ggml-medium.bin",
        "ggml-small.bin",
        "ggml-base.bin",
        "ggml-tiny.bin",
        "ggml-base.en.bin",
        "ggml-custom-finetune.bin",
        "whisper.bin",
    ],
)
def test_model_translates_true_for_everything_else(name):
    assert environment.model_translates(Path("/models") / name) is True


def test_model_translates_ignores_the_directory_name(tmp_path):
    # Only the file name matters: a directory called "turbo" must not make
    # every model inside it look transcription-only.
    path = tmp_path / "turbo" / "ggml-base.bin"

    assert environment.model_translates(path) is True


def test_best_translation_model_follows_catalogue_order(tmp_path):
    installed = [
        tmp_path / "ggml-large-v3-turbo.bin",
        tmp_path / "ggml-tiny.bin",
        tmp_path / "ggml-medium.bin",
        tmp_path / "ggml-small.bin",
    ]

    assert environment.best_translation_model(installed) == tmp_path / "ggml-small.bin"


def test_best_translation_model_skips_turbo_entries(tmp_path):
    installed = [
        tmp_path / "ggml-large-v3-turbo-q5_0.bin",
        tmp_path / "ggml-large-v3-turbo.bin",
        tmp_path / "ggml-base.bin",
    ]

    assert environment.best_translation_model(installed) == tmp_path / "ggml-base.bin"


def test_best_translation_model_is_none_without_a_capable_model(tmp_path):
    installed = [
        tmp_path / "ggml-large-v3-turbo.bin",
        tmp_path / "ggml-large-v3-turbo-q5_0.bin",
        tmp_path / "ggml-unknown-turbo.bin",
    ]

    assert environment.best_translation_model(installed) is None
    assert environment.best_translation_model([]) is None


def test_best_translation_model_falls_back_to_uncatalogued_models(tmp_path):
    # A model the catalogue does not know about is still usable if its name
    # does not mark it as transcription-only.
    installed = [
        tmp_path / "ggml-large-v3-turbo.bin",
        tmp_path / "ggml-custom-finetune.bin",
    ]

    assert (
        environment.best_translation_model(installed)
        == tmp_path / "ggml-custom-finetune.bin"
    )


# ------------------------------------------------------------ model discovery


def test_discover_models_finds_only_ggml_bins(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    _make_model(models / "ggml-base.bin")
    _make_model(models / "ggml-small.bin")
    (models / "notes.txt").write_text("not a model", encoding="utf-8")
    (models / "ggml-medium.bin.part").write_text("", encoding="utf-8")
    (models / "whisper.bin").write_text("", encoding="utf-8")

    monkeypatch.setattr(environment, "model_search_paths", lambda: [models])

    found = environment.discover_models()

    assert [path.name for path in found] == ["ggml-base.bin", "ggml-small.bin"]


def test_discover_models_first_path_wins_on_duplicates(tmp_path, monkeypatch):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    _make_model(first / "ggml-large-v3-turbo.bin")
    _make_model(second / "ggml-large-v3-turbo.bin")

    monkeypatch.setattr(environment, "model_search_paths", lambda: [first, second])

    found = environment.discover_models()

    assert found == [first / "ggml-large-v3-turbo.bin"]


def test_discover_models_sorted_by_display_name_and_stable(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    for name in ("ggml-small.bin", "ggml-large-v3-turbo.bin", "ggml-base.bin"):
        _make_model(models / name)

    monkeypatch.setattr(environment, "model_search_paths", lambda: [models])

    found = environment.discover_models()

    assert [environment.model_display_name(path) for path in found] == [
        "base",
        "large-v3-turbo",
        "small",
    ]
    assert environment.discover_models() == found


def test_discover_models_skips_missing_directories(tmp_path, monkeypatch):
    missing = tmp_path / "missing"
    models = tmp_path / "models"
    models.mkdir()
    _make_model(models / "ggml-base.bin")

    monkeypatch.setattr(environment, "model_search_paths", lambda: [missing, models])

    assert environment.discover_models() == [models / "ggml-base.bin"]


def test_discover_models_empty_when_nothing_found(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()

    monkeypatch.setattr(environment, "model_search_paths", lambda: [models])

    assert environment.discover_models() == []


# -------------------------------------------------------------------- distro


def _os_release(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "os-release"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ID=fedora\nVERSION_ID=44\n", "fedora"),
        ('ID=linuxmint\nID_LIKE="ubuntu debian"\n', "debian"),
        ("ID=ubuntu\nID_LIKE=debian\n", "debian"),
        ('ID="arch"\n', "arch"),
        ('ID=manjaro\nID_LIKE="arch"\n', "arch"),
        ('ID=rocky\nID_LIKE="rhel centos fedora"\n', "fedora"),
        ("ID=gentoo\n", "other"),
        ('NAME="Some Distro"\n', "other"),
    ],
)
def test_detect_distro(tmp_path, text, expected):
    assert environment.detect_distro(_os_release(tmp_path, text)) == expected


def test_detect_distro_unreadable_file(tmp_path):
    assert environment.detect_distro(tmp_path / "does-not-exist") == "other"


def test_detect_distro_skips_comments_and_malformed_lines(tmp_path):
    text = "# ID=debian\n\nno equals sign here\nID = fedora \nID_LIKE =\n"
    assert environment.detect_distro(_os_release(tmp_path, text)) == "fedora"


# -------------------------------------------------------------- install hints


def test_install_hint_fedora(monkeypatch):
    monkeypatch.setattr(environment, "detect_distro", lambda: "fedora")

    assert "dnf install" in environment.install_hint("whisper-cli")
    assert "whisper-cpp" in environment.install_hint("whisper-cli")
    assert environment.install_hint("ffmpeg") == "sudo dnf install -y ffmpeg"


def test_install_hint_arch(monkeypatch):
    monkeypatch.setattr(environment, "detect_distro", lambda: "arch")

    assert "pacman -S whisper-cpp" in environment.install_hint("whisper-cli")
    assert environment.install_hint("ffmpeg") == "sudo pacman -S ffmpeg"


def test_install_hint_fallback_for_whisper_cli(monkeypatch):
    monkeypatch.setattr(environment, "detect_distro", lambda: "other")

    hint = environment.install_hint("whisper-cli")

    assert "build whisper.cpp from source" in hint
    assert "#debian--ubuntu" in hint


def test_install_hint_fallback_for_other_tools(monkeypatch):
    monkeypatch.setattr(environment, "detect_distro", lambda: "other")

    hint = environment.install_hint("ffmpeg")

    assert "install ffmpeg with your package manager" in hint
    assert environment.README_URL in hint


# ------------------------------------------------------------------ flatpak


def test_is_flatpak_detects_marker(tmp_path):
    marker = tmp_path / ".flatpak-info"

    assert environment.is_flatpak(marker) is False

    marker.write_text("[Application]\nname=io.github.psampir.Shept\n", encoding="utf-8")

    assert environment.is_flatpak(marker) is True


@pytest.mark.parametrize("tool", ["whisper-cli", "ffmpeg"])
def test_install_hint_prefers_flatpak_message_over_distro(tmp_path, monkeypatch, tool):
    marker = tmp_path / ".flatpak-info"
    marker.write_text("[Application]\n", encoding="utf-8")
    # install_hint() calls is_flatpak() with the real sandbox marker, so bind
    # the check to the temporary marker instead of looking at /.
    monkeypatch.setattr(environment, "is_flatpak", lambda: marker.exists())
    monkeypatch.setattr(environment, "detect_distro", lambda: "fedora")

    hint = environment.install_hint(tool)

    assert "Flatpak" in hint
    assert environment.ISSUES_URL in hint
    assert "dnf" not in hint


# -------------------------------------------------------------------- locale


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("pl_PL.UTF-8", "pl"),
        ("en_US.UTF-8", "en"),
        ("de_DE@euro", "de"),
        ("pl_PL", "pl"),
        ("uk_UA.utf8", "uk"),
        ("xx_YY.UTF-8", "auto"),
        ("C", "auto"),
        ("C.UTF-8", "auto"),
        ("POSIX", "auto"),
        ("", "auto"),
    ],
)
def test_default_language(locale_env, value, expected):
    if value:
        locale_env(value)
    assert environment.default_language() == expected


def test_default_language_lc_all_beats_lang(locale_env):
    locale_env("en_US.UTF-8", "LANG")
    locale_env("pl_PL.UTF-8", "LC_ALL")

    assert environment.default_language() == "pl"


def test_default_language_lc_messages_beats_lang(locale_env):
    locale_env("en_US.UTF-8", "LANG")
    locale_env("de_DE.UTF-8", "LC_MESSAGES")

    assert environment.default_language() == "de"


# ------------------------------------------------------------------- backend


def _which_only(*names: str):
    def which(name: str) -> str | None:
        return "/usr/bin/whisper-cli" if name in names else None

    return which


def _run_returning(stdout: str, returncode: int = 0):
    def run(cmd, **_kwargs):
        return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr="")

    return run


def test_detect_backend_not_installed(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _name: None)

    assert environment.detect_backend() == "not installed"


@pytest.mark.parametrize(
    ("library", "expected"),
    [
        ("libggml-vulkan.so.1", "Vulkan"),
        ("libggml-cuda.so.0", "CUDA"),
        ("libggml-hip.so.0", "HIP"),
    ],
)
def test_detect_backend_gpu_marker(monkeypatch, library, expected):
    monkeypatch.setattr(shutil, "which", _which_only("whisper-cli"))
    monkeypatch.setattr(
        subprocess, "run", _run_returning(f"{library} => /usr/lib64/{library}\n")
    )

    assert environment.detect_backend() == expected


def test_detect_backend_cpu_when_only_generic_ggml_libs(monkeypatch):
    monkeypatch.setattr(shutil, "which", _which_only("whisper-cli"))
    monkeypatch.setattr(
        subprocess,
        "run",
        _run_returning(
            "libggml.so.0 => /usr/lib64/libggml.so.0\n"
            "libggml-cpu.so.0 => /usr/lib64/libggml-cpu.so.0\n"
        ),
    )

    assert environment.detect_backend() == "CPU"


def test_detect_backend_falls_back_to_ldconfig(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        if cmd[0] == "ldd":
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="nope")
        return subprocess.CompletedProcess(
            cmd, 0, stdout="libggml-vulkan.so.1 => /usr/lib64/libggml-vulkan.so.1\n"
        )

    monkeypatch.setattr(shutil, "which", _which_only("whisper-cli"))
    monkeypatch.setattr(subprocess, "run", fake_run)

    assert environment.detect_backend() == "Vulkan"
    assert [call[0] for call in calls] == ["ldd", "ldconfig"]


def test_detect_backend_tries_fallback_binary_names(monkeypatch):
    seen: list[str] = []

    def which(name: str) -> str | None:
        seen.append(name)
        return "/usr/bin/main" if name == "main" else None

    monkeypatch.setattr(shutil, "which", which)
    monkeypatch.setattr(
        subprocess, "run", _run_returning("libggml-cpu.so.0 => /usr/lib64/x.so\n")
    )

    assert environment.detect_backend() == "CPU"
    assert seen[:3] == ["whisper-cli", "whisper-cpp", "main"]


def test_detect_backend_unknown_on_unexpected_error(monkeypatch):
    def boom(_binary: str) -> list[str]:
        raise RuntimeError("boom")

    monkeypatch.setattr(shutil, "which", _which_only("whisper-cli"))
    monkeypatch.setattr(environment, "_linked_libraries", boom)

    assert environment.detect_backend() == "unknown"
