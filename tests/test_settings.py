"""Tests for shept.settings — persistence, tolerance and atomic writes.

These must run without PyGObject: nothing here imports `gi`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shept import settings as settings_module
from shept.settings import Settings, load, save, settings_path


@pytest.fixture
def locale_env(monkeypatch):
    """Pin the locale so the language default is predictable."""
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    monkeypatch.delenv("LC_ALL", raising=False)
    monkeypatch.delenv("LC_MESSAGES", raising=False)


@pytest.fixture
def target(tmp_path):
    return tmp_path / "config" / "shept" / "settings.json"


def _write(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---------------------------------------------------------------- settings_path


def test_settings_path_honours_xdg_config_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))

    assert settings_path() == tmp_path / "cfg" / "shept" / "settings.json"


def test_settings_path_falls_back_to_home_config(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setenv("HOME", str(home))

    assert settings_path() == home / ".config" / "shept" / "settings.json"


def test_settings_path_ignores_an_empty_xdg_config_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("XDG_CONFIG_HOME", "  ")
    monkeypatch.setenv("HOME", str(home))

    assert settings_path() == home / ".config" / "shept" / "settings.json"


# ---------------------------------------------------------------------- defaults


def test_defaults_follow_the_locale(locale_env):
    settings = Settings()

    assert settings.model is None
    assert settings.language == "en"
    assert settings.skip_silence is False


def test_missing_file_returns_defaults(locale_env, tmp_path):
    settings = load(tmp_path / "nope" / "settings.json")

    assert settings == Settings()
    assert settings.language == "en"


def test_load_without_path_uses_settings_path(locale_env, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))

    assert load() == Settings()


# ------------------------------------------------------------------- round trip


def test_save_then_load_round_trips(target):
    original = Settings(model="small", language="pl", skip_silence=True)

    save(original, target)

    assert load(target) == original


def test_save_creates_parent_directories(target):
    assert not target.parent.exists()

    save(Settings(model="base"), target)

    assert target.is_file()


def test_save_overwrites_an_existing_file(target):
    save(Settings(model="base", language="pl", skip_silence=False), target)
    save(Settings(model="tiny", language="de", skip_silence=True), target)

    assert load(target) == Settings(model="tiny", language="de", skip_silence=True)


def test_saved_file_is_readable_json(target):
    save(Settings(model="small", language="uk", skip_silence=True), target)

    payload = json.loads(target.read_text(encoding="utf-8"))

    assert payload == {
        "model": "small",
        "language": "uk",
        "skip_silence": True,
        "favourites": [],
    }


# ----------------------------------------------------------------- atomic write


def test_save_leaves_no_temporary_file(target):
    save(Settings(model="small"), target)

    assert sorted(path.name for path in target.parent.iterdir()) == ["settings.json"]


def test_save_uses_os_replace(target, monkeypatch):
    calls: list[tuple[Path, Path]] = []
    real_replace = settings_module.os.replace

    def spy(src, dst):
        calls.append((Path(src), Path(dst)))
        return real_replace(src, dst)

    monkeypatch.setattr(settings_module.os, "replace", spy)

    save(Settings(model="small"), target)

    assert calls == [(target.with_name("settings.json.tmp"), target)]
    assert target.is_file()


def test_save_fsyncs_before_replacing(target, monkeypatch):
    calls: list[str] = []
    real_fsync = settings_module.os.fsync
    real_replace = settings_module.os.replace

    def spy_fsync(fd):
        calls.append("fsync")
        return real_fsync(fd)

    def spy_replace(src, dst):
        calls.append("replace")
        return real_replace(src, dst)

    monkeypatch.setattr(settings_module.os, "fsync", spy_fsync)
    monkeypatch.setattr(settings_module.os, "replace", spy_replace)

    save(Settings(model="small"), target)

    # fsync must happen while the temporary file is still open, i.e. before
    # the rename that publishes it under the real name.
    assert calls == ["fsync", "replace"]
    assert target.is_file()


def test_failed_write_leaves_no_temporary_file(target, monkeypatch):
    def boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(settings_module.os, "replace", boom)

    with pytest.raises(OSError, match="disk full"):
        save(Settings(model="small"), target)

    assert list(target.parent.iterdir()) == []


def test_existing_file_survives_a_failed_write(target, monkeypatch):
    save(Settings(model="small"), target)

    def boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(settings_module.os, "replace", boom)

    with pytest.raises(OSError):
        save(Settings(model="tiny"), target)

    assert load(target).model == "small"


# -------------------------------------------------------------------- tolerance


def test_corrupt_json_returns_defaults(locale_env, target):
    target.parent.mkdir(parents=True)
    target.write_text('{"model": "small",', encoding="utf-8")

    assert load(target) == Settings()


def test_non_object_json_returns_defaults(locale_env, target):
    _write(target, ["small", "pl"])

    assert load(target) == Settings()


def test_partial_settings_fill_in_defaults(locale_env, target):
    _write(target, {"model": "small"})

    settings = load(target)

    assert settings.model == "small"
    assert settings.language == "en"
    assert settings.skip_silence is False


def test_extra_keys_are_ignored(locale_env, target):
    _write(
        target,
        {
            "model": "small",
            "language": "pl",
            "skip_silence": True,
            "threads": 12,
            "future_option": {"nested": True},
        },
    )

    assert load(target) == Settings(model="small", language="pl", skip_silence=True)


@pytest.mark.parametrize("language", ["xx", "", "AUTO", 5, None, ["pl"]])
def test_unknown_or_wrong_language_falls_back_to_the_locale(
    locale_env, target, language
):
    _write(target, {"language": language})

    assert load(target).language == "en"


@pytest.mark.parametrize("language", ["pl", "en", "auto"])
def test_known_languages_are_kept(locale_env, target, language):
    _write(target, {"language": language})

    assert load(target).language == language


@pytest.mark.parametrize("model", [5, None, "", [], {"name": "small"}])
def test_wrongly_typed_model_becomes_none(locale_env, target, model):
    _write(target, {"model": model})

    assert load(target).model is None


def test_unknown_model_name_is_kept_for_the_window_to_resolve(locale_env, target):
    _write(target, {"model": "some-local-build"})

    assert load(target).model == "some-local-build"


@pytest.mark.parametrize("value", ["yes", 1, 0, None, [], {}])
def test_wrongly_typed_skip_silence_becomes_false(locale_env, target, value):
    _write(target, {"skip_silence": value})

    assert load(target).skip_silence is False


def test_skip_silence_true_is_kept(locale_env, target):
    _write(target, {"skip_silence": True})

    assert load(target).skip_silence is True


def test_directory_instead_of_file_returns_defaults(locale_env, target):
    target.mkdir(parents=True)

    assert load(target) == Settings()


# ----------------------------------------------------------------- favourites


def test_favourites_default_to_empty(locale_env):
    assert Settings().favourites == []


def test_favourites_round_trip_in_order(target):
    original = Settings(language="pl", favourites=["uk", "ja", "yue"])

    save(original, target)

    assert load(target).favourites == ["uk", "ja", "yue"]


def test_favourites_are_saved_in_the_json(target):
    save(Settings(favourites=["uk", "ja"]), target)

    payload = json.loads(target.read_text(encoding="utf-8"))

    assert payload["favourites"] == ["uk", "ja"]


def test_a_settings_file_without_favourites_still_loads(locale_env, target):
    _write(target, {"model": "small", "language": "pl", "skip_silence": True})

    settings = load(target)

    assert settings == Settings(model="small", language="pl", skip_silence=True)
    assert settings.favourites == []


def test_unknown_favourites_are_dropped(locale_env, target):
    _write(target, {"favourites": ["uk", "xx", "english", "auto", "", "ja"]})

    assert load(target).favourites == ["uk", "ja"]


def test_duplicate_favourites_keep_the_first_position(locale_env, target):
    _write(target, {"favourites": ["ja", "uk", "ja", "uk"]})

    assert load(target).favourites == ["ja", "uk"]


@pytest.mark.parametrize("value", ["pl", 5, None, {}, True])
def test_a_non_list_favourites_value_becomes_empty(locale_env, target, value):
    _write(target, {"favourites": value})

    assert load(target).favourites == []


def test_non_string_favourites_are_dropped(locale_env, target):
    _write(target, {"favourites": [5, None, ["uk"], {"ja": True}, "pl"]})

    assert load(target).favourites == ["pl"]


def test_a_favourites_list_of_only_junk_becomes_empty(locale_env, target):
    _write(target, {"favourites": [5, None, "xx"]})

    assert load(target).favourites == []
