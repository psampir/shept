"""Persisted user settings: transcription model, language, silence handling.

GTK-free on purpose: this module is imported by `shept.app` but is also the
subject of unit tests that run without PyGObject installed. Reading is
deliberately forgiving — a missing file, a half-written one or a value the app
no longer understands must never stop Shept from starting, so `load()` always
returns a usable `Settings`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from shept.environment import default_language
from shept.languages import AUTO, LANGUAGES

#: Language codes Shept knows how to pass to whisper-cli, plus `auto`.
LANGUAGE_CODES = frozenset({AUTO.code, *(language.code for language in LANGUAGES)})


@dataclass
class Settings:
    """Everything Shept remembers between runs.

    `language` defaults to whatever the locale would have selected, so the
    first launch behaves exactly as it did before settings existed.
    """

    model: str | None = None
    language: str = field(default_factory=default_language)
    skip_silence: bool = False


def settings_path() -> Path:
    """Return `$XDG_CONFIG_HOME/shept/settings.json`, defaulting to ~/.config."""
    config_home = os.environ.get("XDG_CONFIG_HOME", "").strip()
    base = Path(config_home) if config_home else Path.home() / ".config"
    return base / "shept" / "settings.json"


def load(path: Path | None = None) -> Settings:
    """Read settings from `path`, falling back to defaults where needed.

    Every failure mode degrades instead of raising: an unreadable or malformed
    file, a JSON document that is not an object, unknown or wrongly typed
    values. Unknown model names are kept as they are — whether that model is
    installed is a question for the window, which falls back to the default
    selection when it is not.
    """
    target = Path(path) if path is not None else settings_path()
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return Settings()
    if not isinstance(raw, dict):
        return Settings()

    model = raw.get("model")
    language = raw.get("language")
    skip_silence = raw.get("skip_silence")
    defaults = Settings()
    return Settings(
        model=model if isinstance(model, str) and model else None,
        language=(
            language
            if isinstance(language, str) and language in LANGUAGE_CODES
            else defaults.language
        ),
        skip_silence=skip_silence if isinstance(skip_silence, bool) else False,
    )


def save(settings: Settings, path: Path | None = None) -> Path:
    """Write `settings` to `path` atomically and return that path.

    The JSON is written to `<name>.tmp` in the same directory and moved into
    place with `os.replace`, so a crash mid-write cannot leave a truncated
    settings file behind for the next start to trip over.
    """
    target = Path(path) if path is not None else settings_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    payload = {
        "model": settings.model,
        "language": settings.language,
        "skip_silence": settings.skip_silence,
    }
    try:
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, target)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise
    return target
