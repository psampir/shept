"""Persisted user settings: model, language, silence handling, model folders.

GTK-free on purpose: this module is imported by `shept.app` but is also the
subject of unit tests that run without PyGObject installed. Reading is
deliberately forgiving: a missing file, a half-written one or a value the app
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

#: Codes that can be favourites. `auto` cannot: it is a mode, not a language,
#: and starring it would only take a slot that belongs to a real entry.
FAVOURITE_CODES = frozenset(language.code for language in LANGUAGES)


def _favourites(value: object) -> list[str]:
    """Tolerant reader for the favourites list.

    Anything a picker cannot show is dropped: a value that is not a list, a
    non-string entry, an unknown code, a duplicate. Order is preserved because
    the user chose it, and the first occurrence wins.
    """
    if not isinstance(value, list):
        return []
    codes: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        code = item.strip().lower()
        if code in FAVOURITE_CODES and code not in codes:
            codes.append(code)
    return codes


def _model_dirs(value: object) -> list[str]:
    """Tolerant reader for the extra model folders, as absolute paths.

    A folder Shept cannot search is worse than no folder at all, so anything
    that would silently resolve elsewhere is dropped: a value that is not a
    list, a non-string entry, an empty string, a relative path. `~` is expanded
    first, because that is the one relative spelling with a useful meaning.
    Order is the user's, and the first occurrence of a duplicate wins.
    """
    if not isinstance(value, list):
        return []
    dirs: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            continue
        path = Path(item.strip()).expanduser()
        if not path.is_absolute():
            continue
        text = str(path)
        if text not in dirs:
            dirs.append(text)
    return dirs


@dataclass
class Settings:
    """Everything Shept remembers between runs.

    `language` defaults to whatever the locale would have selected, so the
    first launch behaves exactly as it did before settings existed.
    """

    model: str | None = None
    language: str = field(default_factory=default_language)
    skip_silence: bool = False
    #: Language codes the picker shows first, in the order the user starred
    #: them. Kept as a plain list so the order survives a save/load round trip.
    favourites: list[str] = field(default_factory=list)
    #: Extra model folders the user added in the model manager, in their order.
    #: Stored as strings so the JSON stays readable and hand-editable.
    model_dirs: list[str] = field(default_factory=list)


def settings_path() -> Path:
    """Return `$XDG_CONFIG_HOME/shept/settings.json`, defaulting to ~/.config."""
    config_home = os.environ.get("XDG_CONFIG_HOME", "").strip()
    base = Path(config_home) if config_home else Path.home() / ".config"
    return base / "shept" / "settings.json"


def load(path: Path | None = None) -> Settings:
    """Read settings from `path`, falling back to defaults where needed.

    Every failure mode degrades instead of raising: an unreadable or malformed
    file, a JSON document that is not an object, unknown or wrongly typed
    values. Unknown model names are kept as they are; whether that model is
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
        favourites=_favourites(raw.get("favourites")),
        model_dirs=_model_dirs(raw.get("model_dirs")),
    )


def save(settings: Settings, path: Path | None = None) -> Path:
    """Write `settings` to `path` atomically and return that path.

    The JSON is written to `<name>.tmp` in the same directory, flushed and
    fsynced, then moved into place with `os.replace`, so a crash mid-write
    cannot leave a truncated settings file behind for the next start to trip
    over, and a crash right after the replace cannot lose the new content
    either.
    """
    target = Path(path) if path is not None else settings_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    payload = {
        "model": settings.model,
        "language": settings.language,
        "skip_silence": settings.skip_silence,
        "favourites": list(settings.favourites),
        "model_dirs": list(settings.model_dirs),
    }
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2) + "\n")
            handle.flush()
            # The rename is atomic, but the bytes it points at may still sit in
            # the page cache; fsync is what makes the "crash-safe" claim true.
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise
    return target
