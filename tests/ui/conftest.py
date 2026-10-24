"""Fixtures for the UI tests: a real GTK 4 / libadwaita window, no display logic.

These tests are the only ones that import `gi`, so the whole directory is
skipped when PyGObject is not installed: the GTK-free `test` job keeps proving
that every other module stays importable without GTK, and the `ui-test` job in
CI installs GTK and runs exactly this directory under Xvfb.

Nothing here calls `application.run()`: the widgets are constructed directly,
which needs a display (Xvfb in CI) but neither a message bus nor a main loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

#: `gi` comes from the system, never from pip. Skipping here (module level, so
#: before any GTK import below) is what keeps the GTK-free job green: pytest
#: reports the whole directory as skipped instead of failing on the import.
gi = pytest.importorskip("gi")
gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from shept.app import SheptApp, SheptWindow  # noqa: E402

#: One application id for the whole session: GtkApplication is a singleton per
#: process, and the tests only need somewhere to hang the windows on. The real
#: class is used, not a plain `Adw.Application`, because the About window is
#: reached through an action that only `SheptApp` installs; the id stays a test
#: one so a Shept the developer has running does not win the bus name.
TEST_APP_ID = "io.github.psampir.Shept.Tests"


@dataclass(frozen=True)
class Sandbox:
    """The temporary directories one test may write to."""

    root: Path
    config: Path
    data: Path
    models: Path
    home: Path


@pytest.fixture(scope="session")
def app():
    """A registered `SheptApp` without a main loop.

    `register()` emits `startup`, which is what lets windows be added, and it
    succeeds without a session bus: the tests must not need one.
    """
    application = SheptApp(application_id=TEST_APP_ID)
    assert application.register(None)
    return application


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Sandbox:
    """Point every directory Shept writes to or searches at `tmp_path`.

    `SHEPT_MODEL_DIR` is the first model search path, and `$HOME` is isolated
    too: `shept.environment` searches `~/.local/share/whisper-cpp/models` and
    `~/.cache/whisper.cpp/models`, so the developer's own models would otherwise
    leak into the window and make the model assertions depend on the machine.
    """
    box = Sandbox(
        root=tmp_path,
        config=tmp_path / "config",
        data=tmp_path / "data",
        models=tmp_path / "models",
        home=tmp_path / "home",
    )
    for directory in (box.config, box.data, box.models, box.home):
        directory.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(box.config))
    monkeypatch.setenv("XDG_DATA_HOME", str(box.data))
    monkeypatch.setenv("SHEPT_MODEL_DIR", str(box.models))
    monkeypatch.setenv("HOME", str(box.home))
    return box


@pytest.fixture
def make_window(app, sandbox: Sandbox):
    """Build `SheptWindow`s, destroying every one of them at the end.

    A factory rather than a single fixture because some tests must patch the
    environment (a missing tool, extra model files) *before* the window reads
    it: everything the window snapshots is read in `__init__`.
    """
    created: list[SheptWindow] = []

    def build() -> SheptWindow:
        window = SheptWindow(app)
        created.append(window)
        return window

    yield build

    for window in created:
        window.destroy()


@pytest.fixture
def window(make_window) -> SheptWindow:
    """A plain window built in the isolated environment."""
    return make_window()
