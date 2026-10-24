"""UI regressions for `shept.app`, run against a real GTK 4 / libadwaita.

Everything here is a defect that shipped in 0.1.0-0.1.2 and could not be
caught by the GTK-free suite, because the bug was in the wiring, not in the
logic underneath it:

* the language rows were `Adw.ActionRow`s appended straight to a
  `Gtk.ListBox`, so they rendered but were never real rows: clicks did
  nothing while Enter still worked;
* the queue/VAD/model rows only exist as widgets.

The assertions deliberately go through the widgets (a row's own activation
path, a star's `set_active`, a dropdown's `set_selected`), never through the
signals the app connects: emitting `row-activated` by hand is exactly what
failed to catch the language-picker bug.
"""

from __future__ import annotations

import json
import shutil

from gi.repository import Adw, Gtk

from shept.models import VAD_SPEC, VadState, human_size

# --------------------------------------------------------------- widget helpers


def _descendants(widget: Gtk.Widget):
    """Every widget below `widget`, depth first."""
    child = widget.get_first_child()
    while child is not None:
        yield child
        yield from _descendants(child)
        child = child.get_next_sibling()


def _picker_rows(picker) -> list[Gtk.ListBoxRow]:
    """The rows currently in the picker's list box, in order."""
    rows: list[Gtk.ListBoxRow] = []
    child = picker.listbox.get_first_child()
    while child is not None:
        rows.append(child)
        child = child.get_next_sibling()
    return rows


def _picker_row(picker, code: str) -> Gtk.ListBoxRow:
    """The picker row for language `code`, found by what the app will do."""
    for row in _picker_rows(picker):
        if picker._codes.get(row) == code:
            return row
    raise AssertionError(f"no row for {code!r} in the picker")


def _star_button(row: Gtk.ListBoxRow) -> Gtk.ToggleButton:
    """The favourite toggle inside one picker row."""
    for widget in _descendants(row):
        if isinstance(widget, Gtk.ToggleButton):
            return widget
    raise AssertionError("picker row has no star button")


def _settings_payload(sandbox) -> dict:
    path = sandbox.config / "shept" / "settings.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _pretend_tools_are_installed(monkeypatch) -> None:
    """Make `shutil.which` find whisper-cli, ffmpeg and ffprobe.

    The CI job installs GTK, not whisper.cpp, so without this every window
    would report a missing tool and Transcribe could never become available.
    """
    monkeypatch.setattr(shutil, "which", lambda name, *a, **k: f"/usr/bin/{name}")


# ------------------------------------------------------------------ 1. language


def test_language_picker_rows_are_real_listbox_rows(window, sandbox):
    """The 0.1.2 bug: an ActionRow placed directly in the ListBox is not a row.

    Gtk.ListBox only emits `row-activated` for actual `Gtk.ListBoxRow`
    children, so the picker must wrap each ActionRow. Clicking Polish through
    the row's own activation path must select it, update the window's row and
    persist the choice: the whole chain that was dead for two releases.
    """
    window._on_choose_language(window.lang_btn)
    picker = window._lang_picker
    assert picker is not None

    rows = _picker_rows(picker)
    assert rows, "the picker opened without any rows"
    # The exact defect: every direct child of the ListBox is a real row.
    assert all(isinstance(row, Gtk.ListBoxRow) for row in rows)
    # ... and each one still shows the libadwaita row the user clicks on.
    assert all(isinstance(row.get_child(), Adw.ActionRow) for row in rows)

    polish = _picker_row(picker, "pl")
    assert polish.get_child().get_title() == "Polish"
    # Activate the row the way a click does, not by emitting `row-activated`.
    polish.activate()

    assert window._language == "pl"
    assert window.lang_row.get_subtitle() == "Polish"
    # The picker closed itself, so the real activation handler ran.
    assert window._lang_picker is None
    assert _settings_payload(sandbox)["language"] == "pl"


def test_enter_in_the_picker_picks_the_first_match_after_filtering(window, sandbox):
    """Enter still works on a filtered list: one match, activated from the entry."""
    window._on_choose_language(window.lang_btn)
    picker = window._lang_picker
    assert picker is not None

    picker.search_entry.set_text("pol")
    assert picker._codes[_picker_rows(picker)[0]] == "pl"
    assert picker.hint.get_text().startswith("1 match")

    # Enter in the search entry is the documented shortcut; the entry emits
    # `activate` for a real Return key press.
    picker.search_entry.emit("activate")

    assert window._language == "pl"
    assert window.lang_row.get_subtitle() == "Polish"
    assert _settings_payload(sandbox)["language"] == "pl"


def test_starring_a_language_updates_the_window_and_is_persisted(window, sandbox):
    """A star click reaches the window's favourites and survives as JSON."""
    window._on_choose_language(window.lang_btn)
    picker = window._lang_picker
    assert picker is not None

    star = _star_button(_picker_row(picker, "pl"))
    assert star.get_active() is False
    assert star.get_icon_name() == "non-starred-symbolic"

    star.set_active(True)  # what clicking the toggle does

    assert "pl" in window._favourites
    assert star.get_icon_name() == "starred-symbolic"
    assert "pl" in picker._favourites
    assert _settings_payload(sandbox)["favourites"] == ["pl"]

    # Un-starring removes it again, and that is persisted too.
    star.set_active(False)

    assert "pl" not in window._favourites
    assert _settings_payload(sandbox)["favourites"] == []


# --------------------------------------------------------------------- 2. models


def test_the_model_dropdown_lists_the_files_and_follows_the_selection(
    make_window, sandbox
):
    """The dropdown is built from `discover_models` and drives `_selected_model`."""
    first = sandbox.models / "ggml-aaa-test.bin"
    second = sandbox.models / "ggml-aab-test.bin"
    first.write_bytes(b"not a real model")
    second.write_bytes(b"not a real model")

    window = make_window()

    model = window.model_row.get_model()
    assert isinstance(model, Gtk.StringList)
    names = [model.get_string(index) for index in range(model.get_n_items())]
    assert "aaa-test" in names
    assert "aab-test" in names
    assert names.index("aaa-test") < names.index("aab-test")

    # Select the second fake model through the dropdown, as a user would.
    window.model_row.set_selected(names.index("aab-test"))

    assert window._selected_model() == second
    assert _settings_payload(sandbox)["model"] == "aab-test"


# ------------------------------------------------------------------ 3. VAD row


def test_vad_row_missing_offers_the_download(window):
    window._set_vad_state(VadState.MISSING)

    assert window.vad_hint.get_visible()
    assert window.vad_hint.get_title() == "Silero VAD is not installed"
    assert window.vad_hint.get_subtitle() == (
        f"Download it ({human_size(VAD_SPEC.size_bytes)}) to skip silence."
    )
    assert window.vad_download_btn.get_label() == "Download"
    assert window.vad_download_btn.get_sensitive()
    assert window.vad_row.get_active() is False
    assert window.vad_row.get_sensitive()


def test_vad_row_downloading_shows_progress_and_locks_the_switch(window):
    window._set_vad_state(VadState.DOWNLOADING, "42%")

    assert window.vad_hint.get_visible()
    assert window.vad_hint.get_title() == "Silero VAD is not installed"
    assert window.vad_hint.get_subtitle() == "Downloading Silero VAD: 42%"
    assert window.vad_download_btn.get_label() == "Downloading…"
    assert not window.vad_download_btn.get_sensitive()
    assert not window.vad_row.get_sensitive()
    assert window.vad_row.get_active() is False


def test_vad_row_failure_drops_every_downloading_label(window):
    """A failed download used to keep its "Downloading…" text behind."""
    window._set_vad_state(VadState.DOWNLOADING, "42%")
    window._set_vad_state(VadState.FAILED, "no network connection")

    assert window.vad_hint.get_visible()
    assert window.vad_hint.get_title() == "Silero VAD is not installed"
    assert window.vad_hint.get_subtitle() == "Download failed: no network connection."
    assert "Downloading" not in window.vad_hint.get_subtitle()
    assert window.vad_download_btn.get_label() == "Retry"
    assert "Downloading" not in window.vad_download_btn.get_label()
    assert window.vad_download_btn.get_sensitive()
    assert window.vad_row.get_sensitive()
    assert window.vad_row.get_active() is False


def test_vad_row_installed_hides_the_hint_and_turns_the_switch_on(window):
    window._set_vad_state(VadState.INSTALLED)

    assert not window.vad_hint.get_visible()
    assert window.vad_hint.get_title() == "Silero VAD is installed"
    assert window.vad_hint.get_subtitle() == ""
    assert window.vad_row.get_active() is True
    assert window.vad_row.get_sensitive()


# -------------------------------------------------------------- 4. missing tool


def test_missing_whisper_cli_shows_the_banner_and_disables_transcribe(
    make_window, monkeypatch
):
    """Without whisper-cli the window explains itself instead of failing later."""
    real_which = shutil.which

    def which(name: str, *args, **kwargs):
        if name == "whisper-cli":
            return None
        return real_which(name, *args, **kwargs)

    monkeypatch.setattr(shutil, "which", which)

    window = make_window()

    banners = [
        widget for widget in _descendants(window) if isinstance(widget, Adw.Banner)
    ]
    assert len(banners) == 1
    assert banners[0].get_revealed() is True
    assert "whisper-cli" in banners[0].get_title()
    assert "whisper-cli" in window.run_btn.get_tooltip_text()
    assert window.run_btn.get_sensitive() is False


# ----------------------------------------------------------------- 5. the queue


def test_adding_two_files_creates_two_rows_and_unblocks_transcribe(
    make_window, sandbox, monkeypatch
):
    """Rows are mirrored from the queue, and a model plus a file unblock Run."""
    _pretend_tools_are_installed(monkeypatch)
    (sandbox.models / "ggml-base.bin").write_bytes(b"not a real model")
    first = sandbox.root / "first.mp3"
    second = sandbox.root / "second.mp3"
    first.write_bytes(b"audio")
    second.write_bytes(b"audio")

    window = make_window()

    # Nothing queued yet, even with a model installed: the blocker names the
    # missing file rather than blaming the model.
    assert window._transcribe_blocker() == "Add an audio file first."

    window.add_files([first, second])

    rows = []
    child = window.job_list.get_first_child()
    while child is not None:
        rows.append(child)
        child = child.get_next_sibling()
    assert len(rows) == 2
    assert [row.get_title() for row in rows] == ["first.mp3", "second.mp3"]
    assert window._transcribe_blocker() == ""
    assert window.run_btn.get_sensitive() is True
