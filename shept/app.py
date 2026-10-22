"""Shept — a GTK4 + libadwaita frontend for whisper-cli.

This is the only module that imports `gi`; the rest of the package is
GTK-free so it can be tested without PyGObject.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import gi

# gi.require_version() must run before the gi.repository import, so the shept
# imports below are pushed past the top of the module.
gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gio, GLib, Gtk  # noqa: E402

from shept import APP_ID  # noqa: E402
from shept.environment import (  # noqa: E402
    LANGUAGES,
    MODELS_URL,
    REQUIRED_TOOLS,
    default_language,
    detect_backend,
    discover_models,
    install_hint,
    model_display_name,
    model_search_paths,
)
from shept.transcribe import (  # noqa: E402
    build_ffmpeg_cmd,
    build_whisper_cmd,
    parse_progress_line,
    parse_segment,
    ts_to_seconds,
)


class SheptWindow(Adw.ApplicationWindow):
    def __init__(self, app: Adw.Application) -> None:
        super().__init__(application=app, title="Shept")
        self.set_default_size(820, 640)

        self._proc: subprocess.Popen[str] | None = None
        self._tmpdir: tempfile.TemporaryDirectory | None = None
        self._input_file: Path | None = None
        self._missing_tools = [
            name for name in REQUIRED_TOOLS if shutil.which(name) is None
        ]

        toolbar = Adw.ToolbarView()
        self.set_content(toolbar)

        header = Adw.HeaderBar()
        toolbar.add_top_bar(header)

        self.copy_btn = Gtk.Button.new_from_icon_name("edit-copy-symbolic")
        self.copy_btn.set_tooltip_text("Copy transcript")
        self.copy_btn.set_sensitive(False)
        self.copy_btn.connect("clicked", self._on_copy)
        header.pack_end(self.copy_btn)

        self.save_btn = Gtk.Button.new_from_icon_name("document-save-symbolic")
        self.save_btn.set_tooltip_text("Save transcript as .txt")
        self.save_btn.set_sensitive(False)
        self.save_btn.connect("clicked", self._on_save)
        header.pack_end(self.save_btn)

        outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        outer.set_margin_top(12)
        outer.set_margin_bottom(12)
        outer.set_margin_start(12)
        outer.set_margin_end(12)
        toolbar.set_content(outer)

        if self._missing_tools:
            names = " and ".join(self._missing_tools)
            hint = install_hint(self._missing_tools[0])
            banner = Adw.Banner(title=f"{names} not found — {hint}")
            banner.set_tooltip_text(f"Missing: {names}")
            banner.set_revealed(True)
            outer.append(banner)

        clamp = Adw.Clamp(maximum_size=720)
        outer.append(clamp)

        controls = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        clamp.set_child(controls)

        group = Adw.PreferencesGroup()
        controls.append(group)

        self.file_row = Adw.ActionRow(title="Audio file", subtitle="No file selected")
        choose_btn = Gtk.Button(label="Choose…")
        choose_btn.set_valign(Gtk.Align.CENTER)
        choose_btn.connect("clicked", self._on_choose_file)
        self.file_row.add_suffix(choose_btn)
        self.file_row.set_activatable_widget(choose_btn)
        group.add(self.file_row)

        models = discover_models()
        self.model_row = Adw.ComboRow(title="Model")
        model_strings = Gtk.StringList()
        if models:
            for m in models:
                model_strings.append(model_display_name(m))
        else:
            model_strings.append("(no models found)")
        self.model_row.set_model(model_strings)
        self._models = models
        for i, m in enumerate(models):
            if model_display_name(m) == "large-v3-turbo":
                self.model_row.set_selected(i)
                break
        group.add(self.model_row)

        if not models:
            controls.append(self._models_hint())

        self.lang_row = Adw.ComboRow(title="Language")
        lang_strings = Gtk.StringList()
        for _code, label in LANGUAGES:
            lang_strings.append(label)
        self.lang_row.set_model(lang_strings)
        lang_default = default_language()
        self.lang_row.set_selected(
            [code for code, _label in LANGUAGES].index(lang_default)
        )
        group.add(self.lang_row)

        self.translate_row = Adw.SwitchRow(
            title="Translate to English",
            subtitle="Output English translation instead of source-language transcript",
        )
        group.add(self.translate_row)

        self.run_btn = Gtk.Button(label="Transcribe")
        self.run_btn.add_css_class("suggested-action")
        self.run_btn.add_css_class("pill")
        self.run_btn.set_halign(Gtk.Align.CENTER)
        self.run_btn.connect("clicked", self._on_run)
        controls.append(self.run_btn)
        self._update_run_state()

        self.progress = Gtk.ProgressBar()
        self.progress.set_show_text(True)
        self.progress.set_visible(False)
        controls.append(self.progress)

        scroller = Gtk.ScrolledWindow(vexpand=True)
        scroller.set_min_content_height(280)
        outer.append(scroller)

        self.buffer = Gtk.TextBuffer()
        self.text_view = Gtk.TextView(buffer=self.buffer)
        self.text_view.set_editable(False)
        self.text_view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        self.text_view.set_top_margin(8)
        self.text_view.set_bottom_margin(8)
        self.text_view.set_left_margin(8)
        self.text_view.set_right_margin(8)
        scroller.set_child(self.text_view)

        self.status = Gtk.Label(xalign=0)
        self.status.add_css_class("dim-label")
        backend = detect_backend()
        if backend == "not installed":
            status_text = "whisper.cpp backend: whisper-cli not found in PATH"
        elif backend == "CPU":
            status_text = "whisper.cpp backend: CPU only"
        else:
            status_text = f"whisper.cpp backend: {backend}"
        self.status.set_text(status_text)
        outer.append(self.status)

    def _models_hint(self) -> Gtk.Widget:
        """Explain where models were searched and how to get one."""
        searched = "\n".join(f"  {path}" for path in model_search_paths())
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        label = Gtk.Label(
            label=(
                "No Whisper model found. Searched:\n"
                f"{searched}\n"
                "Recommended: large-v3-turbo (~1.5 GB)."
            ),
            xalign=0,
        )
        label.set_wrap(True)
        label.add_css_class("dim-label")
        box.append(label)
        link = Gtk.LinkButton.new_with_label(MODELS_URL, "Get models")
        link.set_halign(Gtk.Align.START)
        box.append(link)
        return box

    def _transcribe_blocker(self) -> str:
        """Return why Transcribe is unavailable, or "" when it is available."""
        if self._missing_tools:
            names = " and ".join(self._missing_tools)
            return f"Missing {names} — {install_hint(self._missing_tools[0])}"
        if not self._models:
            return "No Whisper model found — see the hint below the model row."
        if self._input_file is None:
            return "Choose an audio file first."
        return ""

    def _update_run_state(self) -> None:
        blocker = self._transcribe_blocker()
        self.run_btn.set_sensitive(not blocker)
        self.run_btn.set_tooltip_text(blocker or "Transcribe the selected audio file.")

    def _on_choose_file(self, _btn: Gtk.Button) -> None:
        dialog = Gtk.FileDialog(title="Select audio file")
        audio_filter = Gtk.FileFilter()
        audio_filter.set_name("Audio / video files")
        for ext in (
            "mp3",
            "wav",
            "m4a",
            "ogg",
            "opus",
            "flac",
            "aac",
            "mp4",
            "webm",
            "mkv",
            "mov",
            "avi",
            "wmv",
            "ts",
            "m4v",
            "3gp",
        ):
            audio_filter.add_pattern(f"*.{ext}")
            audio_filter.add_pattern(f"*.{ext.upper()}")
        all_filter = Gtk.FileFilter()
        all_filter.set_name("All files")
        all_filter.add_pattern("*")
        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(audio_filter)
        filters.append(all_filter)
        dialog.set_filters(filters)
        dialog.set_default_filter(audio_filter)
        dialog.open(self, None, self._on_file_chosen)

    def _on_file_chosen(self, dialog: Gtk.FileDialog, result: Gio.AsyncResult) -> None:
        try:
            f = dialog.open_finish(result)
        except GLib.Error:
            return
        self.set_input_file(Path(f.get_path()))

    def set_input_file(self, path: Path) -> None:
        """Select `path` as the input to transcribe.

        Used both by the file dialog and by files opened from the command line
        or the desktop, so there is a single code path for choosing an input.
        """
        self._input_file = path
        self.file_row.set_subtitle(str(path))
        self._update_run_state()

    def _selected_model(self) -> Path | None:
        if not self._models:
            return None
        return self._models[self.model_row.get_selected()]

    def _selected_lang(self) -> str:
        return LANGUAGES[self.lang_row.get_selected()][0]

    def _on_run(self, _btn: Gtk.Button) -> None:
        if self._input_file is None:
            return
        model = self._selected_model()
        if model is None:
            return
        self.run_btn.set_sensitive(False)
        self.copy_btn.set_sensitive(False)
        self.save_btn.set_sensitive(False)
        self.buffer.set_text("")
        self.progress.set_visible(True)
        self.progress.set_fraction(0.0)
        self.progress.set_text("Decoding audio…")
        self.status.set_text("Extracting audio with ffmpeg…")
        threading.Thread(
            target=self._worker,
            args=(
                self._input_file,
                model,
                self._selected_lang(),
                self.translate_row.get_active(),
            ),
            daemon=True,
        ).start()

    def _worker(
        self,
        input_path: Path,
        model: Path,
        lang: str,
        translate: bool,
    ) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(prefix="shept-")
        wav_path = Path(self._tmpdir.name) / "input.wav"

        input_duration = self._probe_duration(input_path)
        ffmpeg_cmd = build_ffmpeg_cmd(input_path, wav_path)
        ff = subprocess.Popen(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert ff.stdout is not None
        for line in ff.stdout:
            secs = parse_progress_line(line)
            if secs is None:
                continue
            if input_duration > 0:
                frac = max(0.0, min(1.0, secs / input_duration))
                GLib.idle_add(
                    self._set_progress,
                    frac,
                    f"Decoding audio… {int(frac * 100)}%",
                )
        if ff.wait() != 0:
            err = (ff.stderr.read() if ff.stderr else "") or "unknown error"
            GLib.idle_add(self._finish, False, f"ffmpeg failed: {err.strip()}")
            return

        GLib.idle_add(self._set_progress, 0.0, "Loading model…")
        GLib.idle_add(self.status.set_text, "Running whisper-cli…")

        duration = self._probe_duration(wav_path)
        cmd = build_whisper_cmd(
            model,
            wav_path,
            lang,
            min(8, os.cpu_count() or 4),
            translate,
        )

        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            self._handle_line(line.rstrip("\n"), duration)
        ret = self._proc.wait()
        self._proc = None
        GLib.idle_add(
            self._finish,
            ret == 0,
            "Done." if ret == 0 else f"whisper-cli exited with {ret}",
        )

    @staticmethod
    def _probe_duration(media: Path) -> float:
        try:
            out = subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=nw=1:nk=1",
                    str(media),
                ],
                text=True,
            ).strip()
            return float(out)
        except Exception:
            return 0.0

    def _handle_line(self, line: str, duration: float) -> None:
        segment = parse_segment(line)
        if segment is None:
            return
        end_ts, text = segment
        GLib.idle_add(self._append_text, text + "\n")
        if duration > 0:
            secs = ts_to_seconds(end_ts)
            frac = max(0.0, min(1.0, secs / duration))
            GLib.idle_add(self._set_progress, frac, end_ts)

    def _append_text(self, text: str) -> bool:
        end = self.buffer.get_end_iter()
        self.buffer.insert(end, text)
        return False

    def _set_progress(self, frac: float, label: str) -> bool:
        self.progress.set_fraction(frac)
        self.progress.set_text(f"{label}  ({int(frac * 100)}%)")
        return False

    def _finish(self, _ok: bool, msg: str) -> bool:
        self.progress.set_visible(False)
        self._update_run_state()
        has_text = self.buffer.get_char_count() > 0
        self.copy_btn.set_sensitive(has_text)
        self.save_btn.set_sensitive(has_text)
        self.status.set_text(msg)
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None
        return False

    def _transcript_text(self) -> str:
        start = self.buffer.get_start_iter()
        end = self.buffer.get_end_iter()
        return self.buffer.get_text(start, end, False)

    def _on_copy(self, _btn: Gtk.Button) -> None:
        clipboard = self.get_clipboard()
        clipboard.set(self._transcript_text())
        self.status.set_text("Copied to clipboard.")

    def _on_save(self, _btn: Gtk.Button) -> None:
        dialog = Gtk.FileDialog(title="Save transcript")
        suggested = "transcript.txt"
        if self._input_file is not None:
            suggested = self._input_file.with_suffix(".txt").name
        dialog.set_initial_name(suggested)
        dialog.save(self, None, self._on_save_chosen)

    def _on_save_chosen(self, dialog: Gtk.FileDialog, result: Gio.AsyncResult) -> None:
        try:
            f = dialog.save_finish(result)
        except GLib.Error:
            return
        Path(f.get_path()).write_text(self._transcript_text(), encoding="utf-8")
        self.status.set_text(f"Saved: {f.get_path()}")


class SheptApp(Adw.Application):
    def __init__(self) -> None:
        # HANDLES_OPEN is what makes the MimeType= entries in the desktop file
        # work: double-clicking a media file, or running `shept file.mp4`, is
        # delivered here through do_open() instead of being ignored.
        super().__init__(
            application_id=APP_ID,
            flags=Gio.ApplicationFlags.HANDLES_OPEN,
        )

    def do_activate(self) -> None:
        win = self.get_active_window() or SheptWindow(self)
        win.present()

    def do_open(self, files, n_files: int, _hint: str) -> None:
        """Load a file passed on the command line or from the desktop."""
        win = self.get_active_window() or SheptWindow(self)
        win.present()
        if not files:
            return

        gfile = files[0]
        path_str = gfile.get_path() if gfile is not None else None
        if path_str is None:
            uri = gfile.get_uri() if gfile is not None else ""
            win.status.set_text(f"Cannot open {uri}: not a local file.")
            return

        path = Path(path_str)
        if not path.is_file():
            win.status.set_text(f"Cannot open {path}: no such file.")
            return
        win.set_input_file(path)


def main() -> int:
    # Missing tools and models are reported in the window instead of aborting
    # here: a desktop-menu launch has nowhere to show a stderr message.
    GLib.set_application_name("Shept")
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    return SheptApp().run(sys.argv)
