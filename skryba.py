#!/usr/bin/env python3
"""Skryba — a GTK4 + libadwaita frontend for whisper-cli."""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gio, GLib, Gtk  # noqa: E402


APP_ID = "io.github.psampir.Skryba"
MODEL_GLOB = "ggml-*.bin"

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

LINE_RE = re.compile(
    r"^\[(\d\d:\d\d:\d\d\.\d{3})\s*-->\s*(\d\d:\d\d:\d\d\.\d{3})\]\s+(.*)$"
)

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
            text=True,
            timeout=5,
        )
        if ldconfig.returncode == 0:
            return [
                ln for ln in ldconfig.stdout.splitlines() if "libggml-" in ln
            ]
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
    data_home = os.environ.get("XDG_DATA_HOME", "")
    data_dir = Path(data_home) if data_home else Path.home() / ".local" / "share"
    override = os.environ.get("SKRYBA_MODEL_DIR", "")

    wanted: list[Path] = []
    if override:
        wanted.append(Path(override).expanduser())
    wanted.extend(
        [
            data_dir / "skryba" / "models",
            data_dir / "whisper-cpp" / "models",
            Path.home() / ".cache" / "whisper.cpp" / "models",
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


class SkrybaWindow(Adw.ApplicationWindow):
    def __init__(self, app: Adw.Application) -> None:
        super().__init__(application=app, title="Skryba")
        self.set_default_size(820, 640)

        self._proc: subprocess.Popen[str] | None = None
        self._tmpdir: tempfile.TemporaryDirectory | None = None
        self._input_file: Path | None = None

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

        clamp = Adw.Clamp(maximum_size=720)
        outer.append(clamp)

        controls = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        clamp.set_child(controls)

        group = Adw.PreferencesGroup()
        controls.append(group)

        self.file_row = Adw.ActionRow(
            title="Audio file", subtitle="No file selected"
        )
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

        self.lang_row = Adw.ComboRow(title="Language")
        lang_strings = Gtk.StringList()
        for _code, label in LANGUAGES:
            lang_strings.append(label)
        self.lang_row.set_model(lang_strings)
        self.lang_row.set_selected(0)
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
        self.run_btn.set_sensitive(False)
        self.run_btn.connect("clicked", self._on_run)
        controls.append(self.run_btn)

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

    def _on_choose_file(self, _btn: Gtk.Button) -> None:
        dialog = Gtk.FileDialog(title="Select audio file")
        audio_filter = Gtk.FileFilter()
        audio_filter.set_name("Audio / video files")
        for ext in (
            "mp3", "wav", "m4a", "ogg", "opus", "flac", "aac",
            "mp4", "webm", "mkv", "mov", "avi", "wmv", "ts", "m4v", "3gp",
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
        path = Path(f.get_path())
        self._input_file = path
        self.file_row.set_subtitle(str(path))
        self.run_btn.set_sensitive(bool(self._models))

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
            args=(self._input_file, model, self._selected_lang(),
                  self.translate_row.get_active()),
            daemon=True,
        ).start()

    def _worker(
        self,
        input_path: Path,
        model: Path,
        lang: str,
        translate: bool,
    ) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(prefix="skryba-")
        wav_path = Path(self._tmpdir.name) / "input.wav"

        input_duration = self._probe_duration(input_path)
        ffmpeg_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-nostats", "-progress", "pipe:1",
            "-i", str(input_path),
            "-vn", "-ac", "1", "-ar", "16000",
            str(wav_path), "-y",
        ]
        ff = subprocess.Popen(
            ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        assert ff.stdout is not None
        for line in ff.stdout:
            line = line.strip()
            if line.startswith("out_time_ms="):
                try:
                    secs = int(line.split("=", 1)[1]) / 1_000_000.0
                except ValueError:
                    continue
                if input_duration > 0:
                    frac = max(0.0, min(1.0, secs / input_duration))
                    GLib.idle_add(
                        self._set_progress, frac,
                        f"Decoding audio… {int(frac * 100)}%",
                    )
        if ff.wait() != 0:
            err = (ff.stderr.read() if ff.stderr else "") or "unknown error"
            GLib.idle_add(self._finish, False, f"ffmpeg failed: {err.strip()}")
            return

        GLib.idle_add(self._set_progress, 0.0, "Loading model…")
        GLib.idle_add(self.status.set_text, "Running whisper-cli…")

        duration = self._probe_duration(wav_path)
        cmd = [
            "whisper-cli",
            "-m", str(model),
            "-l", lang,
            "-t", str(min(8, os.cpu_count() or 4)),
            "-f", str(wav_path),
        ]
        if translate:
            cmd.append("--translate")

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
        GLib.idle_add(self._finish, ret == 0,
                      "Done." if ret == 0 else f"whisper-cli exited with {ret}")

    @staticmethod
    def _probe_duration(media: Path) -> float:
        try:
            out = subprocess.check_output(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=nw=1:nk=1", str(media)],
                text=True,
            ).strip()
            return float(out)
        except Exception:
            return 0.0

    def _handle_line(self, line: str, duration: float) -> None:
        m = LINE_RE.match(line)
        if not m:
            return
        end_ts, text = m.group(2), m.group(3).strip()
        GLib.idle_add(self._append_text, text + "\n")
        if duration > 0:
            secs = self._ts_to_seconds(end_ts)
            frac = max(0.0, min(1.0, secs / duration))
            GLib.idle_add(self._set_progress, frac, end_ts)

    @staticmethod
    def _ts_to_seconds(ts: str) -> float:
        h, m, s = ts.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)

    def _append_text(self, text: str) -> bool:
        end = self.buffer.get_end_iter()
        self.buffer.insert(end, text)
        return False

    def _set_progress(self, frac: float, label: str) -> bool:
        self.progress.set_fraction(frac)
        self.progress.set_text(f"{label}  ({int(frac * 100)}%)")
        return False

    def _finish(self, ok: bool, msg: str) -> bool:
        self.progress.set_visible(False)
        self.run_btn.set_sensitive(True)
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


class SkrybaApp(Adw.Application):
    def __init__(self) -> None:
        super().__init__(application_id=APP_ID)

    def do_activate(self) -> None:
        win = self.get_active_window() or SkrybaWindow(self)
        win.present()


def main() -> int:
    if shutil.which("whisper-cli") is None:
        print("whisper-cli not found in PATH", file=sys.stderr)
        return 1
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found in PATH", file=sys.stderr)
        return 1
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    return SkrybaApp().run(sys.argv)


if __name__ == "__main__":
    sys.exit(main())
