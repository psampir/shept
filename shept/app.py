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
from collections.abc import Callable, Iterable
from pathlib import Path

import gi

# gi.require_version() must run before the gi.repository import, so the shept
# imports below are pushed past the top of the module.
gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gdk, Gio, GLib, Gtk  # noqa: E402

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
from shept.models import (  # noqa: E402
    CATALOG,
    DownloadCancelled,
    ModelSpec,
    download,
    download_dir,
    human_size,
    is_downloaded,
)
from shept.queue import Job, JobState, Queue, export_all  # noqa: E402
from shept.transcribe import (  # noqa: E402
    EXPORT_LABELS,
    EXPORTERS,
    Segment,
    build_ffmpeg_cmd,
    build_whisper_cmd,
    format_txt,
    ms_to_ts,
    parse_progress_line,
    parse_segment,
)

#: The extensions offered when adding files; the README lists the same set.
MEDIA_EXTENSIONS = (
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
)


class _Cancelled(Exception):
    """Raised inside a worker thread when the user pressed Cancel."""


class SheptWindow(Adw.ApplicationWindow):
    def __init__(self, app: Adw.Application) -> None:
        super().__init__(application=app, title="Shept")
        self.set_default_size(820, 680)

        #: One process at a time: the ffmpeg of the job being decoded or the
        #: whisper-cli of the job being transcribed. Cancel terminates it.
        self._proc: subprocess.Popen[str] | None = None
        self._cancel = threading.Event()
        self.queue = Queue()
        self._selected: Job | None = None
        #: The job whose segments the text view is currently showing. Segments
        #: stream into the buffer only while this is the running job.
        self._buffer_job: Job | None = None
        self._rows: dict[Job, JobRow] = {}
        #: Model/language/translate are captured when the batch starts, so
        #: changing a dropdown mid-run does not split the batch across settings.
        self._run_model: Path | None = None
        self._run_lang = "auto"
        self._run_translate = False
        self._export_format = "txt"
        self._missing_tools = [
            name for name in REQUIRED_TOOLS if shutil.which(name) is None
        ]

        toolbar = Adw.ToolbarView()
        self.set_content(toolbar)

        header = Adw.HeaderBar()
        toolbar.add_top_bar(header)

        self.copy_btn = Gtk.Button.new_from_icon_name("edit-copy-symbolic")
        self.copy_btn.set_tooltip_text("Copy the selected transcript")
        self.copy_btn.set_sensitive(False)
        self.copy_btn.connect("clicked", self._on_copy)
        header.pack_end(self.copy_btn)

        # Two sections: one item per export format for the selected job (the
        # format is carried in the action target, so the file dialog never has
        # to guess it from the filename the user types), and one entry that
        # writes every completed job into a folder the user picks.
        self.export_menu = Gio.Menu()
        selected_section = Gio.Menu()
        for ext, label in EXPORT_LABELS.items():
            selected_section.append(f"{label} (.{ext})", f"win.export::{ext}")
        self.export_menu.append_section("Selected file", selected_section)
        all_section = Gio.Menu()
        all_section.append("Export all…", "win.export-all")
        self.export_menu.append_section("All completed files", all_section)
        self.save_btn = Gtk.MenuButton()
        self.save_btn.set_icon_name("document-save-symbolic")
        self.save_btn.set_tooltip_text("Save transcripts")
        self.save_btn.set_menu_model(self.export_menu)
        self.save_btn.set_sensitive(False)
        header.pack_end(self.save_btn)

        export_action = Gio.SimpleAction.new("export", GLib.VariantType.new("s"))
        export_action.connect("activate", self._on_export)
        export_all_action = Gio.SimpleAction.new("export-all", None)
        export_all_action.connect("activate", self._on_export_all)
        export_actions = Gio.SimpleActionGroup()
        export_actions.add_action(export_action)
        export_actions.add_action(export_all_action)
        self.insert_action_group("win", export_actions)

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

        queue_header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        queue_title = Gtk.Label(label="Queue", xalign=0)
        queue_title.add_css_class("heading")
        queue_title.set_hexpand(True)
        queue_header.append(queue_title)

        self.clear_btn = Gtk.Button(label="Clear finished")
        self.clear_btn.add_css_class("flat")
        self.clear_btn.set_tooltip_text("Remove finished files from the queue")
        self.clear_btn.connect("clicked", self._on_clear_finished)
        queue_header.append(self.clear_btn)

        self.add_btn = Gtk.Button(label="Add files…")
        self.add_btn.set_tooltip_text("Add one or more audio or video files")
        self.add_btn.connect("clicked", self._on_add_files)
        queue_header.append(self.add_btn)
        controls.append(queue_header)

        # Capped so a long queue scrolls instead of pushing the transcript and
        # the run buttons off the bottom of the window.
        queue_scroller = Gtk.ScrolledWindow()
        queue_scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        queue_scroller.set_min_content_height(120)
        queue_scroller.set_max_content_height(220)
        queue_scroller.set_propagate_natural_height(True)
        controls.append(queue_scroller)

        self.job_list = Gtk.ListBox()
        self.job_list.add_css_class("boxed-list")
        self.job_list.set_selection_mode(Gtk.SelectionMode.SINGLE)
        self.job_list.set_activate_on_single_click(True)
        self.job_list.connect("row-selected", self._on_row_selected)
        queue_scroller.set_child(self.job_list)

        drop = Gtk.DropTarget.new(Gdk.FileList, Gdk.DragAction.COPY)
        drop.connect("drop", self._on_drop)
        self.add_controller(drop)

        group = Adw.PreferencesGroup()
        controls.append(group)

        self._models: list[Path] = []
        self.model_row = Adw.ComboRow(title="Model")
        group.add(self.model_row)

        self.models_row = Adw.ActionRow(
            title="Whisper models",
            subtitle="Download a model without leaving the app",
        )
        manage_btn = Gtk.Button(label="Manage…")
        manage_btn.set_valign(Gtk.Align.CENTER)
        manage_btn.connect("clicked", self._on_manage_models)
        self.models_row.add_suffix(manage_btn)
        self.models_row.set_activatable_widget(manage_btn)
        group.add(self.models_row)
        self._model_dialog: ModelDownloadDialog | None = None

        self._hint_box = self._models_hint()
        controls.append(self._hint_box)

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
        self.run_btn.connect("clicked", self._on_run)

        self.cancel_btn = Gtk.Button(label="Cancel")
        self.cancel_btn.add_css_class("pill")
        self.cancel_btn.set_tooltip_text(
            "Stop the current file and leave the rest of the queue pending"
        )
        self.cancel_btn.set_visible(False)
        self.cancel_btn.connect("clicked", self._on_cancel_clicked)

        run_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        run_box.set_halign(Gtk.Align.CENTER)
        run_box.append(self.run_btn)
        run_box.append(self.cancel_btn)
        controls.append(run_box)
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

        self._reload_models()

    def _models_hint(self) -> Gtk.Widget:
        """Explain where models were searched and how to get one."""
        searched = "\n".join(f"  {path}" for path in model_search_paths())
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        label = Gtk.Label(
            label=(
                "No Whisper model found. Searched:\n"
                f"{searched}\n"
                'Use "Manage…" above to download one, or drop a ggml-*.bin '
                "file into one of those directories."
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

    def model_available(self, name: str) -> bool:
        """True when a model called `name` is already in the dropdown."""
        return any(model_display_name(model) == name for model in self._models)

    def _reload_models(self, select: str | None = None) -> None:
        """Re-read the model search paths and rebuild the dropdown.

        Runs after a download so a freshly fetched model appears immediately,
        without restarting the app.
        """
        if select is None:
            current = self.model_row.get_selected()
            if 0 <= current < len(self._models):
                select = model_display_name(self._models[current])
        if select is None:
            select = "large-v3-turbo"

        self._models = discover_models()
        strings = Gtk.StringList()
        if self._models:
            for model in self._models:
                strings.append(model_display_name(model))
        else:
            strings.append("(no models found)")
        self.model_row.set_model(strings)

        if self._models:
            index = 0
            for i, model in enumerate(self._models):
                if model_display_name(model) == select:
                    index = i
                    break
            self.model_row.set_selected(index)

        self._hint_box.set_visible(not self._models)
        self._update_run_state()

    def _on_manage_models(self, _btn: Gtk.Button) -> None:
        if self._model_dialog is not None:
            self._model_dialog.present()
            return
        self._model_dialog = ModelDownloadDialog(self)
        self._model_dialog.connect("close-request", self._on_model_dialog_closed)
        self._model_dialog.present()

    def _on_model_dialog_closed(self, *args) -> bool:
        self._model_dialog = None
        return False

    def on_model_downloaded(self, path: Path) -> None:
        name = model_display_name(path)
        self._reload_models(select=name)
        self.status.set_text(f"Downloaded model {name}.")

    def _transcribe_blocker(self) -> str:
        """Return why Transcribe is unavailable, or "" when it is available."""
        if self._missing_tools:
            names = " and ".join(self._missing_tools)
            return f"Missing {names} — {install_hint(self._missing_tools[0])}"
        if not self._models:
            return "No Whisper model found — see the hint below the model row."
        if not self.queue.jobs:
            return "Add an audio file first."
        if not self.queue.has_pending:
            return "Nothing left to transcribe — add a file or clear the list."
        return ""

    def _update_run_state(self) -> None:
        """Keep the run/cancel buttons and the queue actions in step."""
        running = self.queue.is_running
        pending = self.queue.counts()[JobState.PENDING]
        if running:
            self.run_btn.set_sensitive(False)
            self.run_btn.set_label("Transcribing…")
            self.run_btn.set_tooltip_text("Waiting for the current file to finish.")
        else:
            blocker = self._transcribe_blocker()
            self.run_btn.set_sensitive(not blocker)
            self.run_btn.set_label("Transcribe" if pending <= 1 else "Transcribe all")
            self.run_btn.set_tooltip_text(
                blocker or f"Transcribe {pending} files one after another."
            )
        self.cancel_btn.set_visible(running)
        self.clear_btn.set_sensitive(
            not running and any(job.is_finished for job in self.queue.jobs)
        )

    # ------------------------------------------------------------------ queue

    def _job_for(self, path: Path) -> Job | None:
        """Return the queued job for `path`, matching on the resolved path."""
        try:
            resolved = Path(path).expanduser().resolve()
        except OSError:
            return None
        for job in self.queue.jobs:
            if job.path == resolved:
                return job
        return None

    def add_files(self, paths: Iterable[Path]) -> list[Job]:
        """Queue `paths`, skipping duplicates and files that do not exist.

        Missing files are reported in the status line instead of raising: a
        dropped folder or a stale playlist entry must not break the batch.
        Returns the jobs that were actually added.
        """
        added = self.queue.add(paths)
        for job in added:
            self._add_row(job)
        skipped = self.queue.last_skipped
        if skipped:
            names = ", ".join(path.name for path in skipped)
            self.status.set_text(f"Skipped (not a file): {names}")
        elif len(added) > 1:
            self.status.set_text(f"Added {len(added)} files to the queue.")
        if added and self._selected is None:
            self._select_job(added[0])
        self._update_run_state()
        return added

    def set_input_file(self, path: Path) -> None:
        """Add `path` to the queue and show it.

        Used both by the add-files dialog and by files opened from the command
        line or the desktop, so there is a single code path for choosing input.
        """
        added = self.add_files([path])
        job = added[0] if added else self._job_for(path)
        if job is not None:
            self._select_job(job)

    def _add_row(self, job: Job) -> None:
        row = JobRow(job, self._remove_job)
        self._rows[job] = row
        self.job_list.append(row)

    def _remove_job(self, job: Job) -> None:
        """Drop one job from the queue; the running one stays until it ends."""
        if job.state is JobState.RUNNING:
            return
        row = self._rows.pop(job, None)
        if row is not None:
            self.job_list.remove(row)
        self.queue.remove(job)
        if self._selected is job:
            self._selected = None
            self._buffer_job = None
            remaining = self.queue.jobs
            if remaining:
                self._select_job(remaining[-1])
            else:
                self._show_job(None)
        self._update_run_state()

    def _on_clear_finished(self, _btn: Gtk.Button) -> None:
        removed = self.queue.clear_finished()
        for job in removed:
            row = self._rows.pop(job, None)
            if row is not None:
                self.job_list.remove(row)
        if any(job is self._selected for job in removed):
            self._selected = None
            self._buffer_job = None
            remaining = self.queue.jobs
            if remaining:
                self._select_job(remaining[-1])
            else:
                self._show_job(None)
        if removed:
            self.status.set_text(f"Removed {len(removed)} finished file(s).")
        self._update_run_state()

    def _on_row_selected(
        self, _list_box: Gtk.ListBox, row: Gtk.ListBoxRow | None
    ) -> None:
        if row is None:
            return
        self._show_job(row.job)  # type: ignore[attr-defined]

    def _show_job(self, job: Job | None) -> None:
        """Render one job's transcript into the text view and wire the actions."""
        self._selected = job
        self._buffer_job = job
        self.buffer.set_text(format_txt(job.segments) if job is not None else "")
        has_segments = job is not None and bool(job.segments)
        self.copy_btn.set_sensitive(has_segments)
        self.save_btn.set_sensitive(has_segments)

    def _select_job(self, job: Job) -> None:
        """Select `job` in the list and show it, even when it already is."""
        row = self._rows.get(job)
        if row is None:
            return
        self.job_list.select_row(row)
        self._show_job(job)

    def _on_add_files(self, _btn: Gtk.Button) -> None:
        dialog = Gtk.FileDialog(title="Add audio or video files")
        audio_filter = Gtk.FileFilter()
        audio_filter.set_name("Audio / video files")
        for ext in MEDIA_EXTENSIONS:
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
        dialog.open_multiple(self, None, self._on_files_chosen)

    def _on_files_chosen(self, dialog: Gtk.FileDialog, result: Gio.AsyncResult) -> None:
        try:
            files = dialog.open_multiple_finish(result)
        except GLib.Error:
            return
        paths = []
        for index in range(files.get_n_items()):
            gfile = files.get_item(index)
            path_str = gfile.get_path() if gfile is not None else None
            if path_str:
                paths.append(Path(path_str))
        if paths:
            self.add_files(paths)

    def _on_drop(self, _target: Gtk.DropTarget, value, _x: float, _y: float) -> bool:
        """Accept a drop of one or more files anywhere on the window."""
        if not isinstance(value, Gdk.FileList):
            return False
        paths = []
        for gfile in value.get_files():
            path_str = gfile.get_path()
            if path_str:
                paths.append(Path(path_str))
        if not paths:
            return False
        self.add_files(paths)
        return True

    def _selected_model(self) -> Path | None:
        if not self._models:
            return None
        return self._models[self.model_row.get_selected()]

    def _selected_lang(self) -> str:
        return LANGUAGES[self.lang_row.get_selected()][0]

    def _on_run(self, _btn: Gtk.Button) -> None:
        if self.queue.is_running or not self.queue.has_pending:
            return
        model = self._selected_model()
        if model is None:
            return
        self._run_model = model
        self._run_lang = self._selected_lang()
        self._run_translate = self.translate_row.get_active()
        self._cancel.clear()
        self.cancel_btn.set_sensitive(True)
        self.progress.set_visible(True)
        self.progress.set_fraction(0.0)
        self.progress.set_text("Starting…")
        self.status.set_text("Transcribing the queue…")
        self._start_next_job()

    def _start_next_job(self) -> None:
        """Run the oldest pending job, or wind the batch down when none is left."""
        job = self.queue.next_pending()
        model = self._run_model
        if job is None or model is None:
            self._finish_batch()
            return
        self.queue.mark_running(job)
        self._rows[job].refresh()
        self._select_job(job)
        self.progress.set_fraction(0.0)
        self.progress.set_text("Decoding audio…")
        self.status.set_text(f"Extracting audio from {job.display_name} with ffmpeg…")
        self._update_run_state()
        threading.Thread(
            target=self._job_worker,
            args=(job, model, self._run_lang, self._run_translate),
            name=f"shept-{job.display_name}",
            daemon=True,
        ).start()

    def _finish_batch(self) -> None:
        """Return the window to its idle state and summarise the batch."""
        self._run_model = None
        self.progress.set_visible(False)
        self.progress.set_text("")
        self._update_run_state()
        counts = self.queue.counts()
        if self._cancel.is_set():
            self.status.set_text(
                f"Cancelled. {counts[JobState.PENDING]} file(s) left in the queue."
            )
            return
        parts = [f"{counts[JobState.DONE]} done"]
        if counts[JobState.FAILED]:
            parts.append(f"{counts[JobState.FAILED]} failed")
        if counts[JobState.CANCELLED]:
            parts.append(f"{counts[JobState.CANCELLED]} cancelled")
        self.status.set_text("Queue finished: " + ", ".join(parts) + ".")

    def _on_cancel_clicked(self, _btn: Gtk.Button) -> None:
        """Abandon the running job and leave the rest of the queue pending."""
        if not self.queue.is_running:
            return
        self._cancel.set()
        self.cancel_btn.set_sensitive(False)
        self.status.set_text("Cancelling…")
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()

    def _job_worker(self, job: Job, model: Path, lang: str, translate: bool) -> None:
        """Decode and transcribe one job, then report back on the main loop.

        Each job gets its own thread and exactly one `_job_finished` call, so a
        broken file or a cancellation can never take the rest of the queue with
        it. The next job starts from the main loop, never from here, which is
        what keeps the batch strictly sequential.
        """
        error: str | None = None
        cancelled = False
        try:
            with tempfile.TemporaryDirectory(prefix="shept-") as tmp:
                wav_path = Path(tmp) / "input.wav"
                self._decode(job, wav_path)
                if self._cancel.is_set():
                    raise _Cancelled
                self._transcribe(job, model, wav_path, lang, translate)
        except _Cancelled:
            cancelled = True
        except Exception as exc:  # a bad file must not stop the batch
            error = str(exc) or exc.__class__.__name__
        finally:
            GLib.idle_add(self._job_finished, job, error, cancelled)

    def _decode(self, job: Job, wav_path: Path) -> None:
        """Decode one job's input to 16 kHz mono WAV with ffmpeg."""
        duration = self._probe_duration(job.path)
        proc = subprocess.Popen(
            build_ffmpeg_cmd(job.path, wav_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._proc = proc
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                if self._cancel.is_set():
                    proc.terminate()
                    break
                secs = parse_progress_line(line)
                if secs is None or duration <= 0:
                    continue
                frac = max(0.0, min(1.0, secs / duration))
                GLib.idle_add(self._job_progress, job, frac, "Decoding audio…")
        finally:
            ret = proc.wait()
            self._proc = None
        if self._cancel.is_set():
            raise _Cancelled
        if ret != 0:
            message = (proc.stderr.read() if proc.stderr else "") or "unknown error"
            raise RuntimeError(f"ffmpeg failed: {message.strip()}")

    def _transcribe(
        self, job: Job, model: Path, wav_path: Path, lang: str, translate: bool
    ) -> None:
        """Run whisper-cli on the decoded WAV, streaming its segments."""
        GLib.idle_add(self._job_progress, job, 0.0, "Loading model…")
        GLib.idle_add(
            self.status.set_text, f"Running whisper-cli on {job.display_name}…"
        )
        duration = self._probe_duration(wav_path)
        proc = subprocess.Popen(
            build_whisper_cmd(
                model,
                wav_path,
                lang,
                min(8, os.cpu_count() or 4),
                translate,
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._proc = proc
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                if self._cancel.is_set():
                    proc.terminate()
                    break
                self._handle_line(job, line.rstrip("\n"), duration)
        finally:
            ret = proc.wait()
            self._proc = None
        if self._cancel.is_set():
            raise _Cancelled
        if ret != 0:
            raise RuntimeError(f"whisper-cli exited with {ret}")

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

    def _handle_line(self, job: Job, line: str, duration: float) -> None:
        segment = parse_segment(line)
        if segment is None:
            return
        GLib.idle_add(self._job_segment, job, segment)
        if duration > 0:
            frac = max(0.0, min(1.0, (segment.end_ms / 1000.0) / duration))
            GLib.idle_add(self._job_progress, job, frac, ms_to_ts(segment.end_ms))

    def _job_segment(self, job: Job, segment: Segment) -> bool:
        """Keep one segment for the exporters and show it when it is on screen."""
        job.segments.append(segment)
        if self._buffer_job is job:
            end = self.buffer.get_end_iter()
            self.buffer.insert(end, segment.text + "\n")
        return False

    def _job_progress(self, job: Job, frac: float, label: str) -> bool:
        job.progress = frac
        row = self._rows.get(job)
        if row is not None:
            row.set_progress(frac)
        self.progress.set_fraction(frac)
        self.progress.set_text(f"{label}  ({int(frac * 100)}%)")
        return False

    def _job_finished(self, job: Job, error: str | None, cancelled: bool) -> bool:
        """Record one job's outcome, then start the next one — failures included."""
        self._proc = None
        if cancelled:
            self.queue.mark_cancelled(job)
        elif error:
            self.queue.mark_failed(job, error)
        else:
            self.queue.mark_done(job)
        row = self._rows.get(job)
        if row is not None:
            row.refresh()
        if self._buffer_job is job:
            self._show_job(job)
        if cancelled:
            self._finish_batch()
            return False
        self._start_next_job()
        return False

    def _on_copy(self, _btn: Gtk.Button) -> None:
        job = self._selected
        if job is None or not job.segments:
            return
        self.get_clipboard().set(format_txt(job.segments))
        self.status.set_text(f"Copied {job.display_name} to the clipboard.")

    def _on_export(
        self,
        _action: Gio.SimpleAction,
        param: GLib.Variant | None,
    ) -> None:
        ext = param.get_string() if param is not None else "txt"
        if ext not in EXPORTERS:
            return
        job = self._selected
        if job is None or not job.segments:
            return
        self._export_format = ext
        dialog = Gtk.FileDialog(title=f"Save {job.display_name} as .{ext}")
        dialog.set_initial_name(job.path.with_suffix(f".{ext}").name)
        dialog.save(
            self,
            None,
            lambda d, r, j=job, e=ext: self._on_save_chosen(d, r, j, e),
        )

    def _on_save_chosen(
        self,
        dialog: Gtk.FileDialog,
        result: Gio.AsyncResult,
        job: Job,
        ext: str,
    ) -> None:
        try:
            f = dialog.save_finish(result)
        except GLib.Error:
            return
        target = f.get_path()
        if target is None:
            return
        formatter = EXPORTERS.get(ext, format_txt)
        Path(target).write_text(formatter(job.segments), encoding="utf-8")
        self.status.set_text(f"Saved: {target}")

    def _on_export_all(
        self,
        _action: Gio.SimpleAction,
        _param: GLib.Variant | None,
    ) -> None:
        """Write every completed job into a folder the user picks.

        The format is whichever one was last chosen from the per-file section
        of the menu, plain text until then.
        """
        if not any(job.segments for job in self.queue.jobs):
            self.status.set_text("Nothing to export yet.")
            return
        dialog = Gtk.FileDialog(title="Choose a folder for the transcripts")
        dialog.select_folder(self, None, self._on_export_folder_chosen)

    def _on_export_folder_chosen(
        self, dialog: Gtk.FileDialog, result: Gio.AsyncResult
    ) -> None:
        try:
            folder = dialog.select_folder_finish(result)
        except GLib.Error:
            return
        path_str = folder.get_path()
        if path_str is None:
            return
        try:
            written = export_all(
                self.queue.jobs,
                self._export_format,
                Path(path_str),
            )
        except OSError as exc:
            self.status.set_text(f"Export failed: {exc}")
            return
        self.status.set_text(
            f"Exported {len(written)} transcript(s) as .{self._export_format} "
            f"to {path_str}."
        )


class JobRow(Adw.ActionRow):
    """One row of the queue: file name, state, per-file progress, remove button.

    `Adw.ActionRow` is a `Gtk.ListBoxRow`, so it can live directly in the
    queue's `Gtk.ListBox` while keeping the libadwaita row look.
    """

    def __init__(self, job: Job, on_remove: Callable[[Job], None]) -> None:
        super().__init__(title=job.display_name)
        self.job = job
        self.set_tooltip_text(str(job.path))

        self.progress = Gtk.ProgressBar()
        self.progress.set_size_request(110, -1)
        self.progress.set_valign(Gtk.Align.CENTER)
        self.progress.set_visible(False)
        self.add_suffix(self.progress)

        self.remove_btn = Gtk.Button.new_from_icon_name("user-trash-symbolic")
        self.remove_btn.add_css_class("flat")
        self.remove_btn.set_valign(Gtk.Align.CENTER)
        self.remove_btn.set_tooltip_text("Remove from the queue")
        self.remove_btn.connect("clicked", lambda _btn: on_remove(job))
        self.add_suffix(self.remove_btn)

        self.refresh()

    def refresh(self) -> None:
        """Re-read the job's state after a transition."""
        job = self.job
        self.set_title(job.display_name)
        self.set_subtitle(state_text(job))
        running = job.state is JobState.RUNNING
        self.progress.set_visible(running)
        self.progress.set_fraction(job.progress)
        self.remove_btn.set_visible(not running)
        self.remove_btn.set_sensitive(not running)

    def set_progress(self, fraction: float) -> None:
        """Advance the small bar while the job runs."""
        self.progress.set_fraction(fraction)
        self.progress.set_text(f"{int(fraction * 100)}%")
        self.set_subtitle(state_text(self.job))


def state_text(job: Job) -> str:
    """The one-line state shown as a queue row's subtitle."""
    if job.state is JobState.PENDING:
        return "Pending"
    if job.state is JobState.RUNNING:
        return f"Transcribing… {int(job.progress * 100)}%"
    if job.state is JobState.DONE:
        count = len(job.segments)
        return f"Done — {count} {'segment' if count == 1 else 'segments'}"
    if job.state is JobState.FAILED:
        return f"Failed: {job.error or 'unknown error'}"
    return "Cancelled"


class ModelDownloadDialog(Adw.Window):
    """Download models from HuggingFace into the user's data directory.

    Downloads run on a worker thread; every widget update is bounced back to
    the main loop with `GLib.idle_add`, because GTK is not thread-safe.
    """

    def __init__(self, parent: SheptWindow) -> None:
        super().__init__(transient_for=parent, modal=True, title="Whisper models")
        self.set_default_size(560, 640)

        self._parent = parent
        self._dest = download_dir()
        self._cancel = threading.Event()
        self._active: ModelSpec | None = None
        self._thread: threading.Thread | None = None
        self._closed = False
        #: spec name -> its row, progress bar and the suffix widget currently
        #: in use (a download/cancel button or the "installed" check mark).
        self._entries: dict[str, dict[str, Gtk.Widget | None]] = {}
        self.connect("close-request", self._on_close_request)

        toolbar = Adw.ToolbarView()
        self.set_content(toolbar)
        toolbar.add_top_bar(Adw.HeaderBar())

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        box.set_margin_top(12)
        box.set_margin_bottom(12)
        box.set_margin_start(12)
        box.set_margin_end(12)
        toolbar.set_content(box)

        self._message = Gtk.Label(xalign=0)
        self._message.set_wrap(True)
        self._message.set_visible(False)
        box.append(self._message)

        scroller = Gtk.ScrolledWindow(vexpand=True)
        box.append(scroller)
        clamp = Adw.Clamp(maximum_size=760)
        scroller.set_child(clamp)

        self._group = Adw.PreferencesGroup(
            title="Available models",
            description=f"Downloads are saved to {self._dest}",
        )
        clamp.set_child(self._group)

        for spec in CATALOG:
            self._add_entry(spec)

    # ---------------------------------------------------------------- widgets

    def _add_entry(self, spec: ModelSpec) -> None:
        row = Adw.ActionRow(
            title=spec.name,
            subtitle=f"{human_size(spec.size_bytes)} · {spec.note}",
        )
        progress = Gtk.ProgressBar()
        progress.set_show_text(True)
        progress.set_visible(False)
        progress.set_size_request(160, -1)
        progress.set_valign(Gtk.Align.CENTER)
        row.add_suffix(progress)
        self._group.add(row)

        self._entries[spec.name] = {
            "row": row,
            "progress": progress,
            "action": None,
        }
        self._refresh_entry(spec)

    def _refresh_entry(self, spec: ModelSpec) -> None:
        entry = self._entries[spec.name]
        row = entry["row"]
        action = entry["action"]
        assert isinstance(row, Adw.ActionRow)
        if action is not None:
            row.remove(action)
        entry["action"] = None

        progress = entry["progress"]
        assert isinstance(progress, Gtk.ProgressBar)
        progress.set_visible(self._active is spec)

        if self._active is spec:
            cancel_btn = Gtk.Button(label="Cancel")
            cancel_btn.add_css_class("destructive-action")
            cancel_btn.set_valign(Gtk.Align.CENTER)
            cancel_btn.connect("clicked", self._on_cancel_clicked)
            row.add_suffix(cancel_btn)
            entry["action"] = cancel_btn
            return

        if is_downloaded(spec, self._dest) or self._parent.model_available(spec.name):
            installed = Gtk.Box(spacing=6)
            check = Gtk.Image.new_from_icon_name("object-select-symbolic")
            installed.append(check)
            installed.append(Gtk.Label(label="Installed"))
            installed.set_tooltip_text("This model is already available")
            row.add_suffix(installed)
            entry["action"] = installed
            return

        download_btn = Gtk.Button(label="Download")
        download_btn.set_valign(Gtk.Align.CENTER)
        download_btn.set_sensitive(self._active is None)
        download_btn.connect("clicked", lambda _btn, s=spec: self._start_download(s))
        row.add_suffix(download_btn)
        entry["action"] = download_btn

    def _refresh_entries(self) -> None:
        for spec in CATALOG:
            self._refresh_entry(spec)

    def _set_message(self, text: str, error: bool = True) -> None:
        self._message.set_text(text)
        self._message.set_visible(bool(text))
        for css_class in ("error", "dim-label"):
            if css_class == ("error" if error else "dim-label"):
                self._message.add_css_class(css_class)
            else:
                self._message.remove_css_class(css_class)

    # ---------------------------------------------------------------- download

    def _start_download(self, spec: ModelSpec) -> None:
        if self._active is not None:
            return
        self._cancel.clear()
        self._active = spec
        self._set_message("")
        self._refresh_entries()
        self._thread = threading.Thread(
            target=self._download_worker,
            args=(spec,),
            name=f"shept-download-{spec.name}",
            daemon=True,
        )
        self._thread.start()

    def _download_worker(self, spec: ModelSpec) -> None:
        try:
            path = download(
                spec,
                self._dest,
                progress=self._report_progress,
                is_cancelled=self._cancel.is_set,
            )
        except DownloadCancelled as exc:
            GLib.idle_add(self._download_finished, spec, None, str(exc), False)
        except Exception as exc:  # surfaced in the dialog, never crashes
            GLib.idle_add(self._download_finished, spec, None, str(exc), True)
        else:
            GLib.idle_add(self._download_finished, spec, path, "", False)

    def _report_progress(self, received: int, total: int) -> None:
        spec = self._active
        if spec is None or self._closed:
            return
        GLib.idle_add(self._update_progress, spec.name, received, total)

    def _update_progress(self, name: str, received: int, total: int) -> bool:
        if self._closed or self._active is None or self._active.name != name:
            return False
        progress = self._entries[name]["progress"]
        assert isinstance(progress, Gtk.ProgressBar)
        if total > 0:
            fraction = min(1.0, received / total)
            progress.set_fraction(fraction)
            progress.set_text(
                f"{human_size(received)} of {human_size(total)} "
                f"({int(fraction * 100)}%)"
            )
        else:
            progress.pulse()
            progress.set_text(human_size(received))
        return False

    def _download_finished(
        self,
        spec: ModelSpec,
        path: Path | None,
        message: str,
        failed: bool,
    ) -> bool:
        if self._closed:
            return False
        self._active = None
        self._thread = None
        if path is not None:
            self._set_message(f"{spec.name} downloaded.", error=False)
            self._parent.on_model_downloaded(path)
        elif message:
            self._set_message(message, error=failed)
            if failed:
                self._parent.status.set_text(f"Model download failed: {message}")
            else:
                self._parent.status.set_text("Model download cancelled.")
        self._refresh_entries()
        return False

    def _on_cancel_clicked(self, _btn: Gtk.Button) -> None:
        if self._active is None:
            return
        self._cancel.set()
        self._set_message("Cancelling…", error=False)
        progress = self._entries[self._active.name]["progress"]
        assert isinstance(progress, Gtk.ProgressBar)
        progress.set_text("Cancelling…")

    def _on_close_request(self, *_args) -> bool:
        self._closed = True
        self._cancel.set()
        if self._active is not None:
            self._parent.status.set_text("Model download cancelled.")
            self._active = None
        return False


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
        """Queue every file passed on the command line or from the desktop."""
        win = self.get_active_window() or SheptWindow(self)
        win.present()
        if not files:
            return

        paths: list[Path] = []
        for gfile in files:
            path_str = gfile.get_path() if gfile is not None else None
            if path_str is None:
                uri = gfile.get_uri() if gfile is not None else ""
                win.status.set_text(f"Cannot open {uri}: not a local file.")
                continue
            path = Path(path_str)
            if not path.is_file():
                win.status.set_text(f"Cannot open {path}: no such file.")
                continue
            paths.append(path)
        if not paths:
            return

        # The first file is added and selected — the single-file flow — and the
        # rest simply extend the queue.
        win.set_input_file(paths[0])
        if len(paths) > 1:
            win.add_files(paths[1:])


def main() -> int:
    # Missing tools and models are reported in the window instead of aborting
    # here: a desktop-menu launch has nowhere to show a stderr message.
    GLib.set_application_name("Shept")
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    return SheptApp().run(sys.argv)
