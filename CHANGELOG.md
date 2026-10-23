# Changelog

All notable changes to Shept are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-10-23

### Fixed

- The language picker reacts to a click again. Its rows were plain
  `Adw.ActionRow` widgets appended to a `Gtk.ListBox`, and a list box only
  emits `row-activated` for real `Gtk.ListBoxRow` children, so choosing a
  language with the mouse did nothing while Enter still worked. Each row is
  now wrapped in a `Gtk.ListBoxRow`, which also restores arrow key navigation.

## [0.1.1] - 2026-10-22

### Changed

- The application icon is new: a head in profile with a hand raised to its
  mouth.

- The desktop entry and icon are now installed by the wheel.

### Fixed

- The Flatpak manifest now grants network access, so the model downloader works
  inside the sandbox.

## [0.1.0] - 2026-10-22

First release.

### Added

- Local transcription of audio and video files with `whisper-cli` from
  whisper.cpp. `ffmpeg` decodes each file to a temporary 16 kHz mono WAV,
  whisper.cpp transcribes it, and the transcript streams into the window as it
  is produced: no uploads, no accounts, no background daemon.

- A GTK4 / libadwaita interface, with the queue, the model and language rows
  and the transcript in a single window.

- A batch queue: add many files with the file dialog, drag and drop, or the
  command line, and transcribe them one after another with a per-file progress
  bar driven by whisper.cpp's own timestamps, a cancellable run, and a summary
  when the batch ends. A failed file is marked in its row and does not stop the
  rest of the queue.

- Five export formats: plain text (`.txt`), SubRip (`.srt`), WebVTT (`.vtt`),
  lyrics (`.lrc`) and timed JSON (`.json`), for the selected file, or for every
  finished file into one folder at once. Copying the selected transcript to the
  clipboard uses the same text.

- A model downloader: the models Shept can fetch are listed with their size and
  what they are good for, downloads show progress, can be cancelled, and a
  finished model appears in the model dropdown immediately.

- Model discovery across several directories (`$SHEPT_MODEL_DIR`,
  `$XDG_DATA_HOME/shept/models`, `$XDG_DATA_HOME/whisper-cpp/models`,
  `~/.local/share/whisper-cpp/models` (searched even when `$XDG_DATA_HOME`
  points elsewhere, which is what makes models visible inside the Flatpak
  sandbox), `~/.cache/whisper.cpp/models`, `/usr/share/whisper-cpp/models` and
  `/usr/local/share/whisper-cpp/models`), with the first directory winning when
  a model exists in more than one.

- A language selector that defaults to the system locale when Shept supports
  that language and to auto-detection otherwise.

- Guidance for missing dependencies: a banner names the tools that are not on
  `PATH` and shows the install command for the detected distribution, and
  **Transcribe** stays disabled until they are available.

- A Flatpak manifest (`build-aux/io.github.psampir.Shept.yml`) that builds
  whisper.cpp with Vulkan and shared libraries, rebuilds `ffmpeg` for the
  `ffmpeg` / `ffprobe` executables, and installs Shept itself.

[Unreleased]: https://github.com/psampir/shept/compare/v0.1.1...dev
[0.1.1]: https://github.com/psampir/shept/releases/tag/v0.1.1
[0.1.0]: https://github.com/psampir/shept/releases/tag/v0.1.0

- **Delete models from the app**: every installed model in the **Manage…**
  window gets a trash button, shown only where the file is genuinely removable
  (the directory must be writable). Read-only locations (system directories,
  and `~/.local/share/whisper-cpp/models` inside the Flatpak sandbox) show a
  lock instead of a button that could only fail. Deletion asks first, is
  disabled while the queue runs, unlinks a symlink rather than its target, and
  the dropdown falls back to `large-v3-turbo` or the next model when the
  selected one is removed.

- **Your own model folders**: the **Manage…** window now lists every directory
  Shept searches, labelled by origin, with **Add folder…** and a remove button
  for the ones you added. Extra folders are persisted in `settings.json`,
  searched ahead of the default locations, and picked up immediately without a
  restart. Removing one stops the search and leaves the files alone.

- **Every language whisper.cpp knows**: the source language list is no
  longer eight hand-picked entries but all 100 languages and dialects
  whisper.cpp v1.8.4 supports, taken from the `g_lang` table in
  `src/whisper.cpp` at the pinned tag instead of a copy that can drift. That
  includes `yue` (Cantonese), `haw` (Hawaiian) and every other three-letter
  code, and locale detection now recognises them too. Languages can be matched
  fuzzily by name or code (`ukr` finds Ukrainian, `jpn` finds Japanese) with
  results ranked so a prefix beats a word-start match and scattered letters
  come last.

- **A searchable language picker, with favourites**: the Language row is an
  `Adw.ActionRow` with a **Choose…** button that opens a picker window: a
  search box, a live-filtered list and a star on every row. Typing filters
  through the fuzzy matcher, Enter takes the first match, Escape closes the
  window and the arrow keys walk the list, so a language is a few keystrokes
  away instead of a 100-row scroll. Starred languages are stored under
  `favourites` in `settings.json`, show up directly below **Auto-detect**
  when the search box is empty, and survive restarts; a star is saved as soon
  as it is toggled. Settings files written before favourites existed keep
  loading unchanged.

- **Skip silence**: a switch under Language drives whisper.cpp's Silero VAD
  stage (`--vad -vm`), so silent stretches are dropped before they reach the
  model and the invented "thank you for watching" that whisper likes to put
  over trailing silence does not appear. The switch is on by default when the
  VAD asset is installed; without it, turning the switch on reveals a
  **Download** button that fetches the 885 kB `ggml-silero-v6.2.0.bin` into
  `$XDG_DATA_HOME/shept/models`. Nothing is downloaded until you press it, and
  the asset never appears in the transcription model list.

- The transcription model, language and Skip silence choice are remembered
  between runs in `$XDG_CONFIG_HOME/shept/settings.json`
  (`~/.config/shept/settings.json`). A missing, damaged or outdated file falls
  back to defaults instead of stopping the app, and a model that is no longer
  installed is replaced by the default selection.

### Fixed

- Closing the window while a file is being transcribed now stops `whisper-cli`
  (or the `ffmpeg` decode) instead of leaving it running in the background
  with the GPU still busy, and the job's temporary directory is removed.
  Cancel and closing the window share one code path, so they behave the same,
  and the worker thread no longer touches GTK widgets after the window is gone.

- A damaged media file can no longer hang the queue. `ffmpeg` used to log its
  errors to a pipe nobody read until the process exited; a file that produced
  more than 64 kB of them filled the pipe and deadlocked the decode. The
  diagnostics now go to a temporary file, only the last 2 kB is read back on
  failure, and `ffmpeg` runs with `-nostdin` so typing in the terminal cannot
  abort it.

- `ffprobe` is bounded (30 s) and can be stopped by **Cancel** and by closing
  the window, so a file on a hung FUSE or SMB mount no longer wedges a job.

- Saving a transcript reports read-only targets, a full disk and over-long
  names in the status line instead of raising silently inside a GTK callback,
  and writes atomically so an existing file is never replaced by a partial one.
  **Manage…** says why it cannot open when the model folder is unwritable.

- A failed row's subtitle is capped to the first couple of lines; the whole
  output of the failing tool is in its tooltip. A failed `whisper-cli` now
  carries the last diagnostic lines (a corrupt model, an OOM, a bad `-vm` path)
  instead of only its exit status.

- **Export all** writes only files that finished successfully, never overwrites
  a file already in the destination folder, and shortens over-long names so a
  batch cannot abort part-way through.

- A cancel that lands between two files leaves the next one pending and
  runnable instead of marking it cancelled without running it, and **Cancel**
  stays usable when a cancel does not end the batch.

- A successful Silero VAD download is saved immediately, so the switch no
  longer reverts on the next launch. A truncated VAD file counts as missing,
  and an asset that disappears while the app runs is re-resolved when the
  switch is touched.

- `settings.json` is fsynced before being renamed into place, and stale
  `*.part` downloads are cleaned up.

### Known limitations

The following are known and deferred to 0.2:

- **Cancel and close can block the main loop for 2–3 s** while the running
  child is stopped with `SIGTERM` and then `SIGKILL`.

- **Truncated model files are not detected against the catalogue sizes**; a
  download cut short by a crash is only noticed when whisper-cli fails on it.

- **Downloads cannot be resumed or retried**; an interrupted transfer starts
  over from zero.

- **The batch worker is not yet a GTK-free module**, so the queue's
  run/finish/cancel transitions can only be tested at the `Queue` level.
