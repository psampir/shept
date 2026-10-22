# Changelog

All notable changes to Shept are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Every language whisper.cpp knows**: the source language list is no
  longer eight hand-picked entries but all 100 languages and dialects
  whisper.cpp v1.8.4 supports, taken from the `g_lang` table in
  `src/whisper.cpp` at the pinned tag instead of a copy that can drift. That
  includes `yue` (Cantonese), `haw` (Hawaiian) and every other three-letter
  code, and locale detection now recognises them too. Languages can be matched
  fuzzily by name or code — `ukr` finds Ukrainian, `jpn` finds Japanese — with
  results ranked so a prefix beats a word-start match and scattered letters
  come last.
- **A searchable language picker, with favourites**: the Language row is an
  `Adw.ActionRow` with a **Choose…** button that opens a picker window — a
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

## [0.1.0] - 2026-10-22

First release.

### Added

- Local transcription of audio and video files with `whisper-cli` from
  whisper.cpp. `ffmpeg` decodes each file to a temporary 16 kHz mono WAV,
  whisper.cpp transcribes it, and the transcript streams into the window as it
  is produced — no uploads, no accounts, no background daemon.
- A GTK4 / libadwaita interface, with the queue, the model and language rows
  and the transcript in a single window.
- A batch queue: add many files with the file dialog, drag and drop, or the
  command line, and transcribe them one after another with a per-file progress
  bar driven by whisper.cpp's own timestamps, a cancellable run, and a summary
  when the batch ends. A failed file is marked in its row and does not stop the
  rest of the queue.
- Five export formats — plain text (`.txt`), SubRip (`.srt`), WebVTT (`.vtt`),
  lyrics (`.lrc`) and timed JSON (`.json`) — for the selected file, or for every
  finished file into one folder at once. Copying the selected transcript to the
  clipboard uses the same text.
- A model downloader: the models Shept can fetch are listed with their size and
  what they are good for, downloads show progress, can be cancelled, and a
  finished model appears in the model dropdown immediately.
- Model discovery across several directories — `$SHEPT_MODEL_DIR`,
  `$XDG_DATA_HOME/shept/models`, `$XDG_DATA_HOME/whisper-cpp/models`,
  `~/.local/share/whisper-cpp/models` (searched even when `$XDG_DATA_HOME`
  points elsewhere, which is what makes models visible inside the Flatpak
  sandbox), `~/.cache/whisper.cpp/models`, `/usr/share/whisper-cpp/models` and
  `/usr/local/share/whisper-cpp/models` — with the first directory winning when
  a model exists in more than one.
- A language selector that defaults to the system locale when Shept supports
  that language and to auto-detection otherwise.
- Guidance for missing dependencies: a banner names the tools that are not on
  `PATH` and shows the install command for the detected distribution, and
  **Transcribe** stays disabled until they are available.
- A Flatpak manifest (`build-aux/io.github.psampir.Shept.yml`) that builds
  whisper.cpp with Vulkan and shared libraries, rebuilds `ffmpeg` for the
  `ffmpeg` / `ffprobe` executables, and installs Shept itself.

[Unreleased]: https://github.com/psampir/shept/compare/v0.1.0...dev
[0.1.0]: https://github.com/psampir/shept/releases/tag/v0.1.0
