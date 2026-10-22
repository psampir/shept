# Changelog

All notable changes to Shept are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Nothing yet.

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
  that language and to auto-detection otherwise, plus optional translation of
  the result into English.
- Guidance for missing dependencies: a banner names the tools that are not on
  `PATH` and shows the install command for the detected distribution, and
  **Transcribe** stays disabled until they are available.
- A Flatpak manifest (`build-aux/io.github.psampir.Shept.yml`) that builds
  whisper.cpp with Vulkan and shared libraries, rebuilds `ffmpeg` for the
  `ffmpeg` / `ffprobe` executables, and installs Shept itself.

[Unreleased]: https://github.com/psampir/shept/compare/v0.1.0...dev
[0.1.0]: https://github.com/psampir/shept/releases/tag/v0.1.0
