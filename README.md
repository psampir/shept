<p align="left"><img src="data/logo.svg" width="112" alt="Shept logo"></p>

# Shept

*Shept* is Polish for "whisper", spelled the way it sounds.

Local speech-to-text for your desktop, powered by [whisper.cpp](https://github.com/ggml-org/whisper.cpp).

![Shept transcribing a queue of files](data/screenshots/queue.png)

## Contents

- [What it does](#what-it-does)
- [Requirements](#requirements)
- [Install](#install)
- [Models](#models)
- [Usage](#usage)
- [How it works](#how-it-works)
- [Performance notes](#performance-notes)
- [Troubleshooting](#troubleshooting)
- [Status](#status)
- [Changelog](#changelog)
- [License](#license)

## What it does

- Add any audio or video file to a queue (mp3, wav, m4a, ogg, opus, flac,
  aac, mp4, mkv, mov, webm, avi, …) with the file dialog, drag & drop, or the
  command line
- Transcribe locally with a Whisper model of your choice, one file after
  another, so a batch can run unattended
- Download models from inside the app, with progress, cancellation and a
  transfer that resumes where it stopped
- Skip silence with whisper.cpp's Silero VAD, so silent stretches never reach
  the model and are not hallucinated into text
- Every language whisper.cpp supports (all 100, auto-detect included) behind
  a searchable picker with fuzzy matching and starred favourites
- Remember the model, language, favourites and silence choice between runs
- Streams output as it transcribes, with a per-file progress bar and a summary
  when the batch ends
- Copy or save the selected file's transcript, or export every finished file
  at once: `.txt`, `.srt`, `.vtt`, `.lrc` or `.json`
- See the version, the licence and the project links in the About window

Everything runs on your machine: no accounts, no uploads, no network access
during transcription. The only network access is a model download you start
yourself; no daemons or background processes either: `whisper-cli` is launched
on demand and exits when done, and the model is loaded into memory only for the
duration of a run.

Only file transcription is supported: there is no microphone / live mode.

## Requirements

- Python 3.10+
- GTK 4, libadwaita and PyGObject (`gi`): system packages, not pip installs
- `ffmpeg` and `ffprobe`: Shept checks for both at startup and names the
  missing one in the banner; without `ffprobe` transcription still works, but
  progress stays at 0 % for the whole file
- `whisper-cli` from whisper.cpp (1.7 or newer)

GPU acceleration is optional. A build of whisper.cpp with Vulkan support makes
transcription considerably faster than a CPU-only build on most modern GPUs
(AMD, Intel, NVIDIA); without it, whisper-cli falls back to the CPU.

## Install

Two ways: a self-contained Flatpak bundle from the releases page (nothing else
to install), or a wheel that uses the `whisper-cli` already on your system.

### Flatpak bundle

Each release ships `shept.flatpak`, which bundles `whisper-cli` and `ffmpeg`, so
neither has to be installed on the host:

```bash
curl -LO https://github.com/psampir/shept/releases/latest/download/shept.flatpak
flatpak install --user shept.flatpak
```

GPU drivers are deliberately not bundled: Vulkan comes from the host through
`--device=dri`, so driver updates keep working. Models are shared with a native
install (see [Models](#models)).

### Wheel

If `whisper-cli` and `ffmpeg` are already on your PATH, install the wheel
attached to the release:

```bash
python3 -m pip install --user ./shept-<version>-py3-none-any.whl
python3 -m shept
```

PyGObject is taken from the system and never from pip. The distribution sections
below cover `whisper-cli` only, so skip them if you used the Flatpak bundle.

### Fedora

The Fedora repositories ship only the whisper.cpp library, so use a COPR that
provides a CLI binary built with Vulkan enabled:

```bash
sudo dnf copr enable -y sneed/whisper-cpp-vulkan
sudo dnf install -y whisper-cpp ffmpeg python3-gobject gtk4 libadwaita
```

`ffmpeg` is in RPM Fusion Free; if you do not have RPM Fusion enabled you can
use `ffmpeg-free` from the main repositories instead.

### Debian / Ubuntu

`whisper.cpp` is not packaged in the distribution repositories and must be
built from source. The build takes only a few minutes:

```bash
# runtime + build dependencies
sudo apt install -y build-essential cmake git ffmpeg \
    python3-gi python3-gi-cairo gir1.2-gtk-4.0 gir1.2-adw-1 \
    libvulkan-dev glslc          # Vulkan build; skip these two for CPU-only

git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp
cmake -B build -DGGML_VULKAN=1   # drop -DGGML_VULKAN=1 for a CPU-only build
cmake --build build -j --config Release

sudo install -Dm755 build/bin/whisper-cli /usr/local/bin/whisper-cli
sudo install -Dm755 build/src/libwhisper.so* /usr/local/lib/
sudo ldconfig
```

### Arch

Arch packages whisper.cpp as `whisper-cpp` (a CPU build). A Vulkan-enabled
build is also available from the AUR:

```bash
sudo pacman -S whisper-cpp ffmpeg python-gobject gtk4 libadwaita

# or, for the Vulkan build, with an AUR helper:
paru -S whisper.cpp-vulkan
```

### Other distributions

If no package exists for your distribution, build `whisper-cli` from source as
shown in the Debian/Ubuntu section above; the CMake steps are the same
everywhere. Make sure the resulting binary and any shared libraries are
reachable through `PATH` and the dynamic linker.

### Checking GPU acceleration

To see which backend whisper.cpp picked up, transcribe a short silent clip and
look at the startup output:

```bash
ffmpeg -hide_banner -loglevel error -f lavfi -i "anullsrc=r=16000:cl=mono" \
    -t 1 /tmp/silence.wav -y
whisper-cli -m /dev/null /tmp/silence.wav 2>&1 | grep -E "Vulkan|backend"
```

If the output mentions a Vulkan device and `whisper_backend_init_gpu`, GPU
acceleration is active. If it only mentions CPU, the Vulkan build was not
picked up: check that `vulkaninfo` reports your GPU and rebuild with
`-DGGML_VULKAN=1`.

## Models

Models are **not bundled** with Shept. Drop one (or more) `ggml-*.bin` files
into any of the directories Shept searches, in this order:

1. `$SHEPT_MODEL_DIR`: set this to keep models somewhere else entirely
2. **folders you added in the app**: see [Adding your own folders](#adding-your-own-folders)
3. `$XDG_DATA_HOME/shept/models` (`~/.local/share/shept/models`)
4. `$XDG_DATA_HOME/whisper-cpp/models`, whisper.cpp's own convention
5. `~/.cache/whisper.cpp/models`, older upstream default
6. `/usr/share/whisper-cpp/models`
7. `/usr/local/share/whisper-cpp/models`

Every `ggml-*.bin` file found in those directories shows up in the model
dropdown; if the same model exists in several places, the first directory wins.
A model Shept knows the published size of is labelled `tiny (incomplete)` when
the file on disk is shorter than that, and the hint below the dropdown names
both sizes: a copy that was interrupted, or one that was still being written,
would otherwise only be noticed when `whisper-cli` failed on it. The file stays
selectable, because a short file may still transcribe something and hiding it
would look like the model had vanished. The one exception is
`ggml-silero-*.bin`, the voice-activity asset behind
[Skipping silence](#skipping-silence): it is picked up from the same
directories but never listed as a transcription model.

### Downloading from the app

Open **Manage…** next to the model row to see the models Shept can fetch. Each
entry shows its size and what it is good for; press **Download** to fetch it.
Progress is shown per model, a download can be cancelled at any time, and
closing the window cancels the active transfer. A finished model appears in the
model dropdown immediately and is selected without a restart.

Downloads always go to `$XDG_DATA_HOME/shept/models`, which is
`~/.local/share/shept/models` by default. That is the first directory Shept
searches and the only model directory that is writable inside the Flatpak
sandbox, so a model downloaded there is visible to both the Flatpak and a
source checkout. A transfer is written as `ggml-<model>.bin.part` and only
renamed once it is complete, so an unfinished download never shows up in the
dropdown.

An interrupted transfer resumes where it stopped: the next attempt asks the
server for the bytes from the current size of the `.part` file, so a download
that lost the connection at 1 GB continues from there instead of starting
again, and cancelling keeps the partial file for the next try. A dropped
connection, a timeout or an HTTP 5xx is retried three times with a growing
wait; a 404, a full disk or a read-only folder fails at once, since another
attempt cannot change the answer. The `.part` is renamed into place only once
its size matches the published one, and one that is already that size is
renamed without a new transfer; a file larger than the published model cannot
be a prefix of the real one, so it is discarded and fetched again. Only a
`.part` nothing has touched for a week counts as garbage and is removed when
the folder is opened, so old leftovers do not accumulate. There is no checksum
to verify against (HuggingFace does not publish per-file hashes), so Shept only
checks the downloaded size against the published one.

### Adding your own folders

The same **Manage…** window lists every directory Shept searches, labelled with
why it is searched (`SHEPT_MODEL_DIR`, *Added by you*, *Default*, *System*).
Press **Add folder…** to point Shept at models kept elsewhere: an external
drive, a shared network folder, a hand-curated collection. Folders you add are
saved to `settings.json`, keep their order, and are searched before the default
locations, so adding one is enough to make its models win over a duplicate
elsewhere. Removing a folder only stops Shept from looking there; the files stay
untouched.

Inside the Flatpak sandbox only paths under your home directory are visible,
because the manifest grants `home:ro`. A folder such as `/mnt/disk/models` needs
an explicit grant:

```bash
flatpak override --user --filesystem=/mnt/disk/models io.github.psampir.Shept
```

### Deleting models

Each installed model in **Manage…** has a trash button, but only where Shept can
actually remove the file: the containing directory has to be writable. Models in
`~/.local/share/whisper-cpp/models` or `/usr/share/whisper-cpp/models` are
read-only inside the Flatpak, so those rows show a lock instead: deleting them
there would only fail, and Shept says so up front rather than after the fact.
Deletion is disabled while the queue is running, asks for confirmation, and
removes a symlink rather than the file it points to.

Recommended: **`large-v3-turbo`** (~1.62 GB, f16), the best accuracy/speed
balance for most languages, including non-English ones.

To download by hand instead:

```bash
mkdir -p ~/.local/share/shept/models
cd ~/.local/share/shept/models
curl -LO https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin
```

Available models:

| Model                 | File size | Quality     | Speed       |
| --------------------- | --------- | ----------- | ----------- |
| `large-v3-turbo`      | 1.62 GB   | excellent   | very fast   |
| `large-v3-turbo-q5_0` | 0.57 GB   | excellent   | very fast   |
| `small`               | 0.49 GB   | acceptable  | fast        |
| `medium`              | 1.53 GB   | good        | medium      |
| `large-v3`            | 3.09 GB   | best        | slower      |
| `base`                | 0.15 GB   | poor        | very fast   |
| `tiny`                | 0.08 GB   | poorest     | very fast   |

Substitute `ggml-<model>.bin` in the URL above for any of these. Avoid the
`.en` variants unless you only ever transcribe English, since they are
English-only.

If you installed the Fedora COPR package, it also ships a download helper:

```bash
whisper-cpp-download-model large-v3-turbo ~/.local/share/whisper-cpp/models
```

## Usage

Run from a source checkout:

```bash
python3 -m shept
```

Once installed (`pip install .`), the same thing is available as the `shept`
command.

1. **Add files…** queues one or more audio or video files. Dropping files onto
   the window works too, as does passing them on the command line
   (`python3 -m shept a.mp3 b.wav`) or opening them from the desktop. Files
   that are already in the queue, and paths that are not regular files, are
   skipped instead of queued twice.
2. Pick a **Model** (`large-v3-turbo` is selected automatically if present).
   Press **Manage…** to download one if the list is empty or you want another.
3. Pick the source **Language** with **Choose…**. The picker holds every
   language whisper.cpp supports (100 of them) and filters as you type:
   `ukr` finds Ukrainian, `jpn` finds Japanese, and matching is a case-
   insensitive subsequence search over both the name and the code, so
   `pol` puts Polish first without you having to remember the whole word. The
   star on a row marks it as a favourite; favourites sit just under
   **Auto-detect** whenever the search box is empty. Enter takes the first
   match, Escape closes the picker. The language defaults to your system
   locale when whisper supports it, otherwise auto-detect. Leave **Skip
   silence** on: it is the default whenever the VAD asset is installed. See
   [Skipping silence](#skipping-silence) for what it changes and how to fetch
   the asset.
4. Click **Transcribe all**: the button reads **Transcribe** when exactly one
   file is waiting. Files are processed one at a time: the active row shows
   `Transcribing…` with its own progress bar, the overall progress bar tracks
   the current file, and finished rows are marked `Done: N segments`.
   **Cancel** abandons the file that is running (it is marked `Cancelled`) and
   leaves the rest pending, so pressing **Transcribe all** again resumes the
   queue where it stopped.
5. Select a row to read that file's transcript in the text view. The running
   file is selected automatically, and switching rows keeps everything that
   was already transcribed: segments stay attached to their file.
6. Use the toolbar buttons to **copy** the selected transcript as plain text,
   or open the **save** menu:
   - **Selected file** writes that one transcript in the chosen format:
     `.txt` (plain text), `.srt` and `.vtt` (subtitles, with the timestamps
     whisper.cpp reported), `.lrc` (lyrics) or `.json` (timed segments,
     versioned schema). The menu entry decides the content; the filename you
     type in the dialog does not affect it.
   - **Export all…** asks for a folder and writes every file that finished
     successfully into it as `<file name>.<ext>`; failed and cancelled rows are
     skipped, as are files that produced no text. An existing file in that
     folder is never overwritten (the new one gets ` (2)`, ` (3)` appended),
     and very long file names are shortened so the batch cannot stop on a
     name-length error. It uses the format you last picked in **Selected
     file** (plain text until you pick one).
7. **Clear finished** removes completed, failed and cancelled rows from the
   queue without touching anything on disk.
8. The menu at the right end of the header bar opens **About Shept**: the
   version, the MIT licence, the copyright line and links to the project and
   its issue tracker.

### Skipping silence

Whisper does not like silence. Give it a clean recording with twenty quiet
seconds at the end and it tends to invent something: "thank you for watching",
"for more information, visit www.fema.org", a repeat of the last sentence.
Those segments are pure fiction, and they end up in every export.

Shept can drop silence before the model ever sees it, using whisper.cpp's
Silero VAD stage (`--vad -vm`):

- **Skip silence** sits under **Language** in the main window.
- It starts **on** when the VAD asset is installed, and off otherwise.
- Turning it on without the asset does not fail a run: the switch stays off and
  an inline **Download** button appears. Pressing it fetches
  `ggml-silero-v6.2.0.bin` (885 kB) from
  [ggml-org/whisper-vad](https://huggingface.co/ggml-org/whisper-vad) into
  `$XDG_DATA_HOME/shept/models`, and the switch turns on when it lands. Shept
  never touches the network on its own.
- While the switch is on, every file in the batch is transcribed with
  `--vad -vm <path>`.

The asset is not a Whisper model, so it is deliberately absent from the model
dropdown even though it lives in the same directories. Shept looks for the
first `ggml-silero-*.bin` of the expected size in the same places as the
models (see [Models](#models)); a truncated or corrupt file counts as missing,
so the hint offers the download again instead of failing the next run. To
install it by hand:

```bash
mkdir -p ~/.local/share/shept/models
curl -L -o ~/.local/share/shept/models/ggml-silero-v6.2.0.bin \
    https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v6.2.0.bin
```

VAD removes silent stretches from the audio that whisper analyses, so the
speech that remains keeps its position in the file and the timestamps in the
subtitle exports stay correct. It affects transcription only: the audio file
itself is never modified. The VAD threshold and padding knobs whisper.cpp
exposes are not in the interface; they are left at upstream's defaults.

### Settings

Shept remembers the selected **Model**, **Language**, language **favourites**
and **Skip silence** choice in:

```
$XDG_CONFIG_HOME/shept/settings.json     # ~/.config/shept/settings.json
```

Favourites live in the same file as a `favourites` list of language codes, in
the order you starred them; starring or unstarring a language in the picker
writes the file straight away. The list is read tolerantly, so a hand-edited
file survives a stray code, a duplicate or an entry of the wrong type; those
are simply dropped.

Delete that file to return to the defaults: the model Shept picks on a fresh
install, the language your locale implies, no favourites, and Skip silence on
only when the VAD asset is available. The file is written atomically (a crash
while saving cannot leave a half-written one behind), and it is read
defensively: a damaged file, unknown language code or wrongly typed value is
ignored, and a model that has since been removed from disk silently gives way
to the default selection.

### Translating a transcript

Shept does not translate. Whisper's built-in translation only produces English
and is considerably worse than dedicated machine translation. If you need
another language, export the transcript as subtitles (`.srt`) and translate
that file with the tool of your choice: DeepL, for instance, accepts `.srt`
and keeps the timestamps intact.

### Desktop launcher (source checkout only)

The Flatpak bundle and the wheel both install the launcher and the icon
themselves, so there is nothing to do. Only running from a source checkout
leaves you without a menu entry; drop one at
`~/.local/share/applications/io.github.psampir.Shept.desktop`:

```ini
[Desktop Entry]
Type=Application
Name=Shept
GenericName=Audio Transcription
Comment=Local audio transcription with whisper.cpp
Exec=python3 -m shept %U
Path=/path/to/shept
Icon=/path/to/shept/data/icons/hicolor/scalable/apps/io.github.psampir.Shept.svg
Terminal=false
Categories=AudioVideo;Utility;
Keywords=transcribe;transcription;whisper;speech;audio;text;
StartupNotify=true
StartupWMClass=io.github.psampir.Shept
MimeType=audio/mpeg;audio/x-wav;audio/mp4;audio/ogg;audio/opus;audio/flac;audio/x-flac;audio/aac;video/mp4;video/webm;video/x-matroska;video/quicktime;video/x-msvideo;video/x-ms-wmv;video/mp2t;video/x-m4v;video/3gpp;
```

Then refresh the launcher cache (most desktops do this automatically):

```bash
update-desktop-database ~/.local/share/applications/
```

## How it works

Every file in the queue goes through the same two stages. `ffmpeg` decodes it
to a temporary 16 kHz mono WAV file, then `whisper-cli` transcribes that WAV and
Shept parses the timestamped segments from its output as they arrive. Each
segment keeps its start and end time: the text is appended to the text view, the
end time drives the progress bars, and the stored segments are what the
subtitle, lyrics and JSON exports are built from.

Files are transcribed strictly one after another. Decoding and whisper-cli are
already CPU/GPU-bound, so running two of them at once would make both slower and
the progress bars meaningless. Each file's temporary WAV is removed as soon as
its run ends, along with the log the tools write their diagnostics to. If a file
fails (an unreadable container, a missing codec, a non-zero exit from
whisper-cli), its row is marked `Failed` with a short reason and the batch
continues with the next file instead of stopping.

## Performance notes

- Reading the source file is often the bottleneck for large videos stored on
  network mounts. Copy the file to local disk first if it feels slow.
- `whisper-cli` only runs while a file is being transcribed. The GUI process
  itself idles at around 80 MB of RAM and negligible CPU.
- Each file costs one model load, because whisper-cli is a fresh process per
  file. A long batch is therefore cheaper per minute than a short one, and a
  queue of many short clips spends a noticeable share of its time loading the
  model.
- Quantized models (`*-q5_0`, `*-q8_0`) roughly halve memory use with
  negligible quality loss.
- GPU acceleration depends on the whisper.cpp build and the drivers, not on
  Shept; a CPU-only build still works, just slower.

## Troubleshooting

**"whisper-cli not found in PATH"**: install it (see [Install](#install)) and
make sure `which whisper-cli` returns a path. Shept shows the missing tool in a
banner at the top of the window together with the install command for your
distribution, and keeps **Transcribe all** disabled until it is available.

**"ffmpeg not found in PATH"**: install `ffmpeg` with your distribution's
package manager; the same banner explains how.

**"ffprobe not found in PATH"**: `ffprobe` ships with `ffmpeg`, so the usual
cause is a split package (`ffmpeg-free` without the probe) or a `PATH` that
only holds one of them. Installing `ffmpeg` fixes it. Transcription still works
without it, but the progress bars cannot follow the file and stay at 0 %.

**One row says "Failed: …"**: that file could not be processed (damaged or
unsupported container, missing codec, whisper-cli exited non-zero, a corrupt
model). The row's subtitle carries the first lines of the reason; hover it to
see the complete output from the failing tool, which is what a bug report
needs. The rest of the queue keeps running. Remove the row, fix the file, and
add it again.

**Model dropdown shows "(no models found)"**: no `ggml-*.bin` file was found in
any of the directories listed under [Models](#models); the window lists the
paths it searched. Press **Manage…** to download one from inside the app, or
drop a file there by hand. Filenames must start with `ggml-` and end with
`.bin`. Set `SHEPT_MODEL_DIR` if your models live elsewhere.

**No GPU acceleration / very slow**: see *Checking GPU acceleration* above.
On Debian/Ubuntu, double-check that you built with `-DGGML_VULKAN=1` and that
`libvulkan-dev` was installed.

**Output is garbled or English-looking for a non-English language**: you are
probably using a `.en` model. Pick a multilingual one (any model name without
an `.en` suffix).

**Random nonsense at the end of long files**: Whisper hallucinates during
silence, so a recording with a long quiet tail often ends in invented text.
Leave **Skip silence** on (see [Skipping silence](#skipping-silence)); if the
VAD asset is missing, press **Download** in the hint under the switch. As a
fallback you can trim silence with ffmpeg (`-af silenceremove`) before adding
the file.

**The Flatpak ignores my GTK theme**: expected, and not specific to Shept.
Shept uses libadwaita, which draws itself with Adwaita and deliberately ignores
`GTK_THEME`; GTK3 had Flatpak theme extensions, GTK4 has no equivalent. On the
host your theme is applied by `~/.config/gtk-4.0/gtk.css`, which is loaded after
libadwaita's stylesheet and overrides its colours, but Flatpak gives every app
its own `$XDG_CONFIG_HOME`, so that file is invisible inside the sandbox.

To apply your theme to this app only, link the theme's `gtk-4.0` directory into
the sandbox config (adjust the theme name):

```bash
mkdir -p ~/.var/app/io.github.psampir.Shept/config
ln -sfn ~/.themes/Catppuccin-Mauve-Dark/gtk-4.0 \
        ~/.var/app/io.github.psampir.Shept/config/gtk-4.0
```

The link is read through the `home:ro` mount, so theme updates are picked up
without repeating this. Uninstalling the app removes the directory. Dark and
light mode still follow the system setting without any of this.

## Status

Shept is pre-release software under active development. Development happens on
the `dev` branch; `main` only carries released versions. The interface,
packaging and supported platforms may change without notice.

## Changelog

Released versions are documented in [CHANGELOG.md](CHANGELOG.md); the
`[Unreleased]` section there collects what has landed on `dev` since the last
tag.

## License

MIT; see [LICENSE](LICENSE). `whisper.cpp` and the Whisper models are
distributed under their own licenses (MIT for whisper.cpp); none of them are
bundled with Shept.
