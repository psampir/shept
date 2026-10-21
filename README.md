# Skryba

Local speech-to-text for your desktop, powered by [whisper.cpp](https://github.com/ggml-org/whisper.cpp).

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
- [License](#license)

## What it does

- Pick any audio or video file (mp3, wav, m4a, ogg, opus, flac, aac, mp4,
  mkv, mov, webm, avi, …)
- Transcribe locally with a Whisper model of your choice
- Streams output as it transcribes, with a progress bar
- Copy or save the transcript

Everything runs on your machine: no accounts, no uploads, no network access
during transcription. No daemons or background processes either — `whisper-cli`
is launched on demand and exits when done, and the model is loaded into memory
only for the duration of a run.

Only file transcription is supported — there is no microphone / live mode.

## Requirements

- Python 3.10+
- GTK 4, libadwaita and PyGObject (`gi`) — system packages, not pip installs
- `ffmpeg` and `ffprobe`
- `whisper-cli` from whisper.cpp (1.7 or newer)

GPU acceleration is optional. A build of whisper.cpp with Vulkan support makes
transcription considerably faster than a CPU-only build on most modern GPUs
(AMD, Intel, NVIDIA); without it, whisper-cli falls back to the CPU.

## Install

Skryba itself is a single Python file; the only real work is getting
`whisper-cli` onto your PATH.

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
shown in the Debian/Ubuntu section above — the CMake steps are the same
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
picked up — check that `vulkaninfo` reports your GPU and rebuild with
`-DGGML_VULKAN=1`.

## Models

Models are **not bundled** with Skryba. Drop one (or more) `ggml-*.bin` files
into any of the directories Skryba searches, in this order:

1. `$SKRYBA_MODEL_DIR` — set this to keep models somewhere else entirely
2. `$XDG_DATA_HOME/skryba/models` (`~/.local/share/skryba/models`)
3. `$XDG_DATA_HOME/whisper-cpp/models` — whisper.cpp's own convention
4. `~/.cache/whisper.cpp/models` — older upstream default
5. `/usr/share/whisper-cpp/models`
6. `/usr/local/share/whisper-cpp/models`

Every `ggml-*.bin` file found in those directories shows up in the model
dropdown; if the same model exists in several places, the first directory wins.

Recommended: **`large-v3-turbo`** (~1.5 GB, f16) — the best accuracy/speed
balance for most languages, including non-English ones.

```bash
mkdir -p ~/.local/share/whisper-cpp/models
cd ~/.local/share/whisper-cpp/models
curl -LO https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin
```

Alternative sizes:

| Model                 | File size | Quality     | Speed       |
| --------------------- | --------- | ----------- | ----------- |
| `large-v3-turbo`      | 1.5 GB    | excellent   | very fast   |
| `large-v3-turbo-q5_0` | 0.7 GB    | excellent   | very fast   |
| `large-v3`            | 3.0 GB    | best        | slower      |
| `medium`              | 1.5 GB    | good        | medium      |
| `small`               | 0.5 GB    | acceptable  | fast        |

Substitute `ggml-<model>.bin` in the URL above for any of these. Avoid the
`.en` variants unless you only ever transcribe English — they are
English-only.

If you installed the Fedora COPR package, it also ships a download helper:

```bash
whisper-cpp-download-model large-v3-turbo ~/.local/share/whisper-cpp/models
```

## Usage

Run from a source checkout:

```bash
python3 skryba.py
```

Once the project is packaged, the same thing is available as the `skryba`
command.

1. **Choose…** an audio or video file
2. Pick a **Model** (`large-v3-turbo` is selected automatically if present)
3. Pick the source **Language** (defaults to your system locale when Skryba
   supports it, otherwise auto-detect)
4. Optionally toggle **Translate to English** to get an English translation
   instead of a transcript in the source language
5. Click **Transcribe**. The progress bar first tracks ffmpeg decoding, then
   the text segments stream in as whisper-cli produces them
6. Use the toolbar buttons to **copy** the transcript or **save** it as `.txt`

### Desktop launcher (optional)

To get Skryba into your application menu, drop a desktop entry at
`~/.local/share/applications/io.github.psampir.Skryba.desktop` and adjust
`Exec=` to the path of your checkout:

```ini
[Desktop Entry]
Type=Application
Name=Skryba
GenericName=Audio Transcription
Comment=Local audio transcription with whisper.cpp
Exec=python3 /path/to/skryba.py
Icon=audio-input-microphone
Terminal=false
Categories=AudioVideo;Utility;
Keywords=transcribe;transcription;whisper;speech;audio;text;
StartupNotify=true
StartupWMClass=io.github.psampir.Skryba
```

Then refresh the launcher cache (most desktops do this automatically):

```bash
update-desktop-database ~/.local/share/applications/
```

## How it works

When you press **Transcribe**, `ffmpeg` decodes the selected file to a
temporary 16 kHz mono WAV file. `whisper-cli` then transcribes that WAV, and
Skryba parses the timestamped segments from its output as they arrive and
appends them to the text view, using the segment timestamps to drive the
progress bar. The temporary WAV is removed when the run finishes or fails.

## Performance notes

- Reading the source file is often the bottleneck for large videos stored on
  network mounts. Copy the file to local disk first if it feels slow.
- `whisper-cli` only runs while the **Transcribe** button is busy. The GUI
  process itself idles at around 80 MB of RAM and negligible CPU.
- Quantized models (`*-q5_0`, `*-q8_0`) roughly halve memory use with
  negligible quality loss.
- GPU acceleration depends on the whisper.cpp build and the drivers, not on
  Skryba; a CPU-only build still works, just slower.

## Troubleshooting

**"whisper-cli not found in PATH"** — install it (see [Install](#install)) and
make sure `which whisper-cli` returns a path. Skryba shows the missing tool in a
banner at the top of the window together with the install command for your
distribution, and keeps **Transcribe** disabled until it is available.

**"ffmpeg not found in PATH"** — install `ffmpeg` with your distribution's
package manager; the same banner explains how.

**Model dropdown shows "(no models found)"** — no `ggml-*.bin` file was found in
any of the directories listed under [Models](#models); the window lists the
paths it searched. Filenames must start with `ggml-` and end with `.bin`. Set
`SKRYBA_MODEL_DIR` if your models live elsewhere.

**No GPU acceleration / very slow** — see *Checking GPU acceleration* above.
On Debian/Ubuntu, double-check that you built with `-DGGML_VULKAN=1` and that
`libvulkan-dev` was installed.

**Output is garbled or English-looking for a non-English language** — you are
probably using a `.en` model. Pick a multilingual one (any model name without
an `.en` suffix).

**Random nonsense at the end of long files** — Whisper sometimes hallucinates
during silence. Trim trailing silence with ffmpeg (`-af silenceremove`) or
clean it up manually.

## Status

Skryba is pre-release software under active development. Development happens on
the `dev` branch; `main` only carries released versions. The interface,
packaging and supported platforms may change without notice.

## License

MIT — see [LICENSE](LICENSE). `whisper.cpp` and the Whisper models are
distributed under their own licenses (MIT for whisper.cpp); none of them are
bundled with Skryba.
