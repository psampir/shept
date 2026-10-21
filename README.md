# Whisper GUI

A small GTK4 + libadwaita wrapper around [`whisper.cpp`](https://github.com/ggml-org/whisper.cpp)
for transcribing local audio and video files. Designed for offline use, with
optional GPU acceleration via Vulkan.

## What it does

- Pick any audio or video file (mp3, wav, m4a, ogg, opus, flac, aac, mp4,
  mkv, mov, webm, avi, …)
- Transcribe locally with a Whisper model of your choice
- Streams output as it transcribes, with a progress bar
- Copy or save the transcript

No daemons, no background processes — `whisper-cli` is launched on demand and
exits when done. The model is loaded into VRAM/RAM only during the run.

Only supports file transcription — there is no microphone / live mode.

## Requirements

- Python 3.10+
- GTK 4, libadwaita, PyGObject (`gi`)
- `ffmpeg` (any recent version)
- `whisper-cli` from whisper.cpp 1.7+ (Vulkan build strongly recommended for
  speed)

A Vulkan-capable GPU is optional but makes things 5–30× faster than pure
CPU. Tested on AMD Radeon 8060S (Strix Halo, RADV) and works on any modern
AMD/Intel/NVIDIA GPU with Vulkan drivers.

## Installing whisper-cli

### Fedora 40+

The Fedora package only ships the library. Use the COPR for a CLI binary
with Vulkan enabled:

```bash
sudo dnf copr enable -y sneed/whisper-cpp-vulkan
sudo dnf install -y whisper-cpp ffmpeg python3-gobject \
    gtk4 libadwaita
```

(`ffmpeg` is in RPM Fusion Free; if you don't have RPM Fusion enabled you
can use `ffmpeg-free` from the main repos.)

### Arch / Manjaro / EndeavourOS

`whisper.cpp-vulkan` is in the AUR:

```bash
# with paru / yay / your AUR helper of choice:
paru -S whisper.cpp-vulkan
sudo pacman -S ffmpeg python-gobject gtk4 libadwaita
```

CPU-only alternative (official repos): `pacman -S whisper.cpp` (note: package
name in AUR; check `pacman -Ss whisper`).

### Ubuntu / Debian / Pop!_OS

`whisper.cpp` isn't packaged. Build from source — it's quick:

```bash
# runtime + build deps
sudo apt install -y build-essential cmake git ffmpeg \
    python3-gi python3-gi-cairo gir1.2-gtk-4.0 gir1.2-adw-1 \
    libvulkan-dev glslc          # Vulkan build (skip if CPU-only)

git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp
cmake -B build -DGGML_VULKAN=1   # drop -DGGML_VULKAN=1 for CPU-only
cmake --build build -j --config Release

# install whisper-cli to /usr/local/bin
sudo install -Dm755 build/bin/whisper-cli /usr/local/bin/whisper-cli
sudo install -Dm755 build/src/libwhisper.so* /usr/local/lib/
sudo ldconfig
```

### Verify Vulkan is detected

```bash
ffmpeg -hide_banner -loglevel error -f lavfi -i "anullsrc=r=16000:cl=mono" \
    -t 1 /tmp/silence.wav -y
whisper-cli -m /dev/null /tmp/silence.wav 2>&1 | grep -E "Vulkan|backend"
```

You should see something like:

```
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon 8060S Graphics (RADV STRIX_HALO)
whisper_backend_init_gpu: using Vulkan0 backend
```

If you only see CPU info, the Vulkan build wasn't picked up — check that
`vulkaninfo` reports your GPU and rebuild with `-DGGML_VULKAN=1`.

## Downloading a model

The GUI auto-discovers any `ggml-*.bin` file in
`~/.local/share/whisper-cpp/models/`.

Recommended: **`large-v3-turbo`** (~1.5 GB, f16) — best accuracy/speed
balance for non-English languages including Polish.

```bash
mkdir -p ~/.local/share/whisper-cpp/models
cd ~/.local/share/whisper-cpp/models
curl -LO https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin
```

Alternative sizes:

| Model                       | File size | Quality (PL) | Speed       |
| --------------------------- | --------- | ------------ | ----------- |
| `large-v3-turbo`            | 1.5 GB    | excellent    | very fast   |
| `large-v3-turbo-q5_0`       | 0.7 GB    | excellent    | very fast   |
| `large-v3`                  | 3.0 GB    | best         | slower      |
| `medium`                    | 1.5 GB    | good         | medium      |
| `small`                     | 0.5 GB    | acceptable   | fast        |

Replace the URL above with `ggml-<model>.bin` for any of these. Avoid the
`.en` variants — they only support English.

If you have the Fedora COPR package installed, the helper script also works:

```bash
whisper-cpp-download-model large-v3-turbo ~/.local/share/whisper-cpp/models
```

## Installing the GUI

```bash
mkdir -p ~/scripts
git clone https://github.com/psampir/dotfiles.git /tmp/dotfiles  # or curl the file
cp /tmp/dotfiles/scripts/whisper-gui/whisper_gui.py ~/scripts/whisper-gui/
chmod +x ~/scripts/whisper-gui/whisper_gui.py
```

Or just drop `whisper_gui.py` anywhere convenient and adjust the `Exec=`
line below accordingly.

Test:

```bash
python3 ~/scripts/whisper-gui/whisper_gui.py
```

### Desktop launcher entry

Save the following as `~/.local/share/applications/whisper-gui.desktop`
(adjust `Exec=` to point at wherever you put the script):

```ini
[Desktop Entry]
Type=Application
Name=Whisper GUI
GenericName=Audio Transcription
Comment=Local audio transcription with whisper.cpp
Exec=python3 /home/YOURUSER/scripts/whisper-gui/whisper_gui.py
Icon=audio-input-microphone
Terminal=false
Categories=AudioVideo;Utility;
Keywords=transcribe;transcription;whisper;speech;audio;text;
StartupNotify=true
StartupWMClass=ink.psampir.WhisperGui
```

Then refresh the launcher cache (most desktops do this automatically):

```bash
update-desktop-database ~/.local/share/applications/
```

## Usage

1. Launch *Whisper GUI* from your application menu (or
   `python3 ~/scripts/whisper-gui/whisper_gui.py`)
2. **Choose…** an audio or video file
3. Pick a **Model** (auto-selects `large-v3-turbo` if present)
4. Pick the source **Language** (defaults to Polish)
5. Optionally toggle **Translate to English** to get an English translation
   instead of a transcript
6. Click **Transcribe**. The progress bar shows ffmpeg decoding first, then
   whisper-cli streams text segments
7. Use the toolbar buttons to **copy** the transcript or **save** it as
   `.txt`

## Performance notes

- Reading the source file is often the bottleneck for big videos on
  network mounts. Copy the file to local disk first if it feels slow.
- `whisper-cli` only runs while the **Transcribe** button is busy. The GUI
  process itself idles at ~80 MB RAM and 0% CPU.
- On unified-memory systems (like AMD Strix Halo) the model lives in normal
  RAM via GTT — no PCIe transfer cost. Expect ~25–30× realtime with
  `large-v3-turbo` + Vulkan.
- Quantized models (`*-q5_0`, `*-q8_0`) cut memory roughly in half with
  negligible quality loss.

## Troubleshooting

**"whisper-cli not found in PATH"** — install it (see above) and make sure
`which whisper-cli` returns a path.

**"ffmpeg not found in PATH"** — `sudo dnf/apt/pacman install ffmpeg`.

**Model dropdown shows "(no models found)"** — download a model into
`~/.local/share/whisper-cpp/models/` (filenames must start with `ggml-` and
end with `.bin`).

**No Vulkan acceleration / very slow** — verify with the snippet under
*Verify Vulkan is detected*. On Ubuntu/Debian, double-check that you built
with `-DGGML_VULKAN=1` and that `libvulkan-dev` was installed.

**Polish output is garbled / English-looking** — you're probably using a
`.en` model. Pick a multilingual one (any model name without `.en` suffix).

**Random nonsense at the end of long files** — Whisper sometimes
hallucinates during silence. Trim trailing silence with ffmpeg
(`-af silenceremove`) or accept and clean up manually.

## License

Same license as the dotfiles repo it ships in (MIT). The bundled
`whisper.cpp` and Whisper models are MIT and MIT-licensed respectively.
