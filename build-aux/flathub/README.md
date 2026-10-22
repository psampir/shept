# Flathub submission

The two files in this directory are the ones that go upstream —
`io.github.psampir.Shept.yml` and `flathub.json` — plus this note. Everything
here is a copy of `build-aux/io.github.psampir.Shept.yml` with the `shept`
module pointed at the release tarball instead of the local checkout; the
differences are marked `FLATHUB:` in the manifest.

## Before opening the pull request

1. **Make the repository public.** Flathub's builders fetch the source over the
   network; a private `psampir/shept` fails the build and the AppStream
   screenshots stay 404.

2. **Tag the release and push the tag.** The manifest asks for
   `https://github.com/psampir/shept/archive/refs/tags/v0.1.0.tar.gz`, so the
   tag `v0.1.0` has to exist on the commit being submitted — the same version
   that `pyproject.toml`, `shept/__init__.py` and the `<release>` entry in
   `data/io.github.psampir.Shept.metainfo.xml` already carry.

3. **Fill in the checksum.** `sha256` is the literal placeholder
   `TODO_SHA256_OF_V0.1.0_TARBALL` until this is done:

   ```bash
   curl -L -o /tmp/shept-0.1.0.tar.gz \
       https://github.com/psampir/shept/archive/refs/tags/v0.1.0.tar.gz
   sha256sum /tmp/shept-0.1.0.tar.gz
   ```

   Paste the first field over the placeholder. If the tag is ever re-created,
   the tarball changes and the checksum has to be updated with it — the same
   goes for every later release, which also needs a new `<release>` entry in
   `data/io.github.psampir.Shept.metainfo.xml`.

4. **Build it locally** with the checksum filled in and the repository public:

   ```bash
   flatpak-builder --user --install --force-clean build-dir \
       build-aux/flathub/io.github.psampir.Shept.yml
   flatpak run io.github.psampir.Shept
   ```

   `flatpak-builder --repo` and `flatpak build-update-repo` are not needed;
   Flathub does its own build with the same manifest.

## Opening the pull request

Flathub expects the manifest at the root of its own repository, not under
`build-aux/`. Only these two files are submitted:

```bash
gh repo fork flathub/flathub --clone --remote
cd flathub
git checkout -b io.github.psampir.Shept
cp /path/to/shept/build-aux/flathub/io.github.psampir.Shept.yml .
cp /path/to/shept/build-aux/flathub/flathub.json .
git add io.github.psampir.Shept.yml flathub.json
git commit -m "Add io.github.psampir.Shept"
git push -u origin io.github.psampir.Shept
gh pr create --repo flathub/flathub --title "Add io.github.psampir.Shept" \
    --body "Local speech-to-text for your desktop, powered by whisper.cpp."
```

Flathub's CI then runs `flatpak-builder` on the manifest, validates the
AppStream metadata with `appstreamcli` and checks the screenshots. The
screenshots are pinned to the `v0.1.0` tag, so they only resolve once step 1
and step 2 above are done.

## flathub.json

`flathub.json` is strict JSON — it takes no comments — so the policy behind it
is written down here instead:

- `only-arches` is `["x86_64"]`: that is the only architecture the bundled
  whisper.cpp + Vulkan stack and `ffmpeg` build have been tested on. Adding
  `aarch64` later means testing a build there first, not just deleting the key.
- `end-of-life` is deliberately **not** set. It is only for retiring an app
  from Flathub (`"end-of-life": "<date>"`, plus `end-of-life-rebase` pointing
  at the replacement app id); setting it now would mark a brand-new submission
  as dead.
