# Building and running locally

Needs the GNOME 49 SDK from Flathub (`flatpak install flathub org.gnome.Sdk//49 org.gnome.Platform//49`).
Run from the repository root:

```bash
flatpak-builder --user --install --force-clean build-dir build-aux/io.github.psampir.Shept.yml
flatpak run io.github.psampir.Shept
```
