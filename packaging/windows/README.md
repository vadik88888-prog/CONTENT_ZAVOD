# Windows onedir foundation

This directory is the source layout for the friend-beta Windows portable build.
It does not create an installer.

The resulting onedir layout is expected to contain:

```text
ContentFactory/
  ContentFactory.exe        # RuntimeLayout.program
  _internal/                # RuntimeLayout.resources (PyInstaller _MEIPASS)
    config.example.yaml
    app/gui/styles/theme.qss
    tools/                   # RuntimeLayout.tools
      ffmpeg.exe
      ffprobe.exe
      yt-dlp.exe
```

Writable settings, projects, engine work, and run metadata use
`%LOCALAPPDATA%\ContentFactoryData` (`RuntimeLayout.data`), never `_internal` or
the repository. Before a build, place redistributable tool binaries and their
required DLLs in `packaging/windows/tools/`; only `.exe` and `.dll` files are
collected. Licenses must be added alongside the eventual portable artifact.

The next packaging step can run:

```powershell
python -m PyInstaller --clean packaging/windows/ContentFactory.spec
```

The executable starts the desktop by default. Its private
`--content-factory-internal-cli` switch is reserved for child processes created
by the desktop runtime; it dispatches the established CLI inside the same exe.
The spec uses the console bootloader with `hide_console="hide-early"` so the Qt
launch does not leave a console visible while QProcess workers retain usable
stdout/stderr. The entrypoint calls `multiprocessing.freeze_support()` before
application imports.
