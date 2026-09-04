# Windows onedir foundation

This directory builds the friend-beta Windows x64 portable ZIP. It does not
create an installer.

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
      deno.exe
    youtube-access-runtime/  # pinned BGutil plugin + Deno server/node_modules
```

Writable settings, projects, engine work, and run metadata use
`%LOCALAPPDATA%\ContentFactoryData` (`RuntimeLayout.data`), never `_internal` or
the repository. Exact runtime and tool inputs are pinned in `runtime.lock.json`
and `binaries.lock.json`. Before a build, place matching binaries in
`packaging/windows/tools/`; only `.exe` and `.dll` files are collected.

Install the pinned build toolchain, build, and smoke a fresh ZIP extraction:

```powershell
python -m pip install -r packaging/windows/build-requirements.txt
python packaging/windows/prepare_binaries.py
python packaging/windows/prepare_youtube_access_runtime.py
python packaging/windows/build_portable.py
python packaging/windows/smoke_portable.py
```

Outputs:

- `packaging/windows/artifacts/ContentFactory-beta-win-x64/`
- `packaging/windows/artifacts/ContentFactory-beta-win-x64.zip`
- `packaging/windows/reports/ContentFactory-beta-win-x64.build.json`
- `packaging/windows/reports/SHA256SUMS`

The executable starts the desktop by default. Its private
`--content-factory-internal-cli` switch is reserved for child processes created
by the desktop runtime; it dispatches the established CLI inside the same exe.
The spec uses PyInstaller's windowed bootloader so a normal double-click opens
only the Qt desktop window. For the private internal CLI switch, the entrypoint
restores QProcess-provided stdout/stderr pipes before importing the CLI, without
attaching a console. It also calls `multiprocessing.freeze_support()` before
application imports.
