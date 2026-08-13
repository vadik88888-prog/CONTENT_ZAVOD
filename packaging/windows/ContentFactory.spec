# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path


spec_directory = Path(SPECPATH).resolve()
project_root = spec_directory.parents[1]
datas = [
    (str(project_root / "config.example.yaml"), "."),
    (str(project_root / "app" / "gui" / "styles" / "theme.qss"), "app/gui/styles"),
]
assets_directory = project_root / "assets"
if assets_directory.is_dir():
    datas.append((str(assets_directory), "assets"))

tools_directory = spec_directory / "tools"
binaries = [
    (str(path), "tools")
    for path in tools_directory.iterdir()
    if path.is_file() and path.suffix.casefold() in {".exe", ".dll"}
]

a = Analysis(
    [str(spec_directory / "desktop_entrypoint.py")],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ContentFactory",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    # Keep standard handles available to the internal CLI started by QProcess.
    # The console bootloader hides a console it owns before the Qt shell starts.
    console=True,
    hide_console="hide-early",
    disable_windowed_traceback=False,
)
collect = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="ContentFactory",
)
