# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata


spec_directory = Path(SPECPATH).resolve()
project_root = spec_directory.parents[1]
datas = [
    (str(project_root / "config.example.yaml"), "."),
    (str(project_root / "app" / "gui" / "styles" / "theme.qss"), "app/gui/styles"),
    (str(spec_directory / "runtime.lock.json"), "manifests"),
    (str(spec_directory / "binaries.lock.json"), "manifests"),
    (str(spec_directory / "youtube-access-runtime.lock.json"), "manifests"),
]
assets_directory = project_root / "assets"
if assets_directory.is_dir():
    datas.append((str(assets_directory), "assets"))

tools_directory = spec_directory / "tools"
youtube_access_runtime = spec_directory / "youtube-access-runtime"
if not youtube_access_runtime.is_dir():
    raise RuntimeError(
        "Pinned YouTube access runtime is missing; run prepare_youtube_access_runtime.py first."
    )
datas.append((str(youtube_access_runtime), "youtube-access-runtime"))
shiboken_directory = Path(__import__("shiboken6").__file__).resolve().parent
binaries = [
    (str(path), "tools")
    for path in tools_directory.iterdir()
    if path.is_file() and path.suffix.casefold() in {".exe", ".dll"}
]
# Keep the Qt binding and its companion MSVC DLLs together.  The runtime hook
# makes this directory the DLL lookup root before the first PySide6 import so
# unrelated extension runtimes collected into ``_internal`` cannot override it.
binaries += [
    (str(path), "PySide6")
    for path in shiboken_directory.iterdir()
    if path.is_file() and path.suffix.casefold() == ".dll"
]
hiddenimports = []
# The PySide6 hook collects the Qt modules used by the app, but Shiboken ships
# additional MSVC helper DLLs beside its binding library.  They are required
# by QtCore at frozen startup and are not a user-installed dependency.
for package in ("faster_whisper", "ctranslate2", "tokenizers", "onnxruntime", "shiboken6"):
    package_datas, package_binaries, package_hiddenimports = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports
hiddenimports += collect_submodules("openai")
hiddenimports += collect_submodules(
    "google.genai", filter=lambda name: "._test" not in name and ".tests" not in name,
)
for distribution in (
    "faster-whisper", "ctranslate2", "av", "tokenizers",
    "huggingface-hub", "openai", "google-genai", "onnxruntime",
):
    datas += copy_metadata(distribution)

a = Analysis(
    [str(spec_directory / "desktop_entrypoint.py")],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(spec_directory / "pyside_runtime_hook.py")],
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
