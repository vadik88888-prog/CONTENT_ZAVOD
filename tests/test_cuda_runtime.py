from __future__ import annotations

import sys
from types import SimpleNamespace

from app.cuda_runtime import probe_cuda_runtime


def test_cuda_device_count_without_required_blas_runtime_is_not_usable(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "ctranslate2",
        SimpleNamespace(get_cuda_device_count=lambda: 1),
    )
    monkeypatch.setattr("app.cuda_runtime.required_cuda_libraries", lambda: ("cublas64_12.dll",))

    def missing_library(library: str) -> None:
        raise OSError("The specified module could not be found")

    monkeypatch.setattr("app.cuda_runtime._load_cuda_library", missing_library)

    probe = probe_cuda_runtime()

    assert probe.device_count == 1
    assert not probe.usable
    assert probe.required_libraries == ("cublas64_12.dll",)
    assert "cublas64_12.dll" in str(probe.fallback_reason)


def test_cuda_is_usable_only_after_all_required_libraries_load(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "ctranslate2",
        SimpleNamespace(get_cuda_device_count=lambda: 1),
    )
    monkeypatch.setattr("app.cuda_runtime.required_cuda_libraries", lambda: ("cublas64_12.dll",))
    monkeypatch.setattr("app.cuda_runtime.platform.system", lambda: "Linux")
    loaded: list[str] = []
    monkeypatch.setattr("app.cuda_runtime._load_cuda_library", loaded.append)

    probe = probe_cuda_runtime()

    assert probe.usable
    assert loaded == ["cublas64_12.dll"]


def test_windows_probe_loads_cuda_path_library_by_absolute_path(monkeypatch, tmp_path) -> None:
    cuda_bin = tmp_path / "CUDA" / "bin"
    cuda_bin.mkdir(parents=True)
    cublas = cuda_bin / "cublas64_12.dll"
    cublas.touch()
    registered: list[str] = []
    loaded: list[str] = []
    monkeypatch.setitem(
        sys.modules,
        "ctranslate2",
        SimpleNamespace(get_cuda_device_count=lambda: 1),
    )
    monkeypatch.setattr("app.cuda_runtime.platform.system", lambda: "Windows")
    monkeypatch.setenv("CUDA_PATH", str(cuda_bin.parent))
    monkeypatch.setattr("app.cuda_runtime.os.add_dll_directory", registered.append, raising=False)
    monkeypatch.setattr("app.cuda_runtime._load_cuda_library", loaded.append)
    monkeypatch.setattr("app.cuda_runtime._WINDOWS_DLL_DIRECTORY_HANDLES", {})

    probe = probe_cuda_runtime()

    assert probe.usable
    assert registered == [str(cuda_bin)]
    assert loaded == [str(cublas)]


def test_windows_probe_uses_normal_loader_when_cuda_path_library_is_missing(monkeypatch, tmp_path) -> None:
    cuda_bin = tmp_path / "CUDA" / "bin"
    cuda_bin.mkdir(parents=True)
    loaded: list[str] = []
    monkeypatch.setitem(
        sys.modules,
        "ctranslate2",
        SimpleNamespace(get_cuda_device_count=lambda: 1),
    )
    monkeypatch.setattr("app.cuda_runtime.platform.system", lambda: "Windows")
    monkeypatch.setenv("CUDA_PATH", str(cuda_bin.parent))
    monkeypatch.setattr("app.cuda_runtime.os.add_dll_directory", lambda _directory: object(), raising=False)
    monkeypatch.setattr("app.cuda_runtime._load_cuda_library", loaded.append)
    monkeypatch.setattr("app.cuda_runtime._WINDOWS_DLL_DIRECTORY_HANDLES", {})

    probe = probe_cuda_runtime()

    assert probe.usable
    assert loaded == ["cublas64_12.dll"]
