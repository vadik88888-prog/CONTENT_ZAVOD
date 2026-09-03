from __future__ import annotations

"""Bounded CUDA-runtime checks shared by transcription and diagnostics."""

import ctypes
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_WINDOWS_DLL_DIRECTORY_HANDLES: dict[Path, Any] = {}


@dataclass(frozen=True, slots=True)
class CudaRuntimeProbe:
    """Whether CTranslate2 can use CUDA without a deferred DLL failure."""

    device_count: int
    usable: bool
    fallback_reason: str | None = None
    required_libraries: tuple[str, ...] = ()


def required_cuda_libraries() -> tuple[str, ...]:
    """Return the CTranslate2 CUDA 12 BLAS library for this platform.

    CTranslate2 4.8 bundles cuDNN on Windows, but CUDA's cuBLAS runtime is a
    separate mandatory dependency.  A detected CUDA device alone therefore
    does not prove that a Whisper model can run on it.
    """

    if platform.system() == "Windows":
        return ("cublas64_12.dll",)
    if platform.system() == "Linux":
        return ("libcublas.so.12",)
    return ()


def _load_cuda_library(library: str) -> None:
    loader = getattr(ctypes, "WinDLL", ctypes.CDLL)
    loader(library)


def _windows_cuda_bin_directory() -> Path | None:
    """Return CUDA_PATH/bin when it is a usable Windows CUDA installation."""

    cuda_path = os.environ.get("CUDA_PATH")
    if not cuda_path:
        return None
    cuda_bin = Path(cuda_path) / "bin"
    return cuda_bin if cuda_bin.is_dir() else None


def _add_windows_dll_directory(directory: Path) -> None:
    """Keep CUDA's directory in the process DLL search path for CTranslate2 too."""

    if directory in _WINDOWS_DLL_DIRECTORY_HANDLES:
        return
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is not None:
        _WINDOWS_DLL_DIRECTORY_HANDLES[directory] = add_dll_directory(str(directory))


def _cuda_library_load_target(library: str) -> str:
    """Prefer the configured CUDA installation over an ambient DLL search path."""

    if platform.system() == "Windows":
        cuda_bin = _windows_cuda_bin_directory()
        if cuda_bin is not None:
            _add_windows_dll_directory(cuda_bin)
            candidate = cuda_bin / library
            if candidate.is_file():
                return str(candidate)
    return library


def probe_cuda_runtime() -> CudaRuntimeProbe:
    """Verify CUDA device discovery and every mandatory dynamic dependency.

    Loading the required library is deliberately performed before the Whisper
    model is created.  It avoids CTranslate2 deferring a missing-cuBLAS error
    until the first inference step, where recovery can otherwise take minutes.
    """

    try:
        import ctranslate2
    except ImportError:
        return CudaRuntimeProbe(
            device_count=0,
            usable=False,
            fallback_reason="CTranslate2 CUDA probe is unavailable",
        )
    try:
        device_count = int(ctranslate2.get_cuda_device_count())
    except Exception as error:
        return CudaRuntimeProbe(
            device_count=0,
            usable=False,
            fallback_reason=f"CTranslate2 could not query CUDA devices: {error}",
        )
    if device_count <= 0:
        return CudaRuntimeProbe(
            device_count=device_count,
            usable=False,
            fallback_reason="CTranslate2 reports no CUDA devices",
        )

    libraries = required_cuda_libraries()
    for library in libraries:
        try:
            _load_cuda_library(_cuda_library_load_target(library))
        except OSError as error:
            return CudaRuntimeProbe(
                device_count=device_count,
                usable=False,
                fallback_reason=(
                    f"CUDA runtime incomplete: required {library} is unavailable ({error})"
                ),
                required_libraries=libraries,
            )
    return CudaRuntimeProbe(
        device_count=device_count,
        usable=True,
        required_libraries=libraries,
    )
