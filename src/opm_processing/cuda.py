"""CUDA runtime library loading for wheel-based Linux installations."""

import ctypes
import sys
import sysconfig
from pathlib import Path


_CUDA_LIBRARIES = (
    ("cuda_runtime", ("libcudart.so.12",)),
    ("cuda_nvrtc", ("libnvrtc.so.12",)),
    ("cuda_cupti", ("libcupti.so.12",)),
    ("cufft", ("libcufft.so.11",)),
    ("cublas", ("libcublasLt.so.12", "libcublas.so.12")),
    ("curand", ("libcurand.so.10",)),
    ("cusparse", ("libcusparse.so.12",)),
    ("cusolver", ("libcusolver.so.11", "libcusolverMg.so.11")),
    ("nvjitlink", ("libnvJitLink.so.12",)),
)

_CUDA_LIBRARY_HANDLES: list[ctypes.CDLL] = []
_CUDA_PRELOAD_ATTEMPTED = False


def _nvidia_library_roots() -> tuple[Path, ...]:
    """Return possible roots for NVIDIA runtime wheels in this environment."""
    candidates = (
        Path(sysconfig.get_path("purelib")) / "nvidia",
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "nvidia",
    )
    return tuple(dict.fromkeys(candidates))


def preload_cuda_libraries() -> None:
    """Load CUDA wheel libraries globally before importing GPU frameworks.

    Linux NVIDIA wheels keep shared libraries in package-specific ``lib``
    directories that are not always visible to the dynamic linker. Missing
    libraries are ignored so CPU-only environments continue to import normally.
    """
    global _CUDA_PRELOAD_ATTEMPTED

    if _CUDA_PRELOAD_ATTEMPTED or not sys.platform.startswith("linux"):
        return
    _CUDA_PRELOAD_ATTEMPTED = True

    roots = _nvidia_library_roots()
    for package, library_names in _CUDA_LIBRARIES:
        for library_name in library_names:
            library_path = next(
                (
                    root / package / "lib" / library_name
                    for root in roots
                    if (root / package / "lib" / library_name).is_file()
                ),
                None,
            )
            if library_path is None:
                continue
            try:
                handle = ctypes.CDLL(str(library_path), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                continue
            _CUDA_LIBRARY_HANDLES.append(handle)
