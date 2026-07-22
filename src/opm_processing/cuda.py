"""CUDA runtime library loading for wheel-based Linux installations."""

import ctypes
import os
import sys
import sysconfig
import warnings
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
_CUPY_CUDA_PATH_WARNING_FILTER = (
    "ignore:CUDA path could not be detected:UserWarning:cupy._environment"
)


def suppress_spurious_cupy_cuda_path_warning() -> None:
    """Hide CuPy's expected warning for Windows CUDA 12 component wheels.

    CuPy deliberately reports no single CUDA root for the CUDA 12 PyPI
    component layout because the runtime and NVRTC packages use separate
    directories. Its Windows initialization currently warns whenever that root
    is absent even though ``cuda-pathfinder`` loads those component wheels.
    Only suppress that exact warning when both required wheel layouts exist.

    Parameters
    ----------
    None
        This callable has no parameters.

    Returns
    -------
    None
        No value is returned.
    """
    if not sys.platform.startswith("win32"):
        return

    nvidia_root = Path(sysconfig.get_path("purelib")) / "nvidia"
    component_bins = (
        nvidia_root / "cuda_runtime" / "bin",
        nvidia_root / "cuda_nvrtc" / "bin",
    )
    if not all(path.is_dir() for path in component_bins):
        return

    warnings.filterwarnings(
        "ignore",
        message=(
            r"CUDA path could not be detected\. Set CUDA_PATH environment "
            r"variable if CuPy fails to load\."
        ),
        category=UserWarning,
        module=r"cupy\._environment",
    )

    # ``spawn`` starts a fresh interpreter before importing the target module,
    # so the in-process warnings filter can be too late. Propagate the same
    # narrow filter without altering the handling of any other warning.
    python_warnings = os.environ.get("PYTHONWARNINGS", "")
    filters = [value for value in python_warnings.split(",") if value]
    if _CUPY_CUDA_PATH_WARNING_FILTER not in filters:
        filters.append(_CUPY_CUDA_PATH_WARNING_FILTER)
        os.environ["PYTHONWARNINGS"] = ",".join(filters)


def _nvidia_library_roots() -> tuple[Path, ...]:
    """Return possible roots for NVIDIA runtime wheels in this environment.

    Parameters
    ----------
    None
        This callable has no parameters.

    Returns
    -------
    tuple[Path, ...]
        Result produced by the callable.
    """
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

    Parameters
    ----------
    None
        This callable has no parameters.

    Returns
    -------
    None
        No value is returned.
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
