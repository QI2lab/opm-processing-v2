"""CUDA runtime library loading for Windows and Linux installations."""

import os
import sys
import sysconfig
import warnings
from pathlib import Path

_CUDA_LIBRARIES = (
    "cudart",
    "nvrtc",
    "cupti",
    "cufft",
    "cublas",
    "curand",
    "cusparse",
    "cusolver",
    "cusolverMg",
    "nvJitLink",
)

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


def preload_cuda_libraries() -> None:
    """Load available CUDA libraries before importing GPU frameworks.

    NVIDIA's loader discovers libraries and their dependencies, handles platform
    differences and retains their handles. Missing packages or optional libraries
    are ignored so CPU-only environments continue to import normally.

    Returns
    -------
    None
        No value is returned.
    """
    global _CUDA_PRELOAD_ATTEMPTED

    if _CUDA_PRELOAD_ATTEMPTED or not sys.platform.startswith(("linux", "win32")):
        return
    _CUDA_PRELOAD_ATTEMPTED = True

    try:
        from cuda.pathfinder import DynamicLibNotFoundError, load_nvidia_dynamic_lib
    except ImportError:
        # CPU installations need not provide CuPy's CUDA loader dependency.
        return
    for name in _CUDA_LIBRARIES:
        try:
            load_nvidia_dynamic_lib(name)
        except DynamicLibNotFoundError:
            continue
