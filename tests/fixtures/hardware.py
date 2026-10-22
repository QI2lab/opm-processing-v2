"""CPU thread limits, CUDA availability and numerical resource cleanup."""

from __future__ import annotations

import os

import numpy as np
import pytest


@pytest.fixture(scope="module")
def cupy_gpu():
    """Return CuPy after proving CUDA execution, or honor GPU skip policy."""

    def unavailable(message: str) -> None:
        """Fail required GPU runs or skip optional GPU runs."""
        if os.environ.get("OPM_REQUIRE_GPU") == "1":
            pytest.fail(message, pytrace=False)
        pytest.skip(message)

    try:
        import cupy as cp
    except ImportError:
        unavailable("CuPy is not installed; install the project's gpu extra")

    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            unavailable("CuPy found no CUDA devices")
        device = cp.cuda.Device(0)
        device.use()
        probe = cp.arange(32, dtype=cp.float32)
        probe = probe * probe + cp.float32(3)
        # Verify the installed FFT and linear algebra runtimes as well as CUDA.
        cp.fft.rfft(probe)
        cp.tensordot(probe.reshape(4, 8), probe.reshape(8, 4), axes=1)
        device.synchronize()
        np.testing.assert_array_equal(
            cp.asnumpy(probe),
            np.arange(32, dtype=np.float32) ** 2 + 3,
        )
    except pytest.skip.Exception:
        raise
    except Exception as error:  # CUDA errors vary with runtime/driver versions.
        unavailable(f"CUDA execution failed: {error}")

    yield cp

    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


@pytest.fixture
def reconstruct(cupy_gpu):
    """Reconstruct full-grid measurements with the native or acquired-plane solver."""
    import importlib

    def run(observed, psf, factor=None, **options):
        """Apply the shared solver to native or selected acquired scan planes."""
        return importlib.import_module("opm_processing.imageprocessing.rlgc").rlgc(
            observed if factor is None else observed[::factor],
            psf,
            scan_upsample_factor=1 if factor is None else factor,
            boundary="reflect" if factor is None else "zero",
            **options,
        )

    return run


@pytest.fixture(scope="module")
def small_cpu_kernels():
    """Bound worker overhead for small parallel numerical unit-test arrays."""
    from numba import get_num_threads, set_num_threads

    previous = get_num_threads()
    set_num_threads(min(2, previous))
    yield
    set_num_threads(previous)


@pytest.fixture(scope="module")
def nvenc_gpu(cupy_gpu):
    """Require a real hardware encoder, without substituting an encoded movie.

    Returns
    -------
    None
        CUDA and NVENC are available for the real video integration tests.
    """
    import PyNvVideoCodec as nvc

    try:
        capabilities = nvc.GetEncoderCaps(0, "h264")
    except RuntimeError as error:
        if not any(
            word in str(error).lower()
            for word in ("no encode device", "no nvenc capable", "unsupported device")
        ):
            raise
        if os.environ.get("OPM_REQUIRE_GPU") == "1":
            pytest.fail(str(error), pytrace=False)
        pytest.skip(str(error))
    assert capabilities["width_min"] > 0 and capabilities["height_min"] > 0
