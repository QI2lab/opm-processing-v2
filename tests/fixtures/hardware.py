"""CUDA availability and resource cleanup for numerical GPU tests."""

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


@pytest.fixture(scope="module")
def experiment(cupy_gpu):
    """Load the experimental operators after requiring actual CUDA execution.

    Parameters
    ----------
    cupy_gpu
        Fixture proving CUDA execution before importing the audit kernels.

    Returns
    -------
    module
        Audit script containing the experimental count sampler and PSF window.
    """
    from scripts import audit_deconvolution

    return audit_deconvolution
