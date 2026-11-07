"""CPU thread limits, CUDA availability and numerical resource cleanup."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="module")
def cupy_gpu():
    """Return CuPy on available CUDA hardware, or honor GPU skip policy."""
    import cupy as cp

    message = "CuPy found no CUDA devices"
    try:
        device_count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as error:
        # Driver/device discovery may legitimately fail on a CPU-only host.
        # Numerical, allocation, FFT and assertion errors must remain failures.
        if error.status not in (35, 100):  # insufficient driver / no device
            raise
        device_count = 0
        message = f"CUDA device discovery failed: {error}"
    if device_count < 1:
        if os.environ.get("OPM_REQUIRE_GPU") == "1":
            pytest.fail(message, pytrace=False)
        pytest.skip(message)
    device = cp.cuda.Device(0)
    device.use()
    cp.empty(1)
    device.synchronize()

    yield cp

    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


@pytest.fixture(autouse=True)
def cpu_registration_backend(monkeypatch, request):
    """Select the actual CPU operators for cases marked as CPU execution.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Restore the previously selected numerical backend after each case.
    request : pytest.FixtureRequest
        Collected CPU/GPU marker supplied by the test directory.
    """
    if request.node.get_closest_marker("cpu") is None:
        return
    from opm_processing.imageprocessing import tilefusion

    for name, operator in tilefusion._cpu_registration_backend().items():
        monkeypatch.setattr(tilefusion, name, operator)


@pytest.fixture(scope="module")
def small_cpu_kernels():
    """Bound worker overhead for small parallel numerical unit-test arrays."""
    from numba import get_num_threads, set_num_threads

    previous = get_num_threads()
    set_num_threads(min(2, previous))
    yield
    set_num_threads(previous)
