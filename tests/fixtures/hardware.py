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

    import cupy as cp

    try:
        device_count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as error:
        # Driver/device discovery may legitimately fail on a CPU-only host.
        # Numerical, allocation, FFT and assertion errors must remain failures.
        if error.status not in (35, 100):  # insufficient driver / no device
            raise
        unavailable(f"CUDA device discovery failed: {error}")
    if device_count < 1:
        unavailable("CuPy found no CUDA devices")
    device = cp.cuda.Device(0)
    device.use()
    probe = cp.arange(32, dtype=cp.float32)
    probe = probe * probe + cp.float32(3)
    cp.fft.rfft(probe)
    cp.tensordot(probe.reshape(4, 8), probe.reshape(8, 4), axes=1)
    device.synchronize()
    np.testing.assert_array_equal(
        cp.asnumpy(probe), np.arange(32, dtype=np.float32) ** 2 + 3
    )

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
    from scipy import ndimage
    from skimage.exposure import match_histograms
    from skimage.measure import block_reduce
    from skimage.metrics import structural_similarity
    from skimage.registration import phase_cross_correlation
    from skimage.registration._masked_phase_cross_correlation import (
        cross_correlate_masked,
    )

    from opm_processing.imageprocessing import tilefusion

    backend = {
        "USING_GPU": False,
        "cp": None,
        "xp": np,
        "ssim_cuda": None,
        "_ssim_cpu": structural_similarity,
        "match_histograms": match_histograms,
        "block_reduce": block_reduce,
        "phase_cross_correlation": phase_cross_correlation,
        "masked_cross_correlation": cross_correlate_masked,
        "sobel_filter": ndimage.sobel,
        "shift_filter": ndimage.shift,
        "minimum_filter": ndimage.minimum_filter,
    }
    for name, operator in backend.items():
        monkeypatch.setattr(tilefusion, name, operator)


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
