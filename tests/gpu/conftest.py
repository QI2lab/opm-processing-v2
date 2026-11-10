"""CUDA resource lifecycle scoped to GPU test modules."""

import pytest


@pytest.fixture(scope="session", autouse=True)
def cupy_gpu():
    """Return CuPy on available CUDA hardware, or skip before GPU data setup."""
    cp = pytest.importorskip("cupy")
    message = "CuPy found no CUDA devices"
    try:
        device_count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as error:
        # Discovery may fail without a driver/device. Numerical errors propagate.
        if error.status not in (35, 100):
            raise
        device_count = 0
        message = f"CUDA device discovery failed: {error}"
    if device_count < 1:
        pytest.skip(message)
    device = cp.cuda.Device(0)
    device.use()
    cp.empty(1)
    device.synchronize()
    return cp


@pytest.fixture(scope="module", autouse=True)
def cuda_cleanup(cupy_gpu):
    """Require CUDA before GPU data setup and release module work buffers."""
    cupy_gpu.cuda.Device(0).use()
    yield
    cupy_gpu.cuda.Stream.null.synchronize()
    cupy_gpu.get_default_memory_pool().free_all_blocks()
    cupy_gpu.get_default_pinned_memory_pool().free_all_blocks()


@pytest.fixture(scope="session")
def torch_gpu(cupy_gpu):
    """Require the actual BaSiC CUDA backend before building optical inputs."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("PyTorch CUDA is unavailable for BaSiC fitting")
    return torch
