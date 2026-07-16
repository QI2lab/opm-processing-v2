import importlib
import inspect
import sys
import types

import numpy as np


def _import_rlgc(monkeypatch):
    try:
        return importlib.import_module("opm_processing.imageprocessing.rlgc")
    except ModuleNotFoundError as error:
        if error.name != "cupy":
            raise

    class Device:
        id = 0

        def __init__(self, *_args):
            pass

        def use(self):
            pass

    cupy = types.ModuleType("cupy")
    cupy.ndarray = np.ndarray
    cupy.ElementwiseKernel = lambda *_args: None
    cupy.cuda = types.SimpleNamespace(
        Device=Device,
        memory=types.SimpleNamespace(OutOfMemoryError=MemoryError),
    )
    monkeypatch.setitem(sys.modules, "cupy", cupy)
    sys.modules.pop("opm_processing.imageprocessing.rlgc", None)
    return importlib.import_module("opm_processing.imageprocessing.rlgc")


def test_chunked_rlgc_uses_y_only_chunking_api(monkeypatch):
    rlgc = _import_rlgc(monkeypatch)
    parameters = inspect.signature(rlgc.chunked_rlgc).parameters

    assert parameters["crop_y"].default == 2048
    assert "crop_yx" not in parameters
    assert "crop_z" not in parameters


def test_y_tiles_keep_full_z_and_x(monkeypatch):
    rlgc = _import_rlgc(monkeypatch)
    solver_shapes = []

    def fake_solver(image, _psf, *_args, **_kwargs):
        solver_shapes.append(image.shape)
        return image.astype(np.float32)

    monkeypatch.setattr(rlgc, "rlgc", fake_solver)
    image = np.ones((5, 1000, 700), dtype=np.uint16)
    result = rlgc._chunked_rlgc_once(
        image,
        np.ones((3, 21, 19), dtype=np.float32),
        crop_y=400,
        release_memory=False,
    )

    assert solver_shapes == [(5, 421, 700), (5, 442, 700), (5, 221, 700)]
    np.testing.assert_array_equal(result, image)


def test_oom_fallback_reduces_only_crop_y(monkeypatch):
    rlgc = _import_rlgc(monkeypatch)
    attempted = []
    successful = []

    def fake_chunked_once(image, psf, **kwargs):
        del psf
        attempted.append((kwargs["crop_y"], image.shape[0], image.shape[2]))
        if len(attempted) == 1:
            raise MemoryError("simulated GPU OOM")
        return image.astype(np.float32)

    monkeypatch.setattr(rlgc, "_chunked_rlgc_once", fake_chunked_once)
    monkeypatch.setattr(rlgc, "clear_rlgc_caches", lambda **_kwargs: None)
    image = np.ones((5, 1000, 700), dtype=np.uint16)
    result = rlgc.chunked_rlgc(
        image,
        np.ones((3, 21, 19), dtype=np.float32),
        crop_y=512,
        on_successful_crop_y=successful.append,
    )

    assert attempted == [(512, 5, 700), (384, 5, 700)]
    assert successful == [384]
    np.testing.assert_array_equal(result, image)
