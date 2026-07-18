"""Test GPU deconvolution chunking and memory fallback behavior."""

import importlib
import inspect
import sys
import types

import numpy as np


def _import_rlgc(monkeypatch):
    """Import RLGC with a CPU-backed CuPy test double when necessary.

    Parameters
    ----------
    monkeypatch : object
        Value supplied for ``monkeypatch``.

    Returns
    -------
    object
        Result produced by the callable.
    """
    rlgc = None
    try:
        rlgc = importlib.import_module("opm_processing.imageprocessing.rlgc")
    except ModuleNotFoundError as error:
        if error.name != "cupy":
            raise

    class Device:
        id = 0

        def __init__(self, *_args):
            """Initialize a simulated CUDA device.

            Parameters
            ----------
            _args : tuple
                Value supplied for ``args``.

            Returns
            -------
            None
                No value is returned.
            """
            pass

        def use(self):
            """Simulate selecting the CUDA device.

            Parameters
            ----------
            None
                This callable has no parameters.

            Returns
            -------
            None
                No value is returned.
            """
            pass

    cupy = types.ModuleType("cupy")
    cupy.ndarray = np.ndarray
    cupy.ElementwiseKernel = lambda *_args: None
    cupy.cuda = types.SimpleNamespace(
        Device=Device,
        memory=types.SimpleNamespace(OutOfMemoryError=MemoryError),
    )
    if rlgc is None:
        monkeypatch.setitem(sys.modules, "cupy", cupy)
        sys.modules.pop("opm_processing.imageprocessing.rlgc", None)
        rlgc = importlib.import_module("opm_processing.imageprocessing.rlgc")
    else:
        # These are backend-independent chunking tests. Keep them isolated from
        # an installed CuPy package and from whether this host exposes a GPU.
        monkeypatch.setattr(rlgc, "cp", cupy)
    return rlgc


def test_chunked_rlgc_uses_y_only_chunking_api(monkeypatch):
    """Verify chunked RLGC exposes only Y-axis chunk sizing.

    Parameters
    ----------
    monkeypatch : object
        Value supplied for ``monkeypatch``.

    Returns
    -------
    None
        No value is returned.
    """
    rlgc = _import_rlgc(monkeypatch)
    parameters = inspect.signature(rlgc.chunked_rlgc).parameters

    assert parameters["crop_y"].default == 2048
    assert "crop_yx" not in parameters
    assert "crop_z" not in parameters


def test_y_tiles_keep_full_z_and_x(monkeypatch):
    """Verify Y chunks retain the complete Z and X dimensions.

    Parameters
    ----------
    monkeypatch : object
        Value supplied for ``monkeypatch``.

    Returns
    -------
    object
        Result produced by the callable.
    """
    rlgc = _import_rlgc(monkeypatch)
    solver_shapes = []

    def fake_solver(image, _psf, *_args, **_kwargs):
        """Record each solver tile shape and return its image.

        Parameters
        ----------
        image : object
            Value supplied for ``image``.
        _psf : object
            Value supplied for ``psf``.
        _args : tuple
            Value supplied for ``args``.
        _kwargs : dict
            Value supplied for ``kwargs``.

        Returns
        -------
        object
            Result produced by the callable.
        """
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
    """Verify an OOM retry reduces only the Y crop size.

    Parameters
    ----------
    monkeypatch : object
        Value supplied for ``monkeypatch``.

    Returns
    -------
    object
        Result produced by the callable.
    """
    rlgc = _import_rlgc(monkeypatch)
    attempted = []
    successful = []

    def fake_chunked_once(image, psf, **kwargs):
        """Raise one simulated OOM before returning the image.

        Parameters
        ----------
        image : object
            Value supplied for ``image``.
        psf : object
            Value supplied for ``psf``.
        kwargs : dict
            Value supplied for ``kwargs``.

        Returns
        -------
        object
            Result produced by the callable.
        """
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


def test_oom_fallback_step_is_configurable(monkeypatch):
    """Verify the Y-crop fallback decrement is configurable.

    Parameters
    ----------
    monkeypatch : object
        Value supplied for ``monkeypatch``.

    Returns
    -------
    object
        Result produced by the callable.
    """
    rlgc = _import_rlgc(monkeypatch)
    attempted = []

    def fake_chunked_once(image, psf, **kwargs):
        """Raise one simulated OOM and record crop sizes.

        Parameters
        ----------
        image : object
            Value supplied for ``image``.
        psf : object
            Value supplied for ``psf``.
        kwargs : dict
            Value supplied for ``kwargs``.

        Returns
        -------
        object
            Result produced by the callable.
        """
        del psf
        attempted.append(kwargs["crop_y"])
        if len(attempted) == 1:
            raise MemoryError("simulated GPU OOM")
        return image.astype(np.float32)

    monkeypatch.setattr(rlgc, "_chunked_rlgc_once", fake_chunked_once)
    monkeypatch.setattr(rlgc, "clear_rlgc_caches", lambda **_kwargs: None)

    rlgc.chunked_rlgc(
        np.ones((3, 800, 20), dtype=np.uint16),
        np.ones((3, 21, 19), dtype=np.float32),
        crop_y=500,
        fallback_step_y=50,
    )

    assert attempted == [500, 450]
