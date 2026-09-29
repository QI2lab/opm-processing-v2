"""Optional cumulative pipeline timings against the local main branch.

Run with ``pytest tests/benchmarks --run-benchmarks -s``. There are no speed
thresholds. Compilation and one full warmup per implementation are excluded.
"""

import ast
import gc
import importlib.util
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import numba
import numpy as np
import pytest

from opm_processing.imageprocessing import opmtools

pytestmark = [pytest.mark.integration, pytest.mark.benchmark]


@pytest.fixture(scope="module")
def main_reference(tmp_path_factory):
    """Load the reference kernels from git without modifying the checkout.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Temporary storage for an isolated snapshot of main.

    Returns
    -------
    module
        Main's deskew functions and its camera/illumination helpers.
    """
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(
        ["git", "rev-parse", "main"], cwd=root, text=True
    ).strip()
    directory = tmp_path_factory.mktemp("deskew_main")
    package = ModuleType("deskew_benchmark_main")
    package.__path__ = [str(directory)]
    sys.modules[package.__name__] = package
    sources = {}
    for name in ("camera", "opmtools", "process"):
        relative = f"src/opm_processing/{'imageprocessing/' if name != 'process' else ''}{name}.py"
        sources[name] = subprocess.check_output(
            ["git", "show", f"{commit}:{relative}"],
            cwd=root,
            text=True,
            encoding="utf-8",
        )
        if name != "process":
            # Resolve camera imports within the reference snapshot, not this checkout.
            source = sources[name].replace(
                "from opm_processing.imageprocessing.camera import",
                "from .camera import",
            )
            (directory / f"{name}.py").write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(
        f"{package.__name__}.opmtools", directory / "opmtools.py"
    )
    reference = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference
    spec.loader.exec_module(reference)
    names = {
        "_camera_calibrated_image",
        "_apply_illumination_correction",
        "_require_float32",
    }
    functions = [
        node
        for node in ast.parse(sources["process"]).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if any(node.name == "_camera_calibrated_image" for node in functions):
        namespace = {"np": np}
        exec(
            compile(
                ast.Module(body=functions, type_ignores=[]),
                "main_process_helpers",
                "exec",
            ),
            namespace,
        )
        reference.camera_correct = namespace["_camera_calibrated_image"]
        reference.illumination_correct = namespace["_apply_illumination_correction"]
    reference.commit = commit
    reference.tqdm = lambda iterable: iterable
    return reference


@pytest.fixture(scope="module", params=["small", "large", "chunked"])
def acquisition(request):
    """Digitize a known blurred emitter with illumination and photon shot noise.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Selects the realistic OPM scan and detector dimensions.

    Returns
    -------
    tuple
        Size name, uint16 camera data, and float32 detector illumination.
    """
    shapes = {
        "small": (10, 256, 1900),
        "large": (500, 512, 1900),
        "chunked": (5000, 512, 1900),
    }
    shape = shapes[request.param]
    raw = np.empty(shape, np.uint16)
    rng = np.random.default_rng(42)
    row = (np.arange(shape[1])[None, :, None] - (shape[1] - 1) / 2) * 0.115
    x = (np.arange(shape[2])[None, None, :] - (shape[2] - 1) / 2) * 0.115
    y_profile = np.linspace(-1, 1, shape[1], dtype=np.float32)[:, None]
    x_profile = np.linspace(-1, 1, shape[2], dtype=np.float32)[None, :]
    illumination = np.asarray(
        0.6 + 0.4 * np.exp(-0.5 * (x_profile**2 + y_profile**2)), np.float32
    )
    for first in range(0, shape[0], 16):
        last = min(first + 16, shape[0])
        scan = (np.arange(first, last)[:, None, None] - (shape[0] - 1) / 2) * 0.4
        y, z = scan + row * np.cos(np.deg2rad(30)), row * 0.5
        photons = (
            20
            + 1000 * np.exp(-0.5 * ((x / 0.3) ** 2 + (y / 0.3) ** 2 + (z / 0.7) ** 2))
        ) * illumination
        raw[first:last] = np.clip(
            np.rint(rng.poisson(photons) / (0.24 / 0.9) + 100), 0, 65535
        ).astype(np.uint16)
    return request.param, raw, illumination


def render(data, illumination, module, scope, chunked):
    """Run the selected cumulative stages with the default chunk geometry.

    Parameters
    ----------
    data : np.ndarray
        Calibrated float32 data for interpolation alone, otherwise uint16 raw.
    illumination : np.ndarray
        Float32 detector response.
    module : module
        Either the main snapshot or the current production opmtools module.
    scope : int
        One for interpolation, two to include camera, three for illumination.
    chunked : bool
        Use the default 15000-row threshold, overlap, crop, and assembly.

    Returns
    -------
    np.ndarray
        Deskewed volume. Scope three uses the actual production chunk wrapper;
        other chunked scopes omit correction stages but retain its schedule.
    """
    if chunked and scope == 3:
        store = MagicMock()
        store.shape = data.shape
        store.__getitem__.side_effect = data.__getitem__
        return module.chunked_orthogonal_deskew(store, illumination=illumination)
    if not chunked:
        if scope >= 2:
            data = module.camera_correct(data, 100, 0.24 / 0.9)
        if scope == 3:
            data = module.illumination_correct(data, illumination)
        return module.orthogonal_deskew(data)
    shape, _, _, _ = module.deskew_shape_estimator(data.shape, crop_after_deskew=False)
    output = np.zeros((shape[0] // 2, shape[1] - 700, shape[2]), np.float32)
    assert output.shape[1] > 15000
    for start, stop in module.chunk_indices(output.shape[1], 15000):
        first = max(0, int(np.ceil((start - 550 if start else 0) * (0.115 / 0.4))))
        last = min(
            data.shape[0],
            int(
                np.ceil(
                    (stop + 550 if stop < output.shape[1] else stop) * (0.115 / 0.4)
                )
            ),
        )
        chunk = data[first:last]
        if scope == 2:
            chunk = module.camera_correct(chunk, 100, 0.24 / 0.9)
        if module is opmtools:
            deskewed = module._orthogonal_deskew_float32(chunk, zero_initialized=False)
        else:
            deskewed = module.orthogonal_deskew(chunk)
        local = max(0, int(np.rint(start - first * (0.4 / 0.115))))
        retained = deskewed[:, local : local + stop - start]
        output[:, start : start + retained.shape[1]] = retained
        del chunk, deskewed, retained
    gc.collect()
    return output


@pytest.mark.parametrize("threads", [4, 16])
@pytest.mark.parametrize(
    "scope", [1, 2, 3], ids=["interpolation", "camera", "illumination"]
)
def test_pipeline_timing(acquisition, main_reference, threads, scope, monkeypatch):
    """Report warmed medians and verify every output against main.

    Parameters
    ----------
    acquisition : tuple
        Physically generated camera data and detector profile.
    main_reference : module
        Isolated main snapshot with its original correction functions.
    threads : int
        Numba worker count, restored after each case.
    scope : int
        Cumulative stage count passed to ``render``.
    monkeypatch : pytest.MonkeyPatch
        Suppresses progress output during timing.
    """
    name, raw, illumination = acquisition
    monkeypatch.setattr(opmtools, "tqdm", lambda iterable: iterable)
    previous = numba.get_num_threads()
    numba.set_num_threads(threads)
    try:
        data = (
            main_reference.camera_correct(raw, 100, 0.24 / 0.9) if scope == 1 else raw
        )
        results = {}
        reference = render(data, illumination, main_reference, scope, name == "chunked")
        actual = render(data, illumination, opmtools, scope, name == "chunked")
        error = 0.0
        for expected_plane, actual_plane in zip(reference, actual, strict=True):
            np.testing.assert_allclose(
                actual_plane, expected_plane, rtol=1e-6, atol=1e-5
            )
            error = max(error, float(np.max(np.abs(actual_plane - expected_plane))))
        del reference, actual, expected_plane, actual_plane
        repeats = 3 if name == "chunked" else 5
        for label, module in (("main", main_reference), ("current", opmtools)):
            samples = []
            for _ in range(repeats):
                started = time.perf_counter()
                output = render(data, illumination, module, scope, name == "chunked")
                samples.append(time.perf_counter() - started)
                del output
            results[label] = {
                "median_s": statistics.median(samples),
                "trials_s": samples,
            }
        print(
            json.dumps(
                dict(
                    size=name,
                    threads=threads,
                    scope=scope,
                    main_commit=main_reference.commit,
                    max_abs_error=error,
                    **results,
                )
            ),
            flush=True,
        )
    finally:
        numba.set_num_threads(previous)
