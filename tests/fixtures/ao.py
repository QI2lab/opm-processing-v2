"""Known AO pupils and native, package-backed tissue geometry for unit tests."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from matplotlib.figure import Figure
from scripts.ao_tissue_geometry import TissueGeometry


@pytest.fixture
def ao_figure_observer(monkeypatch):
    """Observe completed figures while executing Matplotlib's actual file writer."""
    figures = []
    savefig = Figure.savefig

    def observe(figure, *args, **kwargs):
        if figure not in figures:
            figures.append(figure)
        return savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", observe)
    return figures


@pytest.fixture
def ao_geometry_factory():
    """Build the production geometry record from prescribed physical unit inputs."""

    def create(image, **overrides):
        shape = tuple(image.shape[:2])
        dy, dx = overrides.pop("scale_yx_um", (1.0, 1.0))
        oy, ox = overrides.pop("origin_yx_um", (0.0, 0.0))
        defaults = {
            "image": image,
            "shape_yx": shape,
            "origin_yx_um": (oy, ox),
            "scale_yx_um": (dy, dx),
            "preview_origin_yx_um": (oy, ox),
            "preview_scale_yx_um": (dy, dx),
            "extent_yx_um": (
                oy - dy / 2,
                oy + (shape[0] - 0.5) * dy,
                ox - dx / 2,
                ox + (shape[1] - 0.5) * dx,
            ),
            "centers_by_depth_index": {},
            "depth_offsets": {},
            "provenance": {},
        }
        defaults.update(overrides)
        return TissueGeometry(**defaults)

    return create


@dataclass
class SavedAOCase:
    """Actual opm-v2 saved optimization plus its prescribed modal truth."""

    result: Any
    starting: np.ndarray
    final: np.ndarray
    reader_path: Path


@pytest.fixture
def ao_result_factory(tmp_path):
    """Persist AO-format units through the actual opm-v2 optimization writer."""
    import zarr
    from opm_v2.utils.sensorless_ao import save_optimization_results
    from scripts import ao_saved_results

    counter = iter(range(10000))

    def create(
        *,
        path=None,
        starting=None,
        final=None,
        time=0,
        index=0,
        history=None,
        iterations=1,
        modes=(0,),
        zarr_format=3,
    ):
        starting = np.asarray(
            np.zeros(32) if starting is None else starting, dtype=float
        )
        final = np.asarray(starting if final is None else final, dtype=float)
        if final.shape != starting.shape:
            raise ValueError("AO fixture modal vectors must share a basis")
        history = np.asarray(final[None] if history is None else history, dtype=float)
        directory = path or tmp_path / f"saved_ao_{next(counter)}"
        directory.mkdir(parents=True, exist_ok=True)
        image = np.arange(20, dtype=np.uint16).reshape(4, 5)
        with zarr.config.set({"default_zarr_format": zarr_format}):
            save_optimization_results(
                all_images=np.repeat(image[None], 1 + 2 * len(history), axis=0),
                all_metrics=1 + np.arange(1 + 2 * len(history), dtype=float),
                optimal_images=np.repeat(image[None], len(history), axis=0),
                optimal_metrics=3 + 2 * np.arange(len(history), dtype=float),
                optimal_coeffs=history,
                starting_coeffs=starting,
                starting_metric=1,
                starting_image=image,
                update_status=np.ones(len(history), dtype=bool),
                metadata={
                    "num_iterations": iterations,
                    "num_mode_samples": 1,
                    "modes_to_optimize": list(modes),
                },
                save_dir_path=directory,
            )
        result = ao_saved_results.AOOptimizationResult(
            directory / "ao_results.zarr", time, index
        )
        return SavedAOCase(
            result, starting.copy(), final.copy(), Path(ao_saved_results.__file__)
        )

    return create


@pytest.fixture
def ao_analytical_phase():
    """Evaluate prescribed WaveKit modes using SciPy's independent Jacobi operator."""
    import json

    from scipy.special import eval_jacobi

    convention = Path(__file__).with_name("ao_wavekit_phase_sdk42.json")
    modes = tuple(map(tuple, json.loads(convention.read_text())["modal_basis_nm"]))

    def evaluate(coefficients, *, samples=512, wavelength=0.488, radians=True):
        coordinates = np.linspace(-1, 1, samples)
        x, y = np.meshgrid(coordinates, coordinates)
        radius, angle = np.hypot(x, y), np.arctan2(y, x)
        data = np.zeros_like(radius)
        for coefficient, (n, m) in zip(coefficients, modes, strict=False):
            radial = radius ** abs(m) * eval_jacobi(
                (n - abs(m)) // 2, 0, abs(m), 2 * radius**2 - 1
            )
            angular = np.cos(m * angle) if m > 0 else np.sin(abs(m) * angle)
            if m == 0:
                angular = np.ones_like(angle)
            data += (
                coefficient * np.sqrt((n + 1) * (1 if m == 0 else 2)) * radial * angular
            )
        if radians:
            data *= 2 * np.pi / wavelength
        return np.ma.array(data, mask=radius > 1)

    return evaluate


@pytest.fixture
def ao_region_factory():
    """Build actual region records from prescribed pupil and optional saved result."""
    from scripts.export_ao_overview import Region

    def create(*, pupil, depth=0, index=0, result=None):
        return Region(depth, index, result, np.ma.asarray(pupil))

    return create


@pytest.fixture
def analytical_basis(ao_analytical_phase):
    """Tabulate independent normalized pupils using the shared Jacobi evaluator."""
    import json

    from scripts.ao_wavekit_phase import BASIS_VERSION

    header = json.loads(
        Path(__file__).with_name("ao_wavekit_phase_sdk42.json").read_text()
    )
    assert header["basis_version"] == BASIS_VERSION
    modes = tuple(map(tuple, header["modal_basis_nm"]))
    pupils = [
        ao_analytical_phase(row, samples=33, radians=False)
        for row in np.eye(len(modes))
    ]
    return np.array([pupil.data for pupil in pupils]), pupils[0].mask, modes
