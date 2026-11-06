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
def ao_modal_sequence_factory():
    """Construct prescribed warm starts with the actual shared pupil operator."""
    from types import SimpleNamespace

    from scripts.ao_wavekit_phase import phase_from_modal_coeff
    from scripts.export_ao_overview import Region

    def create(baseline, commands):
        baseline = np.asarray(baseline, dtype=float)
        commands = {
            key: np.asarray(value, dtype=float) for key, value in commands.items()
        }
        records, saved = [], []
        for (depth, index), final in commands.items():
            start = baseline if depth == 0 else commands[depth - 1, index]
            records.append(
                {
                    "source": f"saved_z{depth}_position{index}.zarr",
                    "depth_index": depth,
                    "region_index": index,
                    "starting_coefficients": start.tolist(),
                    "final_coefficients": final.tolist(),
                }
            )
            pupil = phase_from_modal_coeff(
                final,
                samples=512,
                norm=True,
                wavelength=0.488,
                out="radians",
            )
            saved.append(Region(depth, index, None, pupil))
        return SimpleNamespace(
            baseline=baseline,
            commands=commands,
            saved=saved,
            manifest={
                "coefficient_state": "final",
                "wavelength_nm": 488,
                "regions": records,
            },
        )

    return create


@pytest.fixture
def ao_modal_chain(ao_modal_sequence_factory):
    """Provide inherited physical modal states for two sites and three depths."""

    def create():
        reference = np.array([0.3, -0.2, 0.1])
        commands = {
            (depth, index): reference
            + np.array(
                [
                    (0.01, 0.04, 0.09)[depth] * (index + 1),
                    -0.02 * (depth + 1),
                    0.03 * (depth + 1),
                ]
            )
            for index in range(2)
            for depth in range(3)
        }
        case = ao_modal_sequence_factory(reference, commands)
        return case.manifest, case.saved, reference

    return create


@pytest.fixture
def ao_two_position_warm_start(ao_modal_sequence_factory):
    """Supply signed updates with independently prescribed nonzero initial state."""
    from scripts import ao_wavekit_phase

    baseline = np.array([0.2, -0.1])
    final = {
        (0, 0): np.array([0.5, 0.4]),
        (1, 0): np.array([0.9, 0.7]),
        (2, 0): np.array([0.6, 1.2]),
        (0, 1): np.array([0.1, 0.3]),
        (1, 1): np.array([-0.2, 0.1]),
        (2, 1): np.array([-0.5, 0.6]),
    }
    case = ao_modal_sequence_factory(baseline, dict(sorted(final.items())))
    return baseline, final, case.manifest, case.saved, ao_wavekit_phase


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
