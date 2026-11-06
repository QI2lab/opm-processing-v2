"""Known AO pupils and native, package-backed tissue geometry for unit tests."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from matplotlib.figure import Figure
from scripts import ao_layer_projections
from scripts.ao_tissue_geometry import TissueGeometry
from yaozarrs import open_group

from opm_processing.dataio.processing_state import ProcessingState


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
class AOLayerCase:
    """Persisted sparse landmarks with registered camera-slab bounds."""

    geometry: TissueGeometry
    volume: np.ndarray
    collection: Any
    state: ProcessingState
    fused: Path
    cache_dir: Path
    expected: dict[int, np.ndarray]

    @property
    def record(self):
        """Return the journal's prescribed registered placement state."""
        return self.state.registration(self.collection.path)


@pytest.fixture
def layer_case(
    tmp_path,
    monkeypatch,
    acquisition_factory,
    position_store_factory,
    fused_store_factory,
    ao_geometry_factory,
):
    """Write two known sparse layers with actual NGFF arrays and journal APIs."""
    volume = np.zeros((1, 1, 8, 5, 7), dtype=np.uint16)
    volume[0, 0, 2, 0, 0] = 10
    volume[0, 0, 3, 0, 0] = 20
    volume[0, 0, 0, 0, 0] = 30
    volume[0, 0, 5, 0, 0] = 999
    volume[0, 0, 5, 3, 5] = 40
    volume[0, 0, 3, 3, 5] = 25
    volume[0, 0, :, 4, 0] = 1000
    stage = ((10, 100, 0), (10, 100, 4), (12, 100, 0), (12, 100, 4))
    raw = acquisition_factory(
        np.zeros((1, 4, 1, 5, 4, 3), dtype=np.uint16),
        name="ao_layer_raw",
        mode="stage",
        stage_positions_zxy=stage,
        pixel_size_um=1,
        scan_axis_step_um=1,
        angle_deg=30,
    )
    collection = position_store_factory(
        np.zeros((1, 4, 1, 2, 3, 3), dtype=np.uint16),
        name="ao_layer_deskewed",
        voxel_size_um=(1, 1, 1),
        stage_positions=stage,
        spatial_offsets_zyx_um=((0, -4, 0),) * 4,
    )
    fused = fused_store_factory(volume, name="ao_layer_fused", origin=(-20, 0, 0))
    preview = fused_store_factory(
        volume.max(axis=2, keepdims=True), name="ao_layer_max_z_fused"
    )
    state = ProcessingState.create(tmp_path / "ao_layer.processing.json", raw.path)
    state.initialize_run(collection.path, configuration={}, overwrite=False)
    state.save_registration(collection.path, configuration={}, pairwise_metrics={})
    origins = ((-18, 0, 0), (-16, 3, 5), (-21, -1, -1), (-17, 3, 5))
    state.complete_registration(
        collection.path,
        fused_path=fused,
        tiles=[
            {"time_index": 0, "position_index": i, "origin_zyx_um": origin}
            for i, origin in enumerate(origins)
        ],
    )
    state.set_registered_max_projection(collection.path, max_projection_path=preview)
    geometry = ao_geometry_factory(
        volume[0, 0].max(axis=0),
        provenance={
            "full_resolution": True,
            "projection": str(preview),
            "processing_state": str(state.path),
        },
    )
    monkeypatch.setattr(ao_layer_projections, "XY_BLOCK_SHAPE", (2, 3))
    expected = {depth: np.zeros((5, 7), dtype=np.uint16) for depth in (0, 1)}
    expected[0][0, 0], expected[0][3, 5] = 20, 40
    expected[1][0, 0], expected[1][3, 5] = 30, 25
    return AOLayerCase(
        geometry,
        volume,
        collection,
        state,
        fused,
        tmp_path / "projection-cache",
        expected,
    )


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
def ao_geometry_case(
    tmp_path,
    acquisition_factory,
    position_store_factory,
    fused_store_factory,
    ao_region_factory,
):
    """Persist image-pyramid calibration and prescribed registered stage assignments."""
    import json
    from types import SimpleNamespace

    from opm_processing.dataio.position_collection import update_group_attributes
    from opm_processing.imageprocessing.coordinates import STAGE_COORDINATE_VERSION

    pixels = np.arange(600 * 800, dtype=np.float32).reshape(600, 800)
    native = np.full((1, 1, 1, 2400, 3200), 123, dtype=np.float32)
    preview = fused_store_factory(
        native,
        name="specimen_max_z_fused",
        spacing=(1, 0.1, 0.2),
        factors=(1, 4),
        level_origins=((0, 0, 0), (0, 0.15, 0.3)),
    )
    open_group(preview)["1"].to_tensorstore().write(pixels[None, None, None]).result()
    ome = dict(open_group(preview).attrs["ome"])
    ome["multiscales"][0]["coordinateTransformations"] = [
        {"type": "scale", "scale": [1, 1, 1, 2, 3]},
        {"type": "translation", "translation": [0, 0, 0, 20, 30]},
    ]
    update_group_attributes(
        preview,
        {
            "ome": ome,
            "opm_processing_coordinates": {"version": STAGE_COORDINATE_VERSION},
        },
    )
    stage = []
    tiles = []
    regions = []
    for depth in range(2):
        layer = tmp_path / f"time_0_z_{depth}_ao_grid_results"
        layer.mkdir()
        correction = 7 * depth
        ao = [
            {"x": x + correction, "y": y, "z": 100 + 100 * depth}
            for y in (15, 30, 45)
            for x in (35, 60, 85)
        ]
        experiment = []
        for scan, x in enumerate((10, 110)):
            for lateral, y in enumerate(range(0, 70, 10)):
                position = len(stage)
                z = 100 + 100 * depth
                stage.append((z, x, y))
                experiment.append(
                    {
                        "z": z,
                        "x": x + correction,
                        "lab_scan_um": x,
                        "y": y,
                        "sample_depth_um": 11.76 * depth,
                    }
                )
                center_y = 100 + 50 * scan + 1.234 * depth
                center_x = 200 + 10 * lateral
                tiles.append(
                    {
                        "time_index": 0,
                        "position_index": position,
                        "origin_zyx_um": [-z, center_y - 20, center_x - 5],
                    }
                )
        (layer / "ao_stage_positions.json").write_text(json.dumps(ao))
        (layer / "exp_stage_positions.json").write_text(json.dumps(experiment))
        regions.extend(
            ao_region_factory(depth=depth, index=index, pupil=np.ma.zeros((2, 2)))
            for index in range(9)
        )
    raw = acquisition_factory(
        np.zeros((1, 28, 1, 5, 4, 5), dtype=np.uint16),
        name="geometry_raw",
        mode="stage",
        stage_positions_zxy=tuple(stage),
    )
    opm = dict(open_group(raw.path).attrs["opm_v2"])
    opm.setdefault("configuration", {}).setdefault("acq_config", {})["AO"] = {
        "num_scan_positions": 3,
        "num_tile_positions": 3,
    }
    update_group_attributes(raw.path, {"opm_v2": opm})
    collection = position_store_factory(
        np.zeros((1, 28, 1, 2, 200, 50), dtype=np.uint16),
        name="specimen_deskewed",
        voxel_size_um=(1, 0.2, 0.2),
        stage_positions=stage,
        attributes={
            "opm_processing_coordinates": {"version": STAGE_COORDINATE_VERSION}
        },
    )
    fused = fused_store_factory(
        np.zeros((1, 1, 2, 1, 1), dtype=np.uint16), name="specimen_fused"
    )
    state = ProcessingState.create(tmp_path / "specimen.processing.json", raw.path)
    state.initialize_run(collection.path, configuration={}, overwrite=False)
    state.save_registration(collection.path, configuration={}, pairwise_metrics={})
    state.complete_registration(collection.path, fused_path=fused, tiles=tiles)
    state.set_registered_max_projection(collection.path, max_projection_path=preview)
    return SimpleNamespace(
        root=tmp_path,
        preview=preview,
        pixels=pixels,
        native=native[0, 0, 0],
        regions=regions,
        state=state,
    )


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
def ao_composite_case(
    tmp_path,
    acquisition_factory,
    position_store_factory,
    fused_store_factory,
    ao_result_factory,
):
    """Persist analytical color bands with real layer geometry and an AO result."""
    import hashlib
    import json
    from types import SimpleNamespace

    from scripts import export_ao_overview
    from scripts.ao_tissue_geometry import read_tissue_geometry

    from opm_processing.dataio.acquisition import ChannelMetadata
    from opm_processing.dataio.position_collection import update_group_attributes
    from opm_processing.imageprocessing.coordinates import STAGE_COORDINATE_VERSION

    volume = np.zeros((1, 4, 2, 100, 100), dtype=np.uint16)
    volume[0, 0, :, :, :30] = 100
    volume[0, 1, :, :, 35:65] = 200
    volume[0, 3, :, :, 70:] = 400
    images = {channel: volume[0, channel, 0] for channel in (0, 1, 3)}
    hashes = {
        str(channel): hashlib.sha256(image.tobytes()).hexdigest()
        for channel, image in images.items()
    }
    stage = ((0, 0, 0),)
    raw = acquisition_factory(
        np.zeros((1, 1, 4, 5, 4, 100), np.uint16),
        name="composite_raw",
        mode="stage",
        stage_positions_zxy=stage,
        pixel_size_um=1,
        scan_axis_step_um=1,
        angle_deg=30,
        channels=tuple(
            ChannelMetadata(index, f"channel-{index}", None, 10, 10)
            for index in range(4)
        ),
    )
    opm = dict(open_group(raw.path).attrs["opm_v2"])
    opm.setdefault("configuration", {}).setdefault("acq_config", {})["AO"] = {
        "num_scan_positions": 1,
        "num_tile_positions": 1,
    }
    update_group_attributes(raw.path, {"opm_v2": opm})
    attributes = {"opm_processing_coordinates": {"version": STAGE_COORDINATE_VERSION}}
    collection = position_store_factory(
        np.zeros((1, 1, 4, 2, 100, 100), np.uint16),
        name="composite_deskewed",
        voxel_size_um=(1, 1, 1),
        stage_positions=stage,
        attributes=attributes,
    )
    fused = fused_store_factory(
        volume, name="composite_fused", origin=(0, 0.5, 0.5), attributes=attributes
    )
    preview = fused_store_factory(
        volume.max(axis=2, keepdims=True),
        name="composite_max_z_fused",
        origin=(0, 0.5, 0.5),
        attributes=attributes,
    )
    state = ProcessingState.create(tmp_path / "composite.processing.json", raw.path)
    state.initialize_run(collection.path, configuration={}, overwrite=False)
    state.save_registration(collection.path, configuration={}, pairwise_metrics={})
    state.complete_registration(
        collection.path,
        fused_path=fused,
        tiles=[{"time_index": 0, "position_index": 0, "origin_zyx_um": (0, 0.5, 0.5)}],
    )
    state.set_registered_max_projection(collection.path, max_projection_path=preview)
    ao_root = tmp_path / "sample_ao_grid_results"
    layer = ao_root / "time_0_z_0_ao_grid_results"
    starting = np.linspace(-0.1234567, 0.2345678, 32)
    final = starting + 0.0123456
    saved = ao_result_factory(path=layer / "grid_pos_0", starting=starting, final=final)
    plan = [{"x": 0, "y": 0, "z": 0}]
    (layer / "ao_stage_positions.json").write_text(json.dumps(plan))
    (layer / "exp_stage_positions.json").write_text(json.dumps(plan))
    regions = [export_ao_overview.Region(0, 0, saved.result, saved.result.get_pupil())]
    geometry = read_tissue_geometry(
        tmp_path,
        preview,
        0,
        0,
        regions,
        planned_grid=True,
        full_resolution=True,
        load_image=False,
    )
    output = tmp_path / "export"
    ao_layer_projections.read_layer_projections(
        geometry,
        timepoint=0,
        channel=1,
        depths=[0],
        cache_dir=output / "projection_cache",
    )
    return SimpleNamespace(
        root=tmp_path,
        ao_root=ao_root,
        preview=preview,
        output=output,
        starting=starting,
        final=final,
        hashes=hashes,
    )


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
