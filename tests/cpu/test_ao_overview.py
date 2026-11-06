"""Export saved AO pupils directly using the SJS publication presentation."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from scripts import export_ao_overview as overview
from scripts.ao_publication_style import write_publication_pupil_grid
from scripts.ao_wavekit_phase import BASIS_VERSION
from scripts.export_ao_overview import read_regions, resolve_ao_root

pytestmark = pytest.mark.unit


@pytest.fixture
def pupil_reader():
    """Supply native saved coefficients and a deliberately forbidden old renderer."""

    class Result:
        def __init__(self, path, time, position):
            self.zarr_path = Path(path)
            self.time_index = time
            self.position_index = position
            self.starting_coefficients = np.linspace(-0.1234567, 0.2345678, 32)
            self.final_coefficients = self.starting_coefficients + 0.0123456
            self.pupil = np.ma.array(
                [[-np.pi, 0.1234567], [position + 0.2345678, 0]],
                mask=[[False, False], [False, True]],
            )
            self.calls = []

        def get_pupil(self, **kwargs):
            raise AssertionError("The legacy reader must only load saved coefficients")

    return SimpleNamespace(AOOptimizationResult=Result)


@pytest.fixture
def phase_calls(monkeypatch):
    """Capture coefficient selection while keeping the SDK renderer independently tested."""
    calls = []

    def phase_from_modal_coeff(coefficients, **kwargs):
        calls.append((np.array(coefficients, copy=True), kwargs))
        return np.ma.array(
            [[coefficients[0], coefficients[1]], [coefficients[2], 0]],
            mask=[[False, False], [False, True]],
        )

    monkeypatch.setattr(overview, "phase_from_modal_coeff", phase_from_modal_coeff)
    return calls


def test_direct_ao_results_keep_numeric_order_and_saved_values(
    tmp_path, pupil_reader, phase_calls
):
    """All nine saved corrections load without stage plans or processed images."""
    layer = tmp_path / "time_0_z_0_ao_grid_results"
    for index in (10, 7, 4, 1, 6, 3, 0, 5, 2):
        (layer / f"grid_pos_{index}" / "ao_results.zarr").mkdir(parents=True)
    args = argparse.Namespace(
        timepoint=0, depth_index=None, coefficient_state="final", wavelength_nm=488
    )
    regions = read_regions(tmp_path, pupil_reader, args)
    assert [region.index for region in regions] == [0, 1, 2, 3, 4, 5, 6, 7, 10]
    assert len(phase_calls) == len(regions)
    for region, (coefficients, kwargs) in zip(regions, phase_calls, strict=True):
        np.testing.assert_array_equal(coefficients, region.result.final_coefficients)
        np.testing.assert_array_equal(kwargs.pop("mode_indices"), np.arange(1, 33))
        assert kwargs == {
            "samples": 512,
            "norm": True,
            "wavelength": 0.488,
            "out": "radians",
        }
        np.testing.assert_array_equal(
            region.pupil.data[:2, 0], region.result.final_coefficients[[0, 2]]
        )
        np.testing.assert_array_equal(region.pupil.mask, region.result.pupil.mask)
        np.testing.assert_array_equal(
            region.result.starting_coefficients,
            np.linspace(-0.1234567, 0.2345678, 32),
        )
        np.testing.assert_array_equal(
            region.result.final_coefficients,
            np.linspace(-0.1234567, 0.2345678, 32) + 0.0123456,
        )


@pytest.mark.parametrize("state", ("starting", "final", "change"))
def test_time_and_depth_selection_use_the_requested_saved_state(
    tmp_path, pupil_reader, phase_calls, state
):
    """Select AO layers without relying on acquisition time/depth metadata."""
    for time, depth in ((0, 0), (1, 0), (1, 2), (1, 5)):
        (
            tmp_path
            / f"time_{time}_z_{depth}_ao_grid_results"
            / "grid_pos_0"
            / "ao_results.zarr"
        ).mkdir(parents=True)
    args = argparse.Namespace(
        timepoint=1, depth_index=2, coefficient_state=state, wavelength_nm=561
    )
    regions = read_regions(tmp_path, pupil_reader, args)
    assert [(region.depth, region.index) for region in regions] == [(2, 0)]
    assert regions[0].result.time_index == 1
    coefficients, kwargs = phase_calls[0]
    result = regions[0].result
    expected = {
        "starting": result.starting_coefficients,
        "final": result.final_coefficients,
        "change": result.final_coefficients - result.starting_coefficients,
    }[state]
    np.testing.assert_array_equal(coefficients, expected)
    assert kwargs["wavelength"] == 0.561


def test_main_reader_uses_the_sdk_horizontal_first_tilt(tmp_path):
    """The corrected first mode is an X slope, fixing the old sine/cosine swap."""
    store = tmp_path / "time_0_z_0_ao_grid_results" / "grid_pos_0" / "ao_results.zarr"
    store.mkdir(parents=True)

    class Result:
        def __init__(self, *args):
            self.starting_coefficients = np.zeros(32)
            self.final_coefficients = np.zeros(32)
            self.final_coefficients[0] = 0.1

        def get_pupil(self, **kwargs):
            raise AssertionError("The old SJS basis must not reconstruct new exports")

    args = argparse.Namespace(
        timepoint=0, depth_index=None, coefficient_state="final", wavelength_nm=488
    )
    pupil = read_regions(tmp_path, SimpleNamespace(AOOptimizationResult=Result), args)[
        0
    ].pupil
    y, x = np.meshgrid(np.linspace(-1, 1, 512), np.linspace(-1, 1, 512), indexing="ij")
    expected = 2 * np.pi / 0.488 * 0.1 * 2 * x
    valid = x * x + y * y <= 1
    np.testing.assert_array_equal(pupil.mask, ~valid)
    np.testing.assert_allclose(pupil.data[valid], expected[valid], atol=1e-14)


def test_export_manifest_identifies_the_local_reconstruction_basis(
    tmp_path, pupil_reader, monkeypatch
):
    """New exports retain native coefficients and fingerprint the corrected basis."""
    (tmp_path / "sample_ao_grid_results" / "grid_pos_0" / "ao_results.zarr").mkdir(
        parents=True
    )
    reader_path = tmp_path / "saved_reader.py"
    reader_path.write_text("# Reader source reference\n", encoding="utf-8")
    reference = tmp_path / "plot_reference.py"
    reference.write_text("# Publication style reference\n", encoding="utf-8")
    output = tmp_path / "output"
    monkeypatch.setattr(overview, "load_reference", lambda _: pupil_reader)
    monkeypatch.setattr(
        overview, "write_publication_pupil_grid", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(
        overview.sys,
        "argv",
        [
            "export_ao_overview.py",
            str(tmp_path),
            "--output",
            str(output),
            "--ao-reader",
            str(reader_path),
            "--plot-reference",
            str(reference),
        ],
    )
    overview.main()
    manifest = json.loads((output / "ao_publication.json").read_text())
    provenance = manifest["pupil_reconstruction"]
    source = Path(provenance["source"])
    assert source.name == "ao_wavekit_phase.py"
    assert provenance["basis_version"] == BASIS_VERSION
    assert (
        provenance["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    )
    assert provenance["quantity"].startswith("requested modal correction target")
    assert provenance["coefficient_state"] == "final"
    assert provenance["normalization"] == "RMS"
    assert provenance["samples"] == 512
    assert provenance["phase_unit"] == "radians"
    np.testing.assert_array_equal(
        manifest["regions"][0]["starting_coefficients"],
        np.linspace(-0.1234567, 0.2345678, 32),
    )
    np.testing.assert_array_equal(
        manifest["regions"][0]["final_coefficients"],
        np.linspace(-0.1234567, 0.2345678, 32) + 0.0123456,
    )


def test_legacy_time_layer_loads_as_depth_zero(tmp_path, pupil_reader):
    """Retain AO-only compatibility with the existing single-depth result layout."""
    (tmp_path / "time_0_ao_grid_results" / "grid_pos_0" / "ao_results.zarr").mkdir(
        parents=True
    )
    args = argparse.Namespace(
        timepoint=0, depth_index=0, coefficient_state="final", wavelength_nm=488
    )
    regions = read_regions(tmp_path, pupil_reader, args)
    assert [(region.depth, region.index) for region in regions] == [(0, 0)]


def test_standalone_grid_directory_is_time_zero_and_depth_zero(tmp_path, pupil_reader):
    """Load the SJS standalone layout from any directory name without layer plans."""
    root = tmp_path / "arbitrary_result_directory"
    for index in (2, 0):
        (root / f"grid_pos_{index}" / "ao_results.zarr").mkdir(parents=True)
    assert resolve_ao_root(root) == root
    args = argparse.Namespace(
        timepoint=0, depth_index=None, coefficient_state="final", wavelength_nm=488
    )
    regions = read_regions(root, pupil_reader, args)
    assert [(region.depth, region.index) for region in regions] == [(0, 0), (0, 2)]
    assert all(region.result.time_index == 0 for region in regions)
    for timepoint, depth in ((1, None), (0, 1)):
        args.timepoint, args.depth_index = timepoint, depth
        with pytest.raises(ValueError, match="No saved AO results"):
            read_regions(root, pupil_reader, args)


def test_publication_grid_uses_linear_phase_colors_and_reference_layout(
    tmp_path, pupil_reader, monkeypatch
):
    """Render every saved pupil unchanged with the reference's linear pi scale."""
    regions = [
        SimpleNamespace(
            index=index,
            pupil=pupil_reader.AOOptimizationResult(tmp_path, 0, index).pupil,
        )
        for index in range(9)
    ]
    originals = [region.pupil.copy() for region in regions]
    reference = tmp_path / "ao_grid_phase_plots.py"
    reference.write_text("# Publication formatting reference\n", encoding="utf-8")
    figures = []
    original_savefig = Figure.savefig

    def capture_figure(figure, *args, **kwargs):
        figures.append(figure)
        return original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture_figure)
    output = tmp_path / "publication"
    write_publication_pupil_grid(
        regions, output, timepoint=0, depth=0, reference_path=reference
    )
    assert len(list(output.glob("*.png"))) == 1
    assert len(list(output.glob("*.svg"))) == 1
    figure = figures[0]
    np.testing.assert_array_equal(figure.get_size_inches(), (12, 25))
    assert figure._suptitle is None
    pupil_axes = [axis for axis in figure.axes if axis.images]
    assert len(pupil_axes) == 9
    for index, axis in enumerate(pupil_axes):
        assert axis.get_subplotspec().get_gridspec().ncols == 2
        assert axis.get_title() == str(index)
        image = axis.images[0]
        assert image.origin == "lower"
        assert tuple(image.get_extent()) == (-1, 1, -1, 1)
        assert type(image.norm) is Normalize
        np.testing.assert_allclose(
            image.norm(np.asarray((-2, -1, 0, 1, 2)) * np.pi),
            (0, 0.25, 0.5, 0.75, 1),
        )
        np.testing.assert_array_equal(image.get_array(), originals[index])
        np.testing.assert_array_equal(regions[index].pupil, originals[index])
