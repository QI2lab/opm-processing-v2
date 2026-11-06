"""Saved-AO serialization, modal selection and publication-coordinate units."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from matplotlib.colors import Normalize
from scripts import ao_publication_style, ao_saved_results
from scripts import export_ao_overview as overview
from scripts.ao_publication_style import write_publication_pupil_grid
from scripts.ao_wavekit_phase import BASIS_VERSION
from scripts.export_ao_overview import Region, read_regions, resolve_ao_root

pytestmark = pytest.mark.unit


def test_direct_ao_results_keep_numeric_order_and_saved_values(
    tmp_path,
    ao_result_factory,
    ao_analytical_phase,
):
    """Read every native coefficient and reconstruct actual pupils in numeric order."""
    layer = tmp_path / "time_0_z_0_ao_grid_results"
    starting = np.linspace(-0.1234567, 0.2345678, 32)
    final = starting + 0.0123456
    for index in (10, 7, 4, 1, 6, 3, 0, 5, 2):
        ao_result_factory(
            path=layer / f"grid_pos_{index}",
            starting=starting,
            final=final,
            index=index,
        )
    args = argparse.Namespace(
        timepoint=0, depth_index=None, coefficient_state="final", wavelength_nm=488
    )
    regions = read_regions(tmp_path, ao_saved_results, args)
    assert [r.index for r in regions] == [0, 1, 2, 3, 4, 5, 6, 7, 10]
    expected = ao_analytical_phase(final)
    for region in regions:
        np.testing.assert_array_equal(region.result.starting_coefficients, starting)
        np.testing.assert_array_equal(region.result.final_coefficients, final)
        np.testing.assert_array_equal(region.pupil.mask, expected.mask)
        np.testing.assert_allclose(
            region.pupil.compressed(), expected.compressed(), rtol=0, atol=4e-12
        )


@pytest.mark.parametrize("state", ("starting", "final", "change"))
def test_time_and_depth_selection_use_the_requested_saved_state(
    tmp_path, ao_result_factory, state
):
    """Read selected saved states and evaluate their known horizontal-tilt phase."""
    starting = np.zeros(32)
    starting[0] = 0.03
    final = starting.copy()
    final[0] = 0.09
    for time, depth in ((0, 0), (1, 0), (1, 2), (1, 5)):
        ao_result_factory(
            path=tmp_path / f"time_{time}_z_{depth}_ao_grid_results" / "grid_pos_0",
            starting=starting,
            final=final,
            time=time,
        )
    args = argparse.Namespace(
        timepoint=1, depth_index=2, coefficient_state=state, wavelength_nm=561
    )
    regions = read_regions(tmp_path, ao_saved_results, args)
    assert [(r.depth, r.index) for r in regions] == [(2, 0)]
    assert regions[0].result.time_index == 1
    coefficient = {"starting": 0.03, "final": 0.09, "change": 0.09 - 0.03}[state]
    y, x = np.meshgrid(np.linspace(-1, 1, 512), np.linspace(-1, 1, 512), indexing="ij")
    expected = 2 * np.pi / 0.561 * coefficient * 2 * x
    valid = x * x + y * y <= 1
    np.testing.assert_array_equal(regions[0].pupil.mask, ~valid)
    np.testing.assert_allclose(
        regions[0].pupil.data[valid], expected[valid], atol=1e-14
    )


def test_main_reader_uses_the_sdk_horizontal_first_tilt(tmp_path, ao_result_factory):
    """The prescribed first SDK mode is a horizontal pupil slope."""
    final = np.zeros(32)
    final[0] = 0.1
    ao_result_factory(
        path=tmp_path / "time_0_z_0_ao_grid_results" / "grid_pos_0", final=final
    )
    args = argparse.Namespace(
        timepoint=0, depth_index=None, coefficient_state="final", wavelength_nm=488
    )
    pupil = read_regions(tmp_path, ao_saved_results, args)[0].pupil
    y, x = np.meshgrid(np.linspace(-1, 1, 512), np.linspace(-1, 1, 512), indexing="ij")
    expected = 2 * np.pi / 0.488 * 0.1 * 2 * x
    valid = x * x + y * y <= 1
    np.testing.assert_array_equal(pupil.mask, ~valid)
    np.testing.assert_allclose(pupil.data[valid], expected[valid], atol=1e-14)


def test_export_manifest_identifies_the_local_reconstruction_basis(
    tmp_path, ao_result_factory, monkeypatch
):
    """Actual reader and rendering output retain coefficient units and source integrity."""
    starting = np.linspace(-0.1234567, 0.2345678, 32)
    final = starting + 0.0123456
    ao_result_factory(
        path=tmp_path / "sample_ao_grid_results" / "grid_pos_0",
        starting=starting,
        final=final,
    )
    output = tmp_path / "output"
    monkeypatch.setattr(
        overview.sys,
        "argv",
        ["export_ao_overview.py", str(tmp_path), "--output", str(output)],
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
        manifest["regions"][0]["starting_coefficients"], starting
    )
    np.testing.assert_array_equal(manifest["regions"][0]["final_coefficients"], final)
    assert list(output.glob("*.png")) and list(output.glob("*.svg"))


def test_legacy_time_layer_loads_as_depth_zero(tmp_path, ao_result_factory):
    """A producer result can live in the historical single-depth directory schema."""
    ao_result_factory(path=tmp_path / "time_0_ao_grid_results" / "grid_pos_0")
    args = argparse.Namespace(
        timepoint=0, depth_index=0, coefficient_state="final", wavelength_nm=488
    )
    assert [
        (r.depth, r.index) for r in read_regions(tmp_path, ao_saved_results, args)
    ] == [(0, 0)]


def test_standalone_grid_directory_is_time_zero_and_depth_zero(
    tmp_path, ao_result_factory
):
    """Standalone saved results use their requested physical acquisition indices."""
    root = tmp_path / "arbitrary_result_directory"
    for index in (2, 0):
        ao_result_factory(path=root / f"grid_pos_{index}", index=index)
    assert resolve_ao_root(root) == root
    args = argparse.Namespace(
        timepoint=0, depth_index=None, coefficient_state="final", wavelength_nm=488
    )
    regions = read_regions(root, ao_saved_results, args)
    assert [(r.depth, r.index) for r in regions] == [(0, 0), (0, 2)]
    assert all(r.result.time_index == 0 for r in regions)
    for timepoint, depth in ((1, None), (0, 1)):
        args.timepoint, args.depth_index = timepoint, depth
        with pytest.raises(ValueError, match="No saved AO results"):
            read_regions(root, ao_saved_results, args)


def test_publication_grid_uses_linear_phase_colors_and_reference_layout(
    tmp_path, ao_result_factory, ao_figure_observer
):
    """Inspect actual Matplotlib coordinate and color mappings after writing figures."""
    regions = []
    for index in range(9):
        final = np.zeros(32)
        final[0] = 0.02 * (index + 1)
        case = ao_result_factory(final=final, index=index)
        regions.append(
            Region(
                0,
                index,
                case.result,
                ao_saved_results.phase_from_modal_coeff(final, samples=33),
            )
        )
    originals = [r.pupil.copy() for r in regions]
    output = tmp_path / "publication"
    write_publication_pupil_grid(
        regions,
        output,
        timepoint=0,
        depth=0,
        reference_path=Path(ao_publication_style.__file__),
    )
    assert len(list(output.glob("*.png"))) == len(list(output.glob("*.svg"))) == 1
    figure = ao_figure_observer[0]
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
            image.norm(np.asarray((-2, -1, 0, 1, 2)) * np.pi), (0, 0.25, 0.5, 0.75, 1)
        )
        np.testing.assert_array_equal(image.get_array(), originals[index])
        np.testing.assert_array_equal(regions[index].pupil, originals[index])
