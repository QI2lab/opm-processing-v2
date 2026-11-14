"""Orthogonal interpolation units and tiled processing/fusion integration."""

import numpy as np
import pytest
import tensorstore as ts

from opm_processing.imageprocessing.opmtools import (
    deskew_coverage,
    deskew_row,
    orthogonal_deskew,
)
from opm_processing.process import deskew_channel_writer


@pytest.mark.unit
@pytest.mark.parametrize(
    "deskew_field",
    [(kind, step) for kind in ("constant", "y", "z") for step in (0.23, 0.4, 1.2)],
    indirect=True,
)
def test_deskew_recovers_analytic_fluorescence(deskew_field, deskew_analytic_truth):
    """Preserve linear laboratory fields with the 30-degree photon-volume Jacobian."""
    case = deskew_field
    actual = orthogonal_deskew(case.data, **case.geometry, downsample_factor=1)
    expected, interior = deskew_analytic_truth(actual.shape)
    assert interior.any()
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual[interior], expected[interior], rtol=8e-7, atol=0)
    np.testing.assert_array_equal(actual[..., case.data.shape[2] :], 0)


@pytest.mark.unit
@pytest.mark.parametrize("factor", [1, 2, 3, 20])
@pytest.mark.parametrize("deskew_field", [("x", 0.4)], indirect=True)
def test_deskew_x_ramp_retains_partial_z_bins(deskew_field, factor, deskew_roundoff):
    """Keep every Z bin and its full divisor, including a partly supported last bin."""
    case = deskew_field
    actual = orthogonal_deskew(case.data, **case.geometry, downsample_factor=factor)
    support = deskew_coverage(
        case.data.shape, actual.shape, **case.geometry, downsample_factor=factor
    )
    ramp = np.pad(case.data[0, 0], (0, actual.shape[-1] - case.data.shape[-1]))
    np.testing.assert_allclose(
        actual, support[..., None] * ramp * (2 / case.step), **deskew_roundoff
    )
    if factor in (3, 20):
        assert np.any(actual[-1] > 0)
    np.testing.assert_array_equal(actual[:, 0], 0)


@pytest.mark.unit
@pytest.mark.parametrize("factor", [1, 2, 3, 20])
@pytest.mark.parametrize(
    "deskew_field, supported_y, width",
    [
        (("constant", 0.115, (4, 4, 1)), (2, 3, 4), 8),
        (("constant", 0.23, (4, 4, 1)), (2, 4, 6), 12),
    ],
    indirect=["deskew_field"],
)
def test_constant_box_has_prescribed_coverage(
    deskew_field, factor, supported_y, width, deskew_roundoff
):
    """Retain a known constant box's three interior rows and zero every other cell.

    At 30 degrees, four detector rows span exactly two laboratory-Z samples.
    Z=0 has its following foot outside the sensor. At Z=1, projecting onto
    the two acquired planes leaves both detector feet within [0,3) only at
    the three prescribed Y coordinates. This supplies a complete analytical
    image, independent of the production mapper used by workflow fixtures.
    """
    case = deskew_field
    actual = orthogonal_deskew(case.data, **case.geometry, downsample_factor=factor)
    shape = (2 if factor == 1 else 1, width, 4)
    assert actual.shape == shape
    coverage = np.zeros(shape[:2], np.float32)
    coverage[1 if factor == 1 else 0, supported_y] = np.float32(1 / factor)
    expected = np.zeros(shape, np.float32)
    expected[..., 0] = coverage * np.float32(0.24 * 2 / case.step)
    np.testing.assert_allclose(actual, expected, **deskew_roundoff)
    np.testing.assert_array_equal(
        deskew_coverage(
            case.data.shape, shape, **case.geometry, downsample_factor=factor
        ),
        coverage,
    )


@pytest.mark.unit
@pytest.mark.parametrize("step_um", [0.23, 0.4, 1.2])
@pytest.mark.parametrize("point", [(4, 20), (8, 30), (12, 40)])
def test_source_rows_are_perpendicular_projections(
    step_um, point, simulation_parameters
):
    """Recover both feet and their distance weights from independent dot products."""
    microscope = simulation_parameters["microscope"]
    theta = np.deg2rad(microscope["angle_deg"])
    step = step_um / microscope["pixel_size_um"]
    tangent, normal = (
        np.array([np.sin(theta), np.cos(theta)]),
        np.array([np.cos(theta), -np.sin(theta)]),
    )
    valid, before, row0, row1, a, b, c, d = deskew_row(
        *point, 48, 64, 32, step, np.tan(theta), np.sin(theta), np.cos(theta)
    )
    assert valid
    origins = np.column_stack((np.zeros(2), np.array([before, before + 1]) * step))
    projected = (point - origins) @ tangent
    feet = origins + projected[:, None] * tangent
    np.testing.assert_allclose((point - feet) @ tangent, 0, atol=2e-14)
    weights = 1 - np.abs((point - feet) @ normal) / (step * np.sin(theta))
    rows = np.floor(projected).astype(int)
    np.testing.assert_array_equal([row0, row1], rows)
    fraction = projected - rows
    expected = weights[:, None] * np.column_stack((fraction, 1 - fraction))
    np.testing.assert_allclose([[c, d], [a, b]], expected, atol=5e-8, rtol=0)
    np.testing.assert_allclose(a + b + c + d, 1, rtol=1e-7)


@pytest.mark.unit
@pytest.mark.parametrize("factor", [1, 2, 3, 20])
@pytest.mark.parametrize("reverse, crop_scan", [(False, 7), (True, 11)])
@pytest.mark.parametrize(
    "deskew_field",
    [("constant", 0.4, (48, 97, 41)), ("affine", 1.2, (48, 97, 41))],
    indirect=True,
)
def test_scan_chunk_deskew_matches_global_grid(
    deskew_field,
    deskew_output,
    deskew_analytic_truth,
    deskew_roundoff,
    factor,
    reverse,
    crop_scan,
):
    """Match scan-core processing to the full grid and independent affine truth."""
    case = deskew_field
    storage, output = deskew_output
    options = dict(
        **case.geometry, downsample_factor=factor, reverse_deskewed_z=reverse
    )
    whole = orthogonal_deskew(case.data, **options)
    tiled = orthogonal_deskew(case.data, **options, out=output)
    assert tiled is output
    np.testing.assert_array_equal(tiled, whole)
    target = ts.array(np.full(whole.shape, np.nan, np.float32))
    consume, _ = deskew_channel_writer(
        case.data.shape, target, **case.geometry, downsample_factor=factor
    )
    for start in range(0, case.data.shape[0], crop_scan):
        end = min(start + crop_scan, case.data.shape[0])
        consume(case.data[start:end], start, end)
    streamed = target.read().result()
    if reverse:
        streamed = streamed[::-1]
    np.testing.assert_array_equal(streamed, whole)
    expected, interior = deskew_analytic_truth(
        tiled.shape, factor=factor, reverse=reverse
    )
    assert interior.any()
    np.testing.assert_allclose(tiled[interior], expected[interior], **deskew_roundoff)
    assert np.isfinite(output).all()
    assert np.isnan(storage[1::2]).all()
    assert np.isnan(storage[:, (0, -1)]).all()
    assert np.isnan(storage[:, :, (0, -1)]).all()
    explicit = orthogonal_deskew(case.data, **options, zero_initialized=False)
    np.testing.assert_array_equal(explicit, whole)


@pytest.mark.unit
@pytest.mark.parametrize(
    "noisy_extended_scene,layout",
    [
        pytest.param((kind, "uniform", 1, layout), layout, id=f"{kind}-{layout}")
        for kind in ("line-pairs", "sphere")
        for layout in ("whole", "tiled")
    ],
    indirect=["noisy_extended_scene"],
    scope="session",
)
@pytest.mark.parametrize("scan_factor", (1, 3), ids=("native", "sparse3"))
def test_large_noisy_deskew_preserves_scientific_bounds_across_tiles(
    noisy_extended_scene, layout, scan_factor, monkeypatch
):
    """Deskew noisy full-camera scenes with scan-only memory-pressure streaming."""
    import importlib
    from dataclasses import replace
    from types import SimpleNamespace

    from ryomen import Slicer

    from opm_processing.imageprocessing import opmtools
    from opm_processing.imageprocessing.camera import camera_correct
    from tests.simulation.camera_likelihood import (
        camera_sampling_envelope,
        sampling_stencil_radii,
    )
    from tests.simulation.metrics import camera_volume_samples

    microscope, expectation, raw, source, _dark = noisy_extended_scene
    processing = importlib.import_module("opm_processing.process")
    assert raw.shape == (50 if layout == "whole" else 200, 256, 1900)
    microscope = replace(microscope, scan_step_um=microscope.scan_step_um * scan_factor)
    expectation, raw, source = (
        array[::scan_factor] for array in (expectation, raw, source)
    )
    camera, pixel = microscope.camera, microscope.pixel_size_um
    data = camera_correct(raw, camera.offset_adu, camera.electrons_per_adu)
    assert data.size >= 1_000_000
    z_bin = 5
    geometry = {
        "theta": microscope.angle_deg,
        "distance": microscope.scan_step_um,
        "pixel_size": pixel,
        "downsample_factor": z_bin,
    }
    shape = opmtools.deskew_shape_estimator(
        data.shape, **geometry, crop_after_deskew=False
    )[0]
    whole = orthogonal_deskew(data, **geometry) if layout == "tiled" else None
    target = ts.array(np.full(shape, np.nan, np.float32))
    slicers, visits = [], []
    interpolate = opmtools.interpolate_rows

    def observe_slicer(*args, **kwargs):
        """Observe the actual upstream scheduler without replacing its tiles."""
        result = Slicer(*args, **kwargs)
        slicers.append(len(result))
        return result

    def observe_rows(*args, **kwargs):
        """Record actual origins while executing the unchanged interpolation."""
        visits.append((args[9], args[1].shape, args[0].shape))
        return interpolate(*args, **kwargs)

    with monkeypatch.context() as execution:
        execution.setattr(processing, "Slicer", observe_slicer)
        execution.setattr(opmtools, "interpolate_rows", observe_rows)
        execution.setattr(processing, "interpolate_rows", observe_rows)
        if layout == "tiled":
            execution.setattr(
                processing.psutil,
                "virtual_memory",
                lambda: SimpleNamespace(available=int(np.prod(shape)) * 3 // 2),
            )
        consume, _ = deskew_channel_writer(data.shape, target, **geometry)
        consume(data, 0, data.shape[0])
    tiled = target.read().result()
    if layout == "tiled":
        assert len(visits) > 1 and slicers == [len(visits)]
        np.testing.assert_array_equal(tiled, whole)
        covered = np.zeros(tiled.shape[1], bool)
        foreground = orthogonal_deskew(source.astype(np.float32), **geometry) > 0
        for origin, shape, source_shape in visits:
            assert origin[0] == origin[2] == 0
            assert shape[0] == tiled.shape[0] and shape[2] == tiled.shape[2]
            assert source_shape[1:] == data.shape[1:]
            covered[origin[1] : origin[1] + shape[1]] = True
        assert covered.all()
        borders = sorted({origin[1] for origin, _shape, _source in visits if origin[1]})
        assert any(
            np.any(foreground[:, border - 1] & foreground[:, border])
            for border in borders
        ), (borders, np.flatnonzero(foreground.any(axis=(0, 2)))[[0, -1]])
    else:
        assert not slicers and visits == [((0, 0, 0), tiled.shape, data.shape)]
    coverage_zy = deskew_coverage(data.shape, tiled.shape, **geometry)
    support = (coverage_zy[..., None] > 0) & (
        np.arange(tiled.shape[-1]) < data.shape[-1]
    )
    np.testing.assert_array_equal(tiled[~support], 0)
    assert 0 < coverage_zy[-1].max() < 1 and np.any(tiled[-1] > 0)
    # Bin centers, photon-volume conversion and the source aperture are fixed
    # by the physical acquisition. No reconstructed intensity chooses a region.
    origin = ((z_bin - 1) * pixel / 2, 0, 0)
    spacing = (z_bin * pixel, pixel, pixel)
    gain = 2 * pixel / microscope.scan_step_um
    prediction = camera_volume_samples(
        tiled / gain, origin, spacing, (0, 0, 0), data.shape, microscope
    )
    coverage = camera_volume_samples(
        coverage_zy[..., None] * (np.arange(tiled.shape[-1]) < data.shape[-1]),
        origin,
        spacing,
        (0, 0, 0),
        data.shape,
        microscope,
    )
    aperture = source & (coverage >= 1 - 32 * np.finfo(np.float32).eps)
    assert aperture.any()
    bounds = camera_sampling_envelope(
        expectation,
        raw,
        camera,
        radii=sampling_stencil_radii(microscope, spacing, z_bin_factor=z_bin),
        z_bin_factor=z_bin,
    )
    assert np.all(
        np.isfinite(prediction[aperture])
        & (prediction[aperture] >= bounds.lower[aperture])
        & (prediction[aperture] <= bounds.upper[aperture])
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "option,value",
    [
        ("downsample_factor", 0),
        ("downsample_factor", 1.5),
        ("pixel_size", 0),
    ],
)
@pytest.mark.parametrize("deskew_field", [("constant", 0.4)], indirect=True)
def test_deskew_rejects_invalid_sampling(deskew_field, option, value):
    """Reject nonphysical sampling before interpolation."""
    options = {**deskew_field.geometry, option: value}
    with pytest.raises(ValueError):
        orthogonal_deskew(deskew_field.data, **options)


@pytest.mark.unit
@pytest.mark.parametrize("factor", [1, 3])
@pytest.mark.parametrize("deskew_field", [("constant", 0.4)], indirect=True)
def test_deskewed_stage_tiles_blend_without_boundary_seams(
    deskew_mosaic, factor, block_fusion
):
    """Preserve uniform fluorescence through real deskew wedges and block fusion."""
    case = deskew_mosaic
    deskewed = orthogonal_deskew(case.data, **case.geometry, downsample_factor=factor)
    fusion = block_fusion(
        [deskewed[None], deskewed[None]],
        case.positions,
        case.shape,
        support=case.support,
        x_profile=case.x_profile,
        block_shape=[1, 1, 2, 13, 5],
        output_dtype=np.dtype(np.float32),
    )
    fusion._fuse_by_blocks()
    actual = fusion.fused_ts.read().result()[0, 0]
    np.testing.assert_allclose(actual, case.expected, rtol=8e-7, atol=0)


@pytest.mark.integration
@pytest.mark.parametrize("mode,factor", [("full", 3), ("roi", 3), ("cropped", 20)])
def test_saved_deskew_retains_complete_z_extent(
    deskew_processing_case, factor, assert_roi_point_accuracy
):
    """Save every supported Z bin and the correct physical full-volume or ROI extent."""
    from opm_processing.dataio.position_collection import open_position_collection
    from opm_processing.imageprocessing.tilefusion import TileFusion
    from opm_processing.process import process_skewed

    case = deskew_processing_case
    process_skewed(
        case.dataset.path,
        case.acquisition,
        save_float32=True,
        z_downsample_level=factor,
        max_projection=False,
        create_fused_max_projection=False,
        roi_selection=case.roi,
        crop_after_deskew=case.crop,
    )
    saved = open_position_collection(case.path)
    actual = saved.arrays[0][0, 0].read().result()
    assert actual.shape == case.expected_shape
    processing_metadata = saved.attributes["opm_processing"]
    assert (
        processing_metadata["timing"]["n_acquired_planes"]
        == case.truth.pixels.shape[-3]
    )
    assert_roi_point_accuracy(
        actual[None, None],
        case.truth,
        saved.spatial_origins_zyx_um[0],
        saved.voxel_size_um,
    )
    width = case.acquisition.index_sizes["x"] - (2 if case.roi is not None else 0)
    # The finite optical object contains genuine vacuum, even inside the
    # camera aperture. Require exact padding zeros without inventing photons.
    np.testing.assert_array_equal(actual[..., :width][case.support == 0], 0)
    np.testing.assert_array_equal(actual[..., width:], 0)
    assert np.any(actual[-1] > 0)
    assert saved.voxel_size_um[0] == pytest.approx(
        round(factor * case.acquisition.pixel_size_um, 3)
    )
    assert saved.spatial_origins_zyx_um[0] == case.origin
    fusion = TileFusion(
        root_path=case.path,
        downsample_factors=(1, 1, 1),
        ssim_window=3,
        threshold=0.1,
        max_registration_shift_zyx=(2, 2, 2),
        multiscale_factors=(),
        resolution_multiples=((1, 1, 1),),
        max_workers=1,
    )
    support = fusion._tile_support_zy[0]
    np.testing.assert_allclose(support, case.support, atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(fusion._tile_positions[0], case.origin, atol=1e-12)
    if case.roi is None:
        width = case.acquisition.index_sizes["x"]
        np.testing.assert_array_equal(fusion._tile_profiles[0][2][width:], 0)
