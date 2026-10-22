"""Orthogonal interpolation units and tiled processing/fusion integration."""

import numpy as np
import pytest
import tensorstore as ts

from opm_processing.imageprocessing.opmtools import deskew_row, orthogonal_deskew

pytestmark = pytest.mark.usefixtures("small_cpu_kernels")


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
    response = orthogonal_deskew(
        np.full(case.data.shape, 1, dtype=np.float32),
        **case.geometry,
        downsample_factor=factor,
    )
    ramp = np.pad(case.data[0, 0], (0, response.shape[-1] - case.data.shape[-1]))
    np.testing.assert_allclose(actual, response * ramp, **deskew_roundoff)
    if factor in (3, 20):
        assert np.any(actual[-1] > 0)
    np.testing.assert_array_equal(actual[:, 0], 0)


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
@pytest.mark.parametrize(
    "reverse, tile_shape", [(False, (2, 17, 5)), (True, (3, 23, 7))]
)
@pytest.mark.parametrize(
    "deskew_field",
    [("constant", 0.4, (48, 97, 41)), ("affine", 1.2, (48, 97, 41))],
    indirect=True,
)
def test_tiled_deskew_matches_global_grid(
    deskew_field,
    deskew_output,
    deskew_analytic_truth,
    deskew_roundoff,
    factor,
    reverse,
    tile_shape,
):
    """Match whole-volume output exactly across Z, Y and X tile boundaries."""
    case = deskew_field
    storage, output = deskew_output
    options = dict(
        **case.geometry, downsample_factor=factor, reverse_deskewed_z=reverse
    )
    whole = orthogonal_deskew(case.data, **options)
    tiled = orthogonal_deskew(case.data, **options, tile_shape=tile_shape, out=output)
    assert tiled is output
    np.testing.assert_array_equal(tiled, whole)
    expected, interior = deskew_analytic_truth(
        tiled.shape, factor=factor, reverse=reverse
    )
    assert interior.any()
    np.testing.assert_allclose(tiled[interior], expected[interior], **deskew_roundoff)
    # Assert that the analytical comparison includes foreground on both sides
    # of actual interior tile borders, rather than only a tile's central region.
    for axis in (1, 2):
        borders = np.arange(tile_shape[axis], tiled.shape[axis], tile_shape[axis])
        assert len(borders) >= 2
        assert np.any(
            np.take(interior, borders - 1, axis=axis)
            & np.take(interior, borders, axis=axis)
        )
    assert np.isfinite(output).all()
    assert np.isnan(storage[1::2]).all()
    assert np.isnan(storage[:, (0, -1)]).all()
    assert np.isnan(storage[:, :, (0, -1)]).all()
    explicit = orthogonal_deskew(
        case.data, **options, tile_shape=tile_shape, zero_initialized=False
    )
    np.testing.assert_array_equal(explicit, whole)


@pytest.mark.unit
@pytest.mark.parametrize(
    "option,value",
    [
        ("downsample_factor", 0),
        ("downsample_factor", 1.5),
        ("tile_shape", (1, 0, 2)),
        ("tile_shape", (1, 2)),
        ("pixel_size", 0),
    ],
)
@pytest.mark.parametrize("deskew_field", [("constant", 0.4)], indirect=True)
def test_deskew_rejects_invalid_sampling_or_tiles(deskew_field, option, value):
    """Reject nonphysical sampling or invalid tile extents before interpolation."""
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
    deskewed = orthogonal_deskew(
        case.data, **case.geometry, downsample_factor=factor, tile_shape=(2, 17, 5)
    )
    fusion = block_fusion(
        [deskewed[None], deskewed[None]],
        case.positions,
        case.shape,
        support=case.support,
        x_profile=case.x_profile,
        block_shape=[1, 1, 2, 13, 5],
        fused_ts=ts.array(np.zeros((1, 1, *case.shape), np.float32)),
        output_dtype=np.dtype(np.float32),
    )
    fusion._fuse_by_blocks()
    actual = fusion.fused_ts.read().result()[0, 0]
    np.testing.assert_allclose(actual, case.expected, rtol=8e-7, atol=0)


@pytest.mark.unit
@pytest.mark.parametrize("factor", [3, 20])
@pytest.mark.parametrize("mode", ["full", "roi", "cropped"])
@pytest.mark.parametrize("deskew_field", [("x", 0.4)], indirect=True)
def test_saved_deskew_retains_complete_z_extent(
    deskew_processing_case, factor, deskew_roundoff
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
    np.testing.assert_allclose(actual, case.expected, **deskew_roundoff)
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
    first_value = 14 if case.roi is not None else 10
    gain = 2 * case.acquisition.pixel_size_um / case.acquisition.scan_axis_step_um
    np.testing.assert_allclose(
        support, case.expected[..., 0] / (first_value * gain), atol=1e-7, rtol=1e-6
    )
    np.testing.assert_allclose(fusion._tile_positions[0], case.origin, atol=1e-12)
    if case.roi is None:
        width = case.acquisition.index_sizes["x"]
        np.testing.assert_array_equal(fusion._tile_profiles[0][2][width:], 0)
