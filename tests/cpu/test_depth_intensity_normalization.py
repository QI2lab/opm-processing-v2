"""Depth gains are shared across XY fields and fitted separately per channel."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d
from typer.testing import CliRunner

from opm_processing.fuse import app as fuse_app
from opm_processing.imageprocessing.tilefusion import TileFusion, _fit_depth_gains


@pytest.mark.integration
def test_directory_fusion_uses_deconvolved_tiles_and_preserves_partial_z_bins(tmp_path):
    """Fuse known depth overlaps through disk while replacing a plain-data fusion.

    Parameters
    ----------
    tmp_path
        Directory for the simulated camera acquisition, both processed inputs,
        processing journal, fused volume and maximum projection.

    Notes
    -----
    A narrow fluorescent line on a constant background has independently known
    intensity. The deeper acquisition has Beer–Lambert attenuation of one half.
    Simulated deconvolved tiles retain the line; plain tiles contain its Gaussian
    optical blur. Exact stage placements are persisted as measured registration
    state to isolate fusion and input selection from feature registration.
    Partially filled Z bins must retain the same calibrated intensity as full
    bins across the overlap. Both fusion passes use real reads and writes.
    """
    from dataclasses import replace

    from opm_processing.dataio.position_collection import (
        create_position_collection,
        open_image_array,
        open_position_collection,
    )
    from opm_processing.dataio.processing_state import ProcessingState
    from opm_processing.process import process
    from tests.fixtures.acquisition import (
        simulated_acquisition_metadata,
        write_simulated_acquisition,
    )

    shape = (1, 2, 1, 64, 64, 32)
    depth_step_um = 24.0
    attenuation_per_um = np.log(2) / depth_step_um
    specimen = np.full(shape[-3:], 100.0)
    specimen[..., 10] += 500
    raw = np.stack(
        [
            np.rint(specimen * np.exp(-attenuation_per_um * path)).astype(np.uint16)
            for path in (0, depth_step_um)
        ]
    )[None, :, None]
    metadata = replace(
        simulated_acquisition_metadata(tmp_path / "coverage.ome.zarr", "mirror", shape),
        array_paths=("0/0", "1/0"),
        stage_positions_zxy=((0, 0, 0), (-depth_step_um, 0, 0)),
        scan_axis_step_um=1.0,
        pixel_size_um=1.0,
        camera_offset=0.0,
        camera_conversion=1.0,
    )
    write_simulated_acquisition(metadata, raw)
    process(
        metadata.path,
        flatfield_correction=False,
        max_projection=False,
        create_fused_max_projection=False,
        z_downsample_level=2,
        save_float32=True,
    )
    plain_path = tmp_path / "coverage_deskewed.ome.zarr"
    decon_path = tmp_path / "coverage_decon_deskewed.ome.zarr"
    plain = open_position_collection(plain_path)
    decon = create_position_collection(
        decon_path,
        plain.shape,
        plain.voxel_size_um,
        dtype=np.float32,
        stage_positions=plain.stage_positions_zxy,
        channels=plain.channel_names,
        chunks=(1, 1, 4, 32, 32),
    )
    state = ProcessingState.read(tmp_path / "coverage.processing.json")
    state.initialize_run(
        decon_path, configuration={"simulated_deconvolution": True}, overwrite=True
    )
    for position, (source, destination) in enumerate(
        zip(plain.arrays, decon.arrays, strict=False)
    ):
        photons = source.read().result()
        destination.write(photons).result()
        source.write(gaussian_filter1d(photons, 1, axis=-1)).result()
        state.complete_tile(decon_path, 0, position)

    for path in (plain_path, decon_path):
        fusion = TileFusion(path, max_registration_shift_zyx=(1, 1, 1), max_workers=1)
        fusion.processing_state.save_registration(
            path,
            configuration=fusion._registration_cache_settings(),
            pairwise_metrics={"0,1": [0, 0, 0, 1.0]},
        )
        if path == plain_path:
            fusion.run()
            written = open_image_array(tmp_path / "coverage_fused.ome.zarr")
            expected = 2 * gaussian_filter1d(specimen[0, 0], 1)
            np.testing.assert_allclose(
                written[0, 0, 12:16, 56:60, :32].read().result(),
                np.broadcast_to(expected, (4, 4, 32)),
                rtol=1e-6,
            )

    result = CliRunner().invoke(
        fuse_app,
        [
            str(tmp_path),
            "--max-registration-shift-zyx",
            "1",
            "1",
            "1",
            "--max-workers",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    written = open_image_array(tmp_path / "coverage_fused.ome.zarr")
    np.testing.assert_allclose(
        written[0, 0, 12:16, 56:60, :32].read().result(),
        np.broadcast_to(2 * specimen[0, 0], (4, 4, 32)),
        rtol=1e-6,
    )
    maximum = open_image_array(tmp_path / "coverage_max_z_fused.ome.zarr")
    np.testing.assert_allclose(
        maximum[0, 0, 0, 56:60, :32].read().result(),
        np.broadcast_to(2 * specimen[0, 0], (4, 32)),
        rtol=1e-6,
    )
    state = ProcessingState.read(state.path)
    assert (
        state.registered_output_for_fused(tmp_path / "coverage_fused.ome.zarr")
        == decon_path
    )
    assert state.registration(plain_path)["pairwise_metrics"] == {"0,1": [0, 0, 0, 1.0]}


@pytest.mark.integration
@pytest.mark.parametrize("normalize", [False, True])
def test_depth_normalization_fuses_simulated_object_from_disk(
    processing_options, tmp_path, normalize
):
    """Fuse a known oblique intensity phantom while preserving its XY contrast."""
    import json
    from dataclasses import replace

    from opm_processing.dataio.acquisition import ChannelMetadata
    from opm_processing.dataio.position_collection import open_image_array
    from opm_processing.process import process
    from tests.fixtures.acquisition import (
        simulated_acquisition_metadata,
        write_simulated_acquisition,
    )

    shape = (1, 4, 2, 64, 64, 32)
    scan, row, col = np.indices(shape[-3:], dtype=float)
    raw = np.empty(shape, dtype=np.uint16)
    depth_step_um = 8.0
    # Known effective excitation/emission attenuation, in inverse micrometers.
    attenuation_per_um = np.asarray((0.08, 0.16))
    rng = np.random.default_rng(637)
    bead_centers_um = rng.uniform((2, 10, 4), (38, 100, 28), (40, 3))
    for tile, (depth, xy_gain) in enumerate(((0, 1), (0, 3), (1, 1), (1, 3))):
        optical_path_um = depth * depth_step_um
        z = row * 0.5 + optical_path_um
        y = scan + row * np.cos(np.pi / 6)
        # Resolve a fixed bead object in laboratory coordinates with an
        # independent continuous Gaussian optical response (sigma = 2 um).
        specimen = np.full(shape[-3:], 10.0)
        for bead_z, bead_y, bead_x in bead_centers_um:
            squared_distance = (
                (z - bead_z) ** 2 + (y - bead_y) ** 2 + (col - bead_x) ** 2
            )
            specimen += 1000 * np.exp(-squared_distance / (2 * 2.0**2))
        for channel in range(2):
            raw[0, tile, channel] = np.rint(
                specimen
                * xy_gain
                * np.exp(-attenuation_per_um[channel] * optical_path_um)
            ).astype(np.uint16)
    metadata = replace(
        simulated_acquisition_metadata(tmp_path / "depth.ome.zarr", "mirror", shape),
        channels=(
            ChannelMetadata(0, "488nm", 488, 10, None),
            ChannelMetadata(1, "637nm", 637, 10, None),
        ),
        array_paths=tuple(f"{tile}/0" for tile in range(4)),
        stage_positions_zxy=((0, 0, 0), (0, 0, 100), (-8, 0, 0), (-8, 0, 100)),
        scan_axis_step_um=1.0,
        pixel_size_um=1.0,
        camera_offset=0.0,
        camera_conversion=1.0,
    )
    write_simulated_acquisition(metadata, raw)
    process(
        root_path=metadata.path,
        save_float32=True,
        **processing_options,
    )
    fusion = TileFusion(
        metadata.path,
        normalize_depth_intensity=normalize,
        max_registration_shift_zyx=(1, 1, 1),
        downsample_factors=(1, 1, 1),
        ssim_window=3,
        threshold=0.3,
        multiscale_factors=(2,),
        chunk_shape_yx=(32, 32),
        max_workers=1,
    )
    fusion.run()
    written = open_image_array(tmp_path / "depth_fused.ome.zarr")
    actual = written[0].read().result()
    assert actual[..., :32].max() > 500  # Recover the known 1000-ADU bead signal.
    pyramid = open_image_array(tmp_path / "depth_fused.ome.zarr", level="1")
    np.testing.assert_array_equal(pyramid[0].read().result(), actual[:, ::2, ::2, ::2])
    # The two separated XY fields differ only by the imposed threefold gain.
    # Quantization contributes at most half an ADU per field before the
    # threefold XY scale and inverse attenuation. Bound it independently
    # using the known Beer gain, including the 2% gain accuracy checked below.
    expected_depth_gains = np.exp(attenuation_per_um * depth_step_um)
    quantization_tolerance = 0.5 * (3 + 1) * expected_depth_gains.max() * 1.02
    np.testing.assert_allclose(
        actual[..., 100:132],
        actual[..., :32] * 3,
        rtol=0.002,
        atol=quantization_tolerance,
    )
    report = json.loads((tmp_path / "depth_depth_intensity_gains.json").read_text())
    assert report["enabled"] == normalize
    if normalize:
        np.testing.assert_allclose(
            np.asarray([channel["depth_gains"] for channel in report["channels"]]).T,
            np.exp(np.asarray((0, depth_step_um))[:, None] * attenuation_per_um),
            rtol=0.02,
        )
        # Both channels sample the same object and must agree after matching.
        np.testing.assert_allclose(
            actual[0],
            actual[1],
            rtol=0.03,
            atol=0.5 * 3 * expected_depth_gains.sum() * 1.02,
        )
    else:
        assert np.max(np.abs(actual[0] - actual[1])) > 500


@pytest.mark.unit
def test_depth_gain_fit_is_robust_and_anchors_unobserved_depths():
    """Recover known gains despite an outlier and retain isolated depth anchors."""
    gains = _fit_depth_gains(
        4,
        [
            (0, 1, np.log(2)),
            (0, 1, np.log(2)),
            (0, 1, np.log(99)),
            (1, 2, np.log(0.5)),
        ],
    )
    np.testing.assert_allclose(gains, (1, 2, 1, 1))


@pytest.mark.unit
def test_normalization_matches_depths_and_preserves_xy_brightness(
    tensorstore_dataset,
    fusion_operator,
):
    """Invert known Beer attenuation while preserving threefold XY contrast."""
    rng = np.random.default_rng(31)
    specimen = rng.uniform(100, 1000, (24, 64, 64)).astype(np.float32)
    arrays = []
    attenuation_per_um = np.log((2.0, 4.0)) / 8.0
    for depth, xy_gain in ((0, 1), (0, 3), (1, 1), (1, 3)):
        signal = specimen[depth * 8 : depth * 8 + 16] * xy_gain
        channels = np.stack(
            (
                signal * np.exp(-attenuation_per_um[0] * depth * 8),
                signal * np.exp(-attenuation_per_um[1] * depth * 8),
                np.zeros_like(signal),
            )
        )
        arrays.append(channels)
    fusion = fusion_operator(
        TileFusion,
        acquisition=SimpleNamespace(
            stage_positions_zxy=[(0, 0, 0), (0, 0, 100), (8, 0, 0), (8, 0, 100)]
        ),
        _tile_positions=[(0, 0, 0), (0, 0, 100), (8, 0, 0), (8, 0, 100)],
        _tile_source_position_indices=list(range(4)),
        _tile_time_indices=[0] * 4,
        _tile_shapes=[(16, 64, 64)] * 4,
        _pixel_size=(1, 1, 1),
        _is_2d=False,
        _variable_roi_tiles=False,
        position_dim=4,
        time_dim=1,
        channels=3,
        position_arrays=tuple(tensorstore_dataset(a[None]) for a in arrays),
        _tile_support_zy=[np.ones((16, 64), dtype=bool)] * 4,
        _tile_channel_nonzero={(i, c): c < 2 for i in range(4) for c in range(3)},
        optimized_pairwise_metrics={(0, 2): (0, 0, 0, 1), (1, 3): (0, 0, 0, 1)},
        _depth_intensity_gains=np.ones((4, 3), dtype=np.float32),
    )
    fusion.estimate_depth_intensity_gains()
    np.testing.assert_allclose(
        fusion._depth_intensity_gains, [(1, 1, 1)] * 2 + [(2, 4, 1)] * 2, rtol=1e-6
    )
    corrected = [fusion._scale_depth_source(i, a) for i, a in enumerate(arrays)]
    np.testing.assert_allclose(corrected[0][:, 8:], corrected[2][:, :8], rtol=1e-6)
    np.testing.assert_allclose(corrected[1][:, 8:], corrected[3][:, :8], rtol=1e-6)
    np.testing.assert_allclose(corrected[1], corrected[0] * 3, rtol=1e-6)


@pytest.mark.unit
def test_scaled_uint16_output_clips_and_float32_preserves_range(
    fusion_operator,
):
    """Check saturation, floating-point range, and immutable source intensities."""
    fusion = fusion_operator(
        TileFusion, _depth_intensity_gains=np.asarray([(2, 0.5)], dtype=np.float32)
    )
    source = np.asarray([40000, 1000], dtype=np.uint16).reshape(2, 1, 1, 1)
    scaled = fusion._scale_depth_source(0, source)
    fusion.output_dtype = np.dtype(np.uint16)
    np.testing.assert_array_equal(
        fusion._cast_fusion_values(scaled).ravel(), (65535, 500)
    )
    fusion.output_dtype = np.dtype(np.float32)
    assert fusion._cast_fusion_values(scaled) is scaled
    np.testing.assert_array_equal(
        fusion._cast_fusion_values(scaled).ravel(), (80000, 500)
    )
    fusion._depth_intensity_gains[:] = 1
    fusion.output_dtype = np.dtype(np.uint16)
    unscaled = fusion._cast_fusion_values(fusion._scale_depth_source(0, source))
    assert unscaled is source
    np.testing.assert_array_equal(unscaled.ravel(), (40000, 1000))
    np.testing.assert_array_equal(source.ravel(), (40000, 1000))
    coverage = np.asarray([[0.5]], np.float32)
    corrected = fusion._scale_depth_source(0, source, coverage_zy=coverage)
    np.testing.assert_array_equal(corrected.ravel(), (80000, 2000))
    np.testing.assert_array_equal(source.ravel(), (40000, 1000))
    fusion._depth_intensity_gains[:] = (2, 0.5)
    corrected = fusion._scale_depth_source(0, source, coverage_zy=coverage)
    np.testing.assert_array_equal(corrected.ravel(), (160000, 1000))
    np.testing.assert_array_equal(source.ravel(), (40000, 1000))
