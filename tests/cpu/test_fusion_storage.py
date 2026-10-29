"""Fusion writer units and registered maximum-projection CLI integration."""

import numpy as np
import pytest
from skimage.measure import block_reduce as block_reduce_cpu
from typer.testing import CliRunner
from yaozarrs import open_group, v05

from opm_processing.dataio.position_collection import (
    open_image_array,
    open_position_collection,
)
from opm_processing.dataio.processing_state import ProcessingState
from opm_processing.fuse import app as fuse_app
from opm_processing.imageprocessing.maxtilefusion import MaxTileFusion
from opm_processing.imageprocessing.tilefusion import TileFusion


@pytest.mark.unit
def test_max_projection_depth_tiles_share_xy_footprint(
    tmp_path, position_store_factory
) -> None:
    """Depth tiles with fixed stage XY must fuse into a single tile footprint."""
    source = np.arange(64, dtype=np.uint16).reshape(1, 1, 1, 8, 8)
    collection = position_store_factory(
        np.stack((source, source), axis=1),
        name="depth_phantom",
        voxel_size_um=(0.23, 0.115, 0.115),
    )
    fusion = MaxTileFusion(
        ts_dataset=open_position_collection(collection.path).arrays,
        tile_positions=((3939.54, -7456.06, 4207.16), (3800.09, -7456.06, 4207.16)),
        output_path=tmp_path / "depth_max_z_fused.ome.zarr",
        pixel_size=(0.23, 0.115, 0.115),
        opm_angle_deg=30.0,
        blend_pixels=(0, 0),
        chunk_size=4,
    )
    fusion.run()

    np.testing.assert_array_equal(fusion.tile_positions[0], fusion.tile_positions[1])
    written = open_image_array(tmp_path / "depth_max_z_fused.ome.zarr")
    np.testing.assert_array_equal(written.read().result(), source)


@pytest.mark.unit
def test_max_projection_fusion_writes_centered_multiscales_and_offsets(
    tmp_path,
    position_store_factory,
) -> None:
    """Persist every fused max level with rounded physical coordinates."""
    source = np.arange(64, dtype=np.uint16).reshape(1, 1, 1, 8, 8)
    output_path = tmp_path / "max_z_fused.ome.zarr"
    collection = position_store_factory(
        source[:, None],
        name="ramp_phantom",
        voxel_size_um=(0.34567, 0.1164, 0.1184),
    )
    fusion = MaxTileFusion(
        ts_dataset=open_position_collection(collection.path).arrays,
        tile_positions=((1.23456, 2.34567, 3.45678),),
        output_path=output_path,
        pixel_size=(0.34567, 0.1164, 0.1184),
        spatial_offset_z_um=4.56789,
        blend_pixels=(0, 0),
        chunk_size=4,
        reverse_stage_y=True,
        reverse_stage_z=False,
    )
    fusion.run()

    reopened = open_group(output_path)
    metadata = reopened.ome_metadata()
    assert isinstance(metadata, v05.Image)
    datasets = metadata.multiscales[0].datasets
    assert [dataset.path for dataset in datasets] == ["0", "1", "2", "3"]
    base_scale = np.asarray(datasets[0].scale_transform.scale[-3:])
    assert datasets[0].translation_transform is not None
    base_translation = np.asarray(datasets[0].translation_transform.translation[-3:])
    expected = source
    for factor, dataset in zip((1, 2, 4, 8), datasets, strict=False):
        assert dataset.scale_transform.scale == [
            1.0,
            1.0,
            0.346,
            round(0.116 * factor, 3),
            round(0.118 * factor, 3),
        ]
        assert dataset.translation_transform is not None
        assert dataset.translation_transform.translation == [
            0.0,
            0.0,
            4.568,
            -2.346,
            3.457,
        ]
        output_index = np.asarray([0, min(2, 7 // factor), min(3, 7 // factor)])
        output_coordinate = np.asarray(
            dataset.translation_transform.translation[-3:]
        ) + output_index * np.asarray(dataset.scale_transform.scale[-3:])
        source_index = output_index * np.asarray([1, factor, factor])
        np.testing.assert_allclose(
            output_coordinate,
            base_translation + source_index * base_scale,
        )
        if factor > 1:
            expected = source[..., ::factor, ::factor]
        np.testing.assert_array_equal(
            reopened[dataset.path].to_tensorstore().read().result(),
            expected,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("downsample_method", "is_2d", "padded_shape", "dtype"),
    (
        ("stride", False, (4, 8, 8), np.uint16),
        ("block_mean", False, (4, 8, 8), np.uint16),
        ("stride", True, (1, 8, 8), np.uint16),
        ("stride", False, (2, 4, 4), np.float32),
    ),
    ids=(
        "volume-stride",
        "volume-block-mean",
        "projection-stride",
        "fractional-photons",
    ),
)
def test_multiscale_storage_round_trip_matches_reference(
    fusion_operator,
    tmp_path,
    downsample_method,
    is_2d,
    padded_shape,
    dtype,
):
    """Reopen exact pyramid data and verify its common physical-space center."""
    fusion = fusion_operator(
        TileFusion,
        padded_shape=padded_shape,
        offset_um=(4.51234, -3.01234, 12.25123),
        _pixel_size=(0.81234, 0.51234, 0.25123),
        time_dim=1,
        channels=1,
        multiscale_factors=(2, 4),
        _is_2d=is_2d,
        chunk_y=4,
        chunk_x=4,
        multiscale_downsample=downsample_method,
        _max_workers=2,
        fusion_ram_fraction=0.4,
        max_in_flight_writes=2,
        _input_chunk_zyx=(2, 4, 4),
        output_dtype=np.dtype(dtype),
    )

    path = tmp_path / "multiscale.ome.zarr"
    scale0, fusion.write_block_shape = fusion._prepare_fused_image(path)
    source_shape = (1, 1, *padded_shape)
    source = np.arange(np.prod(source_shape), dtype=dtype).reshape(source_shape)
    if dtype == np.float32:
        source = source / 4 + np.float32(0.25)
    scale0.write(source).result()
    fusion._write_multiscales()

    reopened = open_group(path)
    metadata = reopened.ome_metadata()
    assert metadata is not None
    datasets = metadata.multiscales[0].datasets
    assert [dataset.path for dataset in datasets] == ["0", "1", "2"]

    expected = source
    rounded_offset = np.asarray([round(float(value), 3) for value in fusion.offset_um])
    rounded_pixel_size = np.asarray(
        [round(float(value), 3) for value in fusion._pixel_size]
    )
    base_center = (
        rounded_offset
        + (np.asarray(padded_shape, dtype=np.float64) - 1.0) * rounded_pixel_size / 2.0
    )
    for absolute_factor, dataset in zip((1, 2, 4), datasets, strict=False):
        z_factor = 1 if is_2d else absolute_factor
        expected_scale = np.asarray(
            [
                round(float(value), 3)
                for value in (
                    1.0,
                    1.0,
                    rounded_pixel_size[0] * z_factor,
                    rounded_pixel_size[1] * absolute_factor,
                    rounded_pixel_size[2] * absolute_factor,
                )
            ]
        )
        center_shift = 0.5 if downsample_method == "block_mean" else 0.0
        expected_translation = np.asarray(
            [
                round(float(value), 3)
                for value in (
                    0.0,
                    0.0,
                    rounded_offset[0]
                    + center_shift * (z_factor - 1) * rounded_pixel_size[0],
                    rounded_offset[1]
                    + center_shift * (absolute_factor - 1) * rounded_pixel_size[1],
                    rounded_offset[2]
                    + center_shift * (absolute_factor - 1) * rounded_pixel_size[2],
                )
            ]
        )
        np.testing.assert_allclose(
            dataset.scale_transform.scale,
            expected_scale,
        )
        assert dataset.translation_transform is not None
        np.testing.assert_allclose(
            dataset.translation_transform.translation,
            expected_translation,
        )
        reopened_array = reopened[dataset.path].to_tensorstore()
        assert np.dtype(reopened_array.dtype.numpy_dtype) == dtype

        if downsample_method == "stride":
            # A pyramid index must have exactly the same physical coordinate
            # as the source voxel selected at index * absolute_factor.
            output_index = np.asarray(
                [
                    min(1, int(reopened_array.shape[-3]) - 1),
                    min(2, int(reopened_array.shape[-2]) - 1),
                    min(3, int(reopened_array.shape[-1]) - 1),
                ],
                dtype=np.float64,
            )
            source_index = output_index * np.asarray(
                [z_factor, absolute_factor, absolute_factor],
                dtype=np.float64,
            )
            output_coordinate = np.asarray(
                dataset.translation_transform.translation[-3:]
            ) + output_index * np.asarray(dataset.scale_transform.scale[-3:])
            source_coordinate = rounded_offset + source_index * rounded_pixel_size
            np.testing.assert_allclose(
                output_coordinate,
                source_coordinate,
            )

        spatial_shape = np.asarray(reopened_array.shape[-3:], dtype=np.float64)
        physical_center = (
            expected_translation[-3:]
            + (spatial_shape - 1.0) * expected_scale[-3:] / 2.0
        )
        if downsample_method == "block_mean":
            np.testing.assert_allclose(physical_center, base_center, atol=0.002)

        if dataset.path == "0":
            np.testing.assert_array_equal(
                reopened_array.read().result(),
                expected,
            )
            continue

        z_relative_factor = 1 if is_2d else 2
        if downsample_method == "stride":
            expected = expected[..., ::z_relative_factor, ::2, ::2]
        else:
            expected = block_reduce_cpu(
                expected,
                block_size=(1, 1, z_relative_factor, 2, 2),
                func=np.mean,
            ).astype(dtype)
        np.testing.assert_array_equal(
            reopened_array.read().result(),
            expected,
        )

    assert not any(key.startswith("opm_") for key in reopened.attrs)


@pytest.mark.unit
def test_regenerate_max_z_flag_preserves_multiscale_pyramid_round_trip(
    registered_projection_case,
) -> None:
    """All pyramid levels include photons from every full-resolution Z plane."""
    case = registered_projection_case
    result = CliRunner().invoke(
        fuse_app,
        [str(case.raw_path), "--regenerate-max-z", "--max-workers", "2"],
    )
    assert result.exit_code == 0, result.output

    reopened = open_group(case.output_path)
    metadata = reopened.ome_metadata()
    assert isinstance(metadata, v05.Image)
    datasets = metadata.multiscales[0].datasets
    assert [dataset.path for dataset in datasets] == ["0", "1"]
    projection = case.truth.max(axis=2, keepdims=True)
    for level, dataset in enumerate(datasets):
        actual = reopened[dataset.path].to_tensorstore().read().result()
        factor = 2**level
        expected = projection[..., ::factor, ::factor]
        if case.downsample == "block_mean":
            expected = np.rint(
                block_reduce_cpu(
                    projection,
                    block_size=(1, 1, 1, factor, factor),
                    func=np.mean,
                )
            ).astype(case.truth.dtype)
        np.testing.assert_array_equal(actual, expected)
        expected_scale = [
            case.scales[level][0],
            case.scales[level][1],
            *(round(value, 3) for value in case.scales[level][2:]),
        ]
        np.testing.assert_allclose(dataset.scale_transform.scale, expected_scale)
        assert dataset.translation_transform is not None
        expected_translation = [
            case.translations[level][0],
            case.translations[level][1],
            *(round(value, 3) for value in case.translations[level][2:]),
        ]
        expected_translation[2] = (
            case.translations[0][2]
            + 0.5 * (case.truth.shape[2] - 1) * case.scales[0][2]
        )
        expected_translation[2:] = [
            round(value, 3) for value in expected_translation[2:]
        ]
        np.testing.assert_allclose(
            dataset.translation_transform.translation,
            expected_translation,
        )
    assert not any(key.startswith("opm_") for key in reopened.attrs)
    state = ProcessingState.read(case.state_path)
    assert (
        state.registered_output_for_max_projection(case.output_path)
        == case.processed_path
    )


@pytest.mark.unit
def test_expanded_cache_uses_initial_settings_and_restores_final_limits(
    tmp_path, processing_state, retry_fusion
):
    """Restore expanded limits from a cache keyed by the original settings."""
    fusion = retry_fusion
    fusion.processing_state = processing_state
    fusion.data = tmp_path / "sample_deskewed.ome.zarr"
    processing_state.initialize_run(fusion.data, configuration={}, overwrite=True)
    initial = fusion._registration_cache_settings()
    fusion._rejected_registration_links = {(0, 1): ((15, -284, 0), (13, 198, 33), 0.9)}
    assert fusion._expand_registration_limits()
    fusion._registration_retry_count = 1
    fusion.pairwise_metrics = {(0, 1): (15, -284, 0, 0.9)}
    assert fusion._registration_cache_settings() == initial
    fusion.save_pairwise_metrics()
    saved = processing_state.path.read_bytes()
    final_pairs = fusion._pair_registration_limits.copy()
    fusion.max_registration_shift_zyx = fusion._initial_registration_shift_zyx
    fusion._pair_registration_limits = fusion._initial_pair_registration_limits.copy()
    fusion.pairwise_metrics = {}
    fusion.register_with_retries()
    assert fusion.max_registration_shift_zyx == (17, 299, 33)
    assert fusion._pair_registration_limits == final_pairs
    assert fusion._registration_retry_count == 1
    assert processing_state.path.read_bytes() == saved
    fusion.registration_retries = 0
    with pytest.raises(ValueError, match="settings do not match"):
        fusion.load_pairwise_metrics()
