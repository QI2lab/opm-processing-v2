"""Minimal numerical fusion operators with explicitly supplied geometry and data."""

from dataclasses import replace
from functools import lru_cache
from itertools import count
from types import SimpleNamespace

import numpy as np
import pytest
import tensorstore as ts

from opm_processing.imageprocessing.tilefusion import (
    TileFusion,
)


@pytest.fixture
def tile_fusion(tmp_path, opm_v2_acquisition_factory):
    """Construct real fusion inputs with native geometry and durable ROI records."""
    from opm_processing.dataio.position_collection import (
        create_variable_position_collection,
    )
    from opm_processing.dataio.processing_state import ProcessingState

    identifiers = count()

    def create(
        tiles,
        positions,
        *,
        spacing=(1, 1, 1),
        time_indices=None,
        time_interval_s=None,
        **options,
    ):
        """Persist Cartesian unit objects as physically sized cropped scan tiles."""
        times = [0] * len(tiles) if time_indices is None else time_indices
        projection = all(tile.shape[-3] == 1 for tile in tiles)
        pixel = spacing[1]
        z_bin = round(spacing[0] / pixel)
        np.testing.assert_allclose(
            spacing, (pixel * z_bin, pixel, pixel), rtol=0, atol=1e-12
        )
        native_shapes, records, counts = [], [], {}
        for tile, time in zip(tiles, times, strict=True):
            z, y, x = tile.shape[-3:]
            rows = y if projection else 2 * z * z_bin
            crop = 0 if projection else int(np.ceil(rows * np.cos(np.pi / 6)))
            native = (1 if projection else y + crop, rows, x)
            position = counts.get(time, 0)
            counts[time] = position + 1
            native_shapes.append(native)
            records.append(
                {
                    "time_index": time,
                    "position_index": position,
                    "skewed_shape_syx": native,
                    "skewed_bounds_sx": (0, native[0], 0, x),
                    "deskew_crop_yx": (crop, crop + y, 0, x),
                }
            )
        native_shape = np.max(native_shapes, axis=0)
        fullest_time = max(counts, key=counts.get)
        source_positions = [
            positions[index] for index, time in enumerate(times) if time == fullest_time
        ]
        stem = f"fusion_{next(identifiers)}"
        source = opm_v2_acquisition_factory(
            np.zeros(
                (
                    max(times) + 1,
                    max(counts.values()),
                    tiles[0].shape[0],
                    *native_shape,
                ),
                np.uint16,
            ),
            name=stem,
            mode="projection" if projection else "mirror",
            pixel_size_um=pixel,
            scan_axis_step_um=pixel,
            stage_positions_zxy=source_positions,
        )
        path = tmp_path / f"{stem}_deskewed.ome.zarr"
        collection = create_variable_position_collection(
            path,
            [(1, *tile.shape) for tile in tiles],
            spacing,
            spatial_origins_zyx_um=positions,
            dtype=tiles[0].dtype,
            time_interval_s=time_interval_s,
        )
        state = ProcessingState.create(
            tmp_path / f"{stem}.processing.json", source.path
        )
        state.initialize_run(
            path,
            configuration={"crop_after_deskew": True},
            roi_series=records,
            overwrite=True,
        )
        for array, tile, record in zip(collection.arrays, tiles, records, strict=True):
            array.write(tile[None]).result()
            time, position = record["time_index"], record["position_index"]
            for channel in range(tile.shape[0]):
                state.complete_channel(
                    path, time, position, channel, is_zero=not np.any(tile[channel])
                )
            state.complete_tile(path, time, position)
        return TileFusion(
            path,
            **{
                "blend_pixels": (0, 0, 0),
                "downsample_factors": (1, 1, 1),
                "ssim_window": 3,
                "max_workers": 1,
                "multiscale_factors": (),
                "resolution_multiples": ((1, 1, 1),),
                "normalize_depth_intensity": False,
                **options,
            },
        )

    return create


@pytest.fixture
def registration_pair(tile_fusion):
    """Share pair geometry and registration state while keeping pixels explicit."""

    def create(
        volumes,
        positions,
        *,
        spacing=(0.23, 0.115, 0.115),
        sampling=(3, 2, 2),
        **options,
    ):
        """Build a fusion operator from prescribed pixels, support and geometry."""
        fusion = tile_fusion(
            [volume[None] for volume in volumes],
            positions,
            spacing=spacing,
            **{
                "downsample_factors": sampling,
                "ssim_window": 15,
                "threshold": 0.7,
                **options,
            },
        )
        # Cartesian unit operands are fully supported numerical inputs.
        fusion._tile_support_zy = [
            np.ones(volume.shape[:2], np.float32) for volume in volumes
        ]
        return fusion

    return create


@pytest.fixture
def block_fusion(tile_fusion):
    """Configure bounded block writes shared by mask and channel-weight tests."""

    def create(
        tiles,
        positions,
        shape,
        *,
        support=None,
        present=None,
        block_shape=None,
        output_dtype=np.uint16,
        time_indices=None,
        profiles=None,
        depth_gains=None,
        x_profile=None,
        y_profile=None,
        max_workers=1,
        max_in_flight_writes=1,
        fusion_ram_fraction=1,
        chunk_shape_yx=None,
    ):
        """Build a fusion operator from prescribed pixels, support and geometry."""
        fusion = tile_fusion(
            tiles,
            positions,
            time_indices=time_indices,
            max_workers=max_workers,
            max_in_flight_writes=max_in_flight_writes,
            fusion_ram_fraction=fusion_ram_fraction,
            chunk_shape_yx=shape[1:] if chunk_shape_yx is None else chunk_shape_yx,
        )
        fusion.offset_um, fusion.padded_shape = (0, 0, 0), shape
        fusion.output_dtype = np.dtype(output_dtype)
        fusion.fused_ts, fusion.write_block_shape = fusion._prepare_fused_image(
            fusion.output_dir / f"{fusion.acquisition_name}_fused.ome.zarr"
        )
        if block_shape is not None:
            fusion.write_block_shape = block_shape
        fusion._tile_support_zy = (
            [np.ones(tile.shape[-3:-1], np.float32) for tile in tiles]
            if support is None
            else [support] * len(tiles)
            if isinstance(support, np.ndarray) and support.ndim == 2
            else support
        )
        if present is not None:
            fusion._tile_channel_nonzero = present
        if profiles is not None:
            fusion._tile_profiles = profiles
        else:
            for _z_weights, y_weights, x_weights in fusion._tile_profiles:
                if y_profile is not None:
                    y_weights[:] = y_profile
                if x_profile is not None:
                    x_weights[:] = x_profile
        if depth_gains is not None:
            fusion._depth_intensity_gains = depth_gains
        return fusion

    return create


@pytest.fixture
def retry_fusion(tile_fusion):
    """Provide a controlled registration graph with independently specified links."""
    limits = {
        (0, 1): (13, 198, 33),
        (1, 2): (13, 198, 33),
        (3, 4): (9, 202, 31),
        (4, 5): (9, 202, 31),
    }
    fusion = tile_fusion(
        [np.ones((2, 1, 16, 16), np.uint16)] * 6,
        [(0, 0, 0)] * 6,
        time_indices=(0, 0, 0, 1, 1, 1),
        max_registration_shift_zyx=(13, 202, 33),
        registration_retries=2,
        downsample_factors=(3, 5, 5),
        ssim_window=15,
        channel_to_use=1,
        threshold=0.7,
        optimization_rel_threshold=0.5,
        optimization_abs_threshold=1.5,
    )
    fusion._automatic_registration_limits = True
    fusion._pair_registration_limits = limits.copy()
    fusion._initial_pair_registration_limits = limits.copy()
    fusion._registration_pair_sampling = dict.fromkeys(limits, (1, 5, 5))
    return fusion


@pytest.fixture(scope="session")
def registration_specimen(microscope_parameters):
    """Share immutable landmark truth for analytical units and oblique cameras."""
    from tests.simulation.scenes import prepare_scene

    @lru_cache(maxsize=8)
    def create(shape, *, seed=51):
        """Distribute known emitters throughout the prescribed physical field."""
        microscope = replace(
            microscope_parameters, fine_spacing_um=0.1, scan_step_um=0.23
        )
        spacing = np.array((microscope.scan_step_um, *[microscope.pixel_size_um] * 2))
        half_extent = np.maximum((np.asarray(shape) - 1) * spacing / 2, spacing)
        # These are object positions, not simulated camera intensities.
        count = max(32, int(np.prod(2 * half_extent) * 4))
        centers = np.random.default_rng(seed).uniform(
            -half_extent, half_extent, (count, 3)
        )
        return prepare_scene(
            "points",
            microscope,
            parameters={
                "half_extent_zyx_um": half_extent + 0.3,
                "features": {"centers_xyz_um": centers[:, ::-1].tolist()},
            },
        )

    return create


@pytest.fixture
def aperiodic_registration_object(registration_specimen):
    """Supply a known unblurred object for exact finite-window FFT alias units."""

    def create(shape, *, seed, equal_column_histograms=False):
        """Center a declared physical object domain before any camera observation."""
        scene = registration_specimen(shape, seed=seed)
        starts = (np.array(scene.density.shape) - np.asarray(shape)) // 2
        selection = tuple(
            slice(start, start + length)
            for start, length in zip(starts, shape, strict=True)
        )
        field = scene.density[selection]
        assert field.shape == shape and np.any(field > 0)
        if equal_column_histograms:
            # This exact alias operand uses permutations of the original known
            # landmark projection. Each column has the same intensity CDF, so
            # matching the two windows' global histograms is the identity.
            # Prescribed permutations vary spatial arrangement along X. These
            # are known landmark intensities, not camera-noise realizations or
            # another physical observation.
            landmarks = field.max(axis=2)
            arrangements = np.random.default_rng(seed)
            field = np.stack(
                [
                    arrangements.permutation(landmarks.ravel()).reshape(shape[:2])
                    for _ in range(shape[2])
                ],
                axis=2,
            )
            field.setflags(write=False)
        return field

    return create


@pytest.fixture(scope="session")
def registration_observation_pair(registration_specimen):
    """Blur Cartesian processed-coordinate unit objects, then digitize independently."""
    from tests.simulation.camera import digitize
    from tests.simulation.forward_model import Acquisition, forward_model
    from tests.simulation.scenes import freeze_metadata

    @lru_cache(maxsize=2)
    def create(shape, delta, *, seed=51):
        """Crop known shifted operands without copying a camera-noise field."""
        delta = np.asarray(delta, int)
        union_shape = tuple(np.asarray(shape) + abs(delta))
        specimen = registration_specimen(union_shape, seed=seed)
        scene = SimpleNamespace(
            expectation=forward_model(
                specimen.density,
                Acquisition(
                    specimen.axes,
                    union_shape,
                    geometry="cartesian",
                    camera_photon_budget=(1000, 30),
                    output="expectation",
                ),
                specimen.microscope,
            ),
            microscope=specimen.microscope,
        )
        scene.expectation = freeze_metadata(scene.expectation)
        starts = (np.maximum(0, -delta), np.maximum(0, delta))
        volumes = []
        for index, start in enumerate(starts):
            selection = tuple(
                slice(offset, offset + length)
                for offset, length in zip(start, shape, strict=True)
            )
            # These are analytical Cartesian operands, not raw OPM frames.
            expected = scene.expectation[selection]
            volumes.append(
                digitize(expected, scene.microscope.camera, seed=seed + index + 1)
            )
        return SimpleNamespace(
            volumes=freeze_metadata(volumes),
            starts=freeze_metadata(starts),
            scene=scene,
        )

    return create


@pytest.fixture
def shifted_slab_pair(registration_observation_pair):
    """Share known Cartesian slabs with independent detector realizations."""

    def create(
        shape, nominal_step, correction, *, seed=51, spacing=(0.23, 0.115, 0.115)
    ):
        """Return paired pixels, nominal placement and independent exact shifts."""
        shift = np.asarray(correction, dtype=int)
        delta = shift + np.array((nominal_step, 0, 0))
        observation = registration_observation_pair(shape, tuple(delta), seed=seed)
        return SimpleNamespace(
            volumes=observation.volumes,
            positions=[(0, 0, 0), (nominal_step * spacing[0], 0, 0)],
            spacing=spacing,
            expected=shift,
            union_shape=observation.scene.expectation.shape,
            starts=observation.starts,
        )

    return create


@pytest.fixture
def projection_fusion(tmp_path):
    """Share projection geometry and native TensorStore arrays for chunk tests."""
    from opm_processing.imageprocessing.maxtilefusion import MaxTileFusion

    def create(tiles, positions, shape, *, chunk_size=3):
        """Build a bounded projection fuser from explicit TCZYX camera data."""
        fusion = MaxTileFusion(
            ts_dataset=tuple(ts.array(tile) for tile in tiles),
            tile_positions=np.asarray(positions, float),
            output_path=tmp_path / "projection_fused.ome.zarr",
            pixel_size=(1, 1),
            blend_pixels=(0, 0),
            chunk_size=chunk_size,
            padding_multiple=1,
            reverse_stage_y=False,
        )
        assert fusion.fused_shape == shape
        return fusion

    return create


@pytest.fixture
def fusion_mosaic(aperiodic_registration_object):
    """Prescribe a Cartesian unit object with attenuated, partial tile operands."""
    from opm_processing.process import format_processed_output

    shape = (28, 46, 69)
    # Preserve the prescribed integer intensity + fractional offset, avoiding
    # cast boundaries while testing masks, gains and block layouts exactly.
    field = aperiodic_registration_object(shape, seed=51)
    assert 0 <= field.min() <= field.max() <= np.iinfo(np.uint16).max
    field = format_processed_output(field, save_float32=False).astype(np.float32)
    field += np.float32(0.375)
    field[:, 2, 4] = 0  # Real darkness inside valid support.
    tiles, origins, supports, profiles, gains, present = (
        [],
        [],
        [],
        [],
        [],
        {},
    )
    expected = np.zeros((2, 2, *shape), np.float32)
    for time in range(2):
        for position, (origin, extent) in enumerate(
            zip(
                ((0, 0, 0), (0, 11, 23), (9, 23, 7)),
                ((21, 35, 47), (21, 29, 41), (17, 21, 51)),
                strict=True,
            )
        ):
            tile_index = len(tiles)
            selection = tuple(
                slice(start, start + length)
                for start, length in zip(origin, extent, strict=True)
            )
            coverage = np.ones(extent[:2], np.float32)
            coverage[0, :2] = 0
            coverage[-1, 1:] = 0.5
            x_profile = np.linspace(0.25, 1, extent[2], dtype=np.float32)
            x_profile[-1] = 0  # No detector support in a padded column.
            gain = np.array((1.25, 2), np.float32) + position * 0.25
            truth = np.stack(
                [field[selection] * (1 + time + channel) for channel in range(2)]
            )
            pixels = truth / gain[:, None, None, None] * coverage[None, ..., None]
            pixels[..., -1] = 0
            for channel in range(2):
                active = not (position == 1 and channel == time)
                present[tile_index, channel] = active
                if not active:
                    pixels[channel] = 0
                    continue
                mask = (coverage[..., None] > 0) & (x_profile[None, None] > 0)
                target = expected[(time, channel, *selection)]
                target[mask] = truth[channel][mask]
            tiles.append(pixels.astype(np.float32))
            origins.append(origin)
            supports.append(coverage)
            profiles.append(
                (
                    np.ones(extent[0], np.float32),
                    np.linspace(0.5, 1, extent[1], dtype=np.float32),
                    x_profile,
                )
            )
            gains.append(gain)
    return SimpleNamespace(
        tiles=tiles,
        origins=origins,
        shape=shape,
        supports=supports,
        profiles=profiles,
        gains=np.asarray(gains),
        present=present,
        expected=expected,
    )


@pytest.fixture
def weighted_fusion_case(strided, shared, partial):
    """Provide affine photon fields and prescribed scalar mixing/coverage truths."""
    first = np.arange(3 * 4 * 5 * 10, dtype=np.float32).reshape(3, 4, 5, 10)
    first[0, 1, 1, 1] = 0
    first[0, 1, 1, 3] = -20
    first[1, 1, 1, 1] = 90000
    gains = np.asarray((2, 4, 8), np.float32)
    # Exact inverse transmissions isolate the operator's arithmetic from the
    # separate physical attenuation-estimation tests.
    second = (first * np.float32(0.5) + np.float32(10)) / gains[:, None, None, None]
    depth_gains = (None, gains)
    crop = np.s_[:, 1:3, 1:4, 1:9:2] if strided else np.s_[:, 1:3, 1:4, 1:5]
    source_views = (first[crop], second[crop])
    present = (
        (np.ones(3, bool), np.ones(3, bool))
        if shared
        else (np.asarray((True, True, False)), np.asarray((True, False, True)))
    )
    support = np.asarray(((True, False, True), (False, True, True)))
    if partial:
        support = support.astype(np.float32)
        support[0, 0] = 0.5
        for pixels in source_views:
            pixels[:, 0, 0] *= np.float32(0.5)
    z_weights = np.asarray((0.5, 1), np.float32)
    y_weights = np.asarray((1, 0.25, 0.5), np.float32)
    x_weights = np.asarray((1, 0.5, 0.25, 1), np.float32)
    accumulated = np.zeros((3, 4, 7, 10), np.float32)
    weights = np.zeros((1 if shared else 3, *accumulated.shape[1:]), np.float32)
    selection = np.s_[:, 1:3, 2:5, 3:7]
    source_origin = (0, 0, 0) if strided else (1, 1, 1)
    coverage_factor = np.float32(0.5 if partial else 1)
    unit_weights = (1,) if shared else (1, 0.5, 0.5)
    half_weights = (0.5,) if shared else (0.5, 0.25, 0.25)
    probes = (
        (
            (1, 2, 3),
            (5, 67505, 350.75) if shared else (5, 90000, 240.5),
            (5, 65535, 350) if shared else (5, 65535, 240),
            unit_weights,
            coverage_factor,
        ),
        (
            (1, 4, 3),
            (65.75, 215.75, 365.75) if shared else (65.75, 281, 250.5),
            (65, 215, 365) if shared else (65, 281, 250),
            half_weights,
            1,
        ),
        (
            (2, 3, 3),
            (95.75, 245.75, 395.75) if shared else (95.75, 321, 270.5),
            (95, 245, 395) if shared else (95, 321, 270),
            half_weights,
            1,
        ),
        (
            (2, 4, 6),
            ((107.75, 257.75, 407.75) if shared else (107.75, 337, 278.5))
            if strided
            else ((105.5, 255.5, 405.5) if shared else (105.5, 334, 277)),
            ((107, 257, 407) if shared else (107, 337, 278))
            if strided
            else ((105, 255, 405) if shared else (105, 334, 277)),
            unit_weights,
            1,
        ),
    )
    untouched = np.ones(accumulated.shape[1:], bool)
    untouched[selection[1:]] = False
    return SimpleNamespace(
        first=first,
        second=second,
        source_views=source_views,
        present=present,
        depth_gains=depth_gains,
        support=support,
        z_weights=z_weights,
        y_weights=y_weights,
        x_weights=x_weights,
        accumulated=accumulated,
        weights=weights,
        origin_sum=(5, 67505, 350.75) if shared else (5, 45000, 120.25),
        coverage_factor=coverage_factor,
        negative_x=1 if strided else 2,
        negative_sum=-5 if strided else -2.5,
        negative_weight=0.5 if strided else 0.25,
        probes=probes,
        untouched=untouched,
        source_origin=source_origin,
    )
