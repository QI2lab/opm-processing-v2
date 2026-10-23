"""Minimal numerical fusion operators with explicitly supplied geometry and data."""

from types import SimpleNamespace

import numpy as np
import pytest
import tensorstore as ts

from opm_processing.imageprocessing.tilefusion import (
    TileFusion,
    _infer_registration_limits,
)


@pytest.fixture
def fusion_operator():
    """Return a factory for isolated fusion calculations without file discovery.

    Tests supply only the state consumed by their calculation.
    """

    def create(operator=TileFusion, *, source_tiles=None, **attributes):
        """Construct an operator from a complete, visible unit-test configuration."""
        if source_tiles is not None:
            channels, z_dim, y_dim, x_dim = source_tiles[0].shape
            attributes = {
                "time_dim": 1,
                "position_dim": len(source_tiles),
                "channels": channels,
                "z_dim": z_dim,
                "y_dim": y_dim,
                "x_dim": x_dim,
                "_is_2d": False,
                "z_profile": np.ones(z_dim, np.float32),
                "y_profile": np.ones(y_dim, np.float32),
                "x_profile": np.ones(x_dim, np.float32),
                "position_arrays": tuple(ts.array(tile[None]) for tile in source_tiles),
                **attributes,
            }
        instance = operator.__new__(operator)
        instance.__dict__.update(attributes)
        return instance

    return create


@pytest.fixture
def registration_pair(fusion_operator):
    """Share pair geometry and registration state while keeping pixels explicit."""

    def create(
        volumes,
        positions,
        *,
        spacing=(0.23, 0.115, 0.115),
        sampling=(3, 2, 2),
        **settings,
    ):
        """Build a fusion operator from prescribed pixels, support and geometry."""
        shapes = [volume.shape for volume in volumes]
        limits = _infer_registration_limits(positions, shapes, spacing, sampling, 30)
        attributes = {
            "_tile_positions": positions,
            "_tile_shapes": shapes,
            "_tiles_by_time": ((0, 1),),
            "_tile_time_indices": [0, 0],
            "_tile_support_zy": [np.ones(shape[:2], np.float32) for shape in shapes],
            "_pixel_size": spacing,
            "downsample_factors": sampling,
            "ssim_window": 15,
            "threshold": 0.7,
            "pairwise_metrics": {},
            "_pair_registration_limits": limits,
            "max_registration_shift_zyx": limits[0, 1],
            "_max_workers": 1,
            "_debug": False,
            "_tile_channel_nonzero": {(0, 0): True, (1, 0): True},
        }
        attributes.update(settings)
        return fusion_operator(
            source_tiles=[volume[None] for volume in volumes], **attributes
        )

    return create


@pytest.fixture
def block_fusion(fusion_operator):
    """Configure bounded block writes shared by mask and channel-weight tests."""

    def create(
        tiles,
        positions,
        shape,
        *,
        support=None,
        present=None,
        block_shape=None,
        **settings,
    ):
        """Build a fusion operator from prescribed pixels, support and geometry."""
        channels, z_dim, y_dim, _ = tiles[0].shape
        attributes = {
            "fused_ts": ts.array(np.zeros((1, channels, *shape), dtype=np.uint16)),
            "write_block_shape": block_shape or [1, 1, *shape],
            "offset_um": (0, 0, 0),
            "padded_shape": shape,
            "_pixel_size": (1, 1, 1),
            "_tile_positions": positions,
            "chunk_y": shape[1],
            "chunk_x": shape[2],
            "fusion_ram_fraction": 1.0,
            "max_in_flight_writes": 1,
            "_max_workers": 1,
            "_debug": False,
            "_deskew_support_zy": np.ones((z_dim, y_dim), bool)
            if support is None
            else support,
            "_tile_channel_nonzero": present
            or {
                (tile, channel): True
                for tile in range(len(tiles))
                for channel in range(channels)
            },
        }
        attributes.update(settings)
        return fusion_operator(source_tiles=tiles, **attributes)

    return create


@pytest.fixture
def registration_cache(tmp_path, processing_state):
    """Attach the production durable registration journal to unit operators."""

    def attach(fusion):
        """Supply persistence paths and retain the real save/load implementations."""
        fusion.data = tmp_path / "sample_deskewed.ome.zarr"
        fusion.processing_state = processing_state
        processing_state.initialize_run(fusion.data, configuration={}, overwrite=True)
        fusion.__dict__.setdefault("reverse_stage_y", True)
        fusion.__dict__.setdefault("reverse_stage_z", True)
        return processing_state

    return attach


@pytest.fixture
def retry_fusion(fusion_operator):
    """Provide a controlled registration graph with independently specified links."""
    from opm_processing.imageprocessing.tilefusion import TileFusion

    limits = {
        (0, 1): (13, 198, 33),
        (1, 2): (13, 198, 33),
        (3, 4): (9, 202, 31),
        (4, 5): (9, 202, 31),
    }
    return fusion_operator(
        TileFusion,
        _tile_positions=[(0, 0, 0)] * 6,
        _tile_shapes=[(1, 1, 1)] * 6,
        _pixel_size=(1, 1, 1),
        _is_2d=False,
        _tiles_by_time=((0, 1, 2), (3, 4, 5)),
        _automatic_registration_limits=True,
        _pair_registration_limits=limits.copy(),
        _initial_pair_registration_limits=limits.copy(),
        _registration_pair_sampling=dict.fromkeys(limits, (1, 5, 5)),
        max_registration_shift_zyx=(13, 202, 33),
        _initial_registration_shift_zyx=(13, 202, 33),
        registration_retries=2,
        downsample_factors=(3, 5, 5),
        ssim_window=15,
        channel_to_use=1,
        threshold=0.7,
        optimization_rel_threshold=0.5,
        optimization_abs_threshold=1.5,
        pairwise_metrics={},
        reverse_stage_y=True,
        reverse_stage_z=True,
        _tile_channel_nonzero={(index, 1): True for index in range(6)},
    )


@pytest.fixture
def shifted_slab_pair():
    """Build known translated slabs from one prescribed blurred random field."""
    from scipy.ndimage import gaussian_filter

    def create(
        shape, nominal_step, correction, *, seed=51, spacing=(0.23, 0.115, 0.115)
    ):
        """Return paired pixels, nominal placement and independent exact shifts."""
        shift = np.asarray(correction, dtype=int)
        delta = shift + np.array((nominal_step, 0, 0))
        truth = np.rint(
            gaussian_filter(
                np.random.default_rng(seed).uniform(
                    0, 1000, np.asarray(shape) + abs(delta)
                ),
                (1, 2, 2),
            )
        ).astype(np.uint16)
        starts = (np.maximum(0, -delta), np.maximum(0, delta))
        volumes = tuple(
            truth[
                tuple(
                    slice(start, start + length)
                    for start, length in zip(origin, shape, strict=True)
                )
            ]
            for origin in starts
        )
        return SimpleNamespace(
            volumes=volumes,
            positions=[(0, 0, 0), (nominal_step * spacing[0], 0, 0)],
            spacing=spacing,
            expected=shift,
            truth=truth,
            starts=starts,
        )

    return create


@pytest.fixture
def projection_fusion(fusion_operator):
    """Share projection geometry and native TensorStore arrays for chunk tests."""
    from opm_processing.imageprocessing.maxtilefusion import MaxTileFusion

    def create(tiles, positions, shape, *, chunk_size=3, **settings):
        """Build a bounded projection fuser from explicit TCZYX camera data."""
        channels, height, width = tiles[0].shape[1], *tiles[0].shape[-2:]
        return fusion_operator(
            MaxTileFusion,
            pad_y=0,
            pad_x=0,
            ts_dataset=tuple(ts.array(tile) for tile in tiles),
            tile_positions=np.asarray(positions, float),
            pixel_size=np.ones(2),
            offset=(0, 0),
            time_dim=tiles[0].shape[0],
            channels=channels,
            tile_shape=(height, width),
            fused_shape=shape,
            time_range=None,
            chunk_size=chunk_size,
            weight_mask=np.ones((height, width), np.float32),
            output_dtype=tiles[0].dtype,
            fused_ts=ts.array(
                np.zeros((tiles[0].shape[0], channels, 1, *shape), tiles[0].dtype)
            ),
            **settings,
        )

    return create


@pytest.fixture
def fusion_mosaic():
    """Prescribe one laboratory field seen by cropped, attenuated, partial tiles."""
    shape = (28, 46, 69)
    z, y, x = np.indices(shape)
    field = 100.375 + 3 * z + 2 * y + x
    field[:, 2, 4] = 0  # Real darkness inside valid support.
    tiles, origins, shapes, supports, profiles, gains, present = (
        [],
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
            shapes.append(extent)
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
        shapes=shapes,
        supports=supports,
        profiles=profiles,
        gains=np.asarray(gains),
        present=present,
        expected=expected,
    )


@pytest.fixture
def weighted_fusion_case(strided, shared, partial):
    """Provide strided photon fields, partial coverage and independent feather weights."""
    first = np.arange(3 * 4 * 5 * 10, dtype=np.float32).reshape(3, 4, 5, 10)
    first[0, 1, 1, 1] = 0
    first[0, 1, 1, 3] = -20
    first[1, 1, 1, 1] = 90000
    second = first * np.float32(0.5) + np.float32(10)
    attenuation_per_um = np.asarray((0.025, 0.05, 0.075), np.float32)
    optical_path_um = np.float32(8)
    second *= np.exp(-attenuation_per_um * optical_path_um)[:, None, None, None]
    depth_gains = (None, np.exp(attenuation_per_um * optical_path_um))
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
    spatial_weights = (z_weights[:, None, None] * y_weights[None, :, None]) * x_weights[
        None, None, :
    ]
    spatial_weights *= (support > 0)[..., None]
    accumulated = np.zeros((3, 4, 7, 10), np.float32)
    weights = np.zeros((1 if shared else 3, *accumulated.shape[1:]), np.float32)
    expected_sum = np.zeros_like(accumulated)
    expected_weight = np.zeros_like(accumulated)
    selection = np.s_[:, 1:3, 2:5, 3:7]
    source_origin = (0, 0, 0) if strided else (1, 1, 1)
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
        spatial_weights=spatial_weights,
        accumulated=accumulated,
        weights=weights,
        expected_sum=expected_sum,
        expected_weight=expected_weight,
        selection=selection,
        source_origin=source_origin,
    )
