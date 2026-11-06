"""Minimal numerical fusion operators with explicitly supplied geometry and data."""

from dataclasses import dataclass, replace
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pytest
import tensorstore as ts

from opm_processing.imageprocessing.opmpsf import centered_sensor_axes
from opm_processing.imageprocessing.tilefusion import (
    TileFusion,
    _infer_registration_limits,
)


@dataclass(frozen=True)
class RegistrationScene:
    """Immutable emitted object, detector axes and one recorded camera exposure."""

    raw: np.ndarray
    means: np.ndarray
    dose: np.ndarray
    ground_truth: object
    microscope: object
    sensor_axes: tuple[np.ndarray, ...]

    def __post_init__(self):
        """Protect camera and optical data reused by multiple registration units."""
        for array in (self.raw, self.means, self.dose, *self.sensor_axes):
            array.setflags(write=False)


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


@pytest.fixture(scope="session")
def registration_scene(microscope_parameters):
    """Render a known fluorescent point specimen with the shared optical model.

    A spatially distributed specimen supplies features throughout thin overlaps.
    Exact signal units use the original object or noiseless optical means;
    physical translations obtain independent exposures from the same dose.
    """
    from tests.simulation.forward_model import Acquisition, forward_model
    from tests.simulation.scenes import prepare_scene

    @lru_cache(maxsize=8)
    def render(shape, *, seed=51):
        """Return unsigned camera samples and the object that produced them."""
        microscope = replace(
            microscope_parameters, fine_spacing_um=0.1, scan_step_um=0.23
        )
        spacing = np.array((microscope.scan_step_um, *[microscope.pixel_size_um] * 2))
        half_extent = np.maximum((np.asarray(shape) - 1) * spacing / 2, spacing)
        # Coordinates are object positions, not random image intensities. The
        # same vectorial PSF and pixel integration used by all optics tests
        # form these photons before the calibrated camera digitizes them.
        count = max(32, int(np.prod(2 * half_extent) * 4))
        centers = np.random.default_rng(seed).uniform(
            -half_extent, half_extent, (count, 3)
        )
        scene = prepare_scene(
            "points",
            microscope,
            parameters={
                "half_extent_zyx_um": half_extent + 0.3,
                "features": {"centers_xyz_um": centers[:, ::-1].tolist()},
            },
        )
        sensor_axes = centered_sensor_axes(shape, spacing)
        dose = forward_model(
            scene.density,
            Acquisition(
                scene.axes,
                shape,
                geometry="cartesian",
                camera_photon_budget=(1000, 30),
                output="object",
            ),
            microscope,
        )
        means = forward_model(
            dose,
            Acquisition(scene.axes, shape, geometry="cartesian", output="expectation"),
            microscope,
        )
        from tests.simulation.camera import digitize

        raw = digitize(means, microscope.camera, seed=seed)
        return RegistrationScene(raw, means, dose, scene, microscope, sensor_axes)

    return render


@pytest.fixture
def aperiodic_registration_object(registration_scene):
    """Supply a known unblurred object for exact finite-window FFT alias units."""

    def create(shape, *, seed, equal_column_histograms=False):
        """Center a declared physical object domain before any camera observation."""
        scene = registration_scene(shape, seed=seed).ground_truth
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


@pytest.fixture
def registration_observation_pair(registration_scene):
    """Observe one fixed emission dose on independently digitized sensor grids."""
    from tests.simulation.camera import digitize
    from tests.simulation.forward_model import Acquisition, forward_model

    def create(shape, delta, *, seed=51):
        """Translate actual sensor coordinates without copying a camera-noise field."""
        delta = np.asarray(delta, int)
        scene = registration_scene(tuple(np.asarray(shape) + abs(delta)), seed=seed)
        starts = (np.maximum(0, -delta), np.maximum(0, delta))
        volumes, means = [], []
        for index, start in enumerate(starts):
            axes = tuple(
                axis[offset : offset + length]
                for axis, offset, length in zip(
                    scene.sensor_axes, start, shape, strict=True
                )
            )
            expected = forward_model(
                scene.dose,
                Acquisition(
                    scene.ground_truth.axes,
                    shape,
                    sensor_axes=axes,
                    geometry="cartesian",
                    output="expectation",
                ),
                scene.microscope,
            )
            means.append(expected)
            volumes.append(
                digitize(expected, scene.microscope.camera, seed=seed + index + 1)
            )
        return SimpleNamespace(
            volumes=tuple(volumes), means=tuple(means), starts=starts, scene=scene
        )

    return create


@pytest.fixture
def shifted_slab_pair(registration_observation_pair):
    """Share known translated slabs with independent detector realizations."""
    from tests.simulation.camera_likelihood import (
        SamplingEnvelope,
        camera_sampling_envelope,
    )

    def create(
        shape, nominal_step, correction, *, seed=51, spacing=(0.23, 0.115, 0.115)
    ):
        """Return paired pixels, nominal placement and independent exact shifts."""
        shift = np.asarray(correction, dtype=int)
        delta = shift + np.array((nominal_step, 0, 0))
        observation = registration_observation_pair(shape, delta, seed=seed)
        union_shape = observation.scene.means.shape
        total = np.zeros(union_shape, np.float64)
        count = np.zeros(union_shape, np.uint8)
        low = np.full(union_shape, np.nan)
        high = np.full(union_shape, np.nan)
        for start, raw, expected in zip(
            observation.starts, observation.volumes, observation.means, strict=True
        ):
            selection = tuple(
                slice(offset, offset + length)
                for offset, length in zip(start, shape, strict=True)
            )
            total[selection] += raw
            count[selection] += 1
            bounds = camera_sampling_envelope(
                expected,
                raw,
                observation.scene.microscope.camera,
                radii=(0, 0, 0),
                views=2,
            )
            low[selection] = np.fmin(low[selection], bounds.lower)
            high[selection] = np.fmax(high[selection], bounds.upper)
        # With unity feathering, two coincident measurements have the closed
        # arithmetic-mean expectation; it is independent of the block renderer.
        expected_pixels = np.divide(
            total, count, out=np.zeros_like(total), where=count > 0
        )
        coordinate_axes = observation.scene.sensor_axes
        extent = tuple(axis[-1] for axis in coordinate_axes)
        halo = observation.scene.microscope.psf_half_extent_zyx_um
        interior = np.ones(union_shape, bool)
        for dimension, (axis, half_extent, margin) in enumerate(
            zip(coordinate_axes, extent, halo, strict=True)
        ):
            reshape = [1, 1, 1]
            reshape[dimension] = len(axis)
            interior &= np.abs(axis).reshape(reshape) < half_extent - margin
        return SimpleNamespace(
            volumes=observation.volumes,
            positions=[(0, 0, 0), (nominal_step * spacing[0], 0, 0)],
            spacing=spacing,
            expected=shift,
            truth=observation.scene.means,
            starts=observation.starts,
            observation=observation,
            expected_pixels=expected_pixels,
            expected_envelope=SamplingEnvelope(low, high),
            accuracy_mask=interior & (count > 0),
        )

    return create


@pytest.fixture
def projection_fusion(fusion_operator):
    """Share projection geometry and native TensorStore arrays for chunk tests."""
    from opm_processing.imageprocessing.maxtilefusion import MaxTileFusion

    def create(tiles, positions, shape, *, chunk_size=3):
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
        )

    return create


@pytest.fixture
def fusion_mosaic(registration_scene):
    """Prescribe one laboratory field seen by cropped, attenuated, partial tiles."""
    shape = (28, 46, 69)
    # Retain the fractional calibrated intensity of this float-output case.
    # Integer-cast boundary arithmetic is checked separately using
    # explicitly prescribed scalar weights and photon counts.
    field = registration_scene(shape).raw.astype(np.float32) + np.float32(0.375)
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
