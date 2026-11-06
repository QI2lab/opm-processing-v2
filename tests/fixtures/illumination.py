"""Known detector fields and shared camera/storage boundaries for illumination tests."""

from types import SimpleNamespace

import numpy as np
import pytest
import tensorstore as ts


@pytest.fixture
def projection_illumination_case(opm_v2_projection_zarr, simulated_camera):
    """Retain original optical camera means for the actual projection producer."""
    dataset = opm_v2_projection_zarr
    means = np.empty_like(dataset.raw_data[:, 0], dtype=np.float64)
    for time, channel in np.ndindex(means.shape[:2]):
        observation = simulated_camera(
            (1, *means.shape[-2:]),
            geometry="cartesian",
            exposure=1 + time / 4 + channel / 2,
        )
        means[time, channel] = observation.expectation[0]
    return SimpleNamespace(dataset=dataset, means=means)


@pytest.fixture
def illumination_calibration_ensemble(roi_camera_scene):
    """Provide identifiable optical calibration before the unchanged point sample."""
    from tests.simulation.camera import digitize
    from tests.simulation.forward_model import Acquisition, forward_model
    from tests.simulation.objects import centered_axis

    def create():
        """Illuminate a uniform physical object covering the sensor and PSF support."""
        scene = roi_camera_scene(shape=(24, 33, 33))
        microscope = scene.microscope
        scans, rows, columns = scene.pixels.shape[-3:]
        pixel = microscope.pixel_size_um
        angle = np.deg2rad(microscope.angle_deg)
        camera_half_extent = (
            rows * pixel * np.sin(angle) / 2,
            ((scans - 1) * microscope.scan_step_um + rows * pixel * np.cos(angle)) / 2,
            columns * pixel / 2,
        )
        axes = tuple(
            centered_axis(extent + halo, microscope.fine_spacing_um)
            for extent, halo in zip(
                camera_half_extent, microscope.psf_half_extent_zyx_um, strict=True
            )
        )
        # Constant emission over the full optical support makes illumination
        # identifiable independently of sparse sample structure at timepoint1.
        calibration_mean = forward_model(
            np.ones(tuple(map(len, axes)), dtype=np.float32),
            Acquisition(
                axes,
                scene.pixels.shape[-3:],
                camera_photon_budget=(2000, 0),
                output="expectation",
            ),
            microscope,
        )
        calibration_raw = digitize(calibration_mean, microscope.camera, seed=71)
        assert calibration_raw.max() < np.iinfo(np.uint16).max
        return SimpleNamespace(
            scene=scene,
            raw=np.concatenate((calibration_raw[None, None, None], scene.pixels)),
            calibration_mean=calibration_mean,
            calibration_photons=2000 * microscope.quantum_efficiency,
            sample_timepoint=1,
        )

    return create


@pytest.fixture
def illumination_observations(request):
    """Observe a uniform analytical calibration object at prescribed exposures."""
    kind = request.param
    tiled = kind == "tiles"
    positions, channels = (6, 2) if tiled else (2, 2 if kind == "masked" else 1)
    height, width = (64, 128) if tiled else (32, 64)
    y = np.linspace(-1, 1, height, dtype=np.float32)[:, None]
    x = np.linspace(-1, 1, width, dtype=np.float32)[None, :]
    if tiled:
        fields = np.stack(
            (
                1 + 0.28 * y + 0.12 * x + 0.08 * y * x,
                1 - 0.22 * y + 0.16 * x - 0.06 * y * x,
            )
        )
        detector_x = 78
        stripe = 1 - 0.12 * np.exp(-0.5 * ((np.arange(width) - detector_x) / 3) ** 2)
        fields *= stripe[None, None]
        fields /= fields.mean(axis=(1, 2), keepdims=True)
        fields = np.broadcast_to(fields[None], (positions, channels, height, width))
    else:
        fields = np.stack((1 + 0.25 * x + 0.1 * y, 1 - 0.2 * x + 0.15 * y))
        fields = np.repeat(fields[:, None], channels, axis=1)
        detector_x = None
    offset, conversion, source = 100.0, 0.25, 1000.0
    specimen = np.broadcast_to(
        source * np.linspace(0.75, 1.25, 12)[None, None, :, None, None],
        (positions, channels, 12, height, width),
    )
    from tests.simulation.camera import Camera, digitize

    raw = digitize(
        specimen * fields[:, :, None],
        Camera(
            read_noise_e=1.6,
            electrons_per_adu=conversion,
            offset_adu=offset,
            bit_depth=16,
        ),
        seed=71,
    )[None]
    coordinates = np.zeros((positions, 3))
    coordinates[:, 1] = np.arange(positions) * 100 if tiled else 20
    coordinates[:, 0] = np.arange(positions) * 10
    mask = np.asarray([[[True, False], [False, False]]]) if kind == "masked" else None
    expected = np.ones((channels, height, width), np.float32)
    for channel in range(channels):
        eligible = np.ones(positions, bool) if mask is None else mask[0, :, channel]
        if eligible.any():
            expected[channel] = fields[eligible, channel].max(axis=0)
    return SimpleNamespace(
        datastore=ts.array(raw),
        offset=offset,
        conversion=conversion,
        coordinates=coordinates,
        mask=mask,
        expected=expected,
        source=source,
        detector_x=detector_x,
    )


@pytest.fixture
def illumination_fit_case(monkeypatch, topology, storage):
    """Observe calibrated samples while the installed BaSiCPy performs the fit."""
    from opm_processing.imageprocessing import flatfield

    positions = 1 if topology == "single" else 3
    gain = np.linspace(0.5, 2, 7, dtype=np.float32)
    levels = (
        8
        + 16 * np.arange(positions)[:, None, None]
        + 32 * np.arange(2)[None, :, None]
        + 8 * np.arange(3)[None, None]
    )
    raw = np.broadcast_to(
        100 + levels[..., None, None] * gain / 0.5, (positions, 2, 3, 5, 7)
    ).astype(np.uint16)
    raw = np.stack((raw, raw + 1000))
    coordinates = None if topology == "single" else np.zeros((positions, 3))
    if topology == "tiles":
        coordinates[:, 1] = np.arange(positions)
    calls = []
    fit = flatfield.BaSiC.fit

    def observe_fit(solver, images, *args, **kwargs):
        """Record the calibrated images and execute the installed optimizer."""
        if isinstance(images, np.ndarray):
            calls.append(images.copy())
        return fit(solver, images, *args, **kwargs)

    monkeypatch.setattr(flatfield.BaSiC, "fit", observe_fit)
    return SimpleNamespace(
        datastore=ts.array(raw) if storage else raw,
        coordinates=coordinates,
        levels=levels,
        gain=gain,
        calls=calls,
        raw=raw,
    )


@pytest.fixture
def constant_illumination_acquisition(acquisition_factory, scan_planes):
    """Persist known constant photon fields over time, position and channel."""
    photons = 20 + 4 * np.arange(8).reshape(2, 2, 2)
    shape = (*photons.shape, scan_planes, 8, 12)
    raw = np.broadcast_to(100 + photons[..., None, None, None] / 0.25, shape).astype(
        np.uint16
    )
    dataset = acquisition_factory(
        raw,
        stage_positions_zxy=((10, 20, 30), (0, 20, 30)),
        camera_offset=100,
        camera_conversion=0.25,
    )
    fields = np.broadcast_to(
        np.array([2, 1.25], np.float32)[None, :, None, None], (1, 2, 8, 12)
    ).copy()
    return SimpleNamespace(dataset=dataset, photons=photons, fields=fields)
