"""Known detector fields and shared camera/storage boundaries for illumination tests."""

from types import SimpleNamespace

import numpy as np
import pytest
import tensorstore as ts

from opm_processing.imageprocessing.camera import camera_correct, illumination_correct
from opm_processing.imageprocessing.opmtools import (
    deskew_coverage,
    deskew_shape_estimator,
)


@pytest.fixture
def uniform_camera_scene(uniform_detector_scene):
    """Share real optical uniform phantoms before independent camera draws."""
    from dataclasses import replace

    from tests.simulation.camera import digitize

    def create(
        *,
        times=1,
        positions=1,
        channels=1,
        projection=False,
        fields=None,
        exposures=None,
        detector_artifacts=False,
        read_noise_e=None,
    ):
        """Declare emitted dose and detector response before sampling photons."""
        optical = uniform_detector_scene(projection)
        microscope = optical.microscope
        if read_noise_e is not None:
            microscope = replace(
                microscope,
                camera=replace(microscope.camera, read_noise_e=read_noise_e),
            )
        shape = (times, positions, channels, *optical.shape)
        response = np.broadcast_to(
            np.ones((channels, *optical.shape[-2:])) if fields is None else fields,
            (channels, *optical.shape[-2:]),
        ).copy()
        doses = (
            0.15 + 0.02 * np.arange(times * positions * channels).reshape(shape[:3])
            if exposures is None
            else np.broadcast_to(exposures, shape[:3])
        )
        source_means = np.broadcast_to(
            optical.means * doses[..., None, None, None], shape
        ).copy()
        means = source_means * response[None, None, :, None]
        if detector_artifacts:
            # A fixed detector-dark-current defect is a camera-plane photon
            # source. It precedes Poisson/read-noise sampling and ADC conversion.
            means[:, :, 0, :, 0, 0] += 100
            means[:, :, 0, 0, 0, 1:5] += 1
        pixels = np.empty(shape, np.uint16)
        for time, position, channel in np.ndindex(shape[:3]):
            pixels[time, position, channel] = digitize(
                means[time, position, channel],
                microscope.camera,
                seed=71 + 100 * time + 10 * position + channel,
            )
        assert pixels.max() < np.iinfo(np.uint16).max
        return SimpleNamespace(
            pixels=pixels,
            source_means=source_means,
            fields=response,
            microscope=microscope,
            projection=projection,
        )

    return create


@pytest.fixture
def assert_uniform_photon_accuracy():
    """Bound positive CPU interpolation using the original phantom and camera noise."""
    from tests.simulation.camera_likelihood import (
        SamplingEnvelope,
        camera_sampling_accuracy,
    )

    def check(data, scene, *, position=0, channels=None):
        """Check calibrated absolute photons, spatial support and output precision."""
        microscope = scene.microscope
        pixel = microscope.pixel_size_um
        gain = 1 if scene.projection else 2 * pixel / microscope.scan_step_um
        native = scene.pixels.shape[-3:]
        if scene.projection:
            assert data.shape[-3:] == native
            coverage = np.ones(data.shape[-3:])
        else:
            expected_shape, _, _, _ = deskew_shape_estimator(
                native,
                microscope.angle_deg,
                microscope.scan_step_um,
                pixel,
                crop_after_deskew=False,
            )
            assert data.shape[-3:] == tuple(expected_shape)
            geometry = {
                "theta": microscope.angle_deg,
                "distance": microscope.scan_step_um,
                "pixel_size": pixel,
            }
            coverage = np.zeros(data.shape[-3:])
            coverage[..., : native[2]] = deskew_coverage(
                native, data.shape[-3:], **geometry
            )[..., None]
        camera = microscope.camera
        selected = range(data.shape[1]) if channels is None else channels
        for time in range(data.shape[0]):
            for channel in selected:
                truth = scene.source_means[time, position, channel]
                np.testing.assert_allclose(truth, truth.flat[0], rtol=2e-5)
                field = scene.fields[channel]
                observed = illumination_correct(
                    camera_correct(
                        scene.pixels[time, position, channel],
                        camera.offset_adu,
                        camera.electrons_per_adu,
                    ),
                    field,
                ).astype(np.float64)
                residual = observed - truth
                expected = float(truth.flat[0]) * gain * coverage
                operations = 64
                error = operations * np.finfo(np.float32).eps
                roundoff = error / (1 - error) * max(float(observed.max()), 1) * gain
                quantum = 0 if data.dtype.kind == "f" else 1
                low = (float(truth.min()) + residual.min()) * gain * coverage
                high = (float(truth.max()) + residual.max()) * gain * coverage
                envelope = SamplingEnvelope(
                    np.maximum(low - roundoff - quantum, 0),
                    high + roundoff + quantum,
                )
                actual = data[time, channel]
                result = camera_sampling_accuracy(
                    actual, expected, envelope, coverage > 0
                )
                assert result.accepted, result
                np.testing.assert_array_equal(actual[coverage == 0], 0)

    return check


@pytest.fixture
def projection_illumination_case(opm_v2_projection_zarr, simulated_camera):
    """Retain original optical camera means for the actual projection producer."""
    dataset = opm_v2_projection_zarr
    means = np.empty_like(dataset.raw_data[:, 0, :, 0], dtype=np.float64)
    for time, channel in np.ndindex(means.shape[:2]):
        observation = simulated_camera(
            (1, *means.shape[-2:]),
            geometry="cartesian",
            exposure=1 + time / 4 + channel / 2,
        )
        means[time, channel] = observation.expectation[0]
    return SimpleNamespace(dataset=dataset, means=means)


@pytest.fixture(scope="session")
def uniform_optical_calibration():
    """Share immutable uniform-phantom means with excitation applied before blur."""
    from functools import lru_cache

    from tests.simulation.forward_model import Acquisition, forward_model
    from tests.simulation.objects import centered_axis

    @lru_cache(maxsize=12)
    def create(microscope, shape, profile="uniform", photon_budget=(2000, 0)):
        """Cover the entire sensor footprint and emission-PSF support before sampling."""
        scans, rows, columns = shape
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
        means = forward_model(
            np.ones(tuple(map(len, axes)), dtype=np.float32),
            Acquisition(
                axes,
                shape,
                illumination_profile=profile,
                camera_photon_budget=photon_budget,
                output="expectation",
            ),
            microscope,
        )
        means.setflags(write=False)
        return means

    return create


@pytest.fixture
def illumination_calibration_ensemble(roi_camera_scene, uniform_optical_calibration):
    """Provide identifiable optical calibration before the unchanged point sample."""
    from tests.simulation.camera import digitize

    scene = roi_camera_scene(shape=(24, 33, 33))
    microscope = scene.microscope
    # Constant emission over the full optical support makes illumination
    # identifiable independently of sparse sample structure at timepoint1.
    calibration_mean = uniform_optical_calibration(microscope, scene.pixels.shape[-3:])
    calibration_raw = digitize(calibration_mean, microscope.camera, seed=71)
    assert calibration_raw.max() < np.iinfo(np.uint16).max
    return SimpleNamespace(
        scene=scene,
        raw=np.concatenate((calibration_raw[None, None, None], scene.pixels)),
        calibration_photons=2000 * microscope.quantum_efficiency,
        sample_timepoint=1,
    )


@pytest.fixture
def illumination_observations(
    request, microscope_parameters, uniform_optical_calibration
):
    """Observe a uniform optical calibration object at prescribed exposures."""
    from tests.simulation.camera import digitize
    from tests.simulation.camera_likelihood import (
        SamplingEnvelope,
        camera_sampling_envelope,
    )

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
    microscope = microscope_parameters
    camera = microscope.camera
    optical_mean = uniform_optical_calibration(
        microscope,
        (12, height, width),
        photon_budget=(5000 / microscope.quantum_efficiency, 0),
    )
    source = float(optical_mean.mean())
    np.testing.assert_allclose(optical_mean, source, rtol=2e-5)
    levels = np.linspace(0.75, 1.25, 12)[:, None, None]
    means = optical_mean[None, None] * levels * fields[:, :, None]
    raw = digitize(
        means,
        camera,
        seed=71,
    )[None]
    assert raw.max() < 2**camera.bit_depth - 1
    coordinates = np.zeros((positions, 3))
    coordinates[:, 1] = np.arange(positions) * 100 if tiled else 20
    coordinates[:, 0] = np.arange(positions) * 10
    mask = np.asarray([[[True, False], [False, False]]]) if kind == "masked" else None
    expected = np.ones((channels, height, width), np.float32)
    # Specify the camera accuracy target before fitting. Bilinear reduction
    # to half size and expansion each span one working-grid pitch in native
    # coordinates. All independent exposures contribute original shot/read/ADC
    # residuals after normalization by their exact declared dose levels.
    radii = (
        optical_mean.shape[0] - 1,
        int(np.ceil(2 * (height - 1) / (height // 2 - 1))),
        int(np.ceil(2 * (width - 1) / (width // 2 - 1))),
    )
    envelopes = []
    for channel in range(channels):
        eligible = np.ones(positions, bool) if mask is None else mask[0, :, channel]
        if eligible.any():
            expected[channel] = fields[eligible, channel].max(axis=0)
            bounds = [
                camera_sampling_envelope(
                    means[position, channel],
                    raw[0, position, channel],
                    camera,
                    radii=radii,
                    response=levels,
                    views=int(eligible.sum()),
                )
                for position in np.flatnonzero(eligible)
            ]
            envelopes.append(
                SamplingEnvelope(
                    np.maximum.reduce(
                        [(bound.lower / levels).min(axis=0) for bound in bounds]
                    ),
                    np.maximum.reduce(
                        [(bound.upper / levels).max(axis=0) for bound in bounds]
                    ),
                )
            )
        else:
            envelopes.append(None)
    return SimpleNamespace(
        datastore=ts.array(raw),
        offset=camera.offset_adu,
        conversion=camera.electrons_per_adu,
        coordinates=coordinates,
        mask=mask,
        expected=expected,
        source=source,
        detector_x=detector_x,
        envelopes=envelopes,
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
def constant_illumination_acquisition(
    opm_v2_acquisition_factory, uniform_camera_scene, scan_planes
):
    """Persist independent optical exposures through the real acquisition writer."""
    fields = np.broadcast_to(
        np.array([2, 1.25], np.float32)[:, None, None], (2, 24, 256)
    ).copy()
    scene = uniform_camera_scene(
        times=2,
        positions=2,
        channels=2,
        projection=scan_planes == 1,
        fields=fields,
    )
    dataset = opm_v2_acquisition_factory(
        scene.pixels,
        mode="projection" if scan_planes == 1 else "mirror",
        stage_positions_zxy=((10, 20, 30), (0, 20, 30)),
        camera_offset=scene.microscope.camera.offset_adu,
        camera_conversion=scene.microscope.camera.electrons_per_adu,
    )
    return SimpleNamespace(dataset=dataset, fields=fields[None], scene=scene)
