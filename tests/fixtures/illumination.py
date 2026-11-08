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
            coverage = np.ones(native)
        else:
            expected_shape, *_ = deskew_shape_estimator(
                native,
                microscope.angle_deg,
                microscope.scan_step_um,
                pixel,
                crop_after_deskew=False,
            )
            assert data.shape[-3:] == tuple(expected_shape)
            coverage = np.zeros(data.shape[-3:])
            coverage[..., : native[2]] = deskew_coverage(
                native,
                data.shape[-3:],
                microscope.angle_deg,
                microscope.scan_step_um,
                pixel,
            )[..., None]
        # Gain is the independent 30-degree photon-volume normalization;
        # coverage only describes the acquisition's geometric support.
        support = gain * coverage
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
                expected = float(truth.flat[0]) * support
                operations = 64
                error = operations * np.finfo(np.float32).eps
                roundoff = error / (1 - error) * max(float(observed.max()), 1) * gain
                quantum = 0 if data.dtype.kind == "f" else 1
                low = (float(truth.min()) + residual.min()) * support
                high = (float(truth.max()) + residual.max()) * support
                envelope = SamplingEnvelope(
                    np.maximum(low - roundoff - quantum, 0),
                    high + roundoff + quantum,
                )
                actual = data[time, channel]
                result = camera_sampling_accuracy(
                    actual, expected, envelope, support > 0
                )
                assert result.accepted, result
                np.testing.assert_array_equal(actual[support == 0], 0)

    return check


@pytest.fixture(scope="session")
def uniform_optical_calibration():
    """Share immutable uniform-phantom means with excitation applied before blur."""
    from dataclasses import replace
    from functools import lru_cache

    from opm_processing.imageprocessing.opmpsf import centered_sensor_axes
    from tests.simulation.forward_model import Acquisition, forward_model
    from tests.simulation.objects import centered_axis

    @lru_cache(maxsize=12)
    def create(
        microscope,
        shape,
        profile="uniform",
        photon_budget=(2000, 0),
        *,
        geometry="oblique",
    ):
        """Cover the entire sensor footprint and emission-PSF support before sampling."""
        # Uniform calibration has no sub-pixel structure; keep its established
        # 0.1 um optical grid instead of allocating a finer full-detector volume.
        microscope = replace(microscope, fine_spacing_um=0.1)
        pixel = microscope.pixel_size_um
        angle = np.deg2rad(microscope.angle_deg)
        scan, row, column = centered_sensor_axes(
            shape, (microscope.scan_step_um, pixel, pixel)
        )
        if geometry == "oblique":
            camera_half_extent = (
                abs(row[0] * np.sin(angle)),
                abs(scan[0]) + abs(row[0] * np.cos(angle)),
                abs(column[0]),
            )
        else:
            assert geometry == "cartesian"
            camera_half_extent = tuple(abs(axis[0]) for axis in (scan, row, column))
        axes = tuple(
            centered_axis(extent + halo + pixel, microscope.fine_spacing_um)
            for extent, halo in zip(
                camera_half_extent, microscope.psf_half_extent_zyx_um, strict=True
            )
        )
        means = forward_model(
            np.broadcast_to(np.float32(1), tuple(map(len, axes))),
            Acquisition(
                axes,
                shape,
                geometry=geometry,
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

    scene = roi_camera_scene()
    microscope = scene.microscope
    # Constant emission over the full optical support makes illumination
    # identifiable independently of sparse sample structure at timepoint1.
    calibration_mean = uniform_optical_calibration(microscope, scene.pixels.shape[-3:])
    calibration_raw = digitize(calibration_mean, microscope.camera, seed=71)
    assert calibration_raw.max() < np.iinfo(np.uint16).max
    return SimpleNamespace(
        scene=scene,
        raw=np.concatenate((calibration_raw[None, None, None], scene.pixels)),
        sample_timepoint=1,
    )


@pytest.fixture
def illumination_observations(
    request,
    microscope_parameters,
    uniform_optical_calibration,
    simulation_parameters,
    tmp_path,
    opm_v2_acquisition_factory,
):
    """Observe a uniform optical calibration object at prescribed exposures."""
    from opm_processing.dataio.acquisition import open_acquisition_datastore
    from tests.simulation.camera import digitize
    from tests.simulation.camera_likelihood import (
        SamplingEnvelope,
        camera_sampling_envelope,
    )

    kind = request.param
    tiled = kind == "tiles"
    positions, channels = (6, 2) if tiled else (2, 2 if kind == "masked" else 1)
    height, width = simulation_parameters["camera_shape_yx"]
    scans = simulation_parameters["tiled_scan_planes" if tiled else "whole_scan_planes"]
    y = np.linspace(-1, 1, height, dtype=np.float32)[:, None]
    x = np.linspace(-1, 1, width, dtype=np.float32)[None, :]
    if tiled:
        fields = np.stack(
            (
                1 + 0.28 * y + 0.12 * x + 0.08 * y * x,
                1 - 0.22 * y + 0.16 * x - 0.06 * y * x,
            )
        )
        detector_x = round(width * 78 / 128)
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
        (scans, height, width),
        photon_budget=(5000 / microscope.quantum_efficiency, 0),
    )
    source = float(optical_mean.mean())
    np.testing.assert_allclose(optical_mean, source, rtol=2e-5)
    levels = np.linspace(0.75, 1.25, scans)[:, None, None]
    raw = np.lib.format.open_memmap(
        tmp_path / "illumination_camera.npy",
        mode="w+",
        dtype=np.uint16,
        shape=(1, positions, channels, scans, height, width),
    )
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
    eligible = np.ones((positions, channels), bool) if mask is None else mask[0]
    envelopes = [None] * channels
    for channel in range(channels):
        if eligible[:, channel].any():
            expected[channel] = fields[eligible[:, channel], channel].max(axis=0)
        for position in range(positions):
            # Form and digitize one real camera view before retaining its bytes
            # on disk. The full acquisition still reaches the opm-v2 writer.
            means = optical_mean * levels * fields[position, channel]
            pixels = digitize(means, camera, seed=71 + 10 * position + channel)
            assert pixels.max() < 2**camera.bit_depth - 1
            raw[0, position, channel] = pixels
            if eligible[position, channel]:
                bounds = camera_sampling_envelope(
                    means,
                    pixels,
                    camera,
                    radii=radii,
                    response=levels,
                    views=int(eligible[:, channel].sum()),
                )
                reduced = SamplingEnvelope(
                    (bounds.lower / levels).min(axis=0),
                    (bounds.upper / levels).max(axis=0),
                )
                previous = envelopes[channel]
                envelopes[channel] = (
                    reduced
                    if previous is None
                    else SamplingEnvelope(
                        np.maximum(previous.lower, reduced.lower),
                        np.maximum(previous.upper, reduced.upper),
                    )
                )
                del bounds
            del means, pixels
    raw.flush()
    dataset = opm_v2_acquisition_factory(
        raw,
        name="illumination_observations",
        camera_offset=camera.offset_adu,
        camera_conversion=camera.electrons_per_adu,
        camera_read_noise_e=camera.read_noise_e,
        pixel_size_um=microscope.pixel_size_um,
        scan_axis_step_um=microscope.scan_step_um,
        angle_deg=microscope.angle_deg,
        stage_positions_zxy=tuple(map(tuple, coordinates)),
    )
    return SimpleNamespace(
        datastore=open_acquisition_datastore(dataset.metadata),
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
    opm_v2_acquisition_factory, uniform_camera_scene, scan_planes, simulation_parameters
):
    """Persist independent optical exposures through the real acquisition writer."""
    fields = np.broadcast_to(
        np.array([2, 1.25], np.float32)[:, None, None],
        (2, *simulation_parameters["camera_shape_yx"]),
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
        camera_read_noise_e=scene.microscope.camera.read_noise_e,
    )
    return SimpleNamespace(dataset=dataset, fields=fields[None], scene=scene)
