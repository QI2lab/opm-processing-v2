"""Camera statistics and independent optical source-to-sensor image formation."""

from dataclasses import replace

import numpy as np
import pytest

from tests.simulation.camera import digitize
from tests.simulation.forward_model import (
    Acquisition,
    camera_psf,
    excitation_field,
    forward_model,
    sensor_integral,
)
from tests.simulation.scenes import acquisition_shape

pytestmark = pytest.mark.unit


def test_line_pairs_have_declared_spacing_and_equal_emitted_photons(
    independent_photon_scene_factory,
):
    """Preserve declared line-pair gaps and equal emitted photon totals."""
    scene = independent_photon_scene_factory("line-pairs")
    positions = np.asarray(scene.features["positions_x_um"]).reshape(-1, 2)
    np.testing.assert_allclose(
        positions[:, 1] - positions[:, 0],
        [0.15, 0.23, 0.35, 0.53, 0.8, 1.2],
        rtol=0,
        atol=2e-15,
    )
    np.testing.assert_allclose(positions[1:, 0] - positions[:-1, 1], 1.8, atol=2e-15)
    np.testing.assert_allclose(
        positions.mean(axis=1), scene.features["pair_centers_x_um"]
    )
    boundaries = (positions[:-1, 1] + positions[1:, 0]) / 2
    profile = scene.density.sum(axis=(0, 1), dtype=np.float64)
    pair_mass = [
        part.sum() * scene.microscope.fine_spacing_um**3
        for part in np.split(profile, np.searchsorted(scene.axes[2], boundaries))
    ]
    np.testing.assert_allclose(pair_mass, 2 * scene.features["length_um"], rtol=1e-6)


@pytest.mark.parametrize(
    "mean,noise,full_model", [(0, 0.7, False), (2, 1.6, False), (30, 1.6, True)]
)
def test_camera_adc_moments(
    independent_photon_scene_factory, mean, noise, full_model, camera_calibration
):
    """Detection/read variances add before gain; unsigned ADC retains its offset."""
    camera = replace(camera_calibration, read_noise_e=noise)
    if full_model:
        scene = independent_photon_scene_factory()
        # A closed exposure has only the specified photon background.
        raw = forward_model(
            scene.density,
            Acquisition(
                scene.axes,
                (1, 400, 1000),
                exposure=0,
                background_electrons=mean,
                seed=637,
            ),
            replace(scene.microscope, camera=camera),
        )
    else:
        raw = digitize(
            np.full(400_000, float(mean)), camera, seed=637 if mean == 0 else 31
        )
    variance = (mean + noise**2) / camera.electrons_per_adu**2 + 1 / 12
    tolerance = (
        (0.02 if mean == 0 else 0.05)
        if not full_model
        else 5 * np.sqrt(variance / raw.size)
    )
    assert raw.dtype == np.uint16
    assert raw.mean() == pytest.approx(
        camera.offset_adu + mean / camera.electrons_per_adu, abs=tolerance
    )
    assert raw.var() == pytest.approx(variance, rel=0.02)
    if mean == 0:
        assert 0.42 < np.mean(raw < 100) < 0.46
        assert np.mean(raw == 0) == 0


@pytest.mark.parametrize("profile", ("uniform", "edge-taper"))
def test_512_by_1900_excitation_varies_only_along_camera_x(
    profile, microscope_parameters
):
    """The taper is positive at both 1900-axis ends and identical on all 512 rows."""
    x = (np.arange(1900) - 949.5) * 0.115
    field = np.broadcast_to(
        excitation_field(
            x,
            x[-1],
            profile,
            edge_fraction=microscope_parameters.illumination_edge_fraction,
        ),
        (512, 1900),
    )
    np.testing.assert_array_equal(field[0], field[-1])
    np.testing.assert_array_equal(field[:, 0], field[:, -1])
    if profile == "edge-taper":
        assert field[0, 0] == pytest.approx(0.25)
        assert field[0, 949] == pytest.approx(1, abs=1e-6)
        assert np.all(np.diff(field[0, :950]) > 0)
    else:
        np.testing.assert_array_equal(field, 1)


@pytest.mark.parametrize("geometry", ("oblique", "cartesian"))
def test_tilted_sensor_integrates_area_instead_of_point_sampling(
    geometry, microscope_parameters
):
    """A quadratic irradiance has an independently known finite-pixel response."""
    p = replace(microscope_parameters, pixel_samples=3)
    axis = np.arange(-20, 21) * 0.005
    z, y, x = np.meshgrid(axis, axis, axis, indexing="ij", sparse=True)
    field = (10 + 2 * x**2 + y**2 + z**2).astype(np.float32)
    actual = sensor_integral(
        field, (axis, axis, axis), (1, 1, 1), p, geometry=geometry
    ).item()
    # Camera row offsets rotate in YZ, preserving y²+z². Midpoint quadrature
    # variance per pixel dimension is pitch²*(1 - 1/n²)/12.
    expected_average = 10 + 3 * p.pixel_size_um**2 * (1 - 1 / 3**2) / 12
    assert actual == pytest.approx(expected_average * p.pixel_size_um**2, abs=2e-7)


@pytest.mark.parametrize("profile", ("uniform", "edge-taper"))
def test_point_illumination_precedes_emission_blur(
    independent_photon_scene_factory, profile
):
    """Known emitter brightness scales its entire PSF, rather than detected pixels."""
    scene = independent_photon_scene_factory("points", profile)
    p = scene.microscope
    shape = acquisition_shape(scene)
    actual = (
        forward_model(
            scene.density,
            Acquisition(
                scene.axes, shape, illumination_profile=profile, output="expectation"
            ),
            p,
        )
        / p.quantum_efficiency
    )
    expected = np.zeros(shape, np.float32)
    # The convolved scene continues into zero-valued voxels beyond the finite
    # PSF support. Include that zero halo in the direct point-source route so
    # trilinear interpolation has the same physical boundary on both routes.
    optical_field = np.pad(scene.psf_irradiance, 1)
    axes_with_halo = tuple(
        np.r_[a[0] - p.fine_spacing_um, a, a[-1] + p.fine_spacing_um]
        for a in scene.psf_axes
    )
    for center in scene.features["centers_xyz_um"]:
        shifted_axes = tuple(
            a + c for a, c in zip(axes_with_halo, center[::-1], strict=True)
        )
        expected += sensor_integral(
            optical_field, shifted_axes, shape, p
        ) * excitation_field(
            np.asarray(center[0]),
            scene.illumination_half_width_um,
            profile,
            edge_fraction=p.illumination_edge_fraction,
        )
    np.testing.assert_allclose(actual, expected, rtol=5e-4, atol=1e-8)
    # Zero padding of GT is a representation choice; it cannot change the
    # emitted light collected beyond the stored object's boundaries.
    crop = tuple(
        slice(int(indices.min()) - 1, int(indices.max()) + 2)
        for indices in np.nonzero(scene.density)
    )
    cropped_axes = tuple(a[s] for a, s in zip(scene.axes, crop, strict=True))
    cropped = (
        forward_model(
            scene.density[crop],
            Acquisition(
                cropped_axes, shape, illumination_profile=profile, output="expectation"
            ),
            p,
        )
        / p.quantum_efficiency
    )
    np.testing.assert_allclose(cropped, actual, rtol=5e-4, atol=1e-8)
    # A stage translation changes global coordinates, not the local image's
    # excitation profile or collected photons.
    shift_xyz = np.array((1.15, 2.0, 10.0))
    shifted_axes = tuple(
        a + shift for a, shift in zip(scene.axes, shift_xyz[::-1], strict=True)
    )
    angle = np.deg2rad(p.angle_deg)
    shift_syx = (
        shift_xyz[1] - shift_xyz[2] / np.tan(angle),
        shift_xyz[2] / np.sin(angle),
        shift_xyz[0],
    )
    sensor_axes = tuple(
        (np.arange(n) - (n - 1) / 2) * step + shift
        for n, step, shift in zip(
            shape,
            (p.scan_step_um, p.pixel_size_um, p.pixel_size_um),
            shift_syx,
            strict=True,
        )
    )
    translated = (
        forward_model(
            scene.density,
            Acquisition(
                shifted_axes,
                sensor_axes=sensor_axes,
                illumination_profile=profile,
                output="expectation",
            ),
            p,
        )
        / p.quantum_efficiency
    )
    np.testing.assert_allclose(translated, actual, rtol=1e-5, atol=1e-9)
    # Each fine-grid point has one emitted photon per exposure, independent of
    # grid spacing. No fitted gain or reconstruction participates in this check.
    assert scene.density.sum(dtype=np.float64) * p.fine_spacing_um**3 == pytest.approx(
        2, rel=1e-6
    )


@pytest.mark.parametrize("step", (0.4, 1.2))
def test_inverse_psf_matches_full_camera_response(
    independent_photon_scene_factory, step
):
    """A common lattice emitter must have the same response in both operators."""
    scene = independent_photon_scene_factory()
    p = replace(scene.microscope, scan_step_um=step)
    shape = acquisition_shape(replace(scene, microscope=p))
    density = np.zeros_like(scene.density)
    density[tuple(n // 2 for n in density.shape)] = 1 / p.fine_spacing_um**3
    actual = forward_model(
        density, Acquisition(scene.axes, shape, output="expectation"), p
    )
    kernel, scale = camera_psf(p)
    # Kernel normalization has a physical Jacobian, not a fitted image gain.
    probability = scale / (step * p.pixel_size_um**2 * np.sin(np.deg2rad(p.angle_deg)))
    expected = np.zeros(shape)
    support = tuple(
        slice(n // 2 - k // 2, n // 2 + k // 2 + 1)
        for n, k in zip(shape, kernel.shape, strict=True)
    )
    expected[support] = kernel * probability
    # The full route uses float32 FFT convolution; allow its arithmetic error.
    np.testing.assert_allclose(
        actual, expected, rtol=0, atol=64 * np.finfo(np.float32).eps * actual.max()
    )


@pytest.mark.parametrize("factor", (3, 4))
def test_subsampling_selects_the_same_physical_exposures(
    independent_photon_scene_factory, factor
):
    """Coarse scans retain dense positions, sensor pixels, illumination and exposure."""
    scene = independent_photon_scene_factory()
    p = scene.microscope
    shape = acquisition_shape(scene)
    dense = forward_model(
        scene.density, Acquisition(scene.axes, shape, output="expectation"), p
    )
    coarse = forward_model(
        scene.density,
        Acquisition(
            scene.axes, ((shape[0] - 1) // factor + 1, *shape[1:]), output="expectation"
        ),
        replace(p, scan_step_um=p.scan_step_um * factor),
    )
    np.testing.assert_allclose(coarse, dense[::factor], rtol=1e-6, atol=1e-10)


def test_pixel_quadrature_convergence(independent_photon_scene_factory):
    """Refine tilted sensor integration while holding the physical scene fixed."""
    fine = independent_photon_scene_factory(spacing=0.025)
    metadata = Acquisition(fine.axes, acquisition_shape(fine), output="expectation")
    images = [
        forward_model(fine.density, metadata, replace(fine.microscope, pixel_samples=n))
        for n in (3, 5)
    ]
    assert np.linalg.norm(images[0] - images[1]) / np.linalg.norm(images[1]) < 0.005


@pytest.mark.parametrize("angle", [30, 38])
@pytest.mark.parametrize("step", [0.2, 0.4, 0.8])
def test_oblique_pixel_coordinates_match_linear_field(
    linear_camera_field, angle, step, microscope_parameters
):
    """A known concentration ramp gives exact values at tilted pixel centers."""
    case = linear_camera_field
    p = replace(
        microscope_parameters,
        pixel_size_um=0.13,
        angle_deg=angle,
        scan_step_um=step,
        pixel_samples=1,
    )
    theta = np.deg2rad(angle)
    shifts = (
        case.center[1] - case.center[2] / np.tan(theta),
        case.center[2] / np.sin(theta),
        case.center[0],
    )
    sensor_axes = tuple(
        (np.arange(n) - (n - 1) / 2) * d + offset
        for n, d, offset in zip(case.shape, (step, 0.13, 0.13), shifts, strict=True)
    )
    actual = sensor_integral(
        case.field,
        case.axes,
        case.shape,
        p,
        sensor_axes=sensor_axes,
        area_integral=False,
    )
    expected = np.empty(case.shape)
    for s, r, c in np.ndindex(case.shape):
        row = (r - 5.5) * 0.13
        xyz = (
            (c - 4) * 0.13 + case.center[0],
            (s - 3) * step + row * np.cos(theta) + case.center[1],
            row * np.sin(theta) + case.center[2],
        )
        expected[s, r, c] = 100 + np.dot((2, 3, 5), xyz)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("step", [0.2, 0.4, 0.8])
def test_optical_grid_convergence(fine_optical_images, step, microscope_parameters):
    """Refine PSF and finite-object grids at the same physical scan positions."""
    p = replace(microscope_parameters, scan_step_um=step, pixel_samples=1)
    sampled_psfs, sampled_images = [], []
    for axes, psf, image in fine_optical_images:
        sampled = sensor_integral(psf, axes, (25, 33, 17), p, area_integral=False)
        sampled_psfs.append(sampled / sampled.sum())
        sampled_images.append(
            sensor_integral(image, axes, (25, 33, 17), p, area_integral=False)
        )
    for samples in (sampled_psfs, sampled_images):
        reference = samples[-1]
        errors = [
            np.linalg.norm(a - reference) / np.linalg.norm(reference)
            for a in samples[:-1]
        ]
        assert errors[1] < errors[0]
        assert errors[1] < 0.02
    if step == 0.8:
        dense = sensor_integral(
            image, axes, (33, 33, 17), replace(p, scan_step_um=0.2), area_integral=False
        )
        coarse = sensor_integral(image, axes, (9, 33, 17), p, area_integral=False)
        np.testing.assert_allclose(coarse, dense[::4], rtol=1e-12, atol=1e-12)
