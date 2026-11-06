"""Camera statistics and independent optical source-to-sensor image formation."""

from dataclasses import replace
from functools import partial
from itertools import pairwise

import numpy as np
import pytest

from opm_processing.imageprocessing.opmpsf import (
    centered_sensor_axes,
    integrate_camera_pixels,
)
from tests.simulation.forward_model import (
    Acquisition,
    camera_psf,
    excitation_field,
    forward_model,
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
def test_camera_adc_moments(camera_adc_case, mean, noise, full_model):
    """Detection/read variances add before gain; unsigned ADC retains its offset."""
    from scipy.stats import binom, norm

    raw, moments = camera_adc_case.raw, camera_adc_case.moments
    count = raw.size
    variance_upper = moments.variance + moments.variance_error
    variance_lower = max(0, moments.variance - moments.variance_error)
    mean_tolerance = moments.mean_error + 5 * np.sqrt(variance_upper / count)
    # Exact variance of the unbiased sample variance for independent observations.
    variance_tolerance = moments.variance_error + 5 * np.sqrt(
        (moments.fourth_moment_upper - (count - 3) / (count - 1) * variance_lower**2)
        / count
    )
    assert raw.dtype == np.uint16
    assert raw.mean() == pytest.approx(moments.mean, abs=mean_tolerance)
    assert raw.var(ddof=1) == pytest.approx(moments.variance, abs=variance_tolerance)
    if mean == 0:
        below = moments.probabilities[
            moments.codes < camera_adc_case.camera.offset_adu
        ].sum()
        binomial_tolerance = (
            moments.omitted_probability
            + moments.summation_roundoff * below
            + 5 * np.sqrt(below * (1 - below) / count)
        )
        assert np.mean(raw < camera_adc_case.camera.offset_adu) == pytest.approx(
            below, abs=binomial_tolerance
        )
        zero_probability_upper = (
            moments.probabilities[0] * (1 + moments.summation_roundoff)
            + moments.omitted_probability
        )
        assert np.count_nonzero(raw == 0) <= binom.ppf(
            norm.cdf(5), count, zero_probability_upper
        )


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
@pytest.mark.parametrize("pixel_samples", (1, 3, 5, 9))
def test_tilted_sensor_integrates_area_instead_of_point_sampling(
    geometry, pixel_samples, quadratic_camera_field, microscope_parameters
):
    """Known quadratic irradiance gives exact midpoint error proportional to n^-2."""
    p = replace(microscope_parameters, pixel_samples=pixel_samples)
    case = quadratic_camera_field
    actual = integrate_camera_pixels(
        case.field,
        case.axes,
        centered_sensor_axes(
            case.shape, (p.scan_step_um, p.pixel_size_um, p.pixel_size_um)
        ),
        pixel_size_um=p.pixel_size_um,
        theta_deg=p.angle_deg,
        samples=p.pixel_samples,
        geometry=geometry,
    ).item()
    # Rotation preserves y²+z². The exact pixel integral of 10+2x²+y²+z² is
    # 10*p²+p^4/4; n midpoint samples undercount it by exactly p^4/(4*n²).
    pitch = p.pixel_size_um
    integral = 10 * pitch**2 + pitch**4 / 4
    midpoint_error = pitch**4 / (4 * pixel_samples**2)
    # Linear interpolation of a*x² over spacing h errs by at most a*h²/4.
    # The three coefficients sum to four. Convex trilinear weights retain this
    # bound; 32 float64 operations per quadrature point cover accumulation.
    spacing = case.axes[0][1] - case.axes[0][0]
    interpolation_error = pitch**2 * spacing**2
    operations = 32 * pixel_samples**2
    gamma = operations * np.finfo(np.float64).eps
    gamma /= 1 - gamma
    roundoff = gamma * case.field.max() * pitch**2
    assert integral - actual == pytest.approx(
        midpoint_error, abs=interpolation_error + roundoff
    )


@pytest.mark.parametrize("profile", ("uniform", "edge-taper"))
def test_point_illumination_precedes_emission_blur(
    independent_photon_scene_factory, profile
):
    """Known emitter brightness scales its entire PSF, rather than detected pixels."""
    scene = independent_photon_scene_factory("points", profile)
    p = scene.microscope
    shape = acquisition_shape(scene)
    native_sensor_axes = centered_sensor_axes(
        shape, (p.scan_step_um, p.pixel_size_um, p.pixel_size_um)
    )
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
        expected += integrate_camera_pixels(
            optical_field,
            shifted_axes,
            native_sensor_axes,
            pixel_size_um=p.pixel_size_um,
            theta_deg=p.angle_deg,
            samples=p.pixel_samples,
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
        axis + shift for axis, shift in zip(native_sensor_axes, shift_syx, strict=True)
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
        axis + offset
        for axis, offset in zip(
            centered_sensor_axes(case.shape, (step, 0.13, 0.13)), shifts, strict=True
        )
    )
    actual = integrate_camera_pixels(
        case.field,
        case.axes,
        sensor_axes,
        pixel_size_um=p.pixel_size_um,
        theta_deg=p.angle_deg,
        samples=p.pixel_samples,
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
    sensor_axes = centered_sensor_axes(
        (25, 33, 17), (step, p.pixel_size_um, p.pixel_size_um)
    )
    sample = partial(
        integrate_camera_pixels,
        pixel_size_um=p.pixel_size_um,
        theta_deg=p.angle_deg,
        samples=p.pixel_samples,
        area_integral=False,
    )
    sampled_psfs, sampled_images = [], []
    for axes, psf, image in fine_optical_images:
        sampled = sample(psf, axes, sensor_axes)
        sampled_psfs.append(sampled / sampled.sum())
        sampled_images.append(sample(image, axes, sensor_axes))
    for samples in (sampled_psfs, sampled_images):
        reference = samples[-1]
        errors = [
            np.linalg.norm(a - reference) / np.linalg.norm(reference)
            for a in samples[:-1]
        ]
        assert errors[1] < errors[0]
        changes = [np.linalg.norm(a - b) for a, b in pairwise(samples)]
        # Finite, inclusive PSF boundaries can add first-order terms to the
        # interior second-order grid error. Verify at least first-order
        # contraction at every available adjacent refinement, rather than
        # assuming the asymptotic h² regime. Conditional on that regime
        # continuing, the medium-to-limit geometric tail is <=2*last change.
        assert changes[1] <= changes[0] / 2
        estimated_limit_error = 2 * changes[-1] / np.linalg.norm(reference)
        # Retain the established 2% accuracy target, applying it to the stronger
        # conditional continuum estimate instead of only the last difference.
        assert estimated_limit_error < 0.02
    if step == 0.8:
        dense = sample(
            image,
            axes,
            centered_sensor_axes((33, 33, 17), (0.2, p.pixel_size_um, p.pixel_size_um)),
        )
        coarse = sample(
            image,
            axes,
            centered_sensor_axes((9, 33, 17), (step, p.pixel_size_um, p.pixel_size_um)),
        )
        np.testing.assert_allclose(coarse, dense[::4], rtol=1e-12, atol=1e-12)
