"""Generated optical kernels must retain their declared physical coordinates."""

from dataclasses import replace

import numpy as np
import pytest

from opm_processing.imageprocessing.opmpsf import (
    create_psf_silicone_100x,
    generate_proj_psf,
    generate_skewed_psf,
    get_skewed_roi_size,
)

pytestmark = pytest.mark.unit


def test_oblique_psf_refinement_against_independent_acquisition(
    production_point_acquisition,
):
    """Pixel integration converges towards a separately refined physical acquisition.

    Compare conditional photon probabilities, keeping the known total detected
    dose fixed. Expected log-likelihood loss measures optical error in units of
    distinguishability under shot noise; it is not a reconstruction error fit.
    """
    from tests.simulation.metrics import poisson_log_ratio_moments

    case = production_point_acquisition
    coarse, expectation = case.expectations
    total = expectation.sum()
    refinement = poisson_log_ratio_moments(expectation, coarse / coarse.sum() * total)
    metrics = {
        name: poisson_log_ratio_moments(expectation, kernel * total)
        for name, kernel in case.kernels.items()
    }
    print({"scan_step_um": case.microscope.scan_step_um, "loss_variance": metrics})
    # At this declared exposure, reference refinement and the finest tested
    # production quadrature are each smaller than one shot-noise standard deviation.
    assert refinement[0] < np.sqrt(refinement[1]), refinement
    assert metrics[3][0] < np.sqrt(metrics[3][1]), metrics
    assert metrics[3][0] < metrics[2][0] < metrics[1][0], metrics


@pytest.mark.parametrize("oversampling", [1, 2, 3])
@pytest.mark.parametrize("step", [0.4, 1.2])
def test_oblique_psf_matches_physical_pixel_integration(
    microscope_parameters,
    production_psf_parameters,
    production_psf_grid,
    oversampling,
    step,
):
    """Integrate the same optical field using the shared physical camera model.

    The reference maps finite pixels through X=c+u, Y=s+(r+v)cos(theta),
    Z=(r+v)sin(theta). This checks all axial positions and both camera dimensions
    without replacing the optical model or constructing another forward model.
    """
    from tests.simulation.forward_model import sensor_integral

    microscope = replace(
        microscope_parameters, scan_step_um=step, pixel_samples=oversampling
    )
    parameters = dict(production_psf_parameters, grid_oversampling=oversampling)
    kernel = generate_skewed_psf(
        microscope.wavelength_um,
        microscope.pixel_size_um,
        step,
        microscope.angle_deg,
        parameters=parameters,
    )
    field, (dxy, dz, nxy, nz) = production_psf_grid[0]
    axes = tuple(
        (np.arange(n) - (n - 1) / 2) * spacing
        for n, spacing in ((nz, dz), (nxy, dxy), (nxy, dxy))
    )
    expected = sensor_integral(field, axes, kernel.shape, microscope)
    np.testing.assert_allclose(
        kernel,
        expected,
        rtol=2e-11,
        atol=1e-15,
    )


@pytest.mark.parametrize("geometry", ["cartesian", "projection"])
def test_lateral_taper_preserves_optical_reflection_symmetry(
    microscope_parameters, production_psf_parameters, geometry
):
    """A centered isotropic pupil and radial taper have no preferred lateral side."""
    microscope = microscope_parameters
    if geometry == "cartesian":
        kernel = create_psf_silicone_100x(
            microscope.pixel_size_um,
            microscope.fine_spacing_um,
            15,
            3,
            microscope.wavelength_um,
            0,
            parameters=production_psf_parameters,
        )
    else:
        kernel = generate_proj_psf(
            microscope.wavelength_um,
            microscope.pixel_size_um,
            parameters=production_psf_parameters,
        )
    np.testing.assert_allclose(kernel, kernel[..., ::-1], rtol=2e-12, atol=1e-15)
    np.testing.assert_allclose(kernel, kernel[..., ::-1, :], rtol=2e-12, atol=1e-15)


@pytest.mark.parametrize("step", [0.115, 0.4, 1.2])
def test_skewed_support_rounds_a_dimensionless_scan_count(microscope_parameters, step):
    """The minimum scan count is a ceiling of physical extent divided by spacing."""
    theta = np.deg2rad(microscope_parameters.angle_deg)
    pitch = microscope_parameters.pixel_size_um
    extent = (0.8, 0.7, 0.6)
    scans, rows, columns = get_skewed_roi_size(extent, theta, pitch, step, False)
    scan_extent = 0.5 * (rows + 1) * pitch * np.cos(theta) + extent[1]
    assert (scans - 1) * step < scan_extent <= scans * step
    assert rows * pitch * np.sin(theta) >= extent[0]
    assert columns * pitch >= extent[2]


@pytest.mark.parametrize("missing", ["NA", "ng", "ni0"])
def test_missing_lens_calibration_cannot_select_package_defaults(
    production_psf_parameters, microscope_parameters, missing
):
    """Theoretical optics require the acquisition's lens, sample and glass data."""
    parameters = dict(
        production_psf_parameters, lens=dict(production_psf_parameters["lens"])
    )
    parameters["lens"].pop(missing)
    microscope = microscope_parameters
    with pytest.raises(ValueError, match=missing):
        generate_skewed_psf(
            microscope.wavelength_um,
            microscope.pixel_size_um,
            microscope.scan_step_um,
            microscope.angle_deg,
            parameters=parameters,
        )
