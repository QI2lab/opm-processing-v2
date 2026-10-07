"""Physical sampling tests without changes to production deskew or PSFs."""

import numpy as np
import pytest
from scipy.signal import fftconvolve

from scripts.psf_sampling_experiment import (
    cartesian_psf,
    sample_skewed,
)


@pytest.mark.unit
@pytest.mark.parametrize("angle", [30, 38])
@pytest.mark.parametrize("step", [0.2, 0.4, 0.8])
def test_physical_concentration_sampled_at_camera_pixel_centers(angle, step):
    """An affine concentration field has exact values at every camera pixel."""
    axes = tuple(np.linspace(-5, 5, 101) for _ in range(3))
    z, y, x = np.meshgrid(*axes, indexing="ij", sparse=True)
    field = 100 + 2 * x + 3 * y + 5 * z
    center = (0.03, -0.12, 0.21)
    shape = (7, 12, 9)  # Include an even camera dimension and a shifted origin.
    actual = sample_skewed(
        field,
        axes,
        shape,
        angle_deg=angle,
        scan_step_um=step,
        pixel_size_um=0.13,
        center_xyz_um=center,
    )
    expected = np.empty(shape)
    for s, r, c in np.ndindex(shape):
        row_um = (r - 5.5) * 0.13
        xyz = (
            (c - 4) * 0.13 + center[0],
            (s - 3) * step + row_um * np.cos(np.deg2rad(angle)) + center[1],
            row_um * np.sin(np.deg2rad(angle)) + center[2],
        )
        expected[s, r, c] = 100 + np.dot((2, 3, 5), xyz)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.fixture(scope="module")
def fine_optical_images():
    """Blur a finite fluorescent object in physical space before sampling.

    A Gaussian fluorophore density describes a smooth, finite-size bead, not
    the optical PSF. The blur is the vectorial diffraction PSF. All three grids
    use the same physical object, wavelength, support and photon-density units.
    """
    result = []
    for spacing in (0.0575, 0.02875, 0.014375):
        # Shared physical boundaries must also coincide across refinements;
        # otherwise changes in zero-filled support masquerade as sampling error.
        axes, psf = cartesian_psf(spacing, (24 * 0.0575, 16 * 0.0575, 16 * 0.0575))
        z, y, x = np.meshgrid(*axes, indexing="ij", sparse=True)
        center = (0.07, 0.11, -0.04)
        sigma = 0.12
        density = np.exp(
            -sum((coord - origin) ** 2 for coord, origin in zip((x, y, z), center))
            / (2 * sigma**2)
        ) / ((2 * np.pi) ** 1.5 * sigma**3)
        image = fftconvolve(density, psf, mode="same")
        result.append((axes, psf, image))
    return result


@pytest.mark.unit
@pytest.mark.parametrize("step", [0.2, 0.4, 0.8])
def test_fine_cartesian_forward_model_converges_at_raw_sampling(
    fine_optical_images, step
):
    """Refine the optical/object grid while holding raw sampling fixed."""
    sampled_psfs = []
    sampled_images = []
    for axes, psf, image in fine_optical_images:
        sampled = sample_skewed(psf, axes, (25, 33, 17), scan_step_um=step)
        sampled_psfs.append(sampled / sampled.sum())
        sampled_images.append(
            sample_skewed(
                image,
                axes,
                (25, 33, 17),
                scan_step_um=step,
            )
        )
    for label, samples in (("psf", sampled_psfs), ("blurred_object", sampled_images)):
        reference = samples[-1]
        errors = [
            np.linalg.norm(a - reference) / np.linalg.norm(reference)
            for a in samples[:-1]
        ]
        print({"scan_step_um": step, "field": label, "relative_errors": errors})
        assert errors[1] < errors[0]
        assert errors[1] < 0.02


@pytest.mark.unit
def test_integer_scan_sampling_preserves_physical_plane_locations(fine_optical_images):
    """0.8 um measurements are every fourth plane of a 0.2 um scan."""
    axes, _, image = fine_optical_images[-1]
    dense = sample_skewed(image, axes, (33, 33, 17), scan_step_um=0.2)
    coarse = sample_skewed(image, axes, (9, 33, 17), scan_step_um=0.8)
    np.testing.assert_allclose(coarse, dense[::4], rtol=1e-12, atol=1e-12)
