"""Physical specimens and independent optical image formation for reconstruction."""

from functools import lru_cache

import numpy as np
import pytest
from scipy import signal
from scipy.signal import fftconvolve
from scripts.opm_simulation import (
    prepare_sphere,
)
from scripts.psf_sampling_experiment import (
    cartesian_psf,
)

from tests.reference.point_sources import (
    fluorescent_specimen,
    optical_field,
    skewed_coordinates,
)


@pytest.fixture(scope="session")
def planar_point_model():
    """Return two known emitters and the normalized central YX PSF used to blur them.

    Returns
    -------
    tuple
        Float32 YX truth, normalized YX PSF and asymmetric SYX PSF whose central
        plane defines native 2D reconstruction. Other planes must not contribute.
    """
    y, x = np.mgrid[-3:4, -3:4]
    central = np.exp(-(y**2 + x**2) / 8).astype(np.float32)
    central /= central.sum()
    psf = np.stack((np.ones_like(central), central * 7, np.ones_like(central) * 100))
    truth = np.zeros((41, 43), np.float32)
    truth[12, 13] = 2000
    truth[28, 30] = 3000
    return truth, central, psf


@pytest.fixture
def planar_camera_acquisition(tmp_path, acquisition_factory, planar_point_model):
    """Persist independently blurred emitters across two times, positions and channels.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated directory receiving the reconstruction PSF.
    acquisition_factory : callable
        Writer for calibrated upstream camera measurements and metadata.
    planar_point_model : tuple
        Known YX emitters, central blur kernel and asymmetric volume PSF.

    Returns
    -------
    types.SimpleNamespace
        Calibrated camera acquisition, unblurred TPCZYX object, photon observations
        and on-disk PSF. Poisson measurements use a fixed seed and unity conversion.
    """
    from types import SimpleNamespace

    from scipy import ndimage

    truth, central, psf = planar_point_model
    gains = (
        np.asarray((1.0, 1.2))[:, None, None]
        * np.asarray((1.0, 1.1))[None, :, None]
        * np.asarray((1.0, 0.8))[None, None, :]
    )
    object_pixels = gains[..., None, None, None] * truth[None, None, None, None]
    blurred = ndimage.convolve(
        object_pixels, central[None, None, None, None], mode="reflect"
    )
    photons = np.random.default_rng(31).poisson(blurred + 0.25).astype(np.float32)
    dataset = acquisition_factory(
        (photons + 100).astype(np.uint16),
        name="planar_points",
        mode="projection",
        camera_conversion=1.0,
    )
    psf_path = tmp_path / "planar_psf.npy"
    np.save(psf_path, psf)
    return SimpleNamespace(
        dataset=dataset, truth=object_pixels, photons=photons, psf_path=psf_path
    )


@pytest.fixture(scope="session")
def sparse_optical_volume():
    """Return a fixed bead object, 637 nm optical PSF and every second photon plane.

    Returns
    -------
    tuple
        Laboratory bead truth and centers, unit-sum optical PSF, independent
        noiseless measurements and seeded Poisson data acquired every 800 nm.
        Reconstruction spacing is 400 nm, with 115 nm camera pixels.
    """
    truth, beads = fluorescent_specimen((49, 65, 49), 0.4)
    psf = compact_optical_psf(0.637, 0.4)
    expectation = np.maximum(
        signal.fftconvolve(truth.astype(np.float64), psf, mode="same"), 0
    )
    photons = np.random.default_rng(637).poisson(expectation)[::2]
    return truth, beads, psf, expectation, photons


@pytest.fixture(scope="session")
def physical_volume():
    """Return a voxel-integrated specimen and independent detector-integrated PSF.

    Returns
    -------
    tuple
        Float32 specimen, bead XYZ centers in micrometers, unit-sum cropped
        reconstruction PSF, and noiseless measurements generated with the
        larger optical PSF. Cropping omits at most 0.15 percent of PSF energy.
    """
    psf = np.zeros((121, 121, 61), np.float64)
    for row_offset in (-1 / 3, 0, 1 / 3):
        for col_offset in (-1 / 3, 0, 1 / 3):
            x, y, z = skewed_coordinates(psf.shape, offset=(0, row_offset, col_offset))
            coordinates = np.stack(np.broadcast_arrays(z, y, x), axis=-1)
            psf += optical_field()(coordinates) / 9
    psf /= psf.sum()
    selection = []
    for axis, size in enumerate(psf.shape):
        marginal = psf.sum(axis=tuple(i for i in range(3) if i != axis))
        center = size // 2
        radius = next(
            r
            for r in range(center + 1)
            if marginal[center - r : center + r + 1].sum() >= 0.9995
        )
        selection.append(slice(center - radius, center + radius + 1))
    cropped = psf[tuple(selection)]
    assert cropped.sum() >= 0.9985
    truth, beads = fluorescent_specimen((73, 81, 49), 0.2)
    expectation = np.maximum(signal.fftconvolve(truth, psf, mode="same"), 0)
    return truth, beads, (cropped / cropped.sum()).astype(np.float32), expectation


@pytest.fixture(scope="session")
def optical_psf():
    """Generate the production vectorial PSF at the reconstruction spacing."""
    return compact_optical_psf(0.610, 0.2)


@lru_cache(maxsize=4)
def compact_optical_psf(wavelength, scan_step):
    """Evaluate the optical model, caching only identical physical parameters.

    Parameters
    ----------
    wavelength : float
        Emission wavelength in micrometers.
    scan_step : float
        Scan-plane spacing in micrometers.

    Returns
    -------
    numpy.ndarray
        Unit-sum float32 skewed PSF at 115 nm detector pitch and 30 degree tilt.
        Cropping retains at least 99.85 percent of the full model's energy.
    """
    from opm_processing.imageprocessing.opmpsf import generate_skewed_psf

    psf = generate_skewed_psf(
        em_wvl=wavelength,
        pixel_size_um=0.115,
        scan_axis_step_um=scan_step,
        theta_deg=30,
    )
    # Compact only negligible tails for GPU test runtime. Bound the total
    # omitted energy explicitly; retain the full skew orientation and center.
    psf /= psf.sum()
    slices = []
    for axis, size in enumerate(psf.shape):
        marginal = psf.sum(axis=tuple(i for i in range(3) if i != axis))
        center = size // 2
        radius = next(
            r
            for r in range(center + 1)
            if marginal[center - r : center + r + 1].sum() >= 0.9995
        )
        slices.append(slice(center - radius, center + radius + 1))
    cropped = psf[tuple(slices)]
    assert cropped.sum() >= 0.9985
    return (cropped / cropped.sum()).astype(np.float32)


@pytest.fixture(scope="session")
def specimen():
    """Integrate bead volumes and a tilted filament over skew-grid voxels."""
    return fluorescent_specimen((73, 65, 49), 0.2)


@pytest.fixture(scope="session")
def physical_sphere():
    """Use the requested 10 um object on a 50 nm Cartesian grid, all in memory."""
    return prepare_sphere()


@pytest.fixture(scope="session")
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
            -sum(
                (coord - origin) ** 2
                for coord, origin in zip((x, y, z), center, strict=False)
            )
            / (2 * sigma**2)
        ) / ((2 * np.pi) ** 1.5 * sigma**3)
        image = fftconvolve(density, psf, mode="same")
        result.append((axes, psf, image))
    return result
