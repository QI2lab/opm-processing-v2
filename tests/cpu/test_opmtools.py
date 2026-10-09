"""Test production deskew coordinates, photon scaling, and Z averaging."""

import numpy as np
import pytest

from opm_processing.imageprocessing import opmtools


@pytest.mark.unit
@pytest.mark.parametrize("scan_step", [0.23, 0.4, 0.8])
def test_deskew_preserves_fractional_photon_density(scan_step) -> None:
    """An interior constant field scales with the raw-to-output voxel volume."""
    calibrated = np.full((4, 8, 5), 0.24, dtype=np.float32)

    output = opmtools.orthogonal_deskew(
        calibrated,
        distance=scan_step,
        pixel_size=0.115,
        downsample_factor=1,
    )

    assert output.dtype == np.float32
    # At 30 degrees, V_raw = scan_step * pixel_size**2 * sin(30),
    # while V_output = pixel_size**3. No integer cast or arbitrary gain.
    expected = 0.24 * 0.115 / (scan_step * 0.5)
    np.testing.assert_allclose(output[1, 7, :5], expected, rtol=1e-6)
    np.testing.assert_array_equal(output[..., 5:], 0)


@pytest.mark.unit
@pytest.mark.parametrize("factor", [1, 2, 3, 20])
def test_deskew_x_ramp_and_z_averaging(factor):
    """A uniform YZ field retains detector X and the existing averaging gain.

    Parameters
    ----------
    factor : int
        Laboratory Z bin size, including a bin larger than the volume height.
    """
    ramp = np.arange(38, dtype=np.float32)[::2] + 10
    data = np.broadcast_to(ramp, (40, 33, 19))
    actual = opmtools.orthogonal_deskew(data, downsample_factor=factor)
    fine = opmtools.orthogonal_deskew(data, downsample_factor=1)
    nz = fine.shape[0]
    expected = np.zeros_like(actual)
    for z in range(actual.shape[0]):
        expected[z] = fine[z * factor : min((z + 1) * factor, nz)].sum(axis=0) / factor
    np.testing.assert_allclose(actual, expected, rtol=3e-7, atol=1e-6)
    np.testing.assert_allclose(
        fine[6, 45, :19], ramp * np.float32(2 * 0.115 / 0.4), rtol=3e-7
    )
    np.testing.assert_array_equal(actual[..., 19:], 0)
    np.testing.assert_array_equal(actual[:, 0, :], 0)
    np.testing.assert_array_equal(
        opmtools.orthogonal_deskew(
            data, downsample_factor=factor, reverse_deskewed_z=True
        ),
        actual[::-1],
    )
