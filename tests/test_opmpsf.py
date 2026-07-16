import numpy as np

from opm_processing.imageprocessing.opmpsf import _interpolate_psf_plane


def test_interpolate_psf_plane_preserves_interp2d_axis_order():
    x_grid = np.array([0.0, 1.0, 2.0])
    y_grid = np.array([0.0, 2.0])
    values = y_grid[:, None] + 2.0 * x_grid[None, :]

    result = _interpolate_psf_plane(
        x_grid,
        y_grid,
        values,
        x_coords=np.array([0.5, 1.5]),
        y_coords=np.array([0.5, 1.5]),
    )

    np.testing.assert_allclose(
        result,
        np.array([[1.5, 3.5], [2.5, 4.5]]),
    )


def test_interpolate_psf_plane_fills_out_of_bounds_with_zero():
    result = _interpolate_psf_plane(
        x_grid=np.array([0.0, 1.0]),
        y_grid=np.array([0.0, 1.0]),
        values=np.ones((2, 2)),
        x_coords=np.array([-1.0, 0.5]),
        y_coords=np.array([0.5]),
    )

    np.testing.assert_allclose(result, np.array([[0.0, 1.0]]))
