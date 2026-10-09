"""Unit and integration tests of physical fluorescence and OPM image formation."""

import numpy as np
import pytest
from scripts.opm_simulation import (
    acquire_sphere,
    camera_noise,
    meridian_sphere,
    pixel_average,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "xyz, expected",
    [
        ((0, 0, 0), 0),  # Hollow interior.
        ((5, 0, 0), 1),  # Equator meridian centerline.
        ((0, 0, 5), 1),  # Pole where all meridians join.
        ((5.4, 0, 0), 1),  # Inside the 0.5 um tube radius.
        ((5.6, 0, 0), 0),  # Outside the outer radius.
        ((4.4, 0, 0), 0),  # Inside the hollow sphere, outside tube.
        ((5, 0.4, 0), 1),  # Tube extends out of its meridian plane.
        ((5, 0.6, 0), 0),
        ((5 * np.cos(np.pi / 12), 5 * np.sin(np.pi / 12), 0), 0),  # Equatorial gap.
    ],
)
def test_tubes_have_defined_diameter_and_empty_meridian_gaps(xyz, expected):
    """Known physical points distinguish tubes from a fluorescent spherical shell."""
    axes = tuple(np.array([v]) for v in xyz[::-1])
    actual = meridian_sphere(axes, diameter_um=10, tube_diameter_um=1, meridians=12)
    assert actual.item() == expected


@pytest.mark.unit
def test_camera_pixel_averaging_has_known_area_response():
    """A quadratic irradiance field has an exact subpixel-quadrature average."""
    axes = tuple(np.arange(-10, 11) * 0.01 for _ in range(3))
    z, y, x = np.meshgrid(*axes, indexing="ij", sparse=True)
    field = (10 + 0 * z + y**2 + 2 * x**2).astype(np.float32)
    target = tuple(np.array([0.0]) for _ in range(3))
    point = pixel_average(field, axes, target, pitch_um=0.12, samples=1).item()
    integrated = pixel_average(field, axes, target, pitch_um=0.12, samples=3).item()
    expected = 10 + 3 * 0.12**2 * (1 - 1 / 3**2) / 12
    assert point == pytest.approx(10)
    assert integrated == pytest.approx(expected, abs=2e-6)


@pytest.mark.unit
def test_camera_shot_and_read_noise_have_physical_moments():
    """Poisson variance is the expected count; read-noise variances add."""
    expectation = np.full(200_000, 102.0)  # 100 signal + 2 background electrons.
    observed = camera_noise(expectation, read_noise_e=1.5, seed=637)
    assert observed.mean() == pytest.approx(102, abs=0.1)
    assert observed.var() == pytest.approx(102 + 1.5**2, rel=0.015)


@pytest.mark.unit
@pytest.mark.parametrize("scan_step, max_error", [(0.2, 0.05), (0.8, 0.12)])
def test_skewed_sphere_deskews_to_cartesian_microscope(
    physical_sphere, scan_step, max_error
):
    """Compare independent convolutions and absolute physical coordinates."""
    acquired = acquire_sphere(physical_sphere, scan_step_um=scan_step)
    metrics = acquired["metrics"]
    assert metrics["forward_routes"]["relative_l2"] < 0.03
    assert metrics["forward_routes"]["correlation"] > 0.999
    assert metrics["deskew_vs_normal"]["relative_l2"] < max_error
    assert metrics["deskew_vs_normal"]["correlation"] > 0.99
    assert metrics["deskew_vs_normal"]["total_signal_ratio"] == pytest.approx(
        1, abs=0.01
    )
    axes = acquired["output_axes"]
    center = tuple(np.argmin(np.abs(a)) for a in axes)
    assert acquired["deskewed"][center] < 0.01 * acquired["deskewed"].max()
    # No image alignment is allowed. The absolute physical centroid must stay
    # at the center of the synthetic sphere, within one native camera pixel.
    mass = acquired["deskewed"]
    centroid = [
        np.dot(mass.sum(axis=tuple(j for j in range(3) if j != i)), a) / mass.sum()
        for i, a in enumerate(axes)
    ]
    assert np.linalg.norm(centroid) < physical_sphere["params"]["pixel_size_um"]
