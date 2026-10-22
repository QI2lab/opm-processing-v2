"""Unit tests of physical fluorescence and OPM image formation."""

import numpy as np
import pytest

from tests.simulation.objects import meridian_sphere

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "radius,phase,z,expected",
    [
        (0, 0, 0, 0.5),
        (4, 0, 0, 1),
        (4, np.pi / 100, 0, 0),
        (4, np.pi / 200, 0, 0.5),
        (4, 0, 0.05, 0),
    ],
)
def test_siemens_target_matches_matlab_angular_pattern(
    radius, phase, z, expected, simulation_parameters
):
    """Check the flat core, angular extrema and finite sheet thickness analytically."""
    from tests.simulation.objects import siemens_star

    axes = tuple(
        np.array([v]) for v in (z, radius * np.sin(phase), radius * np.cos(phase))
    )
    assert siemens_star(
        axes, **simulation_parameters["specimens"]["siemens-star"]["features"]
    ).item() == pytest.approx(expected, abs=2e-7)


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
