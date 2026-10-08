"""CPU checks of two-point profile measurements against known emitter geometry."""

import numpy as np
import pytest

from tests.physics_point_sources import two_point_profile


@pytest.mark.unit
@pytest.mark.parametrize(
    "offset, separation", (((0, 0, 3), 0.69), ((3, 0, 0), 1.2), ((0, 3, 0), 0.69))
)
def test_two_point_measurements_match_known_positions(offset, separation):
    """Recover exact separation and an empty valley for two discrete emitters.

    Parameters
    ----------
    offset : tuple of int
        Known displacement from the midpoint in acquisition indices.
    separation : float
        Independently known laboratory separation in micrometers.
    """
    volume = np.zeros((17, 17, 17), np.float64)
    center = np.array((8, 8, 8))
    volume[tuple(center - offset)] = 100
    volume[tuple(center + offset)] = 200
    measured = two_point_profile(volume, center, offset)
    assert measured["true_separation_um"] == pytest.approx(separation)
    assert measured["peak_separation_um"] == pytest.approx(separation)
    np.testing.assert_allclose(
        measured["peak_positions_um"], (-separation / 2, separation / 2), atol=1e-12
    )
    assert measured["valley_depth"] == 1


@pytest.mark.unit
@pytest.mark.parametrize("constant", (False, True), ids=("single-point", "flat-field"))
def test_two_point_measurement_does_not_split_one_peak(constant):
    """Return no pair measurement for a single emitter or a uniform profile.

    Parameters
    ----------
    constant : bool
        Use a uniform field instead of a single central emitter.
    """
    volume = np.full((17, 17, 17), float(constant), np.float64)
    volume[8, 8, 8] = 1
    measured = two_point_profile(volume, (8, 8, 8), (0, 0, 3))
    assert measured["peak_positions_um"] is None
    assert measured["peak_separation_um"] is None
    assert measured["valley_depth"] is None


@pytest.mark.unit
def test_adjacent_points_use_their_half_pixel_midpoint():
    """Sample adjacent emitters about their true midpoint without inventing a dip."""
    volume = np.zeros((17, 17, 17), np.float64)
    volume[8, 8, 8:10] = 100
    measured = two_point_profile(volume, (8, 8, 8.5), (0, 0, 0.5))
    assert measured["true_separation_um"] == pytest.approx(0.115)
    np.testing.assert_allclose(
        measured["distance_um"], np.linspace(-0.115, 0.115, 9), atol=1e-12
    )
    np.testing.assert_allclose(
        measured["profile"], (50, 75, 100, 100, 100, 100, 100, 75, 50)
    )
    assert measured["peak_positions_um"] is None
