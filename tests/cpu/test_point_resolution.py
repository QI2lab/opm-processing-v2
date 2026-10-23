"""Two-point profile checks against prescribed physical emitter geometry."""

import numpy as np
import pytest

from tests.simulation.metrics import two_point_profile

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "point_profile_case",
    [
        ("pair", (0, 0, 3), 0.69),
        ("pair", (3, 0, 0), 1.2),
        ("pair", (0, 3, 0), 0.69),
        ("single", (0, 0, 3), 0.69),
        ("flat", (0, 0, 3), 0.69),
        ("adjacent", (0, 0, 0.5), 0.115),
    ],
    indirect=True,
    ids=["camera-x", "scan", "camera-y", "single-point", "flat-field", "adjacent"],
)
def test_point_profile_resolves_only_distinct_emitters(point_profile_case):
    """Measure physical pair spacing without inventing a dip in unresolved data."""
    case = point_profile_case
    measured = two_point_profile(case.volume, case.center, case.offset, case.microscope)
    assert measured["true_separation_um"] == pytest.approx(case.separation)
    if case.pattern == "pair":
        assert measured["peak_separation_um"] == pytest.approx(case.separation)
        np.testing.assert_allclose(
            measured["peak_positions_um"],
            (-case.separation / 2, case.separation / 2),
            atol=1e-12,
        )
        assert measured["valley_depth"] == 1
    else:
        assert measured["peak_positions_um"] is None
        assert measured["peak_separation_um"] is None
        assert measured["valley_depth"] is None
    if case.pattern == "adjacent":
        np.testing.assert_allclose(
            measured["distance_um"], np.linspace(-0.115, 0.115, 9), atol=1e-12
        )
        np.testing.assert_allclose(
            measured["profile"], (50, 75, 100, 100, 100, 100, 100, 75, 50)
        )
