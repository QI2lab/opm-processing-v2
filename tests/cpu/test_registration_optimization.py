"""Reject inconsistent cycles while preserving measured depth placement."""

import numpy as np
import pytest

from opm_processing.imageprocessing.tilefusion import TileFusion


@pytest.mark.unit
@pytest.mark.parametrize("method", ["TWO_ROUND", "TWO_ROUND_ITERATIVE"])
def test_cycle_rejection_preserves_depth_links_and_component_anchors(
    fusion_operator, method
):
    """Remove a bad cycle edge without losing valid depth links or graph anchors."""
    fusion = fusion_operator(
        TileFusion,
        _tile_positions=[(0, 0, 0)] * 6,
        _tiles_by_time=(tuple(range(6)),),
        pairwise_metrics={
            (0, 1): (0, 12, 0, 0.71),
            (0, 2): (0, -180, 0, 0.95),
            (1, 3): (0, -180, 0, 0.95),
            (2, 3): (0, 0, 0, 0.95),
            (4, 5): (0, -50, 0, 0.95),
        },
    )
    fusion.optimize_shifts(method=method)
    np.testing.assert_allclose(
        fusion.global_offsets[:, 1], (0, 0, -180, -180, 0, -50), atol=1e-8
    )
    assert set(fusion.optimized_pairwise_metrics) == {(0, 2), (1, 3), (2, 3), (4, 5)}
    assert (0, 1) in fusion.pairwise_metrics  # Keep raw measurements for cache reuse.


@pytest.mark.unit
def test_depth_measurement_weights_reflect_registration_sampling():
    """Match the analytic weighted solution for unequal registration resolutions."""
    links = [
        {"i": 0, "j": 1, "t": np.array((4, 0, 0)), "w": 1, "sampling": (3, 5, 5)},
        {"i": 1, "j": 2, "t": np.array((0, 0, 0)), "w": 1, "sampling": (3, 5, 5)},
        {"i": 0, "j": 2, "t": np.array((2, 0, 0)), "w": 1, "sampling": (1, 5, 5)},
    ]
    shifts = TileFusion._solve_global(links, 3, [0])
    # Two coarse measurements in series have variance 3^2 + 3^2;
    # the independent full-Z-resolution edge has variance one.
    expected = (2 + 4 / 18) / (1 + 1 / 18)
    np.testing.assert_allclose(shifts[2, 0], expected, atol=1e-10)
