"""Test top-level OPM processing configuration."""

import numpy as np

from opm_processing.process import _apply_stage_axis_flips


def test_stage_orientation_is_driven_by_explicit_axis_configuration():
    """Verify explicit axis flips determine stage-position orientation."""
    positions = np.array(
        [
            [2.0, 10.0, 100.0],
            [5.0, 20.0, 130.0],
        ]
    )

    transformed = _apply_stage_axis_flips(positions, axis_flips_xyz=(True, False, True))

    np.testing.assert_array_equal(
        transformed,
        [[3.0, 10.0, 30.0], [0.0, 20.0, 0.0]],
    )
    np.testing.assert_array_equal(
        positions,
        [[2.0, 10.0, 100.0], [5.0, 20.0, 130.0]],
    )
