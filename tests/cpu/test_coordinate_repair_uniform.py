"""Validate uniform scan corrections at the declared spatial precision."""

import numpy as np
import pytest

from opm_processing.dataio.coordinate_repair_layout import _uniform_scan_offset

pytestmark = pytest.mark.unit


def test_scan_endpoint_float_noise_preserves_one_physical_correction():
    """Large-coordinate subtraction noise must not reject identical scan spans."""
    offsets = np.asarray(
        [
            (0, -1025.1999999990676, 0),
            (0, -1025.200000000233, 0),
            (0, -1025.2, 0),
        ]
    )
    np.testing.assert_array_equal(_uniform_scan_offset(offsets), (0, -1025.2, 0))


def test_a_one_nanometer_scan_difference_is_not_a_uniform_translation():
    """A genuine difference at the stored coordinate precision remains rejected."""
    offsets = np.asarray(((0, -1025.2, 0), (0, -1025.201, 0)))
    with pytest.raises(ValueError, match="uniform scan-origin correction"):
        _uniform_scan_offset(offsets)
