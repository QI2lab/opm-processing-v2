"""Resolve FFT translation aliases using physical limits and common image support."""

import numpy as np
import pytest

from opm_processing.imageprocessing import tilefusion

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("sampling", [(1, 1, 1), (1, 2, 2)])
def test_aliases_use_the_actual_registration_patch_periods(sampling):
    """Padded block reduction changes an odd-sized patch's physical FFT period."""
    shape = (7, 65, 97)
    periods = np.ceil(np.array(shape) / sampling).astype(int)
    expected = np.array((1, -8, 4))
    limits = np.array((2, 10, 6)) / sampling
    for wraps in ((-1, 1, -1), (1, -1, 1), (0, 0, -1)):
        aliases = tilefusion._registration_phase_aliases(
            expected / sampling + periods * wraps,
            periods,
            limits,
            shift_sampling=sampling,
        )
        assert len(aliases) == 1
        np.testing.assert_allclose(np.array(aliases[0]) * sampling, expected)
    assert (
        tilefusion._registration_phase_aliases(
            np.array((3, 20, 15)) / sampling, periods, limits, shift_sampling=sampling
        )
        == []
    )


def test_float32_alias_at_a_physical_limit_uses_retained_sampling():
    """Period cancellation must not reject an integer correction at its limit."""
    aliases = tilefusion._registration_phase_aliases(
        np.array((0, 0, -376.8), dtype=np.float32),
        (20, 1828, 380),
        (13, 414 / 5, 16 / 5),
        shift_sampling=(1, 5, 5),
    )
    assert len(aliases) == 1
    np.testing.assert_array_equal(np.rint(np.array(aliases[0]) * (1, 5, 5)), (0, 0, 16))
