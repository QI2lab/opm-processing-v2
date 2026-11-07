"""Resolve FFT translation aliases using physical limits and common image support."""

import numpy as np
import pytest
from skimage.measure import block_reduce

from opm_processing.imageprocessing import tilefusion

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("sampling", [(1, 1, 1), (1, 2, 2)])
@pytest.mark.parametrize("unrelated", [False, True])
def test_thin_depth_crops_are_registered_and_rescored(
    aperiodic_registration_object, sampling, unrelated
):
    """Resolve exact unblurred-object crop aliases, independently of camera resolution."""
    truth = aperiodic_registration_object((8, 73, 101), seed=485)
    fixed = truth[:7, 8:73, :97].astype(np.float32)
    moving = truth[1:8, :65, 4:101].astype(np.float32)
    if unrelated:
        moving = aperiodic_registration_object(moving.shape, seed=492).astype(
            np.float32
        )
    expected = np.array((1, -8, 4))
    reduced = tuple(block_reduce(a, sampling, np.mean) for a in (fixed, moving))
    mask = block_reduce(np.ones_like(fixed, bool), sampling, np.min)
    shift, score = tilefusion.TileFusion.register_and_score(
        *reduced,
        win_size=3,
        max_shift=np.array((2, 10, 6)) / sampling,
        allow_bounded_peak=False,
        support_masks=(mask, mask),
        shift_sampling=sampling,
    )
    assert (score >= 0.7) is not unrelated
    if not unrelated:
        np.testing.assert_allclose(np.asarray(shift) * sampling, expected, atol=1e-4)
        assert score > 0.99


def test_alias_selection_scores_both_allowed_real_space_shifts(
    aperiodic_registration_object,
):
    """Resolve a long exact-object alias with identity global histogram matching."""
    truth = aperiodic_registration_object(
        (15, 65, 116), seed=486, equal_column_histograms=True
    ).astype(np.float32)
    fixed, moving = truth[..., :64], truth[..., 52:116]
    np.testing.assert_array_equal(np.sort(fixed, axis=None), np.sort(moving, axis=None))
    # The false -12 alias would compare these unrelated 52-column arrangements.
    assert not np.array_equal(fixed[..., :52], moving[..., 12:])

    shift, score = tilefusion.TileFusion.register_and_score(
        fixed,
        moving,
        win_size=3,
        max_shift=(2, 4, 56),
        allow_bounded_peak=False,
        support_masks=(np.ones_like(fixed), np.ones_like(moving)),
    )
    np.testing.assert_allclose(shift, (0, 0, 52))
    assert score > 0.99


def test_peak_without_an_allowed_periodic_equivalent_stays_out_of_range(
    aperiodic_registration_object,
):
    """Unwrapping must not clamp a genuinely excessive translation to a limit."""
    truth = aperiodic_registration_object((7, 17, 84), seed=487).astype(np.float32)
    fixed, moving = truth[..., :64], truth[..., 20:84]

    shift, _ = tilefusion.TileFusion.register_and_score(
        fixed,
        moving,
        win_size=3,
        max_shift=(2, 4, 6),
        allow_bounded_peak=False,
    )
    assert abs(shift[2]) > 6


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


def test_depth_registration_refines_a_measured_lateral_peak(
    registration_pair, shifted_slab_pair
):
    """A thin slab must unwrap XY before contextual axial refinement and gating."""
    pair = shifted_slab_pair((64, 160, 88), 51, (-7, -40, 8), seed=20261102)
    fusion = registration_pair(pair.volumes, pair.positions, ssim_window=7)
    fusion.refine_tile_positions_with_cross_correlation()
    np.testing.assert_allclose(fusion.pairwise_metrics[0, 1][:3], pair.expected, atol=1)
    assert fusion.pairwise_metrics[0, 1][3] >= 0.7
    assert not fusion._rejected_registration_links


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
