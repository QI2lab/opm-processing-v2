"""Resolve FFT translation aliases using physical limits and common image support."""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from skimage.measure import block_reduce

from opm_processing.imageprocessing import tilefusion

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("sampling", [(1, 1, 1), (1, 2, 2)])
@pytest.mark.parametrize("unrelated", [False, True])
def test_thin_depth_peak_is_unwrapped_and_rescored(monkeypatch, sampling, unrelated):
    """Recover independent specimen crops after upstream picks the wrong periods."""
    rng = np.random.default_rng(485)
    truth = gaussian_filter(rng.uniform(0, 1000, (8, 73, 101)), (0, 1, 1))
    fixed = truth[:7, 8:73, :97].astype(np.float32)
    moving = truth[1:8, :65, 4:101].astype(np.float32)
    if unrelated:
        moving = gaussian_filter(rng.uniform(0, 1000, moving.shape), (0, 1, 1))
    expected = np.array((1, -8, 4))
    reduced = tuple(block_reduce(a, sampling, np.mean) for a in (fixed, moving))
    mask = block_reduce(np.ones_like(fixed, bool), sampling, np.min)
    # The FFT period includes block-reduction padding: a 97-pixel X patch
    # sampled by two has a 49-bin period, equivalent to 98 original pixels.
    periodic = expected / sampling + np.array(reduced[0].shape) * (-1, 1, -1)

    def upstream_peak(*args, **kwargs):
        """Model upstream disambiguation selecting the opposite crop edges."""
        return periodic, 0, 0

    monkeypatch.setattr(tilefusion, "phase_cross_correlation", upstream_peak)
    shift, score = tilefusion.TileFusion.register_and_score(
        *reduced,
        win_size=3,
        max_shift=np.array((2, 10, 6)) / sampling,
        allow_bounded_peak=False,
        support_masks=(mask, mask),
        shift_sampling=sampling,
    )
    np.testing.assert_allclose(np.asarray(shift) * sampling, expected, atol=1e-4)
    assert (score >= 0.7) is not unrelated
    if not unrelated:
        assert score > 0.99


def test_alias_selection_scores_both_allowed_real_space_shifts(monkeypatch):
    """A genuine long translation must not be folded to the nearest zero shift."""
    rng = np.random.default_rng(486)
    truth = rng.uniform(0, 1000, (7, 17, 116)).astype(np.float32)
    fixed, moving = truth[..., :64], truth[..., 52:116]

    def upstream_peak(*args, **kwargs):
        return np.array((0, 0, -12)), 0, 0

    monkeypatch.setattr(tilefusion, "phase_cross_correlation", upstream_peak)
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


def test_peak_without_an_allowed_periodic_equivalent_stays_out_of_range(monkeypatch):
    """Unwrapping must not clamp a genuinely excessive translation to a limit."""
    rng = np.random.default_rng(487)
    truth = rng.uniform(0, 1000, (7, 17, 84)).astype(np.float32)
    fixed, moving = truth[..., :64], truth[..., 20:84]

    def upstream_peak(*args, **kwargs):
        return np.array((0, 0, -44)), 0, 0

    monkeypatch.setattr(tilefusion, "phase_cross_correlation", upstream_peak)
    shift, _ = tilefusion.TileFusion.register_and_score(
        fixed,
        moving,
        win_size=3,
        max_shift=(2, 4, 6),
        allow_bounded_peak=False,
    )
    assert abs(shift[2]) > 6


def test_aliases_use_the_actual_registration_patch_periods():
    """The three logged X-edge peaks have physically small equivalents."""
    for shift, expected_x in ((-1884, 16), (1885, -15), (1884, -16)):
        aliases = tilefusion._registration_phase_aliases(
            (0, -155 / 5, shift / 5), (20, 1828, 380), (13, 414 / 5, 285 / 5)
        )
        assert len(aliases) == 1
        np.testing.assert_allclose(
            np.array(aliases[0]) * (1, 5, 5), (0, -155, expected_x)
        )
    assert (
        tilefusion._registration_phase_aliases(
            (-2, -2756 / 5, 555 / 5), (20, 1828, 380), (13, 414 / 5, 285 / 5)
        )
        == []
    )
    assert (
        tilefusion._registration_phase_aliases(
            (3, -820 / 5, 1300 / 5), (20, 1828, 380), (13, 414 / 5, 285 / 5)
        )
        == []
    )


def test_depth_registration_refines_an_unwrapped_lateral_peak(
    registration_pair, shifted_slab_pair, monkeypatch
):
    """A thin slab must unwrap XY before contextual axial refinement and gating."""
    pair = shifted_slab_pair((64, 160, 88), 51, (-7, -40, 8), seed=20261102)
    fusion = registration_pair(pair.volumes, pair.positions, ssim_window=7)
    phase = tilefusion.phase_cross_correlation

    def opposite_x_period(fixed, moving, **kwargs):
        """Retain the measured peak but choose its opposite real-space X alias."""
        measured, error, difference = phase(fixed, moving, **kwargs)
        measured = tilefusion.xp.asarray(measured).copy()
        measured[-1] = measured[-1] % fixed.shape[-1] - fixed.shape[-1]
        return measured, error, difference

    monkeypatch.setattr(tilefusion, "phase_cross_correlation", opposite_x_period)
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
