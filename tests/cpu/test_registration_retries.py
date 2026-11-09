"""Adaptive registration must recompute a consistent complete link graph."""

import pytest


@pytest.mark.unit
@pytest.mark.parametrize(
    "correction,expected,automatic,score",
    [
        ((15, -284, 0), (17, 299, 33), True, 0.9),
        ((-15, 46, 0), (17, 198, 33), True, 0.9),
        ((0, 0, 34), (13, 198, 39), True, 0.9),
        ((0, 420, 0), (13, 441, 33), True, 0.9),
        ((15, -284, 0), None, False, 0.9),
        ((15, -284, 0), None, True, 0.69),
        ((15, -284, 0), None, True, float("nan")),
        ((15, -284, 0), (17, 299, 33), True, 0.7),
    ],
    ids=(
        "zy",
        "negative-z",
        "x",
        "large-y",
        "explicit",
        "low-score",
        "invalid-score",
        "threshold",
    ),
)
def test_limits_follow_rejected_peak_and_sampling_without_widening_other_pairs(
    retry_fusion, correction, expected, automatic, score
):
    """Use measured peaks only when automatic limits and similarity permit it."""
    fusion = retry_fusion
    fusion._automatic_registration_limits = automatic
    original = fusion._pair_registration_limits.copy()
    fusion._rejected_registration_links = {
        (1, 2): (correction, original[1, 2], score),
        (3, 4): ((9999, 9999, 9999), original[3, 4], 0.69),
    }
    assert bool(fusion._expand_registration_limits()) == (expected is not None)
    assert fusion._pair_registration_limits == (
        original if expected is None else {**original, (1, 2): expected}
    )
    assert not fusion._expand_registration_limits()
