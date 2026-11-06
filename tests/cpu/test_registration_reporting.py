"""Explicit measurement state determines truthful shift-limit diagnostics."""

import pytest

from opm_processing.imageprocessing.tilefusion import _registration_rejection_messages

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "score,threshold,qualifies",
    [(0.003, 0.7, False), (0.650, 0.7, False), (0.7, 0.7, True), (0.003, 0, True)],
)
def test_shift_limit_warning_requires_a_qualifying_similarity(
    score, threshold, qualifies
):
    """Separate diagnostic categories from prescribed SSIM and correction state."""
    candidates = [(0, 1, (0, 8, 0), (0, 2, 2), score)]
    output = "\n".join(
        _registration_rejection_messages(0, (0, 2, 2), threshold, candidates)
    )
    if qualifies:
        assert "Registration warning: 1 pair" in output
        assert (
            "largest per-axis absolute candidate corrections were (0, 8, 0)" in output
        )
    else:
        assert "low-similarity out-of-range candidates" in output
        assert "Registration warning:" not in output
        assert "--max-registration-shift-zyx" not in output
    assert candidates == [(0, 1, (0, 8, 0), (0, 2, 2), score)]


def test_warning_maxima_exclude_low_similarity_candidates():
    """Report only qualifying measured corrections in the warning envelope."""
    candidates = [
        (0, 1, (0, 8, 0), (0, 2, 2), 0.95),
        (0, 2, (0, -80, 17), (0, 2, 2), 0.650),
        (1, 2, (0, -31, 90), (0, 2, 2), 0.003),
    ]
    lines = _registration_rejection_messages(0, (0, 2, 2), 0.7, candidates)
    warnings = [line for line in lines if line.startswith("Registration warning:")]
    assert len(warnings) == 1
    assert "1 pair" in warnings[0]
    assert "(0, 8, 0)" in warnings[0]
    assert "80" not in warnings[0]
    assert "90" not in warnings[0]
    assert "rejected 2 low-similarity out-of-range candidates" in lines[0]
