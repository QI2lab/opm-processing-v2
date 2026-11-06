"""Shift-limit diagnostics distinguish qualifying measurements from bad matches."""

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def _warning_case(registration_pair, count=2, threshold=0.7):
    """Supply small overlapping projections for controlled measurement diagnostics."""
    image = np.arange(256, dtype=np.float32).reshape(1, 16, 16)
    limits = {(i, j): (0, 2, 2) for i in range(count) for j in range(i + 1, count)}
    return registration_pair(
        [image] * count,
        [(0, 0, 0)] * count,
        spacing=(1, 1, 1),
        sampling=(1, 1, 1),
        ssim_window=3,
        threshold=threshold,
        _is_2d=True,
        _tiles_by_time=(tuple(range(count)),),
        _tile_channel_nonzero={(i, 0): True for i in range(count)},
        _pair_registration_limits=limits,
        max_registration_shift_zyx=(0, 2, 2),
    )


@pytest.mark.parametrize(
    "score,threshold,qualifies",
    [(0.003, 0.7, False), (0.650, 0.7, False), (0.7, 0.7, True), (0.003, 0, True)],
)
def test_shift_limit_warning_requires_a_qualifying_similarity(
    registration_pair, monkeypatch, capsys, score, threshold, qualifies
):
    """Bad peaks stay rejected without suggesting wider registration bounds."""
    fusion = _warning_case(registration_pair, threshold=threshold)
    monkeypatch.setattr(
        fusion, "register_and_score", lambda *args, **kwargs: ((0, 8, 0), score)
    )
    fusion.refine_tile_positions_with_cross_correlation()
    output = capsys.readouterr().out
    assert not fusion.pairwise_metrics
    # Preserve the diagnostic measurement; retries still apply their own gate.
    assert fusion._rejected_registration_links[0, 1][-1] == score
    if qualifies:
        assert "Registration warning: 1 pair" in output
        assert (
            "largest per-axis absolute candidate corrections were (0, 8, 0)" in output
        )
    else:
        assert "low-similarity out-of-range candidates" in output
        assert "Registration warning:" not in output
        assert "--max-registration-shift-zyx" not in output


def test_warning_maxima_exclude_low_similarity_candidates(
    registration_pair, monkeypatch, capsys
):
    """Large bad peaks must not inflate the envelope reported for a valid match."""
    fusion = _warning_case(registration_pair, count=3)
    results = iter(
        [
            ((0, 8, 0), 0.95),
            ((0, -2756, 555), 0.650),
            ((0, -820, 1300), 0.003),
        ]
    )
    monkeypatch.setattr(
        fusion, "register_and_score", lambda *args, **kwargs: next(results)
    )
    fusion.refine_tile_positions_with_cross_correlation()
    output = capsys.readouterr().out
    warnings = [
        line for line in output.splitlines() if line.startswith("Registration warning:")
    ]
    assert len(warnings) == 1
    assert "1 pair" in warnings[0]
    assert "(0, 8, 0)" in warnings[0]
    assert "2756" not in warnings[0]
    assert "1300" not in warnings[0]
    assert "rejected 2 low-similarity out-of-range candidates" in output
