"""Reusable numerical assertions and measurements for synthetic image tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CorrelationMeasurement:
    """A masked Pearson correlation and its sample count."""

    value: float
    sample_count: int


def masked_correlation(
    candidate: np.ndarray,
    truth: np.ndarray,
    *,
    truth_percentile: float,
) -> CorrelationMeasurement:
    """Measure correlation where both reconstruction and truth are supported."""
    candidate_array = np.asarray(candidate)
    truth_array = np.asarray(truth)
    if candidate_array.shape != truth_array.shape:
        raise ValueError("candidate and truth must have identical shapes")
    supported = (candidate_array > 0) & (
        truth_array > np.percentile(truth_array, truth_percentile)
    )
    sample_count = int(np.count_nonzero(supported))
    if sample_count < 2:
        return CorrelationMeasurement(float("nan"), sample_count)
    value = float(
        np.corrcoef(candidate_array[supported], truth_array[supported])[0, 1]
    )
    return CorrelationMeasurement(value, sample_count)


def scale_invariant_rmse(candidate: np.ndarray, truth: np.ndarray) -> float:
    """Return RMSE after fitting one nonnegative candidate intensity scale."""
    candidate_array = np.asarray(candidate, dtype=np.float64)
    truth_array = np.asarray(truth, dtype=np.float64)
    denominator = float(np.sum(candidate_array**2))
    scale = max(
        0.0,
        float(np.sum(candidate_array * truth_array)) / max(denominator, 1e-12),
    )
    return float(np.sqrt(np.mean((scale * candidate_array - truth_array) ** 2)))


def shell_line_width_x(
    volume: np.ndarray,
    *,
    center_zyx: tuple[float, float, float],
    wall_x: float,
    half_window: int,
) -> float:
    """Measure discrete FWHM of an ellipsoidal shell along its X normal."""
    z_index = int(round(center_zyx[0]))
    y_index = int(round(center_zyx[1]))
    x_index = int(round(wall_x))
    start = max(0, x_index - half_window)
    stop = min(volume.shape[2], x_index + half_window + 1)
    profile = np.asarray(volume[z_index, y_index, start:stop], dtype=np.float64)
    if profile.size < 3 or not np.any(profile > profile.min()):
        return float("nan")
    peak = int(np.argmax(profile))
    half_max = profile.min() + 0.5 * (profile[peak] - profile.min())
    left = peak
    while left > 0 and profile[left - 1] >= half_max:
        left -= 1
    right = peak
    while right + 1 < profile.size and profile[right + 1] >= half_max:
        right += 1
    return float(right - left + 1)
