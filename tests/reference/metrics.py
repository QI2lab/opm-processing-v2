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
    observation_mask: np.ndarray | None = None,
) -> CorrelationMeasurement:
    """Measure correlation on truth-selected pixels, including lost signal."""
    candidate_array = np.asarray(candidate)
    truth_array = np.asarray(truth)
    if candidate_array.shape != truth_array.shape:
        raise ValueError("candidate and truth must have identical shapes")
    supported = truth_array > np.percentile(truth_array, truth_percentile)
    if observation_mask is not None:
        supported &= observation_mask
    sample_count = int(np.count_nonzero(supported))
    if sample_count < 2:
        return CorrelationMeasurement(float("nan"), sample_count)
    value = float(np.corrcoef(candidate_array[supported], truth_array[supported])[0, 1])
    return CorrelationMeasurement(value, sample_count)


def shell_line_width_x(
    volume: np.ndarray,
    *,
    center_zyx: tuple[float, float, float],
    wall_x: float,
    half_window: int,
) -> float:
    """Measure discrete FWHM of an ellipsoidal shell along its X normal."""
    z_index = round(center_zyx[0])
    y_index = round(center_zyx[1])
    x_index = round(wall_x)
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
