"""Shared OME-NGFF coordinate and pyramid helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


SPATIAL_DECIMALS = 3


def round_spatial(value: float) -> float:
    """Round one physical coordinate or spacing to one nanometer."""
    return round(float(value), SPATIAL_DECIMALS)


def round_spatial_values(values: Sequence[float]) -> tuple[float, ...]:
    """Round physical coordinate values to three decimal places."""
    return tuple(round_spatial(value) for value in values)


def round_tczyx_transform(values: Sequence[float]) -> list[float]:
    """Round only the spatial ZYX values of a TCZYX transform."""
    if len(values) != 5:
        raise ValueError("A TCZYX coordinate transform must contain five values")
    return [float(values[0]), float(values[1]), *round_spatial_values(values[2:])]


def downsample_yx(
    image: np.ndarray,
    factor: int,
    method: str = "stride",
) -> np.ndarray:
    """Downsample the last two axes, using striding by default."""
    factor = int(factor)
    if factor < 1:
        raise ValueError("Pyramid factors must be positive integers")
    array = np.asarray(image)
    if factor == 1:
        return array
    if method == "stride":
        return array[..., ::factor, ::factor]
    if method != "block_mean":
        raise ValueError('Pyramid downsampling must be "stride" or "block_mean"')
    y_size = (array.shape[-2] // factor) * factor
    x_size = (array.shape[-1] // factor) * factor
    if y_size == 0 or x_size == 0:
        raise ValueError("Pyramid factor exceeds the image dimensions")
    trimmed = array[..., :y_size, :x_size]
    reshaped = trimmed.reshape(
        *trimmed.shape[:-2],
        y_size // factor,
        factor,
        x_size // factor,
        factor,
    )
    reduced = reshaped.mean(axis=(-3, -1))
    if np.issubdtype(array.dtype, np.integer):
        reduced = np.rint(reduced)
    return reduced.astype(array.dtype, copy=False)
