"""Coordinate transforms shared by projection and volumetric tile fusion."""

from collections.abc import Sequence

import numpy as np


def stage_positions_to_image_coordinates(
    positions: Sequence[Sequence[float]] | np.ndarray,
    *,
    reverse_y: bool = True,
) -> np.ndarray:
    """Convert stage positions to image-placement coordinates.

    Positions may be YX or ZYX. In both cases Y is the penultimate axis. OPM
    stage Y motion is opposite image Y, so the default transform reverses only
    that placement coordinate; it never flips image pixels.

    Parameters
    ----------
    positions
        Stage positions in YX or ZYX order.
    reverse_y
        Whether to reverse the stage-derived image-Y placement coordinate.

    Returns
    -------
    numpy.ndarray
        Independent float64 coordinates suitable for tile placement.
    """
    coordinates = np.asarray(positions, dtype=np.float64).copy()
    if (
        coordinates.ndim != 2
        or coordinates.shape[1] not in (2, 3)
        or coordinates.shape[0] == 0
    ):
        raise ValueError("positions must have nonempty shape (n, 2) or (n, 3)")
    if reverse_y:
        coordinates[:, -2] *= -1.0
    return coordinates
