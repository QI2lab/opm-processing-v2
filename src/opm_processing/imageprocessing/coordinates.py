"""Coordinate transforms shared by projection and volumetric tile fusion."""

from collections.abc import Sequence

import numpy as np


def stage_z_level_indices(
    stage_positions_zxy: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Return repeated-depth indices at each physical stage-XY tile location.

    Absolute stage Z cannot identify acquisition depth levels on a tilted
    coverslip because the fitted coverslip height varies across XY. Repeated
    visits to the same rounded XY location define depth level 0, 1, ... in
    acquisition order.

    Parameters
    ----------
    stage_positions_zxy : Sequence[Sequence[float]] | np.ndarray
        Acquisition-ordered physical stage positions in micrometers.

    Returns
    -------
    np.ndarray
        Zero-based depth visit index for each position.
    """
    positions = np.asarray(stage_positions_zxy, dtype=np.float64)

    visits: dict[tuple[float, float], int] = {}
    indices = np.empty(positions.shape[0], dtype=np.int64)
    for position_index, (_stage_z, stage_x, stage_y) in enumerate(positions):
        xy_key = (round(float(stage_x), 3), round(float(stage_y), 3))
        depth_index = visits.get(xy_key, 0)
        indices[position_index] = depth_index
        visits[xy_key] = depth_index + 1
    return indices


def stage_positions_to_image_coordinates(
    positions: Sequence[Sequence[float]] | np.ndarray,
    *,
    reverse_y: bool = True,
    reverse_z: bool = True,
    opm_angle_deg: float | None = None,
) -> np.ndarray:
    """Convert stage positions to image-placement coordinates.

    Positions may be YX or ZYX. In both cases Y is the penultimate axis. OPM
    physical stage motion is opposite motion of the sample in laboratory Z,
    while stage Y motion is opposite image Y. These placement-coordinate
    transforms never flip image pixels or the acquired scan axis.

    Parameters
    ----------
    positions
        Stage positions in YX or ZYX order.
    reverse_y
        Whether to reverse the stage-derived image-Y placement coordinate.
    reverse_z
        Whether to convert physical stage Z into the opposite laboratory-Z
        placement coordinate. Ignored for two-dimensional YX positions.
    opm_angle_deg
        Retained for compatibility. The illumination angle affects deskewing,
        not placement of the resulting orthogonal tiles. Stage Z motion must
        not introduce an additional image-Y translation.

    Returns
    -------
    numpy.ndarray
        Independent float64 coordinates suitable for tile placement.
    """
    coordinates = np.asarray(positions, dtype=np.float64).copy()
    if reverse_y:
        coordinates[:, -2] *= -1.0
    if reverse_z and coordinates.shape[1] == 3:
        coordinates[:, 0] *= -1.0
    # Deskewing already maps the oblique camera planes into orthogonal XYZ.
    # Applying a Z-dependent Y shear here would shear the tile origins again.
    return coordinates
