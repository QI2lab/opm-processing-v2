"""Coordinate transforms shared by projection and volumetric tile fusion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from opm_processing.dataio.acquisition import AcquisitionMetadata


STAGE_COORDINATE_VERSION = "stage-endpoint-origin-v1"


def deskewed_stage_scan_offsets(acquisition: AcquisitionMetadata) -> np.ndarray:
    """Anchor deskewed scan pixels to the recorded stage trajectory.

    The deskew input is reordered and then trimmed along its scan dimension.
    Its first retained plane therefore need not be the first acquired plane.
    Return the displacement from each recorded stage origin to that retained
    plane, expressed in image ZYX micrometers. Camera-reference, post-deskew
    crop, and Z-averaging offsets are separate from this scan displacement.

    Parameters
    ----------
    acquisition : AcquisitionMetadata
        Recorded acquisition geometry and scan trimming settings.

    Returns
    -------
    numpy.ndarray
        One unrounded ZYX displacement per recorded stage position. Non-stage
        acquisitions have zero displacements.

    Raises
    ------
    ValueError
        Scan trimming removes every plane, scan sampling is invalid, or
        recorded trajectories require different scan orientations.
    """
    offsets = np.zeros((len(acquisition.stage_positions_zxy), 3), dtype=np.float64)
    if "stage" not in acquisition.mode.lower():
        return offsets

    count = acquisition.scan_position_count
    leading = (
        acquisition.excess_scan_start_positions or acquisition.excess_scan_positions
    )
    trailing = acquisition.excess_scan_end_positions
    if leading < 0 or trailing < 0 or leading + trailing >= count:
        raise ValueError("Stage scan trimming must retain at least one scan plane")
    reversed_scan = acquisition.scan_axis_reversed
    first_index = count - 1 - leading if reversed_scan else leading
    if first_index == 0:
        return offsets

    fallback_step = acquisition.scan_axis_step_um
    for position in range(len(offsets)):
        step = None
        if (
            count > 1
            and position < len(acquisition.scan_start_positions_xyz)
            and position < len(acquisition.scan_end_positions_xyz)
        ):
            start_x = acquisition.scan_start_positions_xyz[position][0]
            end_x = acquisition.scan_end_positions_xyz[position][0]
            if end_x != start_x:
                step = (end_x - start_x) / (count - 1)
        if step is None:
            step = fallback_step
        if step is None or not np.isfinite(step) or step == 0:
            raise ValueError("Stage scan origin requires a finite nonzero scan step")
        offsets[position, 1] = -float(step) * first_index
    return offsets


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
