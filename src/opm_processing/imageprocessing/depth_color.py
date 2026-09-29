"""Intensity-weighted depth color for orthogonal maximum projections.

The LUT/intensity model follows the ZstackDepthColorCode plugin concept:
https://github.com/UU-cellbiology/ZstackDepthColorCode
This independent NumPy implementation selects the brightest voxel before
coloring, rather than mixing RGB maxima from different depths.
"""

from functools import lru_cache

import numpy as np
from cmap import Colormap


@lru_cache(maxsize=32)
def depth_palette(planes: int, colormap: str = "turbo") -> np.ndarray:
    """Sample a 256-color LUT with the reference plugin's one-based slice mapping."""
    if planes < 1:
        raise ValueError("Depth axis must contain at least one plane.")
    lut = np.rint(Colormap(colormap).lut(256)[:, :3] * 255).astype(np.uint8)
    indices = np.floor(255 * np.arange(1, planes + 1) / planes + 0.5).astype(int)
    palette = lut[indices]
    palette.setflags(write=False)
    return palette


def depth_projection(volume, axis, limits, colormap="turbo"):
    """Color the raw-intensity maximum's depth; first voxel wins an exact tie."""
    low, high = limits
    if volume.ndim != 3 or axis not in (0, 1, 2) or not high > low:
        raise ValueError(
            "Expected a ZYX volume, spatial axis and increasing contrast limits."
        )
    indices = volume.argmax(axis=axis)
    maximum = np.take_along_axis(
        volume, np.expand_dims(indices, axis), axis=axis
    ).squeeze(axis)
    brightness = np.nan_to_num(
        (maximum.astype(np.float32) - low) / (high - low), nan=0, posinf=1, neginf=0
    ).clip(0, 1)
    return (
        depth_palette(volume.shape[axis], colormap)[indices] * brightness[..., None]
    ).astype(np.uint8)


def depth_legends(shape, spacing, colormap="turbo"):
    """Describe local voxel-center depth ranges for XY/Z, XZ/Y and YZ/X."""
    return [
        dict(
            projection=projection,
            axis=axis,
            max_um=(int(n) - 1) * float(step),
            planes=int(n),
            colormap=colormap,
        )
        for projection, axis, n, step in zip(
            ("XY", "XZ", "YZ"), ("Z", "Y", "X"), shape, spacing
        )
    ]
