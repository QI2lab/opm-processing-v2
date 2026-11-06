"""Make bounded-memory viewing copies from native layer projections."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import matplotlib
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any


def make_layer_previews(
    images: Mapping[int, Any],
    display_limits: Mapping[int, tuple[float, float]],
    *,
    max_edge: int = 6000,
) -> tuple[dict[int, np.ndarray], int]:
    """Area-average native display pixels without a full-resolution RGB copy.

    The native per-layer intensity limits are applied before averaging scalar
    layers. RGB layers are averaged directly after their channel compositing.
    Returned uint8 RGB images are ready to display without another colormap.
    """
    if not images or max_edge < 1:
        raise ValueError("Viewing copies require images and a positive edge length")
    shape = tuple(next(iter(images.values())).shape)
    rgb = len(shape) == 3 and shape[-1] == 3
    if (len(shape) != 2 and not rgb) or min(shape) < 1:
        raise ValueError("Layer projections must have a nonempty YX or RGB shape")
    shape_yx = shape[:2]
    factor = 2 ** max(0, math.ceil(math.log2(max(shape_yx) / max_edge)))
    output_shape = tuple((value + factor - 1) // factor for value in shape_yx)
    block = max(1024, factor)
    previews = {}
    for depth, image in sorted(images.items()):
        expected_dtype = np.dtype("uint8" if rgb else "uint16")
        if tuple(image.shape) != shape or np.dtype(image.dtype) != expected_dtype:
            raise ValueError(
                "Viewing copies require equally sized uint16 or RGB uint8 layers"
            )
        if not rgb:
            low, high = display_limits[depth]
            denominator = max(high - low, np.finfo(float).eps)
            normalized = np.clip(
                (np.arange(65536, dtype=float) - low) / denominator, 0, 1
            )
            lookup = matplotlib.colormaps["gray"](normalized, bytes=True)[:, 0]
        suffix = (3,) if rgb else ()
        preview = np.empty((*output_shape, *suffix), dtype=np.uint8)
        for y0 in range(0, shape_yx[0], block):
            for x0 in range(0, shape_yx[1], block):
                key = (slice(y0, y0 + block), slice(x0, x0 + block))
                values = np.asarray(image[(*key, slice(None))] if rgb else image[key])
                if not rgb:
                    values = lookup[values]
                ny, nx = values.shape[:2]
                rows, columns = (ny + factor - 1) // factor, (nx + factor - 1) // factor
                padded = np.zeros(
                    (rows * factor, columns * factor, *suffix), dtype=np.uint8
                )
                padded[:ny, :nx] = values
                total = padded.reshape(rows, factor, columns, factor, *suffix).sum(
                    axis=(1, 3), dtype=np.uint64
                )
                row_counts = np.minimum(factor, ny - np.arange(rows) * factor)
                column_counts = np.minimum(factor, nx - np.arange(columns) * factor)
                count = row_counts[:, None] * column_counts[None, :]
                if rgb:
                    count = count[..., None]
                patch = np.rint(total / count).astype(np.uint8)
                py, px = y0 // factor, x0 // factor
                preview[py : py + rows, px : px + columns] = patch
        previews[depth] = preview if rgb else np.repeat(preview[..., None], 3, axis=-1)
        print(
            f"Prepared layer {depth} viewing background: {output_shape}, {factor}x area reduction",
            flush=True,
        )
    return previews, factor
