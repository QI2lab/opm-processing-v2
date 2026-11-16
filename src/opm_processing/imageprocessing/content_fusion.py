"""GPU DCT-entropy fusion of already registered overlap samples.

Adapted from multiview-stitcher weights.content_based_dct at 708fd8f96576e4e2934f1bc5d3e51e374c97de9f.
The upstream BSD-3-Clause license is retained below.

Copyright (c) 2023, Marvin Albert
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from __future__ import annotations

import itertools
import math

import cupy as cp
import numpy as np
from cupyx.scipy.fft import dctn
from cupyx.scipy.ndimage import affine_transform
from cupyx.scipy.special import entr


def content_aware_fusion(
    overlap: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return float32 CZYX numerator and weights, like weighted accumulation.

    Parameters
    ----------
    overlap : numpy.ndarray
        Coverage-normalized float32 samples in view, channel, Z, Y, X order.
        The caller supplies the overlap and its aligned DCT context.
    weights : numpy.ndarray
        Geometric feather/coverage weights with the same shape. Zero excludes
        geometric padding and missing channels; valid dark samples remain valid.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Weighted numerator and denominator, ready for the shared normalizer.
        Equal or zero content scores retain geometric feathering.
    """
    image = cp.asarray(overlap)
    geometric = cp.asarray(weights)
    valid = geometric > 0
    shape = image.shape[-3:]
    sizes = tuple(min(32, n) for n in shape)
    grid = tuple(math.ceil(n / d) for n, d in zip(shape, sizes, strict=True))
    quality = cp.zeros((*image.shape[:2], *grid), cp.float32)
    parts = [
        [(0, n // d * d, d)] + ([(n // d * d, n, n % d)] if n % d else [])
        for n, d in zip(shape, sizes, strict=True)
    ]
    radius = 0.5 * min(sizes)
    axes = (-3, -2, -1)
    for region in itertools.product(*parts):
        slices = tuple(slice(a, b) for a, b, _ in region)
        blocks = tuple((b - a) // d for a, b, d in region)
        dz, dy, dx = (d for _, _, d in region)
        layout = (*image.shape[:2], blocks[0], dz, blocks[1], dy, blocks[2], dx)
        pixels = image[(..., *slices)].reshape(layout).transpose(0, 1, 2, 4, 6, 3, 5, 7)
        support = (
            valid[(..., *slices)].reshape(layout).transpose(0, 1, 2, 4, 6, 3, 5, 7)
        )
        minimum = cp.min(cp.where(support, pixels, cp.inf), axis=axes, keepdims=True)
        minimum = cp.where(cp.isfinite(minimum) & (minimum > 1e-4), minimum, 0)
        pixels = cp.where(support, pixels, minimum)
        structured = cp.max(pixels, axis=axes) > minimum[..., 0, 0, 0]
        coefficients = dctn(pixels, axes=axes, norm="ortho")
        norm = cp.sqrt(cp.sum(coefficients**2, axis=axes, keepdims=True))
        cp.abs(coefficients, out=coefficients)
        coefficients /= cp.where(norm > 0, norm, 1)
        z, y, x = cp.ogrid[:dz, :dy, :dx]
        coefficients *= z + y + x < radius
        entropy = cp.sum(entr(coefficients), axis=axes) / np.log(2)
        target = tuple(
            slice(a // d, a // d + k)
            for (a, _, _), d, k in zip(region, sizes, blocks, strict=True)
        )
        quality[(..., *target)] = cp.where(
            structured & (support.sum(axis=axes) >= 0.2 * dz * dy * dx),
            entropy * (2 / radius**2),
            0,
        )
        del pixels, support, coefficients, norm
    del valid
    quality -= quality.min(axis=0, keepdims=True)
    total = quality.sum(axis=0, keepdims=True)
    quality /= cp.where(total > 0, total, 1)
    matrix = cp.asarray((1, *(1 / d for d in sizes)))
    offset = (0, *(-(d - 1) / (2 * d) for d in sizes))
    content = affine_transform(
        quality.reshape(-1, *grid),
        matrix,
        offset=offset,
        output_shape=(image.shape[0] * image.shape[1], *shape),
        order=1,
        mode="nearest",
    ).reshape(image.shape)
    total = content.sum(axis=0, keepdims=True)
    content /= cp.where(total > 0, total, 1)
    content *= geometric
    total = content.sum(axis=0, keepdims=True)
    cp.copyto(content, geometric, where=~(total > 0))
    image *= content
    return cp.asnumpy(image.sum(axis=0)), cp.asnumpy(content.sum(axis=0))
