"""Reconstruct requested WaveKit correction targets with the SDK's modal order.

The returned phase describes the requested aberration target in OPD units.
WaveKit's correction inverse produces approximately the opposite physical
mirror-induced OPD, with calibration and actuator-mode truncation errors. This
function does not reconstruct measured mirror shape or residual sample phase.
"""

from __future__ import annotations

from math import factorial
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

BASIS_VERSION = "wavekit-4.2-rms-piston-free-32-sdk-validated-v1"

# Verified against offline WaveKit 4.2 ModalCoef/Phase one-hot reconstructions.
# Positive m is cosine; negative m is sine. The SDK orders odd-|m| pairs as
# cosine then sine, unlike the historical SJS reader's signed-m lookup table.
WAVEKIT_MODE_NM: tuple[tuple[int, int], ...] = (
    (1, 1),
    (1, -1),
    (2, 0),
    (2, 2),
    (2, -2),
    (3, 1),
    (3, -1),
    (4, 0),
    (3, 3),
    (3, -3),
    (4, 2),
    (4, -2),
    (5, 1),
    (5, -1),
    (6, 0),
    (4, 4),
    (4, -4),
    (5, 3),
    (5, -3),
    (6, 2),
    (6, -2),
    (7, 1),
    (7, -1),
    (8, 0),
    (5, 5),
    (5, -5),
    (6, 4),
    (6, -4),
    (7, 3),
    (7, -3),
    (8, 2),
    (8, -2),
)


def _zernike_nm(
    n: int,
    m: int,
    radius: np.ndarray,
    theta: np.ndarray,
    *,
    norm: bool,
) -> np.ndarray:
    """Evaluate a real circular Zernike polynomial with optional RMS scaling."""
    absolute_m = abs(m)
    radial = np.zeros_like(radius, dtype=float)
    for k in range((n - absolute_m) // 2 + 1):
        coefficient = (
            (-1) ** k
            * factorial(n - k)
            / (
                factorial(k)
                * factorial((n + absolute_m) // 2 - k)
                * factorial((n - absolute_m) // 2 - k)
            )
        )
        radial += coefficient * radius ** (n - 2 * k)
    angular = np.cos(absolute_m * theta) if m > 0 else np.sin(absolute_m * theta)
    if m == 0:
        angular = 1.0
    scale = np.sqrt(n + 1) if m == 0 else np.sqrt(2 * (n + 1))
    return (scale if norm else 1.0) * radial * angular


def phase_from_modal_coeff(
    coeffs: Sequence[float] | np.ndarray,
    mode_indices: Sequence[int] | np.ndarray | None = None,
    *,
    samples: int = 512,
    norm: bool = True,
    wavelength: float | None = None,
    out: str = "coefficients",
) -> np.ma.MaskedArray:
    """Reconstruct a circular requested correction target with WaveKit modes.

    Indices are one-based and exclude piston, matching acquisition's 1..32
    ``ModalCoef`` indices. ``norm=True`` matches ``E_ZERNIKE_NORM.RMS``.
    Coefficients and wavelength must share a length unit; acquisition uses
    micrometers. ``out='radians'`` converts OPD by ``2*pi/wavelength``.

    The API and circular aperture match the approved SJS reader. The modal
    lookup corrects its odd-|m| sine/cosine swaps against offline SDK output.
    Rows run from negative to positive pupil Y, for display with origin='lower'.
    """
    coefficients = np.asarray(coeffs, dtype=float).ravel()
    if not np.isfinite(coefficients).all():
        raise ValueError("Modal coefficients must be finite")
    if isinstance(samples, bool) or int(samples) != samples or samples < 2:
        raise ValueError("samples must be an integer of at least 2")
    samples = int(samples)
    if mode_indices is None:
        indices = np.arange(1, coefficients.size + 1, dtype=int)
    else:
        supplied = np.asarray(mode_indices).ravel()
        if (
            not np.isfinite(supplied).all()
            or not np.equal(supplied, supplied.astype(int)).all()
        ):
            raise ValueError("WaveKit mode indices must be finite integers")
        indices = supplied.astype(int)
    if indices.size != coefficients.size:
        raise ValueError("Each modal coefficient requires one WaveKit mode index")
    if np.any((indices < 1) | (indices > len(WAVEKIT_MODE_NM))):
        raise ValueError("WaveKit mode indices must be in [1, 32]")
    axis = np.linspace(-1, 1, samples)
    xx, yy = np.meshgrid(axis, axis)
    radius = np.hypot(xx, yy)
    theta = np.arctan2(yy, xx)
    values = np.zeros_like(radius)
    for coefficient, index in zip(coefficients, indices, strict=True):
        if coefficient:
            n, m = WAVEKIT_MODE_NM[index - 1]
            values += coefficient * _zernike_nm(
                n, m, np.minimum(radius, 1), theta, norm=norm
            )
    pupil = np.ma.array(values, mask=radius > 1)
    if out in {"coefficients", "opd", "meters"}:
        return pupil
    if out not in {"waves", "radians"}:
        raise ValueError(
            "out must be 'coefficients', 'opd', 'meters', 'waves', or 'radians'"
        )
    if wavelength is None or not np.isfinite(wavelength) or wavelength <= 0:
        raise ValueError("A positive finite wavelength is required for converted phase")
    waves = pupil / wavelength
    return waves if out == "waves" else 2 * np.pi * waves
