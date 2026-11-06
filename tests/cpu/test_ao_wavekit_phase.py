"""Check WaveKit modal conventions against analytical Jacobi pupils."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import eval_jacobi
from scripts.ao_wavekit_phase import (
    BASIS_VERSION,
    WAVEKIT_MODE_NM,
    phase_from_modal_coeff,
)

pytestmark = pytest.mark.unit
_CONVENTION = Path(__file__).parents[1] / "fixtures" / "ao_wavekit_phase_sdk42.json"


@pytest.fixture
def analytical_basis():
    """Evaluate prescribed RMS-normalized circular modes with SciPy operators."""
    header = json.loads(_CONVENTION.read_text("utf-8"))
    assert header["basis_version"] == BASIS_VERSION
    modes = tuple(map(tuple, header["modal_basis_nm"]))
    coordinates = np.linspace(-1, 1, 33)
    x, y = np.meshgrid(coordinates, coordinates)
    radius = np.hypot(x, y)
    angle = np.arctan2(y, x)
    mask = radius > 1
    basis = []
    for n, m in modes:
        radial = radius ** abs(m) * eval_jacobi(
            (n - abs(m)) // 2, 0, abs(m), 2 * radius**2 - 1
        )
        angular = np.cos(m * angle) if m > 0 else np.sin(abs(m) * angle)
        if m == 0:
            angular = np.ones_like(angle)
        normalization = np.sqrt((n + 1) * (1 if m == 0 else 2))
        basis.append(normalization * radial * angular)
    return np.array(basis), mask, modes


def test_all_32_modes_match_analytical_phase_and_aperture(analytical_basis):
    """Each recorded modal index maps to its prescribed independent polynomial."""
    basis, mask, modes = analytical_basis
    assert WAVEKIT_MODE_NM == modes
    for index, expected in enumerate(basis, start=1):
        phase = phase_from_modal_coeff([1.0], [index], samples=33)
        np.testing.assert_array_equal(phase.mask, mask)
        # A float64 polynomial of degree at most eight has a small cancellation bound.
        np.testing.assert_allclose(
            phase.compressed(), expected[~mask], rtol=0, atol=2e-12
        )


def test_full_and_cumulative_analytical_targets(analytical_basis):
    """Known nonflat starting states and cumulative changes preserve the modal sum."""
    basis, mask, _ = analytical_basis
    starting = np.linspace(-0.12, 0.18, 32)
    first = starting + 0.1 * np.cos(np.arange(32))
    second = first + 0.07 * np.sin(np.arange(32) * 0.37)
    for coefficients in (starting, first, second, first - starting, second - starting):
        before = coefficients.copy()
        expected = np.tensordot(coefficients, basis, axes=1)
        phase = phase_from_modal_coeff(coefficients, samples=33, out="opd")
        np.testing.assert_array_equal(phase.mask, mask)
        np.testing.assert_allclose(
            phase.compressed(), expected[~mask], rtol=0, atol=2e-12
        )
        np.testing.assert_array_equal(coefficients, before)


def test_sparse_modes_preserve_order_units_and_standard_normalization(analytical_basis):
    """Reordered inputs and wavelength conversions retain the prescribed OPD."""
    basis, mask, _ = analytical_basis
    coefficients = np.array([0.12, -0.23])
    indices = np.array([7, 4])
    expected_opd = coefficients[0] * basis[6] + coefficients[1] * basis[3]
    for unit, factor in (
        ("opd", 1),
        ("waves", 1 / 0.488),
        ("radians", 2 * np.pi / 0.488),
    ):
        phase = phase_from_modal_coeff(
            coefficients, indices, samples=33, wavelength=0.488, out=unit
        )
        np.testing.assert_array_equal(phase.mask, mask)
        np.testing.assert_allclose(
            phase.compressed(),
            expected_opd[~mask] * factor,
            rtol=0,
            atol=2e-12 * factor,
        )
    np.testing.assert_array_equal(coefficients, [0.12, -0.23])
    np.testing.assert_array_equal(indices, [7, 4])
    standard = phase_from_modal_coeff([1], [3], samples=33, norm=False)
    assert standard[16, 16] == -1
    assert standard[16, 32] == 1
    rms = phase_from_modal_coeff([1], [3], samples=33)
    np.testing.assert_allclose(rms.compressed(), standard.compressed() * np.sqrt(3))
