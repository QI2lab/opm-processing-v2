"""Check actual producer AO file formats and independently prescribed saved states."""

import numpy as np
import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_saved_modal_history_preserves_full_precision_and_last_update(
    ao_result_factory, zarr_format
):
    """Both producer storage formats retain signed OPD vectors without rounding."""
    starting = np.linspace(-0.123456789, 0.234567891, 32)
    history = starting + np.arange(4)[:, None] * 0.00123456789
    case = ao_result_factory(
        starting=starting,
        history=history,
        iterations=2,
        modes=(5, 7),
        zarr_format=zarr_format,
    )
    result = case.result
    np.testing.assert_array_equal(result.starting_coefficients, starting)
    np.testing.assert_array_equal(result.optimal_coefficients, history)
    np.testing.assert_array_equal(result.final_coefficients, history[-1])
    assert result.metadata["num_iterations"] == 2
    assert result.metadata["modes_to_optimize"] == [5, 7]


def test_saved_optimization_with_no_modal_updates_keeps_its_start(ao_result_factory):
    """An empty recorded update sequence leaves the nonflat mirror state intact."""
    starting = np.linspace(-0.12, 0.18, 32)
    case = ao_result_factory(
        starting=starting,
        history=np.empty((0, 32)),
        modes=(),
    )
    np.testing.assert_array_equal(case.result.final_coefficients, starting)


def test_legacy_saved_first_tilt_retains_sjs_vertical_orientation(ao_result_factory):
    """Unversioned SJS phase rendering keeps its historical vertical first tilt."""
    coefficients = np.zeros(32)
    coefficients[0] = 0.17
    case = ao_result_factory(final=coefficients)
    pupil = case.result.get_pupil(samples=33, out="opd")
    coordinates = np.linspace(-1, 1, 33)
    x, y = np.meshgrid(coordinates, coordinates)
    aperture = x**2 + y**2 <= 1
    np.testing.assert_array_equal(pupil.mask, ~aperture)
    np.testing.assert_allclose(
        pupil.compressed(), (2 * 0.17 * y)[aperture], rtol=0, atol=1e-14
    )


@pytest.mark.parametrize("history", [np.zeros((1, 31)), np.zeros((2, 32))])
def test_saved_optimization_rejects_incompatible_or_incomplete_history(
    ao_result_factory, history
):
    """A saved basis mismatch or update count cannot silently choose a final state."""
    case = ao_result_factory(history=history)
    with pytest.raises(ValueError, match=r"modal basis|every recorded modal update"):
        _ = case.result.final_coefficients
