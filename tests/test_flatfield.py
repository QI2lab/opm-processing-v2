"""Tests for BaSiCPy illumination estimation configuration."""

import numpy as np

from opm_processing.imageprocessing import flatfield


def test_estimator_uses_basicpy_defaults(monkeypatch):
    """Construct BaSiC without overriding any library model settings.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture.

    Returns
    -------
    None
        No value is returned.
    """
    constructor_calls = []

    class FakeBaSiC:
        """Minimal BaSiC replacement that records constructor arguments."""

        def __init__(self, *args, **kwargs):
            constructor_calls.append((args, kwargs))

        def fit(self, images):
            """Return a unit illumination with the sampled image shape."""
            self.flatfield = np.ones(images.shape[-2:], dtype=np.float32)

    monkeypatch.setattr(flatfield, "BaSiC", FakeBaSiC)
    datastore = np.ones((1, 1, 1, 1, 4, 5), dtype=np.uint16)

    result = flatfield.estimate_illuminations(datastore, 0.0, 1.0)

    assert constructor_calls == [((), {})]
    np.testing.assert_array_equal(result, np.ones((1, 4, 5), dtype=np.float32))
