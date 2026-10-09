"""Unit checks of the required acquisition stages, using a mocked metadata store."""

import json
from pathlib import Path

import pytest
from scripts import deconvolve_opm_simulation as experiment


@pytest.mark.unit
@pytest.mark.parametrize("peak, samples", [(None, 1), (None, 3), (500, 1)])
def test_gc_requires_integrated_and_noisy_acquisition(peak, samples, monkeypatch):
    """Reject incomplete forward simulation before reading pixels or loading CUDA."""
    metadata = {"acquisition": {"peak_electrons": peak, "camera_samples": samples}}
    monkeypatch.setattr(Path, "read_text", lambda self, **kwargs: json.dumps(metadata))
    monkeypatch.setattr(
        experiment, "imread", lambda _: pytest.fail("Unexpected pixel read")
    )
    with pytest.raises(ValueError, match="camera-integrated, noisy"):
        experiment.deconvolve_saved_simulation("mock-acquisition")
