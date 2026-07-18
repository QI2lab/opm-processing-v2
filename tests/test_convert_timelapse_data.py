"""Test timelapse conversion without import-time side effects."""

import importlib
import sys

import numpy as np


def test_conversion_module_import_performs_no_io(monkeypatch):
    """Verify importing the conversion module performs no I/O."""

    def unexpected_io(*_args, **_kwargs):
        """Fail if import unexpectedly attempts file or datastore I/O."""
        raise AssertionError("module import attempted acquisition I/O")

    monkeypatch.setattr("tensorstore.open", unexpected_io)
    sys.modules.pop("opm_processing.dataio.convert_timelapse_data", None)

    importlib.import_module("opm_processing.dataio.convert_timelapse_data")


def test_time_projection_uses_explicit_camera_calibration(monkeypatch, tmp_path):
    """Verify time projections use caller-provided camera calibration."""
    conversion = importlib.import_module("opm_processing.dataio.convert_timelapse_data")
    captured = {}
    monkeypatch.setattr(
        conversion,
        "_write_tiff",
        lambda data, axes, pixel_size, output: captured.update(
            data=data, axes=axes, pixel_size=pixel_size, output=output
        ),
    )
    data = np.array([[[10, 20]], [[30, 40]]], dtype=np.uint16)

    conversion.save_time_projection(
        data,
        0.2,
        tmp_path / "projection.tiff",
        camera_offset=10,
        camera_conversion=2,
    )

    np.testing.assert_array_equal(captured["data"], [[20, 40]])
    assert captured["axes"] == "YX"
    assert captured["pixel_size"] == 0.2
