"""Validate real scalar-cache composition and both exported image formats."""

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image
from scripts import export_ao_overview

pytestmark = pytest.mark.unit


def test_composite_cli_reuses_channel_one_cache_and_exports_matching_rgb(
    ao_composite_case, monkeypatch
):
    """Prescribed spectral bands survive local projection, cache reuse and encoding."""
    case = ao_composite_case
    output, hashes = case.output, case.hashes
    monkeypatch.setattr(
        "sys.argv",
        [
            "export_ao_overview.py",
            str(case.root),
            "--ao-root",
            str(case.ao_root),
            "--projection",
            str(case.preview),
            "--output",
            str(output),
            "--planned-grid",
            "--overwrite",
            "--composite-channels",
            "0",
            "1",
            "3",
            "--pupil-diameter-um",
            "20",
            "--scale-bar-um",
            "5",
            "--preview-max-edge",
            "100",
        ],
    )
    export_ao_overview.main()

    manifest = json.loads((output / "ao_publication.json").read_text("utf-8"))
    np.testing.assert_array_equal(
        manifest["regions"][0]["starting_coefficients"], case.starting
    )
    np.testing.assert_array_equal(
        manifest["regions"][0]["final_coefficients"], case.final
    )
    preview = manifest["tissue_overlay"]["pages"][0]
    native = manifest["native_tissue_overlays"]["pages"][0]
    roi = preview["projection_roi"]
    assert roi["aggregate_z_start_stop"] == [0, 2]
    assert len(roi["tile_rois"]) == 1
    tile = roi["tile_rois"][0]
    assert tile["position_index"] == 0
    assert tile["registered_voxel_box_zyx"] == [[0, 2], [0, 100], [0, 100]]
    assert roi["mip_sha256_by_channel"] == hashes
    assert roi["resumed_xy_blocks_by_channel"] == {"0": 0, "1": 1, "3": 0}
    assert Path(roi["projection_cache_by_channel"]["1"]).is_file()
    assert "channel" not in roi and "mip_sha256" not in roi
    assert native["projection_roi"] == roi
    assert native["background_rgb_passthrough"] is True
    assert native["source_channel_mip_sha256"] == hashes
    assert native["source_mip_sha256"] is None
    assert native["background_composite"]["color_names"] == {
        "0": "magenta",
        "1": "cyan",
        "3": "orange",
    }
    with Image.open(preview["png"]) as image:
        assert image.size == (100, 100)
        png = np.asarray(image.convert("RGB"))
    with tifffile.TiffFile(native["tif"]) as image:
        assert image.is_bigtiff
        rgb = image.pages[0].asarray()
        metadata = json.loads(image.pages[0].description)["AOOverlay"]
        assert metadata["source_channel_mip_sha256"] == hashes
        assert metadata["source_mip_sha256"] is None
        assert metadata["background_rgb_passthrough"] is True
    assert rgb.shape == (100, 100, 3)
    for column, color in (
        (10, (255, 0, 255)),
        (50, (0, 255, 255)),
        (90, (255, 165, 0)),
    ):
        np.testing.assert_array_equal(png[10, column], color)
        np.testing.assert_array_equal(rgb[10, column], color)
