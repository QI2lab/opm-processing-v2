"""Registered commands process the same simulated ground-truth acquisition."""

import json

import numpy as np
import pytest
import tifffile
from ome_types import from_xml

from opm_processing.dataio.position_collection import open_image_array

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("deconvolve", [False])
@pytest.mark.parametrize(
    "pipeline_case",
    [("mirror", False, 0.4, None, False), ("stage", False, 1.2, None, False)],
    indirect=True,
)
@pytest.mark.parametrize(
    "cli_export_options", ["all-gray", "selected-color"], indirect=True
)
def test_inspection_and_exports_process_simulated_acquisition(
    pipeline_case, cli_export_options, invoke_cli, tmp_path
):
    """Parse real commands and reopen pixels, physical coordinates and timestamps."""
    case = pipeline_case
    options = cli_export_options
    metadata = case.dataset.metadata
    source = case.dataset.path.parent
    manifest = json.loads(
        invoke_cli(
            "inspect-opm", source if options["depth_color"] else case.dataset.path
        ).output
    )
    assert manifest["shape"] == list(case.dataset.raw_data.shape)
    assert manifest["scan_axis_step_um"] == metadata.scan_axis_step_um
    assert manifest["camera_offset"] == metadata.camera_offset
    assert manifest["camera_conversion"] == metadata.camera_conversion

    destination = tmp_path / "ome-tiff"
    invoke_cli(
        "export-ome-tiff", case.output, output=destination, workers=options["workers"]
    )
    for suffix in ("fused", "max_z_fused"):
        image = source / f"pipeline_{suffix}.ome.zarr"
        expected = open_image_array(image).read().result()
        with tifffile.TiffFile(destination / f"pipeline_{suffix}.ome.tif") as tif:
            np.testing.assert_array_equal(
                tif.asarray().reshape(expected.shape), expected
            )
            pixels = from_xml(tif.ome_metadata, validate=True).images[0].pixels
            assert (pixels.size_t, pixels.size_c, pixels.size_z) == expected.shape[:3]
            assert pixels.time_increment == pytest.approx(
                case.collection.time_interval_s
            )

    invoke_cli(
        "export-projections",
        source,
        acquisition=case.dataset.path if options["depth_color"] else None,
        **options,
    )
    files = sorted((source / "projection_frames").rglob("*.tiff"))
    times = range(*(options["timepoints"] or (0, metadata.shape[0])))
    assert len(files) == len(times) * metadata.shape[1] * metadata.shape[2]
    for path in files:
        with tifffile.TiffFile(path) as tif:
            saved = tif.shaped_metadata[0]
            position, channel = saved["position"], saved["channel"]
            reference = case.collection.arrays[position][0, channel].read().result()
            np.testing.assert_allclose(
                saved["contrast_limits"], np.percentile(reference, (0.001, 99.999))
            )
            assert saved["timepoint"] in times
            assert saved["elapsed_ms"] == pytest.approx(
                saved["timepoint"] * metadata.volume_interval_s * 1000
            )
            assert saved["source_voxel_size_zyx_um"] == list(
                case.collection.voxel_size_um
            )
            assert tif.pages[0].photometric.name == (
                "RGB" if options["depth_color"] else "MINISBLACK"
            )
            assert np.any(tif.asarray() > 0)
