"""Exercise composite-export orchestration with real color and image writers."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
from PIL import Image
from scripts import ao_layer_projections, ao_tissue_geometry, export_ao_overview
from scripts.ao_tissue_geometry import TissueGeometry

pytestmark = pytest.mark.unit


def test_composite_cli_reuses_channel_one_cache_and_exports_matching_rgb(
    tmp_path, monkeypatch
):
    """Carry local-channel pixels, ROI provenance and cache identity into both formats."""
    ao_root = tmp_path / "sample_ao_grid_results"
    store = ao_root / "time_0_z_0_ao_grid_results" / "grid_pos_0" / "ao_results.zarr"
    store.mkdir(parents=True)
    reader_path, reference_path = tmp_path / "reader.py", tmp_path / "reference.py"
    reader_path.write_text("# Approved reader fixture\n", encoding="utf-8")
    reference_path.write_text("# SJS plotting reference fixture\n", encoding="utf-8")
    output = tmp_path / "export"
    starting = np.linspace(-0.1234567, 0.2345678, 32)
    final = starting + 0.0123456

    class Result:
        def __init__(self, path, time, position):
            self.zarr_path = path
            self.starting_coefficients = starting
            self.final_coefficients = final

        def get_pupil(self, **kwargs):
            assert kwargs == {
                "coefficient_state": "final",
                "samples": 512,
                "norm": True,
                "wavelength": 0.488,
                "out": "radians",
            }
            return np.ma.zeros((8, 8))

    monkeypatch.setattr(
        export_ao_overview,
        "load_reference",
        lambda path: SimpleNamespace(AOOptimizationResult=Result),
    )
    geometry = TissueGeometry(
        image=np.empty((0, 0), np.uint16),
        shape_yx=(100, 100),
        origin_yx_um=(0.5, 0.5),
        scale_yx_um=(1, 1),
        preview_origin_yx_um=(0.5, 0.5),
        preview_scale_yx_um=(1, 1),
        extent_yx_um=(0, 100, 0, 100),
        centers_by_depth_index={0: {0: (50, 50)}},
        depth_offsets={0: 0},
        provenance={"full_resolution": True},
        planned_centers_by_depth_index={0: {0: (50, 50)}},
        planned_grid_by_depth_index={0: {"horizontal_y_um": [], "vertical_x_um": []}},
    )
    geometry_calls = []

    def read_geometry(root, projection, timepoint, channel, regions, **kwargs):
        geometry_calls.append((timepoint, channel, kwargs))
        return geometry

    monkeypatch.setattr(ao_tissue_geometry, "read_tissue_geometry", read_geometry)
    layers = {channel: np.zeros((100, 100), np.uint16) for channel in (0, 1, 3)}
    layers[0][:, :30] = 100
    layers[1][:, 35:65] = 200
    layers[3][:, 70:] = 400
    hashes = {
        str(channel): hashlib.sha256(image.tobytes()).hexdigest()
        for channel, image in layers.items()
    }
    requests = []

    def read_projections(geometry, *, timepoint, channel, depths, cache_dir):
        assert timepoint == 0 and depths == [0]
        requests.append((channel, Path(cache_dir)))
        return {0: layers[channel]}, {
            0: {
                "channel": channel,
                "depth_index": 0,
                "aggregate_z_start_stop": [2, 8],
                "tile_rois": [
                    {
                        "position_index": 0,
                        "registered_voxel_box_zyx": [[2, 8], [0, 100], [0, 100]],
                    }
                ],
                "mip_sha256": hashes[str(channel)],
                "projection_cache": str(Path(cache_dir) / f"layer_c{channel}.npy"),
                "projection_cache_fingerprint": f"source-and-roi-channel-{channel}",
                "resumed_xy_blocks": 1 if channel == 1 else 0,
                "shared_volume_read_bytes": 0
                if channel == 1
                else layers[channel].nbytes,
            }
        }

    monkeypatch.setattr(
        ao_layer_projections, "read_layer_projections", read_projections
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "export_ao_overview.py",
            str(tmp_path),
            "--ao-root",
            str(ao_root),
            "--ao-reader",
            str(reader_path),
            "--plot-reference",
            str(reference_path),
            "--output",
            str(output),
            "--planned-grid",
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

    assert geometry_calls == [
        (0, 0, {"planned_grid": True, "full_resolution": True, "load_image": False})
    ]
    assert requests == [
        (1, output / "projection_cache"),
        (0, output / "projection_cache_channel_0"),
        (3, output / "projection_cache_channel_3"),
    ]
    manifest = json.loads((output / "ao_publication.json").read_text("utf-8"))
    np.testing.assert_array_equal(
        manifest["regions"][0]["starting_coefficients"], starting
    )
    np.testing.assert_array_equal(manifest["regions"][0]["final_coefficients"], final)
    preview = manifest["tissue_overlay"]["pages"][0]
    native = manifest["native_tissue_overlays"]["pages"][0]
    roi = preview["projection_roi"]
    assert roi["aggregate_z_start_stop"] == [2, 8]
    assert roi["tile_rois"] == [
        {"position_index": 0, "registered_voxel_box_zyx": [[2, 8], [0, 100], [0, 100]]}
    ]
    assert roi["mip_sha256_by_channel"] == hashes
    assert roi["resumed_xy_blocks_by_channel"] == {"0": 0, "1": 1, "3": 0}
    assert roi["projection_cache_by_channel"]["1"] == str(
        output / "projection_cache" / "layer_c1.npy"
    )
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
