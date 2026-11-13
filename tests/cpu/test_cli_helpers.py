"""CLI discovery, calibration, selection and export-format function units."""

import json
from pathlib import Path

import numpy as np
import pytest
import tensorstore as ts

from opm_processing import export_ome_tiff
from opm_processing.process import apply_stage_axis_flips, validate_empty_tile_options

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "suffix",
    [
        "_decon_deskewed.ome.zarr",
        "_deskewed.ome.zarr",
        "_fused.ome.zarr",
        ".ome.zarr",
        ".zarr",
    ],
)
def test_export_basename_retains_specimen_identifier(suffix):
    """Strip a processing suffix without truncating the acquisition identifier."""
    source = Path("sample_name" + suffix)
    assert export_ome_tiff.export_source_stem(source) == "sample_name"


@pytest.mark.parametrize("flips", list(np.ndindex(2, 2, 2)))
def test_stage_reflections_preserve_zxy_axis_order(flips):
    """Reflect each declared stage axis about its maximum and retain source data."""
    source = np.asarray(((2, -7, 11), (5, -3, 19)), dtype=float)
    original = source.copy()
    transformed = apply_stage_axis_flips(source, flips)
    for column, flip in enumerate(flips[::-1]):
        expected = (
            source[:, column].max() - source[:, column] if flip else source[:, column]
        )
        np.testing.assert_array_equal(transformed[:, column], expected)
    np.testing.assert_array_equal(source, original)


@pytest.mark.parametrize(
    "threshold,fraction,valid",
    [
        (None, 0.01, True),
        (0, 1, True),
        (10, 0.1, True),
        (-1, 0.1, False),
        (np.nan, 0.1, False),
        (1, 0, False),
        (1, 1.1, False),
        (1, np.inf, False),
    ],
)
def test_empty_channel_options_preserve_physical_thresholds(threshold, fraction, valid):
    """Accept nonnegative count thresholds and finite occupancy fractions."""
    if valid:
        assert validate_empty_tile_options(threshold, fraction) == (threshold, fraction)
    else:
        with pytest.raises(ValueError):
            validate_empty_tile_options(threshold, fraction)


@pytest.mark.parametrize("tile_shape", [(16, 16), (32, 32)])
@pytest.mark.parametrize("dtype", [np.uint16, np.float32])
def test_tiff_tiles_preserve_page_order_partial_edges_and_counts(tile_shape, dtype):
    """Read genuine TensorStore tiles without dropping padded-edge photon values."""
    source = np.arange(2 * 2 * 3 * 17 * 19, dtype=dtype).reshape(2, 2, 3, 17, 19)
    tiles = list(export_ome_tiff.image_tiles(ts.array(source), tile_shape))
    spatial = int(np.prod(np.ceil(np.asarray(source.shape[-2:]) / tile_shape)))
    assert len(tiles) == np.prod(source.shape[:3]) * spatial
    assert sum(tile.sum(dtype=np.float64) for tile in tiles) == source.sum(
        dtype=np.float64
    )
    np.testing.assert_array_equal(
        tiles[0], source[0, 0, 0, : tile_shape[0], : tile_shape[1]]
    )
    np.testing.assert_array_equal(
        tiles[spatial], source[0, 0, 1, : tile_shape[0], : tile_shape[1]]
    )
    assert all(tile.dtype == dtype for tile in tiles)


@pytest.mark.parametrize(
    "export_image_input,bounds",
    [
        (kind, bounds)
        for kind in ("image", "collection")
        for bounds in (None, (1, 4, 2, 5), (2, 2, 0, 3))
    ]
    + [("multiple", None)],
    indirect=["export_image_input"],
)
def test_tiff_image_functions_resolve_metadata_and_crop_pixels(
    export_image_input, bounds, tmp_path
):
    """Resolve real image layouts and retain camera values through bounded export."""
    import tifffile
    from ome_types import from_xml

    case = export_image_input
    if case.kind == "multiple":
        with pytest.raises(ValueError, match="single-position"):
            export_ome_tiff.resolve_export_image(case.path)
        return
    assert export_ome_tiff.resolve_export_image(case.path) == case.image
    template, provenance = export_ome_tiff.load_export_metadata(case.path)
    output = tmp_path / "image.ome.tif"
    if bounds is not None and bounds[0] == bounds[1]:
        with pytest.raises(ValueError, match="ROI bounds"):
            export_ome_tiff.export_image(
                case.path, output, template, provenance, bounds_yx=bounds
            )
        assert not output.exists()
    else:
        export_ome_tiff.export_image(
            case.path, output, template, provenance, bounds_yx=bounds, workers=1
        )
        expected = (
            case.pixels
            if bounds is None
            else case.pixels[..., bounds[0] : bounds[1], bounds[2] : bounds[3]]
        )
        with tifffile.TiffFile(output) as tif:
            np.testing.assert_array_equal(
                tif.asarray().reshape(expected.shape), expected
            )
            exported = from_xml(tif.ome_metadata, validate=True)
            image = exported.images[0]
            pixels = image.pixels
            assert (pixels.size_z, pixels.size_y, pixels.size_x) == expected.shape[-3:]
            spacing = (1, 1, 1) if case.kind == "image" else (1, 0.115, 0.115)
            origin = np.array((0, 0, 0) if case.kind == "image" else (4, 30, 20), float)
            if bounds is not None:
                origin[1:] += np.asarray(bounds[::2]) * spacing[1:]
            assert (image.stage_label.z, image.stage_label.y, image.stage_label.x) == (
                pytest.approx(origin)
            )
            for plane in pixels.planes:
                assert (plane.position_z, plane.position_y, plane.position_x) == (
                    pytest.approx(origin + np.array((plane.the_z * spacing[0], 0, 0)))
                )
            annotations = exported.structured_annotations.map_annotations[
                -1
            ].value.model_dump()
            if bounds is None:
                assert "ROIBoundsLevelZeroYX" not in annotations
            else:
                assert json.loads(annotations["ROIBoundsLevelZeroYX"]) == list(bounds)
                assert json.loads(annotations["ROIOriginTCZYX"]) == pytest.approx(
                    [0, 0, *origin]
                )


@pytest.mark.parametrize("export_image_input", ["collection"], indirect=True)
@pytest.mark.parametrize(
    "variable,damage",
    [
        (False, None),
        (False, "shape"),
        (False, "dtype"),
        (False, "pyramid"),
        (True, None),
        (True, "shape"),
        (True, "dtype"),
    ],
)
def test_resume_readers_require_matching_shape_dtype_and_pyramid(
    export_image_input, variable, damage
):
    """Reject incompatible saved data while accepting the exact recorded geometry."""
    from opm_processing.process import (
        open_resume_collection,
        open_variable_resume_collection,
    )

    case = export_image_input
    shape = list(case.pixels.shape)
    if damage == "shape":
        shape[-3] += 1
    dtype = np.dtype(np.float32 if damage == "dtype" else case.pixels.dtype)
    if variable:
        reader, args = open_variable_resume_collection, ([tuple(shape)], dtype)
    else:
        shape.insert(1, 1)
        reader, args = (
            open_resume_collection,
            (tuple(shape), dtype, (1, 2) if damage == "pyramid" else (1,)),
        )
    if damage:
        with pytest.raises(ValueError, match="incompatible"):
            reader(case.path, *args)
    else:
        output = reader(case.path, *args)
        np.testing.assert_array_equal(output.arrays[0].read().result(), case.pixels)


@pytest.mark.parametrize("zero", [False, True])
@pytest.mark.parametrize("method", ["stride", "block_mean"])
@pytest.mark.parametrize("dtype", [np.uint16, np.float32])
def test_position_pyramid_queue_writes_all_levels(
    position_store_factory, zero, method, dtype
):
    """Retain intensity or an explicit zero across every queued image pyramid."""
    from skimage.measure import block_reduce

    from opm_processing.process import queue_position_pyramid_writes

    source = np.arange(3 * 9 * 518, dtype=np.float32).reshape(3, 9, 518) / 4 + 0.375
    base = source.astype(dtype)
    collection = position_store_factory(
        np.zeros((1, 1, 1, *source.shape), dtype),
        multiscale_factors_yx=(2, 4),
        multiscale_downsample=method,
    )
    writes = queue_position_pyramid_writes(
        collection, 0, 0, 0, dtype(0) if zero else source
    )
    for future in writes:
        future.result()
    assert len(writes) == 3
    for factor, arrays in zip(
        collection.multiscale_factors_yx, collection.multiscale_arrays, strict=True
    ):
        expected = base[..., ::factor, ::factor]
        if method == "block_mean" and factor > 1:
            complete = base[..., : 9 // factor * factor, : 518 // factor * factor]
            expected = block_reduce(complete, (1, factor, factor), func=np.mean)
            if dtype == np.uint16:
                expected = np.rint(expected)
            expected = expected.astype(dtype)
        np.testing.assert_array_equal(
            arrays[0][0, 0].read().result(),
            0 if zero else expected,
        )


@pytest.mark.parametrize("times", [1, 3])
def test_acquisition_manifest_is_json_serializable(acquisition_metadata, times):
    """Retain recorded calibration and derived index sizes in the inspector payload."""
    from dataclasses import replace

    shape = (times, *acquisition_metadata.shape[1:])
    acquisition = replace(acquisition_metadata, shape=shape)
    manifest = json.loads(json.dumps(acquisition.to_dict()))
    assert manifest["path"] == str(acquisition.path)
    assert manifest["shape"] == list(shape)
    assert manifest["index_sizes"] == dict(zip(acquisition.axes, shape, strict=True))
    assert manifest["camera_offset"] == acquisition.camera_offset
    assert manifest["camera_conversion"] == acquisition.camera_conversion


def test_estimation_worker_returns_input_errors_and_closes_pipe():
    """Return the original input error through a real pipe and close it."""
    import multiprocessing as mp
    from contextlib import closing

    from opm_processing.process import run_estimate_illuminations

    receiving, sending = mp.Pipe(duplex=False)
    with closing(receiving), closing(sending):
        invalid = ts.array(np.zeros((1, 1, 1, 4, 5), np.uint16))
        run_estimate_illuminations(invalid, 100, 0.25, None, None, None, sending)
        error = receiving.recv()
        assert isinstance(error, ValueError)
        assert sending.closed
