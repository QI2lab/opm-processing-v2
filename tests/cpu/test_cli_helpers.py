"""CLI discovery, calibration, selection and export-format function units."""

from pathlib import Path

import numpy as np
import pytest
import tensorstore as ts

from opm_processing import encode_projections, export_ome_tiff, export_projections
from opm_processing.process import apply_stage_axis_flips, validate_empty_tile_options

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "cli_file_layout,expected",
    [
        (("sample_deskewed.ome.zarr",), "sample_deskewed.ome.zarr"),
        (
            ("sample_max_z_tissue_deskewed.ome.zarr",),
            "sample_max_z_tissue_deskewed.ome.zarr",
        ),
        (("sample_decon_deskewed.ome.zarr",), "sample_decon_deskewed.ome.zarr"),
        (
            (
                "sample_deskewed.ome.zarr",
                "sample_decon_deskewed.ome.zarr",
                "sample_max_z_decon_deskewed.ome.zarr",
            ),
            "sample_decon_deskewed.ome.zarr",
        ),
        ((), None),
        (("sample_max_z_decon_deskewed.ome.zarr",), None),
        (("first_decon_deskewed.ome.zarr", "second_deskewed.ome.zarr"), None),
    ],
    indirect=["cli_file_layout"],
)
def test_projection_discovery_selects_full_processed_data(cli_file_layout, expected):
    """Choose one full store without confusing previews or distinct acquisitions."""
    if expected is None:
        with pytest.raises(ValueError):
            export_projections.find_datasets(cli_file_layout)
    else:
        assert export_projections.find_datasets(cli_file_layout) == [
            cli_file_layout / expected
        ]
        with pytest.raises(ValueError, match="acquisition root"):
            export_projections.find_datasets(cli_file_layout / expected)


@pytest.mark.parametrize(
    "cli_file_layout,indices",
    [
        (("p0/c0/sample_t9.tiff", "p0/c0/sample_t10.tiff", "notes.txt"), (9, 10)),
        (("sample_t1.tiff", "sample_t3.tiff"), None),
        (("sample_t1.tiff", "sample_t01.tif"), None),
        (("notes.txt",), None),
    ],
    indirect=["cli_file_layout"],
)
def test_video_sequences_require_unique_contiguous_timepoints(cli_file_layout, indices):
    """Sort acquired indices numerically and reject missing or duplicate frames."""
    if indices is None:
        with pytest.raises(ValueError):
            encode_projections.find_sequences(cli_file_layout)
    else:
        sequences = encode_projections.find_sequences(cli_file_layout)
        assert len(sequences) == 1
        assert [int(p.stem.rsplit("t", 1)[1]) for p in sequences[0]] == list(indices)


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


@pytest.mark.parametrize(
    "values,expected",
    [
        ([0, 1, 2, np.nan], (0.00002, 1.99998)),
        ([7, 7, np.inf], (7, 8)),
        ([np.nan, np.inf], None),
    ],
)
def test_projection_contrast_uses_finite_first_exposure(values, expected):
    """Keep fixed intensity scaling with explicit constant and invalid-input cases."""
    if expected is None:
        with pytest.raises(ValueError, match="no finite"):
            export_projections.contrast_limits(np.asarray(values))
    else:
        np.testing.assert_allclose(
            export_projections.contrast_limits(np.asarray(values)), expected
        )


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
    "options",
    [
        {"fps": 0},
        {"fps": 121},
        {"bitrate": 0},
        {"gpu": -1},
        {"codec": "invalid"},
        {"rate_control": "invalid"},
    ],
)
def test_encoder_rejects_invalid_options_before_gpu_allocation(
    encoder_frames, tmp_path, options
):
    """Reject invalid playback and encoder settings without loading CUDA."""
    with pytest.raises(ValueError):
        encode_projections.encode_sequence(
            encoder_frames, tmp_path / "movie.mp4", **options
        )
    assert not (tmp_path / "movie.mp4").exists()


@pytest.mark.parametrize(
    "export_image_input", ["image", "collection", "multiple"], indirect=True
)
@pytest.mark.parametrize("bounds", [None, (1, 4, 2, 5), (2, 2, 0, 3)])
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
            pixels = from_xml(tif.ome_metadata, validate=True).images[0].pixels
            assert (pixels.size_z, pixels.size_y, pixels.size_x) == expected.shape[-3:]


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


@pytest.mark.parametrize("is_zero", [False, True])
@pytest.mark.parametrize("fail_write", [False, True])
def test_channel_checkpoint_requires_a_successful_write(
    processing_state, tmp_path, is_zero, fail_write
):
    """Retain channel pixels and record completion only after the write succeeds."""
    from opm_processing.process import write_checkpointed_roi_channel

    path = tmp_path / "output.ome.zarr"
    processing_state.initialize_run(path, configuration={}, overwrite=True)
    target = ts.array(np.zeros((2, 3, 4), np.uint16))
    if fail_write:
        readonly = ts.open(target.spec(), read=True, write=False).result()
        with pytest.raises(ValueError, match="does not support writing"):
            write_checkpointed_roi_channel(
                readonly,
                0 if is_zero else 17,
                processing_state,
                path,
                (0, 0, 0),
                is_zero=is_zero,
            )
        assert processing_state.completed_channels(path) == set()
    else:
        write_checkpointed_roi_channel(
            target,
            0 if is_zero else 17,
            processing_state,
            path,
            (0, 0, 0),
            is_zero=is_zero,
        )
        np.testing.assert_array_equal(target.read().result(), 0 if is_zero else 17)
        assert processing_state.completed_channels(path) == {(0, 0, 0)}
        assert processing_state.zero_channels(path) == (
            {(0, 0, 0)} if is_zero else set()
        )


@pytest.mark.parametrize("order", [("t", "p"), ("p", "t"), ("c", "z")])
def test_progress_groups_preserve_every_pending_volume(order):
    """Yield each pending tile once while accounting for completed group members."""
    from opm_processing.process import progress_tile_groups

    requested = [(0, 0), (0, 1), (1, 0), (1, 1)]
    pending = [(0, 1), (1, 0), (1, 1)]
    actual = list(progress_tile_groups(pending, requested, {(0, 0)}, order))
    assert actual == [(0, (1,)), (1, (0,)), (1, (1,))]


@pytest.mark.parametrize("zero", [False, True])
def test_position_pyramid_queue_writes_all_levels(position_store_factory, zero):
    """Retain intensity or an explicit zero across every queued image pyramid."""
    from opm_processing.process import queue_position_pyramid_writes

    source = np.arange(3 * 8 * 10, dtype=np.float32).reshape(3, 8, 10)
    collection = position_store_factory(
        np.zeros((1, 1, 1, *source.shape), np.float32),
        multiscale_factors_yx=(2, 4),
    )
    writes = queue_position_pyramid_writes(
        collection, 0, 0, 0, np.float32(0) if zero else source
    )
    for future in writes:
        future.result()
    assert len(writes) == 3
    for factor, arrays in zip(
        collection.multiscale_factors_yx, collection.multiscale_arrays, strict=True
    ):
        np.testing.assert_array_equal(
            arrays[0][0, 0].read().result(),
            0 if zero else source[..., ::factor, ::factor],
        )


@pytest.mark.parametrize("times", [1, 3])
def test_acquisition_manifest_is_json_serializable(acquisition_metadata, times):
    """Retain recorded calibration and derived index sizes in the inspector payload."""
    import json
    from dataclasses import replace

    shape = (times, *acquisition_metadata.shape[1:])
    acquisition = replace(acquisition_metadata, shape=shape)
    manifest = json.loads(json.dumps(acquisition.to_dict()))
    assert manifest["path"] == str(acquisition.path)
    assert manifest["shape"] == list(shape)
    assert manifest["index_sizes"] == dict(zip(acquisition.axes, shape, strict=True))
    assert manifest["camera_offset"] == acquisition.camera_offset
    assert manifest["camera_conversion"] == acquisition.camera_conversion


@pytest.mark.parametrize("export_image_input", ["collection"], indirect=True)
@pytest.mark.parametrize(
    "acquisition_link",
    [
        "override",
        "journal",
        "relative",
        "moved",
        "copied",
        "windows-moved",
        "external",
        "relative-external",
        "missing",
        "ambiguous",
    ],
    indirect=True,
)
def test_projection_metadata_requires_an_unambiguous_source(acquisition_link):
    """Read recorded acquisition calibration from explicit or journal-linked paths."""
    link = acquisition_link
    if link.kind in ("missing", "ambiguous"):
        with pytest.raises(ValueError, match="one processing sidecar"):
            export_projections.acquisition_for(link.processed)
    else:
        metadata = export_projections.acquisition_for(
            link.processed, link.source if link.kind == "override" else None
        )
        assert metadata.path == link.source.resolve()
        assert metadata.shape == (1, 1, 1, 3, 4, 5)
        assert metadata.camera_offset == 100
        assert metadata.camera_conversion == 0.25


def test_estimation_worker_returns_annotated_errors_and_closes_pipe():
    """Return the original input error and its traceback through a real pipe."""
    import multiprocessing as mp
    from contextlib import closing

    from opm_processing.process import run_estimate_illuminations

    receiving, sending = mp.Pipe(duplex=False)
    with closing(receiving), closing(sending):
        invalid = ts.array(np.zeros((1, 1, 1, 4, 5), np.uint16))
        run_estimate_illuminations(invalid, 100, 0.25, None, None, None, sending)
        error = receiving.recv()
        assert isinstance(error, ValueError)
        assert any("Flatfield worker traceback" in note for note in error.__notes__)
        assert sending.closed
