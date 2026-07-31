"""Numerical unit and storage integration tests for tile fusion."""

from types import MethodType, SimpleNamespace

import numpy as np
import pytest
from skimage.measure import block_reduce as block_reduce_cpu
from typer.testing import CliRunner
from yaozarrs import open_group, v05
from yaozarrs.write.v05 import prepare_image

from opm_processing.fuse import app as fuse_app
from opm_processing.imageprocessing import tilefusion as tilefusion_module
from opm_processing.imageprocessing.tilefusion import TileFusion


class _ImmediateFuture:
    """Completed write compatible with the fusion writer."""

    def result(self):
        """Return after an already-completed in-memory write."""


class _ArrayView:
    """Writable selection into an in-memory array."""

    def __init__(self, store, key):
        self.store = store
        self.key = key

    def write(self, value):
        """Write the selected values exactly."""
        self.store.data[self.key] = value
        return _ImmediateFuture()


class _ArrayStore:
    """Small in-memory TensorStore substitute for numerical fusion."""

    def __init__(self, shape):
        self.data = np.zeros(shape, dtype=np.uint16)

    def __getitem__(self, key):
        return _ArrayView(self, key)


def test_registration_pairs_include_every_physical_overlap() -> None:
    """Match main by retaining face, edge, and corner overlaps."""
    positions = [(z, y, x) for z in (0.0, 8.0) for y in (0.0, 8.0) for x in (0.0, 8.0)]

    pairs = tilefusion_module._select_registration_pairs(
        positions,
        tile_shape_zyx=(10, 10, 10),
        pixel_size_zyx=(1.0, 1.0, 1.0),
    )

    expected = {
        (i, j)
        for i in range(len(positions))
        for j in range(i + 1, len(positions))
        if np.all(np.abs(np.asarray(positions[j]) - np.asarray(positions[i])) < 10.0)
    }
    assert set(pairs) == expected


def test_block_fusion_does_not_infer_support_from_pixel_values(monkeypatch):
    """Include legitimate zero-valued pixels when geometry marks them valid."""
    fusion = TileFusion.__new__(TileFusion)
    fusion.fused_ts = _ArrayStore((1, 1, 2, 3, 6))
    fusion.write_block_shape = [1, 1, 1, 2, 2]
    fusion.offset_um = (0.0, 0.0, 0.0)
    fusion.padded_shape = (2, 3, 6)
    fusion._pixel_size = (1.0, 1.0, 1.0)
    fusion._tile_positions = [(0.0, 0.0, 0.0), (0.0, 0.0, 2.0)]
    fusion.time_dim = 1
    fusion.position_dim = 2
    fusion.channels = 1
    fusion.z_dim = 2
    fusion.y_dim = 3
    fusion.x_dim = 4
    fusion.z_profile = np.ones(2, dtype=np.float32)
    fusion.y_profile = np.ones(3, dtype=np.float32)
    fusion.x_profile = np.ones(4, dtype=np.float32)
    fusion.chunk_y = 2
    fusion.chunk_x = 2
    fusion.fusion_ram_fraction = 1.0
    fusion.max_in_flight_writes = 2
    fusion._max_workers = 2
    fusion._debug = False
    fusion._deskew_support_zy = np.ones((2, 3), dtype=bool)
    tiles = [
        np.full((1, 2, 3, 4), 10.0, dtype=np.float32),
        np.full((1, 2, 3, 4), 30.0, dtype=np.float32),
    ]
    # These dark pixels are inside valid geometry and must retain their weight.
    tiles[1][:, 0, :2, :2] = 0
    tiles[1][:, 1, :1, :2] = 0

    def read_tile(self, tile_idx, ch_sel, z_slice, y_slice, x_slice, dtype=np.float32):
        selected = tiles[tile_idx][ch_sel, z_slice, y_slice, x_slice]
        return selected if dtype is None else selected.astype(dtype, copy=False)

    fusion._read_tile_volume = MethodType(read_tile, fusion)
    monkeypatch.setattr(
        tilefusion_module.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=256),
    )

    fusion._fuse_by_blocks()

    expected = np.full((1, 1, 2, 3, 6), 10, dtype=np.uint16)
    expected[..., 2:4] = 20
    expected[..., 4:6] = 30
    expected[:, :, 0, :2, 2:4] = 5
    expected[:, :, 1, :1, 2:4] = 5
    np.testing.assert_array_equal(fusion.fused_ts.data, expected)


def test_block_fusion_has_no_zero_weight_line_at_masked_wedge_boundary(
    monkeypatch,
):
    """Keep constant signal continuous where a deskew wedge meets a tile edge."""
    fusion = TileFusion.__new__(TileFusion)
    fusion.fused_ts = _ArrayStore((1, 1, 2, 6, 3))
    fusion.write_block_shape = [1, 1, 2, 6, 3]
    fusion.offset_um = (0.0, 0.0, 0.0)
    fusion.padded_shape = (2, 6, 3)
    fusion._pixel_size = (1.0, 1.0, 1.0)
    fusion._tile_positions = [(0.0, 0.0, 0.0), (0.0, 2.0, 0.0)]
    fusion.time_dim = 1
    fusion.position_dim = 2
    fusion.channels = 1
    fusion.z_dim = 2
    fusion.y_dim = 4
    fusion.x_dim = 3
    fusion.z_profile = np.ones(2, dtype=np.float32)
    fusion.y_profile = fusion._make_1d_profile(4, 2)
    fusion.x_profile = np.ones(3, dtype=np.float32)
    fusion.chunk_y = 6
    fusion.chunk_x = 3
    fusion.fusion_ram_fraction = 1.0
    fusion.max_in_flight_writes = 1
    fusion._max_workers = 1
    fusion._debug = False
    fusion._deskew_support_zy = np.asarray(
        [[False, False, True, True], [False, True, True, True]],
        dtype=bool,
    )
    tiles = [
        np.full((1, 2, 4, 3), 100.0, dtype=np.float32),
        np.full((1, 2, 4, 3), 100.0, dtype=np.float32),
    ]
    for tile in tiles:
        tile[:, 0, :2, :] = 0
        tile[:, 1, :1, :] = 0

    def read_tile(self, tile_idx, ch_sel, z_slice, y_slice, x_slice, dtype=np.float32):
        selected = tiles[tile_idx][ch_sel, z_slice, y_slice, x_slice]
        return selected if dtype is None else selected.astype(dtype, copy=False)

    fusion._read_tile_volume = MethodType(read_tile, fusion)
    monkeypatch.setattr(
        tilefusion_module.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=4096),
    )

    fusion._fuse_by_blocks()

    expected = np.full((1, 1, 2, 6, 3), 100, dtype=np.uint16)
    expected[:, :, 0, :2] = 0
    expected[:, :, 1, :1] = 0
    np.testing.assert_array_equal(fusion.fused_ts.data, expected)


@pytest.mark.parametrize(
    ("downsample_method", "is_2d", "padded_shape"),
    (
        ("stride", False, (4, 8, 8)),
        ("block_mean", False, (4, 8, 8)),
        ("stride", True, (1, 8, 8)),
    ),
    ids=("volume-stride", "volume-block-mean", "projection-stride"),
)
def test_multiscale_storage_round_trip_matches_reference(
    tmp_path,
    downsample_method,
    is_2d,
    padded_shape,
):
    """Reopen exact pyramid data and verify its common physical-space center."""
    fusion = TileFusion.__new__(TileFusion)
    fusion.padded_shape = padded_shape
    fusion.offset_um = (4.5, -3.0, 12.25)
    fusion._pixel_size = (0.8, 0.5, 0.25)
    fusion.time_dim = 1
    fusion.channels = 1
    fusion.multiscale_factors = (2, 4)
    fusion._is_2d = is_2d
    fusion.chunk_y = 4
    fusion.chunk_x = 4
    fusion.multiscale_downsample = downsample_method
    fusion._max_workers = 2
    fusion.fusion_ram_fraction = 0.4
    fusion.max_in_flight_writes = 2
    fusion._input_chunk_zyx = (2, 4, 4)

    path = tmp_path / "multiscale.ome.zarr"
    scale0, fusion.write_block_shape = fusion._prepare_fused_image(path)
    source_shape = (1, 1, *padded_shape)
    source = np.arange(np.prod(source_shape), dtype=np.uint16).reshape(source_shape)
    scale0.write(source).result()
    fusion._write_multiscales()

    reopened = open_group(path)
    metadata = reopened.ome_metadata()
    assert metadata is not None
    datasets = metadata.multiscales[0].datasets
    assert [dataset.path for dataset in datasets] == ["0", "1", "2"]

    expected = source
    base_center = (
        np.asarray(fusion.offset_um)
        + (np.asarray(padded_shape, dtype=np.float64) - 1.0)
        * np.asarray(fusion._pixel_size)
        / 2.0
    )
    for absolute_factor, dataset in zip((1, 2, 4), datasets):
        z_factor = 1 if is_2d else absolute_factor
        expected_scale = np.array(
            [
                1.0,
                1.0,
                fusion._pixel_size[0] * z_factor,
                fusion._pixel_size[1] * absolute_factor,
                fusion._pixel_size[2] * absolute_factor,
            ]
        )
        expected_translation = np.array(
            [
                0.0,
                0.0,
                fusion.offset_um[0] + 0.5 * (z_factor - 1) * fusion._pixel_size[0],
                fusion.offset_um[1]
                + 0.5 * (absolute_factor - 1) * fusion._pixel_size[1],
                fusion.offset_um[2]
                + 0.5 * (absolute_factor - 1) * fusion._pixel_size[2],
            ]
        )
        np.testing.assert_allclose(
            dataset.scale_transform.scale,
            expected_scale,
        )
        assert dataset.translation_transform is not None
        np.testing.assert_allclose(
            dataset.translation_transform.translation,
            expected_translation,
        )

        reopened_array = reopened[dataset.path].to_tensorstore()
        spatial_shape = np.asarray(reopened_array.shape[-3:], dtype=np.float64)
        physical_center = (
            expected_translation[-3:]
            + (spatial_shape - 1.0) * expected_scale[-3:] / 2.0
        )
        np.testing.assert_allclose(physical_center, base_center)

        if dataset.path == "0":
            np.testing.assert_array_equal(
                reopened_array.read().result(),
                expected,
            )
            continue

        z_relative_factor = 1 if is_2d else 2
        if downsample_method == "stride":
            expected = expected[..., ::z_relative_factor, ::2, ::2]
        else:
            expected = block_reduce_cpu(
                expected,
                block_size=(1, 1, z_relative_factor, 2, 2),
                func=np.mean,
            ).astype(np.uint16)
        np.testing.assert_array_equal(
            reopened_array.read().result(),
            expected,
        )


def test_regenerate_max_z_flag_projects_registered_scale_zero_round_trip(
    tmp_path,
) -> None:
    """Overwrite a fused max-Z image and validate exact pixels and coordinates."""
    raw_path = tmp_path / "sample.zarr"
    fused_path = tmp_path / "sample_fused.ome.zarr"
    output_path = tmp_path / "sample_max_z_fused.ome.zarr"
    shape = (2, 2, 5, 7, 9)
    scale = [1.0, 1.0, 0.8, 0.5, 0.25]
    translation = [0.0, 0.0, 4.5, -3.0, 12.25]
    image = v05.Image(
        multiscales=[
            v05.Multiscale(
                name="registered-fused",
                axes=[
                    {"name": "t", "type": "time"},
                    {"name": "c", "type": "channel"},
                    {"name": "z", "type": "space", "unit": "micrometer"},
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ],
                datasets=[
                    v05.Dataset(
                        path="0",
                        coordinateTransformations=[
                            v05.ScaleTransformation(scale=scale),
                            v05.TranslationTransformation(translation=translation),
                        ],
                    )
                ],
            )
        ]
    )

    _, raw_arrays = prepare_image(
        raw_path,
        image,
        (shape, np.uint16),
        chunks=(1, 1, 2, 4, 5),
        writer="tensorstore",
        overwrite=True,
    )
    raw_arrays["0"].write(np.zeros(shape, dtype=np.uint16)).result()

    rng = np.random.default_rng(281)
    registered = rng.integers(0, 60_000, size=shape, dtype=np.uint16)
    _, fused_arrays = prepare_image(
        fused_path,
        image,
        (shape, np.uint16),
        extra_attributes={"opm_fusion": {"registered": True}},
        chunks=(1, 1, 2, 4, 5),
        writer="tensorstore",
        overwrite=True,
    )
    fused_arrays["0"].write(registered).result()

    sentinel_shape = (shape[0], shape[1], 1, shape[3], shape[4])
    _, sentinel_arrays = prepare_image(
        output_path,
        image,
        (sentinel_shape, np.uint16),
        chunks=(1, 1, 1, 4, 5),
        writer="tensorstore",
        overwrite=True,
    )
    sentinel_arrays["0"].write(np.zeros(sentinel_shape, dtype=np.uint16)).result()
    del raw_arrays, fused_arrays, sentinel_arrays

    result = CliRunner().invoke(
        fuse_app,
        [str(raw_path), "--regenerate-max-z", "--max-workers", "2"],
    )
    assert result.exit_code == 0, result.output

    reopened = open_group(output_path)
    actual = reopened["0"].to_tensorstore().read().result()
    np.testing.assert_array_equal(actual, registered.max(axis=2, keepdims=True))

    metadata = reopened.ome_metadata()
    assert isinstance(metadata, v05.Image)
    dataset = metadata.multiscales[0].datasets[0]
    np.testing.assert_allclose(dataset.scale_transform.scale, scale)
    assert dataset.translation_transform is not None
    expected_translation = translation.copy()
    expected_translation[2] += 0.5 * (shape[2] - 1) * scale[2]
    np.testing.assert_allclose(
        dataset.translation_transform.translation,
        expected_translation,
    )
    assert reopened.attrs["opm_fusion"]["maximum_projection"] == {
        "axis": "z",
        "source": str(fused_path.resolve()),
        "source_dataset": "0",
        "full_resolution": True,
        "registered": True,
    }
