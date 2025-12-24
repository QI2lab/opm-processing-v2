import json
import shutil
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import tensorstore as ts

# Stub GPU modules so package import doesn't fail when CuPy isn't installed.
if "cupy" not in sys.modules:
    cp_stub = types.SimpleNamespace(
        array=np.array,
        asarray=np.asarray,
        zeros=np.zeros,
        ones=np.ones,
        float32=np.float32,
        ndarray=np.ndarray,
    )
    sys.modules["cupy"] = cp_stub
    sys.modules["cupyx"] = types.SimpleNamespace()
    ndimage_stub = types.SimpleNamespace(
        minimum_filter=lambda *a, **k: None,
        gaussian_filter=lambda *a, **k: None,
        convolve=lambda *a, **k: None,
    )
    cupyx_scipy = types.SimpleNamespace(ndimage=ndimage_stub)
    sys.modules["cupyx.scipy"] = cupyx_scipy
    sys.modules["cupyx.scipy.ndimage"] = ndimage_stub
    sys.modules["cupyx.scipy.special"] = types.SimpleNamespace(j1=lambda x: x)

from opm_processing.imageprocessing.tilefusion import TileFusion
from opm_processing.imageprocessing import tilefusion as tfmod


if tfmod.USING_GPU:
    pytest.skip("Integration tests expect CPU backend", allow_module_level=True)


def _per_index_metadata(time_dim: int, position_dim: int, stage_positions):
    meta = {}
    for t in range(time_dim):
        meta[str(t)] = {}
        for p in range(position_dim):
            meta[str(t)][str(p)] = {"0": {"stage_position": list(stage_positions[p])}}
    return meta


def _write_zarr_store(path: Path, data: np.ndarray, stage_positions, pixel_size=(1.0, 1.0, 1.0)) -> None:
    """
    Create a minimal Zarr3 store with required metadata and write the data.
    """
    path.mkdir(parents=True, exist_ok=True)
    dims = data.ndim
    if dims == 5:
        dim_names = ["t", "p", "c", "y", "x"]
    elif dims == 6:
        dim_names = ["t", "p", "c", "z", "y", "x"]
    else:
        raise ValueError("data must be rank 5 or 6")

    chunk_shape = [1] * dims
    # use small chunks to keep tests light
    for idx in range(3, dims):
        chunk_shape[idx] = min(4, data.shape[idx])

    metadata = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": list(data.shape),
        "data_type": "uint16",
        "fill_value": 0,
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": chunk_shape}},
        "chunk_key_encoding": {"name": "default"},
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "attributes": {
            "_ARRAY_DIMENSIONS": dim_names,
            "deskewed_voxel_size_um": list(pixel_size),
            "per_index_metadata": _per_index_metadata(data.shape[0], data.shape[1], stage_positions),
        },
    }

    config = {
        "driver": "zarr3",
        "kvstore": {"driver": "file", "path": str(path)},
        "metadata": metadata,
    }
    store = ts.open(config, create=True, open=True).result()
    store.write(data).result()

    with open(path / "zarr.json", "w") as f:
        json.dump(metadata, f, indent=2)


def _composite_expected(shape, tiles, offsets):
    """Average overlapping tiles into a composite fused volume."""
    expected = np.zeros(shape, dtype=np.float32)
    weight = np.zeros(shape, dtype=np.float32)
    for (oz, oy, ox), tile in zip(offsets, tiles):
        z, y, x = tile.shape
        expected[oz:oz + z, oy:oy + y, ox:ox + x] += tile
        weight[oz:oz + z, oy:oy + y, ox:ox + x] += 1
    return np.divide(expected, weight, out=np.zeros_like(expected), where=weight > 0).astype(np.uint16)


def _offsets_from_stage(stage_positions, pixel_size):
    """Convert stage positions (physical) into voxel offsets."""
    positions = [np.array(p, dtype=float) for p in stage_positions]
    min_pos = np.min(positions, axis=0)
    return [tuple(int(round((p - min_pos)[i] / pixel_size[i])) for i in range(3)) for p in positions]


def _read_ts_array(path: Path):
    """Helper to read a tensorstore-backed array from disk."""
    arr = ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": str(path)}}).result()
    return arr.read().result()


def test_run_pipeline_2d_full(tmp_path):
    """Integration: 2D, two tiles, verifies offsets, fused image, and multiscale."""
    # Build global 2D scene and extract two tiles with a known offset.
    content_delta = (0, -2, 2)  # injected jitter (z,y,x) relative to stage metadata
    rng = np.random.default_rng(0)
    global_im = rng.normal(loc=1000.0, scale=400.0, size=(16, 16)).astype(np.float32)
    # Add features on top of noise
    global_im[4:12, 5:13] += 3000  # square
    global_im[2, 3] += 5000  # asymmetry marker to aid registration
    global_im[6, 8] += 7000  # overlap-localized marker
    yy, xx = np.meshgrid(np.arange(16), np.arange(16), indexing="ij")
    circle = ((yy - 9) ** 2 + (xx - 11) ** 2) <= 9
    global_im[circle] += 4000
    global_im = np.clip(global_im, 0, 60000).astype(np.uint16)

    # Ideal tiles per stage positions (no jitter).
    ideal_tile0 = global_im[0:12, 0:12]
    ideal_tile1 = global_im[2:14, 2:14]

    stage_positions = [(0.0, 0.0, 0.0), (0.0, 2.0, 2.0)]
    # Jittered tiles fed to pipeline (tile1 captured at stage+delta).
    tile0 = ideal_tile0
    true_y1 = int(stage_positions[1][1] + content_delta[1])
    true_x1 = int(stage_positions[1][2] + content_delta[2])
    tile1 = global_im[true_y1 : true_y1 + ideal_tile1.shape[0], true_x1 : true_x1 + ideal_tile1.shape[1]]
    data = np.zeros((1, 2, 1, 12, 12), dtype=np.uint16)
    data[0, 0, 0] = tile0
    data[0, 1, 0] = tile1
    deskewed = tmp_path / "scene2d_deskewed.zarr"
    _write_zarr_store(deskewed, data, stage_positions)

    root = tmp_path / "scene2d.zarr"
    tf = TileFusion(
        root,
        blend_pixels=(0, 0, 0),
        downsample_factors=(1, 1, 1),
        ssim_window=3,
        threshold=0.2,
        multiscale_factors=(2,),
    )
    tf.run()

    fused_root = tmp_path / "scene2d_fused_deskewed.ome.zarr"
    fused_arr = _read_ts_array(fused_root / "scale0" / "image")
    fused_padded = fused_arr[0, 0]
    fused = fused_padded
    sz, sy, sx = tf.unpadded_shape
    fused = fused[:sz, :sy, :sx]

    offsets = _offsets_from_stage(tf._tile_positions, tf.pixel_size)
    expected = _composite_expected((1, sy, sx), [tile0[None, ...], tile1[None, ...]], offsets)
    print("pairwise_metrics (2D):", tf.pairwise_metrics)
    print("global_offsets (2D):", tf.global_offsets)
    assert np.array_equal(fused, expected)

    # verify global offsets recovered (should compensate content_delta)
    assert np.allclose(tf.global_offsets[0], (0, 0, 0), atol=1e-3)
    expect = content_delta
    assert np.allclose(tf.global_offsets[1], expect, atol=0.6)

    # Check first multiscale level (stride downsample by 2 in Y/X).
    scale1 = _read_ts_array(fused_root / "scale1" / "image")
    scale1 = scale1[0, 0]
    fused_down = fused_padded[::1, ::2, ::2]
    expected_scale1 = fused_down[: scale1.shape[0], : scale1.shape[1], : scale1.shape[2]]
    assert np.array_equal(scale1, expected_scale1)
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_run_pipeline_3d_full(tmp_path):
    """Integration: 3D, two tiles, verifies offsets, fused image, and multiscale."""
    # Build global 3D scene and extract two tiles with a known offset.
    content_delta = (0, -2, 2)  # injected jitter (z,y,x)
    zdim, ydim, xdim = 6, 16, 16
    rng = np.random.default_rng(1)
    global_vol = rng.normal(loc=1000.0, scale=400.0, size=(zdim, ydim, xdim)).astype(np.float32)
    global_vol[1:4, 4:12, 4:12] += 3000  # cube
    global_vol[0, 3, 4] += 5000  # asymmetry marker
    global_vol[2, 6, 8] += 7000  # overlap-localized marker
    zz, yy, xx = np.meshgrid(np.arange(zdim), np.arange(ydim), np.arange(xdim), indexing="ij")
    sphere = (zz - 3) ** 2 + (yy - 10) ** 2 + (xx - 11) ** 2 <= 9
    global_vol[sphere] += 4000
    global_vol = np.clip(global_vol, 0, 60000).astype(np.uint16)

    ideal_tile0 = global_vol[0:4, 0:12, 0:12]
    ideal_tile1 = global_vol[1:5, 2:14, 2:14]

    stage_positions = [(0.0, 0.0, 0.0), (1.0, 2.0, 2.0)]
    tile0 = ideal_tile0
    true_z1 = int(stage_positions[1][0] + content_delta[0])
    true_y1 = int(stage_positions[1][1] + content_delta[1])
    true_x1 = int(stage_positions[1][2] + content_delta[2])
    tile1 = global_vol[
        true_z1 : true_z1 + ideal_tile1.shape[0],
        true_y1 : true_y1 + ideal_tile1.shape[1],
        true_x1 : true_x1 + ideal_tile1.shape[2],
    ]
    data = np.zeros((1, 2, 1, 4, 12, 12), dtype=np.uint16)
    data[0, 0, 0] = tile0
    data[0, 1, 0] = tile1
    deskewed = tmp_path / "scene3d_deskewed.zarr"
    _write_zarr_store(deskewed, data, stage_positions)

    root = tmp_path / "scene3d.zarr"
    tf = TileFusion(
        root,
        blend_pixels=(0, 0, 0),
        downsample_factors=(1, 1, 1),
        ssim_window=3,
        threshold=0.2,
        multiscale_factors=(2,),
    )
    tf.run()

    fused_root = tmp_path / "scene3d_fused_deskewed.ome.zarr"
    fused_arr = _read_ts_array(fused_root / "scale0" / "image")
    fused_padded = fused_arr[0, 0]
    fused = fused_padded
    sz, sy, sx = tf.unpadded_shape
    fused = fused[:sz, :sy, :sx]

    offsets = _offsets_from_stage(tf._tile_positions, tf.pixel_size)
    expected = _composite_expected((sz, sy, sx), [tile0, tile1], offsets)
    print("pairwise_metrics (3D):", tf.pairwise_metrics)
    print("global_offsets (3D):", tf.global_offsets)
    assert np.array_equal(fused, expected)

    assert np.allclose(tf.global_offsets[0], (0, 0, 0), atol=1e-3)
    expect = content_delta
    assert np.allclose(tf.global_offsets[1], expect, atol=0.6)

    # Check first multiscale level (stride downsample by 2 in Z/Y/X).
    scale1 = _read_ts_array(fused_root / "scale1" / "image")
    scale1 = scale1[0, 0]
    fused_down = fused_padded[::2, ::2, ::2]
    expected_scale1 = fused_down[: scale1.shape[0], : scale1.shape[1], : scale1.shape[2]]
    assert np.array_equal(scale1, expected_scale1)
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_run_pipeline_2d_four_tiles(tmp_path):
    """Integration: 2D, four tiles, overlapping fiducials only in adjacent tiles."""
    content_deltas = [(0, -2, 0)] * 4
    rng = np.random.default_rng(11)
    yy, xx = np.meshgrid(np.arange(24), np.arange(24), indexing="ij")
    global_im = rng.normal(loc=1000.0, scale=400.0, size=(24, 24)).astype(np.float32)
    global_im += yy * 120.0 + xx * 60.0  # global gradient to stabilize correlation
    # Fiducials confined to adjacencies (vertical and horizontal bands)
    global_im[9:13, :] += 2500  # horizontal band for top/bottom adjacencies
    global_im[:, 9:13] += 2500  # vertical band for left/right adjacencies
    # Additional localized features per overlap (only in adjacent overlaps)
    global_im[6:10, 10:14] += 8000  # top-right (tile0/tile2)
    global_im[10:14, 6:10] += 8200  # top-left (tile0/tile1)
    global_im[14:18, 10:14] += 7800  # bottom-right (tile3/tile1)
    global_im[10:14, 14:18] += 7600  # bottom-left (tile3/tile2)
    # Unique point markers in each overlap to break symmetry
    global_im[8, 11] += 9000   # tile0/tile2
    global_im[12, 7] += 9200   # tile0/tile1
    global_im[15, 11] += 9100  # tile3/tile1
    global_im[12, 15] += 9300  # tile3/tile2
    global_im = np.clip(global_im, 0, 60000).astype(np.uint16)

    tile_size = 12
    stages = [
        (0.0, 0.0, 0.0),
        (0.0, 4.0, 0.0),   # shift y
        (0.0, 0.0, 4.0),   # shift x
        (0.0, 4.0, 4.0),
    ]
    tiles = []
    for _, sy, sx in stages:
        sy, sx = int(sy), int(sx)
        tiles.append(global_im[sy:sy + tile_size, sx:sx + tile_size])

    jittered = []
    for t, delta in zip(tiles, content_deltas):
        sy = int(delta[1])
        sx = int(delta[2])
        # pad then re-crop to apply jitter without changing shape
        pad_y, pad_x = abs(sy), abs(sx)
        padded = np.pad(t, ((pad_y, pad_y), (pad_x, pad_x)), mode="constant")
        y0 = pad_y - sy
        x0 = pad_x - sx
        jittered.append(padded[y0:y0 + tile_size, x0:x0 + tile_size])

    data = np.zeros((1, 4, 1, tile_size, tile_size), dtype=np.uint16)
    for i, t in enumerate(jittered):
        data[0, i, 0] = t

    deskewed = tmp_path / "scene2d_four_deskewed.zarr"
    _write_zarr_store(deskewed, data, stages)

    root = tmp_path / "scene2d_four.zarr"
    tf = TileFusion(
        root,
        blend_pixels=(0, 0, 0),
        downsample_factors=(1, 1, 1),
        ssim_window=3,
        threshold=0.0,
        multiscale_factors=(2,),
    )
    tf.run()

    fused_root = tmp_path / "scene2d_four_fused_deskewed.ome.zarr"
    fused_arr = _read_ts_array(fused_root / "scale0" / "image")
    fused_padded = fused_arr[0, 0]
    fused = fused_padded[: tf.unpadded_shape[0], : tf.unpadded_shape[1], : tf.unpadded_shape[2]]

    offsets = _offsets_from_stage(tf._tile_positions, tf.pixel_size)
    expected = _composite_expected(fused.shape, [t[None, ...] for t in jittered], offsets)
    print("pairwise_metrics 2D four:", tf.pairwise_metrics)
    print("global_offsets 2D four:", tf.global_offsets)
    print("unpadded_shape 2D four:", tf.unpadded_shape)
    print("offsets 2D four:", offsets)
    print("tile_positions 2D four:", tf._tile_positions)
    mask = expected > 0
    assert np.array_equal(mask, fused > 0)
    assert np.allclose(fused[mask], expected[mask], atol=1.0)
    assert tf.global_offsets.shape == (4, 3)
    rel_offsets = [np.subtract(content_deltas[0], d) for d in content_deltas]
    assert np.allclose(tf.global_offsets, rel_offsets, atol=0.6)
    shutil.rmtree(tmp_path, ignore_errors=True)
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_run_pipeline_3d_four_tiles(tmp_path):
    """Integration: 3D, four tiles, overlapping fiducials only in adjacent tiles."""
    content_deltas = [(0, -2, 0)] * 4
    rng = np.random.default_rng(13)
    zdim, ydim, xdim = 6, 24, 24
    zz, yy, xx = np.meshgrid(np.arange(zdim), np.arange(ydim), np.arange(xdim), indexing="ij")
    global_vol = rng.normal(loc=1000.0, scale=400.0, size=(zdim, ydim, xdim)).astype(np.float32)
    global_vol += zz * 80.0 + yy * 120.0 + xx * 60.0
    # Fiducials limited to adjacent overlaps
    global_vol[:, 9:13, :] += 2500  # horizontal band
    global_vol[:, :, 9:13] += 2500  # vertical band
    global_vol[:, 6:10, 10:14] += 8000  # top-right
    global_vol[:, 10:14, 6:10] += 8200  # top-left
    global_vol[:, 14:18, 10:14] += 7800  # bottom-right
    global_vol[:, 10:14, 14:18] += 7600  # bottom-left
    global_vol[:, 8, 11] += 9000
    global_vol[:, 12, 7] += 9200
    global_vol[:, 15, 11] += 9100
    global_vol[:, 12, 15] += 9300
    global_vol = np.clip(global_vol, 0, 60000).astype(np.uint16)

    tile_z, tile_y, tile_x = 4, 12, 12
    stages = [
        (0.0, 0.0, 0.0),
        (1.0, 4.0, 0.0),
        (0.0, 0.0, 4.0),
        (1.0, 4.0, 4.0),
    ]
    tiles = []
    for sz, sy, sx in stages:
        sz, sy, sx = int(sz), int(sy), int(sx)
        tiles.append(global_vol[sz:sz + tile_z, sy:sy + tile_y, sx:sx + tile_x])

    jittered = []
    for t, delta in zip(tiles, content_deltas):
        sz, sy, sx = delta
        pad_z, pad_y, pad_x = abs(sz), abs(sy), abs(sx)
        padded = np.pad(t, ((pad_z, pad_z), (pad_y, pad_y), (pad_x, pad_x)), mode="constant")
        z0 = pad_z - sz
        y0 = pad_y - sy
        x0 = pad_x - sx
        jittered.append(padded[z0:z0 + tile_z, y0:y0 + tile_y, x0:x0 + tile_x])

    data = np.zeros((1, 4, 1, tile_z, tile_y, tile_x), dtype=np.uint16)
    for i, t in enumerate(jittered):
        data[0, i, 0] = t

    deskewed = tmp_path / "scene3d_four_deskewed.zarr"
    _write_zarr_store(deskewed, data, stages)

    root = tmp_path / "scene3d_four.zarr"
    tf = TileFusion(
        root,
        blend_pixels=(0, 0, 0),
        downsample_factors=(1, 1, 1),
        ssim_window=3,
        threshold=0.0,
        multiscale_factors=(2,),
    )
    tf.run()

    fused_root = tmp_path / "scene3d_four_fused_deskewed.ome.zarr"
    fused_arr = _read_ts_array(fused_root / "scale0" / "image")
    fused_padded = fused_arr[0, 0]
    fused = fused_padded[: tf.unpadded_shape[0], : tf.unpadded_shape[1], : tf.unpadded_shape[2]]

    offsets = _offsets_from_stage(tf._tile_positions, tf.pixel_size)
    expected = _composite_expected(fused.shape, jittered, offsets)
    print("pairwise_metrics 3D four:", tf.pairwise_metrics)
    print("global_offsets 3D four:", tf.global_offsets)
    print("unpadded_shape 3D four:", tf.unpadded_shape)
    print("offsets 3D four:", offsets)
    print("tile_positions 3D four:", tf._tile_positions)
    mask = expected > 0
    assert np.array_equal(mask, fused > 0)
    assert np.allclose(fused[mask], expected[mask], atol=1.0)
    assert tf.global_offsets.shape == (4, 3)
    rel_offsets = [np.subtract(content_deltas[0], d) for d in content_deltas]
    assert np.allclose(tf.global_offsets, rel_offsets, atol=0.6)
    shutil.rmtree(tmp_path, ignore_errors=True)
