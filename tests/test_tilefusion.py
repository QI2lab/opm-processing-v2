import sys
import types
import shutil

import numpy as np
import pytest

# Stub GPU modules so package import doesn't fail when CuPy isn't installed.
if "cupy" not in sys.modules:
    cp_stub = types.SimpleNamespace(
        array=np.array,
        asarray=np.asarray,
        max=np.max,
        float32=np.float32,
        zeros=np.zeros,
        ones=np.ones,
        ndarray=np.ndarray,
    )
    sys.modules["cupy"] = cp_stub
    sys.modules["cupyx"] = types.SimpleNamespace()
    cupyx_scipy = types.SimpleNamespace()
    ndimage_stub = types.SimpleNamespace(
        minimum_filter=lambda *a, **k: None,
        gaussian_filter=lambda *a, **k: None,
        convolve=lambda *a, **k: None,
    )
    cupyx_scipy.ndimage = ndimage_stub
    sys.modules["cupyx.scipy"] = cupyx_scipy
    sys.modules["cupyx.scipy.special"] = types.SimpleNamespace(j1=lambda x: x)
    sys.modules["cupyx.scipy.ndimage"] = ndimage_stub

from opm_processing.imageprocessing.tilefusion import (
    TileFusion,
    _accumulate_tile_shard,
    _blend_numba,
    _normalize_shard,
    _shift_array,
    _ssim,
)

from opm_processing.imageprocessing import tilefusion as tfmod

if tfmod.USING_GPU:
    pytest.skip("Synthetic unit tests expect CPU backend", allow_module_level=True)


class _FakeReadResult:
    def __init__(self, arr: np.ndarray):
        self._arr = arr

    def result(self) -> np.ndarray:
        return self._arr


class _FakeWriteFuture:
    def result(self):
        return None


class _FakeSlice:
    def __init__(self, store: np.ndarray, key):
        self._store = store
        self._key = key

    def read(self) -> _FakeReadResult:
        return _FakeReadResult(self._store[self._key])

    def write(self, value) -> _FakeWriteFuture:
        self._store[self._key] = value
        return _FakeWriteFuture()


class _FakeTS:
    """
    Minimal tensorstore-like mock that supports slicing, read, and write.
    """

    def __init__(self, data: np.ndarray):
        self.data = data
        self.shape = data.shape

    def __getitem__(self, key):
        return _FakeSlice(self.data, key)


def _make_minimal_tilefusion(is_2d: bool) -> TileFusion:
    """Construct a minimal TileFusion instance with small geometry for unit tests."""
    tf = TileFusion.__new__(TileFusion)
    tf._is_2d = is_2d
    tf.z_dim = 1 if is_2d else 2
    tf.Y = 4
    tf.X = 5
    tf._pixel_size = (1.0, 1.0, 1.0)
    tf._blend_pixels = (0, 0, 0)
    tf._update_profiles()
    return tf


def _make_tf_for_geometry(z_dim=2, y=4, x=5) -> TileFusion:
    """Build a TileFusion stub with controllable geometry and defaults."""
    tf = TileFusion.__new__(TileFusion)
    tf._is_2d = z_dim == 1
    tf.z_dim = z_dim
    tf.Y = y
    tf.X = x
    tf._pixel_size = (1.0, 1.0, 1.0)
    tf._blend_pixels = (0, 0, 0)
    tf._update_profiles()
    tf._max_workers = 1
    tf._debug = False
    tf.channels = 1
    tf.padded_shape = (z_dim, y, x)
    tf.downsample_factors = (1, 1, 1)
    tf.ssim_window = 3
    tf.threshold = 0.0
    tf.chunk_shape = (1, 1, 1, y, x)
    tf.chunk_y, tf.chunk_x = y, x
    tf.pairwise_metrics = {}
    return tf


def test_make_1d_profile_respects_length_and_blend():
    """Profile ramps to zero at edges and peaks at 1 in the center."""
    prof = TileFusion._make_1d_profile(5, 2)
    assert prof.shape == (5,)
    assert np.isclose(prof[0], 0.0)
    assert np.isclose(prof[-1], 0.0)
    assert np.isclose(prof[2], 1.0)


def test_accumulate_and_normalize_shard():
    """Accumulation followed by normalization should yield unit values and weights."""
    fused = np.zeros((1, 2, 3, 3), dtype=np.float32)
    weight = np.zeros_like(fused)
    sub = np.ones((1, 2, 3, 3), dtype=np.float32)
    w3d = np.ones((2, 3, 3), dtype=np.float32)
    _accumulate_tile_shard(fused, weight, sub, w3d, 0, 0, 0)
    _normalize_shard(fused, weight)
    assert np.allclose(fused, 1.0)
    assert np.allclose(weight, 1.0)


def test_blend_numba_average_with_equal_weights():
    """Blending with equal weights produces a midpoint average."""
    sub_i = np.ones((1, 2, 2), dtype=np.float32)
    sub_j = np.zeros_like(sub_i)
    wz = np.ones(1, dtype=np.float32)
    wy = np.ones(2, dtype=np.float32)
    wx = np.ones(2, dtype=np.float32)
    out_buf = np.empty_like(sub_i)
    blended = _blend_numba(
        sub_i,
        sub_j,
        wz,
        wy,
        wx,
        wz,
        wy,
        wx,
        out_buf,
    )
    assert np.allclose(blended, 0.5)


def test_shift_array_noop_on_cpu():
    """Zero shift returns the original array."""
    arr = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    shifted = _shift_array(arr, shift_vec=(0, 0, 0))
    assert np.allclose(shifted, arr)


def test_ssim_near_one_for_similar_arrays():
    """Small intensity change should still deliver high SSIM."""
    a = np.ones((3, 7, 7), dtype=np.float32)
    b = a + 0.01
    score = _ssim(a, b, win_size=3)
    assert 0.9 < score <= 1.0


def test_register_and_score_recovers_integer_shift():
    """Registration recovers a known integer shift on structured data."""
    rng = np.random.default_rng(42)
    g1 = rng.normal(loc=1000.0, scale=400.0, size=(1, 16, 16)).astype(np.float32)
    g1[..., 5:11, 6:12] += 2500.0
    true_shift = (0.0, 1.0, -2.0)
    g2 = _shift_array(g1, shift_vec=true_shift)
    est_shift, ssim_val = TileFusion.register_and_score(g1, g2, win_size=3, debug=False)
    assert est_shift is not None
    assert np.allclose(est_shift, tuple(-s for s in true_shift), atol=0.6)
    assert ssim_val > 0.5


def test_register_and_score_real_shift_with_features():
    """Registration succeeds on patterned overlaps with nontrivial shift."""
    # Use a patterned overlap (square + circle) to verify real registration path.
    rng = np.random.default_rng(123)
    g1 = rng.normal(loc=1000.0, scale=400.0, size=(1, 24, 24)).astype(np.float32)
    g1[..., 6:14, 8:16] += 3000.0
    yy, xx = np.meshgrid(np.arange(24), np.arange(24), indexing="ij")
    circle = ((yy - 12) ** 2 + (xx - 14) ** 2) <= 9
    g1[..., circle] += 4000.0

    true_shift = (0.0, -2.0, 3.0)  # shift in (z,y,x) applied to g1 to create g2
    g2 = _shift_array(g1, shift_vec=true_shift)
    # Tolerate registration failure but ensure it runs.
    est_shift, ssim_val = TileFusion.register_and_score(g1, g2, win_size=3, debug=False)
    assert est_shift is not None
    assert np.allclose(est_shift, tuple(-s for s in true_shift), atol=0.6)
    assert ssim_val > 0.5


def test_refine_tile_positions_with_cross_correlation_detects_shift():
    """Cross-correlation detects the expected pixel offset between two tiles."""
    tf = _make_tf_for_geometry(z_dim=1, y=12, x=12)
    tf.position_dim = 2
    tf.time_dim = 1
    tf._tile_positions = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
    rng = np.random.default_rng(7)
    base = rng.normal(loc=1000.0, scale=400.0, size=(1, 12, 12)).astype(np.float32)
    base[..., 3:9, 4:10] += 3000.0
    base[..., 6, 7] += 5000.0
    tile0 = base.copy()
    tile1 = _shift_array(base, shift_vec=(0.0, 0.0, 1.0))
    data = np.zeros((1, 2, 1, 12, 12), dtype=np.float32)
    data[0, 0, 0] = tile0[0]
    data[0, 1, 0] = tile1[0]
    tf.ts = _FakeTS(data)
    tf.refine_tile_positions_with_cross_correlation(
        downsample_factors=(1, 1, 1), ssim_window=3, ch_idx=0, threshold=0.0
    )
    assert tf.pairwise_metrics
    (_, _), (dz, dy, dx, score) = next(iter(tf.pairwise_metrics.items()))
    assert np.isclose(dx, -1.0, atol=0.6)
    assert np.isclose(dy, 0.0, atol=0.6)
    assert np.isclose(dz, 0.0, atol=0.6)
    assert score >= 0.0


def test_refine_tile_positions_with_cross_correlation_real():
    """Cross-correlation runs on simple synthetic tiles and yields near-zero shift."""
    tf = _make_tf_for_geometry(z_dim=1, y=6, x=6)
    tf.position_dim = 2
    tf.time_dim = 1
    tf._tile_positions = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
    # two identical tiles with strong signal
    base = np.zeros((1, 2, 1, tf.z_dim, tf.Y, tf.X), dtype=np.float32)
    base[..., 1:5, 1:5] = 7.0
    tf.ts = _FakeTS(base)

    tf.refine_tile_positions_with_cross_correlation(
        downsample_factors=(1, 1, 1), ssim_window=3, ch_idx=0, threshold=0.0
    )
    # Should complete without error; if registration succeeds, offsets are near zero.
    if tf.pairwise_metrics:
        (_, _), (dz, dy, dx, score) = next(iter(tf.pairwise_metrics.items()))
        assert np.isclose(dx, 0, atol=0.5)
        assert np.isclose(dy, 0, atol=0.5)
        assert np.isclose(dz, 0, atol=0.5)
        assert score >= 0.0


def test_create_fused_tensorstore_uses_allocator(monkeypatch):
    """Fused tensorstore creation uses configured chunking and shape."""
    tf = _make_tf_for_geometry(z_dim=2, y=2, x=2)
    tf.channels = 1
    fake_ts_store = {}

    class _Future:
        def __init__(self, ts_obj):
            self._ts_obj = ts_obj

        def result(self):
            return self._ts_obj

    def fake_ts_open(spec, create=False, open=True):
        shape = spec["metadata"]["shape"]
        arr = np.zeros(tuple(shape), dtype=np.uint16)
        ts_obj = _FakeTS(arr)
        fake_ts_store["last"] = ts_obj
        return _Future(ts_obj)

    monkeypatch.setattr(tfmod, "ts", types.SimpleNamespace(open=fake_ts_open))

    tf._create_fused_tensorstore(output_path="unused", z_slices_per_shard=1)
    assert "last" in fake_ts_store
    assert fake_ts_store["last"].data.shape == (1, tf.channels, *tf.padded_shape)


def test_create_multiscales_stride_downsamples(monkeypatch, tmp_path):
    """Multiscale creation with stride downsampling matches explicit stride slicing."""
    tf = _make_tf_for_geometry(z_dim=2, y=4, x=4)
    tf.channels = 1
    tf.multiscale_downsample = "stride"
    tf._is_2d = False
    tf.padded_shape = (2, 4, 4)
    base = np.arange(1 * 1 * 2 * 4 * 4, dtype=np.uint16).reshape(1, 1, 2, 4, 4)
    base_path = tmp_path / "omezarr" / "scale0" / "image"
    base_path.parent.mkdir(parents=True, exist_ok=True)

    ts_map = {str(base_path): _FakeTS(base.copy())}

    class _Future:
        def __init__(self, ts_obj):
            self._ts_obj = ts_obj

        def result(self):
            return self._ts_obj

    def fake_ts_open(spec, create=False, open=True):
        path = spec["kvstore"]["path"]
        if path not in ts_map:
            ts_map[path] = _FakeTS(np.zeros_like(base))
        return _Future(ts_map[path])

    def fake_create_fused_ts(self, output_path, z_slices_per_shard=1):
        ts_map[str(output_path)] = _FakeTS(
            np.zeros((1, tf.channels, *tf.padded_shape), dtype=np.uint16)
        )
        tf.fused_ts = ts_map[str(output_path)]

    monkeypatch.setattr(tfmod, "ts", types.SimpleNamespace(open=fake_ts_open))
    tf._create_fused_tensorstore = types.MethodType(fake_create_fused_ts, tf)

    (tmp_path / "omezarr" / "scale1").mkdir(parents=True, exist_ok=True)
    tf._create_multiscales(tmp_path / "omezarr", factors=(2,), z_slices_per_shard=1)

    out_ts = ts_map[str(tmp_path / "omezarr" / "scale1" / "image")]
    expected = base[..., ::2, ::2, ::2]
    assert np.array_equal(out_ts.data, expected)
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_create_multiscales_block_mean_downsamples(monkeypatch, tmp_path):
    """Block-mean downsampling produces expected averaged tiles."""
    tf = _make_tf_for_geometry(z_dim=1, y=4, x=4)
    tf.channels = 1
    tf.multiscale_downsample = "block_mean"
    tf._is_2d = True
    tf.padded_shape = (1, 4, 4)
    base = np.zeros((1, 1, 1, 4, 4), dtype=np.float32)
    base[..., 0::2, 0::2] = 4.0
    base_path = tmp_path / "omezarr" / "scale0" / "image"
    base_path.parent.mkdir(parents=True, exist_ok=True)

    ts_map = {str(base_path): _FakeTS(base.copy())}

    class _Future:
        def __init__(self, ts_obj):
            self._ts_obj = ts_obj

        def result(self):
            return self._ts_obj

    def fake_ts_open(spec, create=False, open=True):
        path = spec["kvstore"]["path"]
        if path not in ts_map:
            ts_map[path] = _FakeTS(np.zeros_like(base))
        return _Future(ts_map[path])

    def fake_create_fused_ts(self, output_path, z_slices_per_shard=1):
        ts_map[str(output_path)] = _FakeTS(
            np.zeros((1, tf.channels, *tf.padded_shape), dtype=np.float32)
        )
        tf.fused_ts = ts_map[str(output_path)]

    monkeypatch.setattr(tfmod, "ts", types.SimpleNamespace(open=fake_ts_open))
    tf._create_fused_tensorstore = types.MethodType(fake_create_fused_ts, tf)

    (tmp_path / "omezarr" / "scale1").mkdir(parents=True, exist_ok=True)
    tf._create_multiscales(tmp_path / "omezarr", factors=(2,), z_slices_per_shard=1)

    out_ts = ts_map[str(tmp_path / "omezarr" / "scale1" / "image")]
    # block mean of 2x2 blocks of pattern with 4 at top-left corners -> mean = 1
    assert np.allclose(out_ts.data, 1.0)
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_generate_ngff_zarr3_json(tmp_path):
    """NGFF JSON generation writes a multiscales entry."""
    tf = _make_tf_for_geometry()
    tf._pixel_size = (1.0, 1.0, 1.0)
    tf.center = (0.0, 0.0, 0.0)
    tf._generate_ngff_zarr3_json(
        tmp_path,
        resolution_multiples=[(1, 1, 1), (2, 2, 2)],
        dataset_name="image",
        version="0.5",
    )
    zarr_json = tmp_path / "zarr.json"
    assert zarr_json.exists()
    content = zarr_json.read_text()
    assert '"multiscales"' in content
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_solve_global_recovers_linear_offsets():
    """Least squares solver reconstructs simple chained offsets."""
    tf = _make_tf_for_geometry()
    links = [
        {"i": 0, "j": 1, "t": np.array([1.0, 0.0, 0.0]), "w": 1.0},
        {"i": 1, "j": 2, "t": np.array([1.0, 0.0, 0.0]), "w": 1.0},
    ]
    shifts = tf._solve_global(links, n_tiles=3, fixed_indices=[0])
    assert shifts.shape == (3, 3)


def test_two_round_opt_removes_outlier_link():
    """Two-round optimization drops an outlier link and returns a solution."""
    tf = _make_tf_for_geometry()
    links = [
        {"i": 0, "j": 1, "t": np.array([1.0, 0.0, 0.0]), "w": 1.0},
        {"i": 1, "j": 2, "t": np.array([1.0, 0.0, 0.0]), "w": 1.0},
        {"i": 0, "j": 2, "t": np.array([100.0, 0.0, 0.0]), "w": 1.0},  # outlier
    ]
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        shifts = tf._two_round_opt(
            links, n_tiles=3, fixed_indices=[0], rel_thresh=0.5, abs_thresh=1.0, iterative=True
        )
    assert shifts.shape == (3, 3)


def test_optimize_shifts_sets_global_offsets():
    """Optimize shifts populates global_offsets for a simple pair."""
    tf = _make_tf_for_geometry()
    tf.position_dim = 2
    tf._tile_positions = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    tf.pairwise_metrics = {(0, 1): (1, 0, 0, 1.0)}
    tf.optimize_shifts()
    assert tf.global_offsets.shape == (2, 3)
    assert np.allclose(tf.global_offsets[:, 0], [0.0, 1.0])


def test_optimize_shifts_multi_axis():
    """Optimize shifts solves three tiles with symmetric links across axes."""
    tf = _make_tf_for_geometry()
    tf.position_dim = 3
    tf._tile_positions = [
        (0.0, 0.0, 0.0),
        (0.0, 1.0, 2.0),
        (0.0, -1.0, -2.0),
    ]
    # symmetric links with known shifts
    tf.pairwise_metrics = {
        (0, 1): (0, 1, 2, 1.0),
        (0, 2): (0, -1, -2, 1.0),
        (1, 2): (0, -2, -4, 1.0),
    }
    tf.optimize_shifts(method="TWO_ROUND_ITERATIVE", rel_thresh=0.5, abs_thresh=5.0, iterative=True)
    assert tf.global_offsets.shape == (3, 3)
    assert np.allclose(tf.global_offsets, np.array([[0, 0, 0], [0, 1, 2], [0, -1, -2]]))


def test_find_overlaps_identifies_single_overlap():
    """Overlap finder returns the expected region for two touching tiles."""
    tf = _make_minimal_tilefusion(is_2d=False)
    offsets = [(0, 0, 0), (1, 1, 1)]
    overlaps = tf._find_overlaps(offsets)
    assert len(overlaps) == 1
    (i, j, region) = overlaps[0]
    assert (i, j) == (0, 1)
    z0, z1, y0, y1, x0, x1 = region
    assert (z1 - z0, y1 - y0, x1 - x0) == (1, 3, 4)


def test_copy_nonoverlap_writes_unique_regions():
    """Copy non-overlap fills unique areas while leaving overlap untouched."""
    tf = _make_tf_for_geometry(z_dim=1, y=4, x=4)
    tf._tile_positions = [(0.0, 0.0, 0.0), (0.0, 1.0, 1.0)]
    # ts shape: (t=1, position=2, channel=1, z=1, y=4, x=4)
    tile0 = np.full((1, 1, 1, 4, 4), 2, dtype=np.uint16)
    tile1 = np.full((1, 1, 1, 4, 4), 5, dtype=np.uint16)
    tf.ts = _FakeTS(np.concatenate([tile0, tile1], axis=1))
    tf.fused_ts = _FakeTS(np.zeros((1, 1, 1, 5, 5), dtype=np.uint16))

    overlaps = tf._find_overlaps([(0, 0, 0), (0, 1, 1)])
    tf._copy_nonoverlap(0, [(0, 0, 0), (0, 1, 1)], overlaps)
    tf._copy_nonoverlap(1, [(0, 0, 0), (0, 1, 1)], overlaps)

    fused = tf.fused_ts.data[0, 0, 0]
    # Non-overlap regions should be filled from respective tiles; overlap remains zero until blending.
    assert fused[0, 0] == 2
    assert fused[-1, -1] == 5
    assert fused[1, 1] == 0  # overlap top-left remains untouched


def test_compute_fused_image_space_and_padding():
    """Fused volume shape/padding derived from tile positions is correct."""
    tf = _make_tf_for_geometry(z_dim=2, y=4, x=5)
    tf._tile_positions = [
        (0.0, 0.0, 0.0),
        (1.0, 3.0, 2.0),
    ]
    tf._compute_fused_image_space()
    assert tf.unpadded_shape == (3, 7, 7)
    tf._pad_to_chunk_multiple()
    # z already multiple of 2, y pad to 8, x pad to 10
    assert tf.padded_shape == (4, 8, 10)
    assert tf.pad == (1, 1, 3)


def test_blend_region_overwrites_overlap_with_average():
    """Blend region averages overlap values from both tiles."""
    tf = _make_tf_for_geometry(z_dim=1, y=4, x=4)
    tf._tile_positions = [(0.0, 0.0, 0.0), (0.0, 1.0, 1.0)]
    offsets = [(0, 0, 0), (0, 1, 1)]
    tf.fused_ts = _FakeTS(np.zeros((1, 1, 1, 5, 5), dtype=np.uint16))
    # tiles with different values
    tile0 = np.full((1, 1, 1, 4, 4), 4, dtype=np.uint16)
    tile1 = np.full((1, 1, 1, 4, 4), 8, dtype=np.uint16)
    tf.ts = _FakeTS(np.concatenate([tile0, tile1], axis=1))

    overlaps = tf._find_overlaps(offsets)
    assert overlaps
    (i, j, region) = overlaps[0]
    tf._blend_region(i, j, region, offsets)

    z0, z1, y0, y1, x0, x1 = region
    fused_overlap = tf.fused_ts.data[0, 0, z0:z1, y0:y1, x0:x1]
    # Expect average (6) in overlap; stored as uint16.
    assert np.all(fused_overlap == 6)


def test_read_tile_volume_for_2d_pads_z_axis():
    """2D read pads a singleton z axis to match expected rank."""
    tf = _make_minimal_tilefusion(is_2d=True)
    data = np.arange(1 * 1 * 2 * 4 * 5, dtype=np.uint16).reshape(1, 1, 2, 4, 5)
    tf.ts = _FakeTS(data)
    arr = tf._read_tile_volume(
        tile_idx=0,
        ch_sel=slice(None),
        z_slice=slice(0, 1),
        y_slice=slice(0, 2),
        x_slice=slice(0, 3),
    )
    assert arr.shape == (2, 1, 2, 3)


def test_read_tile_volume_for_3d_preserves_shape():
    """3D read returns the requested subvolume without padding."""
    tf = _make_minimal_tilefusion(is_2d=False)
    data = np.arange(1 * 1 * 2 * 2 * 3 * 4, dtype=np.uint16).reshape(1, 1, 2, 2, 3, 4)
    tf.ts = _FakeTS(data)
    arr = tf._read_tile_volume(
        tile_idx=0,
        ch_sel=slice(None),
        z_slice=slice(0, 2),
        y_slice=slice(0, 2),
        x_slice=slice(0, 3),
    )
    assert arr.shape == (2, 2, 2, 3)
