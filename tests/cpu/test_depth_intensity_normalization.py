"""Depth gains are shared across XY fields and fitted separately per channel."""

from types import SimpleNamespace

import numpy as np
import pytest
import tensorstore as ts

from opm_processing.imageprocessing.tilefusion import TileFusion, _fit_depth_gains

pytestmark = pytest.mark.unit


def test_depth_gain_fit_is_robust_and_anchors_unobserved_depths():
    """Recover known gains despite an outlier and retain isolated depth anchors."""
    gains = _fit_depth_gains(
        4,
        [
            (0, 1, np.log(2)),
            (0, 1, np.log(2)),
            (0, 1, np.log(99)),
            (1, 2, np.log(0.5)),
        ],
    )
    np.testing.assert_allclose(gains, (1, 2, 1, 1))


def test_normalization_matches_depths_and_preserves_xy_brightness(
    fusion_operator,
):
    """Invert known Beer attenuation while preserving threefold XY contrast."""
    rng = np.random.default_rng(31)
    specimen = rng.uniform(100, 1000, (24, 64, 64)).astype(np.float32)
    arrays = []
    attenuation_per_um = np.log((2.0, 4.0)) / 8.0
    for depth, xy_gain in ((0, 1), (0, 3), (1, 1), (1, 3)):
        signal = specimen[depth * 8 : depth * 8 + 16] * xy_gain
        channels = np.stack(
            (
                signal * np.exp(-attenuation_per_um[0] * depth * 8),
                signal * np.exp(-attenuation_per_um[1] * depth * 8),
                np.zeros_like(signal),
            )
        )
        arrays.append(channels)
    fusion = fusion_operator(
        TileFusion,
        acquisition=SimpleNamespace(
            stage_positions_zxy=[(0, 0, 0), (0, 0, 100), (8, 0, 0), (8, 0, 100)]
        ),
        _tile_positions=[(0, 0, 0), (0, 0, 100), (8, 0, 0), (8, 0, 100)],
        _tile_source_position_indices=list(range(4)),
        _tile_time_indices=[0] * 4,
        _tile_shapes=[(16, 64, 64)] * 4,
        _pixel_size=(1, 1, 1),
        _is_2d=False,
        _variable_roi_tiles=False,
        position_dim=4,
        time_dim=1,
        channels=3,
        position_arrays=tuple(ts.array(a[None]) for a in arrays),
        _tile_support_zy=[np.ones((16, 64), dtype=bool)] * 4,
        _tile_channel_nonzero={(i, c): c < 2 for i in range(4) for c in range(3)},
        optimized_pairwise_metrics={(0, 2): (0, 0, 0, 1), (1, 3): (0, 0, 0, 1)},
        _depth_intensity_gains=np.ones((4, 3), dtype=np.float32),
    )
    fusion.estimate_depth_intensity_gains()
    np.testing.assert_allclose(
        fusion._depth_intensity_gains, [(1, 1, 1)] * 2 + [(2, 4, 1)] * 2, rtol=1e-6
    )
    corrected = [fusion._scale_depth_source(i, a) for i, a in enumerate(arrays)]
    np.testing.assert_allclose(corrected[0][:, 8:], corrected[2][:, :8], rtol=1e-6)
    np.testing.assert_allclose(corrected[1][:, 8:], corrected[3][:, :8], rtol=1e-6)
    np.testing.assert_allclose(corrected[1], corrected[0] * 3, rtol=1e-6)


def test_scaled_uint16_output_clips_and_float32_preserves_range(
    fusion_operator,
):
    """Check saturation, floating-point range, and immutable source intensities."""
    fusion = fusion_operator(
        TileFusion, _depth_intensity_gains=np.asarray([(2, 0.5)], dtype=np.float32)
    )
    source = np.asarray([40000, 1000], dtype=np.uint16).reshape(2, 1, 1, 1)
    scaled = fusion._scale_depth_source(0, source)
    fusion.output_dtype = np.dtype(np.uint16)
    np.testing.assert_array_equal(
        fusion._cast_fusion_values(scaled).ravel(), (65535, 500)
    )
    fusion.output_dtype = np.dtype(np.float32)
    assert fusion._cast_fusion_values(scaled) is scaled
    np.testing.assert_array_equal(
        fusion._cast_fusion_values(scaled).ravel(), (80000, 500)
    )
    fusion._depth_intensity_gains[:] = 1
    fusion.output_dtype = np.dtype(np.uint16)
    unscaled = fusion._cast_fusion_values(fusion._scale_depth_source(0, source))
    assert unscaled is source
    np.testing.assert_array_equal(unscaled.ravel(), (40000, 1000))
    np.testing.assert_array_equal(source.ravel(), (40000, 1000))
    coverage = np.asarray([[0.5]], np.float32)
    corrected = fusion._scale_depth_source(0, source, coverage_zy=coverage)
    np.testing.assert_array_equal(corrected.ravel(), (80000, 2000))
    np.testing.assert_array_equal(source.ravel(), (40000, 1000))
    fusion._depth_intensity_gains[:] = (2, 0.5)
    corrected = fusion._scale_depth_source(0, source, coverage_zy=coverage)
    np.testing.assert_array_equal(corrected.ravel(), (160000, 1000))
    np.testing.assert_array_equal(source.ravel(), (40000, 1000))


@pytest.mark.parametrize("direct", [False, True])
def test_fusion_source_prefers_deconvolution_unless_explicit(
    processed_directory_factory, direct
):
    """Directory discovery prefers deconvolution; an explicit plain store stays selected."""
    from opm_processing.imageprocessing.tilefusion import resolve_fusion_input

    directory = processed_directory_factory(
        "sample", labels=("deskewed", "decon_deskewed", "projection")
    )
    plain = directory / "sample_deskewed.ome.zarr"
    output, source, stem, raw = resolve_fusion_input(plain if direct else directory)
    assert (output, source, stem, raw) == (
        directory,
        plain if direct else directory / "sample_decon_deskewed.ome.zarr",
        "sample",
        None,
    )
