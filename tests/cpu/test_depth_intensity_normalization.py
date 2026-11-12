"""Processed depth histograms match their reference without changing XY contrast."""

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def test_normalization_matches_depths_and_preserves_xy_brightness(
    tile_fusion,
    aperiodic_registration_object,
    monkeypatch,
):
    """Match attenuated copies while preserving threefold XY contrast."""
    monkeypatch.setattr("opm_processing.imageprocessing.tilefusion.USING_GPU", False)
    specimen = aperiodic_registration_object((24, 64, 64), seed=31).astype(np.uint16)
    # Multiples of four keep both attenuated channels exactly representable.
    specimen = specimen // 4 * 4
    arrays = []
    for depth, xy_brightness in ((0, 1), (0, 3), (1, 1), (1, 3)):
        signal = specimen[:16] * xy_brightness
        channels = np.stack(
            (
                signal // (2**depth),
                signal // (4**depth),
                np.zeros_like(signal),
            )
        )
        arrays.append(channels)
    fusion = tile_fusion(
        arrays,
        [(0, 0, 0), (0, 0, 100), (8, 0, 0), (8, 0, 100)],
    )
    fusion._tile_support_zy = [np.ones((16, 64), dtype=bool)] * 4
    fusion.match_depth_histograms()
    corrected = [
        fusion._render_fusion_block(0, (0, 16, 0, 64, 0, 64), [(tile, (0, 0, 0))])[1]
        for tile in range(4)
    ]
    np.testing.assert_allclose(corrected[0], corrected[2], rtol=1e-6)
    np.testing.assert_allclose(corrected[1], corrected[3], rtol=1e-6)
    np.testing.assert_allclose(corrected[1], corrected[0] * 3, rtol=1e-6)


def test_coverage_normalization_and_output_range(
    tile_fusion,
    monkeypatch,
):
    """Check uint16 saturation, coverage, and immutable source intensities."""
    from opm_processing.imageprocessing import tilefusion

    monkeypatch.setattr(tilefusion, "USING_GPU", False)
    fusion = tile_fusion([np.ones((2, 1, 1, 1), np.uint16)], [(0, 0, 0)])
    source = np.asarray([40000, 1000], dtype=np.uint16).reshape(2, 1, 1, 1)
    coverage = np.asarray([[0.5]], np.float32)
    corrected = fusion._normalize_coverage(source, coverage)
    np.testing.assert_array_equal(corrected.ravel(), (80000, 2000))
    np.testing.assert_array_equal(
        fusion._cast_fusion_values(corrected).ravel(), (65535, 2000)
    )
    unchanged = fusion._normalize_coverage(source, np.ones((1, 1), np.float32))
    assert fusion._cast_fusion_values(unchanged) is source
    np.testing.assert_array_equal(source.ravel(), (40000, 1000))
    mapping, _ = tilefusion._depth_histogram_mapping(
        np.array([10, 10, 30, 30], np.uint16),
        np.array([100, 100, 200, 200], np.uint16),
    )
    fusion._depth_histogram_maps[0, 0] = mapping
    # Unseen uint16 intensities interpolate; endpoints clamp.
    values = np.array([0, 9, 20, 65535, 30], np.uint16).reshape(1, 1, 1, 5)
    weights = np.array([1, 1, 1, 1, 0], np.float32).reshape(values.shape)
    matched = fusion._match_fusion_depth(0, values, weights, channel=0)
    np.testing.assert_array_equal(matched.ravel(), [100, 100, 150, 200, 0])


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
