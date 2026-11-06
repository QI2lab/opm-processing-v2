"""Viewing copies average known display objects using actual native caches."""

import numpy as np
import pytest
from scripts.ao_tissue_preview import make_layer_previews

pytestmark = pytest.mark.unit


def test_area_mean_uses_each_layers_display_colors_and_real_edge_counts(
    ao_partial_preview, ao_memmap_reads
):
    """Normalize before averaging, with correct counts in partial edge bins."""
    first, second = ao_partial_preview
    ao_memmap_reads.clear()
    previews, factor = make_layer_previews(
        {2: first, 5: second}, {2: (100, 300), 5: (1000, 3000)}, max_edge=3
    )
    assert factor == 4
    expected = np.asarray([[16, 128], [255, 128]], dtype=np.uint8)
    np.testing.assert_array_equal(
        previews[2], np.repeat(expected[..., None], 3, axis=-1)
    )
    np.testing.assert_array_equal(previews[5], np.zeros((2, 2, 3), np.uint8))
    for source in (first, second):
        assert [
            shape for path, shape in ao_memmap_reads if path == source.filename
        ] == [(5, 7)]


def test_preview_preserves_components_and_bounded_reads_at_partial_edges(
    ao_checkerboard_preview, ao_memmap_reads
):
    """Gray and RGB objects retain their known averages across real cache blocks."""
    source, limits, expected = ao_checkerboard_preview
    ao_memmap_reads.clear()
    previews, factor = make_layer_previews({0: source}, limits, max_edge=600)
    assert factor == 2
    np.testing.assert_array_equal(previews[0], expected)
    suffix = (3,) if source.ndim == 3 else ()
    assert [shape for path, shape in ao_memmap_reads if path == source.filename] == [
        (*shape, *suffix) for shape in ((1024, 1024), (1024, 19), (7, 1024), (7, 19))
    ]
