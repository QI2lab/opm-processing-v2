import numpy as np
import pytest

from opm_processing.imageprocessing import opmtools


def test_chunk_indices_cover_exact_multiple_and_remainder():
    assert opmtools.chunk_indices(30_000, 15_000) == [
        (0, 15_000),
        (15_000, 30_000),
    ]
    assert opmtools.chunk_indices(31_000, 15_000) == [
        (0, 15_000),
        (15_000, 30_000),
        (30_000, 31_000),
    ]


def test_chunked_deskew_nests_y_only_deconvolution(monkeypatch):
    decon_calls = []
    deskew_calls = []

    monkeypatch.setattr(
        opmtools,
        "deskew_shape_estimator",
        lambda *_args, **_kwargs: ([4, 100, 8], 0, 0, 0),
    )

    def fake_decon(image, psf, crop_y):
        decon_calls.append((image.shape, psf.shape, crop_y))
        return image.astype(np.float32)

    def fake_deskew(image, *, downsample_factor):
        deskew_calls.append((image.shape, downsample_factor))
        return np.full((2, 100, 8), 7, dtype=np.uint16)

    monkeypatch.setattr(opmtools, "_deconvolve_oblique_chunk", fake_decon)
    monkeypatch.setattr(opmtools, "orthogonal_deskew", fake_deskew)

    result = opmtools.chunked_orthogonal_deskew(
        np.ones((10, 6, 8), dtype=np.uint16),
        psf_data=np.ones((3, 5, 3), dtype=np.float32),
        deconvolve=True,
        decon_chunk_size=4,
        chunk_size=15_000,
        scan_crop=0,
        z_downsample_level=2,
    )

    assert decon_calls == [((10, 6, 8), (3, 5, 3), 4)]
    assert deskew_calls == [((10, 6, 8), 2)]
    np.testing.assert_array_equal(result, 7)


def test_chunked_deskew_requires_psf_for_deconvolution():
    with pytest.raises(ValueError, match="psf_data is required"):
        opmtools.chunked_orthogonal_deskew(
            np.ones((2, 3, 4), dtype=np.uint16),
            deconvolve=True,
        )
