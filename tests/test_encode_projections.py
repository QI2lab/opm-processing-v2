"""Physical intensity and frame-preservation checks for NVENC projection movies."""

import imageio_ffmpeg
import numpy as np
import pytest
import tifffile

from opm_processing.encode_projections import (
    encode_sequence,
    find_sequences,
    image_nv12,
)


@pytest.mark.unit
def test_nv12_preserves_gray_calibration_and_padding():
    """Black, midpoint and white map to video-range luma with neutral chroma."""
    frame = np.array([[0, 128, 255]], dtype=np.uint8)
    buffer, width, height = image_nv12(frame)
    planes = buffer.reshape(height * 3 // 2, width)
    np.testing.assert_array_equal(planes[0, :3], [16, 126, 235])
    assert np.all(planes[1:height] == 16)
    assert np.all(planes[height:] == 128)


@pytest.mark.unit
def test_sequence_discovery_keeps_positions_separate(tmp_path):
    """Sort by numeric index and reject missing timepoints instead of shortening time."""
    for pos in ["p000", "p001"]:
        folder = tmp_path / pos / "c000"
        folder.mkdir(parents=True)
        for t in [10, 9, 8]:
            (folder / f"sample_t{t}.tiff").touch()
    sequences = find_sequences(tmp_path)
    assert len(sequences) == 2
    assert [p.name for p in sequences[0]] == [
        "sample_t8.tiff",
        "sample_t9.tiff",
        "sample_t10.tiff",
    ]
    (tmp_path / "p000/c000/sample_t12.tiff").touch()
    with pytest.raises(ValueError, match="missing"):
        find_sequences(tmp_path)


@pytest.mark.integration
@pytest.mark.gpu
def test_nvenc_round_trip_moving_gaussian(tmp_path):
    """A translating Gaussian PSF and calibrated ramp retain order, geometry and intensity."""
    pytest.importorskip("PyNvVideoCodec")
    y, x = np.mgrid[:255, :321]
    originals, paths = [], []
    for t in range(8):
        frame = np.rint(
            255 * np.exp(-((x - 80 - t * 15) ** 2 + (y - 100) ** 2) / (2 * 20**2))
        ).astype(np.uint8)
        frame[210:230, 30:290] = np.linspace(0, 255, 260).astype(np.uint8)
        path = tmp_path / f"psf_t{t:04d}.tiff"
        tifffile.imwrite(
            path,
            frame,
            metadata={"volume_interval_ms": 85.0, "display_pixel_size_um": 0.1},
        )
        originals.append(frame)
        paths.append(path)
    output = encode_sequence(paths, tmp_path / "psf.mp4")
    decoded = imageio_ffmpeg.read_frames(
        str(output), pix_fmt="rgb24", output_params=["-vsync", "0"]
    )
    metadata = next(decoded)
    assert metadata["codec"] == "h264"
    assert metadata["pix_fmt"].startswith("yuv420p")
    assert metadata["fps"] == pytest.approx(200 / 17, abs=0.01)
    import PyNvVideoCodec as nvc

    assert nvc.CreateDemuxer(filename=str(output)).FrameRate() == pytest.approx(
        200 / 17
    )
    assert metadata["size"] == (322, 256)
    frames = list(decoded)
    assert len(frames) == len(originals)
    for raw, reference in zip(frames, originals, strict=False):
        rgb = np.frombuffer(raw, np.uint8).reshape(256, 322, 3)
        restored = rgb[:255, :321, 0].astype(float)
        error = restored - reference
        psnr = 10 * np.log10(255**2 / max(np.mean(error**2), 1e-12))
        assert psnr > 45
        assert abs(error.mean()) < 1
    contents = output.read_bytes()
    assert contents.index(b"moov") < contents.index(b"mdat")


@pytest.mark.unit
def test_rgb_nv12_known_bt709_primaries():
    # Published BT.709 limited-range primary values, each on a full chroma block.
    """Compare color conversion with published BT.709 primary calibration values."""
    frame = np.repeat(
        np.repeat(
            np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8),
            2,
            axis=0,
        ),
        2,
        axis=1,
    )
    buffer, width, height = image_nv12(frame)
    planes = buffer.reshape(height * 3 // 2, width)
    np.testing.assert_array_equal(planes[0, :6:2], [63, 173, 32])
    np.testing.assert_array_equal(planes[height, :6], [102, 240, 42, 26, 240, 118])


@pytest.mark.integration
@pytest.mark.gpu
def test_nvenc_rgb_calibration_round_trip(tmp_path):
    """Encode on-disk color calibration objects and verify decoded output colors."""
    pytest.importorskip("PyNvVideoCodec")
    rgb = np.zeros((256, 384, 3), dtype=np.uint8)
    colors = [[200, 30, 40], [20, 180, 40], [40, 30, 210]]
    paths = []
    for index, color in enumerate(colors):
        rgb[:, index * 128 : (index + 1) * 128] = color
    for t in range(3):
        path = tmp_path / f"colors_t{t:04d}.tiff"
        tifffile.imwrite(
            path, rgb, photometric="rgb", metadata={"volume_interval_ms": 85}
        )
        paths.append(path)
    output = encode_sequence(paths, tmp_path / "colors.mp4")
    decoded = imageio_ffmpeg.read_frames(str(output), pix_fmt="rgb24")
    next(decoded)
    frames = list(decoded)
    assert len(frames) == 3
    for raw in frames:
        frame = np.frombuffer(raw, np.uint8).reshape(256, 384, 3)
        for index, color in enumerate(colors):
            np.testing.assert_allclose(frame[128, index * 128 + 64], color, atol=4)
