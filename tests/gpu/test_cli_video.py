"""Real NVENC CLI round trips of projections from simulated camera acquisitions."""

import imageio_ffmpeg
import numpy as np
import pytest
import tifffile

from opm_processing.encode_projections import find_sequences
from opm_processing.export_projections import acquisition_for

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("nvenc_gpu")]


@pytest.mark.parametrize("roi_preview", [("mirror", 1, False)], indirect=True)
@pytest.mark.parametrize("codec", ["h264", "hevc"])
@pytest.mark.parametrize("rate_control", ["cbr", "vbr"])
def test_video_cli_preserves_simulated_timepoints_and_display_pixels(
    roi_preview, codec, rate_control, invoke_cli, relocate_acquisition
):
    """Encode real TIFF frames and decode MP4s with the installed FFmpeg package."""
    root = relocate_acquisition(
        roi_preview.dataset.path.parent, copy=rate_control == "vbr"
    )
    assert roi_preview.dataset.path.exists() == (rate_control == "vbr")
    assert (
        acquisition_for(root / roi_preview.collection.path.name).path
        == root / roi_preview.dataset.path.name
    )
    color = codec == "hevc"
    fps = 7.5 if rate_control == "vbr" else None
    invoke_cli(
        "export-projections",
        root,
        depth_color=color,
        workers=2,
        video=codec == "h264" and rate_control == "cbr",
        fps=fps,
    )
    # TIFF metadata now records the first location. Encoding must discover the
    # current frame directory after the complete acquisition is relocated again.
    root = relocate_acquisition(root, copy=rate_control == "vbr")
    invoke_cli(
        "encode-projections",
        root,
        codec=codec,
        rate_control=rate_control,
        fps=fps,
        bitrate=8_000_000,
        gpu=0,
    )
    sequences = find_sequences(root / "projection_frames")
    assert len(sequences) == roi_preview.dataset.metadata.shape[2]
    for paths in sequences:
        movie = paths[0].parent / (paths[0].stem.rsplit("_t", 1)[0] + ".mp4")
        decoded = imageio_ffmpeg.read_frames(str(movie), pix_fmt="rgb24")
        information = next(decoded)
        with tifffile.TiffFile(paths[0]) as tif:
            interval = tif.shaped_metadata[0]["volume_interval_ms"]
        assert information["codec"] == codec
        assert information["fps"] == pytest.approx(fps or 1000 / interval)
        width, height = information["size"]
        frames = list(decoded)
        assert len(frames) == len(paths) == roi_preview.dataset.metadata.shape[0]
        for frame, path in zip(frames, paths, strict=True):
            expected = tifffile.imread(path)
            if expected.ndim == 2:
                expected = np.repeat(expected[..., None], 3, axis=-1)
            image = np.frombuffer(frame, np.uint8).reshape(height, width, 3)
            actual = image[: expected.shape[0], : expected.shape[1]]
            # Video is a presentation copy. This bound concerns 8-bit display
            # conversion/compression, not recovered photon counts or PSNR.
            assert np.abs(actual.astype(float) - expected).mean() < 3
        assert not list(movie.parent.glob(".nvenc-*"))
