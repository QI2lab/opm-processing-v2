"""Encode grayscale or RGB projection TIFF sequences with NVIDIA NVENC and MP4 muxing."""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Annotated, Sequence

import imageio_ffmpeg
import numpy as np
import tifffile
import typer
from tqdm import tqdm

app = typer.Typer(pretty_exceptions_enable=False)
FRAME_PATTERN = re.compile(r"(.+)_t(\d+)\.tiff?$", re.IGNORECASE)


def find_sequences(root: Path) -> list[list[Path]]:
    """Group exported TIFFs by parent and stem, and sort numerically by timepoint."""
    groups = defaultdict(list)
    for path in root.rglob("*"):
        match = FRAME_PATTERN.fullmatch(path.name)
        if path.is_file() and match:
            groups[(path.parent, match[1])].append((int(match[2]), path))
    sequences = []
    for key in sorted(groups):
        entries = sorted(groups[key])
        indices = [i for i, _ in entries]
        if indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError(f"Duplicate or missing timepoints in {key[0]} / {key[1]}")
        sequences.append([p for _, p in entries])
    if not sequences:
        raise ValueError(f"No *_t<number>.tiff projection frames found in {root}")
    return sequences


def image_nv12(frame: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Map grayscale or RGB display pixels to BT.709 limited-range NV12.

    Neutral chroma preserves grayscale. Black padding gives even dimensions
    and a minimum 128-pixel extent for small test or cropped canvases.
    """
    if frame.dtype != np.uint8 or not (
        frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 3)
    ):
        raise ValueError("Video input must be uint8 grayscale or RGB projection TIFFs.")
    h, w = frame.shape[:2]
    height, width = max(128, h + h % 2), max(128, w + w % 2)
    nv12 = np.full((height * 3 // 2, width), 128, dtype=np.uint8)
    nv12[:height] = 16
    if frame.ndim == 2:
        nv12[:h, :w] = np.rint(16 + frame.astype(np.float32) * (219 / 255)).astype(
            np.uint8
        )
    else:
        rgb = frame.astype(np.float32)
        luma = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        nv12[:h, :w] = np.rint(16 + luma * (219 / 255)).clip(16, 235).astype(np.uint8)
        for component, coefficient, offset in ((2, 0.0722, 0), (0, 0.2126, 1)):
            chroma = np.full((height, width), 128, dtype=np.float32)
            chroma[:h, :w] += (
                (112 / 255) * (rgb[..., component] - luma) / (1 - coefficient)
            )
            averaged = chroma.reshape(height // 2, 2, width // 2, 2).mean(axis=(1, 3))
            nv12[height:, offset::2] = np.rint(averaged).clip(16, 240).astype(np.uint8)
    return nv12.ravel(), width, height


def encode_sequence(
    frames: Sequence[Path],
    output: Path,
    *,
    fps=None,
    bitrate=8_000_000,
    gpu=0,
    codec="h264",
    rate_control="cbr",
    gop_seconds=2,
    encoder_options=None,
):
    """Encode one ordered sequence and atomically publish a fast-start H.264 MP4."""
    import PyNvVideoCodec as nvc

    if not frames:
        raise ValueError("No frames to encode.")
    if codec not in ("h264", "hevc") or rate_control not in ("cbr", "vbr"):
        raise ValueError("Choose codec h264/hevc and rate control cbr/vbr.")
    with tifffile.TiffFile(frames[0]) as tif:
        metadata = (tif.shaped_metadata or ({},))[0]
    if fps is None:
        interval = metadata.get("volume_interval_ms")
        if interval is None or not np.isfinite(interval) or interval <= 0:
            raise ValueError(
                "Frames must contain a positive volume_interval_ms, or supply --fps."
            )
        fps = Fraction(1000) / Fraction(str(interval))
    rate = Fraction(str(fps)).limit_denominator(1_000_000)
    if not frames or not 0 < rate <= 120 or bitrate <= 0 or gpu < 0:
        raise ValueError(
            "Require frames, fps >0..120, a positive bitrate and a nonnegative GPU index."
        )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    first = tifffile.imread(frames[0])
    _, width, height = image_nv12(first)
    # No B frames: raw elementary packets can be remuxed with monotonically
    # increasing timestamps without inventing presentation/decode reordering.
    encoder = nvc.CreateEncoder(
        width,
        height,
        "NV12",
        True,
        gpu_id=gpu,
        codec=codec,
        preset="P7",
        tuning_info="high_quality",
        rc=rate_control,
        bitrate=bitrate,
        maxbitrate=bitrate,
        fps=max(1, round(rate)),
        bf=0,
        gop=max(1, round(rate * gop_seconds)),
        idrperiod=max(1, round(rate * gop_seconds)),
        **(encoder_options or {}),
    )
    parameters = encoder.GetEncodeReconfigureParams()
    parameters.frameRateNum = rate.numerator
    parameters.frameRateDen = rate.denominator
    if not encoder.Reconfigure(parameters):
        raise RuntimeError("NVENC rejected the exact playback frame rate.")
    # Keep intermediates on the destination filesystem, so replace is atomic.
    with tempfile.TemporaryDirectory(prefix=".nvenc-", dir=output.parent) as work:
        elementary = Path(work) / "video.h264"
        movie = Path(work) / "video.mp4"
        with elementary.open("wb") as stream:
            for index, path in enumerate(tqdm(frames, desc=output.name, unit="frame")):
                frame = first if index == 0 else tifffile.imread(path)
                if frame.shape != first.shape:
                    raise ValueError(f"Frame dimensions changed at {path}")
                nv12, _, _ = image_nv12(frame)
                for packet in encoder.Encode(nv12):
                    stream.write(packet["data"])
            for packet in encoder.EndEncode():
                stream.write(packet["data"])
        del encoder
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-r",
            str(rate),
            "-f",
            codec,
            "-i",
            str(elementary),
            "-map",
            "0:v:0",
            "-c:v",
            "copy",
            "-an",
            "-tag:v",
            "avc1" if codec == "h264" else "hvc1",
            "-bsf:v",
            f"{codec}_metadata=video_full_range_flag=0:colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1:sample_aspect_ratio=1/1",
            "-color_range",
            "tv",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-movflags",
            "+faststart",
            "-y",
            str(movie),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"MP4 muxing failed: {result.stderr.strip()}")
        movie.replace(output)
    return output


@app.command()
def encode_projections(
    root_path: Annotated[
        Path,
        typer.Argument(
            help="Acquisition root directory containing the deconvolved/deskewed OME-Zarr store."
        ),
    ],
    fps: Annotated[
        float | None,
        typer.Option(
            min=0,
            max=120,
            help="Override playback fps; default is the acquired volume rate.",
        ),
    ] = None,
    bitrate: Annotated[
        int,
        typer.Option(min=1, help="Video bitrate in bits per second (default: 8 Mbps)."),
    ] = 8_000_000,
    gpu: Annotated[int, typer.Option(min=0, help="NVENC GPU index.")] = 0,
    codec: Annotated[
        str,
        typer.Option(help="h264 for broad compatibility, or hevc for newer players."),
    ] = "h264",
    rate_control: Annotated[
        str, typer.Option(help="cbr or vbr; both retain the configured bitrate target.")
    ] = "cbr",
):
    """Write one high-quality H.264 MP4 beside each projection TIFF sequence."""
    from opm_processing.export_projections import find_datasets

    dataset = find_datasets(root_path)[0]
    frame_root = (
        dataset.parent / "projection_frames" / dataset.name.removesuffix(".ome.zarr")
    )
    if not frame_root.is_dir():
        raise ValueError(
            f"No projection frames in {frame_root}. Run export-projections on the acquisition root first."
        )
    for frames in find_sequences(frame_root):
        if frames[0].parent.name.startswith("downsample_"):
            continue
        stem = FRAME_PATTERN.fullmatch(frames[0].name)[1]
        output = frames[0].parent / f"{stem}.mp4"
        encode_sequence(
            frames,
            output,
            fps=fps,
            bitrate=bitrate,
            gpu=gpu,
            codec=codec,
            rate_control=rate_control,
        )
        typer.echo(f"Wrote {output}")


def main():
    """Run the video encoding CLI."""
    app()


if __name__ == "__main__":
    main()
