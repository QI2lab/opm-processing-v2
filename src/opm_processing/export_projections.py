"""Export physically scaled orthogonal projections of processed time series."""

from __future__ import annotations


from pathlib import Path

from itertools import islice
from functools import lru_cache

from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

from typing import Annotated


import numpy as np

import tifffile

import typer

from PIL import Image, ImageDraw, ImageFont

from tqdm import tqdm


from opm_processing.dataio.acquisition import inspect_acquisition

from opm_processing.dataio.position_collection import open_position_collection

from opm_processing.dataio.processing_state import ProcessingState


app = typer.Typer(pretty_exceptions_enable=False)

SUFFIX = "_decon_deskewed.ome.zarr"


def find_datasets(path: Path) -> list[Path]:
    """Resolve the full deconvolved/deskewed store directly within an acquisition root.

    Parameters
    ----------
    path : Path
        Acquisition root containing its full deconvolved/deskewed store.

    Returns
    -------
    list[Path]
        Single full deconvolved/deskewed store selected for projection export.
    """
    path = path.expanduser().resolve()

    if not path.is_dir() or path.name.endswith(".zarr"):
        raise ValueError(
            "Pass the acquisition root directory containing the processed OME-Zarr store."
        )

    datasets = sorted(
        p
        for p in path.glob(f"*{SUFFIX}")
        if p.is_dir() and not p.name.endswith("_max_z" + SUFFIX)
    )

    if not datasets:
        raise ValueError(f"No full deconvolved/deskewed OME-Zarr stores in {path}")

    if len(datasets) != 1:
        raise ValueError(
            f"Expected one full deconvolved/deskewed store in {path}; found {len(datasets)}."
        )

    return datasets


def acquisition_for(path: Path, override: Path | None = None):
    """Find the raw acquisition explicitly or through its processing sidecar.

    Parameters
    ----------
    path : Path
        Filesystem location of the image store or metadata document being read.
    override : Path | None
        Explicit raw acquisition path, or None to read it from processing state.

    Returns
    -------
    AcquisitionMetadata
        Raw acquisition metadata associated with the processed tile collection.
    """
    if override is not None:
        return inspect_acquisition(override)

    matches = []

    for sidecar in path.parent.glob("*.processing.json"):
        state = ProcessingState.read(sidecar)

        if path.name in state.document["outputs"]:
            source = Path(state.document["source"]["path"])

            if not source.is_absolute():
                source = sidecar.parent / source

            matches.append(source)

    if len(matches) != 1:
        raise ValueError(
            f"Expected one processing sidecar linking {path.name} to its "
            "raw acquisition; supply --acquisition explicitly."
        )

    return inspect_acquisition(matches[0])


def volume_interval_ms(acquisition) -> float:
    """Compute scan planes times exposure times channels (sum unequal exposures).

    Parameters
    ----------
    acquisition
        Inspected acquisition dimensions, stage geometry, and camera calibration.

    Returns
    -------
    float
        Acquired scan-plane count times the sum of channel exposures, in milliseconds.
    """
    exposures = [c.exposure_ms for c in acquisition.channels]

    if not exposures or any(
        e is None or not np.isfinite(e) or e <= 0 for e in exposures
    ):
        raise ValueError(
            "Acquisition metadata must contain a positive exposure for every channel."
        )

    planes = acquisition.scan_position_count
    return planes * sum(exposures)


def timestamp(milliseconds: float) -> str:
    """Format elapsed time without wrapping minutes at one hour.

    Parameters
    ----------
    milliseconds : float
        Elapsed acquisition time in milliseconds.

    Returns
    -------
    str
        Elapsed time formatted as minutes, seconds, and milliseconds.
    """
    minutes, remainder = divmod(round(milliseconds), 60_000)

    seconds, millis = divmod(remainder, 1000)

    return f"{minutes:02d}:{seconds:02d}:{millis:03d}"


def contrast_limits(volume: np.ndarray) -> tuple[float, float]:
    """Use first-volume 0.001st/99.999th percentiles, with a sparse/constant fallback.

    Parameters
    ----------
    volume : np.ndarray
        ZYX image volume to project.

    Returns
    -------
    tuple[float, float]
        Fixed low and high intensities for the entire channel sequence.
    """
    finite = volume[np.isfinite(volume)]

    if not finite.size:
        raise ValueError("The first volume contains no finite intensities.")

    low, high = np.percentile(finite, (0.001, 99.999))

    if high <= low:
        low, high = float(finite.min()), float(finite.max())

        if high <= low:
            high = low + 1

        typer.echo("Coincident contrast percentiles; using first-volume min/max range.")

    return float(low), float(high)


@lru_cache(maxsize=4)
def annotation_font(size=20):
    """Load a Unicode font instead of Pillow's limited default font.

    Parameters
    ----------
    size
        Annotation font size in pixels.

    Returns
    -------
    ImageFont
        Unicode-capable Pillow font for the requested annotation size.
    """
    for name in ("arial.ttf", "DejaVuSans.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size=size)

        except OSError:
            continue

    raise RuntimeError("Install Arial or DejaVu Sans to render the micrometer label.")


def projection_panels(
    volume, voxel_size_um, limits, depth_color=False, depth_colormap="turbo"
):
    """Compute the three physically scaled panels once per volume.

    Parameters
    ----------
    volume
        ZYX image volume to project.
    voxel_size_um
        Source voxel spacing in ZYX order, in micrometers.
    limits
        Fixed lower and upper intensity display limits.
    depth_color
        Color each maximum projection by the depth of its brightest voxel.
    depth_colormap
        Colormap used for physical depth colors and legends.

    Returns
    -------
    tuple
        Physically scaled XY/XZ/YZ images, display pixel spacing, and depth legends.
    """
    spacing = np.asarray(voxel_size_um, dtype=float)

    pixel_um = float(spacing.min())

    nz, ny, nx = np.maximum(
        1, np.rint(np.asarray(volume.shape) * spacing / pixel_um)
    ).astype(int)

    low, high = limits

    def panel(values, size):
        """Scale and resize one maximum projection into an 8-bit display panel.

        Parameters
        ----------
        values
            Unscaled maximum-projection intensities.
        size
            Requested display width and height in pixels.

        Returns
        -------
        array or scalar
            Resized 8-bit projection image for the output canvas.
        """
        values = np.nan_to_num(
            (values.astype(np.float32) - low) / (high - low), nan=0, posinf=1, neginf=0
        )

        values = np.clip(values, 0, 1)

        resized = Image.fromarray(values).resize(size, Image.Resampling.BILINEAR)

        return Image.fromarray(
            np.rint(np.clip(np.asarray(resized), 0, 1) * 255).astype(np.uint8)
        )

    legends = None

    if depth_color:
        from opm_processing.imageprocessing.depth_color import (
            depth_projection,
            depth_legends,
        )

        xy, xz, yz = (
            depth_projection(volume, axis, limits, depth_colormap) for axis in range(3)
        )

        xy = Image.fromarray(xy).resize((nx, ny), Image.Resampling.BILINEAR)

        xz = Image.fromarray(xz).resize((nx, nz), Image.Resampling.BILINEAR)

        yz = Image.fromarray(yz.transpose(1, 0, 2)).resize(
            (nz, ny), Image.Resampling.BILINEAR
        )

        legends = depth_legends(volume.shape, spacing, depth_colormap)

    else:
        xy = panel(volume.max(axis=0), (nx, ny))

        xz = panel(volume.max(axis=1), (nx, nz))

        yz = panel(volume.max(axis=2).T, (nz, ny))

    return (xy, xz, yz), pixel_um, legends


def assemble_canvas(
    xy, xz, yz, pixel_um, elapsed_ms, scale_bar_um=None, *, font_size=20, legends=None
):
    """Compose panels and supersample text without resampling image data.

    Parameters
    ----------
    xy
        Physically scaled XY projection image.
    xz
        Physically scaled XZ projection image.
    yz
        Physically scaled YZ projection image with vertical Y.
    pixel_um
        Physical display pixel size, in micrometers.
    elapsed_ms
        Acquisition time burned into the exported frame, in milliseconds.
    scale_bar_um
        Scale-bar length in micrometers, or None for an automatic length.
    font_size
        Annotation font size in display pixels.
    legends
        Depth colorbar records for the three projection orientations, or None.

    Returns
    -------
    tuple[np.ndarray, float, float]
        Annotated image pixels, display pixel size, and scale-bar length in micrometers.
    """
    xy, xz, yz = (
        Image.fromarray(p) if isinstance(p, np.ndarray) else p for p in (xy, xz, yz)
    )

    nx, ny = xy.size

    nz = xz.height

    text_sampling = 4

    font = annotation_font(font_size * text_sampling)

    gap, pad = (12, 16)

    label = timestamp(elapsed_ms)

    time_units = "min:s:ms"

    if scale_bar_um is None:
        target = min(nx, ny) * pixel_um / 5

        power = 10 ** np.floor(np.log10(target))

        scale_bar_um = max(
            (v * power for v in (0.1, 0.2, 0.5, 1, 2, 5) if v * power <= target)
        )

    if not np.isfinite(scale_bar_um) or scale_bar_um <= 0:
        raise ValueError("Scale bar length must be positive and finite.")

    bar_pixels = max(1, round(scale_bar_um / pixel_um))

    bar_label = f"{scale_bar_um:g} µm"

    text_width = int(
        np.ceil(
            max((font.getbbox(text)[2] for text in (label, time_units, bar_label)))
            / text_sampling
        )
    )

    corner_width = max(nz, text_width + 2 * pad, bar_pixels + 2 * pad)

    legend_width = max(120, font_size * 10)

    if legends:
        corner_width = max(corner_width, legend_width + 2 * pad)

    corner_height = max(nz, 134 + (len(legends) * 76 if legends else 0))

    mode = "RGB" if xy.mode == "RGB" else "L"

    canvas = Image.new(mode, (nx + gap + corner_width, ny + gap + corner_height), 0)

    canvas.paste(xy, (0, 0))

    canvas.paste(xz, (0, ny + gap))

    canvas.paste(yz, (nx + gap, 0))

    label_pad = max(2, round(font_size / 4))

    for name, panel, origin in (
        ("XY", xy, (0, 0)),
        ("XZ", xz, (0, ny + gap)),
        ("YZ", yz, (nx + gap, 0)),
    ):
        label_size = (
            min(
                panel.width,
                int(np.ceil(font.getlength(name) / text_sampling)) + 2 * label_pad,
            ),
            min(panel.height, font_size + 2 * label_pad),
        )

        overlay = Image.new(
            "RGBA", tuple((v * text_sampling for v in label_size)), (0, 0, 0, 0)
        )

        ImageDraw.Draw(overlay).text(
            (label_pad * text_sampling, label_pad * text_sampling),
            name,
            font=font,
            anchor="lt",
            fill="white",
            stroke_width=text_sampling,
            stroke_fill="black",
        )

        overlay = overlay.resize(label_size, Image.Resampling.LANCZOS)

        canvas.paste(overlay.convert(mode), origin, overlay.getchannel("A"))

    annotation_size = (corner_width, corner_height)

    annotations = Image.new(
        "L", tuple((size * text_sampling for size in annotation_size)), 0
    )

    text_draw = ImageDraw.Draw(annotations)

    for text, offset in ((label, 0), (time_units, 24), (bar_label, 79)):
        text_draw.text(
            (pad * text_sampling, (pad + offset) * text_sampling),
            text,
            font=font,
            fill=255,
        )

    if legends:
        for index, legend in enumerate(legends):
            top = 126 + index * 76

            heading = f"{legend['projection']}: {legend['axis']} depth (µm)"

            text_draw.text(
                (pad * text_sampling, top * text_sampling), heading, font=font, fill=255
            )

            for value, fraction in (
                (0, 0),
                (legend["max_um"] / 2, 0.5),
                (legend["max_um"], 1),
            ):
                text = f"{value:.0f}"

                width = font.getlength(text)

                x = (pad + fraction * legend_width) * text_sampling - fraction * width

                text_draw.text(
                    (x, (top + font_size + 18) * text_sampling),
                    text,
                    font=font,
                    fill=255,
                )

    canvas.paste(
        annotations.resize(annotation_size, Image.Resampling.LANCZOS),
        (nx + gap, ny + gap),
    )

    draw = ImageDraw.Draw(canvas)

    x, y = (nx + gap + pad, ny + gap + pad)

    draw.rectangle(
        (x, y + 66, x + bar_pixels - 1, y + 70), fill="white" if mode == "RGB" else 255
    )

    if legends:
        from opm_processing.imageprocessing.depth_color import depth_palette

        for index, legend in enumerate(legends):
            palette = depth_palette(legend["planes"], legend["colormap"])

            positions = np.rint(np.linspace(0, len(palette) - 1, legend_width)).astype(
                int
            )

            strip = np.tile(palette[positions][None], (10, 1, 1))

            canvas.paste(
                Image.fromarray(strip),
                (nx + gap + pad, ny + gap + 126 + index * 76 + font_size + 4),
            )

    return (np.asarray(canvas), pixel_um, scale_bar_um)


def run_timepoints(function, timepoints, workers):
    """Keep at most workers tasks in flight; propagate failures before encoding.

    Parameters
    ----------
    function
        Frame-writing function called once per selected timepoint.
    timepoints
        Half-open timepoint range for export, or an iterator of scheduled indices.
    workers
        Maximum number of concurrent timepoint reads and renders.

    Yields
    ------
    int
        Timepoint index after its frame write completes; concurrent completion order may vary.
    """
    if workers == 1:
        for t in timepoints:
            function(t)

            yield t

        return

    iterator = iter(timepoints)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(function, t): t for t in islice(iterator, workers)}

        try:
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)

                for future in done:
                    t = pending.pop(future)

                    future.result()

                    yield t

                for t in islice(iterator, workers - len(pending)):
                    pending[pool.submit(function, t)] = t

        finally:
            for future in pending:
                future.cancel()


def export_dataset(
    dataset: Path,
    output: Path,
    acquisition,
    scale_bar_um=None,
    *,
    video=False,
    fps=None,
    bitrate=8_000_000,
    gpu=0,
    depth_color=False,
    depth_colormap="turbo",
    workers=2,
    timepoints=None,
):
    """Export independent timepoints with bounded concurrent reads and rendering.

    Parameters
    ----------
    dataset : Path
        Processed position collection containing the deskewed volumes.
    output : Path
        Directory or file receiving the generated images and figures.
    acquisition
        Inspected acquisition dimensions, stage geometry, and camera calibration.
    scale_bar_um
        Scale-bar length in micrometers, or None for an automatic length.
    video
        Encode each exported position/channel TIFF sequence as an MP4.
    fps
        Playback frames per second, or None to use saved acquisition timing.
    bitrate
        NVENC target bitrate in bits per second.
    gpu
        CUDA device index used by the video encoder.
    depth_color
        Color each maximum projection by the depth of its brightest voxel.
    depth_colormap
        Colormap used for physical depth colors and legends.
    workers
        Maximum number of concurrent timepoint reads and renders.
    timepoints
        Half-open timepoint range for export, or an iterator of scheduled indices.
    """
    if not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer.")

    collection = open_position_collection(dataset)
    if timepoints is not None:
        start, stop = timepoints
        for array in collection.arrays:
            if not 0 <= start < stop <= array.shape[0]:
                raise ValueError(
                    f"Require 0 <= START < STOP <= {array.shape[0]} for --timepoints (STOP is exclusive)."
                )

    from opm_processing.imageprocessing.depth_color import depth_palette, depth_legends

    if depth_color:
        depth_palette(1, depth_colormap)  # Validate before writing any frames.

    interval = volume_interval_ms(acquisition)

    stem = dataset.name.removesuffix(".ome.zarr")

    typer.echo(f"{stem}: {interval / 1000:g} s/volume ({1000 / interval:g} volumes/s)")

    for position, array in enumerate(collection.arrays):
        times, channels, *_ = array.shape

        start, stop = (0, times) if timepoints is None else timepoints
        digits = max(4, len(str(times - 1)))

        for channel in range(channels):
            destination = output / stem / f"p{position:03d}" / f"c{channel:03d}"

            if depth_color:
                destination = destination / "depth_color"

            destination.mkdir(parents=True, exist_ok=True)

            first = np.asarray(array[0, channel].read().result())

            limits = contrast_limits(first)

            def write_timepoint(t, volume=None):
                """Read, project, annotate, and write one timepoint TIFF.

                Parameters
                ----------
                t
                    Output timepoint index.
                volume
                    ZYX image volume to project.

                Returns
                -------
                None
                    Writes one annotated TIFF using the fixed channel contrast limits.
                """
                if volume is None:
                    volume = np.asarray(array[t, channel].read().result())

                panels, display_pixel_um, legends = projection_panels(
                    volume,
                    collection.voxel_size_um,
                    limits,
                    depth_color,
                    depth_colormap,
                )

                canvas, pixel_um, bar_um = assemble_canvas(
                    *panels,
                    display_pixel_um,
                    t * interval,
                    scale_bar_um,
                    legends=legends,
                )

                tifffile.imwrite(
                    destination / f"{stem}_t{t:0{digits}d}.tiff",
                    canvas,
                    photometric="rgb" if depth_color else "minisblack",
                    compression="deflate",
                    compressionargs={"level": 6},
                    resolution=(10000 / pixel_um,) * 2,
                    resolutionunit="CENTIMETER",
                    metadata={
                        "axes": "YXS" if depth_color else "YX",
                        "depth_color": {
                            "method": "brightest_voxel_depth",
                            "reference": "local voxel centers; first voxel is 0 um",
                            "tie_break": "first voxel",
                            "legends": depth_legends(
                                volume.shape,
                                collection.voxel_size_um,
                                depth_colormap,
                            ),
                        }
                        if depth_color
                        else None,
                        "annotation_font_size_px": 20,
                        "display_pixel_size_um": pixel_um,
                        "source_voxel_size_zyx_um": list(collection.voxel_size_um),
                        "source_volume_shape_zyx": list(volume.shape),
                        "timepoint": t,
                        "elapsed_ms": t * interval,
                        "volume_interval_ms": interval,
                        "contrast_limits": limits,
                        "scale_bar_um": bar_um,
                        "position": position,
                        "channel": channel,
                        "channel_name": collection.channel_names[channel],
                    },
                )

            with tqdm(
                total=stop - start,
                desc=f"p{position:03d} c{channel:03d}",
                unit="timepoint",
            ) as progress:
                next_t = start
                if start == 0:
                    write_timepoint(0, first)
                    progress.update(1)
                    next_t = 1
                del first
                for _ in run_timepoints(write_timepoint, range(next_t, stop), workers):
                    progress.update(1)
            frames = [
                destination / f"{stem}_t{t:0{digits}d}.tiff" for t in range(start, stop)
            ]

            if video:
                from opm_processing.encode_projections import encode_sequence

                movie_stem = (
                    stem
                    if timepoints is None
                    else f"{stem}_t{start:0{digits}d}-t{stop - 1:0{digits}d}"
                )
                encode_sequence(
                    frames,
                    destination / f"{movie_stem}.mp4",
                    fps=fps,
                    bitrate=bitrate,
                    gpu=gpu,
                )


@app.command()
def export_projections(
    root_path: Annotated[
        Path,
        typer.Argument(
            help="Acquisition root directory containing the deconvolved/deskewed OME-Zarr store."
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            help="Output root; defaults to projection_frames beside the stores."
        ),
    ] = None,
    acquisition: Annotated[
        Path | None,
        typer.Option(
            help="Raw acquisition supplying scan count and exposure metadata."
        ),
    ] = None,
    scale_bar_um: Annotated[
        float | None,
        typer.Option(
            min=0, help="Scale bar length in micrometers; automatic when omitted."
        ),
    ] = None,
    video: Annotated[
        bool, typer.Option(help="Also encode an H.264 MP4 per channel/position.")
    ] = False,
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
    depth_color: Annotated[
        bool,
        typer.Option(
            help="Color each maximum projection by brightest-voxel depth and add three depth legends."
        ),
    ] = False,
    depth_colormap: Annotated[
        str, typer.Option(help="Depth lookup table name (used with --depth-color).")
    ] = "turbo",
    workers: Annotated[
        int,
        typer.Option(
            min=1,
            help="Concurrent timepoint read/render/write workers; 1 is serial. Memory use grows with workers.",
        ),
    ] = 2,
    timepoints: Annotated[
        tuple[int, int] | None,
        typer.Option(
            help="Render START STOP timepoints (zero-based, STOP exclusive). Default: all. Contrast stays fixed to dataset timepoint 0."
        ),
    ] = None,
):
    """Export annotated, physically scaled XY/XZ/YZ TIFFs for every timepoint.

    Parameters
    ----------
    root_path : Annotated[Path, typer.Argument(help='Acquisition root directory containing the deconvolved/deskewed OME-Zarr store.')]
        Acquisition or processed-output directory selected by the caller.
    output : Annotated[Path | None, typer.Option(help='Output root; defaults to projection_frames beside the stores.')]
        Directory or file receiving the generated images and figures.
    acquisition : Annotated[Path | None, typer.Option(help='Raw acquisition supplying scan count and exposure metadata.')]
        Inspected acquisition dimensions, stage geometry, and camera calibration.
    scale_bar_um : Annotated[float | None, typer.Option(min=0, help='Scale bar length in micrometers; automatic when omitted.')]
        Scale-bar length in micrometers, or None for an automatic length.
    video : Annotated[bool, typer.Option(help='Also encode an H.264 MP4 per channel/position.')]
        Encode each exported position/channel TIFF sequence as an MP4.
    fps : Annotated[float | None, typer.Option(min=0, max=120, help='Override playback fps; default is the acquired volume rate.')]
        Playback frames per second, or None to use saved acquisition timing.
    bitrate : Annotated[int, typer.Option(min=1, help='Video bitrate in bits per second (default: 8 Mbps).')]
        NVENC target bitrate in bits per second.
    gpu : Annotated[int, typer.Option(min=0, help='NVENC GPU index.')]
        CUDA device index used by the video encoder.
    depth_color : Annotated[bool, typer.Option(help='Color each maximum projection by brightest-voxel depth and add three depth legends.')]
        Color each maximum projection by the depth of its brightest voxel.
    depth_colormap : Annotated[str, typer.Option(help='Depth lookup table name (used with --depth-color).')]
        Colormap used for physical depth colors and legends.
    workers : Annotated[int, typer.Option(min=1, help='Concurrent timepoint read/render/write workers; 1 is serial. Memory use grows with workers.')]
        Maximum number of concurrent timepoint reads and renders.
    timepoints : Annotated[tuple[int, int] | None, typer.Option(help='Render START STOP timepoints (zero-based, STOP exclusive). Default: all. Contrast stays fixed to dataset timepoint 0.')]
        Half-open timepoint range for export, or an iterator of scheduled indices.
    """
    datasets = find_datasets(root_path)

    for dataset in datasets:
        export_dataset(
            dataset,
            output or dataset.parent / "projection_frames",
            acquisition_for(dataset, acquisition),
            scale_bar_um,
            video=video,
            fps=fps,
            bitrate=bitrate,
            gpu=gpu,
            depth_color=depth_color,
            depth_colormap=depth_colormap,
            workers=workers,
            timepoints=timepoints,
        )


def main():
    """Run the projection exporter CLI."""
    app()


if __name__ == "__main__":
    main()
