"""Select maximum-Z ROIs, reconstruct raw regions, and export OME-Zarr/TIFF."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import numpy as np
import typer
from yaozarrs import open_group, v05

from opm_processing.export_ome_tiff import (
    export_image,
    load_export_metadata,
    resolve_export_image,
)

if TYPE_CHECKING:
    from napari.components import ViewerModel
    from napari.layers import Image

from opm_processing.dataio.acquisition import acquisition_stem, inspect_acquisition
from opm_processing.dataio.processing_state import (
    ProcessingState,
    processing_state_path,
)
from opm_processing.dataio.roi import (
    ROI_PREVIEW_SUFFIXES,
    projection_tile_footprints,
    roi_from_image_pixel_rectangle,
)
from opm_processing.process_roi import process_roi

app = typer.Typer(pretty_exceptions_enable=False)


def resolve_roi_source(root_path: Path) -> Path:
    """Find an existing maximum-Z preview for one acquisition.

    Parameters
    ----------
    root_path : pathlib.Path
        Raw acquisition, processed directory, or explicit maximum-Z store.

    Returns
    -------
    pathlib.Path
        Fused preview or single-position maximum-Z collection.

    Raises
    ------
    ValueError
        Input is ambiguous or no supported preview exists.
    """
    root_path = root_path.expanduser().resolve()
    suffixes = ROI_PREVIEW_SUFFIXES
    if any(root_path.name.endswith(suffix) for suffix in suffixes):
        if root_path.is_dir():
            return root_path
    store = (root_path / "zarr.json").is_file() or (root_path / ".zgroup").is_file()
    directory = root_path.parent if store else root_path
    stem = acquisition_stem(root_path) if store else None
    for suffix in suffixes:
        candidates = sorted(directory.glob(f"{stem or '*'}{suffix}"))
        candidates = [candidate for candidate in candidates if candidate.is_dir()]
        if len(candidates) > 1:
            raise ValueError(
                "Multiple previews found; pass the intended maximum-Z OME-Zarr path"
            )
        if candidates:
            return candidates[0]
    try:
        raw = inspect_acquisition(root_path).path
    except (ValueError, FileNotFoundError):
        raw = root_path
    raise ValueError(
        f"No deskewed maximum-Z preview found in {directory}. Create one first:\n"
        f'uv run process "{raw}" --no-deconvolve'
    )


def configure_roi_projection(viewer: "ViewerModel", layers: list["Image"]) -> None:
    """Show a full-depth maximum projection using napari's Z slice margins.

    Parameters
    ----------
    viewer : napari.components.ViewerModel
        Viewer containing the acquired time, channel and spatial dimensions.
    layers : list of napari.layers.Image
        Image layers whose full Z extent will be projected.
    """
    viewer.dims.ndisplay = 2
    viewer.dims.order = tuple(range(viewer.dims.ndim))
    z_axis = viewer.dims.ndim - 3
    start, stop, step = viewer.dims.range[z_axis]
    for layer in layers:
        layer.projection_mode = "max"
    viewer.dims.set_point(z_axis, (start + stop) / 2)
    thickness = [0.0] * viewer.dims.ndim
    # Dims ranges are inclusive world coordinates. Pad by one sample on each
    # side so napari's rounded slice bounds include both boundary planes at
    # every pyramid level, including translated stacks with odd Z sizes.
    thickness[z_axis] = stop - start + 2 * step
    viewer.dims.thickness = tuple(thickness)


def rectangle_bounds(
    vertices: np.ndarray, shape: tuple[int, ...]
) -> tuple[int, int, int, int]:
    """Round outward on the level-zero grid and clip to the image extent.

    Parameters
    ----------
    vertices : numpy.ndarray
        Four axis-aligned rectangle vertices in level-zero data coordinates.
    shape : tuple of int
        Full image dimensions, ending in YX.

    Returns
    -------
    tuple[int, int, int, int]
        Clipped start-inclusive, stop-exclusive bounds y0, y1, x0, x1.

    Raises
    ------
    ValueError
        The rectangle is invalid, rotated, nonfinite or outside the image.
    """
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[0] != 4 or vertices.shape[1] < 2:
        raise ValueError("Expected four rectangle vertices")
    if not np.isfinite(vertices).all():
        raise ValueError("ROI coordinates must be finite")
    if not np.allclose(vertices[:, :-2], vertices[0, :-2], rtol=0, atol=1e-7):
        raise ValueError("Draw rectangles in the YX view")
    points = vertices[:, -2:]
    if len(np.unique(points[:, 0])) != 2 or len(np.unique(points[:, 1])) != 2:
        raise ValueError("ROIs must be axis-aligned YX rectangles")
    low = np.maximum(0, np.floor(points.min(axis=0) + 1e-7)).astype(int)
    high = np.minimum(shape[-2:], np.ceil(points.max(axis=0) - 1e-7)).astype(int)
    if np.any(high <= low):
        raise ValueError("ROI does not intersect the image")
    return int(low[0]), int(high[0]), int(low[1]), int(high[1])


def select_rois(
    source: Path, shape: tuple[int, ...]
) -> list[tuple[int, int, int, int]]:
    """Use napari's image transform to obtain level-zero data coordinates.

    Image world_to_data uses the base image transform, regardless of the
    pyramid level currently rendered. Never multiply by the active level's
    downsampling factor.

    Parameters
    ----------
    source : pathlib.Path
        Maximum-Z image or single-position collection to open in napari.
    shape : tuple of int
        Level-zero image dimensions, ending in YX.

    Returns
    -------
    list of tuple[int, int, int, int]
        Selected rectangles as level-zero y0, y1, x0, x1 bounds.
    """
    import napari

    root = open_group(source)
    if isinstance(root.ome_metadata(), v05.Bf2Raw):
        series = root["OME"].ome_metadata().series
        if len(series) != 1:
            raise ValueError("ROI viewing requires a fused or single-position preview")
    viewer = napari.Viewer()
    layers = viewer.open(str(source), plugin="napari-ome-zarr")
    reference = layers[0]
    if isinstance(root.ome_metadata(), v05.Bf2Raw):
        placement = projection_tile_footprints(source)[0]["origin_zyx_um"]
        for layer in layers:
            layer.translate = (*layer.translate[:-3], *placement)
    configure_roi_projection(viewer, layers)
    # Each ROI spans all Z and timepoints. A 2D overlay also avoids napari's
    # integer shape-slice keys hiding rectangles at fractional world Z values.
    rois = viewer.add_shapes(
        name="export ROIs",
        ndim=2,
        # Shapes keeps float32 vertices. Store local pixel coordinates rather
        # than large stage coordinates, preserving subpixel drawing precision.
        scale=reference.scale[-2:],
        translate=reference.translate[-2:],
        shape_type="rectangle",
        edge_color="yellow",
        face_color=[1, 1, 0, 0.15],
    )
    rois.mode = "add_rectangle"
    typer.echo(
        "Viewing a full-Z maximum projection. Right-click the Z slider to adjust "
        "slice thickness. Draw YX rectangles, then close napari to export all Z, "
        "channels, and timepoints."
    )
    napari.run()
    bounds = []
    for vertices, kind in zip(rois.data, rois.shape_type, strict=True):
        if kind != "rectangle":
            raise ValueError("Only rectangle ROIs are supported")
        world = [rois.data_to_world(vertex) for vertex in vertices]
        pixels = np.asarray([reference.world_to_data(vertex) for vertex in world])
        bounds.append(rectangle_bounds(pixels, shape))
    return bounds


@app.command(
    help="Draw multiple XY ROIs on an existing maximum-Z preview. Close napari to process their raw data and export full-depth OME-Zarr images, optionally with OME-TIFF."
)
def export_rois(
    root_path: Annotated[
        Path,
        typer.Argument(
            help="Raw acquisition, processed directory, or maximum-Z preview."
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(help="Destination; defaults to roi_exports beside the preview."),
    ] = None,
    deconvolve: Annotated[
        bool,
        typer.Option(
            "--deconvolve/--no-deconvolve",
            help="Run RLGC on the selected raw regions before deskewing.",
        ),
    ] = False,
    tiff: Annotated[
        bool, typer.Option(help="Also write compressed OME-BigTIFF; retain OME-Zarr.")
    ] = False,
    flatfield_correction: bool = False,
    save_float32: bool = False,
    z_downsample_level: Annotated[int, typer.Option(min=1)] = 1,
    decon_crop_scan: Annotated[int | None, typer.Option(min=1)] = None,
    decon_gpu_id: Annotated[int, typer.Option(min=0)] = 0,
    decon_verbose: int = 1,
    decon_psf_paths: list[Path] | None = None,
    registration_channel: Annotated[int, typer.Option(min=0)] = 0,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume/--no-resume",
            help="Reuse saved selections and compatible checkpoints. --no-resume redraws and restarts.",
        ),
    ] = True,
    workers: Annotated[int, typer.Option(min=1, help="TIFF compression workers.")] = 4,
) -> None:
    """Draw multiple XY ROIs, process their raw volumes, and export each result.

    Parameters
    ----------
    root_path : pathlib.Path
        Raw acquisition, processed directory, or existing maximum-Z preview.
    output : pathlib.Path or None
        Destination directory; None selects roi_exports beside the preview.
    deconvolve : bool
        Apply RLGC before orthogonal deskewing; disabled by default.
    tiff : bool
        Export OME-TIFF in addition to the final OME-Zarr image.
    flatfield_correction : bool
        Reuse or estimate a detector-response flatfield.
    save_float32 : bool
        Preserve float32 intensities instead of writing uint16.
    z_downsample_level : int
        Output Z binning factor; one retains native orthogonal sampling.
    decon_crop_scan : int or None
        Retained scan planes per reconstruction chunk; None selects automatically.
    decon_gpu_id : int
        CUDA device index for RLGC.
    decon_verbose : int
        Reconstruction diagnostic verbosity.
    decon_psf_paths : list[pathlib.Path] or None
        Channel-ordered measured PSFs; None uses recorded theoretical calibration.
    registration_channel : int
        Channel used for fresh local registration of multi-position ROIs.
    resume : bool
        Reuse saved selections and compatible processing checkpoints.
    workers : int
        TIFF compression workers.
    """
    source = resolve_roi_source(root_path)
    footprints = projection_tile_footprints(source)
    if not footprints:
        raise ValueError("The preview has no source tile footprints")
    suffix = next(
        suffix for suffix in ROI_PREVIEW_SUFFIXES if source.name.endswith(suffix)
    )
    stem = source.name.removesuffix(suffix)
    state = ProcessingState.read(processing_state_path(source.parent, stem))
    acquisition = inspect_acquisition(Path(state.document["source"]["path"]))
    acquisition.validate_processing_calibration()
    if acquisition.is_2d:
        raise ValueError("ROI reconstruction requires a raw oblique 3D acquisition")
    if registration_channel >= len(acquisition.channels):
        raise ValueError("Registration channel is outside the acquired channel range")
    destination = (
        output.expanduser().resolve() if output else source.parent / "roi_exports"
    )
    manifest_path = destination / "selections.json"
    if resume and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["source_path"] != str(acquisition.path) or manifest[
            "preview_path"
        ] != str(source):
            raise ValueError(
                "Saved selections belong to a different acquisition or preview; choose another --output"
            )
        selections = [destination / name for name in manifest["rois"]]
    else:
        image_path = resolve_export_image(source)
        root = open_group(image_path)
        dataset = root.ome_metadata().multiscales[0].datasets[0]
        shape = tuple(root[dataset.path].to_tensorstore().shape)
        bounds = select_rois(source, shape)
        if not bounds:
            typer.echo("No ROIs selected; no files exported.")
            return
        rois = [
            roi_from_image_pixel_rectangle(
                source, np.asarray([[y0, x0], [y0, x1], [y1, x1], [y1, x0]])
            )
            for y0, y1, x0, x1 in bounds
        ]
        if any(not roi.position_indices for roi in rois):
            raise ValueError("An ROI does not intersect any acquired tiles")
        selections = [
            destination / f"roi_{index:03d}" / "roi.json"
            for index in range(1, len(rois) + 1)
        ]
        for roi, path in zip(rois, selections, strict=True):
            roi.write(path)
        manifest = {
            "source_path": str(acquisition.path),
            "preview_path": str(source),
            "rois": [path.relative_to(destination).as_posix() for path in selections],
        }
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary.replace(manifest_path)
    for index, selection in enumerate(selections, 1):
        result = process_roi(
            acquisition.path,
            selection,
            output=selection.parent,
            deconvolve=deconvolve,
            flatfield_correction=flatfield_correction,
            save_float32=save_float32,
            z_downsample_level=z_downsample_level,
            decon_crop_scan=decon_crop_scan,
            decon_gpu_id=decon_gpu_id,
            decon_verbose=decon_verbose,
            decon_psf_paths=decon_psf_paths,
            registration_channel=registration_channel,
            resume=resume,
        )
        typer.echo(f"Created {result}")
        if tiff:
            target = selection.parent / f"{stem}_roi_{index:03d}.ome.tiff"
            template, provenance = load_export_metadata(result)
            export_image(result, target, template, provenance, workers=workers)
            typer.echo(f"Created {target}")


def main() -> None:
    """Run the interactive ROI processing and export CLI."""
    app()
