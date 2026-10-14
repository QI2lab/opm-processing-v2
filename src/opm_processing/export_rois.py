"""Select rectangles on a fused volume or timelapse and export level-zero crops."""

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import numpy as np
import typer
from yaozarrs import open_group

from opm_processing.export_ome_tiff import (
    export_image,
    export_source_stem,
    load_export_metadata,
    resolve_export_image,
)

if TYPE_CHECKING:
    from napari.components import ViewerModel
    from napari.layers import Image

app = typer.Typer(pretty_exceptions_enable=False)


def resolve_roi_source(root_path: Path) -> Path:
    """Prefer full fused, deconvolved, then deskewed volumes in a directory."""
    if (root_path / "zarr.json").is_file() or (root_path / ".zgroup").is_file():
        return root_path
    for suffix in ("_fused.ome.zarr", "_decon_deskewed.ome.zarr", "_deskewed.ome.zarr"):
        stores = sorted(
            p
            for p in root_path.glob(f"*{suffix}")
            if p.is_dir()
            and not p.name.endswith(f"_max_z{suffix}")
            and (
                suffix != "_deskewed.ome.zarr"
                or not p.name.endswith("_decon_deskewed.ome.zarr")
            )
        )
        if len(stores) > 1:
            raise ValueError(
                f"Expected one {suffix} store in {root_path}; found {len(stores)}. Pass an OME-Zarr store directly."
            )
        if stores:
            return stores[0]
    raise ValueError(
        f"No full fused, deconvolved, or deskewed OME-Zarr found in {root_path}"
    )


def configure_roi_projection(viewer: "ViewerModel", layers: list["Image"]) -> None:
    """Show a full-depth maximum projection using napari's Z slice margins."""
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
    """Round outward on the level-zero grid and clip to the image extent."""
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
    """
    import napari

    viewer = napari.Viewer()
    layers = list(viewer.open(str(source), plugin="napari-ome-zarr"))
    if not layers:
        viewer.close()
        raise ValueError("The OME-Zarr reader returned no image layers")
    reference = layers[0]
    configure_roi_projection(viewer, layers)
    # Each ROI spans all Z and timepoints. A 2D overlay also avoids napari's
    # integer shape-slice keys hiding rectangles at fractional world Z values.
    rois = viewer.add_shapes(
        name="export ROIs",
        ndim=2,
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


@app.command()
def export_rois(
    root_path: Annotated[
        Path,
        typer.Argument(
            help="Processed output directory or full OME-Zarr store (fused or single position)."
        ),
    ],
    output: Annotated[Path | None, typer.Option(help="Destination directory.")] = None,
    overwrite: Annotated[
        bool, typer.Option(help="Replace existing ROI exports.")
    ] = False,
    workers: Annotated[int, typer.Option(min=1, help="Compression workers.")] = 4,
) -> None:
    """Draw multiple ROIs in napari, then export compressed OME-BigTIFF crops."""
    root_path = root_path.expanduser().resolve()
    source = resolve_roi_source(root_path)
    image_path = resolve_export_image(source)
    root = open_group(image_path)
    dataset = root.ome_metadata().multiscales[0].datasets[0]
    shape = tuple(root[dataset.path].to_tensorstore().shape)
    bounds = select_rois(image_path, shape)
    if not bounds:
        typer.echo("No ROIs selected; no files exported.")
        return
    destination = (
        output.expanduser().resolve() if output else source.parent / "roi_exports"
    )
    stem = export_source_stem(source)
    targets = [
        destination / f"{stem}_roi_{i:03d}.ome.tiff" for i in range(1, len(bounds) + 1)
    ]
    for target in targets:
        if target.exists() and not overwrite:
            raise FileExistsError(
                f"{target} exists; use --overwrite to replace exports"
            )
    template, provenance = load_export_metadata(source)
    for crop, target in zip(bounds, targets, strict=True):
        export_image(
            source, target, template, provenance, workers=workers, bounds_yx=crop
        )
        typer.echo(f"Created {target}")


def main() -> None:
    """Run the interactive ROI export CLI."""
    app()
