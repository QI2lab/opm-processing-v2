"""Display processed OPM data through napari-ome-zarr."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import napari
import numpy as np
import typer
from napari.experimental import link_layers
from yaozarrs import open_group, v05
from opm_processing.dataio.acquisition import (
    acquisition_stem,
    resolve_acquisition_path,
)
from opm_processing.dataio.position_collection import (
    PositionCollection,
    open_position_collection,
)
from opm_processing.dataio.roi import (
    roi_from_image_pixel_rectangle,
    validate_registered_max_projection,
)

app = typer.Typer(pretty_exceptions_enable=False)

_DISPLAY_OUTPUT_SUFFIXES = {
    "max-z": (
        "_max_z_decon_deskewed.ome.zarr",
        "_max_z_deskewed.ome.zarr",
    ),
    "full": (
        "_decon_deskewed.ome.zarr",
        "_deskewed.ome.zarr",
    ),
    "fused-max-z": ("_max_z_fused.ome.zarr",),
    "fused-full": ("_fused.ome.zarr",),
}


def _processed_output_stem(path: Path) -> str | None:
    """Return the acquisition stem encoded by a recognized display output."""
    for suffixes in _DISPLAY_OUTPUT_SUFFIXES.values():
        for suffix in suffixes:
            if path.name.endswith(suffix):
                return path.name[: -len(suffix)]
    return None


def _resolve_display_context(root_path: Path) -> tuple[Path, str]:
    """Resolve the output directory and acquisition stem for display lookup."""
    candidate = Path(root_path).expanduser().resolve()
    direct_stem = _processed_output_stem(candidate)
    if direct_stem is not None:
        return candidate.parent, direct_stem

    try:
        acquisition_path = resolve_acquisition_path(candidate)
    except ValueError as acquisition_error:
        if not candidate.is_dir():
            raise
        processed_stems = {
            stem
            for item in candidate.iterdir()
            if item.is_dir()
            for stem in (_processed_output_stem(item),)
            if stem is not None
        }
        if len(processed_stems) == 1:
            return candidate, processed_stems.pop()
        if len(processed_stems) > 1:
            found = ", ".join(sorted(processed_stems))
            raise ValueError(
                f"Expected processed data for one acquisition in {candidate}, "
                f"found: {found}"
            ) from acquisition_error
        raise
    return acquisition_path.parent, acquisition_stem(acquisition_path)


def _resolve_data_path(root_path: Path, to_display: str) -> Path:
    """Resolve a display mode to the first existing processed dataset.

    Parameters
    ----------
    root_path
        Original acquisition path used to derive output names.
    to_display
        Requested processed-data display mode.

    Returns
    -------
    pathlib.Path
        Existing processed dataset path.

    Raises
    ------
    ValueError
        If the display mode is unsupported.
    FileNotFoundError
        If no output exists for the requested mode.
    """
    if to_display not in _DISPLAY_OUTPUT_SUFFIXES:
        choices = ", ".join(_DISPLAY_OUTPUT_SUFFIXES)
        raise ValueError(f"to_display must be one of: {choices}")
    base, stem = _resolve_display_context(root_path)
    candidates = tuple(
        base / f"{stem}{suffix}" for suffix in _DISPLAY_OUTPUT_SUFFIXES[to_display]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No {to_display} output found. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


def _configure_collection_layers(
    data_path: Path,
    layers: list[Any],
    pos_range: tuple[int, int] | None,
    time_range: tuple[int, int] | None,
) -> PositionCollection | None:
    """Apply collection selections to plugin-created lazy layers.

    Parameters
    ----------
    data_path : Path
        Value supplied for ``data path``.
    layers : list[Any]
        Value supplied for ``layers``.
    pos_range : tuple[int, int] | None
        Value supplied for ``pos range``.
    time_range : tuple[int, int] | None
        Value supplied for ``time range``.

    Returns
    -------
    PositionCollection or None
        The opened multi-position collection, or ``None`` when ``data_path``
        is a single OME-NGFF Image such as a fused output.
    """
    if time_range is not None:
        time_start, time_stop = time_range
        if time_start < 0 or time_start >= time_stop:
            raise ValueError("time_range must satisfy 0 <= start < stop")
        for layer in layers:
            if layer.multiscale:
                layer.data = [level[time_start:time_stop] for level in layer.data]
            else:
                layer.data = layer.data[time_start:time_stop]

    root = open_group(data_path)
    if not isinstance(root.ome_metadata(), v05.Bf2Raw):
        return None

    collection = open_position_collection(data_path)
    positions = len(collection.arrays)
    channels = len(collection.channel_names) or max(1, len(layers) // positions)
    start, stop = pos_range or (0, positions)
    if start < 0 or stop > positions or start >= stop:
        raise ValueError(f"pos_range must satisfy 0 <= start < stop <= {positions}")

    for position in range(positions):
        position_layers = layers[position * channels : (position + 1) * channels]
        for layer in position_layers:
            layer.visible = start <= position < stop
            if collection.stage_positions_zxy:
                z, y, x = collection.stage_positions_zxy[position]
                layer.translate = (0.0, z, y, x)
    return collection


@app.command()
def display(
    root_path: Path,
    to_display: str = "fused-max-z",
    time_range: tuple[int, int] | None = None,
    pos_range: tuple[int, int] | None = None,
    roi_output: Path | None = None,
    roi: Annotated[
        bool,
        typer.Option(
            "--roi/--no-roi",
            help="Create and save one processing ROI rectangle.",
        ),
    ] = True,
) -> None:
    """Display processed OPM data using the napari-ome-zarr reader.

    Parameters
    ----------
    root_path : Path
        Value supplied for ``root path``.
    to_display : str
        Value supplied for ``to display``.
    time_range : tuple[int, int] | None
        Value supplied for ``time range``.
    pos_range : tuple[int, int] | None
        Value supplied for ``pos range``.
    roi_output : pathlib.Path or None
        Output path for the physical-coordinate ROI JSON. When omitted in ROI
        mode, defaults to ``<acquisition>_roi.json`` beside the displayed data.
    roi : bool
        Create and save one napari rectangle. Enabled by default; use
        ``--no-roi`` for display-only use.

    Returns
    -------
    None
        No value is returned.
    """
    data_path = _resolve_data_path(root_path, to_display)
    save_roi = roi or roi_output is not None
    if save_roi:
        if to_display != "fused-max-z":
            raise ValueError(
                "ROI mode requires --to-display fused-max-z. Use --no-roi "
                "to open another display mode."
            )
        try:
            validate_registered_max_projection(data_path)
        except (FileNotFoundError, ValueError):
            command_path = Path(root_path).expanduser().resolve()
            raise typer.BadParameter(
                "ROI mode requires a registered maximum-Z projection from fuse.\n"
                f'Run: uv run fuse "{command_path}"\n'
                "Then rerun display. For display-only use --no-roi.",
                param_hint="root_path",
            ) from None
    if save_roi and roi_output is None:
        stem = _processed_output_stem(data_path)
        if stem is None:
            raise RuntimeError(f"Cannot derive acquisition stem from {data_path}")
        roi_output = data_path.parent / f"{stem}_roi.json"

    viewer = napari.Viewer()
    layers = list(viewer.open(str(data_path), plugin="napari-ome-zarr"))
    collection = _configure_collection_layers(data_path, layers, pos_range, time_range)

    if collection is not None:
        channels = len(collection.channel_names)
        for channel in range(channels):
            channel_layers = layers[channel::channels]
            if len(channel_layers) > 1:
                link_layers(channel_layers, ("contrast_limits", "gamma"))
    roi_layer = None
    if save_roi:
        roi_layer = viewer.add_shapes(
            name="processing ROI",
            ndim=viewer.dims.ndim,
            shape_type="rectangle",
            edge_color="yellow",
            face_color=[1.0, 1.0, 0.0, 0.15],
        )
        print(
            "Draw exactly one rectangle in the 'processing ROI' layer, then "
            "close napari to save it. All Z values will be retained."
        )
    napari.run()
    if save_roi:
        if roi_layer is None or len(roi_layer.data) != 1:
            raise ValueError("Draw exactly one rectangle before closing napari")
        shape_types = np.atleast_1d(roi_layer.shape_type).tolist()
        if shape_types != ["rectangle"]:
            raise ValueError("The processing ROI layer must contain one rectangle")
        vertices_canvas_world = np.asarray(
            roi_layer.data_to_world(np.asarray(roi_layer.data[0])),
            dtype=np.float64,
        )
        reference_layer = layers[0]
        vertices_image_data = np.asarray(
            [reference_layer.world_to_data(vertex) for vertex in vertices_canvas_world],
            dtype=np.float64,
        )
        roi = roi_from_image_pixel_rectangle(data_path, vertices_image_data)
        if roi_output is None:
            raise RuntimeError("ROI output path was not resolved")
        written = roi.write(roi_output)
        selected = (
            "determined during processing"
            if roi.position_indices is None
            else f"{len(roi.position_indices)} source positions"
        )
        print(f"Saved physical ROI selecting {selected}: {written}")


def main() -> None:
    """Run the display command-line application.

    Parameters
    ----------
    None
        This callable has no parameters.

    Returns
    -------
    None
        No value is returned.
    """
    app()


if __name__ == "__main__":
    main()
