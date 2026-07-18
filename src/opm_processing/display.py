"""Display processed OPM data through napari-ome-zarr."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import napari
import typer
from napari.experimental import link_layers
from yaozarrs import open_group

app = typer.Typer(pretty_exceptions_enable=False)


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
    base = root_path.parent
    stem = root_path.stem
    candidates = {
        "max-z": (
            base / f"{stem}_max_z_decon_deskewed.ome.zarr",
            base / f"{stem}_max_z_deskewed.ome.zarr",
            base / f"{stem}_max_z_decon_deskewed.zarr",
            base / f"{stem}_max_z_deskewed.zarr",
        ),
        "full": (
            base / f"{stem}_decon_deskewed.ome.zarr",
            base / f"{stem}_deskewed.ome.zarr",
            base / f"{stem}_decon_deskewed.zarr",
            base / f"{stem}_deskewed.zarr",
        ),
        "fused-max-z": (
            base / f"{stem}_max_z_fused.ome.zarr",
            base / f"{stem}_max_z_fused.zarr",
        ),
        "fused-full": (base / f"{stem}_fused.ome.zarr",),
    }
    if to_display not in candidates:
        choices = ", ".join(candidates)
        raise ValueError(f"to_display must be one of: {choices}")
    for candidate in candidates[to_display]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No {to_display} output found. Checked: "
        + ", ".join(str(path) for path in candidates[to_display])
    )


def _configure_collection_layers(
    data_path: Path,
    layers: list[Any],
    pos_range: tuple[int, int] | None,
    time_range: tuple[int, int] | None,
) -> None:
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
    None
        No value is returned.
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
    if "bioformats2raw.layout" not in root.attrs.get("ome", {}):
        return

    positions = len(root["OME"].attrs["ome"]["series"])
    channels = len(root.attrs.get("channels", ())) or max(1, len(layers) // positions)
    start, stop = pos_range or (0, positions)
    if start < 0 or stop > positions or start >= stop:
        raise ValueError(f"pos_range must satisfy 0 <= start < stop <= {positions}")

    stage_positions = root.attrs.get("stage_positions")
    for position in range(positions):
        position_layers = layers[position * channels : (position + 1) * channels]
        for layer in position_layers:
            layer.visible = start <= position < stop
            if stage_positions is not None:
                z, y, x = (float(value) for value in stage_positions[position])
                layer.translate = (0.0, z, y, x)


@app.command()
def display(
    root_path: Path,
    to_display: str = "max-z",
    time_range: tuple[int, int] | None = None,
    pos_range: tuple[int, int] | None = None,
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

    Returns
    -------
    None
        No value is returned.
    """
    data_path = _resolve_data_path(root_path, to_display)

    viewer = napari.Viewer()
    layers = list(viewer.open(str(data_path), plugin="napari-ome-zarr"))
    _configure_collection_layers(data_path, layers, pos_range, time_range)

    root = open_group(data_path)
    if "bioformats2raw.layout" in root.attrs.get("ome", {}):
        channels = len(root.attrs.get("channels", ()))
        if channels:
            for channel in range(channels):
                channel_layers = layers[channel::channels]
                if len(channel_layers) > 1:
                    link_layers(channel_layers, ("contrast_limits", "gamma"))
    napari.run()


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
