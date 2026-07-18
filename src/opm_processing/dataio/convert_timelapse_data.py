"""Convert an acquisition timelapse without embedding experiment paths.

All acquisition selection and camera calibration values are supplied by the
caller or read from acquisition metadata. Importing this module performs no I/O.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorstore as ts
import typer
import yaml
from tifffile import TiffWriter

from opm_processing.dataio.metadata import find_key


app = typer.Typer()


def _with_suffix(path: Path, suffix: str) -> Path:
    return path if path.suffix == suffix else path.with_suffix(suffix)


def save_raw_with_yaml(data_array: np.ndarray, output_path: Path) -> None:
    """Write a uint16 RAW array and its shape/byte-order sidecar."""
    output_path = _with_suffix(Path(output_path), ".raw")
    yml_path = output_path.with_suffix(".yaml")
    np.asarray(data_array, dtype=np.uint16).tofile(output_path)
    meta = {
        "Frames": data_array.shape[0],
        "Data Type": str(np.dtype("uint16")),
        "Height": data_array.shape[-2],
        "Width": data_array.shape[-1],
        "Byte Order": "<",
    }
    with yml_path.open("w") as stream:
        yaml.safe_dump(meta, stream, sort_keys=False)


def _camera_correct(
    data_array: np.ndarray,
    *,
    camera_offset: float,
    camera_conversion: float,
) -> np.ndarray:
    corrected = (data_array.astype(np.float32) - camera_offset) * camera_conversion
    return np.clip(corrected, 0, np.iinfo(np.uint16).max)


def _tiff_metadata(axes: str, pixel_size_um: float) -> dict[str, object]:
    return {
        "axes": axes,
        "SignificantBits": np.iinfo(np.uint16).bits,
        "PhysicalSizeX": pixel_size_um,
        "PhysicalSizeXUnit": "µm",
        "PhysicalSizeY": pixel_size_um,
        "PhysicalSizeYUnit": "µm",
    }


def save_time_projection(
    data_array: np.ndarray,
    pixel_size_um: float,
    output_path: Path,
    *,
    camera_offset: float,
    camera_conversion: float,
) -> None:
    """Save a mean time projection after explicit camera calibration."""
    output_path = _with_suffix(Path(output_path), ".tiff")
    projection = _camera_correct(
        data_array,
        camera_offset=camera_offset,
        camera_conversion=camera_conversion,
    ).mean(axis=0, dtype=np.float32)
    _write_tiff(projection.astype(np.uint16), "YX", pixel_size_um, output_path)


def _write_tiff(
    data_array: np.ndarray,
    axes: str,
    pixel_size_um: float,
    output_path: Path,
) -> None:
    output_path = _with_suffix(Path(output_path), ".tiff")
    resolution = 1e4 / pixel_size_um
    with TiffWriter(output_path, bigtiff=True) as tif:
        tif.write(
            data_array,
            resolution=(resolution, resolution),
            compression="zlib",
            predictor=True,
            photometric="minisblack",
            resolutionunit="CENTIMETER",
            metadata=_tiff_metadata(axes, pixel_size_um),
        )


def save_as_tiff(
    data_array: np.ndarray,
    pixel_size_um: float,
    output_path: Path,
    *,
    axes: str = "TYX",
) -> None:
    """Save selected timelapse data as OME-TIFF."""
    _write_tiff(data_array, axes, pixel_size_um, output_path)


def _open_datastore(zarr_dir: Path) -> ts.TensorStore:
    last_error: Exception | None = None
    for driver in ("zarr3", "zarr"):
        try:
            return ts.open(
                {
                    "driver": driver,
                    "kvstore": {"driver": "file", "path": str(zarr_dir)},
                }
            ).result()
        except Exception as error:  # TensorStore uses different errors by driver.
            last_error = error
    raise ValueError(f"Unable to open datastore {zarr_dir}") from last_error


def _selection_bounds(
    requested: tuple[int, int] | None,
    length: int,
    name: str,
) -> tuple[int, int]:
    if requested is None:
        return 0, length
    start, stop = requested
    if not 0 <= start < stop <= length:
        raise ValueError(f"{name} must satisfy 0 <= start < stop <= {length}")
    return int(start), int(stop)


def convert_timelapse(
    zarr_dir: Path,
    *,
    output_dir: Path | None = None,
    time_range: tuple[int, int] | None = None,
    stage_range: tuple[int, int] | None = None,
    scan_range: tuple[int, int] | None = None,
    fov_x_range: tuple[int, int] | None = None,
    create_raw: bool = False,
    create_time_projection: bool = False,
    create_tiff: bool = True,
    camera_offset: float | None = None,
    camera_conversion: float | None = None,
) -> list[Path]:
    """Convert selected TPCZYX positions from one acquisition datastore."""
    zarr_dir = Path(zarr_dir)
    datastore = _open_datastore(zarr_dir)
    if datastore.rank != 6:
        raise ValueError(f"Expected TPCZYX rank 6, got shape {datastore.shape}")

    attrs_path = zarr_dir / ".zattrs"
    with attrs_path.open() as stream:
        attributes = json.load(stream)
    pixel_size_um = float(find_key(attributes, "pixel_size_um"))
    if camera_offset is None:
        value = find_key(attributes, "offset")
        camera_offset = None if value is None else float(value)
    if camera_conversion is None:
        value = find_key(attributes, "e_to_ADU")
        camera_conversion = None if value is None else float(value)

    t0, t1 = _selection_bounds(time_range, datastore.shape[0], "time_range")
    p0, p1 = _selection_bounds(stage_range, datastore.shape[1], "stage_range")
    z0, z1 = _selection_bounds(scan_range, datastore.shape[3], "scan_range")
    x0, x1 = _selection_bounds(fov_x_range, datastore.shape[5], "fov_x_range")
    destination = Path(output_dir) if output_dir else zarr_dir.parent / "converted_files"
    destination.mkdir(parents=True, exist_ok=True)

    if create_time_projection and (
        camera_offset is None or camera_conversion is None
    ):
        raise ValueError(
            "Time projection requires camera offset and conversion in metadata "
            "or as explicit arguments."
        )

    written: list[Path] = []
    for position in range(p0, p1):
        for scan in range(z0, z1):
            selected = np.asarray(
                datastore[t0:t1, position, :, scan, :, x0:x1].read().result(),
                dtype=np.uint16,
            )
            channel_count = selected.shape[1]
            axes = "TYX" if channel_count == 1 else "TCYX"
            stem = f"pos_{position}_scan_{scan}"
            if create_raw:
                path = destination / f"{stem}.raw"
                save_raw_with_yaml(selected, path)
                written.extend((path, path.with_suffix(".yaml")))
            if create_time_projection:
                for channel in range(channel_count):
                    path = destination / f"{stem}_c{channel}_time_mean.tiff"
                    save_time_projection(
                        selected[:, channel],
                        pixel_size_um,
                        path,
                        camera_offset=float(camera_offset),
                        camera_conversion=float(camera_conversion),
                    )
                    written.append(path)
            if create_tiff:
                path = destination / f"{stem}.tiff"
                save_as_tiff(
                    np.squeeze(selected, axis=1) if channel_count == 1 else selected,
                    pixel_size_um,
                    path,
                    axes=axes,
                )
                written.append(path)
    return written


@app.command()
def main(
    zarr_dir: Path,
    output_dir: Path | None = None,
    time_range: tuple[int, int] | None = None,
    stage_range: tuple[int, int] | None = None,
    scan_range: tuple[int, int] | None = None,
    fov_x_range: tuple[int, int] | None = None,
    create_raw: bool = False,
    create_time_projection: bool = False,
    create_tiff: bool = True,
    camera_offset: float | None = None,
    camera_conversion: float | None = None,
) -> None:
    """Convert a selected acquisition; no paths or calibration are implicit."""
    convert_timelapse(
        zarr_dir,
        output_dir=output_dir,
        time_range=time_range,
        stage_range=stage_range,
        scan_range=scan_range,
        fov_x_range=fov_x_range,
        create_raw=create_raw,
        create_time_projection=create_time_projection,
        create_tiff=create_tiff,
        camera_offset=camera_offset,
        camera_conversion=camera_conversion,
    )


if __name__ == "__main__":
    app()
