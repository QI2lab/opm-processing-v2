"""Convert an acquisition timelapse without embedding experiment paths.

All acquisition selection and camera calibration values are supplied by the
caller or read from acquisition metadata. Importing this module performs no I/O.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import typer
import yaml
from tifffile import TiffWriter
from tqdm import tqdm

from opm_processing.dataio.acquisition import (
    inspect_acquisition,
    open_acquisition_datastore,
)
from opm_processing.imageprocessing.camera import camera_correct

app = typer.Typer()


def save_raw_with_yaml(data_array: np.ndarray, output_path: Path) -> None:
    """Write a uint16 RAW array and its shape/byte-order sidecar.

    Parameters
    ----------
    data_array : np.ndarray
        Uint16 data in the axis order required by the selected export.
    output_path : Path
        Destination file; the writer applies the RAW or TIFF suffix.

    Returns
    -------
    None
        No value is returned.
    """
    output_path = Path(output_path).with_suffix(".raw")
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


def save_time_projection(
    data_array: np.ndarray,
    pixel_size_um: float,
    output_path: Path,
    *,
    camera_offset: float,
    camera_conversion: float,
) -> None:
    """Save a mean time projection after explicit camera calibration.

    Parameters
    ----------
    data_array : np.ndarray
        Uint16 data in the axis order required by the selected export.
    pixel_size_um : float
        Detector pixel spacing in micrometers.
    output_path : Path
        Destination file; the writer applies the RAW or TIFF suffix.
    camera_offset : float
        Electronic camera background in ADU.
    camera_conversion : float
        Calibrated intensity per ADU.

    Returns
    -------
    None
        No value is returned.
    """
    output_path = Path(output_path).with_suffix(".tiff")
    calibrated = camera_correct(data_array, camera_offset, camera_conversion)
    projection = np.clip(calibrated, 0, np.iinfo(np.uint16).max).mean(
        axis=0, dtype=np.float32
    )
    save_as_tiff(projection.astype(np.uint16), pixel_size_um, output_path, axes="YX")


def save_as_tiff(
    data_array: np.ndarray,
    pixel_size_um: float,
    output_path: Path,
    *,
    axes: str = "TYX",
) -> None:
    """Save selected timelapse data as OME-TIFF.

    Parameters
    ----------
    data_array : np.ndarray
        Uint16 data in the axis order required by the selected export.
    pixel_size_um : float
        Detector pixel spacing in micrometers.
    output_path : Path
        Destination file; the writer applies the RAW or TIFF suffix.
    axes : str
        TIFF axis labels matching the data dimensions.

    Returns
    -------
    None
        No value is returned.
    """
    output_path = Path(output_path).with_suffix(".tiff")
    resolution = 1e4 / pixel_size_um
    with TiffWriter(output_path, bigtiff=True) as tif:
        tif.write(
            data_array,
            resolution=(resolution, resolution),
            compression="zlib",
            predictor=True,
            photometric="minisblack",
            resolutionunit="CENTIMETER",
            metadata={
                "axes": axes,
                "SignificantBits": np.iinfo(np.uint16).bits,
                "PhysicalSizeX": pixel_size_um,
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeY": pixel_size_um,
                "PhysicalSizeYUnit": "µm",
            },
        )


def selection_bounds(
    requested: tuple[int, int] | None,
    length: int,
    name: str,
) -> tuple[int, int]:
    """Validate and normalize a half-open selection range.

    Parameters
    ----------
    requested : tuple[int, int] | None
        Half-open index range, or None for the entire axis.
    length : int
        Available samples along the selected axis.
    name : str
        Option name used in an invalid-range error.

    Returns
    -------
    tuple[int, int]
        Validated start and exclusive stop indices.
    """
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
    """Convert selected TPCZYX positions from one acquisition datastore.

    Parameters
    ----------
    zarr_dir : Path
        Acquisition store or its containing directory.
    output_dir : Path | None
        Destination directory, defaulting to converted_files beside the store.
    time_range : tuple[int, int] | None
        Half-open timepoint range, or all timepoints.
    stage_range : tuple[int, int] | None
        Half-open position range, or all positions.
    scan_range : tuple[int, int] | None
        Half-open scan-plane range, or all planes.
    fov_x_range : tuple[int, int] | None
        Half-open camera-column range, or the full detector width.
    create_raw : bool
        Write each selected timelapse as RAW with a YAML shape sidecar.
    create_time_projection : bool
        Write the calibrated mean over time for each selected channel.
    create_tiff : bool
        Write selected uncalibrated timelapses as OME-TIFF.
    camera_offset : float | None
        Electronic camera background in ADU.
    camera_conversion : float | None
        Calibrated intensity per ADU.

    Returns
    -------
    list[Path]
        Written RAW, YAML, and TIFF paths in export order.
    """
    zarr_dir = Path(zarr_dir)
    acquisition = inspect_acquisition(zarr_dir)
    zarr_dir = acquisition.path
    datastore = open_acquisition_datastore(acquisition)

    pixel_size_um = acquisition.pixel_size_um
    if camera_offset is None:
        camera_offset = acquisition.camera_offset
    if camera_conversion is None:
        camera_conversion = acquisition.camera_conversion

    t0, t1 = selection_bounds(time_range, datastore.shape[0], "time_range")
    p0, p1 = selection_bounds(stage_range, datastore.shape[1], "stage_range")
    z0, z1 = selection_bounds(scan_range, datastore.shape[3], "scan_range")
    x0, x1 = selection_bounds(fov_x_range, datastore.shape[5], "fov_x_range")
    destination = (
        Path(output_dir) if output_dir else zarr_dir.parent / "converted_files"
    )
    destination.mkdir(parents=True, exist_ok=True)

    if create_time_projection and (camera_offset is None or camera_conversion is None):
        raise ValueError(
            "Time projection requires camera offset and conversion in metadata "
            "or as explicit arguments."
        )

    written: list[Path] = []
    planes = ((position, scan) for position in range(p0, p1) for scan in range(z0, z1))
    for position, scan in tqdm(
        planes,
        total=(p1 - p0) * (z1 - z0),
        desc="planes",
        unit="plane",
    ):
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
    """Convert a selected acquisition; no paths or calibration are implicit.

    Parameters
    ----------
    zarr_dir : Path
        Acquisition store or its containing directory.
    output_dir : Path | None
        Destination directory, defaulting to converted_files beside the store.
    time_range : tuple[int, int] | None
        Half-open timepoint range, or all timepoints.
    stage_range : tuple[int, int] | None
        Half-open position range, or all positions.
    scan_range : tuple[int, int] | None
        Half-open scan-plane range, or all planes.
    fov_x_range : tuple[int, int] | None
        Half-open camera-column range, or the full detector width.
    create_raw : bool
        Write each selected timelapse as RAW with a YAML shape sidecar.
    create_time_projection : bool
        Write the calibrated mean over time for each selected channel.
    create_tiff : bool
        Write selected uncalibrated timelapses as OME-TIFF.
    camera_offset : float | None
        Electronic camera background in ADU.
    camera_conversion : float | None
        Calibrated intensity per ADU.

    Returns
    -------
    None
        No value is returned.
    """
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
