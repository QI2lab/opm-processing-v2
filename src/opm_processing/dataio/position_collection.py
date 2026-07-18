"""Read and write multi-position OME-Zarr collections with yaozarrs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from ome_types import to_xml
from ome_types.model import (
    Channel,
    Image,
    MetadataOnly,
    OME,
    Pixels,
    StageLabel,
    UnitsLength,
)
from yaozarrs import DimSpec, open_group, v05
from yaozarrs.write.v05 import Bf2RawBuilder

from opm_processing.dataio.metadata import convert_metadata


@dataclass(frozen=True)
class PositionCollection:
    """A Bf2Raw collection and its ordered per-position array handles."""

    path: Path
    arrays: tuple[Any, ...]
    attributes: dict[str, Any]

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the logical TPCZYX shape of the collection.

        Parameters
        ----------
        None
            This callable has no parameters.

        Returns
        -------
        tuple[int, ...]
            Result produced by the callable.
        """
        t, c, z, y, x = self.arrays[0].shape
        return int(t), len(self.arrays), int(c), int(z), int(y), int(x)


def create_position_collection(
    output_path: str | Path,
    shape: Sequence[int],
    voxel_size_um: Sequence[float],
    *,
    dtype: np.dtype | str = np.uint16,
    stage_positions: Sequence[Sequence[float]] | None = None,
    channels: Sequence[str] | None = None,
    attributes: dict[str, Any] | None = None,
    chunks: tuple[int, ...] | str | None = "auto",
    overwrite: bool = True,
) -> PositionCollection:
    """Create a Bf2Raw collection with one TCZYX image per position.

    Parameters
    ----------
    output_path : str | Path
        Value supplied for ``output path``.
    shape : Sequence[int]
        Value supplied for ``shape``.
    voxel_size_um : Sequence[float]
        Value supplied for ``voxel size um``.
    dtype : np.dtype | str
        Value supplied for ``dtype``.
    stage_positions : Sequence[Sequence[float]] | None
        Value supplied for ``stage positions``.
    channels : Sequence[str] | None
        Value supplied for ``channels``.
    attributes : dict[str, Any] | None
        Value supplied for ``attributes``.
    chunks : tuple[int, ...] | str | None
        Value supplied for ``chunks``.
    overwrite : bool
        Value supplied for ``overwrite``.

    Returns
    -------
    PositionCollection
        Result produced by the callable.
    """
    if len(shape) != 6:
        raise ValueError(f"Expected a TPCZYX shape, received {tuple(shape)}")
    if len(voxel_size_um) != 3:
        raise ValueError("voxel_size_um must contain z, y, and x spacing")

    t, positions, c, z, y, x = (int(value) for value in shape)
    if stage_positions is not None and len(stage_positions) != positions:
        raise ValueError("stage_positions must contain one entry per position")

    root_attributes = convert_metadata(attributes or {})
    if stage_positions is not None:
        root_attributes["stage_positions"] = convert_metadata(stage_positions)
    if channels is not None:
        root_attributes["channels"] = list(channels)

    ome_xml = _build_ome_xml(
        shape=(t, positions, c, z, y, x),
        dtype=np.dtype(dtype),
        voxel_size_um=voxel_size_um,
        stage_positions=stage_positions,
        channels=channels,
    )
    builder = Bf2RawBuilder(
        output_path,
        ome_xml=ome_xml,
        writer="tensorstore",
        chunks=chunks,
        overwrite=overwrite,
        extra_attributes=root_attributes,
    )
    dims = [
        DimSpec(name="t", size=t, scale=1.0),
        DimSpec(name="c", size=c, scale=1.0),
        DimSpec(name="z", size=z, scale=float(voxel_size_um[0]), unit="micrometer"),
        DimSpec(name="y", size=y, scale=float(voxel_size_um[1]), unit="micrometer"),
        DimSpec(name="x", size=x, scale=float(voxel_size_um[2]), unit="micrometer"),
    ]
    array_spec = ((t, c, z, y, x), np.dtype(dtype))
    for position in range(positions):
        image = v05.Image(
            multiscales=[v05.Multiscale.from_dims(dims, name=f"position-{position}")]
        )
        builder.add_series(str(position), image, array_spec)

    path, handles = builder.prepare()
    arrays = tuple(handles[f"{position}/0"] for position in range(positions))
    return PositionCollection(path, arrays, root_attributes)


def _build_ome_xml(
    shape: tuple[int, int, int, int, int, int],
    dtype: np.dtype,
    voxel_size_um: Sequence[float],
    stage_positions: Sequence[Sequence[float]] | None,
    channels: Sequence[str] | None,
) -> str:
    """Build a validated OME-XML companion for a Bf2Raw collection.

    Parameters
    ----------
    shape : tuple[int, int, int, int, int, int]
        Value supplied for ``shape``.
    dtype : np.dtype
        Value supplied for ``dtype``.
    voxel_size_um : Sequence[float]
        Value supplied for ``voxel size um``.
    stage_positions : Sequence[Sequence[float]] | None
        Value supplied for ``stage positions``.
    channels : Sequence[str] | None
        Value supplied for ``channels``.

    Returns
    -------
    str
        Result produced by the callable.
    """
    t, positions, c, z, y, x = shape
    channel_names = list(channels or (f"channel-{index}" for index in range(c)))
    if len(channel_names) != c:
        raise ValueError("channels must contain one name per channel")

    images = []
    for position in range(positions):
        stage_label = None
        if stage_positions is not None:
            stage_z, stage_y, stage_x = (
                float(value) for value in stage_positions[position]
            )
            stage_label = StageLabel(
                name=f"Position {position}",
                x=stage_x,
                x_unit=UnitsLength.MICROMETER,
                y=stage_y,
                y_unit=UnitsLength.MICROMETER,
                z=stage_z,
                z_unit=UnitsLength.MICROMETER,
            )
        images.append(
            Image(
                id=f"Image:{position}",
                name=f"Position {position}",
                stage_label=stage_label,
                pixels=Pixels(
                    id=f"Pixels:{position}",
                    dimension_order="XYZCT",
                    type=dtype.name,
                    size_x=x,
                    size_y=y,
                    size_z=z,
                    size_c=c,
                    size_t=t,
                    physical_size_x=float(voxel_size_um[2]),
                    physical_size_y=float(voxel_size_um[1]),
                    physical_size_z=float(voxel_size_um[0]),
                    channels=[
                        Channel(
                            id=f"Channel:{position}:{channel}",
                            name=name,
                            samples_per_pixel=1,
                        )
                        for channel, name in enumerate(channel_names)
                    ],
                    metadata_only=MetadataOnly(),
                ),
            )
        )
    ome_xml = to_xml(
        OME(images=images, creator="opm-processing-v2"),
        include_namespace=True,
        validate=True,
    )
    return ome_xml.replace(UnitsLength.MICROMETER.value, "&#181;m")


def open_position_collection(path: str | Path) -> PositionCollection:
    """Open a yaozarrs Bf2Raw collection as ordered TensorStore arrays.

    Parameters
    ----------
    path : str | Path
        Value supplied for ``path``.

    Returns
    -------
    PositionCollection
        Result produced by the callable.
    """
    root = open_group(path)
    if "bioformats2raw.layout" not in root.attrs.get("ome", {}):
        raise ValueError(f"Not a Bio-Formats2Raw collection: {path}")

    ome_group = root["OME"]
    series = ome_group.attrs["ome"]["series"]
    arrays = tuple(root[name]["0"].to_tensorstore() for name in series)
    if not arrays:
        raise ValueError(f"Bf2Raw collection has no image series: {path}")
    if any(tuple(array.shape) != tuple(arrays[0].shape) for array in arrays[1:]):
        raise ValueError("All position series must have the same TCZYX shape")

    attributes = {key: value for key, value in root.attrs.items() if key != "ome"}
    return PositionCollection(Path(path), arrays, attributes)


def open_image_array(path: str | Path, level: str = "0") -> Any:
    """Open one dataset from a yaozarrs Image group.

    Parameters
    ----------
    path : str | Path
        Value supplied for ``path``.
    level : str
        Value supplied for ``level``.

    Returns
    -------
    Any
        Result produced by the callable.
    """
    return open_group(path)[level].to_tensorstore()
