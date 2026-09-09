"""Read and write multi-position OME-Zarr collections with yaozarrs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from ome_types import from_xml, to_xml
from ome_types.model import (
    Channel,
    Image,
    MetadataOnly,
    OME,
    Pixels,
    StageLabel,
    UnitsLength,
)
from yaozarrs import open_group, v05
from yaozarrs.write.v05 import Bf2RawBuilder

from opm_processing.dataio.acquisition import inspect_acquisition
from opm_processing.dataio.metadata import convert_metadata
from opm_processing.dataio.ngff import (
    round_spatial,
    round_spatial_values,
    round_tczyx_transform,
)


@dataclass(frozen=True)
class PositionCollection:
    """A Bf2Raw collection and its ordered per-position array handles."""

    path: Path
    arrays: tuple[Any, ...]
    attributes: dict[str, Any]
    multiscale_arrays: tuple[tuple[Any, ...], ...] = ()
    multiscale_factors_yx: tuple[int, ...] = (1,)
    multiscale_downsample: str = "stride"
    voxel_size_um: tuple[float, float, float] = (1.0, 1.0, 1.0)
    stage_positions_zxy: tuple[tuple[float, float, float], ...] = ()
    channel_names: tuple[str, ...] = ()
    spatial_origins_zyx_um: tuple[tuple[float, float, float], ...] = ()

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
    multiscale_factors_yx: Sequence[int] = (),
    multiscale_downsample: str = "stride",
    spatial_offset_um: Sequence[float] = (0.0, 0.0, 0.0),
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
    if len(spatial_offset_um) != 3:
        raise ValueError("spatial_offset_um must contain z, y, and x offsets")
    if multiscale_downsample not in ("stride", "block_mean"):
        raise ValueError('multiscale_downsample must be "stride" or "block_mean"')

    t, positions, c, z, y, x = (int(value) for value in shape)
    voxel_size_um = round_spatial_values(voxel_size_um)
    spatial_offset_um = round_spatial_values(spatial_offset_um)
    factors = tuple(
        dict.fromkeys((1, *(int(value) for value in multiscale_factors_yx)))
    )
    if any(value < 1 for value in factors):
        raise ValueError("multiscale_factors_yx must contain positive integers")
    factors = tuple(value for value in factors if value == 1 or value <= min(y, x))
    if stage_positions is not None and len(stage_positions) != positions:
        raise ValueError("stage_positions must contain one entry per position")
    rounded_stage_positions = (
        None
        if stage_positions is None
        else tuple(round_spatial_values(position) for position in stage_positions)
    )

    root_attributes = convert_metadata(attributes or {})
    ome_xml = _build_ome_xml(
        shape=(t, positions, c, z, y, x),
        dtype=np.dtype(dtype),
        voxel_size_um=voxel_size_um,
        stage_positions=rounded_stage_positions,
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
    axes = [
        {"name": "t", "type": "time"},
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ]
    datasets = []
    array_specs = []
    for level, factor in enumerate(factors):
        scale = round_tczyx_transform(
            (
                1,
                1,
                voxel_size_um[0],
                voxel_size_um[1] * factor,
                voxel_size_um[2] * factor,
            )
        )
        center_shift = (
            0.5 * (factor - 1) if multiscale_downsample == "block_mean" else 0
        )
        translation = round_tczyx_transform(
            (
                0,
                0,
                spatial_offset_um[0],
                spatial_offset_um[1] + center_shift * voxel_size_um[1],
                spatial_offset_um[2] + center_shift * voxel_size_um[2],
            )
        )
        datasets.append(
            v05.Dataset(
                path=str(level),
                coordinateTransformations=[
                    v05.ScaleTransformation(scale=scale),
                    v05.TranslationTransformation(translation=translation),
                ],
            )
        )
        if multiscale_downsample == "stride":
            level_y = (y + factor - 1) // factor
            level_x = (x + factor - 1) // factor
        else:
            level_y = y // factor
            level_x = x // factor
        array_specs.append(((t, c, z, level_y, level_x), np.dtype(dtype)))
    for position in range(positions):
        image = v05.Image(
            multiscales=[
                v05.Multiscale(
                    name=f"position-{position}",
                    axes=axes,
                    datasets=datasets,
                )
            ]
        )
        builder.add_series(str(position), image, array_specs)

    path, handles = builder.prepare()
    arrays = tuple(handles[f"{position}/0"] for position in range(positions))
    multiscale_arrays = tuple(
        tuple(handles[f"{position}/{level}"] for position in range(positions))
        for level in range(len(factors))
    )
    origins = (spatial_offset_um,) * positions
    return PositionCollection(
        path=path,
        arrays=arrays,
        attributes=root_attributes,
        multiscale_arrays=multiscale_arrays,
        multiscale_factors_yx=factors,
        multiscale_downsample=multiscale_downsample,
        voxel_size_um=tuple(voxel_size_um),
        stage_positions_zxy=rounded_stage_positions or (),
        channel_names=tuple(channels or (f"channel-{index}" for index in range(c))),
        spatial_origins_zyx_um=origins,
    )


def create_variable_position_collection(
    output_path: str | Path,
    shapes_tczyx: Sequence[Sequence[int]],
    voxel_size_um: Sequence[float],
    *,
    spatial_origins_zyx_um: Sequence[Sequence[float]],
    dtype: np.dtype | str = np.uint16,
    channels: Sequence[str] | None = None,
    attributes: dict[str, Any] | None = None,
    chunks: tuple[int, ...] | str | None = "auto",
    overwrite: bool = True,
) -> PositionCollection:
    """Create a Bf2Raw collection whose cropped series have different shapes.

    Each series is an independently placed TCZYX image. This is used for ROI
    tiles so no series needs to be padded back to the original camera tile.
    """
    shapes = tuple(tuple(int(value) for value in shape) for shape in shapes_tczyx)
    origins = tuple(round_spatial_values(origin) for origin in spatial_origins_zyx_um)
    if not shapes or any(len(shape) != 5 for shape in shapes):
        raise ValueError("shapes_tczyx must contain nonempty TCZYX shapes")
    if len(origins) != len(shapes):
        raise ValueError("spatial_origins_zyx_um must contain one origin per series")
    if len(voxel_size_um) != 3:
        raise ValueError("voxel_size_um must contain z, y, and x spacing")
    if any(shape[0] != 1 for shape in shapes):
        raise ValueError("Variable ROI series must each contain one timepoint")
    if len({shape[1] for shape in shapes}) != 1:
        raise ValueError("Variable ROI series must have a common channel count")

    voxel_size = round_spatial_values(voxel_size_um)
    output_dtype = np.dtype(dtype)
    root_attributes = convert_metadata(attributes or {})
    ome_xml = _build_variable_ome_xml(
        shapes,
        output_dtype,
        voxel_size,
        origins,
        channels,
    )
    builder = Bf2RawBuilder(
        output_path,
        ome_xml=ome_xml,
        writer="tensorstore",
        chunks=chunks,
        overwrite=overwrite,
        extra_attributes=root_attributes,
    )
    axes = [
        {"name": "t", "type": "time"},
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ]
    scale = round_tczyx_transform((1, 1, *voxel_size))
    for series, (shape, origin) in enumerate(zip(shapes, origins)):
        image = v05.Image(
            multiscales=[
                v05.Multiscale(
                    name=f"roi-tile-{series}",
                    axes=axes,
                    datasets=[
                        v05.Dataset(
                            path="0",
                            coordinateTransformations=[
                                v05.ScaleTransformation(scale=scale),
                                v05.TranslationTransformation(
                                    translation=round_tczyx_transform((0, 0, *origin))
                                ),
                            ],
                        )
                    ],
                )
            ]
        )
        builder.add_series(str(series), image, ((shape, output_dtype),))

    path, handles = builder.prepare()
    arrays = tuple(handles[f"{series}/0"] for series in range(len(shapes)))
    return PositionCollection(
        path=path,
        arrays=arrays,
        attributes=root_attributes,
        voxel_size_um=tuple(voxel_size),
        stage_positions_zxy=origins,
        channel_names=tuple(
            channels or (f"channel-{index}" for index in range(int(shapes[0][1])))
        ),
        spatial_origins_zyx_um=origins,
    )


def _build_variable_ome_xml(
    shapes_tczyx: Sequence[Sequence[int]],
    dtype: np.dtype,
    voxel_size_um: Sequence[float],
    origins_zyx_um: Sequence[Sequence[float]],
    channels: Sequence[str] | None,
) -> str:
    """Build OME-XML for independently shaped and positioned ROI tiles."""
    channel_count = int(shapes_tczyx[0][1])
    channel_names = list(
        channels or (f"channel-{index}" for index in range(channel_count))
    )
    if len(channel_names) != channel_count:
        raise ValueError("channels must contain one name per channel")
    ome_pixel_type = {
        "float32": "float",
        "float64": "double",
        "complex64": "complex",
        "complex128": "double-complex",
    }.get(dtype.name, dtype.name)
    images = []
    for series, (shape, origin) in enumerate(zip(shapes_tczyx, origins_zyx_um)):
        t, c, z, y, x = (int(value) for value in shape)
        origin_z, origin_y, origin_x = (round_spatial(value) for value in origin)
        images.append(
            Image(
                id=f"Image:{series}",
                name=f"ROI tile {series}",
                stage_label=StageLabel(
                    name=f"ROI tile {series}",
                    x=origin_x,
                    x_unit=UnitsLength.MICROMETER,
                    y=origin_y,
                    y_unit=UnitsLength.MICROMETER,
                    z=origin_z,
                    z_unit=UnitsLength.MICROMETER,
                ),
                pixels=Pixels(
                    id=f"Pixels:{series}",
                    dimension_order="XYZCT",
                    type=ome_pixel_type,
                    size_x=x,
                    size_y=y,
                    size_z=z,
                    size_c=c,
                    size_t=t,
                    physical_size_x=round_spatial(voxel_size_um[2]),
                    physical_size_y=round_spatial(voxel_size_um[1]),
                    physical_size_z=round_spatial(voxel_size_um[0]),
                    channels=[
                        Channel(
                            id=f"Channel:{series}:{channel}",
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
    ome_pixel_type = {
        "float32": "float",
        "float64": "double",
        "complex64": "complex",
        "complex128": "double-complex",
    }.get(dtype.name, dtype.name)

    images = []
    for position in range(positions):
        stage_label = None
        if stage_positions is not None:
            stage_z, stage_y, stage_x = (
                round_spatial(value) for value in stage_positions[position]
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
                    type=ome_pixel_type,
                    size_x=x,
                    size_y=y,
                    size_z=z,
                    size_c=c,
                    size_t=t,
                    physical_size_x=round_spatial(voxel_size_um[2]),
                    physical_size_y=round_spatial(voxel_size_um[1]),
                    physical_size_z=round_spatial(voxel_size_um[0]),
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
    if not isinstance(root.ome_metadata(), v05.Bf2Raw):
        raise ValueError(f"Not a Bio-Formats2Raw collection: {path}")

    ome_group = root["OME"]
    series_metadata = ome_group.ome_metadata()
    if not isinstance(series_metadata, v05.Series):
        raise ValueError(f"Collection lacks typed OME series metadata: {path}")
    collection_path = Path(path).expanduser().resolve()
    ome = from_xml(
        (collection_path / "OME" / "METADATA.ome.xml").read_text(encoding="utf-8")
    )
    arrays = []
    series_levels = []
    spatial_origins = []
    factors: tuple[int, ...] | None = None
    for name in series_metadata.series:
        image_group = root[name]
        image_metadata = image_group.ome_metadata()
        if not isinstance(image_metadata, v05.Image):
            raise ValueError(f"Series {name} lacks typed OME image metadata")
        datasets = image_metadata.multiscales[0].datasets
        level_arrays = tuple(
            image_group[dataset.path].to_tensorstore() for dataset in datasets
        )
        if not level_arrays:
            raise ValueError(f"Series {name} contains no multiscale datasets")
        level_scales = tuple(
            int(
                round(
                    float(dataset.scale_transform.scale[-1])
                    / float(datasets[0].scale_transform.scale[-1])
                )
            )
            for dataset in datasets
        )
        if factors is None:
            factors = level_scales
        elif level_scales != factors:
            raise ValueError("All position series must use the same pyramid factors")
        arrays.append(level_arrays[0])
        series_levels.append(level_arrays)
        translation = datasets[0].translation_transform
        spatial_origins.append(
            tuple(
                float(value)
                for value in (
                    (0.0, 0.0, 0.0)
                    if translation is None
                    else translation.translation[-3:]
                )
            )
        )
    arrays = tuple(arrays)
    if not arrays:
        raise ValueError(f"Bf2Raw collection has no image series: {path}")
    attributes = {key: value for key, value in root.attrs.items() if key != "ome"}
    variable_shapes = any(
        tuple(array.shape) != tuple(arrays[0].shape) for array in arrays[1:]
    )

    if "opm_v2" in root.attrs:
        acquisition = inspect_acquisition(path, root=root)
        attributes.setdefault("acquisition", acquisition.to_dict())
        attributes.setdefault(
            "stage_positions", convert_metadata(acquisition.stage_positions_zxy)
        )
        attributes.setdefault("channels", list(acquisition.channel_names))
    multiscale_arrays = tuple(
        tuple(levels[level] for levels in series_levels)
        for level in range(len(series_levels[0]))
    )
    if len(ome.images) != len(arrays):
        raise ValueError("OME-XML image count does not match the position series")
    first_pixels = ome.images[0].pixels
    voxel_size_um = (
        float(first_pixels.physical_size_z or 1.0),
        float(first_pixels.physical_size_y or 1.0),
        float(first_pixels.physical_size_x or 1.0),
    )
    channel_names = tuple(
        channel.name or f"channel-{index}"
        for index, channel in enumerate(first_pixels.channels)
    )
    stage_positions = tuple(
        (
            float(image.stage_label.z or 0.0),
            float(image.stage_label.y or 0.0),
            float(image.stage_label.x or 0.0),
        )
        for image in ome.images
        if image.stage_label is not None
    )
    if not variable_shapes and len(stage_positions) not in (0, len(arrays)):
        raise ValueError("OME-XML stage labels are incomplete")
    return PositionCollection(
        path=collection_path,
        arrays=arrays,
        attributes=attributes,
        multiscale_arrays=multiscale_arrays,
        multiscale_factors_yx=factors or (1,),
        multiscale_downsample="stride",
        voxel_size_um=voxel_size_um,
        stage_positions_zxy=stage_positions,
        channel_names=channel_names,
        spatial_origins_zyx_um=tuple(spatial_origins),
    )


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
