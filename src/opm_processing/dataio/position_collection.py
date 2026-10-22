"""Read and write multi-position OME-Zarr collections with yaozarrs."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ome_types import from_xml, to_xml
from ome_types.model import (
    OME,
    Channel,
    Image,
    MetadataOnly,
    Pixels,
    StageLabel,
    UnitsLength,
)
from yaozarrs import open_group, v05
from yaozarrs.write.v05 import Bf2RawBuilder

from opm_processing.dataio.acquisition import inspect_acquisition
from opm_processing.dataio.ngff import (
    pyramid_yx_sampling,
    round_spatial,
    round_spatial_values,
    round_tczyx_transform,
)
from opm_processing.dataio.processing_state import convert_metadata
from opm_processing.dataio.timing import read_time_interval, time_axis


def update_group_attributes(path: str | Path, attributes: dict[str, Any]) -> None:
    """Update group attributes with Zarr's public attribute API.

    Parameters
    ----------
    path : str or pathlib.Path
        Existing Zarr v3 group; missing groups are never created.
    attributes : dict
        Top-level attribute values to replace, preserving all other metadata
        and pixels. NumPy arrays are converted using the shared metadata encoder.

    Raises
    ------
    ValueError
        The group is not Zarr v3 or attribute values are not JSON serializable.
    """
    import zarr

    values = convert_metadata(attributes)
    json.dumps(values, allow_nan=False)
    group = zarr.open_group(
        str(Path(path).expanduser().resolve()),
        mode="r+",
        zarr_format=3,
        use_consolidated=False,
    )
    group.attrs.update(values)


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
    time_interval_s: float | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the logical TPCZYX shape of the collection.

        Returns
        -------
        tuple[int, ...]
            T, P, C, Z, Y, and X dimensions from the first series.
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
    time_interval_s: float | None = None,
) -> PositionCollection:
    """Create a Bf2Raw collection with one TCZYX image per position.

    Parameters
    ----------
    output_path : str | Path
        Destination Bf2Raw OME-Zarr directory.
    shape : Sequence[int]
        Logical collection dimensions in TPCZYX order.
    voxel_size_um : Sequence[float]
        Physical voxel spacing in ZYX order, in micrometers.
    dtype : np.dtype | str
        Pixel dtype stored in each series.
    stage_positions : Sequence[Sequence[float]] | None
        Per-position stage coordinates in the acquisition ordering, or no stage labels.
    channels : Sequence[str] | None
        Ordered channel names; omitted names become channel-N.
    attributes : dict[str, Any] | None
        Additional JSON-compatible root metadata.
    chunks : tuple[int, ...] | str | None
        TCZYX chunk dimensions or the TensorStore automatic chunk policy.
    overwrite : bool
        Whether an existing destination may be replaced.
    multiscale_factors_yx : Sequence[int]
        YX pyramid reduction factors; level zero is always included.
    multiscale_downsample : str
        Pyramid sampling method: stride or block_mean.
    spatial_offset_um : Sequence[float]
        Image origin in ZYX order, in micrometers.
    time_interval_s : float or None
        Physical snapshot interval in seconds; None leaves time unitless.

    Returns
    -------
    PositionCollection
        Ordered array handles and physical metadata for the collection.
    """
    t_axis, t_scale = time_axis(time_interval_s)
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
    rounded_stage_positions = (
        None
        if stage_positions is None
        else tuple(round_spatial_values(position) for position in stage_positions)
    )

    root_attributes = convert_metadata(attributes or {})
    ome_xml = _build_ome_xml(
        shapes_tczyx=[(t, c, z, y, x)] * positions,
        dtype=np.dtype(dtype),
        voxel_size_um=voxel_size_um,
        origins_zyx_um=rounded_stage_positions,
        channels=channels,
        time_interval_s=time_interval_s,
        name_prefix="Position",
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
        t_axis,
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
                t_scale,
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
        time_interval_s=time_interval_s,
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
    time_interval_s: float | None = None,
) -> PositionCollection:
    """Create a Bf2Raw collection whose cropped series have different shapes.

    Each series is an independently placed TCZYX image. This is used for ROI
    tiles so no series needs to be padded back to the original camera tile.

    Parameters
    ----------
    output_path : str | Path
        Destination Bf2Raw OME-Zarr directory.
    shapes_tczyx : Sequence[Sequence[int]]
        Per-tile TCZYX shapes with one timepoint and a common channel count.
    voxel_size_um : Sequence[float]
        Physical voxel spacing in ZYX order, in micrometers.
    spatial_origins_zyx_um : Sequence[Sequence[float]]
        Per-tile image origins in ZYX order, in micrometers.
    dtype : np.dtype | str
        Pixel dtype stored in each series.
    channels : Sequence[str] | None
        Ordered channel names; omitted names become channel-N.
    attributes : dict[str, Any] | None
        Additional JSON-compatible root metadata.
    chunks : tuple[int, ...] | str | None
        TCZYX chunk dimensions or the automatic chunk policy.
    overwrite : bool
        Whether an existing destination may be replaced.
    time_interval_s : float or None
        Physical snapshot interval in seconds; None leaves time unitless.

    Returns
    -------
    PositionCollection
        Ordered cropped tile handles and their physical origins.
    """
    t_axis, t_scale = time_axis(time_interval_s)
    shapes = tuple(tuple(int(value) for value in shape) for shape in shapes_tczyx)
    origins = tuple(round_spatial_values(origin) for origin in spatial_origins_zyx_um)

    voxel_size = round_spatial_values(voxel_size_um)
    output_dtype = np.dtype(dtype)
    root_attributes = convert_metadata(attributes or {})
    ome_xml = _build_ome_xml(
        shapes,
        output_dtype,
        voxel_size,
        origins,
        channels,
        time_interval_s,
        name_prefix="ROI tile",
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
        t_axis,
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ]
    scale = round_tczyx_transform((t_scale, 1, *voxel_size))
    for series, (shape, origin) in enumerate(zip(shapes, origins, strict=False)):
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
        time_interval_s=time_interval_s,
    )


def _build_ome_xml(
    shapes_tczyx: Sequence[Sequence[int]],
    dtype: np.dtype,
    voxel_size_um: Sequence[float],
    origins_zyx_um: Sequence[Sequence[float]] | None,
    channels: Sequence[str] | None,
    time_interval_s: float | None = None,
    *,
    name_prefix: str,
) -> str:
    """Build validated OME-XML for uniform collections or independently shaped ROIs.

    Parameters
    ----------
    shapes_tczyx : Sequence[Sequence[int]]
        Per-series dimensions in TCZYX order.
    dtype : np.dtype
        Pixel dtype shared by the series.
    voxel_size_um : Sequence[float]
        ZYX voxel spacing in micrometers.
    origins_zyx_um : Sequence[Sequence[float]] or None
        Per-series ZYX physical origins, or None when stage positions are unknown.
    channels : Sequence[str] | None
        Ordered channel names, or generated channel-N labels.
    time_interval_s : float or None
        Recorded interval between volumes in seconds; None leaves timing unknown.

    name_prefix : str
        Image and stage-label prefix, such as Position or ROI tile.

    Returns
    -------
    str
        OME-XML describing every series with its optional physical stage label.
    """
    channel_count = int(shapes_tczyx[0][1])
    channel_names = list(
        channels or (f"channel-{index}" for index in range(channel_count))
    )
    ome_pixel_type = {
        "float32": "float",
        "float64": "double",
        "complex64": "complex",
        "complex128": "double-complex",
    }.get(dtype.name, dtype.name)
    images = []
    for series, shape in enumerate(shapes_tczyx):
        t, c, z, y, x = (int(value) for value in shape)
        name = f"{name_prefix} {series}"
        stage_label = None
        if origins_zyx_um is not None:
            origin_z, origin_y, origin_x = (
                round_spatial(value) for value in origins_zyx_um[series]
            )
            stage_label = StageLabel(
                name=name,
                x=origin_x,
                x_unit=UnitsLength.MICROMETER,
                y=origin_y,
                y_unit=UnitsLength.MICROMETER,
                z=origin_z,
                z_unit=UnitsLength.MICROMETER,
            )
        images.append(
            Image(
                id=f"Image:{series}",
                name=name,
                stage_label=stage_label,
                pixels=Pixels(
                    id=f"Pixels:{series}",
                    dimension_order="XYZCT",
                    type=ome_pixel_type,
                    size_x=x,
                    size_y=y,
                    size_z=z,
                    size_c=c,
                    size_t=t,
                    time_increment=time_interval_s,
                    time_increment_unit="s",
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


def open_position_collection(path: str | Path) -> PositionCollection:
    """Open a yaozarrs Bf2Raw collection as ordered TensorStore arrays.

    Parameters
    ----------
    path : str | Path
        OME-Zarr directory to open.

    Returns
    -------
    PositionCollection
        Ordered array handles, pyramid levels, and physical metadata.
    """
    root = open_group(path)

    ome_group = root["OME"]
    series_metadata = ome_group.ome_metadata()
    collection_path = Path(path).expanduser().resolve()
    ome = from_xml(
        (collection_path / "OME" / "METADATA.ome.xml").read_text(encoding="utf-8")
    )
    arrays = []
    series_levels = []
    time_intervals = []
    spatial_origins = []
    voxel_sizes = []
    factors: tuple[int, ...] | None = None
    downsample = None
    for name in series_metadata.series:
        image_group = root[name]
        image_metadata = image_group.ome_metadata()
        time_intervals.append(read_time_interval(image_metadata.multiscales[0]))
        datasets = image_metadata.multiscales[0].datasets
        level_arrays = tuple(
            image_group[dataset.path].to_tensorstore() for dataset in datasets
        )
        sampling_factors, sampling_method = pyramid_yx_sampling(
            image_metadata.multiscales[0]
        )
        if factors is not None and (
            factors != sampling_factors or downsample != sampling_method
        ):
            raise ValueError("Position series disagree on pyramid sampling")
        factors, downsample = sampling_factors, sampling_method
        spatial_axes = image_metadata.multiscales[0].axes[-3:]
        units_to_um = {"micrometer": 1.0, "nanometer": 0.001, "millimeter": 1000.0}
        if [axis.name for axis in spatial_axes] != list("zyx") or any(
            axis.unit not in units_to_um for axis in spatial_axes
        ):
            raise ValueError("Position series require calibrated NGFF ZYX spatial axes")
        voxel_sizes.append(
            tuple(
                float(value) * units_to_um[axis.unit]
                for value, axis in zip(
                    datasets[0].scale_transform.scale[-3:], spatial_axes, strict=True
                )
            )
        )
        arrays.append(level_arrays[0])
        series_levels.append(level_arrays)
        translation = datasets[0].translation_transform
        spatial_origins.append(
            tuple(
                float(value) * units_to_um[axis.unit]
                for value, axis in zip(
                    (0.0, 0.0, 0.0)
                    if translation is None
                    else translation.translation[-3:],
                    spatial_axes,
                    strict=True,
                )
            )
        )
    arrays = tuple(arrays)
    if any(value != time_intervals[0] for value in time_intervals):
        raise ValueError("Position series disagree on physical time scale")
    attributes = {key: value for key, value in root.attrs.items() if key != "ome"}

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
    first_pixels = ome.images[0].pixels
    voxel_size_um = voxel_sizes[0]
    if any(value != voxel_size_um for value in voxel_sizes):
        raise ValueError("Position series disagree on physical voxel spacing")
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
    return PositionCollection(
        path=collection_path,
        arrays=arrays,
        attributes=attributes,
        multiscale_arrays=multiscale_arrays,
        multiscale_factors_yx=factors or (1,),
        multiscale_downsample=downsample,
        voxel_size_um=voxel_size_um,
        stage_positions_zxy=stage_positions,
        channel_names=channel_names,
        spatial_origins_zyx_um=tuple(spatial_origins),
        time_interval_s=time_intervals[0],
    )


def open_image_array(path: str | Path, level: str = "0") -> Any:
    """Open one dataset from a yaozarrs Image group.

    Parameters
    ----------
    path : str | Path
        OME-Zarr directory to open.
    level : str
        Dataset path within the image group, usually a pyramid level number.

    Returns
    -------
    Any
        TensorStore handle for the requested image dataset.
    """
    return open_group(path)[level].to_tensorstore()
