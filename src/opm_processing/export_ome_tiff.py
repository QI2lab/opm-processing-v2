"""Export level-zero fused volumes and fused max-Z images to lossless OME-BigTIFF.

Image tiles are streamed from TensorStore into zlib-compressed TIFFs. OME-XML
records calibrated TCZYX pixels and plane positions; structured annotations retain
NGFF metadata, source acquisition settings, and the processing journal.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated
from uuid import uuid4

import numpy as np
import tifffile
import typer
from ome_types import from_xml
from ome_types.model import (
    OME,
    AnnotationRef,
    Channel,
    Color,
    Image,
    MapAnnotation,
    Pixels,
    Plane,
    StageLabel,
    TiffData,
)
from tqdm import tqdm
from yaozarrs import open_group, v05

from opm_processing.dataio.acquisition import inspect_acquisition
from opm_processing.dataio.processing_state import (
    ProcessingState,
    processing_state_path,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    import tensorstore as ts

app = typer.Typer(pretty_exceptions_enable=False)


def image_tiles(
    array: ts.TensorStore, tile_shape: tuple[int, int]
) -> Iterator[np.ndarray]:
    """Read bounded YX tiles in TIFF page order, with Z varying fastest.

    Parameters
    ----------
    array : tensorstore.TensorStore
        Level-zero fused TCZYX pixels on disk.
    tile_shape : tuple[int, int]
        TIFF tile height and width, each a multiple of sixteen.

    Yields
    ------
    numpy.ndarray
        One YX tile in TCZ page order, then raster YX tile order. Edge tiles
        retain their actual shape and are zero-padded by tifffile. Separate
        progress bars report timepoints, channels, and Z planes per channel.
    """
    nt, nc, nz, ny, nx = array.shape
    ty, tx = tile_shape
    for t in tqdm(range(nt), desc="export time", unit="timepoint", position=0):
        for c in tqdm(
            range(nc), desc="export channels", unit="channel", position=1, leave=False
        ):
            for z in tqdm(
                range(nz),
                desc=f"export planes (channel {c + 1}/{nc})",
                unit="plane",
                position=2,
                leave=False,
            ):
                for y in range(0, ny, ty):
                    for x in range(0, nx, tx):
                        yield (
                            array[t, c, z, y : min(y + ty, ny), x : min(x + tx, nx)]
                            .read()
                            .result()
                        )


def export_image(
    source: Path,
    output: Path,
    template: OME,
    provenance: dict[str, str],
    *,
    workers: int = 4,
) -> None:
    """Write one fused image as tiled OME-BigTIFF without changing its pixels.

    Parameters
    ----------
    source : Path
        Fused volume or fused maximum-Z OME-Zarr Image group.
    output : Path
        Destination OME-TIFF file, replaced after a successful export.
    template : OME
        Source OME metadata containing available channels, instrument details,
        acquisition date, timing, and annotations. Its image geometry is
        replaced with the fused level-zero NGFF geometry.
    provenance : dict[str, str]
        JSON-encoded acquisition and processing records for OME annotations.
    workers : int
        Number of TIFF compression workers; reads and buffering remain bounded.
    """
    root = open_group(source)
    metadata = root.ome_metadata()
    multiscale = metadata.multiscales[0]
    dataset = multiscale.datasets[0]
    array = root[dataset.path].to_tensorstore()
    nt, nc, nz, ny, nx = array.shape
    dtype = np.dtype(array.dtype.numpy_dtype)
    scale = dataset.scale_transform.scale
    translation = dataset.translation_transform
    origin = translation.translation if translation is not None else [0] * 5

    ome = template.model_copy(deep=True)
    ome.uuid = None
    if ome.images:
        image = ome.images[0]
        pixels = image.pixels
    else:
        pixels = Pixels(
            id="Pixels:0",
            dimension_order="XYZCT",
            type="uint16",
            size_t=nt,
            size_c=nc,
            size_z=nz,
            size_y=ny,
            size_x=nx,
            channels=[
                Channel(id=f"Channel:0:{c}", name=f"channel-{c}", samples_per_pixel=1)
                for c in range(nc)
            ],
        )
        image = Image(id="Image:0", pixels=pixels)
    ome.images = [image]
    image.name = source.name.removesuffix(".ome.zarr")
    if source.name.endswith("_max_z_fused.ome.zarr"):
        image.description = "Maximum-Z projection of the registered fused volume; Z position is the projected volume center."
    image.stage_label = StageLabel(
        name="Fused origin", x=origin[4], y=origin[3], z=origin[2]
    )
    pixels.dimension_order = "XYZCT"
    pixels.type = {"float32": "float", "float64": "double"}.get(dtype.name, dtype.name)
    pixels.size_t, pixels.size_c, pixels.size_z = nt, nc, nz
    pixels.size_y, pixels.size_x = ny, nx
    pixels.physical_size_z, pixels.physical_size_y, pixels.physical_size_x = scale[-3:]
    pixels.physical_size_x_unit = pixels.physical_size_y_unit = (
        pixels.physical_size_z_unit
    ) = "µm"
    pixels.metadata_only = None
    pixels.bin_data_blocks = []
    pixels.tiff_data_blocks = [
        TiffData(ifd=0, first_z=0, first_c=0, first_t=0, plane_count=nt * nc * nz)
    ]
    pixels.big_endian = False
    pixels.interleaved = False

    if metadata.omero is not None:
        for channel, display in zip(
            pixels.channels, metadata.omero.channels, strict=False
        ):
            channel.name = display.label or channel.name
            if display.color is not None:
                channel.color = Color("#" + display.color)
    time_axis = multiscale.axes[0]
    if time_axis.unit == "second":
        pixels.time_increment, pixels.time_increment_unit = scale[0], "s"
    # Raw scan positions and acquisition-plane times do not describe fused voxels.
    # Retain channel exposures, then replace positions using the fused transform.
    exposures = {
        plane.the_c: (plane.exposure_time, plane.exposure_time_unit)
        for plane in pixels.planes
        if plane.exposure_time is not None
    }
    pixels.planes = []
    for t in range(nt):
        elapsed = None
        if time_axis.unit == "second":
            elapsed = origin[0] + t * scale[0]
        elif pixels.time_increment is not None:
            elapsed = t * pixels.time_increment
        for c in range(nc):
            exposure, exposure_unit = exposures.get(c, (None, "s"))
            for z in range(nz):
                pixels.planes.append(
                    Plane(
                        the_t=t,
                        the_c=c,
                        the_z=z,
                        position_x=origin[4],
                        position_y=origin[3],
                        position_z=origin[2] + z * scale[2],
                        exposure_time=exposure,
                        exposure_time_unit=exposure_unit,
                        delta_t=elapsed,
                        delta_t_unit=pixels.time_increment_unit,
                    )
                )
    annotation = MapAnnotation(
        id=f"Annotation:export-{uuid4().hex}",
        namespace="https://github.com/qi2lab/opm-processing-v2/export-ome-tiff",
        value={
            **provenance,
            "SourceOMEZarr": str(source),
            "NGFF": json.dumps(dict(root.attrs), ensure_ascii=False),
        },
    )
    ome.structured_annotations.map_annotations.append(annotation)
    image.annotation_refs.append(AnnotationRef(id=annotation.id))
    # Match expansion-processing's ome-writers TIFF backend: serialize the OME
    # model directly to UTF-8 bytes and disable automatic TIFF shape metadata.
    description = ome.to_xml().encode("utf-8")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
    try:
        with tifffile.TiffWriter(
            temporary, bigtiff=True, ome=False, shaped=False, byteorder="<"
        ) as writer:
            writer.write(
                image_tiles(array, (512, 512)),
                shape=array.shape,
                dtype=dtype,
                tile=(512, 512),
                photometric="minisblack",
                compression="zlib",
                compressionargs={"level": 8},
                predictor=True,
                description=description,
                metadata=None,
                resolution=(10_000 / scale[4], 10_000 / scale[3]),
                resolutionunit="CENTIMETER",
                software="opm-processing-v2",
                maxworkers=workers,
                buffersize=16 * 1024**2,
            )
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


@app.command(
    help="Export level-zero fused volume and maximum-Z OME-Zarr to compressed OME-BigTIFF."
)
def export_fused(
    root_path: Annotated[
        Path,
        typer.Argument(
            help="Acquisition/output directory or full fused OME-Zarr store."
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(help="Output directory; defaults to beside the fused stores."),
    ] = None,
    workers: Annotated[int, typer.Option(min=1, help="TIFF compression workers.")] = 4,
    overwrite: Annotated[
        bool, typer.Option(help="Replace existing OME-TIFF exports.")
    ] = False,
    metadata_only: Annotated[
        bool,
        typer.Option(
            help="Repair OME XML encoding in existing exports without rewriting pixels."
        ),
    ] = False,
) -> None:
    """Export the full-resolution fused volume and its fused maximum-Z image.

    Parameters
    ----------
    root_path : Path
        Acquisition/output directory containing one fused pair, or the full
        fused store itself. No reduced pyramid levels are exported.
    output : Path or None
        Destination directory, or None to write beside the source stores.
    workers : int
        Number of parallel compression workers, default four.
    overwrite : bool
        Replace existing exports only when explicitly enabled.
    metadata_only : bool
        Rewrite existing TIFF ImageDescription tags as UTF-8 OME XML. Image
        planes, compression, and metadata values are preserved; both exports
        must already exist. Source acquisition metadata is not reread.

    Raises
    ------
    ValueError
        If the directory does not identify one full fused store.
    FileNotFoundError
        If either fused store is missing, or metadata-only repair is requested
        without both existing TIFF exports.
    FileExistsError
        If an export exists and overwrite is disabled.
    """
    root_path = root_path.expanduser().resolve()
    if root_path.name.endswith("_fused.ome.zarr") and not root_path.name.endswith(
        "_max_z_fused.ome.zarr"
    ):
        fused = root_path
    else:
        stores = [
            path
            for path in root_path.glob("*_fused.ome.zarr")
            if not path.name.endswith("_max_z_fused.ome.zarr")
        ]
        if len(stores) != 1:
            raise ValueError(
                f"Expected one full fused OME-Zarr in {root_path}; found {len(stores)}."
            )
        fused = stores[0]
    stem = fused.name.removesuffix("_fused.ome.zarr")
    sources = [fused, fused.with_name(f"{stem}_max_z_fused.ome.zarr")]
    destination = output.expanduser().resolve() if output is not None else fused.parent
    targets = [
        destination / (source.name.removesuffix(".ome.zarr") + ".ome.tif")
        for source in sources
    ]
    for source, target in zip(sources, targets, strict=False):
        if not source.is_dir():
            raise FileNotFoundError(source)
        if metadata_only and not target.is_file():
            raise FileNotFoundError(target)
        if target.exists() and not overwrite and not metadata_only:
            raise FileExistsError(
                f"{target} exists; use --overwrite to replace exports."
            )

    if metadata_only:
        for target in targets:
            ome = from_xml(tifffile.tiffcomment(target))
            tifffile.tiffcomment(target, comment=ome.to_xml().encode("utf-8"))
            typer.echo(f"Repaired OME metadata: {target}")
        return

    template = OME(creator="opm-processing-v2")
    provenance = {}
    state_path = processing_state_path(fused.parent, stem)
    if state_path.is_file():
        state = ProcessingState.read(state_path)
        provenance["ProcessingState"] = json.dumps(state.document, ensure_ascii=False)
        processed = state.registered_output_for_fused(fused)
        source = Path(state.document["source"]["path"])
        if not source.is_absolute():
            source = state_path.parent / source
        for candidate in (source, processed):
            companion = candidate / "OME" / "METADATA.ome.xml"
            if companion.is_file():
                source_xml = companion.read_text(encoding="utf-8")
                template = from_xml(source_xml)
                provenance["SourceOME"] = source_xml
                break
        if source.is_dir():
            raw_root = open_group(source)
            if "opm_v2" in raw_root.attrs:
                acquisition = inspect_acquisition(source, root=raw_root)
                provenance["Acquisition"] = json.dumps(
                    acquisition.to_dict(), ensure_ascii=False
                )
                provenance["AcquisitionSettings"] = json.dumps(
                    raw_root.attrs["opm_v2"], ensure_ascii=False
                )
                raw_metadata = raw_root.ome_metadata()
                if isinstance(raw_metadata, v05.Bf2Raw):
                    raw_metadata = raw_root[
                        raw_root["OME"].ome_metadata().series[0]
                    ].ome_metadata()
                if not template.images:
                    template.images = [
                        Image(
                            id="Image:0",
                            pixels=Pixels(
                                id="Pixels:0",
                                dimension_order="XYZCT",
                                type="uint16",
                                size_x=1,
                                size_y=1,
                                size_z=1,
                                size_c=len(acquisition.channels),
                                size_t=1,
                            ),
                        )
                    ]
                pixels = template.images[0].pixels
                if not pixels.channels:
                    pixels.channels = [
                        Channel(id=f"Channel:0:{c.index}", samples_per_pixel=1)
                        for c in acquisition.channels
                    ]
                for channel, acquired in zip(
                    pixels.channels, acquisition.channels, strict=False
                ):
                    channel.name = acquired.name
                    if acquired.wavelength_nm is not None:
                        channel.excitation_wavelength = acquired.wavelength_nm
                        channel.excitation_wavelength_unit = "nm"
                    if acquired.exposure_ms is not None:
                        pixels.planes.append(
                            Plane(
                                the_t=0,
                                the_z=0,
                                the_c=acquired.index,
                                exposure_time=acquired.exposure_ms,
                                exposure_time_unit="ms",
                            )
                        )
                raw_multiscale = raw_metadata.multiscales[0]
                provenance["AcquisitionNGFF"] = raw_metadata.model_dump_json(
                    exclude_none=True
                )
                if raw_metadata.omero is not None:
                    for channel, display in zip(
                        pixels.channels, raw_metadata.omero.channels, strict=False
                    ):
                        if display.color is not None:
                            channel.color = Color("#" + display.color)
                if (
                    pixels.time_increment is None
                    and raw_multiscale.axes[0].unit == "second"
                ):
                    pixels.time_increment = raw_multiscale.datasets[
                        0
                    ].scale_transform.scale[0]
                    pixels.time_increment_unit = "s"
    for source, target in zip(sources, targets, strict=False):
        export_image(
            source,
            target,
            template,
            provenance,
            workers=workers,
        )
        typer.echo(f"Created {target}")


def main() -> None:
    """Run the fused OME-BigTIFF exporter CLI."""
    app()


if __name__ == "__main__":
    main()
