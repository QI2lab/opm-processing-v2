"""Map legacy raw OPM arrays to referenced Bio-Formats2Raw OME-Zarr.

Only OME metadata and a refkv archive are written. Compressed camera chunks
remain in the original Zarr v2 store and are resolved on demand.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any
from urllib.parse import quote

import numpy as np
import typer
from yaozarrs import v05

from opm_processing.dataio.acquisition import acquisition_stem, inspect_acquisition
from opm_processing.dataio.legacy_metadata import read_legacy_metadata
from opm_processing.dataio.position_collection import _build_ome_xml

if TYPE_CHECKING:
    from collections.abc import Iterator

    from refkv import ReferenceStore

app = typer.Typer(pretty_exceptions_enable=False)


def resolve_legacy_source(path: Path) -> Path:
    """Find one legacy bare Zarr v2 array without parsing its frame history."""
    path = path.expanduser().resolve()
    if (path / ".zarray").is_file():
        return path
    candidates = sorted(
        child
        for child in path.iterdir()
        if child.is_dir() and (child / ".zarray").is_file()
    )
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one legacy raw array in {path}; found {len(candidates)}"
        )
    return candidates[0]


def _descriptor(raw: Path) -> dict[str, Any]:
    descriptor = json.loads((raw / ".zarray").read_text())
    shape, chunks = descriptor.get("shape", []), descriptor.get("chunks", [])
    if (
        descriptor.get("zarr_format") != 2
        or descriptor.get("dtype") != "<u2"
        or descriptor.get("order") != "C"
        or descriptor.get("filters") not in (None, [])
        or len(shape) != 6
        or len(chunks) != 6
        or any(type(v) is not int or v < 1 for v in (*shape, *chunks))
        or chunks[1] != 1
        or descriptor.get("compressor", {}).get("id") != "blosc"
        or descriptor.get("dimension_separator", ".") not in ("/", ".")
    ):
        raise ValueError(
            "Require a C-order uint16 TPCZYX Zarr v2 array with singleton position chunks, Blosc and no filters"
        )
    return descriptor


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def _headers(
    metadata: dict[str, Any], descriptor: dict[str, Any], source: Path
) -> dict[str, bytes]:
    """Describe the standard Bio-Formats2Raw hierarchy inside the archive."""
    shape = metadata["shape_tpczyx"]
    t, positions, c, z, y, x = shape
    shape_tczyx = [t, c, z, y, x]
    chunks = descriptor["chunks"]
    chunks_tczyx = [chunks[0], *chunks[2:]]
    compressor = descriptor["compressor"]
    shuffle = compressor.get("shuffle", 1)
    # Original uint16 AUTOSHUFFLE means byte shuffle; Blosc's self-describing
    # headers and compressed bytes are retained unchanged.
    shuffle_name = {-1: "shuffle", 0: "noshuffle", 1: "shuffle", 2: "bitshuffle"}[
        shuffle
    ]
    array = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": shape_tczyx,
        "data_type": "uint16",
        "dimension_names": list("tczyx"),
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": chunks_tczyx},
        },
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": 0,
        "codecs": [
            {"name": "bytes", "configuration": {"endian": "little"}},
            {
                "name": "blosc",
                "configuration": {
                    "cname": compressor.get("cname", "lz4"),
                    "clevel": compressor.get("clevel", 5),
                    "shuffle": shuffle_name,
                    "typesize": 2,
                    "blocksize": compressor.get("blocksize", 0),
                },
            },
        ],
        "attributes": {},
    }
    manifest = {
        "format_version": 1,
        "archive": "references.vzip",
        "refkv_version": "0.2.0",
        "source": str(source),
        "legacy_compressor": compressor,
        "source_metadata_sha256": metadata["report"]["source_metadata_sha256"],
        "raw_pixels_copied": False,
    }
    attributes = dict(metadata["root_attributes"])
    attributes.update(
        {
            "ome": v05.Bf2Raw.model_validate({"bioformats2raw.layout": 3}).model_dump(
                mode="json", by_alias=True, exclude_none=True
            ),
            "opm_processing_refkv": manifest,
        }
    )
    headers = {
        "zarr.json": _json_bytes(
            {"zarr_format": 3, "node_type": "group", "attributes": attributes}
        ),
        "OME/zarr.json": _json_bytes(
            {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {
                    "ome": v05.Series(
                        series=[str(p) for p in range(positions)]
                    ).model_dump(mode="json", by_alias=True, exclude_none=True)
                },
            }
        ),
    }
    axes = [
        {"name": "t", "type": "time"},
        {"name": "c", "type": "channel"},
        *({"name": a, "type": "space", "unit": "micrometer"} for a in "zyx"),
    ]
    image = v05.Image(
        multiscales=[
            v05.Multiscale(
                axes=axes,
                datasets=[
                    v05.Dataset(
                        path="0",
                        coordinateTransformations=[
                            v05.ScaleTransformation(scale=metadata["sampling_tczyx"])
                        ],
                    )
                ],
            )
        ],
        omero={
            "channels": [
                {
                    "label": name,
                    "color": "FFFFFF",
                    "window": {"start": 0, "end": 65535, "min": 0, "max": 65535},
                }
                for name in metadata["channel_names"]
            ]
        },
    ).model_dump(mode="json", by_alias=True, exclude_none=True)
    origins = []
    for p, frames in enumerate(metadata["frame_sets"]):
        headers[f"{p}/zarr.json"] = _json_bytes(
            {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {"ome": image, "ome_writers": {"frame_metadata": frames}},
            }
        )
        headers[f"{p}/0/zarr.json"] = _json_bytes(array)
        stage = frames[0]["mda_event"]["metadata"]["Stage"]
        origins.append((stage["z_pos"], stage["y_pos"], stage["x_pos"]))
    headers["OME/METADATA.ome.xml"] = _build_ome_xml(
        shapes_tczyx=[shape_tczyx] * positions,
        dtype=np.dtype("uint16"),
        voxel_size_um=metadata["sampling_tczyx"][-3:],
        origins_zyx_um=origins,
        channels=metadata["channel_names"],
        time_interval_s=None,
        name_prefix="Raw position",
    ).encode()
    return headers


def _chunk_indices(descriptor: dict[str, Any]) -> Iterator[tuple[int, ...]]:
    return itertools.product(
        *(
            range(math.ceil(n / c))
            for n, c in zip(descriptor["shape"], descriptor["chunks"], strict=True)
        )
    )


def _build_reference_store(
    raw: Path,
    destination: Path,
    descriptor: dict[str, Any],
    headers: dict[str, bytes],
    *,
    workers: int,
) -> tuple[ReferenceStore, int, int]:
    """Build native chunk references using the mapper's single layout definition.

    Metadata inference is separate: callers supply validated hierarchy headers,
    so the same format writer can describe independently acquired timepoints.
    Source pixels and their codecs remain untouched.
    """
    import refkv

    archive = destination / "references.vzip"
    values: dict[str, Any] = dict(headers)
    sources = []
    expected = math.prod(
        math.ceil(n / c)
        for n, c in zip(descriptor["shape"], descriptor["chunks"], strict=True)
    )
    separator = descriptor.get("dimension_separator", ".")

    def describe(indices: tuple[int, ...]) -> tuple[str, Any, int]:
        source_key = separator.join(map(str, indices))
        chunk_path = raw / source_key
        stat = (
            chunk_path.stat()
        )  # Missing raw chunks are errors, never invented zero frames.
        if stat.st_size < 16 or not chunk_path.is_file():
            raise ValueError(f"Invalid raw Blosc chunk: {chunk_path}")
        t, p, c, z, y, x = indices
        key = f"{p}/0/c/{t}/{c}/{z}/{y}/{x}"
        try:
            reference_url = quote(
                os.path.relpath(chunk_path, destination).replace(os.sep, "/"), safe="/"
            )
        except ValueError:
            # Separate Windows drives cannot be expressed as relative paths.
            reference_url = chunk_path.as_uri()
        return key, refkv.Source(url=reference_url, size=stat.st_size), stat.st_size

    indices = iter(_chunk_indices(descriptor))
    count = 0
    compressed_bytes = 0
    print(f"Referencing {expected:,} original compressed chunks...", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while batch := list(itertools.islice(indices, 2048)):
            for key, entry, size in pool.map(describe, batch):
                source_index = len(sources)
                sources.append(entry)
                values[key] = refkv.Range(source=source_index, offset=0, length=size)
                count += 1
                compressed_bytes += size
            if count % 32768 == 0 or count == expected:
                print(f"Referenced {count:,}/{expected:,} chunks", flush=True)
    store = refkv.ReferenceStore(
        values, sources, url=archive.as_uri(), compressed=headers, page_size=65536
    )
    return store, count, compressed_bytes


def _write_reference_headers(destination: Path, headers: dict[str, bytes]) -> None:
    """Write the metadata facade that rejects readers without reference support."""
    for key, data in headers.items():
        # The on-disk hierarchy is a metadata facade. Unsupported readers must
        # fail rather than return fill-zero images for the absent pixel files.
        if key.endswith("/0/zarr.json"):
            node = json.loads(data)
            node["storage_transformers"] = [
                {
                    "name": "opm-processing-refkv",
                    "configuration": {"archive": "references.vzip"},
                }
            ]
            data = _json_bytes(node)
        target = destination / key
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def map_legacy_acquisition(
    source: str | Path, output: str | Path | None = None, *, workers: int = 8
) -> Path:
    """Create a referenced Bio-Formats2Raw acquisition without copying pixels.

    Parameters
    ----------
    source : str or pathlib.Path
        Legacy raw Zarr v2 store or the directory containing it.
    output : str or pathlib.Path or None
        New OME-Zarr mapping directory; defaults beside the original store.
    workers : int
        Parallel filesystem stat requests while validating referenced chunks.

    Returns
    -------
    pathlib.Path
        Published OME-Zarr directory readable by inspect-opm and process.
    """
    import refkv

    if not 1 <= workers <= 32:
        raise ValueError("workers must be between 1 and 32")
    raw = resolve_legacy_source(Path(source))
    output = (
        raw.with_name(acquisition_stem(raw) + ".ome.zarr")
        if output is None
        else Path(output).expanduser().resolve()
    )
    if output.exists():
        raise FileExistsError(f"Mapping destination already exists: {output}")
    if output == raw or raw in output.parents:
        raise ValueError("Mapping must be outside the original raw store")
    descriptor = _descriptor(raw)
    print("Validating legacy frame metadata and stage trajectories...", flush=True)
    metadata = read_legacy_metadata(raw)
    headers = _headers(metadata, descriptor, raw)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.mapping-{uuid.uuid4().hex}")
    staging.mkdir()
    archive = staging / "references.vzip"
    store, count, compressed_bytes = _build_reference_store(
        raw, staging, descriptor, headers, workers=workers
    )
    print("Writing the reference archive and OME metadata...", flush=True)
    summary = store.save(archive, mirror_refs=False)
    _write_reference_headers(staging, headers)
    inspected = inspect_acquisition(staging)
    inspected.validate_processing_calibration()
    if list(inspected.shape) != descriptor["shape"]:
        raise ValueError("Mapped acquisition shape does not match original raw shape")
    report = {
        **metadata["report"],
        "layout": "bioformats2raw",
        "ome_version": "0.5",
        "refkv_version": refkv.__version__,
        "referenced_chunks": count,
        "compressed_source_bytes": compressed_bytes,
        "archive_bytes": summary.bytes,
        "pixel_payload_bytes_copied": 0,
        "raw_store_required": True,
        "mapped_output": str(output),
        "metadata_inspection_passed": True,
    }
    (staging / "mapping_report.json").write_bytes(_json_bytes(report))
    # Publish only a complete, inspectable mapping; never overwrite or move raw data.
    if staging.resolve().parent != output.parent.resolve() or staging.is_symlink():
        raise ValueError("Unexpected mapping staging directory")
    staging.rename(output)
    print(f"Mapped acquisition: {output}", flush=True)
    return output


@app.command()
def map_command(
    source: Annotated[
        Path, typer.Argument(help="Legacy raw store or containing directory.")
    ],
    output: Annotated[
        Path | None, typer.Option(help="New Bio-Formats2Raw OME-Zarr mapping.")
    ] = None,
    workers: Annotated[int, typer.Option(min=1, max=32)] = 8,
) -> None:
    """Create a reference-backed OME-Zarr acquisition for the modern processor."""
    map_legacy_acquisition(source, output, workers=workers)


def main() -> None:
    """Run the legacy Bio-Formats2Raw reference mapping command."""
    app()


if __name__ == "__main__":
    main()
