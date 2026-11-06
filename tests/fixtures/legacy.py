"""Native library-backed legacy arrays and reference archives for format units."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import refkv
import tensorstore as ts
import zarr

from opm_processing.dataio.acquisition import inspect_acquisition
from opm_processing.dataio.legacy_reference import _headers


@pytest.fixture(
    params=[
        ("absolute", True, True, False),
        ("absolute", True, True, True),
        ("absolute", False, True, False),
        ("absolute", False, False, False),
        ("relative", True, False, False),
        ("nested", True, False, False),
        ("nested", False, True, False),
        ("windows", False, True, False),
    ],
    ids=(
        "external-priority",
        "copied-live-local-priority",
        "relocated-absolute",
        "unavailable-absolute",
        "relative",
        "nested-relative",
        "relocated-nested",
        "portable-windows-basename",
    ),
)
def store_link_case(request, fs):
    """Prescribe link provenance and available stores on a shared fake filesystem."""
    kind, original_exists, local_exists, prefer_local = request.param
    directory = Path("/store-links/current").resolve()
    fs.create_dir(directory)
    local = directory / "sample.ome.zarr"
    if kind == "absolute":
        recorded = directory.parent / "original" / local.name
        original = recorded
    elif kind == "windows":
        recorded = Path(r"Z:\unavailable\sample.ome.zarr")
        original = recorded
    else:
        recorded = Path(local.name if kind == "relative" else f"nested/{local.name}")
        original = directory / recorded
    if original_exists:
        fs.create_dir(original)
        fs.create_file(original / "identity", contents="recorded")
    if local_exists:
        fs.create_dir(local)
        fs.create_file(local / "identity", contents="colocated")
    expected = (
        local if local_exists and (prefer_local or not original_exists) else original
    )
    return SimpleNamespace(
        directory=directory,
        recorded=recorded,
        prefer_local=prefer_local,
        expected=expected.resolve(),
        identity="colocated" if expected == local and local_exists else "recorded",
        available=original_exists or local_exists,
    )


@pytest.fixture
def legacy_raw_factory(tmp_path):
    """Persist known camera values through TensorStore's actual Zarr v2 writer.

    Returns
    -------
    callable
        Create a native array and explicitly prescribed historical frame records.
    """

    def create(
        *,
        name="legacy",
        channels=1,
        direction=1,
        interleaved=False,
        shape=None,
        chunks=None,
    ):
        """Write a legacy format unit with independently numbered pixel values."""
        shape = tuple(shape or (1, 2, channels, 3, 4, 5))
        channels = shape[2]
        chunks = tuple(chunks or (1, 1, 1, 1, shape[-2], shape[-1]))
        path = tmp_path / f"{name}.zarr"
        compressor = {
            "id": "blosc",
            "cname": "lz4",
            "clevel": 5,
            "shuffle": -1,
            "blocksize": 0,
        }
        array = ts.open(
            {
                "driver": "zarr",
                "kvstore": {"driver": "file", "path": str(path) + "/"},
                "metadata": {
                    "shape": shape,
                    "chunks": chunks,
                    "dtype": "<u2",
                    "compressor": compressor,
                    "fill_value": None,
                    "order": "C",
                    "filters": None,
                    "dimension_separator": "/",
                },
            },
            create=True,
        ).result()
        data = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape) + 100
        array.write(data).result()
        states = [False, True, False, channels == 2, False]
        names = ["488nm", "561nm"][:channels]
        frames = []
        for t, p in itertools.product(range(shape[0]), range(shape[1])):
            pairs = (
                itertools.product(range(shape[3]), range(channels))
                if interleaved
                else (
                    (z, c)
                    for c, z in itertools.product(range(channels), range(shape[3]))
                )
            )
            for z, c in pairs:
                frames.append(
                    {
                        "format": "frame-dict",
                        "exposure_ms": (10.0303823529, 15.02)[c],
                        "pixel_size_um": 0.115,
                        "mda_event": {
                            "index": {"t": t, "p": p, "c": c, "z": z},
                            "metadata": {
                                "DAQ": {
                                    "mode": "stage",
                                    "scan_axis_step_um": 0.4,
                                    "active_channels": states,
                                    "laser_powers": [0, 20, 0, 35, 0],
                                    "exposure_channels_ms": [5, 10.03, 5, 15.02, 5],
                                    "current_channel": names[c],
                                },
                                "Camera": {
                                    "exposure_ms": (10.03, 15.02)[c],
                                    "camera_crop_x": shape[-1],
                                    "camera_crop_y": shape[-2],
                                    "offset": 100,
                                    "e_to_ADU": 0.24,
                                },
                                "OPM": {
                                    "angle_deg": 30,
                                    "camera_Zstage_orientation": "negative",
                                    "camera_XYstage_orientation": "positive",
                                    "camera_mirror_orientation": "positive",
                                    "excess_scan_positions": 0,
                                },
                                "Stage": {
                                    "x_pos": 100.123456 + p * 20 + direction * z * 0.4,
                                    "y_pos": 200.987654,
                                    "z_pos": 30.123456,
                                },
                            },
                        },
                    }
                )
        # Zarr-python edits attributes only; all camera pixels use TensorStore.
        zarr.open_array(str(path), mode="r+").attrs["frame_metadatas"] = frames
        return SimpleNamespace(path=path, frames=frames, data=data, array=array)

    return create


@pytest.fixture
def refkv_mapping_factory(tmp_path, legacy_raw_factory):
    """Build real reference archives from native chunk files, including edge chunks.

    Returns
    -------
    callable
        Create a mapped raw format case or inject an explicitly requested bad payload.
    """

    def create(*, missing_key=None, corrupt_key=None):
        """Use production OME metadata and refkv's real archive writer."""
        raw = legacy_raw_factory(shape=(2, 3, 2, 5, 7, 9), chunks=(1, 1, 1, 2, 4, 5))
        root = tmp_path / "mapped.ome.zarr"
        root.mkdir()
        frame_sets = []
        for p in range(raw.data.shape[1]):
            initial = copy.deepcopy(
                [
                    frame
                    for frame in raw.frames
                    if frame["mda_event"]["index"]["t"] == 0
                    and frame["mda_event"]["index"]["p"] == p
                    and frame["mda_event"]["index"]["z"] < 2
                ]
            )
            for frame in initial:
                stage = frame["mda_event"]["metadata"]["Stage"]
                stage.update({key: round(value, 3) for key, value in stage.items()})
            frame_sets.append(initial)
        descriptor = json.loads((raw.path / ".zarray").read_text())
        metadata = {
            "shape_tpczyx": list(raw.data.shape),
            "sampling_tczyx": [1, 1, 0.4, 0.115, 0.115],
            "channel_names": ["488nm", "561nm"],
            "frame_sets": frame_sets,
            "root_attributes": {
                "opm_v2": {
                    "acquisition_order": ["t", "p", "c", "z"],
                    "configuration": {"acq_config": {"opm_mode": "stage"}},
                }
            },
            "report": {
                "source_metadata_sha256": {
                    name: hashlib.sha256((raw.path / name).read_bytes()).hexdigest()
                    for name in (".zarray", ".zattrs")
                }
            },
        }
        references = _headers(metadata, descriptor, raw.path)
        sources, source_files = [], []
        # Enumerate actual native keys. No test encoder, chunk-padding routine,
        # or filesystem adapter stands in for TensorStore or refkv.
        keys = sorted(raw.array.kvstore.list().result())
        for encoded_key in keys:
            source_key = encoded_key.decode()
            parts = source_key.split("/")
            if len(parts) != 6 or not all(part.isdecimal() for part in parts):
                continue
            t, p, c, z, y, x = parts
            key = f"{p}/0/c/{t}/{c}/{z}/{y}/{x}"
            raw_file = raw.path / source_key
            if key == corrupt_key:
                damaged = ts.open(
                    {
                        "driver": "zarr",
                        "kvstore": {
                            "driver": "file",
                            "path": str(tmp_path / "short.zarr") + "/",
                        },
                        "metadata": {
                            "shape": [3],
                            "chunks": [3],
                            "dtype": "<u2",
                            "compressor": descriptor["compressor"],
                        },
                    },
                    create=True,
                ).result()
                damaged.write(np.ones(3, np.uint16)).result()
                raw_file.write_bytes((tmp_path / "short.zarr/0").read_bytes())
            source_files.append(raw_file)
            if key != missing_key:
                sources.append(
                    refkv.Source(url=raw_file.as_uri(), size=raw_file.stat().st_size)
                )
                references[key] = refkv.Range(
                    source=len(sources) - 1, offset=0, length=raw_file.stat().st_size
                )
        refkv.ReferenceStore(references, sources).save(root / "references.vzip")
        for key, value in references.items():
            if isinstance(value, bytes):
                target = root / key
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(value)
        return inspect_acquisition(root), raw.data, source_files

    return create
