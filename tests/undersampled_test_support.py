"""In-memory acquisition and output boundaries for numerical CLI tests."""

from pathlib import Path
from types import SimpleNamespace
import importlib

import numpy as np
import tensorstore as ts

from opm_processing.dataio.acquisition import AcquisitionMetadata, ChannelMetadata
from opm_processing.dataio.processing_state import ProcessingState


def mock_acquisition(mode="mirror", shape=(1, 1, 1, 25, 65, 49)):
    """Describe a 0.8 um combined-laser scan without creating image files."""
    return AcquisitionMetadata(
        path=Path.cwd() / "__mock_undersampled__" / "both_lasers.ome.zarr",
        storage_format="opm-v2-ome-zarr-v3",
        mode=mode,
        axes=("t", "p", "c", "z", "y", "x"),
        shape=shape,
        array_paths=("0/0",),
        acquisition_order=("t", "p", "z", "c"),
        channels=(ChannelMetadata(0, "488 + 637", None, 10.0, None),),
        stage_positions_zxy=((4.0, 20.0, 30.0),),
        scan_start_positions_xyz=(),
        scan_end_positions_xyz=(),
        scan_axis="x",
        scan_axis_step_um=0.8,
        pixel_size_um=0.115,
        angle_deg=30.0,
        camera_offset=100.0,
        camera_conversion=0.5,
        excess_scan_positions=0,
        excess_scan_start_positions=0,
        excess_scan_end_positions=0,
        orientations=(),
        sidecar_paths=(),
    )


def mock_processing_store(monkeypatch, metadata, raw):
    """Replace file reads/writes with memory arrays, keeping processing real."""
    process = importlib.import_module("opm_processing.process")
    output = metadata.path.parent / "output"
    collections = {}
    state = ProcessingState(
        output / "both_lasers.processing.json",
        {"source": {"path": str(metadata.path)}, "outputs": {}, "registration": {}},
    )

    def create_collection(path, shape, voxel_size, **kwargs):
        arrays = tuple(
            ts.array(np.zeros((shape[0], *shape[2:]), dtype=kwargs["dtype"]))
            for _ in range(shape[1])
        )
        collection = SimpleNamespace(
            arrays=arrays, voxel_size_um=voxel_size, shape=shape, **kwargs
        )
        collections[Path(path)] = collection
        return collection

    monkeypatch.setattr(process, "inspect_acquisition", lambda _: metadata)
    monkeypatch.setattr(process, "open_acquisition_datastore", lambda _: ts.array(raw))
    monkeypatch.setattr(process, "_resolve_output_directory", lambda *args: output)
    monkeypatch.setattr(process, "create_position_collection", create_collection)
    monkeypatch.setattr(
        ProcessingState, "open", classmethod(lambda cls, *args, **kwargs: state)
    )
    monkeypatch.setattr(ProcessingState, "save", lambda self: None)
    return output, collections, state
