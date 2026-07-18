"""Test Bio-Formats2Raw-compatible position collections."""

import json

import numpy as np

from opm_processing.dataio.position_collection import (
    create_position_collection,
    open_position_collection,
)


def test_bf2raw_position_collection_round_trip(tmp_path):
    """Verify position collections round-trip data and metadata."""
    path = tmp_path / "positions.ome.zarr"
    collection = create_position_collection(
        path,
        (2, 3, 1, 2, 4, 5),
        (1.5, 0.3, 0.3),
        stage_positions=[[0, 10, 20], [0, 30, 40], [0, 50, 60]],
        channels=["488nm"],
        attributes={"camera_corrected": True},
    )

    writes = []
    for position, array in enumerate(collection.arrays):
        data = np.full(array.shape, position + 1, dtype=np.uint16)
        writes.append(array.write(data))
    for write in writes:
        write.result()

    reopened = open_position_collection(path)
    assert reopened.shape == (2, 3, 1, 2, 4, 5)
    assert reopened.attributes["stage_positions"][1] == [0, 30, 40]
    assert reopened.attributes["channels"] == ["488nm"]
    for position, array in enumerate(reopened.arrays):
        np.testing.assert_array_equal(array.read().result(), position + 1)

    root_metadata = json.loads((path / "zarr.json").read_text())
    ome_metadata = json.loads((path / "OME" / "zarr.json").read_text())
    array_metadata = json.loads((path / "0" / "0" / "zarr.json").read_text())
    assert root_metadata["attributes"]["ome"]["bioformats2raw.layout"] == 3
    assert ome_metadata["attributes"]["ome"]["series"] == ["0", "1", "2"]
    assert all(
        codec["name"] != "sharding_indexed" for codec in array_metadata["codecs"]
    )
