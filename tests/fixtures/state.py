"""Durable processing and registration state fixtures for storage and ROI geometry."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from opm_processing.dataio.acquisition import inspect_acquisition
from opm_processing.dataio.position_collection import create_position_collection
from opm_processing.dataio.processing_state import ProcessingState


@pytest.fixture
def partial_processed_run(tmp_path, live_acquisition_factory):
    """Persist camera input, sentinel output pixels and a partial completion journal."""

    def create(*, planar=False, deconvolve=False, completed=None, selection=None):
        """Persist a controlled partial run and its durable completion journal."""
        shape = (2, 3, 1, 1 if planar else 3, 4, 5)
        dataset = live_acquisition_factory(shape=shape)
        destination = tmp_path / "processed"
        label = "2d" if planar else "deskewed"
        if deconvolve:
            label = f"decon_{label}"
        output = destination / f"sample_{label}.ome.zarr"
        collection = create_position_collection(output, shape, (1, 0.115, 0.115))
        collection.arrays[0][0, 0].write(
            np.full(collection.arrays[0].shape[-3:], 42, np.uint16)
        ).result()
        state = ProcessingState.create(
            destination / "sample.processing.json", dataset.path
        )
        state.initialize_run(
            output,
            configuration={"selection": selection or {"time": None, "position": None}},
            overwrite=True,
        )
        for time, position in completed if completed is not None else [(0, 0), (0, 1)]:
            state.complete_tile(output, time, position)
        return SimpleNamespace(
            acquisition=inspect_acquisition(dataset.path),
            output_dir=destination,
            output=output,
            state=state,
        )

    return create


@pytest.fixture
def stored_file_bytes():
    """Capture current file bytes to verify a read-only state decision cannot modify saved data."""
    return lambda root: {
        p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }


@pytest.fixture
def processing_state(tmp_path):
    """Create an empty durable journal for isolated processing state transitions."""
    source = tmp_path / "sample.ome.zarr"
    source.mkdir(exist_ok=True)
    return ProcessingState.create(tmp_path / "sample.processing.json", source)


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def record_registration(tmp_path):
    """Return a writer for measured tile origins and completed processing state."""

    def record(
        processed_path: Path,
        tiles: list[dict[str, object]],
        *,
        roi_series: tuple[dict[str, object], ...] = (),
    ) -> tuple[Path, ProcessingState]:
        """Persist measured placement and completed series for fusion and ROI selection."""
        stem = processed_path.name.split("_", 1)[0]
        source = tmp_path / f"{stem}.ome.zarr"
        source.mkdir(exist_ok=True)
        state = ProcessingState.create(tmp_path / f"{stem}.processing.json", source)
        state.initialize_run(
            processed_path,
            configuration={},
            roi_series=roi_series,
            overwrite=True,
        )
        for time_index, position_index in sorted(
            {(int(tile["time_index"]), int(tile["position_index"])) for tile in tiles}
        ):
            state.complete_tile(processed_path, time_index, position_index)
        state.save_registration(
            processed_path,
            configuration={},
            pairwise_metrics={},
        )
        fused = tmp_path / f"{stem}_fused.ome.zarr"
        maximum = tmp_path / f"{stem}_max_z_fused.ome.zarr"
        state.complete_registration(processed_path, fused_path=fused, tiles=tiles)
        state.set_registered_max_projection(
            processed_path,
            max_projection_path=maximum,
        )
        return maximum, state

    return record
