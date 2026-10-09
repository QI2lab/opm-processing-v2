"""Durable processing and registration state fixtures for storage and ROI geometry."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from opm_processing.dataio.processing_state import ProcessingState


@pytest.fixture
def processing_state(tmp_path):
    """Create an empty durable journal for isolated processing state transitions.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory for the source identity and durable journal.

    Returns
    -------
    ProcessingState
        Journal bound to a temporary sample source directory. Image workflow
        integration fixtures persist their camera data separately.
    """
    source = tmp_path / "sample.ome.zarr"
    source.mkdir(exist_ok=True)
    return ProcessingState.create(tmp_path / "sample.processing.json", source)


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def record_registration(tmp_path):
    """Return a writer for measured tile origins and completed processing state.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory containing the acquisition, journal and derived image paths.

    Returns
    -------
    callable
        Writer persisting physical tile origins and completed ROI series.
    """

    def record(
        processed_path: Path,
        tiles: list[dict[str, object]],
        *,
        roi_series: tuple[dict[str, object], ...] = (),
    ) -> tuple[Path, ProcessingState]:
        """Persist measured placement and completed series for fusion and ROI selection.

        Parameters
        ----------
        processed_path : pathlib.Path
            On-disk processed position collection.
        tiles : list of dict
            Known time/position indices and laboratory voxel origins in micrometers.
        roi_series : tuple of dict
            Optional camera geometry and per-series crop records.

        Returns
        -------
        tuple
            Registered projection path and durable processing state.
        """
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
