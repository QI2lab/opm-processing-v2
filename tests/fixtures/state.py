"""Durable processing and registration state fixtures for storage and ROI geometry."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from opm_processing.dataio.acquisition import inspect_acquisition
from opm_processing.dataio.position_collection import create_position_collection
from opm_processing.dataio.processing_state import (
    ProcessingState,
    processing_state_image_attributes,
)
from opm_processing.imageprocessing.tilefusion import _processed_store_identity


@pytest.fixture
def external_journal_image(acquisition_factory, fused_store_factory):
    """Write a raw source and linked fused image through the real storage writers."""
    pixels = np.zeros((1, 1, 1, 1, 2, 2), np.uint16)
    source = acquisition_factory(pixels, name="original/sample").path
    state = ProcessingState.create(source.parent / "sample.processing.json", source)
    image = source.parent / "rendered" / "sample_fused.ome.zarr"
    path = fused_store_factory(
        pixels[:, 0],
        name="original/rendered/sample_fused",
        attributes=processing_state_image_attributes(state.path, image),
    )
    return SimpleNamespace(source=source, state=state, image=path)


@pytest.fixture
def partial_processed_run(tmp_path, acquisition_factory):
    """Persist camera input, sentinel output pixels and a partial completion journal."""

    def create(*, planar=False, deconvolve=False, completed=None, selection=None):
        """Persist a controlled partial run and its durable completion journal."""
        shape = (2, 3, 1, 1 if planar else 3, 4, 5)
        dataset = acquisition_factory(
            np.zeros(shape, np.uint16),
            mode="stage",
            camera_conversion=0.24,
            stage_positions_zxy=tuple((30.0, 100.0, 200.0 + 2 * p) for p in range(3)),
        )
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
def processing_state(tmp_path):
    """Create an empty durable journal for isolated processing state transitions."""
    source = tmp_path / "sample.ome.zarr"
    source.mkdir(exist_ok=True)
    return ProcessingState.create(tmp_path / "sample.processing.json", source)


@pytest.fixture
def external_registration_state(request, tmp_path, processing_state, monkeypatch):
    """Prescribe another output directory and its filesystem-relative-path policy."""
    processed = tmp_path / "sample_deskewed.ome.zarr"
    destination = tmp_path.parent / f"{tmp_path.name}-fused"
    fused = destination / "sample_fused.ome.zarr"
    maximum = destination / "sample_max_z_fused.ome.zarr"
    cross_drive_calls = []
    if request.param:
        relative = os.path.relpath

        def different_drive(path, start):
            if path in (fused, maximum):
                cross_drive_calls.append(path)
                raise ValueError("path is on another Windows drive")
            return relative(path, start)

        monkeypatch.setattr(os.path, "relpath", different_drive)
    return SimpleNamespace(
        state=processing_state,
        processed=processed,
        fused=fused,
        maximum=maximum,
        cross_drive=bool(request.param),
        cross_drive_calls=cross_drive_calls,
    )


@pytest.fixture
def filesystem_replace_error(monkeypatch):
    """Inject a failure for one actual filesystem replacement and restore it afterward."""

    @contextmanager
    def interrupt(destination):
        """Fail the named destination while letting other real replacements execute."""
        replace = type(destination).replace

        def fail(path, target, *args, **kwargs):
            if Path(target).resolve() == destination.resolve():
                raise OSError("interrupted state replacement")
            return replace(path, target, *args, **kwargs)

        with monkeypatch.context() as filesystem:
            filesystem.setattr(type(destination), "replace", fail)
            yield

    return interrupt


@pytest.fixture
def record_registration(tmp_path):
    """Return a writer for measured tile origins and completed processing state."""

    def record(
        processed_path: Path,
        tiles: list[dict[str, object]],
        *,
        roi_series: tuple[dict[str, object], ...] = (),
    ) -> None:
        """Persist measured placement and completed series for fusion and ROI selection."""
        identity = _processed_store_identity(processed_path)
        assert identity is not None
        stem, _suffix = identity
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

    return record
