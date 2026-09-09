"""Ground-truth tests for the standalone processing-state contract."""

from __future__ import annotations

import json

import pytest

from opm_processing.dataio.processing_state import (
    PROCESSING_STATE_SCHEMA,
    PROCESSING_STATE_VERSION,
    ProcessingState,
    processing_state_path,
)


@pytest.mark.parametrize(
    ("save_float32", "dtype"),
    ((False, "uint16"), (True, "float32")),
)
def test_processing_state_round_trip_has_exact_durable_ground_truth(
    tmp_path,
    save_float32,
    dtype,
) -> None:
    """Persist exact run state independently of the selected output dtype."""
    source = tmp_path / "sample.ome.zarr"
    source.mkdir()
    path = processing_state_path(tmp_path, "sample")
    output = tmp_path / f"sample_{dtype}.ome.zarr"
    state = ProcessingState.open(path, source, overwrite=True)
    state.initialize_run(
        output,
        configuration={"save_float32": save_float32, "factors": (2, 4)},
        overwrite=True,
    )
    state.complete_tile(output, 0, 1, zero_channels=(0, 2))
    state.save_registration(
        output,
        configuration={"channel": 0},
        pairwise_metrics={"0,1": [1, 2, 3, 0.9]},
    )
    fused = tmp_path / "sample_fused.ome.zarr"
    maximum = tmp_path / "sample_max_z_fused.ome.zarr"
    state.complete_registration(
        output,
        fused_path=fused,
        tiles=(
            {
                "time_index": 0,
                "position_index": 1,
                "origin_zyx_um": (1.25, 2.5, 3.75),
            },
        ),
    )
    state.set_registered_max_projection(output, max_projection_path=maximum)

    expected_run = {
        "path": output.name,
        "configuration_sha256": (
            "73112a48df7287d1863e1ce4223ec102d4df8da71095aab6218ff7b970a793b9"
            if save_float32
            else "8413f287f65066224f139bf0d53e5e32714cc6ac0ed5e9b645c6a3fbd3f65a3a"
        ),
        "completed_tiles": [[0, 1]],
        "zero_channels": [[0, 1, 0], [0, 1, 2]],
    }
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document == {
        "schema": PROCESSING_STATE_SCHEMA,
        "schema_version": PROCESSING_STATE_VERSION,
        "source": {"path": str(source.resolve())},
        "outputs": {output.name: expected_run},
        "registration": {
            output.name: {
                "configuration_sha256": (
                    "5952bcdbfeafaeffc3a92bbc81bfa3c260e8acacd2501afc32cc4e87537b24d3"
                ),
                "pairwise_metrics": {"0,1": [1, 2, 3, 0.9]},
                "fused_path": fused.name,
                "max_projection_path": maximum.name,
                "tiles": [
                    {
                        "time_index": 0,
                        "position_index": 1,
                        "origin_zyx_um": [1.25, 2.5, 3.75],
                    }
                ],
            }
        },
    }

    reopened = ProcessingState.read(path)
    assert reopened.completed_tiles(output) == {(0, 1)}
    assert reopened.zero_channels(output) == {(0, 1, 0), (0, 1, 2)}
    assert reopened.registered_output_for_fused(fused) == output
    assert reopened.registered_output_for_max_projection(maximum) == output


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ({"schema": "old"}, "schema"),
        ({"schema_version": 0}, "version"),
        ({"outputs": []}, "outputs"),
    ),
)
def test_processing_state_rejects_noncurrent_contract(
    tmp_path,
    mutation,
    message,
) -> None:
    """Reject legacy or structurally invalid state documents."""
    source = tmp_path / "sample.ome.zarr"
    source.mkdir()
    path = processing_state_path(tmp_path, "sample")
    ProcessingState.create(path, source)
    document = json.loads(path.read_text(encoding="utf-8"))
    document.update(mutation)
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        ProcessingState.read(path)


def test_processing_state_resume_requires_exact_run_configuration(tmp_path) -> None:
    """Allow resume only when every output-affecting setting is unchanged."""
    source = tmp_path / "sample.ome.zarr"
    source.mkdir()
    path = processing_state_path(tmp_path, "sample")
    output = tmp_path / "sample_deskewed.ome.zarr"
    state = ProcessingState.create(path, source)
    state.initialize_run(
        output,
        configuration={"deconvolve": False},
        overwrite=False,
    )

    with pytest.raises(ValueError, match="incompatible"):
        state.initialize_run(
            output,
            configuration={"deconvolve": True},
            overwrite=False,
        )


def test_overwriting_one_run_preserves_siblings_and_invalidates_only_its_registration(
    tmp_path,
) -> None:
    """One JSON can safely track independent deskew and deconvolution outputs."""
    source = tmp_path / "sample.ome.zarr"
    source.mkdir()
    state = ProcessingState.create(
        processing_state_path(tmp_path, "sample"),
        source,
    )
    deskewed = tmp_path / "sample_deskewed.ome.zarr"
    deconvolved = tmp_path / "sample_decon_deskewed.ome.zarr"
    for output in (deskewed, deconvolved):
        state.initialize_run(
            output,
            configuration={"output": output.name},
            overwrite=True,
        )
        state.complete_tile(output, 0, 0)
        state.save_registration(
            output,
            configuration={"channel": 0},
            pairwise_metrics={},
        )

    state = ProcessingState.open(state.path, source, overwrite=True)
    state.initialize_run(
        deconvolved,
        configuration={"output": deconvolved.name, "recomputed": True},
        overwrite=True,
    )

    assert state.completed_tiles(deskewed) == {(0, 0)}
    assert state.completed_tiles(deconvolved) == set()
    assert state.registration(deskewed)["pairwise_metrics"] == {}
    with pytest.raises(ValueError, match="no registration"):
        state.registration(deconvolved)
