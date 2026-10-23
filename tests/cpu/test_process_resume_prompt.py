"""Read-only resume decisions protect saved photons and their completion journal.

Terminal decisions use controlled input; shared fixtures provide isolated temporary files
and journals containing known completion and pixel state.
"""

from contextlib import nullcontext
from unittest.mock import Mock

import pytest
import typer

from opm_processing.dataio.position_collection import create_position_collection
from opm_processing.process import confirm_partial_output_resume

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("answer", "expected", "damage", "planar"),
    [
        (answer, expected, None, planar)
        for answer, expected in (
            ("resume", True),
            ("overwrite", False),
            ("cancel", None),
        )
        for planar in (False, True)
    ]
    + [
        ("cancel", None, damage, False)
        for damage in ("missing-journal", "invalid-journal", "metadata")
    ],
)
def test_partial_output_decision_preserves_pixels_and_journal(
    partial_processed_run,
    stored_file_bytes,
    monkeypatch,
    planar,
    answer,
    expected,
    damage,
):
    """Apply the resume decision without prematurely changing durable output."""
    run = partial_processed_run(
        planar=planar,
        completed=[(t, p) for t in range(2) for p in range(3)] if damage else None,
    )
    if damage == "missing-journal":
        run.state.path.unlink()
    elif damage == "invalid-journal":
        run.state.path.write_text("{invalid json")
    elif damage == "metadata":
        (run.output / "0" / "zarr.json").unlink()
    before = stored_file_bytes(run.output_dir)
    monkeypatch.setattr(typer, "prompt", Mock(return_value=answer))
    with pytest.raises(typer.Abort) if expected is None else nullcontext():
        assert (
            confirm_partial_output_resume(
                run.acquisition,
                run.output_dir,
                deconvolve=False,
                max_projection=False,
                resume=False,
            )
            is expected
        )
    assert stored_file_bytes(run.output_dir) == before


@pytest.mark.parametrize(
    "selection,completed",
    [
        (
            {"time": None, "position": None},
            [(t, p) for t in range(2) for p in range(3)],
        ),
        ({"time": [1, 2], "position": [1, 3]}, [(1, 1), (1, 2)]),
    ],
)
def test_complete_selected_output_allows_overwrite(
    partial_processed_run, selection, completed
):
    """Allow a completed selected dataset to be intentionally overwritten."""
    run = partial_processed_run(selection=selection, completed=completed)
    assert not confirm_partial_output_resume(
        run.acquisition,
        run.output_dir,
        deconvolve=False,
        max_projection=False,
        resume=False,
    )


@pytest.mark.parametrize("deconvolve", [False, True])
def test_resume_and_distinct_output_families(partial_processed_run, deconvolve):
    """Keep resume decisions independent across processed output families."""
    run = partial_processed_run(deconvolve=deconvolve)
    assert confirm_partial_output_resume(
        run.acquisition,
        run.output_dir,
        deconvolve=deconvolve,
        max_projection=False,
        resume=True,
    )
    assert not confirm_partial_output_resume(
        run.acquisition,
        run.output_dir,
        deconvolve=not deconvolve,
        max_projection=False,
        resume=False,
    )


@pytest.mark.parametrize("requested", [False, True])
def test_projection_completion_only_matters_when_requested(
    partial_processed_run, monkeypatch, requested
):
    """Check projection completeness only when that output is requested."""
    run = partial_processed_run(completed=[(t, p) for t in range(2) for p in range(3)])
    maximum = run.output.with_name("sample_max_z_deskewed.ome.zarr")
    create_position_collection(maximum, (2, 3, 1, 1, 4, 5), (1, 0.115, 0.115))
    run.state.initialize_run(
        maximum,
        configuration={"selection": {"time": None, "position": None}},
        overwrite=True,
    )
    run.state.complete_tile(maximum, 0, 0)
    monkeypatch.setattr(typer, "prompt", lambda *a, **k: "resume")
    assert (
        confirm_partial_output_resume(
            run.acquisition,
            run.output_dir,
            deconvolve=False,
            max_projection=requested,
            resume=False,
        )
        is requested
    )
