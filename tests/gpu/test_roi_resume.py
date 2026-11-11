"""Command-level ROI registration, fusion, projection and default resume."""

import numpy as np
import pytest

from opm_processing import process_roi as roi_command
from opm_processing.dataio.processing_state import ProcessingState

pytestmark = [pytest.mark.usefixtures("cupy_gpu")]


@pytest.mark.integration
def test_roi_command_fuses_selected_tiles_and_projects_pixels(
    roi_run, assert_roi_point_accuracy, monkeypatch
):
    """Recover the original optical emitters through ROI fusion and max projection."""
    from yaozarrs import open_group

    from opm_processing.dataio.position_collection import open_image_array

    run = roi_run
    checkpoints = []
    complete_channel = ProcessingState.complete_channel

    def observe_checkpoint(state, *args, **kwargs):
        """Count durable production writes across two default-resume commands."""
        checkpoints.append(args)
        return complete_channel(state, *args, **kwargs)

    monkeypatch.setattr(ProcessingState, "complete_channel", observe_checkpoint)
    options = {
        "registration_channel": 1,
        "deconvolve": False,
        "flatfield_correction": False,
        "save_float32": True,
    }
    roi_command.process_roi(run.source, run.roi_path, **options)
    fused_path = run.output_dir / "sample_fused.ome.zarr"
    fused = open_image_array(fused_path).read().result()
    projected = (
        open_image_array(run.output_dir / "sample_max_z_fused.ome.zarr").read().result()
    )
    dataset = open_group(fused_path).ome_metadata().multiscales[0].datasets[0]
    state = ProcessingState.read(run.output_dir / "sample.processing.json")
    source_tiles = run.output_dir / "sample_deskewed.ome.zarr"
    records = state.roi_series(source_tiles)
    for field in ("skewed_bounds_sx", "deskew_crop_yx"):
        assert all(record[field] == records[0][field] for record in records)
    # The prescribed source plans are coincident and every view has identical
    # crop/support. Correct registration must retain that known placement,
    # proving equal contributor weights for the original-dose mean below.
    expected_origin = (
        round(run.metadata.pixel_size_um / 2, 3),
        run.roi.bounds_yx_um[0],
        run.roi.bounds_yx_um[2],
    )
    registration = state.registration(source_tiles)
    np.testing.assert_allclose(
        [tile["origin_zyx_um"] for tile in registration["tiles"]],
        np.broadcast_to(expected_origin, (len(records), 3)),
        rtol=0,
        atol=1e-6,
    )
    assert_roi_point_accuracy(
        fused,
        run.truth,
        dataset.translation_transform.translation[-3:],
        dataset.scale_transform.scale[-3:],
    )
    # Projection is a separate numerical reduction, checked with NumPy after
    # the stored fused volume passes the original-object accuracy requirement.
    np.testing.assert_array_equal(projected, np.max(fused, axis=2, keepdims=True))
    assert state.completed_tiles(run.output_dir / "sample_deskewed.ome.zarr") == {
        (0, 1),
        (0, 2),
    }
    assert len(checkpoints) == 6
    completed = checkpoints.copy()
    roi_command.process_roi(run.source, run.roi_path, **options)
    assert checkpoints == completed
