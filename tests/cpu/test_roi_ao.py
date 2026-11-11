"""Physical ROI correction selection from saved acquisition stage plans."""

import json
from dataclasses import replace

import pytest

from opm_processing.dataio.ao_results import select_roi_ao
from opm_processing.dataio.roi import PhysicalRoi

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "failure,recorded",
    [
        (None, False),
        (None, True),
        ("missing", False),
        ("ambiguous", False),
        ("bad-recorded", True),
    ],
)
def test_roi_ao_uses_union_area_and_saved_stage_depths(
    acquisition_metadata, tmp_path, failure, recorded
):
    """Keep stage identity separate from image placement and count overlaps once."""
    root = tmp_path
    acquisition = replace(
        acquisition_metadata,
        path=root / "calibration.ome.zarr",
        shape=(4, 6, *acquisition_metadata.shape[2:]),
        stage_positions_zxy=tuple((z, x, 0) for z in (5, 1) for x in (0, 1, 10)),
    )
    roi = PhysicalRoi(
        bounds_yx_um=(100, 110, 200, 212),
        source_path=root / "preview.ome.zarr",
        grid_origin_yx_um=(100, 200),
        pixel_size_yx_um=(1, 1),
        tile_footprints=tuple(
            {
                "time_index": time,
                "position_index": position,
                "bounds_yx_um": (
                    100,
                    110,
                    *[(200, 206), (201, 207), (204, 212)][position % 3],
                ),
            }
            for time in (0, 3)
            for position in range(6)
        ),
    )
    assert select_roi_ao(acquisition, roi) == []
    for time in (0, 2):
        for depth, z in enumerate((5, 1)):
            layer = (
                root
                / "calibration.ome_ao_grid_results"
                / f"time_{time}_z_{depth}_ao_grid_results"
            )
            (layer / "grid_pos_0" / "ao_results.zarr").mkdir(parents=True)
            (layer / "grid_pos_1" / "ao_results.zarr").mkdir(parents=True)
            if failure == "ambiguous":
                z = 5
            points = [{"z": z, "x": x, "y": 0} for x in (0, 1, 10)]
            ao = [points[0], points[2]][:: (-1 if time else 1)]
            if failure == "missing":
                points.pop()
            for name, data in (("ao", ao), ("exp", points)):
                (layer / f"{name}_stage_positions.json").write_text(json.dumps(data))
            if recorded:
                geometry = {
                    "schema_version": 1,
                    "units": "um",
                    "correction_scope": "entire_tile",
                    "position_indices": list(range(3 * depth, 3 * depth + 3)),
                    # Recorded assignment deliberately contradicts nearest XY.
                    "tile_ao_indices": [0, 0, 9]
                    if failure
                    else ([0, 0, 1] if time else [1, 1, 0]),
                }
                (layer / "ao_grid_geometry.json").write_text(json.dumps(geometry))
    if failure:
        message = {
            "missing": "do not identify ROI source tiles",
            "ambiguous": "multiple saved AO depth plans",
            "bad-recorded": "Invalid recorded AO mapping",
        }[failure]
        with pytest.raises(ValueError, match=message):
            select_roi_ao(acquisition, roi)
        return
    records = select_roi_ao(acquisition, roi)
    first, last = (0, 1) if recorded else (1, 0)
    assert [
        (r["time_index"], r["depth_index"], r["ao_time_index"], r["region_index"])
        for r in records
    ] == [(0, 0, 0, first), (0, 1, 0, first), (3, 0, 2, last), (3, 1, 2, last)]
    assert all(r["area_um2"] == 80 for r in records)
    assert all(sorted(r["candidate_areas_um2"].values()) == [70, 80] for r in records)
    assert all(
        r["assignment"].startswith("recorded" if recorded else "inferred")
        for r in records
    )
