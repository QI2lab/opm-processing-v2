"""Illumination function units and simulated processing CLI integration."""

from pathlib import Path

import numpy as np
import pytest
import tensorstore as ts

from opm_processing.dataio.position_collection import open_position_collection
from opm_processing.imageprocessing.camera import camera_correct, illumination_correct


@pytest.mark.integration
@pytest.mark.parametrize(
    "opm_v2_projection_zarr,output_flatfield",
    [(("projection", 2.0), False), (("projection", 1.25), True)],
    indirect=["opm_v2_projection_zarr"],
    ids=["source-field", "output-field"],
)
def test_processing_uses_existing_flatfield_with_separate_output(
    opm_v2_projection_zarr,
    tmp_path: Path,
    output_flatfield: bool,
) -> None:
    """Verify source reuse and output precedence through saved pixel values."""
    from opm_processing.dataio.acquisition import acquisition_stem
    from opm_processing.dataio.position_collection import open_position_collection
    from opm_processing.process import process, write_flatfield

    acquisition = opm_v2_projection_zarr
    output_dir = tmp_path / "processed"
    output_dir.mkdir()
    filename = f"{acquisition_stem(acquisition.path)}_flatfield.ome.tif"
    field_shape = (1, acquisition.raw_data.shape[2], *acquisition.raw_data.shape[-2:])
    source_field = np.full(field_shape, 2.0, dtype=np.float32)
    selected_field = source_field
    write_flatfield(
        acquisition.path.parent / filename,
        source_field,
        acquisition.metadata.pixel_size_um,
    )
    if output_flatfield:
        selected_field = np.full(field_shape, 1.25, dtype=np.float32)
        write_flatfield(
            output_dir / filename, selected_field, acquisition.metadata.pixel_size_um
        )

    process(
        root_path=acquisition.path,
        output=output_dir,
        flatfield_correction=True,
        deconvolve=False,
        save_float32=True,
        write_fused_max_projection_tiff=False,
    )
    saved = (
        open_position_collection(
            output_dir / f"{acquisition_stem(acquisition.path)}_projection.ome.zarr"
        )
        .arrays[0]
        .read()
        .result()
    )
    calibrated = camera_correct(
        acquisition.raw_data[:, 0, :, 0],
        acquisition.metadata.camera_offset,
        acquisition.metadata.camera_conversion,
    )
    expected = illumination_correct(calibrated, selected_field[0])[:, :, np.newaxis]
    np.testing.assert_allclose(saved, expected, rtol=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("channels", [1, 3])
def test_shared_flatfield_round_trips_and_is_reused_at_multiple_depths(
    tmp_path: Path, channels
):
    """The saved shared field is reused unchanged for every stage depth."""
    from opm_processing.process import (
        load_or_estimate_flatfield,
        read_current_flatfield,
        write_flatfield,
    )

    path = tmp_path / "flatfield.ome.tif"
    flatfields = np.linspace(0.5, 1.5, channels * 8 * 12, dtype=np.float32).reshape(
        1, channels, 8, 12
    )
    write_flatfield(path, flatfields, 0.115)
    actual = read_current_flatfield(path, flatfields.shape)
    assert actual is not None
    np.testing.assert_array_equal(actual, flatfields)
    actual = load_or_estimate_flatfield(
        path,
        ts.array(np.zeros((1, 2, channels, 4, 8, 12), dtype=np.uint16)),
        100,
        0.25,
        0.115,
        np.asarray(((10, 20, 30), (0, 20, 30))),
        detector_gain_x=None,
        signal_threshold=None,
        minimum_signal_fraction=0.01,
        signal_mask=None,
    )
    np.testing.assert_array_equal(actual, flatfields)


@pytest.mark.unit
@pytest.mark.parametrize(
    "software,depths",
    [
        ("opm-processing/basicpy-stage-depth-camera-contract-v10", 2),
        ("opm-processing/basicpy-stage-z-max-camera-contract-v12", 1),
    ],
)
def test_obsolete_flatfield_estimators_are_not_reused(tmp_path, software, depths):
    """Refresh depth-specific and maximum-of-depth-fits calibration caches."""
    from tifffile import imwrite

    from opm_processing.process import read_current_flatfield

    path = tmp_path / "old-flatfield.ome.tif"
    imwrite(
        path,
        np.ones((depths, 1, 8, 12), dtype=np.float32),
        photometric="minisblack",
        metadata={"axes": "ZCYX"},
        software=software,
    )
    assert read_current_flatfield(path, (1, 1, 8, 12)) is None


@pytest.mark.integration
@pytest.mark.parametrize("scan_planes", [1, 4])
def test_processing_applies_the_same_field_at_every_depth(
    tmp_path,
    constant_illumination_acquisition,
    scan_planes,
    assert_uniform_photon_accuracy,
):
    """Calibrate every channel and depth against analytically known photons."""
    from opm_processing.process import process, write_flatfield

    case = constant_illumination_acquisition
    acquisition = case.dataset
    write_flatfield(
        tmp_path / "sample_flatfield.ome.tif",
        case.fields,
        acquisition.metadata.pixel_size_um,
    )
    process(
        root_path=acquisition.path,
        output=tmp_path / "corrected",
        flatfield_correction=True,
        deconvolve=False,
        save_float32=True,
        z_downsample_level=1,
        max_projection=False,
        create_fused_max_projection=False,
        write_fused_max_projection_tiff=False,
    )
    suffix = "projection" if scan_planes == 1 else "deskewed"
    corrected = open_position_collection(
        tmp_path / "corrected" / f"sample_{suffix}.ome.zarr"
    )
    for position, array in enumerate(corrected.arrays):
        assert_uniform_photon_accuracy(
            array.read().result(), case.scene, position=position
        )


@pytest.mark.unit
@pytest.mark.parametrize("level", (0, -1, np.nan))
def test_invalid_cached_fields_are_not_reused(tmp_path, level):
    """Reject invalid fields at write time and refresh invalid existing caches."""
    from tifffile import imwrite

    from opm_processing.process import (
        FLATFIELD_SOFTWARE,
        read_current_flatfield,
        write_flatfield,
    )

    path = tmp_path / "invalid.ome.tif"
    field = np.full((1, 1, 5, 7), level, np.float32)
    with pytest.raises(ValueError, match="finite and positive"):
        write_flatfield(path, field, 0.115)
    assert not path.exists()
    imwrite(
        path,
        field,
        photometric="minisblack",
        metadata={"axes": "ZCYX"},
        software=FLATFIELD_SOFTWARE,
    )
    assert read_current_flatfield(path, field.shape) is None
