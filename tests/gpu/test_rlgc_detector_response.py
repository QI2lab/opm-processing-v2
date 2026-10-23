"""Detector operator units and calibrated acquisition processing integration."""

import importlib

import numpy as np
import pytest
from scipy import ndimage

from tests.simulation.metrics import predictive_deviance


@pytest.mark.integration
@pytest.mark.parametrize(
    "mosaic_acquisition", [("mirror", 0.4), ("stage", 1.2)], indirect=True
)
def test_roi_deconvolves_recorded_sampling(
    cupy_gpu, mosaic_acquisition, monkeypatch, invoke_cli
):
    """Read bounded raw planes and reconstruct the inferred fine ROI geometry."""
    from typer.testing import CliRunner

    from opm_processing import export_rois
    from opm_processing.dataio.position_collection import open_image_array
    from opm_processing.dataio.processing_state import ProcessingState

    case = mosaic_acquisition
    invoke_cli(
        "process",
        case.dataset.path,
        max_projection=True,
        create_fused_max_projection=True,
        z_downsample_level=1,
    )
    preview = case.dataset.path.parent / "mosaic_max_z_fused.ome.zarr"
    projection = open_image_array(preview).read().result()[0, 0, 0]
    center = int(np.argmax(projection.max(axis=1)))
    bounds = (center - 16, center + 16, 40, 88)
    monkeypatch.setattr(export_rois, "select_rois", lambda source, shape: [bounds])
    result = CliRunner().invoke(
        export_rois.app,
        [
            str(preview),
            "--deconvolve",
            "--save-float32",
            "--decon-verbose",
            "0",
            "--decon-psf-paths",
            str(case.psf_path),
        ],
    )
    if result.exception:
        raise result.exception
    assert result.exit_code == 0
    directory = preview.parent / "roi_exports" / "roi_001"
    state = ProcessingState.read(directory / "mosaic.processing.json")
    tiles = directory / "mosaic_decon_deskewed.ome.zarr"
    configuration = state.run(tiles)
    factor = round(case.microscope.scan_step_um / 0.4)
    assert (
        configuration["configuration"]["deconvolution"].get("scan_upsample_factor", 1)
        == factor
    )
    if factor > 1:
        assert configuration["reconstruction"]["scan_axis_step_um"] == pytest.approx(
            0.4
        )
    for tile in state.roi_series(tiles):
        start, stop = tile["skewed_bounds_sx"][:2]
        assert 0 <= start < stop <= case.dataset.raw_data.shape[-3]
        assert tile["skewed_shape_syx"][0] == stop - start
        shape = tile.get("reconstruction_shape_syx", tile["skewed_shape_syx"])
        assert shape[0] == (stop - start - 1) * factor + 1
    image = open_image_array(directory / "mosaic_fused.ome.zarr").read().result()
    assert image.shape[-2:] == (bounds[1] - bounds[0], bounds[3] - bounds[2])
    assert np.isfinite(image).all() and np.any(image > 0)


@pytest.mark.integration
@pytest.mark.parametrize(
    "roi_preview", [("mirror", 1, False), ("stage", 2, False)], indirect=True
)
def test_roi_cli_deconvolves_every_timepoint_and_channel(
    cupy_gpu, roi_preview, roi_psf, invoke_cli, monkeypatch, tmp_path
):
    """Retain all measured spectral/time samples through physical-PSF ROI RLGC."""
    from opm_processing import export_rois
    from opm_processing.dataio.position_collection import open_image_array
    from opm_processing.dataio.processing_state import ProcessingState

    case = roi_preview
    bounds = (3, 12, 2, 10)
    monkeypatch.setattr(export_rois, "select_rois", lambda source, shape: [bounds])
    destination = tmp_path / "selected"
    invoke_cli(
        "export-rois",
        case.preview,
        output=destination,
        deconvolve=True,
        save_float32=True,
        decon_verbose=0,
        decon_crop_scan=4 if case.positions > 1 else None,
        decon_psf_paths=[roi_psf] * case.dataset.metadata.shape[2],
        registration_channel=1,
    )
    directory = destination / "roi_001"
    image = open_image_array(directory / "sample_fused.ome.zarr").read().result()
    assert image.shape[:2] == case.dataset.metadata.shape[:3:2]
    assert image.shape[-2:] == (bounds[1] - bounds[0], bounds[3] - bounds[2])
    assert image.dtype == np.float32 and np.isfinite(image).all()
    assert np.all(image.max(axis=(2, 3, 4)) > 0)
    state = ProcessingState.read(directory / "sample.processing.json")
    tiles = directory / "sample_decon_deskewed.ome.zarr"
    assert state.completed_channels(tiles) == set(np.ndindex(2, case.positions, 2))
    settings = state.run(tiles)["configuration"]["deconvolution"]
    assert settings["gc_stopping"]["read_noise"] == 0
    assert settings["gc_stopping"]["kl_enabled"] is True


@pytest.mark.unit
@pytest.mark.parametrize(
    "kl_history", [False, True], ids=["iteration-cap", "both-halves-KL"]
)
@pytest.mark.parametrize("read_noise", [0, 1.6], ids=["photons", "read-noise"])
@pytest.mark.parametrize(
    "crop_scan,storage", [(2, "gpu"), (2, "host")], ids=["tiled-fft", "host-tiled-fft"]
)
@pytest.mark.parametrize("factor", [None, 1, 3], ids=["native", "sparse1", "sparse3"])
@pytest.mark.parametrize(
    "nonuniform", [False, True], ids=["unity", "detector-response"]
)
def test_tiled_consensus_recovers_known_blurred_object(
    cupy_gpu,
    factor,
    nonuniform,
    kl_history,
    read_noise,
    crop_scan,
    storage,
    discrete_optical_case,
):
    """Full and tiled GC improve the likelihood of a known blurred 3D object."""
    solver = importlib.import_module("opm_processing.imageprocessing.rlgc")
    case = discrete_optical_case
    options = {
        "use_kl": kl_history,
        "max_iterations": 3,
        "limit": 0,
        "max_delta": 0,
        "detector_response": case.response,
        "read_noise": read_noise,
        "rng_seed": 42,
        "scan_upsample_factor": factor or 1,
        "boundary": "reflect" if factor is None else "zero",
    }
    whole = solver.rlgc(case.measured, case.psf, storage=storage, **options)
    tiled = solver.rlgc(
        case.measured, case.psf, crop_scan=crop_scan, storage=storage, **options
    )
    assert tiled.shape == case.truth.shape
    baseline = case.measured / case.response
    if factor is not None:
        from scipy.interpolate import interp1d

        baseline = interp1d(np.arange(baseline.shape[0]) * factor, baseline, axis=0)(
            np.arange(case.truth.shape[0])
        )
    baseline_loss = predictive_deviance(
        baseline,
        case.expectation,
        case.psf,
        boundary=case.boundary,
        response=case.response,
    )
    # The GC gate is discontinuous at zero: single-precision FFT roundoff can
    # change an accepted update. Verify physical accuracy against truth for each
    # layout; forward/adjoint units separately bound convolution seam error.
    for recovered in (whole, tiled):
        assert (
            predictive_deviance(
                recovered,
                case.expectation,
                case.psf,
                boundary=case.boundary,
                response=case.response,
            )
            < baseline_loss
        )


@pytest.mark.unit
@pytest.mark.parametrize("scan_planes", (1, 7))
@pytest.mark.parametrize("crop_scan", (3, 20))
def test_nonuniform_response_preserves_uniform_fluorescence(
    cupy_gpu, scan_planes, crop_scan
):
    """A uniform object remains uniform through blur and nonuniform detection."""
    solver = importlib.import_module("opm_processing.imageprocessing.rlgc")

    response = np.broadcast_to(np.array((0.25, 0.5, 1, 2), np.float32), (16, 4))
    truth = np.full((scan_planes, *response.shape), 1200, dtype=np.float32)
    psf = np.ones((1 if scan_planes == 1 else 3, 3, 3), dtype=np.float32)
    psf /= psf.sum()
    measured = ndimage.convolve(truth, psf, mode="reflect") * response
    restored = solver.rlgc(
        measured, psf, crop_scan=crop_scan, detector_response=response, verbose=0
    )
    np.testing.assert_allclose(restored, truth, rtol=2e-6, atol=2e-3)


@pytest.mark.unit
def test_calibrated_input_preserves_subcount_intensity(cupy_gpu):
    """An identity optical model retains calibrated precision without uint16 overflow."""
    solver = importlib.import_module("opm_processing.imageprocessing.rlgc")

    observed = np.broadcast_to(
        np.array((-0.4, 31.9, 64.1, 80000.2), np.float32), (8, 4)
    )
    expected = np.broadcast_to(np.array((0, 31.9, 64.1, 80000.2), np.float32), (8, 4))
    actual = solver.rlgc(
        observed,
        np.ones((1, 1), np.float32),
        gradient_consensus=False,
        max_iterations=1,
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=0.02)


@pytest.mark.unit
@pytest.mark.parametrize("crop_scan", (4, 20))
@pytest.mark.parametrize("seed", (31, 83))
def test_detector_response_recovers_point_flux(cupy_gpu, crop_scan, seed):
    """Recover known emitters from integer Poisson draws under a varying response."""
    solver = importlib.import_module("opm_processing.imageprocessing.rlgc")
    z, y, x = np.mgrid[-2:3, -3:4, -3:4]
    psf = np.exp(-0.5 * (z**2 + (y / 1.5) ** 2 + (x / 1.5) ** 2)).astype(np.float32)
    psf /= psf.sum()
    truth = np.zeros((9, 35, 37), np.float32)
    truth[3, 10, 10], truth[6, 25, 27] = 20000, 30000
    response = np.broadcast_to(np.linspace(0.25, 2, 37, dtype=np.float32), (35, 37))
    expectation = response * (ndimage.convolve(truth, psf, mode="reflect") + 0.25)
    measured = np.random.default_rng(seed).poisson(expectation).astype(np.float32)
    actual = solver.rlgc(
        measured, psf, crop_scan=crop_scan, detector_response=response, verbose=0
    )
    assert predictive_deviance(
        actual, expectation, psf, boundary="reflect", response=response
    ) < predictive_deviance(
        measured / response, expectation, psf, boundary="reflect", response=response
    )
    for point in ((3, 10, 10), (6, 25, 27)):
        neighborhood = actual[tuple(slice(p - 1, p + 2) for p in point)]
        assert np.unravel_index(np.argmax(neighborhood), neighborhood.shape) == (
            1,
            1,
            1,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("uniform_detector_acquisition", "roi_mode"),
    (
        ("projection", False),
        ("mirror", False),
        ("stage", False),
        ("mirror", True),
        ("stage", True),
        *[
            pytest.param((mode, "generated-PSF"), False, id=f"generated-psf-{mode}")
            for mode in ("mirror", "stage", "projection")
        ],
    ),
    indirect=["uniform_detector_acquisition"],
)
def test_saved_fluorescence_with_detector_response(
    cupy_gpu, uniform_detector_acquisition
):
    """Check calibrated processing orchestration against known absolute fluorescence."""
    from opm_processing.dataio.position_collection import open_position_collection
    from opm_processing.dataio.processing_state import (
        ProcessingState,
        processing_state_path,
    )
    from opm_processing.process import process
    from opm_processing.process_roi import process_roi

    case = uniform_detector_acquisition
    run = process_roi if case.roi_mode else process
    run(case.dataset.path, **case.options)
    saved = open_position_collection(case.output).arrays[0][0, 0].read().result()
    np.testing.assert_allclose(saved, case.expected, rtol=0.005, atol=1)
    state = ProcessingState.read(
        processing_state_path(
            case.output.parent, case.dataset.path.stem.removesuffix(".ome")
        )
    )
    settings = state.run(case.output)["configuration"]["deconvolution"]
    assert settings["gc_stopping"]["read_noise"] == 0
    assert settings["gc_stopping"]["read_noise_reference"] == "gc_deconv.m"
    assert settings["observations"] == "calibrated-detector-counts"
    assert settings["observation_model"] == "poisson"
    assert settings["gc_stopping"]["accepted_fraction_limit"] == 0.001
    assert settings["gc_stopping"]["kl_enabled"] is True
    assert settings["gc_stopping"]["kl_require_both"] is True
    assert (
        settings["gc_stopping"]["consensus_product"]
        == "relative-directions-before-blur"
    )
    if "decon_psf_paths" in case.options:
        assert settings["psf_parameters"] is None
        np.save(case.psf_path, np.full((3, 3, 3), 2 / 27, np.float32))
    else:
        from yaozarrs import open_group

        from opm_processing.dataio.position_collection import update_group_attributes

        root = open_group(case.dataset.path)
        calibration = root.attrs["opm_processing"]
        assert (
            settings["psf_parameters"] == calibration["deconvolution"]["psf_parameters"]
        )
        calibration["deconvolution"]["psf_parameters"]["support_fwhms"] *= 1.2
        update_group_attributes(case.dataset.path, {"opm_processing": calibration})
    with pytest.raises(ValueError, match=r"incompatible|configuration"):
        run(case.dataset.path, **case.options, resume=True)
    np.testing.assert_array_equal(
        open_position_collection(case.output).arrays[0][0, 0].read().result(), saved
    )
