"""Test OPM deskew geometry and chunked processing."""

from dataclasses import dataclass

import numpy as np
import pytest
from scipy import ndimage

from opm_processing.imageprocessing import opmtools
from tests.testing_utils import scale_invariant_rmse, shell_line_width_x


@dataclass(frozen=True)
class ChunkedDeskewTestConfig:
    """Geometry, chunking, and acceptance criteria for deskew tests."""

    skewed_shape_zyx: tuple[int, int, int] = (12, 24, 24)
    theta_deg: float = 30.0
    scan_axis_step_um: float = 0.4
    pixel_size_um: float = 0.115
    decon_chunk_size: int = 5
    deskew_chunk_size: int = 22
    overlap_size: int = 24
    line_profile_half_window: int = 5
    minimum_lab_correlation: float = 0.75
    minimum_chunk_correlation: float = 0.99
    maximum_normalized_chunk_error: float = 0.03
    minimum_decon_correlation_gain: float = 0.01


@dataclass(frozen=True)
class ChunkedDeskewSample:
    """Synthetic hollow-shell acquisition and its known lab-frame truth."""

    sharp_skewed: np.ndarray
    blurred_skewed: np.ndarray
    psf: np.ndarray
    ground_truth: np.ndarray
    center_zyx: tuple[float, float, float]
    radii_zyx: tuple[float, float, float]


@pytest.fixture(scope="module")
def chunked_deskew_config() -> ChunkedDeskewTestConfig:
    """Return immutable deskew geometry and quantitative thresholds.

    Parameters
    ----------
    None
        This callable has no parameters.

    Returns
    -------
    ChunkedDeskewTestConfig
        Result produced by the callable.
    """
    return ChunkedDeskewTestConfig()


@pytest.fixture(scope="module")
def chunked_deskew_sample(chunked_deskew_config) -> ChunkedDeskewSample:
    """Forward-project a lab-frame hollow ellipsoid into skewed OPM data.

    Parameters
    ----------
    chunked_deskew_config : object
        Value supplied for ``chunked deskew config``.

    Returns
    -------
    ChunkedDeskewSample
        Result produced by the callable.
    """
    rng = np.random.default_rng(91)
    config = chunked_deskew_config
    skewed_shape = config.skewed_shape_zyx
    lab_shape, _, _, _ = opmtools.deskew_shape_estimator(
        skewed_shape,
        theta=config.theta_deg,
        distance=config.scan_axis_step_um,
        pixel_size=config.pixel_size_um,
        crop_after_deskew=False,
    )
    zz, yy, xx = np.indices(tuple(lab_shape), dtype=np.float32)
    center_zyx = (5.5, 31.0, 12.0)
    radii_zyx = (3.5, 13.0, 6.5)
    elliptical_radius = np.sqrt(
        ((zz - center_zyx[0]) / radii_zyx[0]) ** 2
        + ((yy - center_zyx[1]) / radii_zyx[1]) ** 2
        + ((xx - center_zyx[2]) / radii_zyx[2]) ** 2
    )
    ground_truth = (
        7000.0 * np.exp(-0.5 * ((elliptical_radius - 1.0) / 0.13) ** 2)
    ).astype(np.float32)

    scan, camera_y, camera_x = np.indices(skewed_shape, dtype=np.float32)
    theta = np.deg2rad(config.theta_deg)
    sample_z = camera_y * np.sin(theta)
    sample_y = scan * (
        config.scan_axis_step_um / config.pixel_size_um
    ) + camera_y * np.cos(theta)
    sharp = ndimage.map_coordinates(
        ground_truth,
        (sample_z, sample_y, camera_x),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    ).astype(np.uint16)

    pz, py, px = np.mgrid[-2:3, -3:4, -3:4]
    psf = np.exp(-(pz**2 / 1.0**2 + py**2 / 1.5**2 + px**2 / 1.5**2) / 2).astype(
        np.float32
    )
    psf /= psf.sum()
    blurred = rng.poisson(
        ndimage.convolve(sharp.astype(np.float32), psf, mode="constant")
    ).astype(np.uint16)
    return ChunkedDeskewSample(
        sharp_skewed=sharp,
        blurred_skewed=blurred,
        psf=psf,
        ground_truth=ground_truth,
        center_zyx=center_zyx,
        radii_zyx=radii_zyx,
    )


def _chunked_deskew(
    image,
    config: ChunkedDeskewTestConfig,
    *,
    psf=None,
    deconvolve=False,
):
    """Run chunked deskewing with shared test configuration.

    Parameters
    ----------
    image : object
        Value supplied for ``image``.
    config : ChunkedDeskewTestConfig
        Value supplied for ``config``.
    psf : object
        Value supplied for ``psf``.
    deconvolve : object
        Value supplied for ``deconvolve``.

    Returns
    -------
    object
        Result produced by the callable.
    """
    return opmtools.chunked_orthogonal_deskew(
        image,
        psf_data=psf,
        deconvolve=deconvolve,
        decon_chunk_size=config.decon_chunk_size,
        chunk_size=config.deskew_chunk_size,
        overlap_size=config.overlap_size,
        scan_crop=0,
        camera_bkd=0,
        camera_cf=1.0,
        camera_qe=1.0,
        z_downsample_level=1,
        theta_deg=config.theta_deg,
        scan_axis_step_um=config.scan_axis_step_um,
        pixel_size_um=config.pixel_size_um,
    )


def test_chunked_deskew_matches_direct_deskew_without_deconvolution(
    chunked_deskew_sample,
    chunked_deskew_config,
):
    """Verify chunked deskewing matches direct deskewing without deconvolution.

    Parameters
    ----------
    chunked_deskew_sample : object
        Value supplied for ``chunked deskew sample``.
    chunked_deskew_config : object
        Value supplied for ``chunked deskew config``.

    Returns
    -------
    None
        No value is returned.
    """
    sample = chunked_deskew_sample
    config = chunked_deskew_config

    actual = _chunked_deskew(sample.sharp_skewed, config)
    direct = opmtools.orthogonal_deskew(
        sample.sharp_skewed,
        theta=config.theta_deg,
        distance=config.scan_axis_step_um,
        pixel_size=config.pixel_size_um,
        downsample_factor=1,
    )

    assert actual.shape == direct.shape == sample.ground_truth.shape
    shell_mask = sample.ground_truth > 0.01 * sample.ground_truth.max()
    assert (
        np.corrcoef(direct[shell_mask], sample.ground_truth[shell_mask])[0, 1]
        > config.minimum_lab_correlation
    )
    center = tuple(int(round(value)) for value in sample.center_zyx)
    assert direct[center] < 0.2 * direct.max()

    supported = (actual > 0) & (direct > 0)
    assert (
        np.corrcoef(actual[supported], direct[supported])[0, 1]
        > config.minimum_chunk_correlation
    )
    normalized_error = scale_invariant_rmse(actual, direct) / np.sqrt(
        np.mean(direct.astype(np.float64) ** 2)
    )
    assert normalized_error < config.maximum_normalized_chunk_error


@pytest.mark.gpu
def test_chunked_deskew_with_gpu_deconvolution_improves_ground_truth(
    chunked_deskew_sample,
    chunked_deskew_config,
    cupy_gpu,
):
    """Verify GPU deconvolution improves shell recovery during chunked deskew.

    Parameters
    ----------
    chunked_deskew_sample : object
        Value supplied for ``chunked deskew sample``.
    chunked_deskew_config : object
        Value supplied for ``chunked deskew config``.
    cupy_gpu : object
        Value supplied for ``cupy gpu``.

    Returns
    -------
    None
        No value is returned.
    """
    del cupy_gpu  # Shared fixture has already proved CUDA execution.
    sample = chunked_deskew_sample
    config = chunked_deskew_config
    assert config.decon_chunk_size < sample.blurred_skewed.shape[0]

    without_deconvolution = _chunked_deskew(sample.blurred_skewed, config)
    with_deconvolution = _chunked_deskew(
        sample.blurred_skewed,
        config,
        psf=sample.psf,
        deconvolve=True,
    )

    assert with_deconvolution.shape == sample.ground_truth.shape
    support = (without_deconvolution > 0) | (with_deconvolution > 0)
    correlation_without = np.corrcoef(
        without_deconvolution[support], sample.ground_truth[support]
    )[0, 1]
    correlation_with = np.corrcoef(
        with_deconvolution[support], sample.ground_truth[support]
    )[0, 1]
    assert (
        correlation_with > correlation_without + config.minimum_decon_correlation_gain
    )

    wall_x = sample.center_zyx[2] + sample.radii_zyx[2]
    truth_width = shell_line_width_x(
        sample.ground_truth,
        center_zyx=sample.center_zyx,
        wall_x=wall_x,
        half_window=config.line_profile_half_window,
    )
    width_without = shell_line_width_x(
        without_deconvolution,
        center_zyx=sample.center_zyx,
        wall_x=wall_x,
        half_window=config.line_profile_half_window,
    )
    width_with = shell_line_width_x(
        with_deconvolution,
        center_zyx=sample.center_zyx,
        wall_x=wall_x,
        half_window=config.line_profile_half_window,
    )
    assert width_with < width_without
    assert abs(width_with - truth_width) < abs(width_without - truth_width)
