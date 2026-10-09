"""Overlapping physical specimens, placement errors and reconstruction settings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy import ndimage

from opm_processing.imageprocessing.opmtools import deskew_shape_estimator
from tests.fixtures.acquisition import opm_v2_frame_metadata, write_opm_v2_zarr

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class OpmV2TiledGroundTruthFixture:
    """A blurred multi-tile acquisition rendered from a known 3D object.

    Attributes
    ----------
    path : pathlib.Path
        Persisted upstream acquisition.
    configuration, mode : str
        Spatial scenario and mirror or stage scan mode.
    ground_truth : numpy.ndarray
        Unblurred laboratory ZYX fluorescence.
    raw_data : numpy.ndarray
        Quantized TPCZYX photon measurements, including stage excess planes.
    true_stage_positions_zxy, recorded_stage_positions_zxy : numpy.ndarray
        Physical and deliberately perturbed stage positions in micrometers.
    tile_offsets_zyx_px : tuple
        True laboratory origins in pixels.
    recorded_position_errors_zyx_px : numpy.ndarray
        Known placement errors for registration to recover.
    ellipsoids_zyx_radii : tuple
        Shell centers and radii in laboratory pixels.
    pixel_size_um, scan_axis_step_um : float
        Detector pitch and physical scan increment.
    psf_path : pathlib.Path
        Unit-sum skewed PSF saved for reconstruction.
    """

    path: Path
    configuration: str
    mode: str
    ground_truth: np.ndarray
    raw_data: np.ndarray
    true_stage_positions_zxy: np.ndarray
    recorded_stage_positions_zxy: np.ndarray
    tile_offsets_zyx_px: tuple[tuple[int, int, int], ...]
    recorded_position_errors_zyx_px: np.ndarray
    ellipsoids_zyx_radii: tuple[tuple[float, float, float, float, float, float], ...]
    pixel_size_um: float
    scan_axis_step_um: float
    psf_path: Path


@dataclass(frozen=True)
class TiledAcquisitionConfig:
    """Immutable spatial and acquisition configuration for a synthetic run.

    Parameters
    ----------
    name : str
        Spatial scenario used in dataset names.
    mode : str
        Mirror or stage scanning.
    tile_offsets_zyx_px, recorded_position_errors_zyx_px : tuple
        True laboratory origins and added recording errors in pixels.
    camera_shape_zyx : tuple of int
        Scan planes, detector rows and detector columns before excess planes.
    pixel_size_um, scan_axis_step_um : float
        Detector pitch and scan increment in micrometers.
    theta_deg : float
        Camera-plane tilt relative to the laboratory frame.
    rng_seed : int
        Seed for shell geometry, brightness and photon noise.
    """

    name: str
    mode: str
    tile_offsets_zyx_px: tuple[tuple[int, int, int], ...]
    recorded_position_errors_zyx_px: tuple[tuple[int, int, int], ...]
    camera_shape_zyx: tuple[int, int, int] = (16, 32, 32)
    pixel_size_um: float = 0.115
    scan_axis_step_um: float = 0.4
    theta_deg: float = 30.0
    rng_seed: int = 2468


def tiled_acquisition_config(
    name: str, *, mode: str = "mirror"
) -> TiledAcquisitionConfig:
    """Build a known spatial scenario with prescribed recording errors.

    Parameters
    ----------
    name : str
        ``x_overlap``, ``yx_grid``, ``z_staggered``, ``thin_z_staggered`` or
        ``yx_grid_z_staggered``.
    mode : str
        Mirror or stage scanning.

    Returns
    -------
    TiledAcquisitionConfig
        Geometry and repeatable acquisition settings for the named scenario.
    """
    configurations = {
        "x_overlap": (
            ((0, 0, 0), (0, 0, 14)),
            ((0, 0, 0), (0, 0, 2)),
        ),
        "yx_grid": (
            ((0, 0, 0), (0, 0, 14), (0, 8, 0), (0, 8, 14)),
            ((0, 0, 0), (0, 0, 2), (0, 2, 0), (0, 2, 2)),
        ),
        "z_staggered": (
            ((0, 0, 0), (4, 0, 0), (8, 0, 0)),
            ((0, 0, 0), (1, 0, 0), (1, 0, 0)),
        ),
        "thin_z_staggered": (
            # Explicit Y moves keep the thin overlap's oblique support aligned.
            # These lab origins previously came from the erroneous Z-to-Y shear.
            ((0, 0, 0), (11, 19, 0), (22, 38, 0)),
            ((0, 0, 0), (0, 0, 2), (0, 0, 2)),
        ),
        "yx_grid_z_staggered": (
            ((0, 0, 0), (2, 0, 14), (4, 8, 0), (6, 8, 14)),
            ((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)),
        ),
    }
    offsets, errors = configurations[name]
    return TiledAcquisitionConfig(
        name=name,
        mode=mode,
        tile_offsets_zyx_px=offsets,
        recorded_position_errors_zyx_px=errors,
    )


def create_opm_v2_tiled_ground_truth_zarr(
    tmp_path: Path,
    *,
    config: TiledAcquisitionConfig,
) -> OpmV2TiledGroundTruthFixture:
    """Render an overlapping tiled OPM-v2 acquisition from a known object.

    The forward model samples a laboratory-frame ground truth on the tilted
    camera planes used by OPM and convolves each skewed stack with the same
    compact PSF supplied to processing.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated directory receiving the acquisition and PSF.
    config : TiledAcquisitionConfig
        Physical sampling, true tile origins and known placement errors.

    Returns
    -------
    OpmV2TiledGroundTruthFixture
        Persisted camera measurements and independent laboratory truth.
    """
    mode = config.mode
    path = tmp_path / f"opm_v2_{mode}_{config.name}.zarr"
    pixel_size_um = config.pixel_size_um
    scan_axis_step_um = config.scan_axis_step_um
    theta_deg = config.theta_deg
    camera_shape = config.camera_shape_zyx
    true_stage_offsets = np.asarray(config.tile_offsets_zyx_px, dtype=np.float64)
    # Stage translations are orthogonal; the camera-plane tilt is modeled
    # below when sampling each tile, not by shearing its laboratory origin.
    true_image_offsets = np.rint(true_stage_offsets).astype(np.int64)
    excess_scan_positions = 1 if mode == "stage" else 0
    stored_scan_count = camera_shape[0] + excess_scan_positions
    shape = (
        1,
        len(config.tile_offsets_zyx_px),
        1,
        stored_scan_count,
        camera_shape[1],
        camera_shape[2],
    )
    tile_zyx, _, _, _ = deskew_shape_estimator(
        camera_shape,
        theta=theta_deg,
        distance=scan_axis_step_um,
        pixel_size=pixel_size_um,
        crop_after_deskew=False,
    )
    max_offsets = np.max(true_image_offsets, axis=0)
    ground_truth_shape = tuple(
        int(tile_size + offset)
        for tile_size, offset in zip(tile_zyx, max_offsets, strict=False)
    )

    rng = np.random.default_rng(config.rng_seed)
    ground_truth = np.zeros(ground_truth_shape, dtype=np.float32)
    ellipsoids = []
    ellipsoid_amplitudes = []
    for z_index, center_z in enumerate(
        np.arange(4.0, ground_truth_shape[0] - 2.0, 5.0)
    ):
        for y_index, center_y in enumerate(
            np.arange(12.0, ground_truth_shape[1] - 5.0, 18.0)
        ):
            for x_index, center_x in enumerate(
                np.arange(7.0, ground_truth_shape[2] - 4.0, 11.0)
            ):
                if (z_index + y_index + x_index) % 2 == 0:
                    ellipsoids.append(
                        (
                            center_z + rng.uniform(-0.6, 0.6),
                            center_y + rng.uniform(-1.2, 1.2),
                            center_x + rng.uniform(-1.0, 1.0),
                            2.2 + rng.uniform(0.0, 0.8),
                            4.8 + rng.uniform(0.0, 1.8),
                            3.4 + rng.uniform(0.0, 1.4),
                        )
                    )
                    ellipsoid_amplitudes.append(rng.uniform(4500.0, 8500.0))
    ellipsoids_zyx_radii = tuple(ellipsoids)
    zz, yy, xx = np.indices(ground_truth_shape, dtype=np.float32)
    for amplitude, (
        center_z,
        center_y,
        center_x,
        radius_z,
        radius_y,
        radius_x,
    ) in zip(ellipsoid_amplitudes, ellipsoids_zyx_radii, strict=False):
        elliptical_radius = np.sqrt(
            ((zz - center_z) / radius_z) ** 2
            + ((yy - center_y) / radius_y) ** 2
            + ((xx - center_x) / radius_x) ** 2
        )
        ground_truth += amplitude * np.exp(
            -0.5 * ((elliptical_radius - 1.0) / 0.13) ** 2
        )
    ground_truth = np.clip(ground_truth, 0.0, 40_000.0)

    scan, camera_y, camera_x = np.indices(camera_shape, dtype=np.float32)
    theta_rad = np.deg2rad(theta_deg)
    sample_z = camera_y * np.sin(theta_rad)
    sample_y = scan * (scan_axis_step_um / pixel_size_um) + camera_y * np.cos(theta_rad)
    psf_z, psf_y, psf_x = np.mgrid[-2:3, -3:4, -3:4]
    psf = np.exp(
        -(psf_z**2 / 1.0**2 + psf_y**2 / 1.6**2 + psf_x**2 / 1.6**2) / 2
    ).astype(np.float32)
    psf /= psf.sum()
    psf_path = tmp_path / f"{mode}_synthetic_psf.npy"
    np.save(psf_path, psf)

    raw_data = np.zeros(shape, dtype=np.uint16)
    for position, (tile_z_offset, tile_y_offset, tile_x_offset) in enumerate(
        true_image_offsets
    ):
        tile_sample_z = sample_z + tile_z_offset
        tile_sample_y = sample_y + tile_y_offset
        sample_x = camera_x + tile_x_offset
        ideal_skewed = ndimage.map_coordinates(
            ground_truth,
            (tile_sample_z, tile_sample_y, sample_x),
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        blurred = ndimage.convolve(ideal_skewed, psf, mode="constant", cval=0.0)
        blurred = rng.poisson(np.clip(blurred, 0.0, None)).astype(np.uint16)
        if mode == "stage":
            # process_skewed reverses stage scans and then removes the leading
            # excess plane. Arrange stored data so that operation recovers the
            # same physical scan order as the mirror acquisition.
            post_flip = np.concatenate((np.zeros_like(blurred[:1]), blurred), axis=0)
            blurred = post_flip[::-1]
        raw_data[0, position, 0] = blurred

    recorded_errors = np.asarray(
        config.recorded_position_errors_zyx_px, dtype=np.float64
    )
    true_stage_positions_zxy = true_stage_offsets * pixel_size_um
    recorded_stage_positions_zxy = (
        true_stage_offsets + recorded_errors
    ) * pixel_size_um
    # Physical specimen-stage Z motion moves the sampled laboratory plane in
    # the opposite direction. Fusion performs this stage-to-lab transform.
    true_stage_positions_zxy[:, 0] *= -1
    recorded_stage_positions_zxy[:, 0] *= -1
    # OPM stage Y and image-placement Y point in opposite directions.
    true_stage_positions_zxy[:, 1] *= -1
    recorded_stage_positions_zxy[:, 1] *= -1
    frame_metadatas = []
    for position in range(shape[1]):
        for stored_scan in range(shape[3]):
            daq_metadata = {
                "mode": mode,
                "channel_states": [True],
                "exposure_channels_ms": [10.0],
                "laser_powers": [10.0],
                "interleaved": True,
                "blanking": True,
                "current_channel": "488nm",
            }
            opm_metadata = {
                "angle_deg": theta_deg,
                "camera_Zstage_orientation": "normal",
                "camera_XYstage_orientation": "normal",
                "camera_mirror_orientation": "normal",
            }
            stage_x = recorded_stage_positions_zxy[position, 1]
            if mode == "mirror":
                daq_metadata.update(
                    {
                        "image_mirror_position": float(stored_scan),
                        "image_mirror_range_um": (camera_shape[0] * scan_axis_step_um),
                        "image_mirror_step_um": scan_axis_step_um,
                    }
                )
            else:
                daq_metadata["scan_axis_step_um"] = scan_axis_step_um
                opm_metadata.update(
                    {
                        "excess_scan_positions": excess_scan_positions,
                        "excess_scan_start_positions": excess_scan_positions,
                        "excess_scan_end_positions": excess_scan_positions,
                    }
                )
                stage_x += stored_scan * scan_axis_step_um

            frame_metadatas.append(
                opm_v2_frame_metadata(
                    index={
                        "t": 0,
                        "p": position,
                        "c": 0,
                        "z": stored_scan,
                    },
                    daq_metadata=daq_metadata,
                    opm_metadata=opm_metadata,
                    stage_metadata={
                        "x_pos": stage_x,
                        "y_pos": recorded_stage_positions_zxy[position, 2],
                        "z_pos": recorded_stage_positions_zxy[position, 0],
                        "excess_image": (
                            mode == "stage" and stored_scan < excess_scan_positions
                        ),
                    },
                    camera_shape_yx=camera_shape[1:],
                    pixel_size_um=pixel_size_um,
                    camera_offset=0.0,
                    camera_conversion=1.0,
                    runner_time_ms=float(position * shape[3] + stored_scan),
                    hardware_triggered=True,
                )
            )

    write_opm_v2_zarr(
        path=path,
        raw_data=raw_data,
        labels=("t", "p", "c", "z", "y", "x"),
        chunks=(1, 1, 1, 1, camera_shape[1], camera_shape[2]),
        frame_metadatas=frame_metadatas,
    )
    return OpmV2TiledGroundTruthFixture(
        path=path,
        configuration=config.name,
        mode=mode,
        ground_truth=ground_truth,
        raw_data=raw_data,
        true_stage_positions_zxy=true_stage_positions_zxy,
        recorded_stage_positions_zxy=recorded_stage_positions_zxy,
        tile_offsets_zyx_px=tuple(
            tuple(int(value) for value in offset) for offset in true_image_offsets
        ),
        recorded_position_errors_zyx_px=recorded_errors,
        ellipsoids_zyx_radii=ellipsoids_zyx_radii,
        pixel_size_um=pixel_size_um,
        scan_axis_step_um=scan_axis_step_um,
        psf_path=psf_path,
    )


@pytest.fixture(params=("mirror", "stage"), ids=("mirror-tiled", "stage-tiled"))
def opm_v2_tiled_ground_truth_zarr(request, tmp_path) -> OpmV2TiledGroundTruthFixture:
    """Create two physically overlapping tiles in each scan mode.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Parametrized mirror or stage scan mode.
    tmp_path : pathlib.Path
        Isolated acquisition directory.

    Returns
    -------
    OpmV2TiledGroundTruthFixture
        Known object, camera measurements and perturbed stage coordinates.
    """
    return create_opm_v2_tiled_ground_truth_zarr(
        tmp_path,
        config=tiled_acquisition_config("x_overlap", mode=str(request.param)),
    )


@pytest.fixture(
    params=(
        ("yx_grid", "mirror"),
        ("z_staggered", "mirror"),
        ("thin_z_staggered", "mirror"),
        ("yx_grid_z_staggered", "mirror"),
        ("yx_grid", "stage"),
        ("z_staggered", "stage"),
        ("thin_z_staggered", "stage"),
        ("yx_grid_z_staggered", "stage"),
    ),
    ids=(
        "yx-grid",
        "z-staggered",
        "thin-z-staggered",
        "yx-grid-z-staggered",
        "stage-yx-grid",
        "stage-z-staggered",
        "stage-thin-z-staggered",
        "stage-yx-grid-z-staggered",
    ),
)
def opm_v2_spatial_tiling_ground_truth_zarr(
    request,
    tmp_path,
) -> OpmV2TiledGroundTruthFixture:
    """Create each multi-axis spatial scenario in both scan modes.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Spatial scenario and scan mode selected by parametrization.
    tmp_path : pathlib.Path
        Isolated acquisition directory.

    Returns
    -------
    OpmV2TiledGroundTruthFixture
        Known object, camera measurements and perturbed stage coordinates.
    """
    configuration, mode = request.param
    return create_opm_v2_tiled_ground_truth_zarr(
        tmp_path,
        config=tiled_acquisition_config(configuration, mode=mode),
    )


@dataclass(frozen=True)
class ReconstructionTestConfig:
    """Processing options and quantitative acceptance criteria.

    Parameters
    ----------
    correlation_percentile : float
        Truth intensity percentile defining meaningful fluorescence support.
    minimum_correlation_samples : int
        Required support samples for a correlation measurement.
    minimum_tile_correlation, minimum_deskewed_correlation : float
        Minimum truth correlation before and after deconvolution.
    minimum_fused_correlation, minimum_overlap_correlation : float
        Minimum truth correlation in the fused volume and tile overlaps.
    minimum_line_width_samples : int
        Required resolved shell profiles for width comparisons.
    line_profile_half_window : int
        Pixel radius of each shell profile.
    blend_pixels_zyx : tuple of int
        Fusion taper widths in laboratory pixels.
    registration_downsample_zyx : tuple of int
        Sampling factors used for pair registration.
    maximum_registration_shift_zyx : tuple of int
        Allowed residual shifts after stage placement, in laboratory pixels.
    decon_scan_chunk_size : int
        Scan-plane block size for chunked deconvolution.
    """

    correlation_percentile: float = 35.0
    minimum_correlation_samples: int = 100
    minimum_tile_correlation: float = 0.55
    minimum_deskewed_correlation: float = 0.45
    minimum_fused_correlation: float = 0.50
    minimum_overlap_correlation: float = 0.50
    minimum_line_width_samples: int = 4
    line_profile_half_window: int = 5
    blend_pixels_zyx: tuple[int, int, int] = (1, 4, 4)
    # Preserve the resolved Z structure of these tiny deconvolved overlaps.
    registration_downsample_zyx: tuple[int, int, int] = (1, 1, 1)
    maximum_registration_shift_zyx: tuple[int, int, int] = (2, 2, 4)
    decon_scan_chunk_size: int = 6

    def fusion_options(self) -> dict[str, object]:
        """Return registration, blending and storage settings for synthetic tiles.

        Returns
        -------
        dict
            Keyword arguments shared by full and deskew-only fusion runs.
        """
        return {
            "blend_pixels": self.blend_pixels_zyx,
            "downsample_factors": self.registration_downsample_zyx,
            "ssim_window": 3,
            "threshold": 0.1,
            "multiscale_factors": (2,),
            "resolution_multiples": ((1, 1, 1), (2, 2, 2)),
            "chunk_shape_yx": (16, 16),
            "max_registration_shift_zyx": self.maximum_registration_shift_zyx,
        }


@pytest.fixture
def reconstruction_config() -> ReconstructionTestConfig:
    """Return one immutable configuration shared by all tiled cases."""
    return ReconstructionTestConfig()
