"""Ground-truth reconstruction test for tiled OPM-v2 acquisitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from opm_processing.dataio.position_collection import (
    open_image_array,
    open_position_collection,
)
from opm_processing.imageprocessing import tilefusion as tilefusion_module
from opm_processing.imageprocessing.tilefusion import TileFusion
from opm_processing.process import process
from tests.testing_utils import masked_correlation, shell_line_width_x


pytestmark = pytest.mark.gpu


@dataclass(frozen=True)
class ReconstructionTestConfig:
    """Processing options and quantitative acceptance criteria."""

    correlation_percentile: float = 35.0
    minimum_correlation_samples: int = 100
    minimum_tile_correlation: float = 0.55
    minimum_fused_correlation: float = 0.60
    minimum_line_width_samples: int = 4
    line_profile_half_window: int = 5
    blend_pixels_zyx: tuple[int, int, int] = (1, 4, 4)
    registration_downsample_zyx: tuple[int, int, int] = (1, 1, 1)
    maximum_registration_shift_zyx: tuple[int, int, int] = (2, 2, 4)

    def processing_options(self) -> dict[str, object]:
        """Options shared by deskew-only and deconvolved processing calls."""
        return {
            "max_projection": False,
            "flatfield_correction": False,
            "create_fused_max_projection": False,
            "z_downsample_level": 1,
            "stage_x_flipped": False,
            "stage_y_flipped": False,
            "stage_z_flipped": False,
        }

    def fusion_options(self) -> dict[str, object]:
        """Small-data registration and fusion settings for synthetic tiles."""
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


def _measure_correlation(
    candidate: np.ndarray,
    truth: np.ndarray,
    config: ReconstructionTestConfig,
) -> float:
    measurement = masked_correlation(
        candidate,
        truth,
        truth_percentile=config.correlation_percentile,
    )
    assert measurement.sample_count > config.minimum_correlation_samples
    return measurement.value


def _measure_shell_width(
    volume: np.ndarray,
    center_zyx: tuple[float, float, float],
    wall_x: float,
    config: ReconstructionTestConfig,
) -> float:
    return shell_line_width_x(
        volume,
        center_zyx=center_zyx,
        wall_x=wall_x,
        half_window=config.line_profile_half_window,
    )


def _assert_tiled_reconstruction(
    fixture,
    config: ReconstructionTestConfig,
    cupy_gpu,
):
    """Run and validate deskew, deconvolution, registration, and fusion."""
    assert tilefusion_module.USING_GPU
    assert tilefusion_module.xp is cupy_gpu
    processing_options = config.processing_options()

    process(root_path=fixture.path, deconvolve=False, **processing_options)
    process(
        root_path=fixture.path,
        deconvolve=True,
        decon_crop_y=fixture.raw_data.shape[-2],
        decon_gpu_id=0,
        decon_psf_paths=[fixture.psf_path],
        **processing_options,
    )

    deskewed_path = fixture.path.parent / f"{fixture.path.stem}_deskewed.zarr"
    deconvolved_path = (
        fixture.path.parent / f"{fixture.path.stem}_decon_deskewed.zarr"
    )
    deskewed_collection = open_position_collection(deskewed_path)
    deconvolved_collection = open_position_collection(deconvolved_path)
    assert deskewed_collection.shape[:3] == (
        1,
        len(fixture.tile_offsets_zyx_px),
        1,
    )
    assert deconvolved_collection.shape == deskewed_collection.shape
    np.testing.assert_allclose(
        deconvolved_collection.attributes["stage_positions"],
        fixture.recorded_stage_positions_zxy,
    )

    deskewed_scores = []
    deconvolved_scores = []
    truth_line_widths = []
    deskewed_line_widths = []
    deconvolved_line_widths = []
    tile_shape = deskewed_collection.shape[-3:]
    for position, (z_offset, y_offset, x_offset) in enumerate(
        fixture.tile_offsets_zyx_px
    ):
        truth_tile = fixture.ground_truth[
            z_offset : z_offset + tile_shape[0],
            y_offset : y_offset + tile_shape[1],
            x_offset : x_offset + tile_shape[2],
        ]
        deskewed = deskewed_collection.arrays[position][0, 0].read().result()
        deconvolved = deconvolved_collection.arrays[position][0, 0].read().result()
        deskewed_scores.append(_measure_correlation(deskewed, truth_tile, config))
        deconvolved_scores.append(
            _measure_correlation(deconvolved, truth_tile, config)
        )
        for center_z, center_y, center_x, _, _, radius_x in (
            fixture.ellipsoids_zyx_radii
        ):
            for wall_x in (center_x - radius_x, center_x + radius_x):
                local_center_z = center_z - z_offset
                local_center_y = center_y - y_offset
                local_center_x = center_x - x_offset
                local_wall_x = wall_x - x_offset
                if (
                    0 <= local_center_z < truth_tile.shape[0]
                    and 0 <= local_center_y < truth_tile.shape[1]
                    and 5 <= local_wall_x < truth_tile.shape[2] - 5
                ):
                    center = (local_center_z, local_center_y, local_center_x)
                    widths = (
                        _measure_shell_width(
                            truth_tile, center, local_wall_x, config
                        ),
                        _measure_shell_width(
                            deskewed, center, local_wall_x, config
                        ),
                        _measure_shell_width(
                            deconvolved, center, local_wall_x, config
                        ),
                    )
                    if np.all(np.isfinite(widths)):
                        truth_width, deskewed_width, deconvolved_width = widths
                        truth_line_widths.append(truth_width)
                        deskewed_line_widths.append(deskewed_width)
                        deconvolved_line_widths.append(deconvolved_width)

    assert min(deconvolved_scores) > config.minimum_tile_correlation
    assert len(truth_line_widths) >= config.minimum_line_width_samples
    truth_width = np.mean(truth_line_widths)
    deskewed_width = np.mean(deskewed_line_widths)
    deconvolved_width = np.mean(deconvolved_line_widths)
    assert deconvolved_width < deskewed_width
    assert abs(deconvolved_width - truth_width) < abs(deskewed_width - truth_width)

    fusion = TileFusion(
        root_path=fixture.path,
        **config.fusion_options(),
    )
    assert fusion.data == deconvolved_path
    fusion.run()

    assert len(fusion.pairwise_metrics) >= len(fixture.tile_offsets_zyx_px) - 1
    assert fusion.global_offsets is not None
    corrected_errors_px = np.abs(
        (
            fixture.recorded_stage_positions_zxy
            + fusion.global_offsets * fixture.pixel_size_um
            - fixture.true_stage_positions_zxy
        )
        / fixture.pixel_size_um
    )
    for initial_error, corrected_error in zip(
        fixture.recorded_position_errors_zyx_px,
        corrected_errors_px,
    ):
        if np.any(initial_error):
            assert np.linalg.norm(corrected_error) < np.linalg.norm(initial_error)

    fused_path = fixture.path.parent / f"{fixture.path.stem}_fused.ome.zarr"
    fused = open_image_array(fused_path).read().result()[0, 0]
    truth_shape = fixture.ground_truth.shape
    reconstructed = fused[
        : truth_shape[0], : truth_shape[1], : truth_shape[2]
    ]
    assert (
        _measure_correlation(reconstructed, fixture.ground_truth, config)
        > config.minimum_fused_correlation
    )
    fused_line_widths = []
    for center_z, center_y, center_x, _, _, radius_x in (
        fixture.ellipsoids_zyx_radii
    ):
        for wall_x in (center_x - radius_x, center_x + radius_x):
            if 5 <= wall_x < reconstructed.shape[2] - 5:
                width = _measure_shell_width(
                    reconstructed,
                    (center_z, center_y, center_x),
                    wall_x,
                    config,
                )
                if np.isfinite(width):
                    fused_line_widths.append(width)
    assert len(fused_line_widths) >= config.minimum_line_width_samples
    fused_width = np.mean(fused_line_widths)
    assert abs(fused_width - truth_width) <= abs(deskewed_width - truth_width)


def test_tiled_opm_v2_reconstructs_registered_ground_truth(
    opm_v2_tiled_ground_truth_zarr,
    reconstruction_config,
    cupy_gpu,
):
    """Recover the original X-overlap acquisition in mirror and stage modes."""
    _assert_tiled_reconstruction(
        opm_v2_tiled_ground_truth_zarr,
        reconstruction_config,
        cupy_gpu,
    )


def test_spatial_tiling_reconstructs_registered_ground_truth(
    opm_v2_spatial_tiling_ground_truth_zarr,
    reconstruction_config,
    cupy_gpu,
):
    """Validate YX-grid, Z-staggered, and combined configurations."""
    _assert_tiled_reconstruction(
        opm_v2_spatial_tiling_ground_truth_zarr,
        reconstruction_config,
        cupy_gpu,
    )
