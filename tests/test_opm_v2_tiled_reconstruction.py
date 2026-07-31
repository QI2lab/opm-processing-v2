"""Ground-truth reconstruction test for tiled OPM-v2 acquisitions."""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pytest

from opm_processing.dataio.position_collection import (
    open_image_array,
    open_position_collection,
)
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
    minimum_deskewed_correlation: float = 0.45
    minimum_fused_correlation: float = 0.50
    minimum_overlap_correlation: float = 0.50
    minimum_line_width_samples: int = 4
    line_profile_half_window: int = 5
    blend_pixels_zyx: tuple[int, int, int] = (1, 4, 4)
    registration_downsample_zyx: tuple[int, int, int] = (3, 1, 1)
    maximum_registration_shift_zyx: tuple[int, int, int] = (2, 2, 4)
    decon_scan_chunk_size: int = 6

    def processing_options(self) -> dict[str, object]:
        """Options shared by deskew-only and deconvolved processing calls.

        Parameters
        ----------
        None
            This callable has no parameters.

        Returns
        -------
        dict[str, object]
            Result produced by the callable.
        """
        return {
            "max_projection": False,
            "flatfield_correction": False,
            "create_fused_max_projection": False,
            "z_downsample_level": 1,
        }

    def fusion_options(self) -> dict[str, object]:
        """Small-data registration and fusion settings for synthetic tiles.

        Parameters
        ----------
        None
            This callable has no parameters.

        Returns
        -------
        dict[str, object]
            Result produced by the callable.
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
    """Return one immutable configuration shared by all tiled cases.

    Parameters
    ----------
    None
        This callable has no parameters.

    Returns
    -------
    ReconstructionTestConfig
        Result produced by the callable.
    """
    return ReconstructionTestConfig()


def _measure_correlation(
    candidate: np.ndarray,
    truth: np.ndarray,
    config: ReconstructionTestConfig,
) -> float:
    """Measure masked correlation and enforce sufficient support.

    Parameters
    ----------
    candidate : np.ndarray
        Value supplied for ``candidate``.
    truth : np.ndarray
        Value supplied for ``truth``.
    config : ReconstructionTestConfig
        Value supplied for ``config``.

    Returns
    -------
    float
        Result produced by the callable.
    """
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
    """Measure the reconstructed hollow-ellipsoid shell width.

    Parameters
    ----------
    volume : np.ndarray
        Value supplied for ``volume``.
    center_zyx : tuple[float, float, float]
        Value supplied for ``center zyx``.
    wall_x : float
        Value supplied for ``wall x``.
    config : ReconstructionTestConfig
        Value supplied for ``config``.

    Returns
    -------
    float
        Result produced by the callable.
    """
    return shell_line_width_x(
        volume,
        center_zyx=center_zyx,
        wall_x=wall_x,
        half_window=config.line_profile_half_window,
    )


def _align_fused_to_ground_truth(
    fused: np.ndarray,
    truth: np.ndarray,
    *,
    offset_um: tuple[float, float, float],
    pixel_size_um: tuple[float, float, float],
) -> np.ndarray:
    """Place a translated fused volume in the synthetic world-coordinate grid.

    Parameters
    ----------
    fused : np.ndarray
        Fused volume whose voxel zero is at ``offset_um``.
    truth : np.ndarray
        Ground-truth volume whose voxel zero is the world-coordinate origin.
    offset_um : tuple[float, float, float]
        Fused image origin in ZYX physical coordinates.
    pixel_size_um : tuple[float, float, float]
        Fused voxel spacing in ZYX order.

    Returns
    -------
    np.ndarray
        Fused samples aligned to the ground-truth array coordinates.
    """
    offset_pixels = np.rint(
        np.asarray(offset_um, dtype=np.float64)
        / np.asarray(pixel_size_um, dtype=np.float64)
    ).astype(np.int64)
    fused_start = np.maximum(-offset_pixels, 0)
    truth_start = np.maximum(offset_pixels, 0)
    common_shape = np.minimum(
        np.asarray(fused.shape, dtype=np.int64) - fused_start,
        np.asarray(truth.shape, dtype=np.int64) - truth_start,
    )
    if np.any(common_shape <= 0):
        raise AssertionError(
            "Fused output does not overlap the synthetic world-coordinate volume"
        )

    fused_slices = tuple(
        slice(int(start), int(start + size))
        for start, size in zip(fused_start, common_shape)
    )
    truth_slices = tuple(
        slice(int(start), int(start + size))
        for start, size in zip(truth_start, common_shape)
    )
    aligned = np.zeros_like(truth, dtype=fused.dtype)
    aligned[truth_slices] = fused[fused_slices]
    return aligned


def _tile_overlap_mask(
    world_shape: tuple[int, int, int],
    tile_shape: tuple[int, int, int],
    tile_offsets_zyx_px,
) -> np.ndarray:
    """Return voxels covered by at least two ground-truth tiles."""
    coverage = np.zeros(world_shape, dtype=np.uint8)
    for offset in tile_offsets_zyx_px:
        slices = tuple(
            slice(int(start), int(start) + int(length))
            for start, length in zip(offset, tile_shape)
        )
        coverage[slices] += 1
    return coverage > 1


def _assert_fused_overlap_matches_ground_truth(
    fixture,
    config: ReconstructionTestConfig,
    fusion: TileFusion,
) -> tuple[np.ndarray, float]:
    """Validate the registered fused pixels specifically inside tile overlaps."""
    fused_path = fixture.path.parent / f"{fixture.path.stem}_fused.ome.zarr"
    fused = open_image_array(fused_path).read().result()[0, 0]
    assert fusion.offset_um is not None
    reconstructed = _align_fused_to_ground_truth(
        fused,
        fixture.ground_truth,
        offset_um=fusion.offset_um,
        pixel_size_um=fusion._pixel_size,
    )
    overlap = _tile_overlap_mask(
        fixture.ground_truth.shape,
        tuple(int(value) for value in fusion.position_arrays[0].shape[-3:]),
        fixture.tile_offsets_zyx_px,
    )
    overlap_correlation = _measure_correlation(
        reconstructed[overlap],
        fixture.ground_truth[overlap],
        config,
    )
    assert overlap_correlation > config.minimum_overlap_correlation, {
        "overlap_correlation": overlap_correlation,
        "pairwise_metrics": fusion.pairwise_metrics,
        "global_offsets": fusion.global_offsets,
    }
    return reconstructed, overlap_correlation


def _assert_tiled_reconstruction(
    fixture,
    config: ReconstructionTestConfig,
    cupy_gpu,
):
    """Run and validate deskew, deconvolution, registration, and fusion.

    Parameters
    ----------
    fixture : object
        Value supplied for ``fixture``.
    config : ReconstructionTestConfig
        Value supplied for ``config``.
    cupy_gpu : object
        Value supplied for ``cupy gpu``.

    Returns
    -------
    None
        No value is returned.
    """
    del cupy_gpu
    processing_options = config.processing_options()

    process(root_path=fixture.path, deconvolve=False, **processing_options)
    process(
        root_path=fixture.path,
        deconvolve=True,
        decon_crop_scan=config.decon_scan_chunk_size,
        decon_gpu_id=0,
        decon_psf_paths=[fixture.psf_path],
        **processing_options,
    )

    deskewed_path = fixture.path.parent / f"{fixture.path.stem}_deskewed.ome.zarr"
    deconvolved_path = (
        fixture.path.parent / f"{fixture.path.stem}_decon_deskewed.ome.zarr"
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
        deconvolved_scores.append(_measure_correlation(deconvolved, truth_tile, config))
        for (
            center_z,
            center_y,
            center_x,
            _,
            _,
            radius_x,
        ) in fixture.ellipsoids_zyx_radii:
            for wall_x in (center_x - radius_x, center_x + radius_x):
                local_center_z = center_z - z_offset
                local_center_y = center_y - y_offset
                local_center_x = center_x - x_offset
                local_wall_x = wall_x - x_offset
                if (
                    0 <= round(local_center_z) < truth_tile.shape[0]
                    and 0 <= round(local_center_y) < truth_tile.shape[1]
                    and 5 <= local_wall_x < truth_tile.shape[2] - 5
                ):
                    center = (local_center_z, local_center_y, local_center_x)
                    widths = (
                        _measure_shell_width(truth_tile, center, local_wall_x, config),
                        _measure_shell_width(deskewed, center, local_wall_x, config),
                        _measure_shell_width(deconvolved, center, local_wall_x, config),
                    )
                    if np.all(np.isfinite(widths)):
                        truth_width, deskewed_width, deconvolved_width = widths
                        truth_line_widths.append(truth_width)
                        deskewed_line_widths.append(deskewed_width)
                        deconvolved_line_widths.append(deconvolved_width)

    assert min(deconvolved_scores) > config.minimum_tile_correlation
    assert min(deskewed_scores) > config.minimum_deskewed_correlation
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
    fusion.run()
    if fixture.configuration == "thin_z_staggered":
        expected_x_corrections = -fixture.recorded_position_errors_zyx_px[:, 2]
        np.testing.assert_allclose(
            np.asarray(fusion.global_offsets)[:, 2],
            expected_x_corrections,
            atol=0.5,
        )
    adjacency = {
        tile_index: set() for tile_index in range(len(fixture.tile_offsets_zyx_px))
    }
    for left, right in fusion.pairwise_metrics:
        adjacency[left].add(right)
        adjacency[right].add(left)
    connected = {0}
    frontier = [0]
    while frontier:
        current = frontier.pop()
        unseen_neighbors = adjacency[current] - connected
        connected.update(unseen_neighbors)
        frontier.extend(unseen_neighbors)
    assert connected == set(range(len(fixture.tile_offsets_zyx_px))), {
        "connected_tiles": connected,
        "pairwise_metrics": fusion.pairwise_metrics,
    }

    reconstructed, overlap_correlation = _assert_fused_overlap_matches_ground_truth(
        fixture,
        config,
        fusion,
    )
    fused_correlation = _measure_correlation(
        reconstructed,
        fixture.ground_truth,
        config,
    )
    assert fused_correlation > config.minimum_fused_correlation, {
        "fused_correlation": fused_correlation,
        "deskewed_correlations": deskewed_scores,
        "deconvolved_correlations": deconvolved_scores,
        "pairwise_metrics": fusion.pairwise_metrics,
        "global_offsets": fusion.global_offsets,
        "offset_um": fusion.offset_um,
        "overlap_correlation": overlap_correlation,
    }
    fused_line_widths = []
    for center_z, center_y, center_x, _, _, radius_x in fixture.ellipsoids_zyx_radii:
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
    assert abs(fused_width - deskewed_width) <= 1.0


def test_tiled_opm_v2_reconstructs_registered_ground_truth(
    opm_v2_tiled_ground_truth_zarr,
    reconstruction_config,
    cupy_gpu,
):
    """Recover the original X-overlap acquisition in mirror and stage modes.

    Parameters
    ----------
    opm_v2_tiled_ground_truth_zarr : object
        Value supplied for ``opm v2 tiled ground truth zarr``.
    reconstruction_config : object
        Value supplied for ``reconstruction config``.
    cupy_gpu : object
        Value supplied for ``cupy gpu``.

    Returns
    -------
    None
        No value is returned.
    """
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
    """Validate YX-grid, Z-staggered, and combined configurations.

    Parameters
    ----------
    opm_v2_spatial_tiling_ground_truth_zarr : object
        Value supplied for ``opm v2 spatial tiling ground truth zarr``.
    reconstruction_config : object
        Value supplied for ``reconstruction config``.
    cupy_gpu : object
        Value supplied for ``cupy gpu``.

    Returns
    -------
    None
        No value is returned.
    """
    _assert_tiled_reconstruction(
        opm_v2_spatial_tiling_ground_truth_zarr,
        reconstruction_config,
        cupy_gpu,
    )


def test_reprocessing_recomputes_registration_before_fusing(
    opm_v2_tiled_ground_truth_zarr,
    reconstruction_config,
    cupy_gpu,
):
    """Reprocess fake tiles, reject their old cache, and validate fused overlap."""
    del cupy_gpu
    fixture = opm_v2_tiled_ground_truth_zarr
    options = reconstruction_config.processing_options()
    metrics_path = fixture.path.parent / "stitching_metrics.json"

    process(root_path=fixture.path, deconvolve=False, **options)
    first_fusion = TileFusion(
        root_path=fixture.path,
        **reconstruction_config.fusion_options(),
    )
    first_fusion.run()
    first_cache = json.loads(metrics_path.read_text(encoding="utf-8"))

    process(root_path=fixture.path, deconvolve=False, **options)
    second_fusion = TileFusion(
        root_path=fixture.path,
        **reconstruction_config.fusion_options(),
    )
    with pytest.raises(ValueError, match="different processed data"):
        second_fusion.load_pairwise_metrics(metrics_path)

    second_fusion.run()
    second_cache = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert (
        second_cache["source"]["processing_created_at"]
        != first_cache["source"]["processing_created_at"]
    )
    processed_path = fixture.path.parent / f"{fixture.path.stem}_deskewed.ome.zarr"
    current_created_at = open_position_collection(processed_path).attributes[
        "opm_processing"
    ]["created_at"]
    assert second_cache["source"]["processing_created_at"] == current_created_at
    _assert_fused_overlap_matches_ground_truth(
        fixture,
        reconstruction_config,
        second_fusion,
    )
