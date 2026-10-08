"""
Deskew qi2lab OPM data.

This file deskews and creates maximum projections of raw qi2lab OPM data.
"""

import hashlib
import math
import multiprocessing as mp
import sys
import time
import traceback
from collections.abc import Iterable, Iterator, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer
from tifffile import TiffFile, TiffWriter, imread
from tqdm import tqdm

from opm_processing.dataio.acquisition import (
    AcquisitionMetadata,
    acquisition_stem,
    inspect_acquisition,
    open_acquisition_datastore,
)
from opm_processing.dataio.live import (
    LIVE_POLL_INTERVAL_SECONDS,
    LiveManifest,
    ZarrTileReadiness,
    iter_live_tiles,
)
from opm_processing.dataio.ngff import downsample_yx, round_spatial
from opm_processing.dataio.position_collection import (
    PositionCollection,
    create_position_collection,
    create_variable_position_collection,
    open_image_array,
    open_position_collection,
)
from opm_processing.dataio.processing_state import (
    ProcessingState,
    processing_state_path,
)
from opm_processing.dataio.roi import (
    PhysicalRoi,
    world_roi_to_skewed_bounds,
)
from opm_processing.imageprocessing.coordinates import (
    stage_z_level_indices,
)
from opm_processing.imageprocessing.camera import (
    camera_correct,
    illumination_correct,
    QI2LAB_STAGE_SCAN_DETECTOR_WIDTH,
)
from opm_processing.imageprocessing.maxtilefusion import MaxTileFusion
from opm_processing.imageprocessing.opmpsf import generate_proj_psf, generate_skewed_psf
from opm_processing.imageprocessing.opmtools import (
    deskew_shape_estimator,
    orthogonal_deskew,
)

app = typer.Typer()
app.pretty_exceptions_enable = False

# Reject cached fields made with a different camera-calibration contract.
FLATFIELD_SOFTWARE = "opm-processing/basicpy-stage-depth-camera-contract-v10"


def progress_tile_groups(
    tiles: Iterable[tuple[int, int]],
    requested_tiles: Iterable[tuple[int, int]],
    completed_tiles: Iterable[tuple[int, int]],
    acquisition_order: Sequence[str],
) -> Iterator[tuple[int, tuple[int]]]:
    """Report acquisition groups and resume-aware progress within each group.

    Parameters
    ----------
    tiles : iterable of tuple
        Pending tile keys in time and position order.
    requested_tiles : iterable of tuple
        All tile keys requested for this processing run.
    completed_tiles : iterable of tuple
        Tile keys already processed before this run.
    acquisition_order : sequence of str
        Acquisition axis order used to select the outer progress tracker.

    Yields
    ------
    tuple
        Time index and a singleton position tuple for each pending tile,
        while updating the outer and inner progress bars.
    """
    axis = next((axis for axis in acquisition_order if axis in ("t", "p")), None)
    description, unit = {
        "t": ("time", "timepoint"),
        "p": ("positions", "position"),
        None: ("volumes", "volume"),
    }[axis]

    axis_index = {"t": 0, "p": 1, None: None}[axis]
    requested = set(requested_tiles)
    completed = requested.intersection(completed_tiles)
    groups = {}
    for tile in requested:
        key = tile[axis_index] if axis_index is not None else tile
        counts = groups.setdefault(key, [0, 0])
        counts[0] += 1
        counts[1] += tile in completed
    with (
        tqdm(
            total=len(groups),
            initial=sum(total == done for total, done in groups.values()),
            desc=description,
            unit=unit,
        ) as progress,
        ExitStack() as inner_stack,
    ):
        inner_progress = None
        current_group = None
        for time_index, position_index in tiles:
            tile = (time_index, position_index)
            key = tile[axis_index] if axis_index is not None else tile
            if axis is not None and key != current_group:
                inner_stack.close()
                current_group = key
                inner_description, inner_unit = (
                    ("positions", "position") if axis == "t" else ("time", "timepoint")
                )
                inner_progress = inner_stack.enter_context(
                    tqdm(
                        total=groups[key][0],
                        initial=groups[key][1],
                        desc=inner_description,
                        unit=inner_unit,
                        leave=False,
                    )
                )
            yield time_index, (position_index,)
            # Resume after the caller has processed and checkpointed this
            # volume. A timepoint may contain several selected positions.
            if tile in requested and tile not in completed:
                completed.add(tile)
                if inner_progress is not None:
                    inner_progress.update(1)
                groups[key][1] += 1
                if groups[key][1] == groups[key][0]:
                    progress.update(1)


def open_resume_collection(
    path: Path,
    expected_shape: tuple[int, int, int, int, int, int],
    expected_dtype: np.dtype,
    expected_multiscale_factors_yx: tuple[int, ...] = (1,),
) -> PositionCollection:
    """Open and validate a processed collection from an interrupted run.

    Parameters
    ----------
    path : pathlib.Path
        Filesystem path of the artifact to read or write.
    expected_shape : tuple[int, int, int, int, int, int]
        Required collection dimensions in TPCZYX order.
    expected_dtype : numpy.dtype
        Required dtype of every existing output image array.
    expected_multiscale_factors_yx : tuple[int, ...]
        Required absolute YX pyramid factors, including level zero factor 1.

    Returns
    -------
    PositionCollection
        Existing collection after shape, dtype, and pyramid validation.
    """
    collection = open_position_collection(path)
    if collection.shape != expected_shape:
        raise ValueError(
            "Existing output shape is incompatible with this run: "
            f"{collection.shape} != {expected_shape}. Remove or relocate {path}."
        )
    actual_dtypes = {np.dtype(array.dtype.numpy_dtype) for array in collection.arrays}
    if actual_dtypes != {np.dtype(expected_dtype)}:
        raise ValueError(
            "Existing output dtype is incompatible with this run: "
            f"{sorted(str(dtype) for dtype in actual_dtypes)} != "
            f"{expected_dtype}. Remove or relocate {path}."
        )
    if collection.multiscale_factors_yx != expected_multiscale_factors_yx:
        raise ValueError(
            "Existing output pyramid is incompatible with this run: "
            f"{collection.multiscale_factors_yx} != "
            f"{expected_multiscale_factors_yx}. Remove or relocate {path}."
        )
    return collection


def open_variable_resume_collection(
    path: Path,
    expected_shapes: list[tuple[int, ...]],
    expected_dtype: np.dtype,
) -> PositionCollection:
    """Open and validate an interrupted variable-size ROI collection.

    Parameters
    ----------
    path : pathlib.Path
        Filesystem path of the artifact to read or write.
    expected_shapes : list[tuple[int, ...]]
        Required TCZYX dimensions for each cropped ROI series, in storage order.
    expected_dtype : numpy.dtype
        Required dtype of every existing output image array.

    Returns
    -------
    PositionCollection
        Existing ROI collection after per-series shape and dtype validation.
    """
    collection = open_position_collection(path)
    actual_shapes = [
        tuple(int(value) for value in array.shape) for array in collection.arrays
    ]
    if actual_shapes != expected_shapes:
        raise ValueError(
            "Existing ROI output shapes are incompatible with this run: "
            f"{actual_shapes} != {expected_shapes}. Use --no-resume to overwrite it."
        )
    actual_dtypes = {np.dtype(array.dtype.numpy_dtype) for array in collection.arrays}
    if actual_dtypes != {np.dtype(expected_dtype)}:
        raise ValueError(
            "Existing ROI output dtype is incompatible with this run: "
            f"{sorted(str(dtype) for dtype in actual_dtypes)} != {expected_dtype}. "
            "Use --no-resume to overwrite it."
        )
    return collection


def initialize_processing_state(
    *,
    output_dir: Path,
    source_path: Path,
    output_path: Path,
    configuration: dict[str, Any],
    roi_series: tuple[dict[str, Any], ...] = (),
    resume: bool,
    output_preexisting: bool,
) -> ProcessingState:
    """Create or validate the only durable state for one processing run.

    Parameters
    ----------
    output_dir : pathlib.Path
        Directory containing processed images and state.
    source_path : pathlib.Path
        Raw acquisition path recorded as the processing source.
    output_path : pathlib.Path
        Processed store identified by this checkpoint or processing run.
    configuration : dict[str, Any]
        Pixel-affecting processing options used to validate a resumed run.
    roi_series : tuple[dict[str, Any], ...]
        Source indices and crop geometry for each variable-size ROI series.
    resume : bool
        Reuse compatible output checkpoints; False overwrites this output run.
    output_preexisting : bool
        Whether the output store existed before initialization.

    Returns
    -------
    ProcessingState
        Initialized journal with compatible checkpoints, or a fresh output run.
    """
    state_path = processing_state_path(output_dir, acquisition_stem(source_path))
    if resume and output_preexisting and not state_path.is_file():
        raise ValueError(
            "Existing output has no current processing-state JSON and cannot be "
            f"resumed: {state_path}. Use overwrite mode to restart."
        )
    state = ProcessingState.open(
        state_path,
        source_path,
        overwrite=not resume,
    )
    if resume and output_preexisting:
        try:
            state.run(output_path)
        except ValueError as error:
            raise ValueError(
                "Existing output has no current processing run and cannot be "
                f"resumed: {output_path}. Use overwrite mode to restart."
            ) from error
    state.initialize_run(
        output_path,
        configuration=configuration,
        roi_series=roi_series,
        overwrite=not resume or not output_preexisting,
    )
    return state


def format_processed_output(data: np.ndarray, save_float32: bool) -> np.ndarray:
    """Convert a completed float32 processing product for persistent storage.

    Parameters
    ----------
    data : numpy.ndarray
        Loaded image values; float32 is required at processing boundaries.
    save_float32 : bool
        Preserve float32 intensities; False clips and casts final output to uint16.

    Returns
    -------
    numpy.ndarray
        Original float32 data or a uint16 copy clipped to 0 through 65535.
    """
    if save_float32:
        return data
    return np.clip(data, 0, np.iinfo(np.uint16).max).astype(np.uint16)


def write_checkpointed_roi_channel(
    target: Any,
    value: np.ndarray | np.generic,
    state: ProcessingState,
    output_path: Path,
    channel_key: tuple[int, int, int],
    *,
    is_zero: bool,
) -> None:
    """Finish a cropped channel write before recording it as resumable.

    Parameters
    ----------
    target : Any
        Writable TensorStore selection for one cropped output channel.
    value : numpy.ndarray or np.generic
        Channel image or scalar fill value to write before checkpointing.
    state : ProcessingState
        Durable processing journal receiving the completed-channel checkpoint.
    output_path : pathlib.Path
        Processed store identified by this checkpoint or processing run.
    channel_key : tuple[int, int, int]
        Source time, position, and channel indices, in that order.
    is_zero : bool
        Record that this channel contains no signal.

    Returns
    -------
    None
        The channel is checkpointed only after its write finishes successfully.
    """
    target.write(value).result()
    state.complete_channel(output_path, *channel_key, is_zero=is_zero)


def queue_position_pyramid_writes(
    collection: PositionCollection,
    position: int,
    timepoint: int,
    channel: int,
    level_zero: np.ndarray | np.generic,
) -> list[Any]:
    """Queue one TCZYX selection across every per-position pyramid level.

    Parameters
    ----------
    collection : PositionCollection
        Output position collection containing level-zero and pyramid arrays.
    position : int
        Zero-based position index in the image collection.
    timepoint : int
        Zero-based time index in the output collection.
    channel : int
        Zero-based channel index.
    level_zero : numpy.ndarray or np.generic
        ZYX image or scalar fill used to write each YX pyramid level.

    Returns
    -------
    list
        Write futures that the caller must finish before checkpointing the tile.
    """
    writes = []
    for factor, level_arrays in zip(
        collection.multiscale_factors_yx,
        collection.multiscale_arrays,
    ):
        value = (
            level_zero
            if factor == 1 or np.ndim(level_zero) == 0
            else downsample_yx(
                np.asarray(level_zero),
                factor,
                collection.multiscale_downsample,
            )
        )
        writes.append(level_arrays[position][timepoint, channel].write(value))
    return writes


def validate_empty_tile_options(
    threshold: float | None,
    min_signal_fraction: float,
) -> tuple[float | None, float]:
    """Validate options controlling the empty-tile fast path.

    Parameters
    ----------
    threshold : float or None
        Minimum calibrated intensity counted as signal; None disables empty
        detection.
    min_signal_fraction : float
        Required fraction of volume pixels at or above the signal threshold.

    Returns
    -------
    tuple
        Validated signal threshold and minimum signal fraction.
    """
    if threshold is not None:
        threshold = float(threshold)
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("--skip-empty-below must be finite and nonnegative")
    min_signal_fraction = float(min_signal_fraction)
    if (
        not np.isfinite(min_signal_fraction)
        or min_signal_fraction <= 0
        or min_signal_fraction > 1
    ):
        raise ValueError(
            "--skip-empty-min-signal-fraction must be greater than 0 and at most 1"
        )
    return threshold, min_signal_fraction


def is_empty_tile(
    data: np.ndarray,
    threshold: float | None,
    min_signal_fraction: float,
) -> bool:
    """Return whether too little of a calibrated channel volume contains signal.

    Parameters
    ----------
    data : numpy.ndarray
        Camera-calibrated channel image or volume, before illumination correction.
    threshold : float or None
        Minimum calibrated intensity counted as signal; None disables empty
        detection.
    min_signal_fraction : float
        Required fraction of volume pixels at or above the signal threshold.

    Returns
    -------
    bool
        True when too few pixels meet the threshold; False when detection is
        disabled.
    """
    if threshold is None:
        return False
    signal_fraction = np.count_nonzero(data >= threshold) / data.size
    return bool(signal_fraction < min_signal_fraction)


def build_illumination_signal_decisions(
    datastore,
    stage_z_indices: np.ndarray,
    camera_offset: float,
    camera_conversion: float,
    threshold: float | None,
    min_signal_fraction: float,
    *,
    apply_stage_scan_gain: bool,
) -> np.ndarray | None:
    """Find nonempty illumination candidates without pre-scanning every tile.

    Parameters
    ----------
    datastore : tensorstore.TensorStore
        Readable raw TensorStore with TPCZYX axes and uint16 camera values.
    stage_z_indices : numpy.ndarray
        Depth-level index for each acquisition position.
    camera_offset : float
        Camera baseline in ADU, subtracted before applying the conversion.
    camera_conversion : float
        Calibrated intensity units per ADU after baseline subtraction.
    threshold : float or None
        Minimum calibrated intensity counted as signal; None disables empty
        detection.
    min_signal_fraction : float
        Required fraction of volume pixels at or above the signal threshold.
    apply_stage_scan_gain : bool
        Apply the fixed qi2lab detector-X gain during camera calibration.

    Returns
    -------
    numpy.ndarray or None
        TPC tri-state decisions for sampled time-zero candidates, or None when
        disabled.
    """
    if threshold is None:
        return None
    decisions = np.full(
        tuple(int(value) for value in datastore.shape[:3]),
        -1,
        dtype=np.int8,
    )
    stage_z_indices = np.asarray(stage_z_indices, dtype=np.int64)
    position_groups = tuple(
        tuple(int(value) for value in np.flatnonzero(stage_z_indices == stage_level))
        for stage_level in range(int(stage_z_indices.max()) + 1)
    )
    target_total = sum(
        min(32, len(positions)) * decisions.shape[2] for positions in position_groups
    )
    progress = tqdm(
        total=target_total,
        desc="Finding illumination tiles",
        unit="nonzero tile",
    )
    try:
        for stage_level, positions in enumerate(position_groups):
            target_count = min(32, len(positions))
            candidates = positions
            if len(positions) > target_count:
                # Try evenly spaced tiles first, then replace empty candidates.
                primary_indices = np.unique(
                    np.linspace(0, len(positions) - 1, target_count, dtype=np.int64)
                )
                primary = tuple(positions[int(index)] for index in primary_indices)
                primary_set = set(primary)
                candidates = primary + tuple(
                    position for position in positions if position not in primary_set
                )
            for channel in range(decisions.shape[2]):
                nonempty_count = 0
                for position in candidates:
                    raw = np.squeeze(datastore[0, position, channel, :].read().result())
                    calibrated = camera_correct(
                        raw,
                        camera_offset,
                        camera_conversion,
                        apply_stage_scan_gain=apply_stage_scan_gain,
                    )
                    is_empty = is_empty_tile(
                        calibrated,
                        threshold,
                        min_signal_fraction,
                    )
                    decisions[0, position, channel] = 0 if is_empty else 1
                    if not is_empty:
                        nonempty_count += 1
                        progress.update()
                    if nonempty_count >= target_count:
                        break
    finally:
        progress.close()
    return decisions


def tile_is_empty(
    signal_decisions: np.ndarray | None,
    time_index: int,
    position: int,
    channel: int,
    calibrated: np.ndarray,
    threshold: float | None,
    min_signal_fraction: float,
) -> bool:
    """Reuse or populate a tri-state decision while processing loaded data.

    Parameters
    ----------
    signal_decisions : numpy.ndarray or None
        TPC decision array: -1 unknown, 0 empty, 1 nonempty; None disables caching.
    time_index : int
        Zero-based acquisition time index.
    position : int
        Zero-based position index in the image collection.
    channel : int
        Zero-based channel index.
    calibrated : numpy.ndarray
        Camera-corrected channel volume before illumination correction.
    threshold : float or None
        Minimum calibrated intensity counted as signal; None disables empty
        detection.
    min_signal_fraction : float
        Required fraction of volume pixels at or above the signal threshold.

    Returns
    -------
    bool
        Cached or measured empty decision; the supplied cache is updated in place.
    """
    if signal_decisions is not None:
        decision = int(signal_decisions[time_index, position, channel])
        if decision >= 0:
            return decision == 0
    is_empty = is_empty_tile(calibrated, threshold, min_signal_fraction)
    if signal_decisions is not None:
        signal_decisions[time_index, position, channel] = 0 if is_empty else 1
    return is_empty


def read_current_flatfield(
    path: Path,
    expected_shape: tuple[int, int, int, int],
) -> np.ndarray | None:
    """Read a flatfield only when it was made by the current estimator.

    Parameters
    ----------
    path : pathlib.Path
        Filesystem path of the artifact to read or write.
    expected_shape : tuple[int, int, int, int]
        Required depth, channel, Y, X dimensions.

    Returns
    -------
    numpy.ndarray or None
        Depth-CYX fields, or None when the estimator version or acquisition
        dimensions differ from this run.
    """
    with TiffFile(path) as tif:
        tag = tif.pages[0].tags.get("Software")
        if tag is None or tag.value != FLATFIELD_SOFTWARE:
            return None
        flatfields = tif.asarray().astype(np.float32)
    if expected_shape[0] == 1 and flatfields.shape == expected_shape[1:]:
        flatfields = flatfields[np.newaxis, ...]
    if flatfields.shape != expected_shape:
        return None
    return flatfields


def write_flatfield(
    path: Path,
    flatfields: np.ndarray,
    pixel_size_um: float,
) -> None:
    """Write reusable physical-stage-Z illumination fields.

    Parameters
    ----------
    path : pathlib.Path
        Filesystem path of the artifact to read or write.
    flatfields : numpy.ndarray
        Positive float32 illumination fields in depth, channel, Y, X order.
    pixel_size_um : float
        Camera pixel pitch in micrometers.

    Returns
    -------
    None
        Fields are written with spatial metadata and the current estimator
        identifier.
    """
    with TiffWriter(path, bigtiff=True) as tif:
        metadata = {
            "axes": "ZCYX",
            "SignificantBits": 32,
            "PhysicalSizeX": pixel_size_um,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": pixel_size_um,
            "PhysicalSizeYUnit": "µm",
        }
        tif.write(
            flatfields,
            resolution=(1e4 / pixel_size_um, 1e4 / pixel_size_um),
            photometric="minisblack",
            resolutionunit="CENTIMETER",
            software=FLATFIELD_SOFTWARE,
            metadata=metadata,
        )


def load_or_estimate_flatfield(
    path: Path,
    datastore,
    camera_offset: float,
    camera_conversion: float,
    pixel_size_um: float,
    stage_positions_zxy: np.ndarray,
    *,
    apply_stage_scan_gain: bool,
    signal_threshold: float | None,
    minimum_signal_fraction: float,
    signal_mask: np.ndarray | None,
) -> np.ndarray:
    """Reuse a current flatfield or replace an obsolete estimator artifact.

    Parameters
    ----------
    path : pathlib.Path
        Filesystem path of the artifact to read or write.
    datastore : tensorstore.TensorStore
        Readable raw TensorStore with TPCZYX axes and uint16 camera values.
    camera_offset : float
        Camera baseline in ADU, subtracted before applying the conversion.
    camera_conversion : float
        Calibrated intensity units per ADU after baseline subtraction.
    pixel_size_um : float
        Camera pixel pitch in micrometers.
    stage_positions_zxy : numpy.ndarray
        Physical stage Z, scan-axis X, and lateral Y coordinates in micrometers.
    apply_stage_scan_gain : bool
        Apply the fixed qi2lab detector-X gain during camera calibration.
    signal_threshold : float or None
        Calibrated signal cutoff for selecting illumination samples; None disables
        it.
    minimum_signal_fraction : float
        Minimum fraction of candidate pixels that must meet the signal cutoff.
    signal_mask : numpy.ndarray or None
        TPC mask selecting nonempty estimation tiles; None allows automatic
        selection.

    Returns
    -------
    numpy.ndarray
        Depth-CYX illumination fields reused from disk or estimated and saved.
    """
    stage_z_indices = stage_z_level_indices(stage_positions_zxy)
    expected_shape = (
        int(stage_z_indices.max()) + 1,
        datastore.shape[2],
        datastore.shape[-2],
        datastore.shape[-1],
    )
    if path.exists():
        calibration = read_current_flatfield(
            path,
            expected_shape,
        )
        if calibration is not None:
            print(f"Using existing flatfield: {path}")
            return calibration
        print("Existing flatfield uses an obsolete estimator; re-estimating it.")

    if signal_threshold is not None and signal_mask is None:
        signal_mask = build_illumination_signal_decisions(
            datastore,
            stage_z_indices,
            camera_offset,
            camera_conversion,
            signal_threshold,
            minimum_signal_fraction,
            apply_stage_scan_gain=apply_stage_scan_gain,
        )

    flatfields = call_estimate_illuminations(
        datastore,
        camera_offset,
        camera_conversion,
        np.asarray(stage_positions_zxy, dtype=float),
        apply_stage_scan_gain,
        None if signal_mask is None else signal_mask == 1,
    )
    write_flatfield(
        path,
        flatfields,
        pixel_size_um,
    )
    return flatfields


def file_sha256(path: Path) -> str:
    """Return a stable SHA-256 digest for a processing input artifact.

    Parameters
    ----------
    path : pathlib.Path
        Filesystem path of the artifact to read or write.

    Returns
    -------
    str
        Hexadecimal SHA-256 digest of the file contents.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def open_live_acquisition(
    root_path: Path,
) -> tuple[AcquisitionMetadata, LiveManifest, Path]:
    """Wait for and resolve a live manifest plus its OME-Zarr hierarchy.

    Parameters
    ----------
    root_path : pathlib.Path
        Raw acquisition store, or its containing directory for live processing.

    Returns
    -------
    tuple
        Acquisition metadata, its live manifest, and lifecycle log path after
        metadata appears.

    Notes
    -----
    Poll until a single manifest and the required OME-Zarr metadata exist.
    The manifest's data_path locates the source store.
    """
    requested_path = Path(root_path).expanduser().resolve()
    if requested_path.name.endswith((".zarr", ".ome.zarr")):
        raise ValueError(
            "--live requires the containing acquisition directory, not the "
            "OME-Zarr path"
        )
    if not requested_path.is_dir():
        raise ValueError("--live requires the containing acquisition directory")
    while True:
        manifests = sorted(requested_path.glob("*.manifest.json"))
        if len(manifests) == 1:
            break
        if len(manifests) > 1:
            raise ValueError(
                "Live acquisition directory contains multiple manifests: "
                + ", ".join(str(path) for path in manifests)
            )
        print(
            f"Waiting {LIVE_POLL_INTERVAL_SECONDS:g} seconds for live manifest "
            f"in {requested_path}..."
        )
        time.sleep(LIVE_POLL_INTERVAL_SECONDS)
    manifest_path = manifests[0]
    manifest_stem = manifest_path.name.removesuffix(".manifest.json")
    log_path = manifest_path.with_name(f"{manifest_stem}.log.jsonl")
    manifest = LiveManifest.read(manifest_path)
    root_path = manifest.data_path
    required_metadata = [root_path / "zarr.json", root_path / "OME" / "zarr.json"]
    required_metadata.extend(
        metadata_path
        for position in range(manifest.index_sizes["p"])
        for metadata_path in (
            root_path / str(position) / "zarr.json",
            root_path / str(position) / "0" / "zarr.json",
        )
    )
    while not all(path.is_file() for path in required_metadata):
        print(
            f"Waiting {LIVE_POLL_INTERVAL_SECONDS:g} seconds for OME-Zarr metadata..."
        )
        time.sleep(LIVE_POLL_INTERVAL_SECONDS)
    acquisition = manifest.apply(inspect_acquisition(root_path))
    return acquisition, manifest, log_path


def plan_roi_tiles(
    roi: PhysicalRoi,
    time_indices: tuple[int, ...],
    position_indices: tuple[int, ...],
    deskew_input_shape: tuple[int, int, int],
    *,
    channel_count: int,
    pixel_size_um: float,
    scan_step_um: float,
    angle_deg: float,
    z_downsample_level: int,
    halo_scan: int,
    halo_x: int,
) -> tuple[dict[str, Any], ...]:
    """Plan variable-size cropped ROI tiles without full-tile zero padding.

    Parameters
    ----------
    roi : PhysicalRoi
        Physical ROI with registered tile footprints and origins.
    time_indices : tuple[int, ...]
        Source time indices included in the cropped output.
    position_indices : tuple[int, ...]
        Source positions included in the cropped output.
    deskew_input_shape : tuple[int, int, int]
        Raw scan, camera-Y, and camera-X dimensions after scan trimming.
    channel_count : int
        Number of channels stored in each cropped output series.
    pixel_size_um : float
        Camera pixel pitch in micrometers.
    scan_step_um : float
        Distance between adjacent acquisition scan planes in micrometers.
    angle_deg : float
        Oblique imaging-plane angle in degrees.
    z_downsample_level : int
        Integer reduction factor along deskewed Z.
    halo_scan : int
        Extra scan planes retained around each ROI for deconvolution support.
    halo_x : int
        Extra camera-X pixels retained around each ROI for deconvolution support.

    Returns
    -------
    tuple of dict
        Per-series source indices, scan/X crops, output TCZYX shapes, and physical
        origins.
    """
    plans: list[dict[str, Any]] = []
    roi_y0, roi_y1, roi_x0, roi_x1 = roi.bounds_yx_um
    for time_index in time_indices:
        for position_index in position_indices:
            registered_origin = roi.registered_tile_origin_zyx_um(
                time_index,
                position_index,
            )
            skewed = world_roi_to_skewed_bounds(
                roi.bounds_yx_um,
                registered_origin[-2:],
                deskew_input_shape,
                pixel_size_um=pixel_size_um,
                scan_step_um=scan_step_um,
                angle_deg=angle_deg,
                halo_scan=halo_scan,
                halo_x=halo_x,
            )
            if skewed is None:
                continue
            local_input_shape = (
                skewed.scan_stop - skewed.scan_start,
                int(deskew_input_shape[1]),
                skewed.x_stop - skewed.x_start,
            )
            local_deskew_shape, _pad_y, _pad_x, _crop_y = deskew_shape_estimator(
                local_input_shape,
                theta=angle_deg,
                distance=scan_step_um,
                pixel_size=pixel_size_um,
                crop_after_deskew=False,
            )
            local_z = int(local_deskew_shape[0]) // int(z_downsample_level)
            local_y = int(local_deskew_shape[1])
            local_x = int(local_deskew_shape[2])
            local_origin_y = float(registered_origin[1]) + int(
                skewed.scan_start
            ) * float(scan_step_um)
            local_origin_x = float(registered_origin[2]) + int(skewed.x_start) * float(
                pixel_size_um
            )
            crop_y0 = max(
                0,
                math.floor((roi_y0 - local_origin_y) / pixel_size_um),
            )
            crop_y1 = min(
                local_y,
                math.ceil((roi_y1 - local_origin_y) / pixel_size_um),
            )
            crop_x0 = max(
                0,
                math.floor((roi_x0 - local_origin_x) / pixel_size_um),
            )
            crop_x1 = min(
                local_x,
                math.ceil((roi_x1 - local_origin_x) / pixel_size_um),
            )
            if crop_y0 >= crop_y1 or crop_x0 >= crop_x1:
                continue
            output_origin = (
                round_spatial(registered_origin[0]),
                round_spatial(local_origin_y + crop_y0 * pixel_size_um),
                round_spatial(local_origin_x + crop_x0 * pixel_size_um),
            )
            plans.append(
                {
                    "time_index": int(time_index),
                    "position_index": int(position_index),
                    "origin_zyx_um": list(output_origin),
                    "shape_tczyx": [
                        1,
                        int(channel_count),
                        local_z,
                        crop_y1 - crop_y0,
                        crop_x1 - crop_x0,
                    ],
                    "skewed_bounds": skewed,
                    "skewed_bounds_sx": [
                        int(skewed.scan_start),
                        int(skewed.scan_stop),
                        int(skewed.x_start),
                        int(skewed.x_stop),
                    ],
                    "skewed_shape_syx": list(local_input_shape),
                    "deskew_crop_yx": [crop_y0, crop_y1, crop_x0, crop_x1],
                }
            )
    if not plans:
        raise ValueError("The ROI does not contain any processable tile regions")
    return tuple(plans)


def apply_stage_axis_flips(
    stage_positions: np.ndarray,
    axis_flips_xyz: tuple[bool, bool, bool],
) -> np.ndarray:
    """Apply configured camera-to-stage orientation flips to ZXY positions.

    Parameters
    ----------
    stage_positions : numpy.ndarray
        Stage coordinates in Z, scan-axis X, and lateral Y order.
    axis_flips_xyz : tuple[bool, bool, bool]
        X, Y, Z flags reflecting coordinates about each selected axis maximum.

    Returns
    -------
    numpy.ndarray
        Transformed floating-point coordinate copy; the input is unchanged.
    """
    transformed = np.asarray(stage_positions, dtype=float).copy()
    for should_flip, column in zip(axis_flips_xyz, (2, 1, 0)):
        if should_flip:
            transformed[:, column] = (
                np.max(transformed[:, column]) - transformed[:, column]
            )
    return transformed


@app.command()
def process(
    root_path: Path,
    deconvolve: bool = False,
    save_float32: Annotated[
        bool,
        typer.Option(
            "--save-float32",
            help=(
                "Keep calibrated fractional intensities and save processed "
                "outputs as float32 instead of uint16."
            ),
        ),
    ] = False,
    skip_empty_below: Annotated[
        float | None,
        typer.Option(
            "--skip-empty-below",
            help=(
                "Skip deconvolution and deskew when too little of the calibrated "
                "channel volume is at or above this intensity. The check occurs "
                "before illumination correction."
            ),
        ),
    ] = None,
    skip_empty_min_signal_fraction: Annotated[
        float,
        typer.Option(
            "--skip-empty-min-signal-fraction",
            help=(
                "Fraction of the calibrated channel volume that must meet "
                "--skip-empty-below."
            ),
        ),
    ] = 0.01,
    max_projection: bool = True,
    flatfield_correction: bool = False,
    create_fused_max_projection: bool = True,
    write_fused_max_projection_tiff: bool = False,
    z_downsample_level: int = 2,
    crop_after_deskew: bool = False,
    time_range: tuple[int, int] = None,
    pos_range: tuple[int, int] = None,
    eager_mode: bool = False,
    decon_crop_scan: int | None = None,
    decon_gpu_id: int = 0,
    decon_verbose: int = 1,
    decon_fallback_step_scan: int | None = None,
    decon_psf_paths: list[Path] | None = None,
    decon_scan_upsample: Annotated[
        int | None,
        typer.Option(
            "--decon-scan-upsample",
            min=1,
            help=(
                "Opt into experimental undersampled RLGC with this integer scan "
                "upsampling factor. Requires --deconvolve and a 3D scan. "
                "Uses a full volume on the GPU; supplied PSFs must use the finer "
                "scan spacing."
            ),
        ),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume",
            help=(
                "Reuse compatible processed outputs and continue after the last "
                "durably completed tile. Without this flag, outputs are overwritten."
            ),
        ),
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            help=(
                "Directory for processed artifacts. The directory is created "
                "when it does not exist."
            ),
        ),
    ] = None,
    live: Annotated[
        Path | None,
        typer.Option(
            "--live",
            help=(
                "Process complete tiles while acquisition is running, using this "
                "CYX illumination OME-TIFF. Illumination is never estimated."
            ),
        ),
    ] = None,
):
    """Process an OPM acquisition using its recorded scan and camera calibration.

    Parameters
    ----------
    root_path : pathlib.Path
        Raw acquisition store, or its containing directory for live processing.
    deconvolve : bool
        Apply Richardson–Lucy gradient-consensus deconvolution before saving.
    save_float32 : bool
        Preserve float32 intensities; False clips and casts final output to uint16.
    skip_empty_below : float or None
        Calibrated signal cutoff for skipping empty channels; None disables
        skipping.
    skip_empty_min_signal_fraction : float
        Minimum fraction of channel-volume pixels required above the cutoff.
    max_projection : bool
        Write per-tile maximum-Z images; False also disables their fused mosaic.
    flatfield_correction : bool
        Estimate or reuse illumination fields and divide calibrated data by them.
    create_fused_max_projection : bool
        Fuse per-tile maximum-Z images using nominal stage placements.
    write_fused_max_projection_tiff : bool
        Export the stage-placed maximum projection as OME-TIFF.
    z_downsample_level : int
        Integer reduction factor along deskewed Z.
    crop_after_deskew : bool
        Crop deskewed Y to the region with support across the complete Z range.
    time_range : tuple[int, int] or None
        Start-inclusive, stop-exclusive time selection; None processes all times.
    pos_range : tuple[int, int] or None
        Start-inclusive, stop-exclusive position selection; None processes all
        positions.
    eager_mode : bool
        Use eager stopping for planar deconvolution; unused for skewed volumes.
    decon_crop_scan : int or None
        Retained scan-plane chunk size for 3D deconvolution; None calibrates it
        automatically.
    decon_gpu_id : int
        Zero-based CUDA device index used by the deconvolution solver.
    decon_verbose : int
        Solver diagnostic verbosity, with zero suppressing iteration reports.
    decon_fallback_step_scan : int or None
        Scan planes removed from a chunk after GPU allocation failure; None uses the
        solver default.
    decon_psf_paths : list[Path] or None
        Channel-ordered PSF files; None generates theoretical PSFs. Planar
        processing uses a central Z slice of 3D PSFs.
    decon_scan_upsample : int or None
        Experimental integer scan upsampling factor; None uses the standard solver.
        Requires 3D deconvolution and a full GPU volume.
    resume : bool
        Reuse compatible output checkpoints; False overwrites this output run.
    output : Path or None
        Output directory to create; None uses the acquisition directory.
    live : Path or None
        Supplied CYX illumination TIFF enabling live processing; None processes
        offline.

    Returns
    -------
    None
        Processed images and durable checkpoints are written to the output
        directory.
    """
    skip_empty_below, skip_empty_min_signal_fraction = validate_empty_tile_options(
        skip_empty_below,
        skip_empty_min_signal_fraction,
    )

    if decon_scan_upsample is not None:
        if not deconvolve:
            raise typer.BadParameter("--decon-scan-upsample requires --deconvolve")
        if decon_scan_upsample < 1 or int(decon_scan_upsample) != decon_scan_upsample:
            raise typer.BadParameter("--decon-scan-upsample must be a positive integer")
        if decon_crop_scan is not None or decon_fallback_step_scan is not None:
            raise typer.BadParameter(
                "Experimental undersampled RLGC uses a full GPU volume; "
                "--decon-crop-scan and --decon-fallback-step-scan are unsupported"
            )

    if live is not None:
        if flatfield_correction:
            raise ValueError(
                "--live supplies illumination directly and cannot be combined with "
                "--flatfield-correction"
            )
        if time_range is not None or pos_range is not None:
            raise ValueError("--live does not support time or position subranges")
        acquisition, live_manifest, live_log_path = open_live_acquisition(root_path)
        root_path = acquisition.path
        opm_mode = acquisition.mode.casefold()
        if "mirror" not in opm_mode and "stage" not in opm_mode:
            raise ValueError(
                "--live currently supports mirror- and stage-scan acquisitions only"
            )
        process_skewed(
            root_path=root_path,
            output_dir=output,
            acquisition=acquisition,
            deconvolve=deconvolve,
            save_float32=save_float32,
            skip_empty_below=skip_empty_below,
            skip_empty_min_signal_fraction=skip_empty_min_signal_fraction,
            max_projection=max_projection,
            flatfield_correction=True,
            create_fused_max_projection=create_fused_max_projection,
            write_fused_max_projection_tiff=write_fused_max_projection_tiff,
            z_downsample_level=z_downsample_level,
            crop_after_deskew=crop_after_deskew,
            decon_crop_scan=decon_crop_scan,
            decon_gpu_id=decon_gpu_id,
            decon_verbose=decon_verbose,
            decon_fallback_step_scan=decon_fallback_step_scan,
            decon_psf_paths=decon_psf_paths,
            decon_scan_upsample=decon_scan_upsample,
            illumination_path=live,
            live_manifest=live_manifest,
            live_log_path=live_log_path,
            resume=True,
        )
        return

    acquisition = inspect_acquisition(root_path)
    root_path = acquisition.path
    opm_mode = acquisition.mode.casefold()
    sizes = acquisition.index_sizes
    print(
        "Acquisition metadata: "
        f"T={sizes.get('t', 1)}, P={sizes.get('p', 1)}, "
        f"C={sizes.get('c', 1)}, Z={sizes.get('z', 1)}; "
        f"channels={list(acquisition.channel_names)}"
    )
    print(f"Processing OPM mode: {opm_mode}")
    common = {
        "root_path": root_path,
        "output_dir": output,
        "acquisition": acquisition,
        "deconvolve": deconvolve,
        "save_float32": save_float32,
        "skip_empty_below": skip_empty_below,
        "skip_empty_min_signal_fraction": skip_empty_min_signal_fraction,
        "flatfield_correction": flatfield_correction,
        "write_fused_max_projection_tiff": write_fused_max_projection_tiff,
        "time_range": time_range,
        "pos_range": pos_range,
        "decon_gpu_id": decon_gpu_id,
        "decon_verbose": decon_verbose,
        "decon_psf_paths": decon_psf_paths,
        "resume": resume,
    }
    if acquisition.is_2d or "projection" in opm_mode:
        if decon_scan_upsample is not None:
            raise typer.BadParameter(
                "--decon-scan-upsample requires a 3D mirror or stage scan"
            )
        process_projection(
            **common,
            eager_deconvolution=eager_mode,
        )
    elif "mirror" in opm_mode or "stage" in opm_mode:
        process_skewed(
            **common,
            decon_crop_scan=decon_crop_scan,
            decon_fallback_step_scan=decon_fallback_step_scan,
            decon_scan_upsample=decon_scan_upsample,
            max_projection=max_projection,
            create_fused_max_projection=create_fused_max_projection,
            z_downsample_level=z_downsample_level,
            crop_after_deskew=crop_after_deskew,
        )
    else:
        raise ValueError(f"Unsupported OPM acquisition mode: {acquisition.mode}")


def process_skewed(
    root_path: Path,
    acquisition: AcquisitionMetadata,
    deconvolve: bool = False,
    save_float32: bool = False,
    skip_empty_below: float | None = None,
    skip_empty_min_signal_fraction: float = 0.01,
    max_projection: bool = True,
    flatfield_correction: bool = False,
    create_fused_max_projection: bool = True,
    write_fused_max_projection_tiff: bool = False,
    z_downsample_level: int = 2,
    crop_after_deskew: bool = False,
    time_range: tuple[int, int] = None,
    pos_range: tuple[int, int] = None,
    decon_crop_scan: int | None = None,
    decon_gpu_id: int = 0,
    decon_verbose: int = 1,
    decon_fallback_step_scan: int | None = None,
    decon_psf_paths: list[Path] | None = None,
    output_dir: Path | None = None,
    illumination_path: Path | None = None,
    live_manifest: LiveManifest | None = None,
    live_log_path: Path | None = None,
    roi_selection: PhysicalRoi | None = None,
    resume: bool = False,
    decon_scan_upsample: int | None = None,
):
    """Calibrate, optionally deconvolve, and deskew mirror or stage scan tiles.

    Parameters
    ----------
    root_path : pathlib.Path
        Resolved path to the raw acquisition store.
    acquisition : AcquisitionMetadata
        Inspected scan geometry, channels, stage placement, and camera calibration.
    deconvolve : bool
        Apply Richardson–Lucy gradient-consensus deconvolution before saving.
    save_float32 : bool
        Preserve float32 intensities; False clips and casts final output to uint16.
    skip_empty_below : float or None
        Calibrated signal cutoff for skipping empty channels; None disables
        skipping.
    skip_empty_min_signal_fraction : float
        Minimum fraction of channel-volume pixels required above the cutoff.
    max_projection : bool
        Write per-tile maximum-Z images; False also disables their fused mosaic.
    flatfield_correction : bool
        Estimate or reuse illumination fields and divide calibrated data by them.
    create_fused_max_projection : bool
        Fuse per-tile maximum-Z images using nominal stage placements.
    write_fused_max_projection_tiff : bool
        Export the stage-placed maximum projection as OME-TIFF.
    z_downsample_level : int
        Integer reduction factor along deskewed Z.
    crop_after_deskew : bool
        Crop deskewed Y to the region with support across the complete Z range.
    time_range : tuple[int, int] or None
        Start-inclusive, stop-exclusive time selection; None processes all times.
    pos_range : tuple[int, int] or None
        Start-inclusive, stop-exclusive position selection; None processes all
        positions.
    decon_crop_scan : int or None
        Retained scan-plane chunk size for 3D deconvolution; None calibrates it
        automatically.
    decon_gpu_id : int
        Zero-based CUDA device index used by the deconvolution solver.
    decon_verbose : int
        Solver diagnostic verbosity, with zero suppressing iteration reports.
    decon_fallback_step_scan : int or None
        Scan planes removed from a chunk after GPU allocation failure; None uses the
        solver default.
    decon_psf_paths : list[Path] or None
        Channel-ordered PSF files; None generates theoretical PSFs. Planar
        processing uses a central Z slice of 3D PSFs.
    output_dir : Path or None
        Directory for processed images and state; None uses the source directory.
    illumination_path : Path or None
        Supplied CYX illumination TIFF; None uses the offline flatfield setting.
    live_manifest : LiveManifest or None
        Acquisition plan enabling live tile discovery; None processes offline.
    live_log_path : Path or None
        Acquisition lifecycle log used to detect live completion.
    roi_selection : PhysicalRoi or None
        Registered physical ROI selecting and cropping source tiles; None retains
        full tiles.
    resume : bool
        Reuse compatible output checkpoints; False overwrites this output run.
    decon_scan_upsample : int or None
        Experimental integer scan upsampling factor; None uses the standard solver.
        Requires 3D deconvolution and a full GPU volume.

    Returns
    -------
    None
        Deskewed tiles, optional maximum projections, and checkpoints are written to
        disk.
    """
    # Fusion consumes the per-position maximum-projection datastore.  Treat
    # --no-max-projection as disabling that dependent output as well instead of
    # trying to open a datastore that was intentionally never created.
    create_fused_max_projection = bool(max_projection and create_fused_max_projection)
    skip_empty_below, skip_empty_min_signal_fraction = validate_empty_tile_options(
        skip_empty_below,
        skip_empty_min_signal_fraction,
    )
    if deconvolve and decon_scan_upsample is None:
        from opm_processing.imageprocessing.rlgc import (
            RlgcChunkState,
            chunked_rlgc,
        )

        decon_chunk_state = RlgcChunkState(decon_crop_scan)
    elif deconvolve:
        from opm_processing.imageprocessing.rlgc_undersampled import rlgc_undersampled

    max_z_factors_yx = (2, 4, 8, 16, 32)
    output_dir = (
        root_path.parent
        if output_dir is None
        else Path(output_dir).expanduser().resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    datastore = open_acquisition_datastore(acquisition)
    opm_mode = acquisition.mode
    scan_axis_step_um = acquisition.scan_axis_step_um
    reconstruction_step_um = scan_axis_step_um / (decon_scan_upsample or 1)
    if decon_scan_upsample is not None:
        print(
            f"Experimental undersampled RLGC: factor={decon_scan_upsample}, "
            f"acquired step={scan_axis_step_um:g} um, "
            f"reconstruction step={reconstruction_step_um:g} um (full GPU volume)"
        )
    excess_scan_positions = (
        acquisition.excess_scan_start_positions or acquisition.excess_scan_positions
    )
    flyback_crop = acquisition.excess_scan_end_positions or None
    scan_axis_reversed = acquisition.scan_axis_reversed
    stage_axis_flips = acquisition.stage_axis_flips_xyz
    pixel_size_um = acquisition.pixel_size_um
    opm_tilt_deg = acquisition.angle_deg
    camera_offset = acquisition.camera_offset
    camera_conversion = acquisition.camera_conversion
    channels = list(acquisition.channel_names)
    stage_positions_raw = np.asarray(acquisition.stage_positions_zxy, dtype=float)
    pixel_size_um = float(pixel_size_um)
    opm_tilt_deg = float(opm_tilt_deg)
    camera_offset = float(camera_offset)
    camera_conversion = float(camera_conversion)
    apply_qi2lab_stage_scan_gain = (
        "stage" in str(opm_mode).casefold()
        and int(datastore.shape[-1]) == QI2LAB_STAGE_SCAN_DETECTOR_WIDTH
    )
    output_dtype = np.dtype(np.float32 if save_float32 else np.uint16)
    psfs: list[np.ndarray] | None = None
    if deconvolve:
        if decon_psf_paths is not None:
            if len(decon_psf_paths) != len(channels):
                raise ValueError("decon_psf_paths must contain one PSF per channel")
            psfs = [
                (
                    np.load(path)
                    if Path(path).suffix == ".npy"
                    else np.asarray(imread(path))
                )
                for path in tqdm(decon_psf_paths, desc="PSFs", unit="PSF")
            ]
        else:
            psfs = [
                generate_skewed_psf(
                    em_wvl=max(
                        float(wavelength.strip().lower().removesuffix("nm"))
                        for wavelength in channel.split("+")
                    )
                    / 1000,
                    pixel_size_um=pixel_size_um,
                    scan_axis_step_um=reconstruction_step_um,
                    theta_deg=opm_tilt_deg,
                    pz=0.0,
                    plot=False,
                )
                for channel in tqdm(channels, desc="PSFs", unit="PSF")
            ]
    stage_positions = apply_stage_axis_flips(stage_positions_raw, stage_axis_flips)
    stage_z_indices = stage_z_level_indices(stage_positions_raw)
    stage_z_flipped = stage_axis_flips[2]

    # Estimate the shape of one deskewed volume.
    deskew_input_shape = (
        datastore.shape[-3]
        - excess_scan_positions
        - (flyback_crop if flyback_crop is not None else 0),
        datastore.shape[-2],
        datastore.shape[-1],
    )
    if decon_scan_upsample is not None:
        deskew_input_shape = (
            (deskew_input_shape[0] - 1) * decon_scan_upsample + 1,
            *deskew_input_shape[1:],
        )
    deskewed_shape, _pad_y, _pad_x, crop_y = deskew_shape_estimator(
        deskew_input_shape,
        theta=opm_tilt_deg,
        distance=reconstruction_step_um,
        pixel_size=pixel_size_um,
        crop_after_deskew=crop_after_deskew,
    )

    roi_position_indices = (
        None if roi_selection is None else roi_selection.position_indices
    )
    roi_halo_scan = 0 if psfs is None else max(int(psf.shape[0]) // 2 for psf in psfs)
    roi_halo_x = 0 if psfs is None else max(int(psf.shape[-1]) // 2 for psf in psfs)

    if time_range is not None:
        time_shape = time_range[1]
    else:
        time_shape = datastore.shape[0]

    if pos_range is not None:
        pos_shape = pos_range[1]
    else:
        pos_shape = datastore.shape[1]

    roi_plans: tuple[dict[str, Any], ...] = ()
    roi_plan_lookup: dict[tuple[int, int], tuple[int, dict[str, Any]]] = {}
    if roi_selection is not None:
        plan_times = tuple(
            range(time_range[0], time_range[1])
            if time_range is not None
            else range(int(datastore.shape[0]))
        )
        plan_positions = tuple(
            index
            for index in (
                range(pos_range[0], pos_range[1])
                if pos_range is not None
                else range(int(datastore.shape[1]))
            )
            if roi_position_indices is not None and index in roi_position_indices
        )
        roi_plans = plan_roi_tiles(
            roi_selection,
            plan_times,
            plan_positions,
            tuple(int(value) for value in deskew_input_shape),
            channel_count=int(datastore.shape[2]),
            pixel_size_um=pixel_size_um,
            scan_step_um=scan_axis_step_um,
            angle_deg=opm_tilt_deg,
            z_downsample_level=z_downsample_level,
            halo_scan=roi_halo_scan,
            halo_x=roi_halo_x,
        )
        roi_plan_lookup = {
            (int(plan["time_index"]), int(plan["position_index"])): (index, plan)
            for index, plan in enumerate(roi_plans)
        }

    datastore_shape = [
        time_shape,
        pos_shape,
        datastore.shape[2],
        deskewed_shape[0] // z_downsample_level,
        deskewed_shape[1],
        deskewed_shape[2],
    ]
    deskewed: np.ndarray | None = None

    if illumination_path is not None:
        flatfield_path = Path(illumination_path).expanduser().resolve()
    else:
        flatfield_path = output_dir / f"{acquisition_stem(root_path)}_flatfield.ome.tif"
        source_flatfield = root_path.parent / flatfield_path.name
        if not flatfield_path.exists() and source_flatfield.exists():
            flatfield_path = source_flatfield
    provided_flatfields = (
        np.asarray(imread(flatfield_path), dtype=np.float32).reshape(
            datastore.shape[2], datastore.shape[-2], datastore.shape[-1]
        )
        if illumination_path is not None
        else None
    )
    signal_mask = (
        None
        if (
            live_manifest is not None
            or not flatfield_correction
            or provided_flatfields is not None
            or flatfield_path.exists()
        )
        else build_illumination_signal_decisions(
            datastore,
            stage_z_indices,
            camera_offset,
            camera_conversion,
            skip_empty_below,
            skip_empty_min_signal_fraction,
            apply_stage_scan_gain=apply_qi2lab_stage_scan_gain,
        )
    )
    output_kind = "deconvolved_deskewed" if deconvolve else "deskewed"
    deskewed_voxel_size_um = (
        round_spatial(z_downsample_level * pixel_size_um),
        round_spatial(pixel_size_um),
        round_spatial(pixel_size_um),
    )
    processing_configuration = {
        "kind": output_kind,
        "dtype": output_dtype.name,
        "flatfield": bool(flatfield_correction),
        "selection": {
            "time": time_range,
            "position": pos_range,
            "roi_yx_um": (
                None if roi_selection is None else roi_selection.bounds_yx_um
            ),
        },
        "empty": (
            None
            if skip_empty_below is None
            else (skip_empty_below, skip_empty_min_signal_fraction)
        ),
        "z_downsample": int(z_downsample_level),
        "crop_after_deskew": bool(crop_after_deskew),
        "deconvolution": (
            None
            if not deconvolve
            else {
                "model_revision": "opm-physical-rlgc-20261008",
                "crop_scan": decon_crop_scan,
                "fallback_step_scan": decon_fallback_step_scan,
                "psf_sha256": (
                    None
                    if decon_psf_paths is None
                    else tuple(file_sha256(path) for path in decon_psf_paths)
                ),
            }
        ),
    }
    if decon_scan_upsample is not None:
        processing_configuration["deconvolution"].update(
            backend="rlgc_undersampled",
            scan_upsample_factor=int(decon_scan_upsample),
        )
        processing_configuration["reconstruction"] = {
            "shape_syx": tuple(int(size) for size in deskew_input_shape),
            "scan_axis_step_um": float(reconstruction_step_um),
        }
    roi_series = tuple(
        {
            key: plan[key]
            for key in (
                "time_index",
                "position_index",
                "origin_zyx_um",
                "skewed_bounds_sx",
                "skewed_shape_syx",
                "deskew_crop_yx",
            )
        }
        for plan in roi_plans
    )

    if not (deconvolve):
        output_path = output_dir / Path(
            acquisition_stem(root_path) + "_deskewed.ome.zarr"
        )
    else:
        output_path = output_dir / Path(
            acquisition_stem(root_path) + "_decon_deskewed.ome.zarr"
        )
    state_resume = bool(resume or live_manifest is not None)
    output_preexisting = output_path.exists()
    output_shape = tuple(int(value) for value in datastore_shape)
    roi_output_shapes = [
        tuple(int(value) for value in plan["shape_tczyx"]) for plan in roi_plans
    ]
    if roi_selection is not None and resume and output_path.exists():
        output_collection = open_variable_resume_collection(
            output_path,
            roi_output_shapes,
            output_dtype,
        )
    elif roi_selection is not None:
        output_collection = create_variable_position_collection(
            output_path,
            roi_output_shapes,
            deskewed_voxel_size_um,
            spatial_origins_zyx_um=[plan["origin_zyx_um"] for plan in roi_plans],
            channels=channels,
            dtype=output_dtype,
            overwrite=True,
        )
    elif (live_manifest is not None or resume) and output_path.exists():
        output_collection = open_resume_collection(
            output_path,
            output_shape,
            output_dtype,
        )
    else:
        output_collection = create_position_collection(
            output_path,
            output_shape,
            deskewed_voxel_size_um,
            stage_positions=stage_positions[:pos_shape],
            channels=channels,
            dtype=output_dtype,
            overwrite=True,
        )
    ts_store = output_collection.arrays

    if max_projection:
        max_z_datastore_shape = [
            time_shape,
            pos_shape,
            datastore.shape[2],
            1,
            deskewed_shape[1],
            deskewed_shape[2],
        ]

        # create array to hold one maximum projection deskewed volume
        max_z_deskewed = np.zeros(
            (1, deskewed_shape[1], deskewed_shape[2]), dtype=np.float32
        )

        if not (deconvolve):
            max_z_output_path = output_dir / Path(
                acquisition_stem(root_path) + "_max_z_deskewed.ome.zarr"
            )
        else:
            max_z_output_path = output_dir / Path(
                acquisition_stem(root_path) + "_max_z_decon_deskewed.ome.zarr"
            )
        max_z_output_preexisting = max_z_output_path.exists()
        max_z_voxel_size_um = deskewed_voxel_size_um
        max_z_offset_um = (
            round_spatial(0.5 * (datastore_shape[3] - 1) * max_z_voxel_size_um[0]),
            0.0,
            0.0,
        )
        max_z_output_shape = tuple(int(value) for value in max_z_datastore_shape)
        if (live_manifest is not None or resume) and max_z_output_path.exists():
            max_z_collection = open_resume_collection(
                max_z_output_path,
                max_z_output_shape,
                output_dtype,
                tuple(
                    factor
                    for factor in (1, *max_z_factors_yx)
                    if factor <= min(max_z_output_shape[-2:])
                ),
            )
        else:
            max_z_collection = create_position_collection(
                max_z_output_path,
                max_z_output_shape,
                max_z_voxel_size_um,
                stage_positions=stage_positions[:pos_shape],
                channels=channels,
                dtype=output_dtype,
                multiscale_factors_yx=max_z_factors_yx,
                spatial_offset_um=max_z_offset_um,
                overwrite=True,
            )
        max_z_ts_store = max_z_collection.arrays

    if flatfield_correction:
        if provided_flatfields is not None:
            flatfields = provided_flatfields[np.newaxis, ...]
        else:
            flatfields = load_or_estimate_flatfield(
                flatfield_path,
                datastore,
                camera_offset,
                camera_conversion,
                pixel_size_um,
                stage_positions_raw,
                apply_stage_scan_gain=apply_qi2lab_stage_scan_gain,
                signal_threshold=skip_empty_below,
                minimum_signal_fraction=skip_empty_min_signal_fraction,
                signal_mask=signal_mask,
            )
    else:
        flatfields = np.ones(
            (
                int(stage_z_indices.max()) + 1,
                datastore.shape[2],
                datastore.shape[-2],
                datastore.shape[-1],
            ),
            dtype=np.float32,
        )

    processing_configuration["illumination_sha256"] = (
        file_sha256(flatfield_path) if flatfield_correction else None
    )
    processing_state = initialize_processing_state(
        output_dir=output_dir,
        source_path=root_path,
        output_path=output_path,
        configuration=processing_configuration,
        roi_series=roi_series,
        resume=state_resume,
        output_preexisting=output_preexisting,
    )
    if max_projection:
        if state_resume and max_z_output_preexisting:
            try:
                processing_state.run(max_z_output_path)
            except ValueError as error:
                raise ValueError(
                    "Existing maximum-projection output has no current processing "
                    f"run and cannot be resumed: {max_z_output_path}"
                ) from error
        processing_state.initialize_run(
            max_z_output_path,
            configuration={
                **processing_configuration,
                "kind": f"max_z_{output_kind}",
                "multiscale_factors_yx": max_z_factors_yx,
            },
            overwrite=not state_resume,
        )

    # A tile is checkpointed only after every requested output write resolves.
    if live_manifest is not None:
        readiness = ZarrTileReadiness(acquisition)
        completed_live_tiles = processing_state.completed_tiles(output_path)
        if max_projection:
            completed_live_tiles &= processing_state.completed_tiles(max_z_output_path)
        live_timepoints = int(live_manifest.index_sizes["t"])
        live_positions = int(live_manifest.index_sizes["p"])
        live_tile_count = live_timepoints * live_positions
        if completed_live_tiles:
            print(
                "Resuming live processing with "
                f"{len(completed_live_tiles)} of {live_tile_count} tiles complete."
            )
        tile_groups = progress_tile_groups(
            iter_live_tiles(
                readiness,
                live_manifest,
                live_log_path,
                completed_tiles=completed_live_tiles,
            ),
            (
                (time_index, position_index)
                for time_index in range(live_timepoints)
                for position_index in range(live_positions)
            ),
            completed_live_tiles,
            live_manifest.acquisition_order,
        )
    else:
        completed_tiles = processing_state.completed_tiles(output_path)
        if max_projection:
            completed_tiles &= processing_state.completed_tiles(max_z_output_path)
        time_indices = (
            tuple(range(time_range[0], time_range[1]))
            if time_range is not None
            else tuple(range(datastore.shape[0]))
        )
        position_indices = (
            tuple(range(pos_range[0], pos_range[1]))
            if pos_range is not None
            else tuple(range(datastore.shape[1]))
        )
        if roi_position_indices is not None:
            position_indices = tuple(
                index for index in position_indices if index in roi_position_indices
            )
        requested_tiles = tuple(
            (int(t_idx), int(pos_idx))
            for t_idx in time_indices
            for pos_idx in position_indices
        )
        completed_requested_tiles = set(requested_tiles) & completed_tiles
        if resume and completed_requested_tiles:
            print(
                "Resuming processing with "
                f"{len(completed_requested_tiles)} of {len(requested_tiles)} "
                "tiles complete."
            )
        remaining_tiles = (
            tile for tile in requested_tiles if tile not in completed_requested_tiles
        )
        tile_groups = progress_tile_groups(
            remaining_tiles,
            requested_tiles,
            completed_requested_tiles,
            acquisition.acquisition_order,
        )

    wrote_tile = False
    zero_channels = processing_state.zero_channels(output_path)
    completed_channels = (
        processing_state.completed_channels(output_path)
        if roi_selection is not None
        else set()
    )
    for t_idx, current_positions in tile_groups:
        for pos_idx in current_positions:
            ts_writes = []
            ts_max_writes = []
            skewed_roi = None
            roi_series_index = None
            roi_plan = None
            if roi_selection is not None:
                plan_entry = roi_plan_lookup.get((int(t_idx), int(pos_idx)))
                if plan_entry is None:
                    continue
                roi_series_index, roi_plan = plan_entry
                skewed_roi = roi_plan["skewed_bounds"]
            wrote_tile = True
            tile_completed_channels = {
                channel
                for time, position, channel in completed_channels
                if time == int(t_idx) and position == int(pos_idx)
            }
            if tile_completed_channels:
                print(
                    f"Resuming time {t_idx}, position {pos_idx}: "
                    f"{len(tile_completed_channels)} of {datastore.shape[2]} "
                    "channels complete."
                )
            for chan_idx in tqdm(
                (
                    channel
                    for channel in range(datastore.shape[2])
                    if channel not in tile_completed_channels
                ),
                total=datastore.shape[2],
                initial=len(tile_completed_channels),
                desc="c",
                leave=False,
            ):
                channel_key = (int(t_idx), int(pos_idx), int(chan_idx))
                if (
                    signal_mask is not None
                    and signal_mask[t_idx, pos_idx, chan_idx] == 0
                ):
                    zero_channels.add((int(t_idx), int(pos_idx), int(chan_idx)))
                    zero_value = output_dtype.type(0)
                    if roi_series_index is None:
                        ts_writes.append(
                            ts_store[pos_idx][t_idx, chan_idx].write(zero_value)
                        )
                    else:
                        write_checkpointed_roi_channel(
                            ts_store[roi_series_index][0, chan_idx],
                            zero_value,
                            processing_state,
                            output_path,
                            channel_key,
                            is_zero=True,
                        )
                    if max_projection:
                        ts_max_writes.extend(
                            queue_position_pyramid_writes(
                                max_z_collection,
                                pos_idx,
                                t_idx,
                                chan_idx,
                                zero_value,
                            )
                        )
                    continue
                if skewed_roi is None:
                    raw_selection = datastore[t_idx, pos_idx, chan_idx, :]
                    illumination = flatfields[stage_z_indices[pos_idx], chan_idx, :]
                    detector_x_offset = 0
                else:
                    useful_start = int(skewed_roi.scan_start)
                    useful_stop = int(skewed_roi.scan_stop)
                    if scan_axis_reversed:
                        raw_start = int(datastore.shape[-3]) - (
                            excess_scan_positions + useful_stop
                        )
                        raw_stop = int(datastore.shape[-3]) - (
                            excess_scan_positions + useful_start
                        )
                    else:
                        raw_start = excess_scan_positions + useful_start
                        raw_stop = excess_scan_positions + useful_stop
                    raw_selection = datastore[
                        t_idx,
                        pos_idx,
                        chan_idx,
                        raw_start:raw_stop,
                        :,
                        skewed_roi.x_start : skewed_roi.x_stop,
                    ]
                    illumination = flatfields[
                        stage_z_indices[pos_idx],
                        chan_idx,
                        :,
                        skewed_roi.x_start : skewed_roi.x_stop,
                    ]
                    detector_x_offset = int(skewed_roi.x_start)
                raw_data = np.asarray(raw_selection.read().result())
                if skewed_roi is not None:
                    raw_data = raw_data.reshape(
                        skewed_roi.scan_stop - skewed_roi.scan_start,
                        int(deskew_input_shape[1]),
                        skewed_roi.x_stop - skewed_roi.x_start,
                    )
                else:
                    raw_data = np.squeeze(raw_data)
                camera_calibrated_data = camera_correct(
                    raw_data,
                    camera_offset,
                    camera_conversion,
                    detector_x_offset=detector_x_offset,
                    apply_stage_scan_gain=apply_qi2lab_stage_scan_gain,
                )
                if skewed_roi is not None and scan_axis_reversed:
                    camera_calibrated_data = np.flip(camera_calibrated_data, axis=0)
                if tile_is_empty(
                    signal_mask,
                    t_idx,
                    pos_idx,
                    chan_idx,
                    camera_calibrated_data,
                    skip_empty_below,
                    skip_empty_min_signal_fraction,
                ):
                    zero_channels.add((int(t_idx), int(pos_idx), int(chan_idx)))
                    zero_value = output_dtype.type(0)
                    if roi_series_index is None:
                        ts_writes.append(
                            ts_store[pos_idx][t_idx, chan_idx].write(zero_value)
                        )
                    else:
                        write_checkpointed_roi_channel(
                            ts_store[roi_series_index][0, chan_idx],
                            zero_value,
                            processing_state,
                            output_path,
                            channel_key,
                            is_zero=True,
                        )
                    if max_projection:
                        ts_max_writes.extend(
                            queue_position_pyramid_writes(
                                max_z_collection,
                                pos_idx,
                                t_idx,
                                chan_idx,
                                zero_value,
                            )
                        )
                    continue
                camera_corrected_data = illumination_correct(
                    camera_calibrated_data,
                    illumination,
                )
                if skewed_roi is None and scan_axis_reversed:
                    camera_corrected_data = np.flip(camera_corrected_data, axis=0)

                if deconvolve:
                    if skewed_roi is not None:
                        decon_input = camera_corrected_data
                    elif flyback_crop is not None:
                        decon_input = camera_corrected_data[
                            excess_scan_positions:-flyback_crop, :, :
                        ]
                    else:
                        decon_input = camera_corrected_data[
                            excess_scan_positions:, :, :
                        ]
                    if decon_scan_upsample is not None:
                        deconvolved_data = rlgc_undersampled(
                            decon_input,
                            np.asarray(psfs[chan_idx]),
                            scan_upsample_factor=decon_scan_upsample,
                            gpu_id=decon_gpu_id,
                        )
                    else:
                        effective_crop_scan = decon_chunk_state.determine_once(
                            tuple(int(size) for size in decon_input.shape),
                            [tuple(int(size) for size in psf.shape) for psf in psfs],
                            gpu_id=decon_gpu_id,
                        )
                        deconvolved_data = chunked_rlgc(
                            decon_input,
                            np.asarray(psfs[chan_idx]),
                            crop_scan=effective_crop_scan,
                            gpu_id=decon_gpu_id,
                            verbose=decon_verbose,
                            fallback_step_scan=decon_fallback_step_scan,
                            on_successful_crop_scan=(
                                decon_chunk_state.remember_successful_crop
                            ),
                        )
                    deskewed = orthogonal_deskew(
                        deconvolved_data,
                        theta=opm_tilt_deg,
                        distance=reconstruction_step_um,
                        pixel_size=pixel_size_um,
                        downsample_factor=z_downsample_level,
                    )
                else:
                    if skewed_roi is not None:
                        deskewed = orthogonal_deskew(
                            camera_corrected_data,
                            theta=opm_tilt_deg,
                            distance=scan_axis_step_um,
                            pixel_size=pixel_size_um,
                            downsample_factor=z_downsample_level,
                        )
                    elif flyback_crop is not None:
                        deskewed = orthogonal_deskew(
                            camera_corrected_data[
                                excess_scan_positions:-flyback_crop, :, :
                            ],
                            theta=opm_tilt_deg,
                            distance=scan_axis_step_um,
                            pixel_size=pixel_size_um,
                            downsample_factor=z_downsample_level,
                        )
                    else:
                        deskewed = orthogonal_deskew(
                            camera_corrected_data[excess_scan_positions:, :, :],
                            theta=opm_tilt_deg,
                            distance=scan_axis_step_um,
                            pixel_size=pixel_size_um,
                            downsample_factor=z_downsample_level,
                        )

                if roi_plan is not None:
                    crop_y0, crop_y1, crop_x0, crop_x1 = (
                        int(value) for value in roi_plan["deskew_crop_yx"]
                    )
                    deskewed = deskewed[
                        :,
                        crop_y0:crop_y1,
                        crop_x0:crop_x1,
                    ]
                elif crop_after_deskew:
                    deskewed = deskewed[:, crop_y:-crop_y, :]

                if max_projection:
                    max_z_deskewed = np.max(deskewed, axis=0, keepdims=True)
                    # create future objects for async data writing
                    ts_max_writes.extend(
                        queue_position_pyramid_writes(
                            max_z_collection,
                            pos_idx,
                            t_idx,
                            chan_idx,
                            format_processed_output(
                                max_z_deskewed,
                                save_float32,
                            ),
                        )
                    )

                # create future objects for async data writing
                formatted = format_processed_output(deskewed, save_float32)
                if not np.any(formatted):
                    zero_channels.add((int(t_idx), int(pos_idx), int(chan_idx)))
                if roi_series_index is None:
                    ts_writes.append(
                        ts_store[pos_idx][t_idx, chan_idx].write(formatted)
                    )
                else:
                    write_checkpointed_roi_channel(
                        ts_store[roi_series_index][0, chan_idx],
                        formatted,
                        processing_state,
                        output_path,
                        channel_key,
                        is_zero=channel_key in zero_channels,
                    )
            for ts_write in ts_writes:
                ts_write.result()
            if max_projection:
                for ts_max_write in ts_max_writes:
                    ts_max_write.result()
            tile_zero_channels = (
                channel_index
                for time_index, position_index, channel_index in zero_channels
                if time_index == int(t_idx) and position_index == int(pos_idx)
            )
            processing_state.complete_tile(
                output_path,
                t_idx,
                pos_idx,
                zero_channels=tile_zero_channels,
            )
            if max_projection:
                processing_state.complete_tile(max_z_output_path, t_idx, pos_idx)

    if wrote_tile and deskewed is not None:
        del deskewed
        if max_projection:
            del max_z_deskewed
    del ts_store
    if create_fused_max_projection:
        if deconvolve:
            max_z_output_path = output_dir / Path(
                acquisition_stem(root_path) + "_max_z_decon_deskewed.ome.zarr"
            )
        else:
            max_z_output_path = output_dir / Path(
                acquisition_stem(root_path) + "_max_z_deskewed.ome.zarr"
            )
        max_z_collection = open_position_collection(max_z_output_path)
        max_z_ts_store = max_z_collection.arrays

        print("\nFusing max projection using stage positions...")
        fused_output_path = output_dir / Path(
            acquisition_stem(root_path) + "_max_z_fused.ome.zarr"
        )

        if pos_range is not None:
            tile_positions = stage_positions[pos_range[0] : pos_range[1]]

        else:
            tile_positions = stage_positions

        tile_fusion = MaxTileFusion(
            ts_dataset=max_z_ts_store,
            tile_positions=tile_positions,
            output_path=fused_output_path,
            pixel_size=np.asarray(
                max_z_collection.voxel_size_um,
                dtype=np.float64,
            ),
            spatial_offset_z_um=float(max_z_collection.spatial_origins_zyx_um[0][0]),
            reverse_stage_z=not stage_z_flipped,
            opm_angle_deg=opm_tilt_deg,
        )
        tile_fusion.run()

        if write_fused_max_projection_tiff:
            tiff_dir_path = max_z_output_path.parent / Path(
                "fused_max_projection_tiff_output"
            )
            tiff_dir_path.mkdir(exist_ok=True)
            max_proj_datastore = open_image_array(fused_output_path)
            for t_idx in tqdm(range(max_proj_datastore.shape[0]), desc="t"):
                max_projection = np.squeeze(
                    np.asarray(max_proj_datastore[t_idx].read().result())
                )

                filename = Path(f"fused_z_max_projection_t{t_idx}.ome.tiff")
                filename_path = tiff_dir_path / Path(filename)
                if len(max_projection.shape) == 2:
                    axes = "YX"
                else:
                    axes = "CYX"

                with TiffWriter(filename_path, bigtiff=True) as tif:
                    metadata = {
                        "axes": axes,
                        "SignificantBits": 32 if save_float32 else 16,
                        "PhysicalSizeX": pixel_size_um,
                        "PhysicalSizeXUnit": "µm",
                        "PhysicalSizeY": pixel_size_um,
                        "PhysicalSizeYUnit": "µm",
                    }
                    options = dict(
                        compression="zlib",
                        compressionargs={"level": 8},
                        predictor=True,
                        photometric="minisblack",
                        resolutionunit="CENTIMETER",
                    )
                    tif.write(
                        max_projection,
                        resolution=(1e4 / pixel_size_um, 1e4 / pixel_size_um),
                        **options,
                        metadata=metadata,
                    )


def process_projection(
    root_path: Path,
    acquisition: AcquisitionMetadata,
    deconvolve: bool = True,
    save_float32: bool = False,
    skip_empty_below: float | None = None,
    skip_empty_min_signal_fraction: float = 0.01,
    flatfield_correction: bool = True,
    write_fused_max_projection_tiff: bool = True,
    time_range: tuple[int, int] = None,
    pos_range: tuple[int, int] = None,
    eager_deconvolution: bool = False,
    resume: bool = False,
    decon_gpu_id: int = 0,
    decon_verbose: int = 1,
    decon_psf_paths: list[Path] | None = None,
    output_dir: Path | None = None,
):
    """Calibrate and optionally deconvolve planar acquisition tiles.

    Parameters
    ----------
    root_path : pathlib.Path
        Resolved path to the raw acquisition store.
    acquisition : AcquisitionMetadata
        Inspected scan geometry, channels, stage placement, and camera calibration.
    deconvolve : bool
        Apply Richardson–Lucy gradient-consensus deconvolution before saving.
    save_float32 : bool
        Preserve float32 intensities; False clips and casts final output to uint16.
    skip_empty_below : float or None
        Calibrated signal cutoff for skipping empty channels; None disables
        skipping.
    skip_empty_min_signal_fraction : float
        Minimum fraction of channel-volume pixels required above the cutoff.
    flatfield_correction : bool
        Estimate or reuse illumination fields and divide calibrated data by them.
    write_fused_max_projection_tiff : bool
        Export the stage-placed maximum projection as OME-TIFF.
    time_range : tuple[int, int] or None
        Start-inclusive, stop-exclusive time selection; None processes all times.
    pos_range : tuple[int, int] or None
        Start-inclusive, stop-exclusive position selection; None processes all
        positions.
    eager_deconvolution : bool
        Use eager stopping in the planar gradient-consensus solver.
    resume : bool
        Reuse compatible output checkpoints; False overwrites this output run.
    decon_gpu_id : int
        Zero-based CUDA device index used by the deconvolution solver.
    decon_verbose : int
        Solver diagnostic verbosity, with zero suppressing iteration reports.
    decon_psf_paths : list[Path] or None
        Channel-ordered PSF files; None generates theoretical PSFs. Planar
        processing uses a central Z slice of 3D PSFs.
    output_dir : Path or None
        Directory for processed images and state; None uses the source directory.

    Returns
    -------
    None
        Processed planar tiles, stage-placed mosaics, and checkpoints are written to
        disk.
    """
    overwrite = not resume
    if deconvolve:
        from opm_processing.imageprocessing.rlgc import rlgc_2d

    skip_empty_below, skip_empty_min_signal_fraction = validate_empty_tile_options(
        skip_empty_below,
        skip_empty_min_signal_fraction,
    )

    output_dir = (
        root_path.parent
        if output_dir is None
        else Path(output_dir).expanduser().resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    datastore = open_acquisition_datastore(acquisition)
    opm_mode = acquisition.mode
    pixel_size_um = acquisition.pixel_size_um
    opm_tilt_deg = acquisition.angle_deg
    camera_offset = acquisition.camera_offset
    camera_conversion = acquisition.camera_conversion
    channels = list(acquisition.channel_names)
    stage_positions_raw = np.asarray(acquisition.stage_positions_zxy, dtype=float)
    stage_axis_flips = acquisition.stage_axis_flips_xyz
    pixel_size_um = float(pixel_size_um)
    opm_tilt_deg = float(opm_tilt_deg)
    camera_offset = float(camera_offset)
    camera_conversion = float(camera_conversion)
    output_dtype = np.dtype(np.float32 if save_float32 else np.uint16)
    psfs: list[np.ndarray] | None = None
    if deconvolve:
        if decon_psf_paths is not None:
            if len(decon_psf_paths) != len(channels):
                raise ValueError("decon_psf_paths must contain one PSF per channel")
            psfs = [
                (
                    np.load(path)
                    if Path(path).suffix == ".npy"
                    else np.asarray(imread(path))
                )
                for path in tqdm(decon_psf_paths, desc="PSFs", unit="PSF")
            ]
        else:
            psfs = [
                generate_proj_psf(
                    em_wvl=max(
                        float(wavelength.strip().lower().removesuffix("nm"))
                        for wavelength in channel.split("+")
                    )
                    / 1000,
                    pixel_size_um=pixel_size_um,
                )
                for channel in tqdm(channels, desc="PSFs", unit="PSF")
            ]
    stage_positions = apply_stage_axis_flips(stage_positions_raw, stage_axis_flips)
    stage_z_indices = stage_z_level_indices(stage_positions_raw)
    stage_z_flipped = stage_axis_flips[2]

    if time_range is not None:
        time_shape = time_range[1]
    else:
        time_shape = datastore.shape[0]

    if pos_range is not None:
        pos_shape = pos_range[1]
    else:
        pos_shape = datastore.shape[1]

    if datastore.rank == 5:
        datastore = datastore[:, :, :, None, :, :]

    filename_label = "projection" if "projection" in opm_mode.casefold() else "2d"
    output_kind = filename_label
    if deconvolve:
        output_kind = f"deconvolved_{filename_label}"
        filename_label = f"decon_{filename_label}"
    output_path = output_dir / Path(
        f"{acquisition_stem(root_path)}_{filename_label}.ome.zarr"
    )
    output_preexisting = output_path.exists()
    fused_output_path: Path | None = None
    signal_mask: np.ndarray | None = None
    flatfield_path = output_dir / f"{acquisition_stem(root_path)}_flatfield.ome.tif"
    source_flatfield = root_path.parent / flatfield_path.name
    if not flatfield_path.exists() and source_flatfield.exists():
        flatfield_path = source_flatfield
    signal_mask = (
        None
        if not flatfield_correction or flatfield_path.exists()
        else build_illumination_signal_decisions(
            datastore,
            stage_z_indices,
            camera_offset,
            camera_conversion,
            skip_empty_below,
            skip_empty_min_signal_fraction,
            apply_stage_scan_gain=False,
        )
    )
    voxel_size_um = (1.0, pixel_size_um, pixel_size_um)
    processing_configuration = {
        "kind": output_kind,
        "dtype": output_dtype.name,
        "flatfield": bool(flatfield_correction),
        "selection": {"time": time_range, "position": pos_range},
        "empty": (
            None
            if skip_empty_below is None
            else (skip_empty_below, skip_empty_min_signal_fraction)
        ),
        "deconvolution": (
            None
            if not deconvolve
            else {
                "model_revision": "opm-physical-rlgc-20261008",
                "eager": bool(eager_deconvolution),
                "psf_sha256": (
                    None
                    if decon_psf_paths is None
                    else tuple(file_sha256(path) for path in decon_psf_paths)
                ),
            }
        ),
    }
    if resume and output_path.exists():
        output_collection = open_resume_collection(
            output_path,
            tuple(int(value) for value in datastore.shape),
            output_dtype,
        )
    else:
        output_collection = create_position_collection(
            output_path,
            datastore.shape,
            voxel_size_um,
            stage_positions=stage_positions,
            channels=channels,
            overwrite=True,
            dtype=output_dtype,
        )
    ts_store = output_collection.arrays

    if flatfield_correction:
        flatfields = load_or_estimate_flatfield(
            flatfield_path,
            datastore,
            camera_offset,
            camera_conversion,
            pixel_size_um,
            stage_positions_raw,
            apply_stage_scan_gain=False,
            signal_threshold=skip_empty_below,
            minimum_signal_fraction=skip_empty_min_signal_fraction,
            signal_mask=signal_mask,
        )
    else:
        flatfields = np.ones(
            (
                int(stage_z_indices.max()) + 1,
                datastore.shape[2],
                datastore.shape[-2],
                datastore.shape[-1],
            ),
            dtype=np.float32,
        )

    processing_configuration["illumination_sha256"] = (
        file_sha256(flatfield_path) if flatfield_correction else None
    )
    processing_state = initialize_processing_state(
        output_dir=output_dir,
        source_path=root_path,
        output_path=output_path,
        configuration=processing_configuration,
        resume=resume,
        output_preexisting=output_preexisting,
    )

    completed_tiles = processing_state.completed_tiles(output_path)
    requested_times = (
        range(time_range[0], time_range[1])
        if time_range is not None
        else range(time_shape)
    )
    requested_positions = (
        range(pos_range[0], pos_range[1]) if pos_range is not None else range(pos_shape)
    )
    requested_tiles = tuple(
        (int(time_index), int(position))
        for time_index in requested_times
        for position in requested_positions
    )
    completed_requested = set(requested_tiles) & completed_tiles
    if resume and completed_requested:
        print(
            "Resuming processing with "
            f"{len(completed_requested)} of {len(requested_tiles)} tiles complete."
        )
    zero_channels = processing_state.zero_channels(output_path)
    remaining_tiles = (
        tile for tile in requested_tiles if tile not in completed_requested
    )
    for t_idx, pos_idx in tqdm(
        remaining_tiles,
        total=len(requested_tiles),
        initial=len(completed_requested),
        desc="tiles",
        unit="tile",
    ):
        tile_writes = []
        for chan_idx in tqdm(range(datastore.shape[2]), desc="c", leave=False):
            if signal_mask is not None and signal_mask[t_idx, pos_idx, chan_idx] == 0:
                zero_channels.add((int(t_idx), int(pos_idx), int(chan_idx)))
                tile_writes.append(
                    ts_store[pos_idx][t_idx, chan_idx].write(output_dtype.type(0))
                )
                continue
            raw_data = np.squeeze(
                datastore[t_idx, pos_idx, chan_idx, :].read().result()
            )
            camera_calibrated_data = camera_correct(
                raw_data,
                camera_offset,
                camera_conversion,
            )

            if tile_is_empty(
                signal_mask,
                t_idx,
                pos_idx,
                chan_idx,
                camera_calibrated_data,
                skip_empty_below,
                skip_empty_min_signal_fraction,
            ):
                zero_channels.add((int(t_idx), int(pos_idx), int(chan_idx)))
                tile_writes.append(
                    ts_store[pos_idx][t_idx, chan_idx].write(output_dtype.type(0))
                )
                continue
            camera_corrected_data = illumination_correct(
                camera_calibrated_data,
                flatfields[stage_z_indices[pos_idx], chan_idx, :],
            )

            if deconvolve:
                deconvolved_data = rlgc_2d(
                    image=camera_corrected_data,
                    skewed_psf=np.asarray(psfs[chan_idx]),
                    gpu_id=decon_gpu_id,
                    verbose=decon_verbose,
                    safe_mode=not eager_deconvolution,
                )
            else:
                deconvolved_data = camera_corrected_data

            formatted = format_processed_output(
                deconvolved_data,
                save_float32,
            )
            if not np.any(formatted):
                zero_channels.add((int(t_idx), int(pos_idx), int(chan_idx)))
            tile_writes.append(ts_store[pos_idx][t_idx, chan_idx].write(formatted))
        for tile_write in tile_writes:
            tile_write.result()
        tile_zero_channels = (
            channel_index
            for time_index, position_index, channel_index in zero_channels
            if time_index == int(t_idx) and position_index == int(pos_idx)
        )
        processing_state.complete_tile(
            output_path,
            t_idx,
            pos_idx,
            zero_channels=tile_zero_channels,
        )

    if pos_range is not None:
        tile_positions = stage_positions[pos_range[0] : pos_range[1]]
    else:
        tile_positions = stage_positions
    if len(tile_positions) > 1:
        print("\nFusing using stage positions...")
        fused_output_path = output_dir / Path(
            acquisition_stem(root_path) + "_stagefused.ome.zarr"
        )
        tile_fusion = MaxTileFusion(
            ts_dataset=ts_store,
            tile_positions=tile_positions,
            output_path=fused_output_path,
            pixel_size=np.asarray((pixel_size_um, pixel_size_um), dtype=np.float32),
            time_range=time_range,
            reverse_stage_z=not stage_z_flipped,
            opm_angle_deg=opm_tilt_deg,
        )
        tile_fusion.run()
    else:
        print("\nSkipping stage fusion: acquisition has one stage position.")
    del ts_store

    if write_fused_max_projection_tiff and fused_output_path is not None:
        tiff_dir_path = fused_output_path.parent / Path("fused_tiff_output")
        tiff_dir_path.mkdir(exist_ok=True)
        max_proj_datastore = open_image_array(fused_output_path)

        filename = Path("deconvolved_stagefused.ome.tiff")
        filename_path = tiff_dir_path / Path(filename)

        if not (filename_path.exists()) or overwrite:
            max_projection = np.squeeze(np.asarray(max_proj_datastore.read().result()))

            print(f"maxprojection dimensions: {max_projection.ndim}")
            if max_projection.ndim == 3:
                if datastore.shape[0] > 1 and datastore.shape[2] == 1:
                    axes = "TYX"
                elif datastore.shape[2] > 1 and datastore.shape[0] == 1:
                    axes = "CYX"
            elif max_projection.ndim == 4:
                axes = "TCYX"
            elif max_projection.ndim == 2:
                axes = "YX"

            with TiffWriter(filename_path, bigtiff=True) as tif:
                metadata = {
                    "axes": axes,
                    "SignificantBits": 32 if save_float32 else 16,
                    "PhysicalSizeX": pixel_size_um,
                    "PhysicalSizeXUnit": "µm",
                    "PhysicalSizeY": pixel_size_um,
                    "PhysicalSizeYUnit": "µm",
                    "PhysicalSizeZ": 1.0,
                    "PhysicalSizeZUnit": "µm",
                }
                options = dict(
                    compression="zlib",
                    compressionargs={"level": 8},
                    predictor=True,
                    photometric="minisblack",
                    resolutionunit="CENTIMETER",
                )
                tif.write(
                    max_projection,
                    resolution=(1e4 / pixel_size_um, 1e4 / pixel_size_um),
                    **options,
                    metadata=metadata,
                )


def run_estimate_illuminations(
    datastore,
    camera_offset,
    camera_conversion,
    stage_positions_zxy,
    apply_stage_scan_gain,
    signal_mask,
    conn,
):
    """Run ``estimate_illuminations`` in a subprocess.

    Parameters
    ----------
    datastore : tensorstore.TensorStore
        Readable raw TensorStore with TPCZYX axes and uint16 camera values.
    camera_offset : float
        Camera baseline in ADU, subtracted before applying the conversion.
    camera_conversion : float
        Calibrated intensity units per ADU after baseline subtraction.
    stage_positions_zxy : numpy.ndarray
        Physical stage Z, scan-axis X, and lateral Y coordinates in micrometers.
    apply_stage_scan_gain : bool
        Apply the fixed qi2lab detector-X gain during camera calibration.
    signal_mask : numpy.ndarray or None
        TPC mask selecting nonempty estimation tiles; None allows automatic
        selection.
    conn : multiprocessing.connection.Connection
        Child pipe endpoint sending either fitted fields or an exception to the
        parent.

    Returns
    -------
    None
        The result or annotated exception is sent through the pipe, which is then
        closed.

    Notes
    -----
    BaSiCPy runs in an isolated process so its GPU allocations are released
    before reconstruction begins.
    """
    try:
        from opm_processing.cuda import preload_cuda_libraries

        preload_cuda_libraries()

        from opm_processing.imageprocessing.flatfield import estimate_illuminations

        flatfields = estimate_illuminations(
            datastore,
            camera_offset,
            camera_conversion,
            stage_positions_zxy,
            apply_stage_scan_gain=apply_stage_scan_gain,
            signal_mask=signal_mask,
        )
        conn.send(flatfields)
    except Exception as e:
        e.add_note("Flatfield worker traceback:\n" + traceback.format_exc())
        conn.send(e)
    finally:
        conn.close()


def call_estimate_illuminations(
    datastore,
    camera_offset,
    camera_conversion,
    stage_positions_zxy,
    apply_stage_scan_gain,
    signal_mask,
):
    """Call ``estimate_illuminations`` in an isolated subprocess.

    Parameters
    ----------
    datastore : tensorstore.TensorStore
        Readable raw TensorStore with TPCZYX axes and uint16 camera values.
    camera_offset : float
        Camera baseline in ADU, subtracted before applying the conversion.
    camera_conversion : float
        Calibrated intensity units per ADU after baseline subtraction.
    stage_positions_zxy : numpy.ndarray
        Physical stage Z, scan-axis X, and lateral Y coordinates in micrometers.
    apply_stage_scan_gain : bool
        Apply the fixed qi2lab detector-X gain during camera calibration.
    signal_mask : numpy.ndarray or None
        TPC mask selecting nonempty estimation tiles; None allows automatic
        selection.

    Returns
    -------
    numpy.ndarray
        Estimated depth-CYX fields after the isolated worker exits successfully.

    Notes
    -----
    BaSiCPy runs in an isolated process so its GPU allocations are released
    before reconstruction begins.
    """
    context = mp.get_context(
        "forkserver" if sys.platform.startswith("linux") else "spawn"
    )
    parent_conn, child_conn = context.Pipe()
    p = context.Process(
        target=run_estimate_illuminations,
        args=(
            datastore,
            camera_offset,
            camera_conversion,
            stage_positions_zxy,
            apply_stage_scan_gain,
            signal_mask,
            child_conn,
        ),
    )
    p.start()
    child_conn.close()
    try:
        try:
            result = parent_conn.recv()
        except EOFError as error:
            p.join()
            raise RuntimeError(
                f"Flatfield estimation subprocess exited without a result "
                f"(exit code {p.exitcode})"
            ) from error
    finally:
        parent_conn.close()
        p.join()

    if p.exitcode != 0:
        raise RuntimeError("Subprocess failed")

    if isinstance(result, Exception):
        raise result

    return result


# entry for point for CLI
def main():
    """Run the OPM processing command-line application.

    Returns
    -------
    None
        The Typer application parses arguments and executes processing.
    """
    app()


if __name__ == "__main__":
    main()
