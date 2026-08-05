"""
Deskew qi2lab OPM data.

This file deskews and creates maximum projections of raw qi2lab OPM data.
"""

import multiprocessing as mp
import sys

if sys.platform.startswith("linux"):
    mp.set_start_method("forkserver", force=True)
elif sys.platform.startswith("win"):
    mp.set_start_method("spawn", force=True)

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=FutureWarning)

import hashlib
import json
import time
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import tensorstore as ts
import typer
from tifffile import TiffFile, TiffWriter, imread
from tqdm import tqdm

from opm_processing.dataio.acquisition import (
    AcquisitionMetadata,
    acquisition_stem,
    inspect_acquisition,
    open_acquisition_datastore,
)
from opm_processing.dataio.metadata import (
    extract_channels,
    extract_stage_positions,
    find_key,
)
from opm_processing.dataio.live import (
    LIVE_POLL_INTERVAL_SECONDS,
    LiveManifest,
    ZarrTileReadiness,
    iter_live_tiles,
    resolve_live_sidecars,
)
from opm_processing.dataio.position_collection import (
    create_position_collection,
    open_image_array,
    open_position_collection,
)
from opm_processing.imageprocessing.maxtilefusion import MaxTileFusion
from opm_processing.imageprocessing.opmpsf import (
    ASI_generate_skewed_psf,
    generate_proj_psf,
    generate_skewed_psf,
)
from opm_processing.imageprocessing.opmtools import (
    deskew_shape_estimator,
    orthogonal_deskew,
)

app = typer.Typer()
app.pretty_exceptions_enable = False


def _resolve_output_directory(root_path: Path, output: Path | None) -> Path:
    """Return an existing directory for all processing artifacts."""
    output_dir = (
        root_path.parent if output is None else Path(output).expanduser().resolve()
    )
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Processing output is not a directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _processed_dtype(save_float32: bool) -> np.dtype:
    """Return the requested dtype for calibrated processing products."""
    return np.dtype(np.float32 if save_float32 else np.uint16)


def _camera_corrected_image(
    raw: np.ndarray,
    camera_offset: float,
    camera_conversion: float,
    illumination: np.ndarray,
) -> np.ndarray:
    """Apply camera and illumination correction without premature quantization."""
    corrected = (
        (np.asarray(raw, dtype=np.float32) - camera_offset) * camera_conversion
    ) / np.asarray(illumination, dtype=np.float32)
    return np.maximum(corrected, 0).astype(np.float32, copy=False)


def _format_processed_output(data: np.ndarray, save_float32: bool) -> np.ndarray:
    """Convert a completed float32 processing product for persistent storage."""
    if save_float32:
        return np.asarray(data, dtype=np.float32)
    return np.clip(data, 0, np.iinfo(np.uint16).max).astype(np.uint16)


def _output_conversion_step(save_float32: bool) -> dict[str, Any]:
    """Describe the requested output conversion in processing provenance."""
    if save_float32:
        return {
            "name": "float32_output",
            "applied": True,
            "parameters": {"dtype": "float32", "quantized": False},
        }
    return {
        "name": "uint16_conversion",
        "applied": True,
        "parameters": {"clip_min": 0, "clip_max": 2**16 - 1},
    }


def _flatfield_software_tag() -> str:
    """Return the estimator identifier stored with reusable flatfields."""
    return "opm-processing/basicpy-autotuned-distributed-sampling-v7"


def _read_current_flatfield(
    path: Path,
    expected_shape: tuple[int, int, int],
) -> np.ndarray | None:
    """Read a flatfield only when it was made by the current estimator."""
    try:
        with TiffFile(path) as tif:
            tag = tif.pages[0].tags.get("Software")
            if tag is None or tag.value != _flatfield_software_tag():
                return None
            flatfields = tif.asarray().astype(np.float32)
    except (OSError, ValueError):
        return None
    if flatfields.shape != expected_shape:
        return None
    if not np.all(np.isfinite(flatfields)) or np.any(flatfields <= 0):
        return None
    return flatfields


def _write_flatfield(
    path: Path,
    flatfields: np.ndarray,
    pixel_size_um: float,
) -> None:
    """Write a reusable, versioned CYX flatfield OME-TIFF."""
    with TiffWriter(path, bigtiff=True) as tif:
        metadata = {
            "axes": "CYX",
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
            software=_flatfield_software_tag(),
            metadata=metadata,
        )


def _load_or_estimate_flatfield(
    path: Path,
    datastore,
    camera_offset: float,
    camera_conversion: float,
    pixel_size_um: float,
) -> np.ndarray:
    """Reuse a current flatfield or replace an obsolete estimator artifact."""
    expected_shape = (
        datastore.shape[2],
        datastore.shape[-2],
        datastore.shape[-1],
    )
    if path.exists():
        flatfields = _read_current_flatfield(path, expected_shape)
        if flatfields is not None:
            return flatfields
        print("Existing flatfield uses an obsolete estimator; re-estimating it.")

    flatfields = call_estimate_illuminations(
        datastore,
        camera_offset,
        camera_conversion,
    )
    _write_flatfield(path, flatfields, pixel_size_um)
    return flatfields


def _load_provided_illumination(
    path: Path,
    expected_shape: tuple[int, int, int],
) -> np.ndarray:
    """Load a user-supplied CYX illumination image without estimating one."""
    illumination_path = Path(path).expanduser().resolve()
    if not illumination_path.is_file():
        raise ValueError(f"Illumination file does not exist: {illumination_path}")
    with TiffFile(illumination_path) as tif:
        axes = tif.series[0].axes.upper()
        illumination = np.asarray(tif.asarray(), dtype=np.float32)
    if axes == "YX":
        illumination = illumination[np.newaxis, ...]
    elif axes != "CYX":
        raise ValueError(
            f"Live illumination must have CYX axes, or YX for one channel; got {axes!r}"
        )
    if tuple(illumination.shape) != expected_shape:
        raise ValueError(
            f"Live illumination shape {illumination.shape} does not match "
            f"acquisition CYX shape {expected_shape}"
        )
    if not np.all(np.isfinite(illumination)) or np.any(illumination <= 0):
        raise ValueError(
            "Live illumination values must be finite and strictly positive"
        )
    return illumination


def _file_sha256(path: Path) -> str:
    """Return a stable SHA-256 digest for a processing input artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _open_live_acquisition(
    root_path: Path,
) -> tuple[AcquisitionMetadata, LiveManifest, Path]:
    """Wait for the manifest and OME-Zarr hierarchy required by live mode."""
    root_path = Path(root_path).expanduser().resolve()
    if not root_path.name.endswith((".zarr", ".ome.zarr")):
        raise ValueError("--live requires the path of an OME-Zarr acquisition")
    sidecars = resolve_live_sidecars(root_path)
    while not sidecars.manifest.is_file():
        print(
            "Waiting "
            f"{LIVE_POLL_INTERVAL_SECONDS:g} seconds for live manifest "
            f"{sidecars.manifest}..."
        )
        time.sleep(LIVE_POLL_INTERVAL_SECONDS)
    manifest = LiveManifest.read(sidecars.manifest)
    if manifest.data_path != root_path:
        raise ValueError(
            "Live manifest data_path does not match the requested acquisition: "
            f"{manifest.data_path} != {root_path}"
        )
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
    return acquisition, manifest, sidecars.log


def _distribution_version(distribution: str) -> str:
    """Return an installed distribution version for provenance."""
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unknown"


def _processing_provenance(
    source_path: Path,
    output_kind: str,
    steps: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build namespaced, JSON-compatible raw-to-processed provenance."""
    return {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "software": {
            "name": "opm-processing-v2",
            "version": _distribution_version("opm-processing-v2"),
        },
        "metadata": {
            "writer": "yaozarrs.Bf2RawBuilder.extra_attributes",
            "yaozarrs_version": _distribution_version("yaozarrs"),
        },
        "source": {
            "path": str(source_path.resolve()),
            "format": "OME-Zarr" if source_path.is_dir() else "OME-TIFF",
        },
        "output": {"kind": output_kind},
        "steps": steps,
    }


def _derived_processing_metadata(
    metadata: dict[str, Any],
    output_kind: str,
    step: dict[str, Any],
) -> dict[str, Any]:
    """Copy processing metadata and append one derived-output operation."""
    derived = deepcopy(metadata)
    provenance = derived["opm_processing"]
    provenance["output"]["kind"] = output_kind
    provenance["steps"].append(step)
    return derived


def _selection_step(
    time_range: tuple[int, int] | None,
    position_range: tuple[int, int] | None,
) -> dict[str, Any]:
    """Describe an optional temporal/spatial subset of the source acquisition."""
    return {
        "name": "input_selection",
        "applied": time_range is not None or position_range is not None,
        "parameters": {
            "time_range": None if time_range is None else list(time_range),
            "position_range": None if position_range is None else list(position_range),
        },
    }


def _apply_stage_axis_flips(
    stage_positions: np.ndarray,
    axis_flips_xyz: tuple[bool, bool, bool],
) -> np.ndarray:
    """Apply configured camera-to-stage orientation flips to ZXY positions.

    Parameters
    ----------
    stage_positions : np.ndarray
        Value supplied for ``stage positions``.
    axis_flips_xyz : tuple[bool, bool, bool]
        Value supplied for ``axis flips xyz``.

    Returns
    -------
    np.ndarray
        Result produced by the callable.
    """
    if len(axis_flips_xyz) != 3:
        raise ValueError("axis_flips_xyz must contain X, Y, and Z flags")
    transformed = np.asarray(stage_positions, dtype=float).copy()
    for should_flip, column in zip(axis_flips_xyz, (2, 1, 0)):
        if should_flip and transformed.shape[0] > 0:
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
    asi_camera_conversion: float | None = None,
    write_fused_tiff: bool = False,
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
    """Postprocess qi2lab OPM dataset.

    This code assumes data is generated by opm-v2 GUI and the resulting data is     saved using OPMMirrorHandler. All revelant metadata is read from imaging     files, including stage transformation, camera parameters, and channels.

    Usage: `process "/path/to/qi2lab_acquisition.zarr"`

    See docstring for the various options available.


    Parameters
    ----------
    root_path: Path
        Path to OPM pymmcoregui zarr file.
    output: Path or None, default = None
        Directory for processed artifacts. Defaults to the source directory.
    live: Path or None, default = None
        Supplied illumination image and live-processing mode switch.
    deconvolve: bool, default = False
        Deconvolve the data using RLGC.
    save_float32: bool, default = False
        Save calibrated processing products as float32 instead of uint16.
    max_projection: bool, default = True
        Create a maximum projection datastore.
    flatfield_correction: bool, default = False
        Estimate and apply flatfield correction on raw data.
    create_fused_max_projection: bool, default = True
        Create stage position fused max Z projection.
    write_fused_max_projection_tiff: bool, default = False
        Write fused maxZ  projection to OME-TIFF file.
    z_downsample_level: int, default = 2
        Amount to downsample deskewed data in z.
    time_range: list[int,int], default = None
        Range of timepoints to reconstruct.
    pos_range: list[int,int], default = None
        Range of stage positions to reconstruct.
    eager_mode: bool, default = False
        Use stricter iteration cutoff, potentially leading to over-fitting.
    decon_psf_paths: list[Path] or None, default = None
        Optional channel-ordered ``.npy`` or image PSFs. When omitted,
        wavelength-specific theoretical skewed PSFs are generated.

    crop_after_deskew : bool
        Value supplied for ``crop after deskew``.
    decon_crop_scan : int
        Retained deconvolution tile size along the acquisition scan axis.
    decon_gpu_id : int
        Value supplied for ``decon gpu id``.
    decon_verbose : int
        Value supplied for ``decon verbose``.
    decon_fallback_step_scan : int
        Scan planes removed from the tile after a GPU allocation failure.
    asi_camera_conversion : float | None
        Value supplied for ``asi camera conversion``.
    write_fused_tiff : bool
        Value supplied for ``write fused tiff``.

    Returns
    -------
    None
        No value is returned.
    """
    # Retired Zarr-v2 and ASI inputs do not consistently record orientation.
    # Current OPM-v2 acquisitions replace this fallback from normalized metadata.
    stage_axis_flips = (False, False, False)

    if live is not None:
        if flatfield_correction:
            raise ValueError(
                "--live supplies illumination directly and cannot be combined with "
                "--flatfield-correction"
            )
        if time_range is not None or pos_range is not None:
            raise ValueError("--live does not support time or position subranges")
        acquisition, live_manifest, live_log_path = _open_live_acquisition(root_path)
        root_path = acquisition.path
        opm_mode = acquisition.mode.casefold()
        if "mirror" not in opm_mode and "stage" not in opm_mode:
            raise ValueError(
                "--live currently supports mirror- and stage-scan acquisitions only"
            )
        output_dir = _resolve_output_directory(root_path, output)
        process_skewed(
            root_path=root_path,
            zattrs=None,
            output_dir=output_dir,
            acquisition=acquisition,
            deconvolve=deconvolve,
            save_float32=save_float32,
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
            stage_axis_flips=acquisition.stage_axis_flips_xyz,
            illumination_path=live,
            live_manifest=live_manifest,
            live_log_path=live_log_path,
        )
        return

    if root_path.suffix == ".zarr" or root_path.is_dir():
        acquisition: AcquisitionMetadata | None = None
        zattrs: dict | None = None
        if (root_path / ".zattrs").is_file():
            # Compatibility for the retired root-array Zarr v2 writer.
            with (root_path / ".zattrs").open(encoding="utf-8") as stream:
                zattrs = json.load(stream)
            opm_mode = str(find_key(zattrs, "mode"))
        else:
            acquisition = inspect_acquisition(root_path)
            root_path = acquisition.path
            opm_mode = acquisition.mode
            stage_axis_flips = acquisition.stage_axis_flips_xyz
            sizes = acquisition.index_sizes
            print(
                "Acquisition metadata: "
                f"T={sizes.get('t', 1)}, P={sizes.get('p', 1)}, "
                f"C={sizes.get('c', 1)}, Z={sizes.get('z', 1)}; "
                f"channels={list(acquisition.channel_names)}"
            )
        output_dir = _resolve_output_directory(root_path, output)
        print(f"Processing OPM mode: {opm_mode}")
        if "mirror" in opm_mode or "stage" in opm_mode:
            process_skewed(
                root_path=root_path,
                zattrs=zattrs,
                output_dir=output_dir,
                acquisition=acquisition,
                deconvolve=deconvolve,
                save_float32=save_float32,
                max_projection=max_projection,
                flatfield_correction=flatfield_correction,
                create_fused_max_projection=create_fused_max_projection,
                write_fused_max_projection_tiff=write_fused_max_projection_tiff,
                z_downsample_level=z_downsample_level,
                crop_after_deskew=crop_after_deskew,
                time_range=time_range,
                pos_range=pos_range,
                decon_crop_scan=decon_crop_scan,
                decon_gpu_id=decon_gpu_id,
                decon_verbose=decon_verbose,
                decon_fallback_step_scan=decon_fallback_step_scan,
                decon_psf_paths=decon_psf_paths,
                stage_axis_flips=stage_axis_flips,
            )
        elif "projection" in opm_mode:
            process_projection(
                root_path=root_path,
                zattrs=zattrs,
                output_dir=output_dir,
                acquisition=acquisition,
                deconvolve=deconvolve,
                save_float32=save_float32,
                flatfield_correction=flatfield_correction,
                write_fused_max_projection_tiff=write_fused_max_projection_tiff,
                time_range=time_range,
                pos_range=pos_range,
                eager_deconvolution=eager_mode,
                decon_crop_scan=decon_crop_scan,
                decon_gpu_id=decon_gpu_id,
                decon_verbose=decon_verbose,
                decon_fallback_step_scan=decon_fallback_step_scan,
                stage_axis_flips=stage_axis_flips,
            )
    elif root_path.suffixes[-2:] == [".ome", ".tif"]:
        output_dir = _resolve_output_directory(root_path, output)
        with TiffFile(root_path) as tif:
            axes = tif.series[0].axes
            all_metadata = dict(tif.micromanager_metadata)
            micromanager_metadata = all_metadata["Summary"]
            asi_metadata = json.loads(all_metadata["Summary"]["SPIMAcqSettings"])
            process_ASI_SCOPE(
                root_path=root_path,
                output_dir=output_dir,
                axes=axes,
                micromanager_metadata=micromanager_metadata,
                asi_metadata=asi_metadata,
                deconvolve=deconvolve,
                save_float32=save_float32,
                max_projection=max_projection,
                flatfield_correction=flatfield_correction,
                create_fused_max_projection=create_fused_max_projection,
                write_fused_max_projection_tiff=write_fused_max_projection_tiff,
                write_fused_tiff=write_fused_tiff,
                z_downsample_level=z_downsample_level,
                crop_after_deskew=crop_after_deskew,
                time_range=time_range,
                pos_range=pos_range,
                decon_crop_scan=decon_crop_scan,
                decon_gpu_id=decon_gpu_id,
                decon_verbose=decon_verbose,
                decon_fallback_step_scan=decon_fallback_step_scan,
                camera_conversion_override=asi_camera_conversion,
                stage_axis_flips=stage_axis_flips,
            )


def process_skewed(
    root_path: Path,
    zattrs: dict | None,
    acquisition: AcquisitionMetadata | None = None,
    deconvolve: bool = False,
    save_float32: bool = False,
    max_projection: bool = True,
    flatfield_correction: bool = False,
    create_fused_max_projection: bool = True,
    write_fused_max_projection_tiff: bool = False,
    z_downsample_level: int = 2,
    crop_after_deskew: bool = False,
    time_range: tuple[int, int] = None,
    pos_range: tuple[int, int] = None,
    excess_overide: int = None,
    flyback_crop: int = None,
    decon_crop_scan: int | None = None,
    decon_gpu_id: int = 0,
    decon_verbose: int = 1,
    decon_fallback_step_scan: int | None = None,
    decon_psf_paths: list[Path] | None = None,
    stage_axis_flips: tuple[bool, bool, bool] = (False, True, True),
    output_dir: Path | None = None,
    illumination_path: Path | None = None,
    live_manifest: LiveManifest | None = None,
    live_log_path: Path | None = None,
):
    """Postprocess qi2lab OPM dataset.

    This code assumes data is generated by opm-v2 GUI and the resulting data is     saved using OPMMirrorHandler. All revelant metadata is read from imaging     files, including stage transformation, camera parameters, and channels.

    Usage: `deskew "/path/to/qi2lab_acquisition.zarr"`

    See docstring for the various options available.

    Outputs are in:
    - Deskewed 3D individual deskewed tiles:         `"/path/to/qi2lab_acquisition_deskewed.ome.zarr"`
    - Maximum Z projected individual deskewed tiles:         `"/path/to/qi2lab_acquisition_max_z_deskewed.ome.zarr"`
    - Maximum Z projection fused deskewed tiles:         `"/path/to/qi2lab_acquisition_max_z_fused.ome.zarr"`

    Parameters
    ----------
    root_path: Path
        Path to OPM pymmcoregui zarr file.
    zattrs: dict
        Metadata dictionary containing OPM data attributes.
    output_dir: Path or None, default = None
        Directory for processed artifacts. Defaults to the source directory.
    illumination_path: Path or None, default = None
        User-supplied CYX illumination image for live processing.
    live_manifest: LiveManifest or None, default = None
        Upfront acquisition plan enabling live tile discovery.
    live_log_path: Path or None, default = None
        Append-only acquisition lifecycle log.
    deconvolve: bool, default = False
        Deconvolve the data using RLGC.
    save_float32: bool, default = False
        Save calibrated processing products as float32 instead of uint16.
    max_projection: bool, default = True
        Create a maximum projection datastore.
    flatfield_correction: bool, default = True
        Estimate and apply flatfield correction on raw data.
    create_fused_max_projection: bool, default = True
        Create stage position fused max Z projection.
    write_fused_max_projection_tiff: bool, default = False
        Write fused maxZ  projection to OME-TIFF file.
    z_downsample_level: int, default = 2
        Amount to downsample deskewed data in z.
    time_range: list[int,int], default = None
        Range of timepoints to reconstruct.
    pos_range: list[int,int], default = None
        Range of stage positions to reconstruct.
    decon_psf_paths: list[Path] or None, default = None
        Optional channel-ordered ``.npy`` or image PSFs. When omitted,
        wavelength-specific theoretical skewed PSFs are generated.

    crop_after_deskew : bool
        Value supplied for ``crop after deskew``.
    excess_overide : int
        Value supplied for ``excess overide``.
    flyback_crop : int
        Value supplied for ``flyback crop``.
    decon_crop_scan : int
        Retained deconvolution tile size along the acquisition scan axis.
    decon_gpu_id : int
        Value supplied for ``decon gpu id``.
    decon_verbose : int
        Value supplied for ``decon verbose``.
    decon_fallback_step_scan : int
        Scan planes removed from the tile after a GPU allocation failure.
    stage_axis_flips : tuple[bool, bool, bool]
        Value supplied for ``stage axis flips``.

    Returns
    -------
    None
        No value is returned.
    """
    if deconvolve:
        from opm_processing.imageprocessing.rlgc import (
            RlgcChunkState,
            chunked_rlgc,
        )

        decon_chunk_state = RlgcChunkState(decon_crop_scan)

    output_dir = _resolve_output_directory(root_path, output_dir)

    if acquisition is not None:
        datastore = open_acquisition_datastore(acquisition)
        opm_mode = acquisition.mode
        if acquisition.scan_axis_step_um is None:
            raise ValueError("Acquisition metadata lacks a scan-axis step")
        scan_axis_step_um = acquisition.scan_axis_step_um
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
    else:
        if zattrs is None:
            raise ValueError("Legacy processing requires Zarr attributes")
        spec = {
            "driver": "zarr",
            "kvstore": {"driver": "file", "path": str(root_path)},
        }
        datastore = ts.open(spec).result()
        opm_mode = str(find_key(zattrs, "mode"))
        if "mirror" in opm_mode:
            scan_axis_step_um = float(find_key(zattrs, "image_mirror_step_um"))
            excess_scan_positions = 0
        elif "stage" in opm_mode:
            scan_axis_step_um = float(find_key(zattrs, "scan_axis_step_um"))
            excess_scan_positions = int(find_key(zattrs, "excess_scan_positions"))
        pixel_size_um = find_key(zattrs, "pixel_size_um")
        opm_tilt_deg = find_key(zattrs, "angle_deg")
        camera_offset = find_key(zattrs, "offset")
        camera_conversion = find_key(zattrs, "e_to_ADU")
        channels = extract_channels(zattrs)
        stage_positions_raw = extract_stage_positions(zattrs)
        scan_axis_reversed = "stage" in opm_mode

    if "stage" in opm_mode:
        if excess_overide is not None:
            excess_scan_positions = excess_overide
        if flyback_crop is not None:
            flyback_crop = int(flyback_crop)
    if None in (pixel_size_um, opm_tilt_deg, camera_offset, camera_conversion):
        raise ValueError("Acquisition metadata lacks required OPM/camera calibration")
    pixel_size_um = float(pixel_size_um)
    opm_tilt_deg = float(opm_tilt_deg)
    camera_offset = float(camera_offset)
    camera_conversion = float(camera_conversion)
    output_dtype = _processed_dtype(save_float32)
    if decon_psf_paths is not None:
        if len(decon_psf_paths) != len(channels):
            raise ValueError("decon_psf_paths must contain one PSF per channel")
        psfs = [
            (np.load(path) if Path(path).suffix == ".npy" else np.asarray(imread(path)))
            for path in decon_psf_paths
        ]
    else:
        psfs = None
    stage_positions = _apply_stage_axis_flips(stage_positions_raw, stage_axis_flips)
    stage_x_flipped, stage_y_flipped, stage_z_flipped = stage_axis_flips

    # Estimate the shape of one deskewed volume.
    deskew_input_shape = (
        datastore.shape[-3]
        - excess_scan_positions
        - (flyback_crop if flyback_crop is not None else 0),
        datastore.shape[-2],
        datastore.shape[-1],
    )
    if flyback_crop is not None:
        deskewed_shape, pad_y, pad_x, crop_y = deskew_shape_estimator(
            deskew_input_shape,
            theta=opm_tilt_deg,
            distance=scan_axis_step_um,
            pixel_size=pixel_size_um,
            crop_after_deskew=crop_after_deskew,
        )
    else:
        deskewed_shape, pad_y, pad_x, crop_y = deskew_shape_estimator(
            deskew_input_shape,
            theta=opm_tilt_deg,
            distance=scan_axis_step_um,
            pixel_size=pixel_size_um,
            crop_after_deskew=crop_after_deskew,
        )

    if time_range is not None:
        time_shape = time_range[1]
    else:
        time_shape = datastore.shape[0]

    if pos_range is not None:
        pos_shape = pos_range[1]
    else:
        pos_shape = datastore.shape[1]

    datastore_shape = [
        time_shape,
        pos_shape,
        datastore.shape[2],
        deskewed_shape[0] // z_downsample_level,
        deskewed_shape[1],
        deskewed_shape[2],
    ]
    # create array to hold one deskewed volume
    deskewed = np.zeros(
        (deskewed_shape[0] // z_downsample_level, deskewed_shape[1], deskewed_shape[2]),
        dtype=np.float32,
    )

    flatfield_path = (
        Path(illumination_path).expanduser().resolve()
        if illumination_path is not None
        else output_dir / Path(acquisition_stem(root_path) + "_flatfield.ome.tif")
    )
    provided_flatfields = (
        _load_provided_illumination(
            flatfield_path,
            (
                int(datastore.shape[2]),
                int(datastore.shape[-2]),
                int(datastore.shape[-1]),
            ),
        )
        if illumination_path is not None
        else None
    )
    illumination_parameters = (
        {
            "source": "provided",
            "artifact": str(flatfield_path),
            "sha256": _file_sha256(flatfield_path),
            "estimated": False,
        }
        if illumination_path is not None
        else {
            "source": "estimated",
            "estimator": "BaSiCPy",
            "estimator_version": _distribution_version("basicpy"),
            "configuration": "autotuned_then_smoothness_2_with_residual_calibration",
            "sort_intensity": True,
            "smoothness_flatfield": 2.0,
            "sampling": "per_position_summaries_across_all_positions",
            "planes_per_position": 10,
            "fit_summary": "median",
            "residual_summary": "75th_percentile",
            "residual_calibration": "smoothed_separable_yx",
            "working_size": "half_resolution_preserving_aspect_ratio",
            "darkfield_estimation": False,
            "artifact_schema": _flatfield_software_tag(),
            "artifact": str(flatfield_path.resolve()),
            "estimated": True,
        }
    )
    output_kind = "deconvolved_deskewed" if deconvolve else "deskewed"
    processing_steps = [
        _selection_step(time_range, pos_range),
        {
            "name": "camera_correction",
            "applied": True,
            "parameters": {
                "offset": camera_offset,
                "e_to_ADU": camera_conversion,
                "formula": "(raw - offset) * e_to_ADU",
            },
        },
        {
            "name": "illumination_correction",
            "applied": flatfield_correction,
            "parameters": illumination_parameters,
        },
        {
            "name": "live_acquisition",
            "applied": live_manifest is not None,
            "parameters": None
            if live_manifest is None
            else {
                "acquisition_id": live_manifest.acquisition_id,
                "manifest": str(live_manifest.path),
                "readiness": "raw_zarr_chunk_presence",
            },
        },
        {
            "name": "scan_crop",
            "applied": bool(excess_scan_positions or flyback_crop),
            "parameters": {
                "leading_positions": int(excess_scan_positions),
                "trailing_positions": None
                if flyback_crop is None
                else int(flyback_crop),
            },
        },
        {
            "name": "scan_axis_flip",
            "applied": scan_axis_reversed,
            "parameters": {"axis": "z"},
        },
        {
            "name": "deconvolution",
            "applied": deconvolve,
            "parameters": {
                "method": "RLGC",
                "crop_scan": (
                    "auto" if decon_crop_scan is None else int(decon_crop_scan)
                ),
                "gpu_id": int(decon_gpu_id),
                "verbose": int(decon_verbose),
                "fallback_step_scan": (
                    "adaptive"
                    if decon_fallback_step_scan is None
                    else int(decon_fallback_step_scan)
                ),
                "psf_source": "provided" if decon_psf_paths else "theoretical",
                "psf_paths": None
                if decon_psf_paths is None
                else [str(Path(path).resolve()) for path in decon_psf_paths],
            },
        },
        {
            "name": "deskew",
            "applied": True,
            "parameters": {
                "theta_deg": opm_tilt_deg,
                "scan_axis_step_um": scan_axis_step_um,
                "raw_pixel_size_um": pixel_size_um,
                "z_downsample_level": int(z_downsample_level),
            },
        },
        {
            "name": "crop_after_deskew",
            "applied": crop_after_deskew,
            "parameters": {"crop_y_pixels_per_side": int(crop_y)},
        },
        _output_conversion_step(save_float32),
    ]
    processing_metadata = {
        "scan_axis_step_um": scan_axis_step_um,
        "raw_pixel_size_um": pixel_size_um,
        "opm_tilt_deg": opm_tilt_deg,
        "camera_corrected": True,
        "camera_offset": camera_offset,
        "camera_e_to_ADU": camera_conversion,
        "deskewed_voxel_size_um": [
            z_downsample_level * pixel_size_um,
            pixel_size_um,
            pixel_size_um,
        ],
        "deskew_input_shape_zyx": [int(value) for value in deskew_input_shape],
        "stage_x_flipped": stage_x_flipped,
        "stage_y_flipped": stage_y_flipped,
        "stage_z_flipped": stage_z_flipped,
        "flatfield_corrected": flatfield_correction,
        "pad_y": pad_y,
        "pad_x": pad_x,
        "opm_processing": _processing_provenance(
            root_path,
            output_kind,
            processing_steps,
        ),
    }

    if not (deconvolve):
        output_path = output_dir / Path(
            acquisition_stem(root_path) + "_deskewed.ome.zarr"
        )
    else:
        output_path = output_dir / Path(
            acquisition_stem(root_path) + "_decon_deskewed.ome.zarr"
        )
    output_collection = create_position_collection(
        output_path,
        datastore_shape,
        processing_metadata["deskewed_voxel_size_um"],
        stage_positions=stage_positions[:pos_shape],
        channels=channels,
        attributes=processing_metadata,
        dtype=output_dtype,
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
        max_z_metadata = _derived_processing_metadata(
            processing_metadata,
            f"max_z_{output_kind}",
            {
                "name": "maximum_projection",
                "applied": True,
                "parameters": {"axis": "z"},
            },
        )
        max_z_metadata["deskewed_voxel_size_um"] = [
            pixel_size_um,
            pixel_size_um,
            pixel_size_um,
        ]
        max_z_collection = create_position_collection(
            max_z_output_path,
            max_z_datastore_shape,
            max_z_metadata["deskewed_voxel_size_um"],
            stage_positions=stage_positions[:pos_shape],
            channels=channels,
            attributes=max_z_metadata,
            dtype=output_dtype,
        )
        max_z_ts_store = max_z_collection.arrays

    if flatfield_correction:
        if provided_flatfields is not None:
            flatfields = provided_flatfields
        else:
            flatfields = _load_or_estimate_flatfield(
                flatfield_path,
                datastore,
                camera_offset,
                camera_conversion,
                pixel_size_um,
            )
    else:
        flatfields = np.ones(
            (datastore.shape[2], datastore.shape[-2], datastore.shape[-1]),
            dtype=np.float32,
        )

    # Loop over complete tiles and await each tile's output before moving on.
    if live_manifest is not None:
        if live_log_path is None:
            raise ValueError("Live processing requires an acquisition lifecycle log")
        readiness = ZarrTileReadiness(acquisition)
        tile_groups = (
            (t_idx, (pos_idx,))
            for t_idx, pos_idx in iter_live_tiles(
                readiness, live_manifest, live_log_path
            )
        )
    else:
        time_indices = (
            range(time_range[0], time_range[1])
            if time_range is not None
            else range(datastore.shape[0])
        )
        position_indices = (
            range(pos_range[0], pos_range[1])
            if pos_range is not None
            else range(datastore.shape[1])
        )
        tile_groups = (
            (t_idx, position_indices) for t_idx in tqdm(time_indices, desc="t")
        )

    wrote_tile = False
    for t_idx, current_positions in tile_groups:
        for pos_idx in tqdm(current_positions, desc="p", leave=False):
            wrote_tile = True
            ts_writes = []
            ts_max_writes = []
            for chan_idx in tqdm(range(datastore.shape[2]), desc="c", leave=False):
                camera_corrected_data = _camera_corrected_image(
                    np.squeeze(
                        datastore[t_idx, pos_idx, chan_idx, :].read().result()
                    ),
                    camera_offset,
                    camera_conversion,
                    flatfields[chan_idx, :],
                )
                if scan_axis_reversed:
                    camera_corrected_data = np.flip(camera_corrected_data, axis=0)

                if deconvolve:
                    if psfs is None:
                        psfs = []
                        for psf_idx in range(datastore.shape[2]):
                            psf = generate_skewed_psf(
                                em_wvl=float(
                                    int(str(channels[psf_idx]).rstrip("nm")) / 1000
                                ),
                                pixel_size_um=pixel_size_um,
                                scan_axis_step_um=scan_axis_step_um,
                                theta_deg=opm_tilt_deg,
                                pz=0.0,
                                plot=False,
                            )
                            psfs.append(psf)

                    if flyback_crop is not None:
                        decon_input = camera_corrected_data[
                            excess_scan_positions:-flyback_crop, :, :
                        ]
                    else:
                        decon_input = camera_corrected_data[
                            excess_scan_positions:, :, :
                        ]
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
                        distance=scan_axis_step_um,
                        pixel_size=pixel_size_um,
                        downsample_factor=z_downsample_level,
                        output_dtype=np.float32,
                    )
                else:
                    if flyback_crop is not None:
                        deskewed = orthogonal_deskew(
                            camera_corrected_data[
                                excess_scan_positions:-flyback_crop, :, :
                            ],
                            theta=opm_tilt_deg,
                            distance=scan_axis_step_um,
                            pixel_size=pixel_size_um,
                            downsample_factor=z_downsample_level,
                            output_dtype=np.float32,
                        )
                    else:
                        deskewed = orthogonal_deskew(
                            camera_corrected_data[excess_scan_positions:, :, :],
                            theta=opm_tilt_deg,
                            distance=scan_axis_step_um,
                            pixel_size=pixel_size_um,
                            downsample_factor=z_downsample_level,
                            output_dtype=np.float32,
                        )

                if crop_after_deskew:
                    deskewed = deskewed[:, crop_y:-crop_y, :]

                if max_projection:
                    max_z_deskewed = np.max(deskewed, axis=0, keepdims=True)
                    # create future objects for async data writing
                    ts_max_writes.append(
                        max_z_ts_store[pos_idx][t_idx, chan_idx].write(
                            _format_processed_output(max_z_deskewed, save_float32)
                        )
                    )

                # create future objects for async data writing
                ts_writes.append(
                    ts_store[pos_idx][t_idx, chan_idx].write(
                        _format_processed_output(deskewed, save_float32)
                    )
                )
            for ts_write in ts_writes:
                ts_write.result()
            if max_projection:
                for ts_max_write in ts_max_writes:
                    ts_max_write.result()

    if wrote_tile:
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
        max_z_ts_store = open_position_collection(max_z_output_path).arrays

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
            pixel_size=np.asarray((pixel_size_um, pixel_size_um), dtype=np.float32),
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
    zattrs: dict | None,
    acquisition: AcquisitionMetadata | None = None,
    deconvolve: bool = True,
    save_float32: bool = False,
    flatfield_correction: bool = True,
    write_fused_max_projection_tiff: bool = True,
    time_range: tuple[int, int] = None,
    pos_range: tuple[int, int] = None,
    eager_deconvolution: bool = False,
    overwrite: bool = False,
    decon_crop_scan: int | None = None,
    decon_gpu_id: int = 0,
    decon_verbose: int = 1,
    decon_fallback_step_scan: int | None = None,
    stage_axis_flips: tuple[bool, bool, bool] = (False, True, True),
    output_dir: Path | None = None,
):
    """Postprocess qi2lab OPM dataset.

    This code assumes data is generated by opm-v2 GUI and the resulting data is     saved using OPMMirrorHandler. All revelant metadata is read from imaging     files, including stage transformation, camera parameters, and channels.

    Usage: `process "/path/to/qi2lab_acquisition.zarr"`

    See docstring for the various options available.

    Outputs are in:
    - Deconvolved individual projection tiles:         `"/path/to/qi2lab_acquisition_decon_projection.ome.zarr"`
    - Stage position fused projection tiles:         `"/path/to/qi2lab_acquisition_stagefused.ome.zarr"`

    Parameters
    ----------
    root_path: Path
        Path to OPM pymmcoregui zarr file.
    zattrs: dict
        Metadata dictionary containing OPM data attributes.
    output_dir: Path or None, default = None
        Directory for processed artifacts. Defaults to the source directory.
    deconvolve: bool, default = False
        Deconvolve the data using RLGC.
    save_float32: bool, default = False
        Save calibrated processing products as float32 instead of uint16.
    flatfield_correction: bool, default = True
        Estimate and apply flatfield correction on raw data.
    write_fused_max_projection_tiff: bool, default = False
        Write fused maxZ  projection to OME-TIFF file.
    time_range: list[int,int], default = None
        Range of timepoints to process.
    pos_range: list[int,int], default = None
        Range of stage positions to process.
    eager_mode: bool, default = False
        Use stricter iteration cutoff, potentially leading to over-fitting.

    eager_deconvolution : bool
        Value supplied for ``eager deconvolution``.
    overwrite : bool
        Value supplied for ``overwrite``.
    decon_crop_scan : int
        Retained deconvolution tile size along the acquisition scan axis.
    decon_gpu_id : int
        Value supplied for ``decon gpu id``.
    decon_verbose : int
        Value supplied for ``decon verbose``.
    decon_fallback_step_scan : int
        Scan planes removed from the tile after a GPU allocation failure.
    stage_axis_flips : tuple[bool, bool, bool]
        Value supplied for ``stage axis flips``.

    Returns
    -------
    None
        No value is returned.
    """
    if deconvolve:
        from opm_processing.imageprocessing.rlgc import (
            RlgcChunkState,
            chunked_rlgc,
        )

        decon_chunk_state = RlgcChunkState(decon_crop_scan)

    output_dir = _resolve_output_directory(root_path, output_dir)

    if acquisition is not None:
        datastore = open_acquisition_datastore(acquisition)
        pixel_size_um = acquisition.pixel_size_um
        opm_tilt_deg = acquisition.angle_deg
        camera_offset = acquisition.camera_offset
        camera_conversion = acquisition.camera_conversion
        channels = list(acquisition.channel_names)
        stage_positions_raw = np.asarray(acquisition.stage_positions_zxy, dtype=float)
        stage_axis_flips = acquisition.stage_axis_flips_xyz
    else:
        if zattrs is None:
            raise ValueError("Legacy processing requires Zarr attributes")
        spec = {
            "driver": "zarr",
            "kvstore": {"driver": "file", "path": str(root_path)},
        }
        datastore = ts.open(spec).result()
        pixel_size_um = find_key(zattrs, "pixel_size_um")
        opm_tilt_deg = find_key(zattrs, "angle_deg")
        camera_offset = find_key(zattrs, "offset")
        camera_conversion = find_key(zattrs, "e_to_ADU")
        channels = extract_channels(zattrs)
        stage_positions_raw = extract_stage_positions(zattrs)
    if None in (pixel_size_um, opm_tilt_deg, camera_offset, camera_conversion):
        raise ValueError("Acquisition metadata lacks required OPM/camera calibration")
    pixel_size_um = float(pixel_size_um)
    opm_tilt_deg = float(opm_tilt_deg)
    camera_offset = float(camera_offset)
    camera_conversion = float(camera_conversion)
    output_dtype = _processed_dtype(save_float32)
    stage_positions = _apply_stage_axis_flips(stage_positions_raw, stage_axis_flips)
    stage_x_flipped, stage_y_flipped, stage_z_flipped = stage_axis_flips

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
    elif datastore.rank != 6 or datastore.shape[3] != 1:
        raise ValueError(
            "Projection acquisitions must have TPCYX shape or a singleton Z axis; "
            f"got {datastore.shape}"
        )

    if deconvolve:
        output_path = output_dir / Path(
            acquisition_stem(root_path) + "_decon_projection.ome.zarr"
        )
    else:
        output_path = output_dir / Path(
            acquisition_stem(root_path) + "_projection.ome.zarr"
        )
    if not (output_path.exists()) or overwrite:
        flatfield_path = output_dir / Path(
            acquisition_stem(root_path) + "_flatfield.ome.tif"
        )
        output_kind = "deconvolved_projection" if deconvolve else "projection"
        processing_metadata = {
            "raw_pixel_size_um": pixel_size_um,
            "opm_tilt_deg": opm_tilt_deg,
            "camera_corrected": True,
            "camera_offset": camera_offset,
            "camera_e_to_ADU": camera_conversion,
            "deskewed_voxel_size_um": [1.0, pixel_size_um, pixel_size_um],
            "stage_x_flipped": stage_x_flipped,
            "stage_y_flipped": stage_y_flipped,
            "stage_z_flipped": stage_z_flipped,
            "flatfield_corrected": flatfield_correction,
            "opm_processing": _processing_provenance(
                root_path,
                output_kind,
                [
                    _selection_step(time_range, pos_range),
                    {
                        "name": "camera_correction",
                        "applied": True,
                        "parameters": {
                            "offset": camera_offset,
                            "e_to_ADU": camera_conversion,
                            "formula": "(raw - offset) * e_to_ADU",
                        },
                    },
                    {
                        "name": "illumination_correction",
                        "applied": flatfield_correction,
                        "parameters": {
                            "estimator": "BaSiCPy",
                            "estimator_version": _distribution_version("basicpy"),
                            "configuration": "autotuned_then_smoothness_2_with_residual_calibration",
                            "sort_intensity": True,
                            "smoothness_flatfield": 2.0,
                            "sampling": "per_position_summaries_across_all_positions",
                            "planes_per_position": 10,
                            "fit_summary": "median",
                            "residual_summary": "75th_percentile",
                            "residual_calibration": "smoothed_separable_yx",
                            "working_size": "half_resolution_preserving_aspect_ratio",
                            "darkfield_estimation": False,
                            "artifact_schema": _flatfield_software_tag(),
                            "artifact": str(flatfield_path.resolve()),
                        },
                    },
                    {
                        "name": "deconvolution",
                        "applied": deconvolve,
                        "parameters": {
                            "method": "RLGC",
                            "crop_scan": (
                                "auto"
                                if decon_crop_scan is None
                                else int(decon_crop_scan)
                            ),
                            "gpu_id": int(decon_gpu_id),
                            "verbose": int(decon_verbose),
                            "fallback_step_scan": (
                                "adaptive"
                                if decon_fallback_step_scan is None
                                else int(decon_fallback_step_scan)
                            ),
                            "safe_mode": not eager_deconvolution,
                            "psf_source": "theoretical",
                        },
                    },
                    _output_conversion_step(save_float32),
                ],
            ),
        }
        output_collection = create_position_collection(
            output_path,
            datastore.shape,
            processing_metadata["deskewed_voxel_size_um"],
            stage_positions=stage_positions,
            channels=channels,
            attributes=processing_metadata,
            overwrite=overwrite,
            dtype=output_dtype,
        )
        ts_store = output_collection.arrays

        if flatfield_correction:
            flatfields = _load_or_estimate_flatfield(
                flatfield_path,
                datastore,
                camera_offset,
                camera_conversion,
                pixel_size_um,
            )
        else:
            flatfields = np.ones(
                (datastore.shape[2], datastore.shape[-2], datastore.shape[-1]),
                dtype=np.float32,
            )

        # loop over all components and stream to zarr using tensorstore
        ts_writes = []

        if time_range is not None:
            time_iterator = tqdm(
                range(time_range[0], time_range[1]), desc="t", leave=True
            )
            if time_range[1] > 1:
                refresh_position_iterator = True
            else:
                refresh_position_iterator = False
        else:
            time_iterator = tqdm(range(time_shape), desc="t", leave=True)
            if time_shape > 1:
                refresh_position_iterator = True
            else:
                refresh_position_iterator = False

        if pos_range is not None:
            pos_iterator = tqdm(
                range(pos_range[0], pos_range[1]), desc="p", leave=False
            )
        else:
            pos_iterator = tqdm(range(pos_shape), desc="p", leave=False)

        for t_idx in time_iterator:
            for pos_idx in pos_iterator:
                for chan_idx in tqdm(range(datastore.shape[2]), desc="c", leave=False):
                    camera_corrected_data = _camera_corrected_image(
                        np.squeeze(
                            datastore[t_idx, pos_idx, chan_idx, :].read().result()
                        ),
                        camera_offset,
                        camera_conversion,
                        flatfields[chan_idx, :],
                    )

                    if deconvolve:
                        if pos_idx == 0 and chan_idx == 0:
                            psfs = []
                            for psf_idx in range(datastore.shape[2]):
                                psf = generate_proj_psf(
                                    em_wvl=float(
                                        int(str(channels[psf_idx]).rstrip("nm")) / 1000
                                    ),
                                    pixel_size_um=pixel_size_um,
                                )
                                psfs.append(psf)

                        if eager_deconvolution:
                            safe_stop = False
                        else:
                            safe_stop = True
                        decon_shape = tuple(
                            int(size) for size in camera_corrected_data.shape
                        )
                        if len(decon_shape) == 2:
                            decon_shape = (1, *decon_shape)
                        effective_crop_scan = decon_chunk_state.determine_once(
                            decon_shape,
                            [tuple(int(size) for size in psf.shape) for psf in psfs],
                            gpu_id=decon_gpu_id,
                        )
                        deconvolved_data = chunked_rlgc(
                            image=camera_corrected_data,
                            psf=np.asarray(psfs[chan_idx]),
                            crop_scan=effective_crop_scan,
                            gpu_id=decon_gpu_id,
                            verbose=decon_verbose,
                            fallback_step_scan=decon_fallback_step_scan,
                            safe_mode=safe_stop,
                            on_successful_crop_scan=(
                                decon_chunk_state.remember_successful_crop
                            ),
                        )
                    else:
                        deconvolved_data = camera_corrected_data.copy()

                    # create future objects for async data writing
                    ts_writes.append(
                        ts_store[pos_idx][t_idx, chan_idx].write(
                            _format_processed_output(
                                deconvolved_data, save_float32
                            )
                        )
                    )
            if refresh_position_iterator:
                if pos_range is not None:
                    pos_iterator = tqdm(
                        range(pos_range[0], pos_range[1]), desc="p", leave=False
                    )
                else:
                    pos_iterator = tqdm(range(pos_shape), desc="p", leave=False)

        # wait for writes to finish
        for ts_write in ts_writes:
            ts_write.result()

        print("\nFusing using stage positions...")
        fused_output_path = output_dir / Path(
            acquisition_stem(root_path) + "_stagefused.ome.zarr"
        )

        if pos_range is not None:
            tile_positions = stage_positions[pos_range[0] : pos_range[1]]

        else:
            tile_positions = stage_positions

        tile_fusion = MaxTileFusion(
            ts_dataset=ts_store,
            tile_positions=tile_positions,
            output_path=fused_output_path,
            pixel_size=np.asarray((pixel_size_um, pixel_size_um), dtype=np.float32),
            time_range=time_range,
            opm_angle_deg=opm_tilt_deg,
        )
        tile_fusion.run()
        del deconvolved_data, ts_write, ts_store

    if write_fused_max_projection_tiff:
        try:
            tiff_dir_path = fused_output_path.parent / Path("fused_tiff_output")
        except Exception:
            fused_output_path = output_dir / Path(
                acquisition_stem(root_path) + "_stagefused.ome.zarr"
            )
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


def process_ASI_SCOPE(
    root_path: Path,
    axes: str,
    micromanager_metadata: dict,
    asi_metadata: dict,
    deconvolve: bool = False,
    save_float32: bool = False,
    max_projection: bool = True,
    flatfield_correction: bool = False,
    create_fused_max_projection: bool = True,
    write_fused_max_projection_tiff: bool = False,
    write_fused_tiff: bool = True,
    z_downsample_level: int = 4,
    crop_after_deskew: bool = False,
    time_range: tuple[int, int] = None,
    pos_range: tuple[int, int] = None,
    decon_crop_scan: int | None = None,
    decon_gpu_id: int = 0,
    decon_verbose: int = 1,
    decon_fallback_step_scan: int | None = None,
    camera_conversion_override: float | None = None,
    stage_axis_flips: tuple[bool, bool, bool] = (False, True, True),
    output_dir: Path | None = None,
):
    """Postprocess ASI SCOPE OPM dataset.

    This code assumes data is generated by ASI MM plugin GUI and the resulting data is     saved using ome.tiff handler. All revelant metadata is read from imaging     files, including stage transformation, camera parameters, and channels.

    Usage: `process "/path/to/asi_scope_acquisition.ome.tiff"`

    See docstring for the various options available.

    Outputs are in:
    - Deskewed 3D individual deskewed tiles:         `"/path/to/qi2lab_acquisition_deskewed.ome.zarr"`
    - Maximum Z projected individual deskewed tiles:         `"/path/to/qi2lab_acquisition_max_z_deskewed.ome.zarr"`
    - Maximum Z projection fused deskewed tiles:         `"/path/to/qi2lab_acquisition_max_z_fused.ome.zarr"`

    Parameters
    ----------
    root_path: Path
        Path to OPM pymmcoregui zarr file.
    axes: str
        Axes of the OME-TIFF file.
    micromanager_metadata: dict
        Micromanager metadata dictionary containing OPM data attributes.
    asi_metadata: dict
        ASI metadata dictionary containing OPM data attributes.
    output_dir: Path or None, default = None
        Directory for processed artifacts. Defaults to the source directory.
    deconvolve: bool, default = False
        Deconvolve the data using RLGC.
    max_projection: bool, default = True
        Create a maximum projection datastore.
    flatfield_correction: bool, default = True
        Estimate and apply flatfield correction on raw data.
    create_fused_max_projection: bool, default = True
        Create stage position fused max Z projection.
    write_fused_max_projection_tiff: bool, default = False
        Write fused maxZ  projection to OME-TIFF file.
    z_downsample_level: int, default = 2
        Amount to downsample deskewed data in z.
    time_range: list[int,int], default = None
        Range of timepoints to reconstruct.
    pos_range: list[int,int], default = None
        Range of stage positions to reconstruct.

    write_fused_tiff : bool
        Value supplied for ``write fused tiff``.
    crop_after_deskew : bool
        Value supplied for ``crop after deskew``.
    decon_crop_scan : int
        Retained deconvolution tile size along the acquisition scan axis.
    decon_gpu_id : int
        Value supplied for ``decon gpu id``.
    decon_verbose : int
        Value supplied for ``decon verbose``.
    decon_fallback_step_scan : int
        Scan planes removed from the tile after a GPU allocation failure.
    camera_conversion_override : float | None
        Value supplied for ``camera conversion override``.
    stage_axis_flips : tuple[bool, bool, bool]
        Value supplied for ``stage axis flips``.

    Returns
    -------
    None
        No value is returned.
    """
    if deconvolve:
        from opm_processing.imageprocessing.rlgc import (
            RlgcChunkState,
            chunked_rlgc,
        )

        decon_chunk_state = RlgcChunkState(decon_crop_scan)

    output_dir = _resolve_output_directory(root_path, output_dir)

    import re

    import zarr

    store = imread(root_path, aszarr=True)
    datastore = zarr.open(store, mode="r")

    if datastore.ndim != 6:
        if axes == "TZCYX":
            datastore = np.swapaxes(datastore, 1, 2)
            datastore = datastore[:, None, :, :, :]
        elif axes == "ZCYX":
            datastore = np.swapaxes(datastore, 0, 1)
            datastore = datastore[None, None, :, :, :, :]
        elif axes == "CZYX":
            datastore = datastore[None, None, :, :, :, :]
        elif axes == "TZYX":
            datastore = datastore[:, None, None, :, :, :]

    tif = TiffFile(root_path, is_mmstack=False)
    per_image_metadata = dict(tif.series[0].pages[0].tags["MicroManagerMetadata"].value)
    camera_offset = float(per_image_metadata["pvcam-Offset"])
    if camera_conversion_override is not None:
        camera_conversion = float(camera_conversion_override)
    elif "e_to_ADU" in per_image_metadata:
        camera_conversion = float(per_image_metadata["e_to_ADU"])
    else:
        raise ValueError(
            "ASI camera conversion is absent from metadata; pass "
            "--asi-camera-conversion explicitly."
        )
    output_dtype = _processed_dtype(save_float32)
    del tif

    asi_step_um = float(micromanager_metadata["z-step_um"])
    pixel_size_um = float(micromanager_metadata["PixelSize_um"])
    asi_tilt_deg = float(micromanager_metadata["StageScanAnglePathA"])
    opm_tilt_deg = 90 - asi_tilt_deg
    scan_axis_step_um = asi_step_um / np.tan(np.deg2rad(asi_tilt_deg))

    if asi_metadata["isStageScanning"]:
        opm_mode = "stage"
    else:
        opm_mode = "mirror"

    em_values = []
    for ch in asi_metadata["channels"]:
        config = ch.get("config_", "")
        match = re.search(r"(\d+(?:\.\d+)?)\s*em", config)
        if match:
            em_values.append(float(match.group(1)) / 1000.0)

    stage_positions = np.asarray(
        [
            [
                float(0.0),
                float(micromanager_metadata["Position_Y"].strip(" µm")),
                float(micromanager_metadata["Position_X"].strip(" µm")),
            ]
        ]
    )

    stage_positions = _apply_stage_axis_flips(stage_positions, stage_axis_flips)
    stage_x_flipped, stage_y_flipped, stage_z_flipped = stage_axis_flips

    # estimate shape of one deskewed volume
    deskew_input_shape = (
        int(datastore.shape[-3]),
        int(datastore.shape[-2]),
        int(datastore.shape[-1]),
    )
    deskewed_shape, pad_y, pad_x, crop_y = deskew_shape_estimator(
        deskew_input_shape,
        theta=opm_tilt_deg,
        distance=scan_axis_step_um,
        pixel_size=pixel_size_um,
        crop_after_deskew=crop_after_deskew,
    )

    if time_range is not None:
        time_shape = time_range[1]
    else:
        time_shape = datastore.shape[0]

    if pos_range is not None:
        pos_shape = pos_range[1]
    else:
        pos_shape = datastore.shape[1]

    datastore_shape = [
        time_shape,
        pos_shape,
        datastore.shape[2],
        deskewed_shape[0] // z_downsample_level,
        deskewed_shape[1],
        deskewed_shape[2],
    ]
    # create array to hold one deskewed volume
    deskewed = np.zeros(
        (deskewed_shape[0] // z_downsample_level, deskewed_shape[1], deskewed_shape[2]),
        dtype=np.float32,
    )

    processing_metadata = {
        "scan_axis_step_um": scan_axis_step_um,
        "raw_pixel_size_um": pixel_size_um,
        "opm_tilt_deg": opm_tilt_deg,
        "camera_corrected": True,
        "camera_offset": camera_offset,
        "camera_e_to_ADU": camera_conversion,
        "deskewed_voxel_size_um": [
            z_downsample_level * pixel_size_um,
            pixel_size_um,
            pixel_size_um,
        ],
        "deskew_input_shape_zyx": list(deskew_input_shape),
        "stage_x_flipped": stage_x_flipped,
        "stage_y_flipped": stage_y_flipped,
        "stage_z_flipped": stage_z_flipped,
        "flatfield_corrected": flatfield_correction,
        "pad_y": pad_y,
        "pad_x": pad_x,
    }

    if not (deconvolve):
        output_path = output_dir / Path(
            acquisition_stem(root_path) + "_deskewed.ome.zarr"
        )
    else:
        output_path = output_dir / Path(
            acquisition_stem(root_path) + "_decon_deskewed.ome.zarr"
        )
    flatfield_path = (
        output_dir
        / Path("flatfield")
        / Path(acquisition_stem(root_path) + "_flatfield.ome.tif")
    )
    output_kind = "deconvolved_deskewed" if deconvolve else "deskewed"
    processing_metadata["opm_processing"] = _processing_provenance(
        root_path,
        output_kind,
        [
            _selection_step(time_range, pos_range),
            {
                "name": "camera_correction",
                "applied": True,
                "parameters": {
                    "offset": camera_offset,
                    "e_to_ADU": camera_conversion,
                    "formula": "(raw - offset) * e_to_ADU",
                },
            },
            {
                "name": "illumination_correction",
                "applied": flatfield_correction,
                "parameters": {
                    "estimator": "BaSiCPy",
                    "estimator_version": _distribution_version("basicpy"),
                    "configuration": "autotuned_then_smoothness_2_with_residual_calibration",
                    "sort_intensity": True,
                    "smoothness_flatfield": 2.0,
                    "sampling": "per_position_summaries_across_all_positions",
                    "planes_per_position": 10,
                    "fit_summary": "median",
                    "residual_summary": "75th_percentile",
                    "residual_calibration": "smoothed_separable_yx",
                    "working_size": "half_resolution_preserving_aspect_ratio",
                    "darkfield_estimation": False,
                    "artifact_schema": _flatfield_software_tag(),
                    "artifact": str(flatfield_path.resolve()),
                },
            },
            {
                "name": "deconvolution",
                "applied": deconvolve,
                "parameters": {
                    "method": "RLGC",
                    "crop_scan": (
                        "auto" if decon_crop_scan is None else int(decon_crop_scan)
                    ),
                    "gpu_id": int(decon_gpu_id),
                    "verbose": int(decon_verbose),
                    "fallback_step_scan": (
                        "adaptive"
                        if decon_fallback_step_scan is None
                        else int(decon_fallback_step_scan)
                    ),
                    "psf_source": "theoretical",
                },
            },
            {
                "name": "deskew",
                "applied": True,
                "parameters": {
                    "theta_deg": opm_tilt_deg,
                    "scan_axis_step_um": scan_axis_step_um,
                    "raw_pixel_size_um": pixel_size_um,
                    "z_downsample_level": int(z_downsample_level),
                },
            },
            {
                "name": "crop_after_deskew",
                "applied": crop_after_deskew,
                "parameters": {"crop_y_pixels_per_side": int(crop_y)},
            },
            _output_conversion_step(save_float32),
        ],
    )
    output_collection = create_position_collection(
        output_path,
        datastore_shape,
        processing_metadata["deskewed_voxel_size_um"],
        stage_positions=stage_positions[:pos_shape],
        channels=[str(value) for value in em_values],
        attributes=processing_metadata,
        dtype=output_dtype,
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
        max_z_metadata = _derived_processing_metadata(
            processing_metadata,
            f"max_z_{output_kind}",
            {
                "name": "maximum_projection",
                "applied": True,
                "parameters": {"axis": "z"},
            },
        )
        max_z_metadata["deskewed_voxel_size_um"] = [
            pixel_size_um,
            pixel_size_um,
            pixel_size_um,
        ]
        max_z_collection = create_position_collection(
            max_z_output_path,
            max_z_datastore_shape,
            max_z_metadata["deskewed_voxel_size_um"],
            stage_positions=stage_positions[:pos_shape],
            channels=[str(value) for value in em_values],
            attributes=max_z_metadata,
            dtype=output_dtype,
        )
        max_z_ts_store = max_z_collection.arrays

    if flatfield_correction:
        flatfield_dir = output_dir / Path("flatfield")
        if not (flatfield_dir.exists()):
            flatfield_dir.mkdir()
        flatfields = _load_or_estimate_flatfield(
            flatfield_path,
            datastore,
            camera_offset,
            camera_conversion,
            pixel_size_um,
        )
    else:
        flatfields = np.ones(
            (datastore.shape[2], datastore.shape[-2], datastore.shape[-1]),
            dtype=np.float32,
        )

    # loop over all components and stream to zarr using tensorstore
    ts_writes = []
    if max_projection:
        ts_max_writes = []

    if time_range is not None:
        time_iterator = tqdm(range(time_range[0], time_range[1]), desc="t")
    else:
        time_iterator = tqdm(range(datastore.shape[0]), desc="t")

    if pos_range is not None:
        pos_iterator = tqdm(range(pos_range[0], pos_range[1]), desc="p", leave=False)
    else:
        pos_iterator = tqdm(range(datastore.shape[1]), desc="p", leave=False)

    for t_idx in time_iterator:
        for pos_idx in pos_iterator:
            for chan_idx in tqdm(range(datastore.shape[2]), desc="c", leave=False):
                camera_corrected_data = _camera_corrected_image(
                    np.squeeze(datastore[t_idx, pos_idx, chan_idx, :]),
                    camera_offset,
                    camera_conversion,
                    flatfields[chan_idx, :],
                )
                if "stage" in opm_mode:
                    flip_scan = True
                else:
                    flip_scan = False

                if flip_scan:
                    camera_corrected_data = np.flip(camera_corrected_data, axis=0)

                if deconvolve:
                    if pos_idx == 0 and chan_idx == 0:
                        psfs = []
                        for psf_idx in range(datastore.shape[2]):
                            psf = ASI_generate_skewed_psf(
                                em_wvl=em_values[psf_idx],
                                pixel_size_um=pixel_size_um,
                                scan_axis_step_um=scan_axis_step_um,
                                theta_deg=opm_tilt_deg,
                                pz=0.0,
                                plot=False,
                            )
                            psfs.append(psf)

                    effective_crop_scan = decon_chunk_state.determine_once(
                        tuple(int(size) for size in camera_corrected_data.shape),
                        [tuple(int(size) for size in psf.shape) for psf in psfs],
                        gpu_id=decon_gpu_id,
                    )
                    deconvolved_data = chunked_rlgc(
                        camera_corrected_data,
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
                        distance=scan_axis_step_um,
                        pixel_size=pixel_size_um,
                        downsample_factor=z_downsample_level,
                        output_dtype=np.float32,
                    )
                else:
                    deskewed = orthogonal_deskew(
                        camera_corrected_data,
                        theta=opm_tilt_deg,
                        distance=scan_axis_step_um,
                        pixel_size=pixel_size_um,
                        downsample_factor=z_downsample_level,
                        output_dtype=np.float32,
                    )

                if crop_after_deskew:
                    deskewed = deskewed[:, crop_y:-crop_y, :]

                if max_projection:
                    max_z_deskewed = np.max(deskewed, axis=0, keepdims=True)
                    # create future objects for async data writing
                    ts_max_writes.append(
                        max_z_ts_store[pos_idx][t_idx, chan_idx].write(
                            _format_processed_output(max_z_deskewed, save_float32)
                        )
                    )

                # create future objects for async data writing
                ts_writes.append(
                    ts_store[pos_idx][t_idx, chan_idx].write(
                        _format_processed_output(deskewed, save_float32)
                    )
                )

    # wait for writes to finish
    for ts_write in ts_writes:
        ts_write.result()

    if max_projection:
        for ts_max_write in ts_max_writes:
            ts_max_write.result()

    del deskewed, ts_write, ts_store
    if max_projection:
        del max_z_deskewed, ts_max_write

    if create_fused_max_projection:
        if deconvolve:
            max_z_output_path = output_dir / Path(
                acquisition_stem(root_path) + "_max_z_decon_deskewed.ome.zarr"
            )
        else:
            max_z_output_path = output_dir / Path(
                acquisition_stem(root_path) + "_max_z_deskewed.ome.zarr"
            )
        max_z_ts_store = open_position_collection(max_z_output_path).arrays

        print("\nFusing max projection using stage positions...")
        fused_output_path = output_dir / Path(
            acquisition_stem(root_path) + "_max_z_fused.ome.zarr"
        )

        if pos_range is not None:
            tile_positions = stage_positions[pos_range[0] : pos_range[1]]

        else:
            tile_positions = stage_positions

        # apply max-tile-fusion
        tile_fusion = MaxTileFusion(
            ts_dataset=max_z_ts_store,
            tile_positions=tile_positions,
            output_path=fused_output_path,
            pixel_size=np.asarray((pixel_size_um, pixel_size_um), dtype=np.float32),
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
        if write_fused_tiff:
            tiff_dir_path = output_path.parent / Path("tiff_output")
            tiff_dir_path.mkdir(exist_ok=True)
            datastore = open_position_collection(output_path).arrays[0]
            for t_idx in tqdm(range(datastore.shape[0]), desc="t"):
                image = np.asarray(datastore[t_idx].read().result())
                image = np.swapaxes(image, 0, 1)
                filename = Path(f"fused_t{t_idx}.ome.tiff")
                filename_path = tiff_dir_path / Path(filename)
                axes = "ZCYX"

                with TiffWriter(filename_path, bigtiff=True) as tif:
                    metadata = {
                        "axes": axes,
                        "SignificantBits": 32 if save_float32 else 16,
                        "PhysicalSizeX": pixel_size_um,
                        "PhysicalSizeXUnit": "µm",
                        "PhysicalSizeY": pixel_size_um,
                        "PhysicalSizeYUnit": "µm",
                        "PhysicalSizeZ": z_downsample_level * pixel_size_um,
                    }
                    options = dict(
                        compression="zlib",
                        compressionargs={"level": 8},
                        predictor=True,
                        photometric="minisblack",
                        resolutionunit="CENTIMETER",
                    )
                    tif.write(
                        image,
                        resolution=(1e4 / pixel_size_um, 1e4 / pixel_size_um),
                        **options,
                        metadata=metadata,
                    )


def run_estimate_illuminations(datastore, camera_offset, camera_conversion, conn):
    """Run ``estimate_illuminations`` in a subprocess.

    BaSiCPy uses PyTorch on the GPU, so the fit runs in an isolated process to
    ensure all GPU allocations are released before other processing begins.

    Parameters
    ----------
    datastore: TensorStore
        TensorStore object containing the data.
    camera_offset: float
        Camera offset value.
    camera_conversion: float
        Camera conversion value.
    conn: Pipe
        Pipe connection to send the result back to the main process.

    Returns
    -------
    None
        No value is returned.
    """
    from opm_processing.cuda import preload_cuda_libraries

    preload_cuda_libraries()

    from opm_processing.imageprocessing.flatfield import estimate_illuminations

    try:
        flatfields = estimate_illuminations(
            datastore,
            camera_offset,
            camera_conversion,
        )
        conn.send(flatfields)
    except Exception as e:
        conn.send(e)
    finally:
        conn.close()


def call_estimate_illuminations(datastore, camera_offset, camera_conversion):
    """Call ``estimate_illuminations`` in an isolated subprocess.

    BaSiCPy uses PyTorch on the GPU, so the fit runs in an isolated process to
    ensure all GPU allocations are released before other processing begins.

    Parameters
    ----------
    datastore: TensorStore
        TensorStore object containing the data.
    camera_offset: float
        Camera offset value.
    camera_conversion: float
        Camera conversion value.

    Returns
    -------
    flatfields: np.ndarray
        Estimated illuminations.
    """
    parent_conn, child_conn = mp.Pipe()
    p = mp.Process(
        target=run_estimate_illuminations,
        args=(
            datastore,
            camera_offset,
            camera_conversion,
            child_conn,
        ),
    )
    p.start()
    result = parent_conn.recv()
    p.join()

    if p.exitcode != 0:
        raise RuntimeError("Subprocess failed")

    if isinstance(result, Exception):
        raise result

    return result


# entry for point for CLI
def main():
    """Run the OPM processing command-line application.

    Parameters
    ----------
    None
        This callable has no parameters.

    Returns
    -------
    None
        No value is returned.
    """
    app()


if __name__ == "__main__":
    main()
