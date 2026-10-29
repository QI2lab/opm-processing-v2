"""Resolve deconvolution settings from acquisition metadata and OPM defaults."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from opm_processing.dataio.acquisition import AcquisitionMetadata


TARGET_SCAN_STEP_UM = 0.4


def scan_upsample_from_metadata(acquisition: AcquisitionMetadata) -> int:
    """Infer integer scan refinement toward the 0.4 micrometer target.

    Parameters
    ----------
    acquisition : AcquisitionMetadata
        Acquisition with recorded physical scan spacing.

    Returns
    -------
    int
        One for planar or already finer acquisitions, otherwise the acquired
        scan spacing divided by the target spacing.

    Raises
    ------
    ValueError
        Scan spacing is invalid or a coarser scan is not an integer multiple
        of the target. The acquired-plane solver requires an integer factor.
    """
    if acquisition.is_2d:
        return 1
    step = acquisition.scan_axis_step_um
    if step is None or not np.isfinite(step) or step <= 0:
        raise ValueError(
            "Automatic scan refinement requires a positive recorded scan_axis_step_um"
        )
    if step <= TARGET_SCAN_STEP_UM:
        return 1
    ratio = step / TARGET_SCAN_STEP_UM
    factor = round(ratio)
    if not np.isclose(ratio, factor, rtol=1e-6, atol=0):
        raise ValueError(
            f"Recorded scan step {step:g} um is not an integer multiple of the {TARGET_SCAN_STEP_UM:g} um reconstruction target"
        )
    return factor


def read_noise_from_metadata(acquisition: AcquisitionMetadata) -> float:
    """Return RMS noise in calibrated detector units; absent calibration means zero.

    ``opm_processing.deconvolution.read_noise`` contains ``rms`` and ``units``.
    ADU noise is multiplied by the same recorded conversion as camera data.
    Electron noise already has the units of the camera-corrected image.

    Parameters
    ----------
    acquisition : AcquisitionMetadata
        Inspected OME-Zarr acquisition and its recorded calibration attributes.

    Returns
    -------
    float
        Read-noise RMS in calibrated detector units, or zero when absent.

    Raises
    ------
    ValueError
        Recorded noise has invalid units or an unrepresentable RMS.
    """
    calibration = (acquisition.deconvolution or {}).get("read_noise")
    if calibration is None:
        return 0.0
    try:
        value = float(calibration["rms"])
        units = calibration["units"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Recorded read noise requires numeric rms and explicit units"
        ) from exc
    if units == "ADU":
        conversion = acquisition.camera_conversion
        if conversion is None or not np.isfinite(conversion) or conversion <= 0:
            raise ValueError(
                "ADU read noise requires a positive recorded camera conversion"
            )
        value *= conversion
    elif units not in ("electrons", "calibrated-detector-counts"):
        raise ValueError(f"Unsupported read-noise units: {units!r}")
    if not np.isfinite(value) or not 0 <= value <= np.sqrt(np.finfo(np.float32).max):
        raise ValueError(
            "Recorded read-noise RMS must be finite, nonnegative and float32-representable"
        )
    return value


def psf_parameters_from_metadata(acquisition: AcquisitionMetadata) -> dict:
    """Read theoretical PSF settings, defaulting to the qi2lab silicone objective.

    Parameters
    ----------
    acquisition : AcquisitionMetadata
        Inspected OME-Zarr acquisition and its recorded calibration attributes.

    Returns
    -------
    dict
        Recorded settings when supplied; otherwise the processing defaults for
        the 100x, NA 1.35 silicone objective. Each call returns fresh defaults.
    """
    parameters = (acquisition.deconvolution or {}).get("psf_parameters")
    if parameters is not None:
        return parameters
    return {
        "lens": {
            "NA": 1.35,
            "ni0": 1.4,
            "ni": 1.4,
            "ns": 1.38,
            "ng0": 1.515,
            "ng": 1.515,
            "tg0": 170.0,
            "tg": 170.0,
            "ti0": 300.0,
        },
        "skewed_depth_um": 0.0,
        "projection_depth_um": 15.0,
        "projection_nz": 20,
        "projection_dz_um": 0.115,
        "projection_nxy": 71,
        "taper_sigma_px_3d": 10.0,
        "taper_sigma_px_2d": 15.0,
        "depth_scale_um": 15.0,
        "depth_width_scale": 5.0,
        "support_fwhms": 12.0,
        "grid_oversampling": 2,
    }


def detector_gain_from_metadata(
    acquisition: AcquisitionMetadata, detector_width: int
) -> np.ndarray | None:
    """Read an optional measured detector-X response; never infer it from width.

    Parameters
    ----------
    acquisition : AcquisitionMetadata
        Inspected OME-Zarr acquisition and its recorded calibration attributes.
    detector_width : int
        Number of columns in the stored camera image.

    Returns
    -------
    numpy.ndarray or None
        Positive float32 detector-X response, or None when absent.

    Raises
    ------
    ValueError
        The recorded response has invalid values or a different width.
    """
    values = (acquisition.deconvolution or {}).get("detector_gain_x")
    if values is None:
        return None
    gain = np.asarray(values, dtype=np.float32)
    if gain.shape != (detector_width,) or not np.all(np.isfinite(gain) & (gain > 0)):
        raise ValueError(
            "Recorded detector_gain_x must contain one positive finite value per column"
        )
    return gain


def emission_wavelengths_from_metadata(
    acquisition: AcquisitionMetadata,
) -> tuple[float, ...]:
    """Resolve channel wavelengths, preferring recorded emission calibration.

    As-acquired stores need no processing-specific attributes. In their absence,
    use the channel wavelength as the established PSF approximation. Combined
    labels such as ``561 + 637`` select the longest wavelength for that one
    stored camera channel.

    Parameters
    ----------
    acquisition : AcquisitionMetadata
        Inspected OME-Zarr acquisition and its recorded calibration attributes.

    Returns
    -------
    tuple of float
        Positive PSF wavelengths in micrometers, in channel order.

    Raises
    ------
    ValueError
        A channel wavelength is absent, nonfinite or nonpositive.
    """
    calibration = acquisition.deconvolution or {}
    if "emission_wavelengths_nm" in calibration:
        values = calibration["emission_wavelengths_nm"]
    else:
        values = []
        for channel in acquisition.channels:
            parts = [
                re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(?:nm)?\s*", part, re.IGNORECASE)
                for part in channel.name.split("+")
            ]
            if all(parts):
                values.append(max(float(part[1]) for part in parts))
            elif channel.wavelength_nm is not None:
                values.append(channel.wavelength_nm)
            else:
                raise ValueError(
                    f"Cannot resolve a PSF wavelength for channel {channel.name!r}. "
                    "Supply emission_wavelengths_nm in opm_processing.deconvolution "
                    "or measured --decon-psf-paths."
                )
    try:
        wavelengths = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Emission wavelengths must be numeric nanometer values"
        ) from exc
    if wavelengths.shape != (len(acquisition.channels),):
        raise ValueError(
            "emission_wavelengths_nm must contain one value per stored channel"
        )
    if not np.all(np.isfinite(wavelengths) & (wavelengths > 0)):
        raise ValueError("Recorded emission wavelengths must be finite and positive")
    return tuple(float(value) / 1000 for value in wavelengths)


def load_deconvolution_psfs(
    acquisition: AcquisitionMetadata,
    measured_paths: Sequence[str | Path] | None = None,
    *,
    projection: bool = False,
    reconstruction_step_um: float | None = None,
) -> tuple[list[np.ndarray], dict | None]:
    """Load measured kernels or generate them from resolved optical settings.

    Parameters
    ----------
    acquisition : AcquisitionMetadata
        Inspected acquisition with channel order and physical camera calibration.
    measured_paths : sequence of str or pathlib.Path or None
        One NumPy or TIFF PSF per channel. None selects theoretical production PSFs.
    projection : bool
        Generate planar kernels when True, oblique scan kernels otherwise.
    reconstruction_step_um : float or None
        Reconstruction scan spacing. None uses the recorded acquisition step.

    Returns
    -------
    tuple[list[numpy.ndarray], dict or None]
        Channel-ordered kernels and the optical parameters used to generate them.
        Measured kernels return None for the theoretical parameter dictionary.

    Raises
    ------
    ValueError
        Measured PSF count differs from channel count or a wavelength is unresolved.
    """
    from tqdm import tqdm

    if measured_paths is not None:
        if len(measured_paths) != len(acquisition.channels):
            raise ValueError("decon_psf_paths must contain one PSF per channel")
        from tifffile import imread

        return [
            np.load(path) if Path(path).suffix == ".npy" else np.asarray(imread(path))
            for path in tqdm(measured_paths, desc="PSFs", unit="PSF")
        ], None
    from opm_processing.imageprocessing.opmpsf import (
        generate_proj_psf,
        generate_skewed_psf,
    )

    parameters = psf_parameters_from_metadata(acquisition)
    settings = {"pixel_size_um": acquisition.pixel_size_um, "parameters": parameters}
    generate = generate_proj_psf if projection else generate_skewed_psf
    if not projection:
        settings.update(
            scan_axis_step_um=acquisition.scan_axis_step_um
            if reconstruction_step_um is None
            else reconstruction_step_um,
            theta_deg=acquisition.angle_deg,
        )
    kernels = [
        generate(em_wvl=wavelength, **settings)
        for wavelength in tqdm(
            emission_wavelengths_from_metadata(acquisition), desc="PSFs", unit="PSF"
        )
    ]
    return kernels, parameters
