"""Camera calibration and detector illumination correction."""

import numpy as np
from numba import njit, prange


@njit(inline="always")
def calibrated_pixel(
    raw: np.uint16, offset: np.float32, conversion: np.float32
) -> np.float32:
    """Apply the camera transfer function to one ADC count.

    Parameters
    ----------
    raw : np.uint16
        Digitized camera value.
    offset, conversion : np.float32
        Electronic offset in ADU and conversion per ADU.

    Returns
    -------
    np.float32
        Calibrated intensity, clipped at zero.
    """
    value = np.float32(raw) - offset
    value = np.float32(value * conversion)
    return np.maximum(value, np.float32(0))


@njit(parallel=True, error_model="numpy")
def camera_correct_rows(
    raw: np.ndarray, offset: np.float32, conversion: np.float32, gain: np.ndarray
) -> np.ndarray:
    """Correct independent camera rows in parallel.

    Parameters
    ----------
    raw : np.ndarray
        Three-dimensional uint16 camera data.
    offset, conversion : np.float32
        Offset in ADU and conversion per ADU.
    gain : np.ndarray
        Float32 detector-X gain, or an empty array to disable gain correction.

    Returns
    -------
    np.ndarray
        Newly allocated float32 calibrated volume.
    """
    output = np.empty(raw.shape, dtype=np.float32)
    _, ny, nx = raw.shape
    for row in prange(raw.shape[0] * ny):
        s, y = row // ny, row % ny
        if gain.size:
            for x in range(nx):
                output[s, y, x] = (
                    calibrated_pixel(raw[s, y, x], offset, conversion) / gain[x]
                )
        else:
            for x in range(nx):
                output[s, y, x] = calibrated_pixel(raw[s, y, x], offset, conversion)
    return output


def camera_correct(
    raw: np.ndarray,
    camera_offset: float,
    camera_conversion: float,
    *,
    detector_x_offset: int = 0,
    detector_gain_x: np.ndarray | None = None,
) -> np.ndarray:
    """Convert raw camera counts to nonnegative float32 intensities.

    Parameters
    ----------
    raw : np.ndarray
        Uint16 image or volume. Leading dimensions are independent images.
    camera_offset : float
        Electronic background in ADU.
    camera_conversion : float
        Calibrated intensity per ADU, including QE when required by the caller.
    detector_x_offset : int
        First detector column of a cropped acquisition.
    detector_gain_x : numpy.ndarray or None
        Recorded detector-X response to divide out after clipping. None disables it.

    Returns
    -------
    np.ndarray
        Calibrated float32 data with the input shape. The input is unchanged.
    """
    raw = np.asarray(raw)
    gain = np.empty(0, np.float32)
    if detector_gain_x is not None:
        profile = np.asarray(detector_gain_x, dtype=np.float32)
        start = int(detector_x_offset)
        stop = start + raw.shape[-1]
        if profile.ndim != 1 or start < 0 or stop > profile.size:
            raise ValueError("Detector-X crop is outside the recorded gain calibration")
        if not np.all(np.isfinite(profile) & (profile > 0)):
            raise ValueError("Recorded detector gain must be finite and positive")
        gain = profile[start:stop]
    offset, conversion = np.float32(camera_offset), np.float32(camera_conversion)
    if raw.ndim == 3:
        return camera_correct_rows(raw, offset, conversion, gain)
    if raw.ndim == 2:
        return camera_correct_rows(raw[None], offset, conversion, gain)[0]
    # Preserve NumPy broadcasting for less common input dimensionalities.
    calibrated = np.maximum(
        (raw.astype(np.float32) - offset) * conversion, np.float32(0)
    )
    if gain.size:
        calibrated /= gain
    return calibrated


@njit(parallel=True, error_model="numpy")
def illumination_correct_rows(
    calibrated: np.ndarray, illumination: np.ndarray
) -> np.ndarray:
    """Divide independent camera rows by a detector illumination profile.

    Parameters
    ----------
    calibrated : np.ndarray
        Three-dimensional float32 calibrated image.
    illumination : np.ndarray
        Two-dimensional float32 detector profile.

    Returns
    -------
    np.ndarray
        Corrected float32 volume in a separate allocation.
    """
    output = np.empty(calibrated.shape, dtype=np.float32)
    _, ny, nx = calibrated.shape
    for row in prange(calibrated.shape[0] * ny):
        s, y = row // ny, row % ny
        for x in range(nx):
            output[s, y, x] = calibrated[s, y, x] / illumination[y, x]
    return output


def illumination_correct(
    calibrated: np.ndarray, illumination: np.ndarray
) -> np.ndarray:
    """Apply illumination correction independently of camera calibration.

    Parameters
    ----------
    calibrated : np.ndarray
        Float32 image or volume after camera correction and empty-tile checks.
    illumination : np.ndarray
        Float32 detector profile broadcastable to the image shape.

    Returns
    -------
    np.ndarray
        Float32 intensities divided by the profile. Neither input is modified.
    """
    calibrated, illumination = np.asarray(calibrated), np.asarray(illumination)
    if illumination.shape == calibrated.shape[-2:]:
        if calibrated.ndim == 3:
            return illumination_correct_rows(calibrated, illumination)
        if calibrated.ndim == 2:
            return illumination_correct_rows(calibrated[None], illumination)[0]
    return calibrated / illumination
