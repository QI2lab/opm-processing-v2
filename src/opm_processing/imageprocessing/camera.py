"""Camera calibration and detector illumination correction."""

import numpy as np
from numba import njit, prange


QI2LAB_STAGE_SCAN_DETECTOR_WIDTH = 1900
QI2LAB_STAGE_SCAN_GAIN_START_X = 1046
QI2LAB_STAGE_SCAN_GAIN_VALUES = np.asarray(
    [
        0.998620,
        0.991085,
        0.987864,
        0.980038,
        0.972417,
        0.963074,
        0.952404,
        0.946026,
        0.936700,
        0.923635,
        0.912184,
        0.901147,
        0.886930,
        0.872415,
        0.864196,
        0.850803,
        0.839574,
        0.825066,
        0.812804,
        0.801067,
        0.790662,
        0.780457,
        0.769833,
        0.761351,
        0.753541,
        0.744542,
        0.738491,
        0.730549,
        0.729804,
        0.723986,
        0.720666,
        0.717408,
        0.718568,
        0.717388,
        0.720602,
        0.721850,
        0.725791,
        0.729051,
        0.732097,
        0.739335,
        0.748341,
        0.752445,
        0.762228,
        0.770891,
        0.781373,
        0.794978,
        0.811252,
        0.826110,
        0.844273,
        0.860149,
        0.879094,
        0.896383,
        0.912213,
        0.929968,
        0.947805,
        0.965706,
        0.978314,
        0.992173,
    ],
    dtype=np.float32,
)


def qi2lab_stage_scan_camera_gain() -> np.ndarray:
    """Return the fixed detector-X gain measured for the qi2lab stage camera.

    Returns
    -------
    np.ndarray
        Float32 detector-X response with unity outside the measured gain interval.
    """
    gain = np.ones(QI2LAB_STAGE_SCAN_DETECTOR_WIDTH, dtype=np.float32)
    stop = QI2LAB_STAGE_SCAN_GAIN_START_X + QI2LAB_STAGE_SCAN_GAIN_VALUES.size
    gain[QI2LAB_STAGE_SCAN_GAIN_START_X:stop] = QI2LAB_STAGE_SCAN_GAIN_VALUES
    return gain


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
    apply_stage_scan_gain: bool = False,
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
    apply_stage_scan_gain : bool
        Divide by the measured qi2lab detector-X response after clipping.

    Returns
    -------
    np.ndarray
        Calibrated float32 data with the input shape. The input is unchanged.
    """
    raw = np.asarray(raw)
    gain = np.empty(0, np.float32)
    if apply_stage_scan_gain:
        start = int(detector_x_offset)
        stop = start + raw.shape[-1]
        if start < 0 or stop > QI2LAB_STAGE_SCAN_DETECTOR_WIDTH:
            raise ValueError("Detector-X crop is outside the calibrated detector")
        gain = qi2lab_stage_scan_camera_gain()[start:stop]
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
