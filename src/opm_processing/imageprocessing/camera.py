"""Camera-coordinate artifact correction."""

import numpy as np


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
    """Return the fixed detector-X gain measured for the qi2lab stage camera."""
    gain = np.ones(QI2LAB_STAGE_SCAN_DETECTOR_WIDTH, dtype=np.float32)
    stop = QI2LAB_STAGE_SCAN_GAIN_START_X + QI2LAB_STAGE_SCAN_GAIN_VALUES.size
    gain[QI2LAB_STAGE_SCAN_GAIN_START_X:stop] = QI2LAB_STAGE_SCAN_GAIN_VALUES
    return gain


def correct_qi2lab_stage_scan_camera(
    images: np.ndarray,
    *,
    x_offset: int = 0,
    copy: bool = True,
) -> np.ndarray:
    """Divide by the fixed qi2lab stage-scan gain, broadcasting over scan/Y."""
    corrected = np.asarray(images, dtype=np.float32)
    if copy:
        corrected = corrected.copy()
    start = int(x_offset)
    stop = start + int(corrected.shape[-1])
    if start < 0 or stop > QI2LAB_STAGE_SCAN_DETECTOR_WIDTH:
        raise ValueError(
            "The qi2lab stage-scan gain requires detector-X coordinates within "
            f"0:{QI2LAB_STAGE_SCAN_DETECTOR_WIDTH}; received {start}:{stop}"
        )
    gain = qi2lab_stage_scan_camera_gain()[start:stop]
    corrected /= gain.reshape((1,) * (corrected.ndim - 1) + (gain.size,))
    return corrected
