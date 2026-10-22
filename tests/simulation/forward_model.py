"""One fluorescence-to-camera model for test fixtures and synthetic comparisons.

For fine-grid emitted-photon density f, excitation L and detection PSF h,
lambda = QE * exposure * integral_pixel[(f L) * h] + background.
Convolution includes the fine voxel volume; pixel integration includes sensor
area. Detection is Poisson(lambda), followed by electron read noise, gain,
offset and unsigned ADC quantization. The PSF has unit focal-plane integral
times the declared collection efficiency, never a gain fitted to an image.
An optional camera-plane photon budget chooses source and background emission
doses from deterministic unit-dose responses before QE and camera sampling.
It never rescales a camera realization or a reconstruction.

Oblique coordinates are X=c, Y=s+r*cos(theta), Z=r*sin(theta). Cartesian
comparison stacks use ZYX coordinates and XY sensor pixels. Neither route
resamples the object before optical convolution. This shift-invariant model
assumes static excitation during a scan; it does not simulate a moving sheet.
Optical generation and pixel quadrature use production opmpsf functions. The
inverse kernel samples the same cached optical field used to blur ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.signal import fftconvolve

from opm_processing.imageprocessing.opmpsf import (
    create_psf_silicone_100x,
    integrate_camera_pixels,
)
from tests.simulation.camera import Camera, digitize
from tests.simulation.objects import centered_axis

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class Microscope:
    """Optics, detector geometry and camera calibration; lengths are micrometers."""

    fine_spacing_um: float
    pixel_size_um: float
    scan_step_um: float
    angle_deg: float
    wavelength_um: float
    numerical_aperture: float
    detection_efficiency: float
    quantum_efficiency: float
    pixel_samples: int
    psf_half_extent_zyx_um: tuple[float, float, float]
    camera: Camera
    illumination_edge_fraction: float
    scan_endpoint_divisor: int
    psf_parameters: tuple[tuple[str, float], ...]
    psf_depth_um: float

    def __post_init__(self) -> None:
        """Keep explicitly supplied PSF calibration immutable for optical caching."""
        object.__setattr__(
            self,
            "psf_parameters",
            tuple((str(k), float(v)) for k, v in self.psf_parameters),
        )


@dataclass(frozen=True)
class Acquisition:
    """Physical GT axes and exposure metadata for one acquisition.

    GT is emitted photons/exposure/um^3 on uniform isotropic Cartesian ZYX
    axes. Sensor axes are SYX for oblique scans and ZYX for Cartesian stacks;
    explicit axes allow sparse scans, tiles and shifted origins. With only a
    shape, centered axes use the microscope's scan step and pixel pitch.
    ``expectation`` returns electron means; ``irradiance`` returns the fine
    pre-sensor optical field for plotting. ``object`` returns the illuminated
    emission density after choosing the source/background doses and exposure,
    before optical blur or QE. Default output is unsigned raw ADU.

    Optional ``camera_photon_budget=(source_peak, background_center)`` specifies
    incident photons/pixel/exposure, before QE. In this mode GT specifies source
    geometry. Its emission dose and a uniform fluorescent-background density
    are chosen using their noiseless optical responses, before detection. The
    background traverses the same illumination, PSF and sensor integration.
    No noisy image is rescaled. Exposure multiplies both nominal photon doses.
    """

    gt_axes: tuple[np.ndarray, ...]
    shape: tuple[int, int, int] | None = None
    sensor_axes: tuple[np.ndarray, ...] | None = None
    geometry: Literal["oblique", "cartesian"] = "oblique"
    illumination_profile: str = "uniform"
    illumination_half_width_um: float | None = None
    exposure: float = 1.0
    background_electrons: float = 0.0
    seed: int = 42
    output: Literal["camera", "expectation", "irradiance", "object"] = "camera"
    camera_photon_budget: tuple[float, float] | None = None


def excitation_field(
    x_um: np.ndarray,
    half_width_um: float,
    profile: str = "uniform",
    *,
    edge_fraction: float,
) -> np.ndarray:
    """Positive excitation with the configured taper at both camera-X ends.

    Parameters
    ----------
    x_um : numpy.ndarray
        Camera-X coordinates in micrometers.
    half_width_um : float
        Physical half-width of the excitation field.
    profile : str
        Uniform excitation or the configured edge-taper profile.
    edge_fraction : float
        Positive excitation fraction at either long-axis endpoint.

    Returns
    -------
    numpy.ndarray
        Dimensionless float32 illumination at the supplied coordinates.
    """
    if profile == "uniform":
        return np.ones_like(x_um, dtype=np.float32)
    if profile != "edge-taper" or half_width_um <= 0:
        raise ValueError("Require uniform or edge-taper and a positive image width")
    coordinate = np.clip(x_um / half_width_um, -1, 1)
    if not 0 < edge_fraction <= 1:
        raise ValueError("Illumination edge fraction must be in (0,1]")
    return (
        edge_fraction + (1 - edge_fraction) * np.cos(np.pi * coordinate / 2) ** 2
    ).astype(np.float32)


@lru_cache(maxsize=4)
def emission_psf(
    microscope: Microscope, spacing: float
) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    """Finite vectorial detection density, normalized by its focal-plane area.

    Parameters
    ----------
    microscope : Microscope
        Explicit optical calibration and collection efficiency.
    spacing : float
        Cartesian integration-grid spacing in micrometers.

    Returns
    -------
    tuple[tuple of numpy.ndarray, numpy.ndarray]
        Physical ZYX axes and vectorial detection density.
    """
    extent = np.asarray(microscope.psf_half_extent_zyx_um, dtype=float)
    optics = dict(microscope.psf_parameters)
    if not {"ni0", "ni", "ns", "tg0", "tg", "ti0"} <= optics.keys():
        raise ValueError(
            "Supply immersion/sample indices and coverglass/objective calibration"
        )
    if (
        not np.isfinite(spacing)
        or spacing <= 0
        or extent.shape != (3,)
        or not np.isfinite(extent).all()
        or np.any(extent <= 0)
        or extent[1] != extent[2]
    ):
        raise ValueError("Require positive spacing and ZYX extents with equal XY")
    if (
        not np.isfinite(microscope.wavelength_um)
        or microscope.wavelength_um <= 0
        or not 0 < microscope.numerical_aperture <= optics["ni0"]
    ):
        raise ValueError(
            "Require a positive wavelength and NA at most the supplied immersion RI"
        )
    axes = tuple(centered_axis(e, spacing) for e in extent)
    psf = np.asarray(
        create_psf_silicone_100x(
            spacing,
            spacing,
            axes[2].size,
            axes[0].size,
            microscope.wavelength_um,
            microscope.psf_depth_um,
            parameters={
                "lens": {
                    **optics,
                    "NA": microscope.numerical_aperture,
                    "sf": 1,
                    "mode": 1,
                },
                "taper_sigma_px_3d": np.inf,
            },
        ),
        dtype=np.float64,
    )
    psf /= psf.sum()
    psf *= microscope.detection_efficiency / (psf[len(axes[0]) // 2].sum() * spacing**2)
    psf = psf.astype(np.float32)
    psf.setflags(write=False)
    return axes, psf


def sensor_integral(
    optical_field: np.ndarray,
    axes: Sequence[np.ndarray],
    shape: Sequence[int] | None,
    microscope: Microscope,
    *,
    area_integral: bool = True,
    sensor_axes: Sequence[np.ndarray] | None = None,
    geometry: str = "oblique",
) -> np.ndarray:
    """Pixel quadrature primitive, also used for analytic sensor unit checks.

    Parameters
    ----------
    optical_field : numpy.ndarray
        Blurred Cartesian emission density.
    axes : sequence of numpy.ndarray
        Physical Cartesian ZYX grid axes in micrometers.
    shape : sequence of int or None
        Requested detector-grid dimensions; None uses sensor_axes.
    microscope : Microscope
        Pixel pitch, scan geometry and quadrature settings.
    area_integral : bool
        Integrate pixel area when True; otherwise evaluate sampled intensity.
    sensor_axes : sequence of numpy.ndarray or None
        Explicit detector coordinates for analytic sampling checks.
    geometry : str
        Oblique acquisition or Cartesian detector sampling.

    Returns
    -------
    numpy.ndarray
        Sensor-integrated intensities on the declared acquisition grid.
    """
    p = microscope
    if not isinstance(p.pixel_samples, int) or p.pixel_samples < 1:
        raise ValueError("Pixel quadrature samples must be positive integers")
    if geometry not in ("oblique", "cartesian"):
        raise ValueError("Require oblique or cartesian sensor geometry")
    if sensor_axes is None:
        if (
            shape is None
            or len(shape) != 3
            or any(
                isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1
                for n in shape
            )
        ):
            raise ValueError(
                "Require three positive sensor dimensions or explicit axes"
            )
        sensor_axes = tuple(
            (np.arange(n) - (n - 1) / 2) * step
            for n, step in zip(
                shape, (p.scan_step_um, p.pixel_size_um, p.pixel_size_um), strict=True
            )
        )
    if len(sensor_axes) != 3 or any(
        np.ndim(a) != 1 or not len(a) or not np.isfinite(a).all() for a in sensor_axes
    ):
        raise ValueError("Sensor axes must be three nonempty finite coordinate arrays")
    if shape is not None and tuple(shape) != tuple(map(len, sensor_axes)):
        raise ValueError("Sensor shape and coordinate axes disagree")
    return integrate_camera_pixels(
        optical_field,
        axes,
        sensor_axes,
        pixel_size_um=p.pixel_size_um,
        theta_deg=p.angle_deg,
        samples=p.pixel_samples,
        geometry=geometry,
        area_integral=area_integral,
    )


def camera_psf(microscope: Microscope) -> tuple[np.ndarray, float]:
    """Integrate the simulation's production optical PSF onto the camera grid.

    Parameters
    ----------
    microscope : Microscope
        Explicit optics, camera pixel pitch and scan sampling.

    Returns
    -------
    tuple[numpy.ndarray, float]
        Unit-mass oblique kernel and its physical integrated detection factor.
    """
    p = microscope
    axes, psf = emission_psf(p, p.fine_spacing_um)
    # Full optical convolution interpolates the final PSF sample towards the
    # next, zero-valued fine-grid sample. Extend the kernel identically before
    # pixel quadrature; filling out-of-bounds queries with zero would instead
    # cut off that interpolation interval and change the camera response.
    psf = np.pad(psf, 1)
    axes = tuple(
        np.r_[a[0] - p.fine_spacing_um, a, a[-1] + p.fine_spacing_um] for a in axes
    )
    theta = np.deg2rad(p.angle_deg)
    hz, hy, hx = (a[-1] for a in axes)
    steps = (p.scan_step_um, p.pixel_size_um, p.pixel_size_um)
    shape = tuple(
        len(centered_axis(e, d))
        for e, d in zip(
            (
                hy + hz / np.tan(theta),
                hz / np.sin(theta) + p.pixel_size_um / 2,
                hx + p.pixel_size_um / 2,
            ),
            steps,
            strict=True,
        )
    )
    sensor_axes = tuple(
        (np.arange(n) - (n - 1) / 2) * step
        for n, step in zip(shape, steps, strict=True)
    )
    probabilities = (
        sensor_integral(psf, axes, shape, p, sensor_axes=sensor_axes)
        * p.quantum_efficiency
    )
    total = probabilities.sum(dtype=np.float64)
    volume = np.prod(steps) * np.sin(theta)
    return probabilities / total, float(total * volume)


def forward_model(
    ground_truth: np.ndarray, metadata: Acquisition, microscope: Microscope
) -> np.ndarray:
    """Return simulated data from GT, acquisition metadata and microscope setup.

    Parameters
    ----------
    ground_truth : numpy.ndarray
        Nonnegative emitted-photon density on the fine Cartesian ZYX grid.
    metadata : Acquisition
        GT axes, detector sampling, exposure, background and requested output.
    microscope : Microscope
        Explicit optics, oblique geometry, illumination, QE and ADC calibration.

    Returns
    -------
    numpy.ndarray
        Unsigned digitized camera data, or the requested noiseless intermediate.

    Raises
    ------
    ValueError
        Grid, density, exposure or detector calibration is nonphysical.
    """
    axes = tuple(np.asarray(a, dtype=float) for a in metadata.gt_axes)
    if len(axes) != 3 or ground_truth.shape != tuple(map(len, axes)):
        raise ValueError("GT must match three Cartesian ZYX coordinate axes")
    if any(a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all() for a in axes):
        raise ValueError("GT axes must contain at least two finite coordinates")
    # Use the declared grid pitch for both the simulated optical field and its
    # inverse kernel. Cropping or translating GT must not alter PSF support
    # through floating-point cancellation in inferred coordinate differences.
    spacing = microscope.fine_spacing_um
    if spacing <= 0 or any(
        not np.allclose(np.diff(a), spacing, rtol=1e-8, atol=1e-12) for a in axes
    ):
        raise ValueError("GT requires a uniform isotropic fine Cartesian grid")
    if not np.isfinite(ground_truth).all() or np.any(ground_truth < 0):
        raise ValueError("GT photon density must be finite and nonnegative")
    values = (metadata.exposure, metadata.background_electrons)
    if not np.isfinite(values).all() or min(values) < 0:
        raise ValueError("Exposure and background must be finite and nonnegative")
    if (
        not 0 <= microscope.quantum_efficiency <= 1
        or not 0 < microscope.detection_efficiency <= 1
    ):
        raise ValueError("QE and collection efficiency must be probabilities")
    if (
        not np.isfinite(
            (microscope.angle_deg, microscope.pixel_size_um, microscope.scan_step_um)
        ).all()
        or not 0 < microscope.angle_deg < 90
        or min(microscope.pixel_size_um, microscope.scan_step_um) <= 0
        or spacing >= microscope.pixel_size_um
    ):
        raise ValueError(
            "Require positive sampling, a finer GT grid and an angle between 0 and 90"
        )
    if metadata.output not in ("camera", "expectation", "irradiance", "object"):
        raise ValueError("Require camera, expectation, irradiance or object output")
    if metadata.output == "object" and metadata.background_electrons:
        raise ValueError(
            "An additive detector background has no emission-density truth"
        )
    budget = metadata.camera_photon_budget
    if budget is not None and (
        len(budget) != 2
        or not np.isfinite(budget).all()
        or min(budget) < 0
        or metadata.output == "irradiance"
    ):
        raise ValueError(
            "Require two nonnegative camera photon doses and sensor output"
        )
    # Excitation belongs to this individual camera image, including its stage
    # origin; a translated tile must not inherit the laboratory-zero taper.
    center_x = 0.0
    half_width = metadata.illumination_half_width_um
    if metadata.sensor_axes is not None:
        columns = np.asarray(metadata.sensor_axes[2])
        center_x = float((columns[0] + columns[-1]) / 2)
        if half_width is None:
            half_width = max(
                float(np.max(np.abs(columns - center_x))), microscope.pixel_size_um / 2
            )
    elif metadata.shape is not None and half_width is None:
        half_width = max(
            (metadata.shape[2] - 1) * microscope.pixel_size_um / 2,
            microscope.pixel_size_um / 2,
        )
    if half_width is None:
        half_width = (
            np.ceil(max(abs(axes[2] - center_x)) / microscope.pixel_size_um)
            * microscope.pixel_size_um
        )
    illumination = excitation_field(
        axes[2][None, None, :] - center_x,
        half_width,
        metadata.illumination_profile,
        edge_fraction=microscope.illumination_edge_fraction,
    )
    psf_axes, psf = emission_psf(microscope, spacing)
    # The object domain is not the camera field of view: light from a source
    # near a GT boundary still reaches pixels outside that boundary. Retain the
    # full convolution halo for acquisitions. Fine-field plotting explicitly
    # requests the original GT grid instead.
    incident = None
    object_density = (
        np.zeros_like(
            ground_truth, dtype=np.result_type(ground_truth.dtype, np.float32)
        )
        if metadata.output == "object"
        else None
    )
    components = (
        (ground_truth,)
        if budget is None
        else (ground_truth, np.ones_like(ground_truth))
    )
    for index, density in enumerate(components):
        irradiance = np.maximum(
            fftconvolve(
                density * illumination,
                psf,
                mode="same" if metadata.output == "irradiance" else "full",
            )
            * spacing**3,
            0,
        )
        if metadata.output == "irradiance":
            return irradiance * metadata.exposure
        optical_axes = tuple(
            source[0] + kernel[0] + np.arange(size) * spacing
            for source, kernel, size in zip(
                axes, psf_axes, irradiance.shape, strict=True
            )
        )
        response = sensor_integral(
            irradiance,
            optical_axes,
            metadata.shape,
            microscope,
            sensor_axes=metadata.sensor_axes,
            geometry=metadata.geometry,
        )
        dose = 1.0
        if budget is not None:
            # Select physical emission densities from the known optical model:
            # a = source_peak / max(H[L f]); b = background_center / H[L 1]_center.
            # These are input doses, shared by every reconstruction method.
            reference = (
                float(response.max())
                if index == 0
                else float(response[tuple(n // 2 for n in response.shape)])
            )
            if reference <= 0 and budget[index] > 0:
                raise ValueError(
                    "Camera photon budget requires a nonzero optical response"
                )
            dose = budget[index] / reference if reference > 0 else 0
            response *= dose
        if object_density is not None:
            object_density += density * illumination * (dose * metadata.exposure)
        if incident is None:
            incident = response
        else:
            incident += response
    if object_density is not None:
        return object_density
    expected = incident * (microscope.quantum_efficiency * metadata.exposure)
    expected += metadata.background_electrons
    if metadata.output == "expectation":
        return expected
    return digitize(expected, microscope.camera, seed=metadata.seed)
