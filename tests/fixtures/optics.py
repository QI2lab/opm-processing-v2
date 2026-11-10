"""Physical specimens and shared production optics for reconstruction."""

import json
from dataclasses import dataclass, replace
from functools import lru_cache, partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from opm_processing.imageprocessing.opmpsf import centered_sensor_axes
from opm_processing.imageprocessing.opmtools import (
    deskew_coverage,
    deskew_shape_estimator,
)
from tests.simulation.forward_model import (
    Acquisition,
    Microscope,
    camera_psf,
    emission_psf,
    forward_model,
)
from tests.simulation.objects import deposit_point
from tests.simulation.scenes import OpticalScene, freeze_metadata


@dataclass(frozen=True)
class CameraObservation:
    """One immutable optical scene and its independent detector observation."""

    ground_truth: OpticalScene
    microscope: Microscope
    expectation: np.ndarray
    raw: np.ndarray


@dataclass(frozen=True)
class AnalyticalCameraField:
    """Known analytical concentration on an immutable physical Cartesian grid."""

    axes: tuple[np.ndarray, ...]
    field: np.ndarray
    shape: tuple[int, int, int]
    center: tuple[float, float, float]


@pytest.fixture(scope="session")
def simulation_parameters():
    """Load the shared physical acquisition and specimen configuration."""
    return freeze_metadata(
        json.loads(
            (
                Path(__file__).resolve().parent / "configs/deconvolution_figures.json"
            ).read_text()
        )
    )


@pytest.fixture(scope="session")
def camera_calibration(simulation_parameters):
    """Provide the camera calibration declared in the shared configuration."""
    from tests.simulation.camera import Camera

    return Camera(**simulation_parameters["microscope"]["camera"])


@pytest.fixture(scope="session")
def microscope_parameters(simulation_parameters, camera_calibration):
    """Provide immutable microscope settings from the shared acquisition calibration."""
    options = dict(simulation_parameters["microscope"], camera=camera_calibration)
    options["psf_half_extent_zyx_um"] = tuple(options["psf_half_extent_zyx_um"])
    return Microscope(**options)


@pytest.fixture(scope="session")
def independent_photon_scene_factory(microscope_parameters, simulation_parameters):
    """Cache independent fine-grid scenes without retaining an entire parameter sweep."""
    from tests.simulation.scenes import prepare_scene

    @lru_cache(maxsize=2)
    def build(kind="points", *, large=False, integration_scan_planes=None):
        """Reuse physical scenes on the shared calibrated fine grid."""
        specimen = simulation_parameters["specimens"][kind]
        if large and kind == "line-pairs":
            z, _, x = specimen["half_extent_zyx_um"]
            length = simulation_parameters["large_scene"]["line_length_um"]
            features = dict(specimen["features"])
            if integration_scan_planes is not None:
                integration = simulation_parameters["integration_line_pairs"]
                z = max(
                    z,
                    simulation_parameters["specimens"]["points"]["half_extent_zyx_um"][
                        0
                    ],
                )
                if (
                    integration_scan_planes
                    == simulation_parameters["whole_scan_planes"]
                ):
                    # At feature Z=0, both endpoints and their full optical
                    # support fit even the coarse camera's common scan range.
                    length = integration["whole_line_length_um"]
                # Two finite cylinders straddle actual X tile borders; scan
                # subsampling cannot remove this independent border coverage.
                x = integration["half_extent_x_um"]
                features.update(
                    pair_gap_um=integration["pair_gap_um"],
                    offset_x_um=integration["offset_x_um"],
                )
            specimen = {
                "half_extent_zyx_um": (
                    z,
                    length / 2 + microscope_parameters.psf_half_extent_zyx_um[1],
                    x,
                ),
                "features": dict(features, length_um=length),
            }
        return prepare_scene(
            kind,
            microscope_parameters,
            parameters=specimen,
        )

    return build


@pytest.fixture(scope="session")
def simulated_camera(
    microscope_parameters, simulation_parameters, independent_photon_scene_factory
):
    """Cache native noisy camera fields and preserve their acquired scan coordinates.

    Returns
    -------
    callable
        Render an SYX acquisition using the single forward model. The result
        contains the object, optical calibration, camera means and unsigned ADU.
        Geometry, exposure and detector calibration are explicit inputs.
    """
    from tests.simulation.camera import digitize

    @lru_cache(maxsize=8)
    def optical_data(
        shape,
        *,
        kind="points",
        large=False,
        integration=False,
        illumination_profile="uniform",
        step=0.4,
        geometry="oblique",
        budget=None,
    ):
        """Reuse unit-exposure optics independently of electronic gain and noise."""
        scene = independent_photon_scene_factory(
            kind,
            large=large,
            integration_scan_planes=shape[0] if integration else None,
        )
        microscope = replace(microscope_parameters, scan_step_um=step)
        expectation = forward_model(
            scene.density,
            Acquisition(
                scene.axes,
                shape,
                geometry=geometry,
                illumination_profile=illumination_profile,
                illumination_half_width_um=scene.illumination_half_width_um,
                camera_photon_budget=(
                    tuple(simulation_parameters["photon_budgets"][0])
                    if budget is None
                    else budget
                ),
                output="expectation",
            ),
            microscope,
        )
        return scene, microscope, freeze_metadata(expectation)

    @lru_cache(maxsize=2)
    def render(
        shape=None,
        *,
        kind="points",
        large=False,
        integration=False,
        illumination_profile="uniform",
        step=0.4,
        conversion=0.25,
        geometry="oblique",
        exposure=1.0,
        seed=42,
        budget=None,
    ):
        """Digitize the shared optical means with the requested detector seed."""
        if not np.isfinite(exposure) or exposure < 0:
            raise ValueError("Exposure must be finite and nonnegative")
        if shape is None:
            shape = (
                simulation_parameters["whole_scan_planes"],
                *simulation_parameters["camera_shape_yx"],
            )
        native_step = simulation_parameters["microscope"]["scan_step_um"]
        if step > native_step:
            factor = round(step / native_step)
            if not np.isclose(step, factor * native_step, rtol=1e-6, atol=0):
                raise ValueError(
                    "Coarse camera sampling must select native scan planes"
                )
            native = render(
                shape,
                kind=kind,
                large=large,
                integration=integration,
                illumination_profile=illumination_profile,
                step=native_step,
                conversion=conversion,
                geometry=geometry,
                exposure=exposure,
                seed=seed,
                budget=budget,
            )
            return CameraObservation(
                ground_truth=native.ground_truth,
                microscope=replace(native.microscope, scan_step_um=step),
                expectation=native.expectation[::factor],
                raw=native.raw[::factor],
            )
        scene, microscope, expectation = optical_data(
            shape,
            kind=kind,
            large=large,
            integration=integration,
            illumination_profile=illumination_profile,
            step=step,
            geometry=geometry,
            budget=budget,
        )
        microscope = replace(
            microscope,
            camera=replace(microscope.camera, electrons_per_adu=conversion),
        )
        if exposure != 1:
            expectation = freeze_metadata(expectation * exposure)
        raw = digitize(expectation, microscope.camera, seed=seed)
        return CameraObservation(
            ground_truth=scene,
            microscope=microscope,
            expectation=expectation,
            raw=freeze_metadata(raw),
        )

    return render


@pytest.fixture(scope="session")
def cartesian_registration_pair(
    independent_photon_scene_factory, simulation_parameters
):
    """Observe known emitters on registration's orthogonal Cartesian ZYX grid.

    Both acquisitions share one known emission dose and calibrated microscope.
    Independent detector realizations avoid matching a reused camera-noise field.
    """
    from opm_processing.imageprocessing.camera import camera_correct
    from tests.simulation.camera import digitize

    @lru_cache(maxsize=4)
    def optical_data(applied_shift):
        """Reuse two immutable optical means at the prescribed physical shift."""
        scene = independent_photon_scene_factory("points")
        microscope = scene.microscope
        shape = (
            simulation_parameters["whole_scan_planes"],
            *simulation_parameters["camera_shape_yx"],
        )
        spacing = np.array(
            (
                microscope.scan_step_um,
                microscope.pixel_size_um,
                microscope.pixel_size_um,
            )
        )
        axes = centered_sensor_axes(shape, spacing)
        dose = forward_model(
            scene.density,
            Acquisition(
                scene.axes,
                shape,
                geometry="cartesian",
                camera_photon_budget=(10000, 30),
                output="object",
            ),
            microscope,
        )
        means = []
        for displacement in (np.zeros(3), np.array(applied_shift)):
            sensor_axes = tuple(
                axis - shift * step
                for axis, shift, step in zip(axes, displacement, spacing, strict=True)
            )
            mean = forward_model(
                dose,
                Acquisition(
                    scene.axes,
                    shape,
                    sensor_axes=sensor_axes,
                    geometry="cartesian",
                    output="expectation",
                ),
                microscope,
            )
            means.append(mean)
        return (
            microscope.camera,
            freeze_metadata(means),
            freeze_metadata(-np.array(applied_shift)),
        )

    @lru_cache(maxsize=2)
    def render(applied_shift, *, noise=True):
        """Draw independent exposures and use production camera calibration."""
        camera, means, expected = optical_data(applied_shift)
        images = []
        for index, mean in enumerate(means):
            observed = (
                camera_correct(
                    digitize(mean, camera, seed=42 + index),
                    camera.offset_adu,
                    camera.electrons_per_adu,
                )
                if noise
                else mean.astype(np.float32)
            )
            images.append(observed)
        return (*freeze_metadata(images), expected)

    return render


@pytest.fixture(scope="session")
def optical_psf(microscope_parameters):
    """Share one immutable calibrated PSF between optical operator units."""
    microscope = replace(
        microscope_parameters,
        fine_spacing_um=0.04,
        psf_half_extent_zyx_um=(3.0, 2.0, 2.0),
    )
    psf, _ = camera_psf(microscope)
    return freeze_metadata(psf.astype(np.float32))


@pytest.fixture(scope="session")
def ideal_photon_camera(camera_calibration):
    """Use the shared camera digitizer for ideal, unsigned photon observations."""
    return replace(
        camera_calibration, read_noise_e=0, electrons_per_adu=1, offset_adu=0
    )


@pytest.fixture
def camera_adc_case(
    independent_photon_scene_factory,
    mean,
    noise,
    full_model,
    camera_calibration,
    simulation_parameters,
):
    """Compare calibrated pixel draws with independent exact ADC moments."""
    from tests.simulation.camera import digitize
    from tests.simulation.camera_statistics import adc_moments

    camera = replace(camera_calibration, read_noise_e=noise)
    if full_model:
        scene = independent_photon_scene_factory()
        raw = forward_model(
            scene.density,
            Acquisition(
                scene.axes,
                (1, *simulation_parameters["camera_shape_yx"]),
                exposure=0,
                background_electrons=mean,
                seed=637,
            ),
            replace(scene.microscope, camera=camera),
        )
    else:
        raw = digitize(
            np.full(1_000_000, float(mean)), camera, seed=637 if mean == 0 else 31
        )
    return SimpleNamespace(raw=raw, camera=camera, moments=adc_moments(mean, camera))


@pytest.fixture
def measured_psf_calibration(tmp_path, psf_format):
    """Persist prescribed normalized calibration pixels through actual file writers."""
    import tifffile

    path = tmp_path / f"measured_psf.{psf_format}"
    kernel = np.arange(1, 28, dtype=np.float32).reshape(3, 3, 3)
    kernel /= kernel.sum()

    write = (
        partial(np.save, path)
        if psf_format == "npy"
        else partial(tifffile.imwrite, path, photometric="minisblack", metadata=None)
    )
    write(kernel)
    return SimpleNamespace(path=path, kernel=kernel, write=write)


@pytest.fixture
def uniform_detector_scene(
    microscope_parameters, uniform_optical_calibration, simulation_parameters
):
    """Form a uniform physical emitter field with the existing optical model."""

    def create(projection=False):
        """Fix the emission dose before applying detector sensitivity and noise."""
        microscope = replace(
            microscope_parameters,
            fine_spacing_um=0.1,
            scan_step_um=0.4,
            quantum_efficiency=0.3,
            camera=replace(microscope_parameters.camera, electrons_per_adu=0.5),
        )
        shape = (
            1 if projection else simulation_parameters["whole_scan_planes"],
            *simulation_parameters["camera_shape_yx"],
        )
        means = uniform_optical_calibration(
            microscope,
            shape,
            geometry="cartesian" if projection else "oblique",
            photon_budget=(12000 / microscope.quantum_efficiency, 0),
        )
        np.testing.assert_allclose(means, 12000, rtol=2e-5)
        return SimpleNamespace(
            microscope=microscope,
            means=means,
            shape=shape,
            nominal=12000.0,
        )

    return create


@pytest.fixture
def uniform_response_case(scan_planes, detector_shape):
    """Share prescribed constant fluorescence and an asymmetric detector response."""
    from scipy import ndimage

    response = np.broadcast_to(
        np.tile(np.array((0.25, 0.5, 1, 2), np.float32), detector_shape[1] // 4),
        detector_shape,
    )
    truth = np.full((scan_planes, *response.shape), 1200, np.float32)
    psf = np.ones((1 if scan_planes == 1 else 3, 3, 3), np.float32)
    psf /= psf.sum()
    measured = ndimage.convolve(truth, psf, mode="reflect") * response
    return SimpleNamespace(response=response, truth=truth, psf=psf, measured=measured)


@pytest.fixture
def uniform_detector_acquisition(
    request,
    tmp_path,
    opm_v2_acquisition_factory,
    uniform_detector_scene,
    roi_mode,
    camera_gain_profile,
):
    """Write physical camera frames and real preview coordinates for calibration workflows."""
    from opm_processing.dataio.acquisition import ChannelMetadata
    from opm_processing.dataio.roi import PhysicalRoi, projection_tile_footprints
    from opm_processing.imageprocessing.coordinates import STAGE_COORDINATE_VERSION
    from opm_processing.process import process_skewed, write_flatfield
    from tests.simulation.camera import digitize

    save_float = not isinstance(request.param, tuple)
    mode = request.param[0] if not save_float else request.param
    scene = uniform_detector_scene(mode == "projection")
    shape = scene.shape
    response = np.broadcast_to(
        np.repeat(np.array((0.25, 0.5, 1, 2), np.float32), shape[-1] // 4), shape[-2:]
    ).copy()
    if mode == "stage":
        response *= np.interp(
            np.linspace(0, len(camera_gain_profile) - 1, shape[-1]),
            np.arange(len(camera_gain_profile)),
            camera_gain_profile,
        )
    # Detector sensitivity changes quantum efficiency after optical integration.
    # Nominal QE .3 times the largest relative response remains below one.
    assert scene.microscope.quantum_efficiency * response.max() < 1
    means = scene.means * response
    raw = digitize(means, scene.microscope.camera, seed=31)
    assert raw.max() < np.iinfo(np.uint16).max
    if mode == "stage":
        raw = raw[::-1].copy()
    dataset = opm_v2_acquisition_factory(
        raw[None, None, None],
        name="uniform_detector",
        mode=mode,
        camera_offset=scene.microscope.camera.offset_adu,
        camera_conversion=scene.microscope.camera.electrons_per_adu,
        camera_read_noise_e=scene.microscope.camera.read_noise_e,
        channels=(ChannelMetadata(0, "637", 637, 10, 10),),
        scan_axis_step_um=scene.microscope.scan_step_um,
        stage_positions_zxy=((0, 0, 0),),
    )
    write_flatfield(
        tmp_path / "uniform_detector_flatfield.ome.tif",
        response[None, None],
        scene.microscope.pixel_size_um,
    )
    output_dir = tmp_path
    options = {
        "deconvolve": True,
        "flatfield_correction": True,
        "save_float32": save_float,
        "z_downsample_level": 1,
        "max_projection": False,
        "create_fused_max_projection": False,
        "write_fused_max_projection_tiff": False,
    }
    pixel = scene.microscope.pixel_size_um
    if mode == "projection":
        support = np.ones(shape[:2], bool)
    else:
        native_shape, _, _, _ = deskew_shape_estimator(
            shape,
            scene.microscope.angle_deg,
            scene.microscope.scan_step_um,
            pixel,
            crop_after_deskew=False,
        )
        support = (
            deskew_coverage(
                shape,
                native_shape,
                theta=scene.microscope.angle_deg,
                distance=scene.microscope.scan_step_um,
                pixel_size=pixel,
            )
            == 1
        )
    if roi_mode:
        process_skewed(
            dataset.path,
            acquisition=dataset.metadata,
            max_projection=True,
            create_fused_max_projection=False,
            save_float32=True,
            z_downsample_level=1,
        )
        preview = tmp_path / "uniform_detector_max_z_deskewed.ome.zarr"
        footprints = projection_tile_footprints(preview)
        origin = footprints[0]["origin_zyx_um"]
        y_stop, x_stop = support.shape[-1] - 3, shape[-1] - 16
        roi = PhysicalRoi(
            bounds_yx_um=(
                origin[1] + 3.25 * pixel,
                origin[1] + (y_stop - 0.25) * pixel,
                origin[2] + 16.25 * pixel,
                origin[2] + (x_stop - 0.25) * pixel,
            ),
            source_path=preview,
            grid_origin_yx_um=tuple(origin[1:]),
            pixel_size_yx_um=(pixel, pixel),
            position_indices=(0,),
            tile_footprints=tuple(footprints),
            coordinate_version=STAGE_COORDINATE_VERSION,
        )
        roi_path = roi.write(tmp_path / "uniform_detector_roi.json")
        support = support[:, 3:y_stop]
        options = {
            k: v
            for k, v in options.items()
            if k
            not in (
                "max_projection",
                "create_fused_max_projection",
                "write_fused_max_projection_tiff",
            )
        }
        output_dir = tmp_path / "roi"
        options.update(roi_json=roi_path, output=output_dir, registration_channel=0)
    suffix = "projection" if mode == "projection" else "deskewed"
    return SimpleNamespace(
        dataset=dataset,
        scene=scene,
        support=support,
        options=options,
        roi_mode=roi_mode,
        output=output_dir / f"uniform_detector_decon_{suffix}.ome.zarr",
    )


@pytest.fixture(scope="session")
def fine_optical_images(microscope_parameters):
    """Blur a finite fluorescent object in physical space before sampling.

    A Gaussian fluorophore density describes a smooth, finite-size bead, not
    the optical PSF. The blur is the vectorial diffraction PSF. All three grids
    use the same physical object, wavelength, support and photon-density units.
    """
    result = []
    for spacing in (0.0575, 0.02875, 0.014375):
        # Shared physical boundaries must also coincide across refinements;
        # otherwise changes in zero-filled support masquerade as sampling error.
        microscope = replace(
            microscope_parameters,
            fine_spacing_um=spacing,
            psf_half_extent_zyx_um=(24 * 0.0575, 16 * 0.0575, 16 * 0.0575),
        )
        axes, psf = emission_psf(microscope, spacing)
        z, y, x = np.meshgrid(*axes, indexing="ij", sparse=True)
        center = (0.07, 0.11, -0.04)
        sigma = 0.12
        density = np.exp(
            -sum(
                (coord - origin) ** 2
                for coord, origin in zip((x, y, z), center, strict=False)
            )
            / (2 * sigma**2)
        ) / ((2 * np.pi) ** 1.5 * sigma**3)
        image = forward_model(
            density,
            Acquisition(axes, output="irradiance"),
            microscope,
        )
        result.append((axes, psf, image))
    return freeze_metadata(result)


@pytest.fixture(scope="session")
def linear_camera_field():
    """Known concentration ramp, with an even row count and shifted XYZ origin."""
    axes = tuple(np.linspace(-5, 5, 101) for _ in range(3))
    z, y, x = np.meshgrid(*axes, indexing="ij", sparse=True)
    return AnalyticalCameraField(
        axes=freeze_metadata(axes),
        field=freeze_metadata(100 + 2 * x + 3 * y + 5 * z),
        shape=(7, 12, 9),
        center=(0.03, -0.12, 0.21),
    )


@pytest.fixture(scope="session")
def quadratic_camera_field():
    """Share a quadratic irradiance with a known trilinear interpolation bound."""
    axes = tuple(np.arange(-40, 41) * 0.0025 for _ in range(3))
    z, y, x = np.meshgrid(*axes, indexing="ij", sparse=True)
    return AnalyticalCameraField(
        axes=freeze_metadata(axes),
        field=freeze_metadata(10 + 2 * x**2 + y**2 + z**2),
        shape=(1, 1, 1),
        center=(0, 0, 0),
    )


@pytest.fixture
def discrete_optical_case(factor, nonuniform):
    """Blur a prescribed positive 3D object for full and tiled operator units."""
    from scipy import ndimage

    scan, row, column = np.indices((31, 24, 26), dtype=float)
    truth = (
        100 + 50 * np.cos(scan / 3) ** 2 + 30 * np.sin((row + column) / 4) ** 2
    ).astype(np.float32)
    psf = np.arange(1, 28, dtype=np.float32).reshape(3, 3, 3)
    psf /= psf.sum()
    response = (
        np.broadcast_to(np.linspace(0.3, 2, truth.shape[-1]), truth.shape[-2:])
        if nonuniform
        else np.ones(truth.shape[-2:])
    )
    boundary = "reflect" if factor is None else "constant"
    expectation = ndimage.convolve(truth, psf, mode=boundary) * response
    measured = expectation if factor is None else expectation[::factor]
    return SimpleNamespace(
        measured=measured.astype(np.float32),
        psf=psf,
        response=response,
        truth=truth,
        expectation=expectation,
    )


@pytest.fixture
def production_psf_parameters(microscope_parameters):
    """Provide explicit optical support and taper calibration for generated kernels."""
    microscope = microscope_parameters
    return {
        "lens": dict(microscope.psf_parameters, NA=microscope.numerical_aperture),
        "skewed_depth_um": 0.0,
        "projection_depth_um": 0.0,
        "projection_nz": 3,
        "projection_dz_um": microscope.fine_spacing_um,
        "projection_nxy": 15,
        "taper_sigma_px_3d": 4.0,
        "taper_sigma_px_2d": 4.0,
        "depth_scale_um": 100.0,
        "depth_width_scale": 4.0,
        "support_fwhms": 4.0,
        "grid_oversampling": 2,
    }


@pytest.fixture(params=[(0.4, (5, 17, 17)), (1.2, (3, 17, 17))])
def production_point_acquisition(
    request, microscope_parameters, production_psf_parameters
):
    """Share production optics between fine-grid simulation and deconvolution.

    Kernel probabilities are conditioned on the finite camera ROI. Unit-mass
    normalization is fixed before sampling camera data and does not fit brightness.
    An infinite taper width selects the untapered PSF used by this simulation.
    """
    from opm_processing.imageprocessing.opmpsf import generate_skewed_psf

    step, shape = request.param
    microscope = replace(
        microscope_parameters,
        scan_step_um=step,
        pixel_samples=12,
        psf_half_extent_zyx_um=(0.6, 2.2, 2.2),
    )
    expectations = []
    for spacing in (0.02, 0.01):
        axis = np.arange(-1, 2) * spacing
        density = np.zeros((3, 3, 3))
        deposit_point(density, (axis,) * 3, (0, 0, 0), 1)
        expectations.append(
            forward_model(
                density,
                Acquisition(
                    (axis,) * 3,
                    shape,
                    output="expectation",
                    camera_photon_budget=(1000, 0),
                ),
                replace(microscope, fine_spacing_um=spacing),
            )
        )
    parameters = dict(production_psf_parameters, taper_sigma_px_3d=np.inf)
    kernels = {}
    for samples in (1, 2, 3):
        kernel = generate_skewed_psf(
            microscope.wavelength_um,
            microscope.pixel_size_um,
            step,
            microscope.angle_deg,
            parameters=dict(parameters, grid_oversampling=samples),
        )
        region = tuple(
            slice((n - size) // 2, (n + size) // 2)
            for n, size in zip(kernel.shape, shape, strict=True)
        )
        kernel = kernel[region]
        kernels[samples] = kernel / kernel.sum()
    return SimpleNamespace(
        expectations=freeze_metadata(expectations),
        kernels=freeze_metadata(kernels),
    )


@pytest.fixture(scope="session")
def noisy_scene_masks(microscope_parameters, independent_photon_scene_factory):
    """Reuse fixed physical apertures independently of dose and illumination."""
    from opm_processing.imageprocessing.opmpsf import (
        camera_grid_center,
        get_skewed_coords,
        integrate_camera_pixels,
    )

    @lru_cache(maxsize=2)
    def create(kind, shape, integration):
        """Sample the original object and its finite optical support once."""
        microscope = microscope_parameters
        scene = independent_photon_scene_factory(
            kind,
            large=True,
            integration_scan_planes=shape[0] if integration else None,
        )
        sensor_axes = centered_sensor_axes(
            shape,
            (
                microscope.scan_step_um,
                microscope.pixel_size_um,
                microscope.pixel_size_um,
            ),
        )
        columns = sensor_axes[2]
        supported = (columns >= scene.axes[2][0] - microscope.pixel_size_um / 2) & (
            columns <= scene.axes[2][-1] + microscope.pixel_size_um / 2
        )
        source = np.zeros(shape, bool)
        if supported.any():
            source[..., supported] = (
                integrate_camera_pixels(
                    scene.density,
                    scene.axes,
                    (*sensor_axes[:2], columns[supported]),
                    pixel_size_um=microscope.pixel_size_um,
                    theta_deg=microscope.angle_deg,
                    samples=microscope.pixel_samples,
                    area_integral=False,
                )
                > 0
            )
        theta = np.deg2rad(microscope.angle_deg)
        x, y, z = get_skewed_coords(
            shape, microscope.pixel_size_um, microscope.scan_step_um, theta
        )
        center = camera_grid_center(
            shape,
            microscope.pixel_size_um,
            microscope.scan_step_um,
            microscope.angle_deg,
        )
        dark = np.zeros(shape, bool)
        for coordinate, gt_axis, psf_axis, half_footprint in zip(
            (z - center[0], y - center[1], x - center[2]),
            scene.axes,
            scene.psf_axes,
            microscope.pixel_size_um / 2 * np.array((np.sin(theta), np.cos(theta), 1)),
            strict=True,
        ):
            dark |= (coordinate + half_footprint < gt_axis[0] + psf_axis[0]) | (
                coordinate - half_footprint > gt_axis[-1] + psf_axis[-1]
            )
        assert dark.any()
        for array in (source, dark):
            array.setflags(write=False)
        return source, dark

    return create


@pytest.fixture(
    scope="session",
    params=[
        (kind, profile, budget, "tiled")
        for kind in ("line-pairs", "sphere")
        for profile in ("uniform", "edge-taper")
        for budget in range(3)
    ],
    ids=lambda case: f"{case[0]}-{case[1]}-dose{case[2]}-{case[3]}",
)
def noisy_extended_scene(
    request,
    microscope_parameters,
    simulation_parameters,
    simulated_camera,
    noisy_scene_masks,
):
    """Observe fixed extended emitters with calibrated noisy camera exposures."""
    config = simulation_parameters
    kind, profile, budget, layout = request.param[:4]
    integration = len(request.param) > 4 and request.param[4]
    microscope = replace(
        microscope_parameters, scan_step_um=config["microscope"]["scan_step_um"]
    )
    shape = (
        config[f"{layout}_scan_planes"],
        *config["camera_shape_yx"],
    )
    observation = simulated_camera(
        shape,
        kind=kind,
        large=True,
        integration=integration,
        illumination_profile=profile,
        step=microscope.scan_step_um,
        conversion=microscope.camera.electrons_per_adu,
        budget=tuple(config["photon_budgets"][budget]),
        seed=config["camera_seed"],
    )
    expectation, raw = observation.expectation, observation.raw
    source, dark = noisy_scene_masks(kind, shape, integration)
    assert not np.any(expectation, where=dark)
    for array in (expectation, raw):
        array.setflags(write=False)
    return microscope, expectation, raw, source, dark
