"""Physical specimens and shared production optics for reconstruction."""

import json
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator

from tests.simulation.forward_model import (
    Acquisition,
    Microscope,
    camera_psf,
    emission_psf,
    forward_model,
)
from tests.simulation.objects import centered_axis, deposit_point


@pytest.fixture(scope="session")
def simulation_parameters():
    """Load the shared physical acquisition and specimen configuration."""
    return json.loads(
        (
            Path(__file__).resolve().parent / "configs/deconvolution_figures.json"
        ).read_text()
    )


@pytest.fixture(scope="session")
def camera_calibration(simulation_parameters):
    """Provide the camera calibration declared in the shared configuration."""
    from tests.simulation.camera import Camera

    return Camera(**simulation_parameters["microscope"]["camera"])


@pytest.fixture(scope="session")
def microscope_parameters(simulation_parameters, camera_calibration):
    """Provide immutable microscope settings for the compact optical fixtures."""
    options = dict(simulation_parameters["microscope"], camera=camera_calibration)
    options["psf_half_extent_zyx_um"] = tuple(options["psf_half_extent_zyx_um"])
    # This small physical unit-test grid predates the figure acquisition grid.
    return replace(Microscope(**options), scan_step_um=0.2)


@pytest.fixture(scope="session")
def independent_photon_scene_factory(microscope_parameters, simulation_parameters):
    """Cache independent fine-grid scenes without retaining an entire parameter sweep."""
    from tests.simulation.scenes import prepare_scene

    @lru_cache(maxsize=2)
    def build(kind="points", profile="uniform", spacing=0.05):
        """Reuse cached physical scenes at the requested illumination and grid spacing."""
        return prepare_scene(
            kind,
            replace(microscope_parameters, fine_spacing_um=spacing),
            parameters=simulation_parameters["specimens"][kind],
            illumination_profile=profile,
        )

    return build


@pytest.fixture(scope="session")
def simulated_camera(
    microscope_parameters, simulation_parameters, independent_photon_scene_factory
):
    """Cache camera observations of the shared physical ground-truth objects.

    Returns
    -------
    callable
        Render an SYX acquisition using the single forward model. The result
        contains the object, optical calibration, camera means and unsigned ADU.
        Geometry, exposure and detector calibration are explicit inputs.
    """
    from tests.simulation.camera import digitize

    @lru_cache(maxsize=24)
    def render(
        shape,
        *,
        step=0.4,
        offset=100.0,
        conversion=0.25,
        geometry="oblique",
        exposure=1.0,
        seed=42,
        kind="points",
        profile="uniform",
    ):
        """Simulate optical formation before camera sampling and digitization."""
        scene = independent_photon_scene_factory(kind)
        microscope = replace(
            microscope_parameters,
            scan_step_um=step,
            camera=replace(
                microscope_parameters.camera,
                offset_adu=offset,
                electrons_per_adu=conversion,
            ),
        )
        expectation = forward_model(
            scene.density,
            Acquisition(
                scene.axes,
                shape,
                geometry=geometry,
                exposure=exposure,
                illumination_profile=profile,
                camera_photon_budget=tuple(simulation_parameters["photon_budgets"][0]),
                output="expectation",
            ),
            microscope,
        )
        raw = digitize(expectation, microscope.camera, seed=seed)
        return SimpleNamespace(
            ground_truth=scene,
            microscope=microscope,
            expectation=expectation,
            raw=raw,
        )

    return render


@pytest.fixture(scope="session")
def optical_setup(microscope_parameters):
    """Share one calibrated PSF between profiles, operators and optical specimens."""
    microscope = replace(
        microscope_parameters,
        fine_spacing_um=0.04,
        psf_half_extent_zyx_um=(3.0, 2.0, 2.0),
    )
    axes, field = emission_psf(microscope, microscope.fine_spacing_um)
    psf, scale = camera_psf(microscope)
    return SimpleNamespace(
        microscope=microscope,
        psf=psf.astype(np.float32),
        density_scale=scale,
        field=RegularGridInterpolator(axes, field, bounds_error=False, fill_value=0),
    )


@pytest.fixture(scope="session", params=["x", "y", "z"])
def rayleigh_pair(request, optical_setup):
    """Direction-specific Rayleigh pair using production optical sampling.

    Fine optical samples locate the diffraction PSF's first minimum. The target
    contrast is measured from the noise-free forward acquisition on the same
    grid and interpolated profile as the reconstruction. A continuous optical
    profile is not a contrast threshold for discrete camera samples.
    The two sources are placed at that physical separation, without snapping
    their positions to acquisition voxels or creating reconstruction-grid delta PSFs.
    The common detector field contains the full PSF response of every pair;
    its scan endpoints also coincide for the tested sparse acquisitions.
    """
    from tests.simulation.metrics import first_minimum, two_point_profile

    axis = request.param
    index = "xyz".index(axis)
    directions = np.eye(3)
    direction = directions[index]
    radius = np.arange(0, 1.8001, 0.005)
    separations = np.array(
        [
            first_minimum(radius, optical_setup.field((radius[:, None] * d)[:, ::-1]))
            for d in directions
        ]
    )
    separation = separations[index]
    microscope = optical_setup.microscope
    theta = np.deg2rad(microscope.angle_deg)
    xyz_offsets = directions * separations[:, None] / 2
    offsets = np.column_stack(
        (
            (xyz_offsets[:, 1] - xyz_offsets[:, 2] / np.tan(theta))
            / microscope.scan_step_um,
            xyz_offsets[:, 2] / (microscope.pixel_size_um * np.sin(theta)),
            xyz_offsets[:, 0] / microscope.pixel_size_um,
        )
    )
    # Detector half-width >= PSF half-width + maximum source displacement.
    shape = (
        2
        * np.ceil(
            (np.array(optical_setup.psf.shape) - 1) / 2
            + np.max(np.abs(offsets), axis=0)
        ).astype(int)
        + 1
    )
    scan_multiple = np.lcm(3, 4)
    shape[0] = ((shape[0] - 1 + scan_multiple - 1) // scan_multiple) * scan_multiple + 1
    offset = offsets[index]
    axes = tuple(centered_axis(e, 0.04) for e in (4.0, 3.0, 3.0))
    density = np.zeros(tuple(map(len, axes)), np.float32)
    for sign in (-1, 1):
        deposit_point(density, axes, direction * sign * separation / 2, 100000)
    expectation = forward_model(
        density, Acquisition(axes, tuple(shape), output="expectation"), microscope
    )
    return SimpleNamespace(
        axis=axis,
        separation=separation,
        microscope=microscope,
        acquisition_dip=two_point_profile(
            expectation, (np.array(expectation.shape) - 1) / 2, offset, microscope
        )["valley_depth"],
        offset=offset,
        expectation=expectation,
        psf=optical_setup.psf,
    )


@pytest.fixture(scope="session")
def planar_point_model():
    """Return two known emitters and the normalized central YX PSF used to blur them."""
    y, x = np.mgrid[-3:4, -3:4]
    central = np.exp(-(y**2 + x**2) / 8).astype(np.float32)
    central /= central.sum()
    psf = np.stack((np.ones_like(central), central * 7, np.ones_like(central) * 100))
    truth = np.zeros((41, 43), np.float32)
    truth[12, 13] = 2000
    truth[28, 30] = 3000
    return truth, central, psf


@pytest.fixture
def uniform_detector_acquisition(
    request,
    acquisition_factory,
    tmp_path,
    roi_mode,
    camera_gain_profile,
    production_psf_parameters,
    microscope_parameters,
):
    """Persist uniform fluorescence with a known nonuniform camera response."""
    from opm_processing.imageprocessing.opmtools import orthogonal_deskew
    from opm_processing.process import write_flatfield

    generated = isinstance(request.param, tuple)
    mode = request.param[0] if generated else request.param
    deconvolution = {}
    if generated:
        deconvolution.update(
            psf_parameters=production_psf_parameters,
            emission_wavelengths_nm=[microscope_parameters.wavelength_um * 1000],
        )
    if mode == "stage":
        deconvolution["detector_gain_x"] = camera_gain_profile.tolist()
    shape = (1 if mode == "projection" else 8, 12, 1900)
    field = np.broadcast_to(
        np.repeat(np.array((0.25, 0.5, 1.0, 2.0), np.float32), 475), shape[-2:]
    ).copy()
    response = field.copy()
    if mode == "stage":
        response *= camera_gain_profile
    truth = np.full(shape, 1200, dtype=np.float32)
    measured = truth * response
    raw = np.clip(np.rint(100 + measured / 0.5), 0, 65535).astype(np.uint16)
    dataset = acquisition_factory(
        raw[None, None, None],
        name="uniform_detector",
        mode=mode,
        camera_conversion=0.5,
        deconvolution=deconvolution or None,
    )
    write_flatfield(
        tmp_path / "uniform_detector_flatfield.ome.tif", field[None, None], 0.115
    )
    psf_path = tmp_path / "uniform_detector_psf.npy"
    psf = np.ones((3, 3, 3), np.float32)
    np.save(psf_path, psf / psf.sum())
    expected = (
        truth
        if mode == "projection"
        else orthogonal_deskew(
            np.full(shape, 1200, dtype=np.float32),
            theta=30.0,
            distance=0.4,
            pixel_size=0.115,
            downsample_factor=1,
        )
    )
    options = {
        "deconvolve": True,
        "flatfield_correction": True,
        "decon_psf_paths": [psf_path],
        "save_float32": True,
        "z_downsample_level": 1,
        "max_projection": False,
        "create_fused_max_projection": False,
        "write_fused_max_projection_tiff": False,
    }
    if generated:
        options.pop("decon_psf_paths")
    if roi_mode:
        from opm_processing.dataio.roi import PhysicalRoi

        y_stop, x_stop = expected.shape[-2] - 3, expected.shape[-1] - 50
        roi_path = PhysicalRoi(
            # Bounds lie inside edge voxels: intersecting-voxel selection is
            # unambiguous despite binary rounding of physical coordinates.
            bounds_yx_um=(
                3.25 * 0.115,
                (y_stop - 0.25) * 0.115,
                50.25 * 0.115,
                (x_stop - 0.25) * 0.115,
            ),
            source_path=tmp_path / "uniform_detector_fused.ome.zarr",
            grid_origin_yx_um=(0, 0),
            pixel_size_yx_um=(0.115, 0.115),
            position_indices=(0,),
            tile_footprints=(
                {
                    "time_index": 0,
                    "position_index": 0,
                    "origin_zyx_um": [0, 0, 0],
                    "bounds_yx_um": [
                        0,
                        expected.shape[-2] * 0.115,
                        0,
                        expected.shape[-1] * 0.115,
                    ],
                },
            ),
        ).write(tmp_path / "uniform_detector_roi.json")
        expected = expected[:, 3:y_stop, 50:x_stop]
        for key in (
            "max_projection",
            "create_fused_max_projection",
            "write_fused_max_projection_tiff",
        ):
            options.pop(key)
        options.update(roi_json=roi_path, output=tmp_path)
    suffix = "projection" if mode == "projection" else "deskewed"
    return SimpleNamespace(
        dataset=dataset,
        psf_path=psf_path,
        expected=expected,
        options=options,
        roi_mode=roi_mode,
        output=tmp_path / f"uniform_detector_decon_{suffix}.ome.zarr",
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
    return result


@pytest.fixture(scope="session")
def linear_camera_field():
    """Known concentration ramp, with an even row count and shifted XYZ origin."""
    axes = tuple(np.linspace(-5, 5, 101) for _ in range(3))
    z, y, x = np.meshgrid(*axes, indexing="ij", sparse=True)
    return SimpleNamespace(
        axes=axes,
        field=100 + 2 * x + 3 * y + 5 * z,
        shape=(7, 12, 9),
        center=(0.03, -0.12, 0.21),
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
        boundary=boundary,
    )


@pytest.fixture
def point_profile_case(request, microscope_parameters):
    """Provide prescribed emitters and independent expectations on a shared grid."""
    pattern, offset, separation = request.param
    center = np.array((8.0, 8.0, 8.0))
    offset = np.asarray(offset)
    volume = np.full((17, 17, 17), float(pattern == "flat"))
    if pattern == "pair":
        volume[tuple((center - offset).astype(int))] = 100
        volume[tuple((center + offset).astype(int))] = 200
    elif pattern == "adjacent":
        center[2] += 0.5
        volume[8, 8, 8:10] = 100
    else:
        volume[8, 8, 8] = 1
    return SimpleNamespace(
        volume=volume,
        center=center,
        offset=offset,
        separation=separation,
        microscope=microscope_parameters,
        pattern=pattern,
    )


@pytest.fixture(scope="session")
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


@pytest.fixture
def production_psf_grid(monkeypatch):
    """Observe production optical fields and their declared sample coordinates."""
    from opm_processing.imageprocessing import opmpsf

    grids = []
    evaluate = opmpsf.create_psf_silicone_100x

    def capture(*args, **kwargs):
        """Retain the actual optical field and its declared Cartesian spacing."""
        field = evaluate(*args, **kwargs)
        grids.append((field, args[:4]))
        return field

    monkeypatch.setattr(opmpsf, "create_psf_silicone_100x", capture)
    return grids


@pytest.fixture(scope="module", params=[(0.4, (5, 17, 17)), (1.2, (3, 17, 17))])
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
    kernel, _ = camera_psf(replace(microscope, fine_spacing_um=spacing))
    region = tuple(
        slice((n - size) // 2, (n + size) // 2)
        for n, size in zip(kernel.shape, shape, strict=True)
    )
    kernel = kernel[region]
    return SimpleNamespace(
        microscope=microscope,
        expectations=expectations,
        kernels=kernels,
        kernel=kernel / kernel.sum(),
    )
