"""Overlapping physical specimens, placement errors and reconstruction settings."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture(params=[(0, 14), (8, 0), (8, 14)], ids=["x", "y", "yx"])
def orthogonal_registration_pair(request, simulation_parameters):
    """Known blurred beads observed through two overlapping oblique camera stacks."""
    from opm_processing.imageprocessing.opmtools import orthogonal_deskew

    calibration = simulation_parameters["microscope"]
    geometry = {
        "theta": calibration["angle_deg"],
        "distance": calibration["scan_step_um"],
        "pixel_size": calibration["pixel_size_um"],
    }
    theta = np.deg2rad(geometry["theta"])
    shape = (20, 48, 48)
    scan, row, x = np.indices(shape, dtype=float)
    z = row * np.sin(theta)
    y = scan * (geometry["distance"] / geometry["pixel_size"]) + row * np.cos(theta)
    delta_y, delta_x = request.param
    error = np.array([0, 2 * bool(delta_y), 2 * bool(delta_x)])
    origins = [np.zeros(3), np.array([0, delta_y, delta_x])]
    images = []
    for offset in origins:
        image = np.zeros(shape, dtype=float)
        # Prescribed Gaussian optical bead images isolate registration from a
        # diffraction-PSF accuracy claim. All positions are laboratory pixels.
        for center, amplitude in [
            ((8, 40, 18), 1000),
            ((15, 65, 32), 1800),
            ((19, 53, 27), 1300),
        ]:
            radius = sum(
                ((coordinate + shift - position) / width) ** 2
                for coordinate, shift, position, width in zip(
                    (z, y, x), offset, center, (2, 4, 3), strict=True
                )
            )
            image += amplitude * np.exp(-radius / 2)
        images.append(
            orthogonal_deskew(image.astype(np.float32), **geometry, downsample_factor=1)
        )
    support = (
        orthogonal_deskew(np.ones(shape, np.float32), **geometry, downsample_factor=1)
        > 0
    )
    nominal = origins[1].astype(int) + error
    left = tuple(
        slice(max(0, d), min(n, n + d))
        for n, d in zip(images[0].shape, nominal, strict=True)
    )
    right = tuple(
        slice(max(0, -d), min(n, n - d))
        for n, d in zip(images[0].shape, nominal, strict=True)
    )
    return SimpleNamespace(
        fixed=images[0][left],
        moving=images[1][right],
        masks=(support[left], support[right]),
        expected=-error,
        axis=int(np.argmax(nominal / np.array(images[0].shape))),
    )


@pytest.fixture
def deskew_field(request, simulation_parameters):
    """Provide calibrated analytic fields and explicit oblique acquisition geometry."""
    kind, step_um, *shape = request.param
    microscope = simulation_parameters["microscope"]
    geometry = {
        "theta": microscope["angle_deg"],
        "distance": step_um,
        "pixel_size": microscope["pixel_size_um"],
    }
    angle = np.deg2rad(geometry["theta"])
    step = step_um / geometry["pixel_size"]
    scan, row, x = np.indices(shape[0] if shape else (24, 33, 9), dtype=float)
    if kind == "constant":
        source = np.full(scan.shape, 0.24)
    elif kind == "x":
        source = 10 + 2 * x
    elif kind == "y":
        source = 100 + scan * step + row * np.cos(angle)
    elif kind == "z":
        source = 100 + row * np.sin(angle)
    elif kind == "affine":
        source = 100 + 2 * x + scan * step + row * (np.cos(angle) + np.sin(angle))
    else:
        source = np.random.default_rng(31).uniform(0.1, 500, scan.shape)
    # A strided source also exercises camera-X addressing across output tiles.
    storage = np.zeros((*source.shape[:2], 2 * source.shape[-1]), np.float32)
    storage[..., ::2] = source
    return SimpleNamespace(
        data=storage[..., ::2], kind=kind, geometry=geometry, step=step
    )


@pytest.fixture
def deskew_analytic_truth(deskew_field):
    """Evaluate affine fluorescence and fully sampled Z-bin interiors analytically.

    Returns
    -------
    callable
        Compute known laboratory fluorescence on an output grid, including the
        30-degree photon-volume Jacobian and the mean position of each Z bin.
    """
    case = deskew_field

    def evaluate(shape, *, factor=1, reverse=False):
        """Return analytic values and the region whose interpolation stencil is supported."""
        z, y, x = np.indices(shape, dtype=float)
        first = (shape[0] - 1 - z if reverse else z) * factor
        last = first + factor - 1
        mean_z = (first + last) / 2
        theta = np.deg2rad(case.geometry["theta"])
        guard = case.step * np.cos(theta) + 2
        interior = (
            ((y - last / np.tan(theta)) / case.step > 2)
            & ((y - first / np.tan(theta)) / case.step < case.data.shape[0] - 3)
            & (first / np.sin(theta) > guard)
            & (last / np.sin(theta) < case.data.shape[1] - guard - 1)
            & (x < case.data.shape[2])
        )
        fields = {
            "constant": 0.24,
            "x": 10 + 2 * x,
            "y": 100 + y,
            "z": 100 + mean_z,
            "affine": 100 + 2 * x + y + mean_z,
        }
        expected = np.broadcast_to(
            np.asarray(fields[case.kind]) * (2 / case.step), shape
        )
        return expected, interior

    return evaluate


@pytest.fixture
def deskew_roundoff(factor):
    """Bound two float32 weighted sums and their final intensity multiplication.

    Each Z bin has at most four positive interpolation terms per source plane,
    followed by division. Comparing two evaluations gives 2e/(1-2e), where
    e=(4*factor+2)*eps; unsupported zeros remain exact.
    """
    error = (4 * factor + 2) * np.finfo(np.float32).eps
    return {"rtol": 2 * error / (1 - 2 * error), "atol": 0}


@pytest.fixture
def deskew_output(deskew_field, factor):
    """Provide a dirty strided output view with independently calculated dimensions."""
    data, geometry = deskew_field.data, deskew_field.geometry
    angle = np.deg2rad(geometry["theta"])
    nz = int(np.ceil(data.shape[1] * np.sin(angle)))
    ny = int(np.ceil(data.shape[0] * deskew_field.step + data.shape[1] * np.cos(angle)))
    shape = (
        (nz + factor - 1) // factor,
        ((ny + 3) // 4) * 4,
        ((data.shape[2] + 3) // 4) * 4,
    )
    storage = np.full((2 * shape[0], shape[1] + 2, shape[2] + 2), np.nan, np.float32)
    return storage, storage[::2, 1:-1, 1:-1]


@pytest.fixture
def deskew_mosaic(deskew_field, factor):
    """Provide independent coverage and fluorescence for overlapping oblique tiles."""
    from opm_processing.imageprocessing.opmtools import orthogonal_deskew

    data = deskew_field.data
    gain = 2 / deskew_field.step
    response = orthogonal_deskew(
        np.full(data.shape, 1, dtype=np.float32),
        **deskew_field.geometry,
        downsample_factor=factor,
    )
    support = response[..., 0] / gain
    positions = ((0, 0, 0), (0, 13, 3))
    shape = (response.shape[0], response.shape[1] + 13, response.shape[2] + 3)
    covered = np.zeros(shape, bool)
    for _z, y, x in positions:
        covered[:, y : y + response.shape[1], x : x + data.shape[2]] |= (
            support[..., None] > 0
        )
    return SimpleNamespace(
        data=data,
        geometry=deskew_field.geometry,
        support=support,
        positions=positions,
        shape=shape,
        x_profile=np.r_[
            np.ones(data.shape[2], np.float32),
            np.zeros(response.shape[2] - data.shape[2], np.float32),
        ],
        expected=covered.astype(np.float32) * (0.24 * gain),
    )


@pytest.fixture
def deskew_processing_case(deskew_field, factor, mode, acquisition_factory):
    """Provide raw camera data, physical ROI bounds and independent saved-pixel truth."""
    from opm_processing.dataio.acquisition import inspect_acquisition
    from opm_processing.dataio.roi import PhysicalRoi
    from opm_processing.imageprocessing.opmtools import orthogonal_deskew

    case = deskew_field
    dataset = acquisition_factory(
        (100 + case.data).astype(np.uint16)[None, None, None],
        name="deskew_tail",
        camera_conversion=1,
        scan_axis_step_um=case.geometry["distance"],
        pixel_size_um=case.geometry["pixel_size"],
        angle_deg=case.geometry["theta"],
        stage_positions_zxy=((0, 0, 0),),
    )
    response = orthogonal_deskew(
        np.full(case.data.shape, 1, dtype=np.float32),
        **case.geometry,
        downsample_factor=factor,
    )
    ramp = np.pad(case.data[0, 0], (0, response.shape[-1] - case.data.shape[-1]))
    expected = response * ramp
    pixel = case.geometry["pixel_size"]
    roi = None
    origin = (0, 0, 0)
    if mode == "roi":
        y_end, x_end = response.shape[1] * pixel, case.data.shape[-1] * pixel
        roi = PhysicalRoi(
            bounds_yx_um=(2 * pixel, y_end, 2 * pixel, x_end),
            source_path=dataset.path,
            grid_origin_yx_um=(0, 0),
            pixel_size_yx_um=(pixel, pixel),
            position_indices=(0,),
            tile_footprints=(
                {
                    "time_index": 0,
                    "position_index": 0,
                    "origin_zyx_um": [0, 0, 0],
                    "bounds_yx_um": [0, y_end, 0, x_end],
                },
            ),
        )
        expected = expected[:, 2:, 2 : case.data.shape[-1]]
        origin = (0, round(2 * pixel, 3), round(2 * pixel, 3))
    if mode == "cropped":
        angle = np.deg2rad(case.geometry["theta"])
        crop_y = int(np.ceil(case.data.shape[1] * np.cos(angle)))
        raw_y = int(
            np.ceil(case.data.shape[0] * case.step + case.data.shape[1] * np.cos(angle))
        )
        core = expected[:, crop_y : raw_y - crop_y]
        expected = np.pad(core, ((0, 0), (0, (-core.shape[1]) % 4), (0, 0)))
        origin = (0, round(crop_y * pixel, 3), 0)
    return SimpleNamespace(
        dataset=dataset,
        acquisition=inspect_acquisition(dataset.path),
        roi=roi,
        crop=mode == "cropped",
        expected=expected,
        origin=origin,
        path=dataset.path.parent / "deskew_tail_deskewed.ome.zarr",
    )


@pytest.fixture
def stage_placement_acquisition(
    tmp_path,
    angle,
    mode,
    simulation_parameters,
    microscope_parameters,
    independent_photon_scene_factory,
):
    """Image one fixed point through the shared physical oblique camera model."""
    from dataclasses import replace

    from tests.fixtures.acquisition import opm_v2_frame_metadata, write_opm_v2_zarr
    from tests.simulation.forward_model import Acquisition, forward_model
    from tests.simulation.objects import deposit_point

    pixel_um = simulation_parameters["microscope"]["pixel_size_um"]
    step_um = 0.23
    scans, camera_height, camera_width = 40, 64, 48
    theta = np.deg2rad(angle)
    bead_zyx_um = np.asarray((20, 64, 24)) * pixel_um
    stage_z_values_um = -np.asarray((0, 6, 12)) * pixel_um
    microscope = replace(microscope_parameters, scan_step_um=step_um, angle_deg=angle)
    scene = independent_photon_scene_factory()
    density = np.zeros_like(scene.density)
    deposit_point(density, scene.axes, (0, 0, 0), 1.0)
    scan_axis = np.arange(scans) * step_um
    if mode == "stage":
        scan_axis = scan_axis[::-1].copy()
    column_axis = np.arange(camera_width) * pixel_um - bead_zyx_um[2]
    # Sensor coordinates refer to the same fixed object for every stage depth.
    nominal_axes = (
        scan_axis - bead_zyx_um[1] + bead_zyx_um[0] / np.tan(theta),
        np.arange(camera_height) * pixel_um - bead_zyx_um[0] / np.sin(theta),
        column_axis,
    )
    density = forward_model(
        density,
        Acquisition(
            scene.axes,
            sensor_axes=nominal_axes,
            camera_photon_budget=tuple(simulation_parameters["photon_budgets"][0]),
            output="object",
        ),
        microscope,
    )
    raw = np.zeros((1, 3, 1, scans, camera_height, camera_width), dtype=np.uint16)
    frames = []
    for position, stage_z_um in enumerate(stage_z_values_um):
        sensor_axes = (
            nominal_axes[0] + stage_z_um / np.tan(theta),
            nominal_axes[1] - stage_z_um / np.sin(theta),
            column_axis,
        )
        raw[0, position, 0] = forward_model(
            density,
            Acquisition(scene.axes, sensor_axes=sensor_axes, seed=position),
            microscope,
        )
        for scan in range(scans):
            if mode == "stage":
                # Forward acquisition: the stationary plane sees the specimen
                # move from negative scan-axis displacement back to zero.
                stage_scan_um = (scan - (scans - 1)) * step_um
            else:
                stage_scan_um = 0.0
            frames.append(
                opm_v2_frame_metadata(
                    index={"t": 0, "p": position, "c": 0, "z": scan},
                    daq_metadata={
                        "mode": mode,
                        "channel_states": [True],
                        "exposure_channels_ms": [10.0],
                        "laser_powers": [10.0],
                        "interleaved": True,
                        "blanking": True,
                        "current_channel": "488nm",
                        "scan_axis_step_um": step_um,
                        "image_mirror_position": float(scan),
                        "image_mirror_step_um": step_um,
                        "image_mirror_range_um": scans * step_um,
                    },
                    opm_metadata={
                        "angle_deg": angle,
                        "camera_Zstage_orientation": "normal",
                        "camera_XYstage_orientation": "normal",
                        "camera_mirror_orientation": "normal",
                    },
                    stage_metadata={
                        "x_pos": stage_scan_um,
                        "y_pos": 0.0,
                        "z_pos": stage_z_um,
                        "excess_image": False,
                    },
                    camera_shape_yx=(camera_height, camera_width),
                    pixel_size_um=pixel_um,
                    camera_offset=microscope.camera.offset_adu,
                    camera_conversion=microscope.camera.electrons_per_adu,
                    runner_time_ms=float(position * scans + scan)
                    * simulation_parameters["pipeline"]["plane_period_s"]
                    * 1000,
                    hardware_triggered=True,
                )
            )
    path = tmp_path / "physical_stage.zarr"
    write_opm_v2_zarr(
        path=path,
        raw_data=raw,
        labels=("t", "p", "c", "z", "y", "x"),
        chunks=(1, 1, 1, 1, camera_height, camera_width),
        frame_metadatas=frames,
    )
    return SimpleNamespace(path=path, pixel_um=pixel_um)
