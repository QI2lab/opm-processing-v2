"""Calibrated fluorescent specimens and provenance for OME-TIFF exports."""

from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pytest
from ome_types import from_xml, to_xml
from ome_types.model import AnnotationRef, CommentAnnotation
from yaozarrs import open_group, v05

from opm_processing.dataio.position_collection import (
    create_position_collection,
    update_group_attributes,
)
from opm_processing.dataio.processing_state import ProcessingState


@pytest.fixture
def fused_export(tmp_path, source_xml, fused_store_factory, simulated_camera):
    """Persist a two-channel object, depth projection and calibrated provenance."""
    dtype = np.uint16
    from opm_processing.imageprocessing.camera import camera_correct
    from tests.simulation.camera import digitize

    observed = simulated_camera((3, 515, 529), geometry="cartesian")
    camera = observed.microscope.camera
    raw_frames = np.stack(
        [
            np.stack(
                [
                    digitize(
                        observed.expectation * exposure,
                        camera,
                        seed=10 * time + channel,
                    )
                    for channel, exposure in enumerate(exposures)
                ]
            )
            for time, exposures in enumerate(((1.0, 2.3), (0.7, 1.4)))
        ]
    )
    truth = camera_correct(
        raw_frames, camera.offset_adu, camera.electrons_per_adu
    ).astype(dtype)
    projection = truth.max(axis=2, keepdims=True)
    raw_path = tmp_path / "sample.ome.zarr"
    raw = create_position_collection(
        raw_path,
        (2, 1, 2, 3, 515, 529),
        (0.4, 0.115, 0.115),
        channels=("488nm", "637nm"),
        dtype=dtype,
        attributes={
            "opm_v2": {
                "index_sizes": {"t": 2, "p": 1, "c": 2, "z": 3},
                "configuration": {
                    "acq_config": {
                        "opm_mode": "mirror",
                        "DAQ": {
                            "channel_states": [True, True],
                            "channel_exposures_ms": [10, 15],
                            "channel_powers": [12, 18],
                            "image_mirror_step_um": 0.4,
                        },
                    }
                },
            }
        },
    )
    raw.arrays[0].write(raw_frames).result()
    image_metadata = (
        open_group(raw_path)["0"]
        .ome_metadata()
        .model_copy(
            update={
                "omero": v05.Omero(
                    channels=[
                        v05.OmeroChannel(
                            label=name,
                            color=color,
                            window=v05.OmeroWindow(start=0, end=2000, min=0, max=2000),
                        )
                        for name, color in (("488nm α", "33CC99"), ("637nm", "FF3300"))
                    ]
                )
            }
        )
    )
    update_group_attributes(
        raw_path / "0",
        {"ome": image_metadata.model_dump(mode="json", exclude_none=True)},
    )
    companion = raw_path / "OME/METADATA.ome.xml"
    source_ome = from_xml(companion.read_text(encoding="utf-8"))
    if source_xml:
        source_ome.images[0].acquisition_date = datetime(
            2026, 10, 5, 19, 12, 32, tzinfo=UTC
        )
        source_ome.images[0].pixels.time_increment = 8.5
        source_ome.images[0].pixels.channels[0].emission_wavelength = (
            observed.microscope.wavelength_um * 1000
        )
        source_ome.structured_annotations.comment_annotations.append(
            CommentAnnotation(id="Annotation:sample", value="Simulated specimen μ")
        )
        source_ome.images[0].annotation_refs.append(
            AnnotationRef(id="Annotation:sample")
        )
        companion.write_text(to_xml(source_ome), encoding="utf-8")
    else:
        companion.unlink()

    fused_path = tmp_path / "sample_fused.ome.zarr"
    projection_path = tmp_path / "sample_max_z_fused.ome.zarr"
    spacing = (0.4, 0.115, 0.115)
    origin = (-40.5, 20.25, -11.75)
    for path, data, z_origin in (
        (fused_path, truth, origin[0]),
        (projection_path, projection, origin[0] + spacing[0]),
    ):
        fused_store_factory(
            data,
            name=path.name.removesuffix(".ome.zarr"),
            spacing=spacing,
            origin=(z_origin, *origin[1:]),
            factors=(1, 2),
            attributes={"specimen": "Simulated α"},
        )
        # Coarse pixels deliberately differ so exporting the wrong scale fails.
        open_group(path)["1"].to_tensorstore().write(0).result()
    processed = tmp_path / "sample_decon_deskewed.ome.zarr"
    state = ProcessingState.create(tmp_path / "sample.processing.json", raw_path)
    state.initialize_run(processed, configuration={"deconvolve": True}, overwrite=True)
    state.save_registration(processed, configuration={}, pairwise_metrics={})
    state.complete_registration(processed, fused_path=fused_path, tiles=[])
    state.set_registered_max_projection(processed, max_projection_path=projection_path)
    return SimpleNamespace(
        truth=truth,
        projection=projection,
        fused_path=fused_path,
        projection_path=projection_path,
        spacing=spacing,
        origin=origin,
        source_ome=source_ome,
        state=state,
    )


@pytest.fixture
def projection_sphere():
    """Share an analytical physical display object and its real storage setup."""
    spacing = (0.4, 0.2, 0.1)
    shape = (31, 61, 121)
    z, y, x = np.meshgrid(
        *[
            (np.arange(n) - (n - 1) / 2) * d
            for n, d in zip(shape, spacing, strict=False)
        ],
        indexing="ij",
    )
    sphere = (x * x + y * y + z * z <= 4**2).astype(float)
    return sphere, spacing


@pytest.fixture
def projection_export_case(position_store_factory, acquisition_metadata, stored_timing):
    """Share an analytical physical display object and its real storage setup."""
    first = np.linspace(0, 100, 10 * 48 * 64).reshape(10, 48, 64)
    data = np.stack([np.stack([first, first * 2]), np.stack([first * 0.5, first])])
    collection = position_store_factory(
        np.repeat(data[:, None], 2, axis=1).astype(np.float32),
        name="sample_decon_deskewed",
        voxel_size_um=(0.1, 0.1, 0.1),
        channels=("488", "637"),
        time_interval_s=0.5 if stored_timing else None,
    )
    source_path = collection.path
    acquisition = replace(
        acquisition_metadata,
        acquired_plane_count=None if stored_timing else 25,
        plane_period_s=None if stored_timing else 0.02,
    )
    return SimpleNamespace(
        data=data,
        source_path=source_path,
        acquisition=acquisition,
    )


@pytest.fixture
def planar_fused_export(fused_store_factory, simulated_camera):
    """Persist one real single-channel planar object with explicit physical timing."""
    from opm_processing.imageprocessing.camera import camera_correct

    observed = simulated_camera((1, 17, 19), geometry="cartesian")
    microscope = observed.microscope
    spacing = (
        microscope.scan_step_um,
        microscope.pixel_size_um,
        microscope.pixel_size_um,
    )
    truth = camera_correct(
        observed.raw, microscope.camera.offset_adu, microscope.camera.electrons_per_adu
    ).astype(np.uint16)[None, None]
    for name in ("sample_fused", "sample_max_z_fused"):
        fused_store_factory(
            truth,
            name=name,
            spacing=spacing,
            time_step_s=3.5,
            time_origin_s=9.75,
            chunks=(1, 1, 1, 17, 19),
            omero=v05.Omero(
                channels=[
                    v05.OmeroChannel(
                        label="simulated points",
                        color="00FF00",
                        window=v05.OmeroWindow(start=0, end=100, min=0, max=100),
                    )
                ]
            ),
        )
    return SimpleNamespace(truth=truth, spacing=spacing)
