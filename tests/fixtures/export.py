"""Calibrated fluorescent specimens and provenance for OME-TIFF exports."""

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
        dtype=dtype,
    )
