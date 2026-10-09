"""Calibrated fluorescent specimens and provenance for OME-TIFF exports."""

from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pytest
import zarr
from ome_types import from_xml, to_xml
from ome_types.model import AnnotationRef, CommentAnnotation
from yaozarrs import open_group, v05

from opm_processing.dataio.position_collection import create_position_collection
from opm_processing.dataio.processing_state import ProcessingState


@pytest.fixture
def fused_export(tmp_path, source_xml, fused_store_factory):
    """Persist a two-channel object, depth projection and calibrated provenance.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary acquisition and processed image directory.
    source_xml : bool
        Test parameter selecting rich OME metadata or NGFF-only acquisition.
    fused_store_factory
        Shared writer for calibrated root images and their pyramids.

    Returns
    -------
    types.SimpleNamespace
        Known pixels, image paths, calibration, source OME and processing state.
    """
    dtype = np.uint16
    z, y, x = np.mgrid[:3, :515, :529]
    object_pixels = 900 * np.exp(
        -((z - 1) ** 2 + ((y - 251) / 45) ** 2 + ((x - 287) / 39) ** 2)
    )
    # Add known emitters at TIFF tile boundaries and at both partial-tile edges.
    object_pixels[:, 511:514, 510:516] += 233
    object_pixels[:, -1, -1] = (7, 11, 17)
    truth = np.stack(
        [
            np.stack([object_pixels, object_pixels * 2.3]),
            np.stack([object_pixels * 0.7, object_pixels * 1.4]),
        ]
    ).astype(dtype)
    projection = truth.max(axis=2, keepdims=True)
    raw_path = tmp_path / "sample.ome.zarr"
    raw = create_position_collection(
        raw_path,
        (2, 1, 2, 3, 5, 7),
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
    raw.arrays[0].write(truth[..., :5, :7]).result()
    raw_group = zarr.open_group(raw_path, mode="a")["0"]
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
    raw_group.attrs["ome"] = image_metadata.model_dump(mode="json", exclude_none=True)
    companion = raw_path / "OME/METADATA.ome.xml"
    source_ome = from_xml(companion.read_text(encoding="utf-8"))
    if source_xml:
        source_ome.images[0].acquisition_date = datetime(
            2026, 10, 5, 19, 12, 32, tzinfo=UTC
        )
        source_ome.images[0].pixels.time_increment = 8.5
        source_ome.images[0].pixels.channels[0].emission_wavelength = 525
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
    spacing = (0.23, 0.115, 0.119)
    origin = (-40.5, 20.25, -11.75)
    for path, data, z_origin in (
        (fused_path, truth, origin[0]),
        (projection_path, projection, origin[0] + 0.23),
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
        zarr.open_group(path, mode="r+")["1"][:] = 0
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
