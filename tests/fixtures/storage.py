"""Persist known objects as processed position collections or fused NGFF images."""

from types import SimpleNamespace

import numpy as np
import pytest
from skimage.measure import block_reduce
from yaozarrs import v05
from yaozarrs.write.v05 import prepare_image

from opm_processing.dataio.position_collection import create_position_collection


@pytest.fixture
def position_store_factory(tmp_path):
    """Return a writer for calibrated processed objects in TPCZYX order."""

    def create(
        data, *, name="sample_deskewed", voxel_size_um=(1.0, 1.0, 1.0), **settings
    ):
        """Persist each position with common spacing and optional NGFF settings."""
        collection = create_position_collection(
            tmp_path / f"{name}.ome.zarr",
            data.shape,
            voxel_size_um,
            dtype=data.dtype,
            **settings,
        )
        for position, array in enumerate(collection.arrays):
            array.write(data[:, position]).result()
        return collection

    return create


@pytest.fixture
def fused_store_factory(tmp_path):
    """Return a writer for a known TCZYX object and its calibrated NGFF pyramid."""

    def create(
        data,
        *,
        name="sample_fused",
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        level_origins=None,
        downsample="stride",
        time_step_s=None,
        time_origin_s=0.0,
        factors=(1,),
        attributes=None,
        omero=None,
        chunks=(1, 1, 1, 128, 128),
    ):
        """Write a root image with explicit physical coordinates at each scale."""
        path = tmp_path / f"{name}.ome.zarr"
        levels = [
            data[..., ::factor, ::factor, ::factor]
            if downsample == "stride"
            else block_reduce(
                data,
                block_size=(1, 1, factor, factor, factor),
                func=np.mean,
            ).astype(data.dtype)
            for factor in factors
        ]
        datasets = [
            v05.Dataset(
                path=str(index),
                coordinateTransformations=[
                    v05.ScaleTransformation(
                        scale=[
                            1 if time_step_s is None else time_step_s,
                            1,
                            *(factor * value for value in spacing),
                        ]
                    ),
                    v05.TranslationTransformation(
                        translation=[
                            time_origin_s,
                            0,
                            *(
                                origin
                                if level_origins is None
                                else level_origins[index]
                            ),
                        ]
                    ),
                ],
            )
            for index, factor in enumerate(factors)
        ]
        image = v05.Image(
            multiscales=[
                v05.Multiscale(
                    axes=[
                        v05.TimeAxis(
                            name="t", unit=None if time_step_s is None else "second"
                        ),
                        v05.ChannelAxis(name="c"),
                        *(
                            v05.SpaceAxis(name=axis, unit="micrometer")
                            for axis in "zyx"
                        ),
                    ],
                    datasets=datasets,
                )
            ],
            omero=omero,
        )
        _, arrays = prepare_image(
            path,
            image,
            [(level.shape, np.dtype(data.dtype)) for level in levels],
            chunks=chunks,
            writer="tensorstore",
            extra_attributes=attributes or {},
            overwrite=True,
        )
        for index, level in enumerate(levels):
            arrays[str(index)].write(level).result()
        return path

    return create


def ptv_manifest(image_paths=("0",)):
    """Return explicit synthetic processing provenance, without implied calibration."""
    return {
        "schema_version": 1,
        "dataset_id": "synthetic-processed",
        "producer_version": "0.3.0",
        "product_kind": "deconvolved_deskewed",
        "processing_history": [{"operation": "synthetic_processed_fixture"}],
        "registration": {"applied": False, "coordinate_frame": "stored_processed"},
        "calibration": {"gain_offset_applied": False, "uncertainty_validated": False},
        "images": [
            {
                "path": name,
                "source_indices": [0, 1, 3],
                "volume_status_by_channel": [["valid", "invalid", "incomplete"]],
            }
            for name in image_paths
        ],
    }


@pytest.fixture
def ptv_collection(tmp_path):
    """Allocate a tiny actual processed writer output; no numerical processing runs."""
    path = tmp_path / "processed.ome.zarr"
    create_position_collection(
        path,
        (3, 1, 1, 7, 12, 14),
        (0.23, 0.115, 0.15),
        spatial_offset_um=(-2.1, 3.2, -4.3),
        time_interval_s=0.085,
        multiscale_factors_yx=(2,),
        channels=["beads"],
        attributes={
            "opm_processing": {
                "timing": {
                    "schema_version": 1,
                    "n_acquired_planes": 17,
                    "dt_plane_s": 0.005,
                    "dt_volume_s": 0.085,
                    "source": "synthetic_trigger",
                }
            }
        },
    )
    return path


@pytest.fixture
def processed_directory_factory(tmp_path):
    """Create derived-store identities for source/output path resolution tests."""

    def create(stem, *, labels=("deskewed", "fused", "max_z_deskewed", "max_z_fused")):
        """Create only the requested processed-store directory structure."""
        output_dir = tmp_path / "processed"
        output_dir.mkdir()
        for label in labels:
            create_position_collection(
                output_dir / f"{stem}_{label}.ome.zarr", (1, 1, 1, 1, 1, 1), (1, 1, 1)
            )
        return output_dir

    return create


@pytest.fixture(params=["stride", "block_mean"])
def registered_projection_case(
    request, tmp_path, fused_store_factory, simulated_camera
):
    """Persist raw, registered and stale projection pyramids with distinct origins."""
    downsample = request.param
    shape = (2, 2, 5, 7, 9) if downsample == "stride" else (2, 2, 6, 8, 10)
    spacing = (0.81234, 0.51234, 0.25123)
    origin = (4.51234, -3.01234, 12.25123)
    origins = (
        origin,
        tuple(
            np.asarray(origin)
            + (0.5 * np.asarray(spacing) if downsample == "block_mean" else 0)
        ),
    )
    registered = np.broadcast_to(
        simulated_camera(shape[-3:], geometry="cartesian").raw, shape
    ).copy()
    paths = [
        fused_store_factory(
            data,
            name=name,
            spacing=spacing,
            level_origins=origins,
            downsample=downsample,
            factors=(1, 2),
            chunks=(1, 1, 2, 4, 5),
        )
        for name, data in (
            ("sample", np.zeros(shape, np.uint16)),
            ("sample_fused", registered),
            ("sample_max_z_fused", np.zeros((*shape[:2], 1, *shape[-2:]), np.uint16)),
        )
    ]
    from opm_processing.dataio.processing_state import ProcessingState

    processed_path = tmp_path / "sample_projection.ome.zarr"
    state = ProcessingState.create(tmp_path / "sample.processing.json", paths[0])
    state.initialize_run(processed_path, configuration={}, overwrite=True)
    state.save_registration(processed_path, configuration={}, pairwise_metrics={})
    state.complete_registration(processed_path, fused_path=paths[1], tiles=())
    return SimpleNamespace(
        raw_path=paths[0],
        output_path=paths[2],
        processed_path=processed_path,
        truth=registered,
        downsample=downsample,
        scales=tuple(
            [1.0, 1.0, *(factor * value for value in spacing)] for factor in (1, 2)
        ),
        translations=tuple([0.0, 0.0, *origin] for origin in origins),
        state_path=state.path,
    )
