"""Persist known objects as processed position collections or fused NGFF images."""

from types import SimpleNamespace

import numpy as np
import pytest
from skimage.measure import block_reduce
from yaozarrs.write.v05 import prepare_image

from opm_processing.dataio.ngff import tczyx_image_metadata
from opm_processing.dataio.position_collection import create_position_collection


@pytest.fixture(params=["stride", "block_mean"])
def spatial_metadata_collection(request, position_store_factory):
    """Write one calibrated object with fractional stage positions and a pyramid."""
    return position_store_factory(
        np.zeros((1, 1, 1, 1, 17, 21), np.uint16),
        name="precise_projection",
        voxel_size_um=(0.34567, 0.11549, 0.11651),
        stage_positions=((1.23456, 2.34567, 3.45678),),
        spatial_offset_um=(4.56789, 5.67891, 6.78912),
        multiscale_factors_yx=(2, 4),
        multiscale_downsample=request.param,
    )


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
        image = tczyx_image_metadata(
            [tuple(factor * value for value in spacing) for factor in factors],
            [
                origin if level_origins is None else level_origins[index]
                for index in range(len(factors))
            ],
            time_interval_s=time_step_s,
            time_origin_s=time_origin_s,
            omero=omero,
            round_coordinates=False,
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
