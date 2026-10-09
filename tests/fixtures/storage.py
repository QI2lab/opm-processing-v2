"""Persist known objects as processed position collections or fused NGFF images."""

import numpy as np
import pytest
from yaozarrs import v05
from yaozarrs.write.v05 import prepare_image

from opm_processing.dataio.position_collection import create_position_collection


@pytest.fixture
def memory_store(tensorstore_dataset):
    """Return zero-filled dataset mocks for numerical output buffers.

    Parameters
    ----------
    tensorstore_dataset : callable
        Shared TensorStore double factory with functional reads and writes.

    Returns
    -------
    callable
        Allocator for a mocked output buffer with the requested shape and dtype.
    """

    def create(shape, dtype=np.uint16):
        """Allocate a zero-filled TensorStore dataset double.

        Parameters
        ----------
        shape : tuple of int
            Required image or output buffer dimensions.
        dtype : numpy dtype
            Element type of the numerical buffer.

        Returns
        -------
        unittest.mock.MagicMock
            Spec-constrained writable dataset with known backing pixels.
        """
        return tensorstore_dataset(np.zeros(shape, dtype=dtype))

    return create


@pytest.fixture
def position_store_factory(tmp_path):
    """Return a writer for calibrated processed objects in TPCZYX order.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory containing each independent processed dataset.

    Returns
    -------
    callable
        Writer accepting known pixels, physical spacing and position metadata.
    """

    def create(
        data, *, name="sample_deskewed", voxel_size_um=(1.0, 1.0, 1.0), **settings
    ):
        """Persist each position with common spacing and optional NGFF settings.

        Parameters
        ----------
        data : numpy.ndarray
            Known processed fluorescence in TPCZYX order.
        name : str
            Dataset basename in the temporary test directory.
        voxel_size_um : tuple of float
            Laboratory voxel spacing in ZYX order.
        **settings
            Position collection settings, including channels and stage positions.

        Returns
        -------
        PositionCollection
            Written collection and its per-position arrays.
        """
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
    """Return a writer for a known TCZYX object and its calibrated NGFF pyramid.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory containing saved fused images.

    Returns
    -------
    callable
        Writer accepting pixels, pyramid factors and calibrated voxel coordinates.
    """

    def create(
        data,
        *,
        name="sample_fused",
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        time_step_s=None,
        time_origin_s=0.0,
        factors=(1,),
        attributes=None,
        omero=None,
        chunks=(1, 1, 1, 128, 128),
    ):
        """Write a root image with explicit physical coordinates at each scale.

        Parameters
        ----------
        data : numpy.ndarray
            Known fluorescence in TCZYX order.
        name : str
            Dataset basename in the temporary test directory.
        spacing, origin : tuple of float
            Base voxel spacing and voxel-center origin in ZYX micrometers.
        time_step_s : float or None
            Timepoint interval in seconds; None leaves time units unspecified
            so acquisition metadata provides the timing.
        time_origin_s : float
            First timepoint offset in seconds when time_step_s is provided.
        factors : tuple of int
            Absolute isotropic stride factors; one denotes the original pixels.
        attributes : dict or None
            Additional root NGFF attributes.
        omero : yaozarrs.v05.Omero or None
            Channel labels, colors and display limits stored with the image.
        chunks : tuple of int
            TCZYX chunk dimensions for the persisted image.

        Returns
        -------
        pathlib.Path
            Saved image path; callers reopen it through the real workflow.
        """
        path = tmp_path / f"{name}.ome.zarr"
        levels = [data[..., ::factor, ::factor, ::factor] for factor in factors]
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
                        translation=[time_origin_s, 0, *origin]
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
