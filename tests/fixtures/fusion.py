"""Minimal numerical fusion operators with explicitly supplied geometry and data."""

import numpy as np
import pytest

from opm_processing.imageprocessing.tilefusion import TileFusion


@pytest.fixture
def fusion_operator(tensorstore_dataset):
    """Return a factory for isolated fusion calculations without file discovery.

    Numerical unit tests supply only the state consumed by their calculation.
    Integration tests construct TileFusion normally from a persisted dataset.

    Parameters
    ----------
    tensorstore_dataset : callable
        Shared dataset double retaining known source pixels and storage behavior.

    Returns
    -------
    callable
        Factory populating the state consumed by one numerical fusion calculation.
    """

    def create(operator=TileFusion, *, source_tiles=None, **attributes):
        """Construct an operator from a complete, visible unit-test configuration.

        Parameters
        ----------
        operator : type
            Fusion class whose numerical method is under test.
        source_tiles : sequence of numpy.ndarray or None
            CZYX tile objects for the mocked TensorStore reader. These supply
            array dimensions, unity blending profiles and per-position sources.
        **attributes
            Geometry, source arrays and execution settings required by that method.

        Returns
        -------
        object
            Uninitialized operator populated with the supplied attributes.
        """
        if source_tiles is not None:
            channels, z_dim, y_dim, x_dim = source_tiles[0].shape
            attributes = {
                "time_dim": 1,
                "position_dim": len(source_tiles),
                "channels": channels,
                "z_dim": z_dim,
                "y_dim": y_dim,
                "x_dim": x_dim,
                "_is_2d": False,
                "z_profile": np.ones(z_dim, np.float32),
                "y_profile": np.ones(y_dim, np.float32),
                "x_profile": np.ones(x_dim, np.float32),
                "position_arrays": tuple(
                    tensorstore_dataset(tile[None]) for tile in source_tiles
                ),
                **attributes,
            }
        instance = operator.__new__(operator)
        instance.__dict__.update(attributes)
        return instance

    return create
