"""Read, write, and convert OPM acquisition data."""

from . import acquisition as acquisition
from . import metadata as metadata
from . import position_collection as position_collection
from .acquisition import (
    AcquisitionMetadata,
    ChannelMetadata,
    acquisition_stem,
    inspect_acquisition,
    open_acquisition_datastore,
)

__all__ = [
    "AcquisitionMetadata",
    "ChannelMetadata",
    "acquisition_stem",
    "acquisition",
    "inspect_acquisition",
    "metadata",
    "open_acquisition_datastore",
    "position_collection",
]
