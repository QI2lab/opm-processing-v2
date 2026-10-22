"""Physical snapshot cadence shared by processed OME-Zarr writers."""

from math import isfinite

from yaozarrs import v05


def time_axis(interval_s: float | None) -> tuple[dict, float]:
    """Return an NGFF time axis and scale without inventing a physical interval.

    Parameters
    ----------
    interval_s : float or None
        Positive finite snapshot interval in seconds; None means unknown timing.

    Returns
    -------
    axis : dict
        Time-axis descriptor, with unit second only when timing is known.
    scale : float
        Physical interval or a unitless index scale of one.
    """
    if interval_s is None:
        return {"name": "t", "type": "time"}, 1.0
    if isinstance(interval_s, bool) or not isfinite(interval_s) or interval_s <= 0:
        raise ValueError("time_interval_s must be finite and positive")
    return {"name": "t", "type": "time", "unit": "second"}, float(interval_s)


def hardware_volume_interval(
    acquired_planes: int | None, plane_period_s: float | None
) -> float | None:
    """Validate recorded hardware timing and return N_acquired_planes * dt_plane_s.

    Parameters
    ----------
    acquired_planes : int or None
        Original acquired count, including discarded planes; never processed Z size.
    plane_period_s : float or None
        Hardware period in seconds per scan position, including acquired channels.

    Returns
    -------
    float or None
        Physical volume interval, or None when both timing fields are absent.
    """
    if acquired_planes is None and plane_period_s is None:
        return None
    if (
        type(acquired_planes) is not int
        or acquired_planes <= 0
        or plane_period_s is None
    ):
        raise ValueError(
            "Hardware timing requires a positive integer acquired-plane count and plane period"
        )
    time_axis(plane_period_s)
    interval = acquired_planes * plane_period_s
    time_axis(interval)
    return interval


def read_time_interval(multiscale: v05.Multiscale) -> float | None:
    """Read a physical NGFF T scale in seconds, preserving unknown time units.

    Parameters
    ----------
    multiscale : yaozarrs.v05.Multiscale
        Producer image metadata; all datasets must have the same physical T scale.

    Returns
    -------
    float or None
        Seconds per snapshot, or None for an axis without physical units.
    """
    index = next(
        (i for i, axis in enumerate(multiscale.axes) if axis.name == "t"), None
    )
    if index is None or multiscale.axes[index].unit is None:
        return None
    factors = {
        "second": 1.0,
        "millisecond": 0.001,
        "microsecond": 1e-6,
        "nanosecond": 1e-9,
        "minute": 60.0,
        "hour": 3600.0,
    }
    unit = multiscale.axes[index].unit
    if unit not in factors:
        raise ValueError(f"Unsupported physical time unit {unit}")
    values = [
        dataset.scale_transform.scale[index] * factors[unit]
        for dataset in multiscale.datasets
    ]
    for value in values:
        time_axis(value)
    if any(value != values[0] for value in values):
        raise ValueError("Pyramid levels disagree on physical time scale")
    return values[0]
