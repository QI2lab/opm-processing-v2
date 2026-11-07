"""Shared OME-NGFF coordinate and pyramid helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from yaozarrs import v05

from opm_processing.dataio.timing import time_axis

if TYPE_CHECKING:
    from collections.abc import Sequence

SPATIAL_DECIMALS = 3


def round_spatial(value: float) -> float:
    """Round one physical coordinate or spacing to one nanometer.

    Parameters
    ----------
    value : float
        Physical coordinate or spacing in micrometers.

    Returns
    -------
    float
        Value rounded to three decimal places.
    """
    return round(float(value), SPATIAL_DECIMALS)


def round_spatial_values(values: Sequence[float]) -> tuple[float, ...]:
    """Round physical coordinate values to three decimal places.

    Parameters
    ----------
    values : Sequence[float]
        Physical coordinates or spacings in micrometers.

    Returns
    -------
    tuple[float, ...]
        Rounded values in the original axis order.
    """
    return tuple(round_spatial(value) for value in values)


def round_tczyx_transform(values: Sequence[float]) -> list[float]:
    """Round only the spatial ZYX values of a TCZYX transform.

    Parameters
    ----------
    values : Sequence[float]
        Five TCZYX scale or translation values; spatial units are micrometers.

    Returns
    -------
    list[float]
        Transform preserving T and C and rounding the spatial values.
    """
    return [float(values[0]), float(values[1]), *round_spatial_values(values[2:])]


def tczyx_image_metadata(
    spatial_scales: Sequence[Sequence[float]],
    spatial_origins: Sequence[Sequence[float]],
    *,
    time_interval_s: float | None = None,
    time_origin_s: float = 0.0,
    name: str | None = None,
    omero: v05.Omero | None = None,
    round_coordinates: bool = True,
) -> v05.Image:
    """Describe a root TCZYX image using calibrated per-level spatial transforms.

    Parameters
    ----------
    spatial_scales : sequence of sequence of float
        Per-level ZYX voxel spacings in micrometers.
    spatial_origins : sequence of sequence of float
        Per-level ZYX origins in micrometers.
    time_interval_s : float or None
        Recorded snapshot interval; None keeps the time axis unitless.
    time_origin_s : float
        Time coordinate of the first stored snapshot.
    name : str or None
        Optional multiscale image name.
    omero : yaozarrs.v05.Omero or None
        Optional channel display metadata.
    round_coordinates : bool
        Round spatial transforms to the processing precision. False preserves
        explicitly supplied coordinates, including historical acquisition data.

    Returns
    -------
    yaozarrs.v05.Image
        Canonical axes and consecutive dataset paths with the supplied transforms.
    """
    axis, interval = time_axis(time_interval_s)
    datasets = []
    for level, (scale, origin) in enumerate(
        zip(spatial_scales, spatial_origins, strict=True)
    ):
        scale_tczyx = [interval, 1.0, *(float(value) for value in scale)]
        origin_tczyx = [float(time_origin_s), 0.0, *(float(value) for value in origin)]
        if round_coordinates:
            scale_tczyx = round_tczyx_transform(scale_tczyx)
            origin_tczyx = round_tczyx_transform(origin_tczyx)
        datasets.append(
            v05.Dataset(
                path=str(level),
                coordinateTransformations=[
                    v05.ScaleTransformation(scale=scale_tczyx),
                    v05.TranslationTransformation(translation=origin_tczyx),
                ],
            )
        )
    return v05.Image(
        multiscales=[
            v05.Multiscale(
                name=name,
                axes=[
                    axis,
                    {"name": "c", "type": "channel"},
                    *(
                        {"name": dimension, "type": "space", "unit": "micrometer"}
                        for dimension in "zyx"
                    ),
                ],
                datasets=datasets,
            )
        ],
        omero=omero,
    )


def downsample_yx(
    image: np.ndarray,
    factor: int,
    method: str = "stride",
) -> np.ndarray:
    """Downsample the last two axes, using striding by default.

    Parameters
    ----------
    image : np.ndarray
        Image with YX as its last two dimensions.
    factor : int
        Positive integer reduction along Y and X.
    method : str
        stride selects pixels; block_mean averages complete blocks.

    Returns
    -------
    np.ndarray
        Reduced image preserving the input dtype and leading axes.
    """
    factor = int(factor)
    if factor < 1:
        raise ValueError("Pyramid factors must be positive integers")
    array = np.asarray(image)
    if factor == 1:
        return array
    if method == "stride":
        return array[..., ::factor, ::factor]
    if method != "block_mean":
        raise ValueError('Pyramid downsampling must be "stride" or "block_mean"')
    y_size = (array.shape[-2] // factor) * factor
    x_size = (array.shape[-1] // factor) * factor
    if y_size == 0 or x_size == 0:
        raise ValueError("Pyramid factor exceeds the image dimensions")
    trimmed = array[..., :y_size, :x_size]
    reshaped = trimmed.reshape(
        *trimmed.shape[:-2],
        y_size // factor,
        factor,
        x_size // factor,
        factor,
    )
    reduced = reshaped.mean(axis=(-3, -1))
    if np.issubdtype(array.dtype, np.integer):
        reduced = np.rint(reduced)
    return reduced.astype(array.dtype, copy=False)


def pyramid_yx_sampling(multiscale: v05.Multiscale) -> tuple[tuple[int, ...], str]:
    """Recover YX reduction factors and sampling from recorded grid coordinates.

    Parameters
    ----------
    multiscale : yaozarrs.v05.Multiscale
        Pyramid with equal integer reductions of the final YX axes.

    Returns
    -------
    factors : tuple of int
        Reduction at each level relative to level zero.
    method : str
        ``stride`` for aligned sample origins or ``block_mean`` for block centers.

    Raises
    ------
    ValueError
        Scale factors or origins disagree with either supported sampling method.
    """
    datasets = multiscale.datasets
    base_scale = np.asarray(datasets[0].scale_transform.scale[-2:], dtype=float)
    base_origin = np.asarray(
        datasets[0].translation_transform.translation[-2:]
        if datasets[0].translation_transform is not None
        else (0, 0),
        dtype=float,
    )
    tolerance = 10**-SPATIAL_DECIMALS
    factors, methods = [], set()
    for dataset in datasets:
        scale = np.asarray(dataset.scale_transform.scale[-2:], dtype=float)
        factor = np.rint(scale / base_scale).astype(int)
        if (
            np.any(factor < 1)
            or factor[0] != factor[1]
            or not np.allclose(scale, factor * base_scale, atol=tolerance, rtol=0)
        ):
            raise ValueError("Pyramids require equal integer YX scale factors")
        factor = int(factor[0])
        factors.append(factor)
        origin = np.asarray(
            dataset.translation_transform.translation[-2:]
            if dataset.translation_transform is not None
            else (0, 0),
            dtype=float,
        )
        if np.allclose(origin, base_origin, atol=tolerance, rtol=0):
            if factor != 1:
                methods.add("stride")
        elif np.allclose(
            origin,
            base_origin + 0.5 * (factor - 1) * base_scale,
            atol=tolerance,
            rtol=0,
        ):
            methods.add("block_mean")
        else:
            raise ValueError(
                "Pyramid origins do not match stride or block-mean sampling"
            )
    if len(methods) > 1:
        raise ValueError("Pyramid levels disagree on YX sampling")
    return tuple(factors), next(iter(methods), "stride")
