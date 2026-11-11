"""Bounded additive-color views of native per-channel tissue projections."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping


MAX_COMPOSITE_TILE_PIXELS = 16 * 1024**2


class CompositeLayerProxy:
    """Expose a native RGB image while materializing only requested XY tiles."""

    def __init__(
        self,
        images: Mapping[int, Any],
        colors: Mapping[int, tuple[int, int, int]],
        display_limits: Mapping[int, tuple[float, float]],
    ) -> None:
        """Validate matching native channels, colors, and linear display limits."""
        if (
            not images
            or set(images) != set(colors)
            or set(images) != set(display_limits)
        ):
            raise ValueError(
                "Composite images, colors, and display limits must identify the same channels"
            )
        self._images = dict(images)
        self._colors = {}
        self._limits = {}
        shape = None
        for channel in sorted(images):
            image = images[channel]
            current = tuple(int(value) for value in image.shape)
            if (
                len(current) != 2
                or min(current) < 1
                or (shape is not None and current != shape)
            ):
                raise ValueError(
                    "Composite channels must share one nonempty native XY shape"
                )
            if (
                np.dtype(getattr(image.dtype, "numpy_dtype", image.dtype)).kind
                not in "uif"
            ):
                raise ValueError(
                    "Composite source channels must contain real numeric intensities"
                )
            shape = current
            color = colors[channel]
            if len(color) != 3 or any(
                not isinstance(value, (int, np.integer))
                or isinstance(value, bool)
                or not 0 <= value <= 255
                for value in color
            ):
                raise ValueError(
                    "Composite channel colors must contain three RGB integers in 0..255"
                )
            self._colors[channel] = tuple(int(value) for value in color)
            try:
                low, high = (float(value) for value in display_limits[channel])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "Composite display limits must contain two finite numbers"
                ) from error
            if not np.isfinite((low, high)).all() or high < low:
                raise ValueError(
                    "Composite display limits must be finite and increasing"
                )
            self._limits[channel] = low, high
        self.shape = (*shape, 3)
        self.dtype = np.dtype(np.uint8)
        self.ndim = 3

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        """Prevent implicit materialization of a full native composite."""
        raise TypeError(
            "Read the composite through bounded XY slices before converting to an array"
        )

    def __getitem__(self, key: Any) -> np.ndarray:
        """Normalize and combine only the selected XY samples and RGB components."""
        if (
            not isinstance(key, tuple)
            or len(key) not in (2, 3)
            or not all(isinstance(value, slice) for value in key[:2])
        ):
            raise IndexError(
                "Composite reads require Y and X slices, with an optional RGB selector"
            )
        y, x = key[:2]
        ny, nx = (
            len(range(*selection.indices(length)))
            for selection, length in zip((y, x), self.shape[:2], strict=True)
        )
        if ny * nx > MAX_COMPOSITE_TILE_PIXELS:
            raise ValueError(
                "Read the native composite in bounded XY tiles or strided previews"
            )
        component = key[2] if len(key) == 3 else slice(None)
        if (
            not isinstance(component, (slice, int, np.integer))
            and component is not Ellipsis
        ):
            raise IndexError("Composite RGB selection must be a channel index or slice")
        rgb = np.zeros((ny, nx, 3), dtype=np.float64)
        if ny and nx:
            for channel in sorted(self._images):
                low, high = self._limits[channel]
                normalized = np.asarray(self._images[channel][y, x], dtype=np.float64)
                if normalized.shape != (ny, nx):
                    raise ValueError(
                        "Composite source returned an incompatible XY tile"
                    )
                normalized = np.clip(
                    (normalized - low) / max(high - low, np.finfo(float).eps), 0, 1
                )
                for axis, color in enumerate(self._colors[channel]):
                    if color:
                        rgb[..., axis] += normalized * color
        result = np.rint(np.clip(rgb, 0, 255)).astype(np.uint8)
        return result if component is Ellipsis else result[..., component]


def make_composite_layers(
    images_by_channel: Mapping[int, Mapping[int, Any]],
    colors: Mapping[int, tuple[int, int, int]],
    display_limits: Mapping[int, Mapping[int, tuple[float, float]]],
) -> dict[int, CompositeLayerProxy]:
    """Create lazy additive RGB views for corresponding native acquisition layers.

    Parameters
    ----------
    images_by_channel : mapping
        Native two-dimensional layer images keyed by channel and logical depth.
    colors : mapping
        RGB display colors keyed by actual acquisition channel identifiers.
    display_limits : mapping
        Linear low/high intensity limits keyed by channel and logical depth.
        Equal limits keep a constant background black.

    Returns
    -------
    dict
        Logical depths mapped to bounded, native-resolution RGB image proxies.
    """
    if (
        not images_by_channel
        or set(images_by_channel) != set(colors)
        or set(images_by_channel) != set(display_limits)
    ):
        raise ValueError(
            "Composite images, colors, and display limits must identify the same channels"
        )
    depths = set(next(iter(images_by_channel.values())))
    if not depths or any(
        set(images_by_channel[channel]) != depths
        or set(display_limits[channel]) != depths
        for channel in images_by_channel
    ):
        raise ValueError(
            "Composite channels and display limits must identify the same acquisition depths"
        )
    return {
        depth: CompositeLayerProxy(
            {channel: layers[depth] for channel, layers in images_by_channel.items()},
            colors,
            {channel: limits[depth] for channel, limits in display_limits.items()},
        )
        for depth in sorted(depths)
    }
