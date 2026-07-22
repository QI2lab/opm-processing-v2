"""Fuse maximum-projection image tiles using stage positions."""

from pathlib import Path
from collections.abc import Sequence

import numpy as np
from tqdm import tqdm

from yaozarrs import DimSpec, v05
from yaozarrs.write.v05 import prepare_image


class MaxTileFusion:
    """
    A class for fusing multiple overlapping image tiles stored in a TensorStore dataset.

    Parameters
    ----------
    tensorstore_path : str or Path
        Path to the input TensorStore dataset.
    tile_positions : list of tuple of float
        List of (y, x) coordinates representing tile positions.
    output_path : Path
        Path to the output TensorStore dataset.
    pixel_size : tuple of float
        Pixel size (y, x) in physical units.
    pad_yx : list of int, default  = [0, 0].
        Padding in y and x dimensions already applied to the dataset
    time_range: list of int, default = None
    """

    def __init__(
        self,
        ts_dataset: str | Path,
        tile_positions: list[float, float],
        output_path: str | Path,
        pixel_size: tuple[float, float],
        pad_yx: Sequence[int] = (0, 0),
        time_range: tuple[int, int] | None = None,
        blend_pixels: tuple[int, int] = (380, 380),
        chunk_size: int = 512,
        padding_multiple: int = 8,
    ):
        """Initialize a maximum-projection tile fusion operation.

        Parameters
        ----------
        ts_dataset
            Per-position TensorStore arrays to fuse.
        tile_positions
            Physical YX position of each tile.
        output_path
            Destination for the fused image.
        pixel_size
            Physical YX pixel spacing.
        pad_yx
            Existing YX padding to remove from each tile.
        time_range
            Optional half-open time range to fuse.
        blend_pixels
            Feathering width along Y and X.
        chunk_size
            Spatial output chunk size.
        padding_multiple
            Multiple to which the fused shape is padded.

        Returns
        -------
        None
            No value is returned.
        """
        self.pad_y = pad_yx[0]
        self.pad_x = pad_yx[1]

        self.ts_dataset = tuple(ts_dataset)

        self.tile_positions = np.array(tile_positions)
        self.output_path = Path(output_path)
        self.pixel_size = pixel_size

        self.time_dim, self.channels, self.z_dim, height, width = self.ts_dataset[
            0
        ].shape
        self.position_dim = len(self.ts_dataset)
        height -= 2 * self.pad_y
        width -= 2 * self.pad_x

        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        if padding_multiple < 1:
            raise ValueError("padding_multiple must be at least 1")
        if len(blend_pixels) != 2 or any(value < 0 for value in blend_pixels):
            raise ValueError("blend_pixels must contain two nonnegative values")
        self.padding_multiple = int(padding_multiple)
        self.chunk_size = int(chunk_size)
        self.blend_pixels = tuple(int(value) for value in blend_pixels)

        self.tile_shape = (height, width)
        self.time_range = time_range

        self.fused_shape, self.offset = self.compute_fused_image_space()
        self.weight_mask = self.generate_blending_weights(self.blend_pixels)
        self.fused_ts = self.create_fused_image()

    def compute_fused_image_space(self):
        """Compute the overall fused image size in yx given tile positions.

        Parameters
        ----------
        None
            This callable has no parameters.

        Returns
        -------
        tuple of int
            Shape (H, W) of the fused image.
        tuple of int
            (min_y, min_x) global offset for positioning tiles.
        """
        min_y = np.min(self.tile_positions[:, 0])
        min_x = np.min(self.tile_positions[:, 1])
        max_y = np.max(self.tile_positions[:, 0]) + (
            self.tile_shape[0] * self.pixel_size[0]
        )
        max_x = np.max(self.tile_positions[:, 1]) + (
            self.tile_shape[1] * self.pixel_size[1]
        )

        fused_shape_unpadded = (
            int((max_y - min_y) / self.pixel_size[0]),
            int((max_x - min_x) / self.pixel_size[1]),
        )

        pad_y = (
            self.padding_multiple - (fused_shape_unpadded[0] % self.padding_multiple)
        ) % self.padding_multiple
        pad_x = (
            self.padding_multiple - (fused_shape_unpadded[1] % self.padding_multiple)
        ) % self.padding_multiple
        padded_final_ny = fused_shape_unpadded[0] + pad_y
        padded_final_nx = fused_shape_unpadded[1] + pad_x

        fused_shape = (padded_final_ny, padded_final_nx)

        return fused_shape, (min_y, min_x)

    def generate_blending_weights(self, blend_pixels: tuple[int, int] | None = None):
        """Generate a feathered blending weight mask for a tile.

        Parameters
        ----------
        blend_pixels : int
            Fraction of the tile edge used for blending, by default 0.2.

        Returns
        -------
        weight_mask: ndarray
            A (h, w) weight array for blending.
        """
        h, w = self.tile_shape
        if blend_pixels is None:
            blend_pixels = self.blend_pixels

        # Clamp blend_pixels to half the tile size
        blend_pixels_y = min(blend_pixels[0], h // 2)
        blend_pixels_x = min(blend_pixels[1], w // 2)

        y = np.ones(h, dtype=np.float32)
        x = np.ones(w, dtype=np.float32)

        # Y blending
        blend_zone_y = np.linspace(0, np.pi, blend_pixels_y)
        feather_y = 0.5 * (1 - np.cos(blend_zone_y))

        y[:blend_pixels_y] = feather_y  # top
        y[-blend_pixels_y:] = feather_y[::-1]  # bottom

        # X blending
        blend_zone_x = np.linspace(0, np.pi, blend_pixels_x)
        feather_x = 0.5 * (1 - np.cos(blend_zone_x))

        x[:blend_pixels_x] = feather_x  # left
        x[-blend_pixels_x:] = feather_x[::-1]  # right

        weight_mask = np.outer(y, x)
        # Every covered pixel must retain a positive contribution. Exact zeros
        # at exterior tile edges otherwise create holes when no neighbor exists.
        return np.maximum(weight_mask, np.finfo(np.float32).eps)

    def create_fused_image(self):
        """Create the fused TCZYX image through yaozarrs.

        Parameters
        ----------
        None
            This callable has no parameters.

        Returns
        -------
        object
            Result produced by the callable.
        """
        if self.time_range is not None:
            time_to_use = self.time_range[1] - self.time_range[0]
        else:
            time_to_use = self.time_dim
        shape = (time_to_use, self.channels, 1, *self.fused_shape)
        dims = [
            DimSpec(name="t", size=time_to_use),
            DimSpec(name="c", size=self.channels),
            DimSpec(name="z", size=1, scale=1.0, unit="micrometer"),
            DimSpec(
                name="y",
                size=self.fused_shape[0],
                scale=self.pixel_size[0],
                unit="micrometer",
            ),
            DimSpec(
                name="x",
                size=self.fused_shape[1],
                scale=self.pixel_size[1],
                unit="micrometer",
            ),
        ]
        image = v05.Image(
            multiscales=[v05.Multiscale.from_dims(dims, name="fused-max-projection")]
        )
        _, arrays = prepare_image(
            self.output_path,
            image,
            (shape, np.uint16),
            chunks=(1, 1, 1, self.chunk_size, self.chunk_size),
            writer="tensorstore",
            overwrite=True,
        )
        return arrays["0"]

    def fuse_tiles(self):
        """Fuse all tiles using weighted averaging and asynchronous writes.

        This method ensures that all channels are fused separately and blended correctly.

        Parameters
        ----------
        None
            This callable has no parameters.

        Returns
        -------
        None
            No value is returned.
        """
        if self.time_range is not None:
            source_times = range(self.time_range[0], self.time_range[1])
        else:
            source_times = range(self.time_dim)

        for output_time, source_time in enumerate(
            tqdm(source_times, desc="t", leave=True)
        ):
            accumulation = np.zeros(
                (self.channels, 1, *self.fused_shape), dtype=np.float32
            )
            weight_sum = np.zeros_like(accumulation)
            for tile_idx, (y, x) in enumerate(
                tqdm(self.tile_positions, desc="p", leave=False)
            ):
                # Read tile correctly from its respective position
                tile = self.ts_dataset[tile_idx]
                y_slice = slice(self.pad_y, -self.pad_y or None)
                x_slice = slice(self.pad_x, -self.pad_x or None)
                tile_data = (
                    tile[source_time, :, 0, y_slice, x_slice]
                    .read()
                    .result()
                    .astype(np.float32)
                )

                # Ensure tile_data has explicit Z-dimension
                tile_data = tile_data[
                    :, np.newaxis, :, :
                ]  # Shape: (C, 1, H_tile, W_tile)

                # Compute global indices
                y_start = int((y - self.offset[0]) / self.pixel_size[0])
                x_start = int((x - self.offset[1]) / self.pixel_size[1])
                y_end = y_start + self.tile_shape[0]
                x_end = x_start + self.tile_shape[1]

                # Expand weight mask to match shape (C, 1, H_tile, W_tile)
                weight_mask_reshaped = np.broadcast_to(
                    self.weight_mask, (self.channels, 1, *self.weight_mask.shape)
                )

                accumulation[:, :, y_start:y_end, x_start:x_end] += (
                    tile_data * weight_mask_reshaped
                )
                weight_sum[:, :, y_start:y_end, x_start:x_end] += weight_mask_reshaped

            fused = np.divide(
                accumulation,
                weight_sum,
                out=np.zeros_like(accumulation),
                where=weight_sum > 0,
            )
            self.fused_ts[output_time].write(
                np.rint(np.clip(fused, 0, np.iinfo(np.uint16).max)).astype(np.uint16)
            ).result()

    def run(self):
        """Run the full fusion pipeline.

        Parameters
        ----------
        None
            This callable has no parameters.

        Returns
        -------
        None
            No value is returned.
        """
        self.fuse_tiles()
