import numpy as np
from pathlib import Path
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
        ts_dataset: str|Path, 
        tile_positions: list[float,float], 
        output_path: str|Path, 
        pixel_size: tuple[float,float],
        pad_yx: list[int,int] = [0,0],
        time_range: list[int] = None
    ):
        self.pad_y = pad_yx[0]
        self.pad_x = pad_yx[1]

        self.ts_dataset = tuple(ts_dataset)

        self.tile_positions = np.array(tile_positions)
        self.output_path = Path(output_path)
        self.pixel_size = pixel_size
        
        self.time_dim, self.channels, self.z_dim, height, width = self.ts_dataset[0].shape
        self.position_dim = len(self.ts_dataset)
        height -= 2 * self.pad_y
        width -= 2 * self.pad_x
        
        self.padding_multiple = 8
        self.chunk_size = 512
        
        self.tile_shape = (height, width)
        self.time_range = time_range

        self.fused_shape, self.offset = self.compute_fused_image_space()
        self.weight_mask = self.generate_blending_weights()
        self.fused_ts = self.create_fused_image()
        
        

    def compute_fused_image_space(self):
        """
        Compute the overall fused image size in yx given tile positions.

        Returns
        -------
        tuple of int
            Shape (H, W) of the fused image.
        tuple of int
            (min_y, min_x) global offset for positioning tiles.
        """
        min_y = np.min(self.tile_positions[:, 0])
        min_x = np.min(self.tile_positions[:, 1])
        max_y = np.max(self.tile_positions[:, 0]) + (self.tile_shape[0] * self.pixel_size[0])
        max_x = np.max(self.tile_positions[:, 1]) + (self.tile_shape[1] * self.pixel_size[1])
        

        fused_shape_unpadded = (
            int((max_y - min_y) / self.pixel_size[0]),
            int((max_x - min_x) / self.pixel_size[1])
        )
        
        pad_y = (
            self.padding_multiple
            - (fused_shape_unpadded[0] % self.padding_multiple)
        ) % self.padding_multiple
        pad_x = (
            self.padding_multiple
            - (fused_shape_unpadded[1] % self.padding_multiple)
        ) % self.padding_multiple
        padded_final_ny = fused_shape_unpadded[0] + pad_y 
        padded_final_nx = fused_shape_unpadded[1] + pad_x
        
        fused_shape = (
            padded_final_ny,
            padded_final_nx
        )

        return fused_shape, (min_y, min_x)

    def generate_blending_weights(self, blend_pixels: list[int,int] = [380,380]):
        """
        Generate a feathered blending weight mask for a tile.

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

        return weight_mask

    def create_fused_image(self):
        """Create the fused TCZYX image through yaozarrs."""
        if self.time_range is not None:
            time_to_use = self.time_range[1] - self.time_range[0]
        else:
            time_to_use = self.time_dim
        shape = (time_to_use, self.channels, 1, *self.fused_shape)
        dims = [
            DimSpec(name="t", size=time_to_use),
            DimSpec(name="c", size=self.channels),
            DimSpec(name="z", size=1, scale=1.0, unit="micrometer"),
            DimSpec(name="y", size=self.fused_shape[0], scale=self.pixel_size[0], unit="micrometer"),
            DimSpec(name="x", size=self.fused_shape[1], scale=self.pixel_size[1], unit="micrometer"),
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
        """
        Fuse all tiles into the final image using weighted averaging and asynchronous writes.
        This method ensures that all channels are fused separately and blended correctly.
        """
        write_futures = []
        
        if self.time_range is not None:
            time_iterator = tqdm(range(self.time_range[0],self.time_range[1]),desc="t",leave=True)
        else:
            time_iterator = tqdm(range(self.time_dim),desc="t",leave=True)
        
        # if self.time_range is not None:
        if (self.time_dim > 1) or (self.time_range is not None and self.time_range[1] > 1):
            refresh_position_iterator = True
        else:
            refresh_position_iterator = False
            
        pos_iterator = enumerate(tqdm(self.tile_positions, desc="p",leave=False))

        for time_idx in time_iterator:
            for tile_idx, (y, x) in pos_iterator:
                # Read tile correctly from its respective position
                tile = self.ts_dataset[tile_idx]
                y_slice = slice(self.pad_y, -self.pad_y or None)
                x_slice = slice(self.pad_x, -self.pad_x or None)
                tile_data = tile[time_idx, :, 0, y_slice, x_slice].read().result().astype(np.float32)

                # Ensure tile_data has explicit Z-dimension
                tile_data = tile_data[:, np.newaxis, :, :]  # Shape: (C, 1, H_tile, W_tile)

                # Compute global indices
                y_start = int((y - self.offset[0]) / self.pixel_size[0])
                x_start = int((x - self.offset[1]) / self.pixel_size[1])
                y_end = y_start + self.tile_shape[0]
                x_end = x_start + self.tile_shape[1]

                # Expand weight mask to match shape (C, 1, H_tile, W_tile)
                weight_mask_reshaped = np.broadcast_to(self.weight_mask, (self.channels, 1, *self.weight_mask.shape))

                # **Read existing fused image region before modifying**
                existing_fused = self.fused_ts[time_idx, :, :, y_start:y_end, x_start:x_end].read().result().astype(np.float32)
                existing_weight_sum = np.ones_like(existing_fused)  # Initialize weight sum if uninitialized

                # **Ensure weighted_tile shape matches existing_fused**
                weighted_tile = tile_data * weight_mask_reshaped
                
                existing_fused += weighted_tile
                existing_weight_sum += weight_mask_reshaped

                existing_weight_sum[existing_weight_sum == 0] = 1
                existing_fused /= existing_weight_sum

                # Convert to uint16
                fused_patch = np.clip(existing_fused, 0, 65535).astype(np.uint16)
                
                # **Asynchronously write the normalized region**
                future = self.fused_ts[time_idx, :, :, y_start:y_end, x_start:x_end].write(fused_patch)
                write_futures.append(future)

            # **Wait for all writes to complete**
            for future in write_futures:
                future.result()
                
            if refresh_position_iterator:    
                pos_iterator = enumerate(tqdm(self.tile_positions, desc="p",leave=False))

    def run(self):
        """
        Run the full fusion pipeline.
        """
        
        self.fuse_tiles()
