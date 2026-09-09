"""Fuse maximum-projection image tiles using stage positions."""

import math
from collections.abc import Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
from tqdm import tqdm

from yaozarrs import open_group, v05
from yaozarrs.write.v05 import prepare_image

from opm_processing.dataio.ngff import (
    downsample_yx,
    round_spatial,
    round_spatial_values,
    round_tczyx_transform,
)
from opm_processing.dataio.processing_state import (
    ProcessingState,
    processing_state_path,
)
from opm_processing.imageprocessing.coordinates import (
    stage_positions_to_image_coordinates,
)


def regenerate_fused_max_projection(
    fused_path: str | Path,
    output_path: str | Path,
    *,
    max_workers: int | None = None,
) -> Path:
    """Stream a maximum-Z projection from registered full-resolution fused data.

    Parameters
    ----------
    fused_path
        Full-resolution fused OME-Zarr image.
    output_path
        OME-Zarr image to overwrite with the maximum-Z projection.
    max_workers
        Maximum number of spatial chunks projected concurrently.

    Returns
    -------
    pathlib.Path
        Path to the regenerated maximum-projection image.
    """
    fused_path = Path(fused_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if not fused_path.is_dir():
        raise FileNotFoundError(
            f"Full-resolution fused data store not found: {fused_path}"
        )
    fused_suffix = "_fused.ome.zarr"
    if not fused_path.name.endswith(fused_suffix):
        raise ValueError(f"Unexpected registered fused-image name: {fused_path}")
    acquisition_name = fused_path.name[: -len(fused_suffix)]
    processing_state = ProcessingState.read(
        processing_state_path(fused_path.parent, acquisition_name)
    )
    processed_path = processing_state.registered_output_for_fused(fused_path)

    source_root = open_group(fused_path)
    source_metadata = source_root.ome_metadata()
    if not isinstance(source_metadata, v05.Image):
        raise ValueError(f"Not an OME-Zarr v0.5 Image: {fused_path}")
    source_multiscale = source_metadata.multiscales[0]
    projection_datasets = []
    projection_specs = []
    levels = []
    expected_time_channels: tuple[int, int] | None = None
    for source_dataset in source_multiscale.datasets:
        source = source_root[source_dataset.path].to_tensorstore()
        if source.rank != 5:
            raise ValueError(
                "Expected TCZYX fused data at every multiscale level; "
                f"{source_dataset.path!r} has shape {source.shape}"
            )
        time_dim, channel_dim, z_dim, y_dim, x_dim = (
            int(value) for value in source.shape
        )
        if expected_time_channels is None:
            expected_time_channels = (time_dim, channel_dim)
        elif (time_dim, channel_dim) != expected_time_channels:
            raise ValueError(
                "Fused multiscale levels must have identical T and C dimensions"
            )

        scale = np.asarray(source_dataset.scale_transform.scale, dtype=np.float64)
        if scale.shape != (5,):
            raise ValueError("Fused dataset scale transform must contain TCZYX values")
        if source_dataset.translation_transform is None:
            translation = np.zeros(5, dtype=np.float64)
        else:
            translation = np.asarray(
                source_dataset.translation_transform.translation,
                dtype=np.float64,
            )
        if translation.shape != (5,):
            raise ValueError(
                "Fused dataset translation transform must contain TCZYX values"
            )
        output_scale = np.asarray(round_tczyx_transform(scale), dtype=np.float64)
        projection_translation = np.asarray(
            round_tczyx_transform(translation),
            dtype=np.float64,
        )
        projection_translation[2] += 0.5 * (z_dim - 1) * output_scale[2]
        projection_datasets.append(
            v05.Dataset(
                path=source_dataset.path,
                coordinateTransformations=[
                    v05.ScaleTransformation(scale=output_scale.tolist()),
                    v05.TranslationTransformation(
                        translation=round_tczyx_transform(projection_translation)
                    ),
                ],
            )
        )

        source_chunks = tuple(
            int(value) for value in source.chunk_layout.read_chunk.shape
        )
        chunk_y = min(y_dim, max(1, source_chunks[-2]))
        chunk_x = min(x_dim, max(1, source_chunks[-1]))
        source_dtype = np.dtype(source.dtype.numpy_dtype)
        projection_specs.append(
            ((time_dim, channel_dim, 1, y_dim, x_dim), source_dtype)
        )
        levels.append(
            {
                "path": source_dataset.path,
                "source": source,
                "shape": (time_dim, channel_dim, z_dim, y_dim, x_dim),
                "chunk_y": chunk_y,
                "chunk_x": chunk_x,
                "z_step": min(z_dim, max(16, source_chunks[-3])),
            }
        )

    if not levels:
        raise ValueError("Fused image contains no multiscale datasets")
    projection_image = v05.Image(
        multiscales=[
            v05.Multiscale(
                name="registered-fused-max-z",
                axes=source_multiscale.axes,
                datasets=projection_datasets,
            )
        ]
    )

    _, arrays = prepare_image(
        output_path,
        projection_image,
        projection_specs,
        extra_attributes={},
        chunks=(
            1,
            1,
            1,
            int(levels[0]["chunk_y"]),
            int(levels[0]["chunk_x"]),
        ),
        writer="tensorstore",
        overwrite=True,
    )
    chunk_total = sum(
        int(level["shape"][0])
        * int(level["shape"][1])
        * math.ceil(int(level["shape"][3]) / int(level["chunk_y"]))
        * math.ceil(int(level["shape"][4]) / int(level["chunk_x"]))
        for level in levels
    )
    workers = max_workers
    if workers is None:
        workers = min(8, chunk_total)
    workers = max(1, min(int(workers), chunk_total))

    def project_chunk(
        level_index: int,
        time_index: int,
        channel_index: int,
        y_start: int,
        y_stop: int,
        x_start: int,
        x_stop: int,
    ) -> None:
        """Project one output-aligned YX chunk through every source Z plane."""
        level = levels[level_index]
        source = level["source"]
        output = arrays[level["path"]]
        z_dim = int(level["shape"][2])
        z_step = int(level["z_step"])
        projection = None
        for z_start in range(0, z_dim, z_step):
            z_stop = min(z_start + z_step, z_dim)
            slab = (
                source[
                    time_index,
                    channel_index,
                    z_start:z_stop,
                    y_start:y_stop,
                    x_start:x_stop,
                ]
                .read()
                .result()
            )
            slab_max = np.max(slab, axis=0)
            if projection is None:
                projection = slab_max
            else:
                np.maximum(projection, slab_max, out=projection)
        if projection is None:
            raise RuntimeError("Cannot project an empty Z axis")
        output[
            time_index,
            channel_index,
            0,
            y_start:y_stop,
            x_start:x_stop,
        ].write(projection).result()

    progress = tqdm(
        total=chunk_total,
        desc="registered max-z",
        leave=True,
        unit="chunk",
    )
    pending: set[Future[None]] = set()

    def collect(done: set[Future[None]]) -> None:
        for future in done:
            future.result()
            progress.update()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for level_index, level in enumerate(levels):
            time_dim, channel_dim, _z_dim, y_dim, x_dim = level["shape"]
            chunk_y = int(level["chunk_y"])
            chunk_x = int(level["chunk_x"])
            for time_index in range(int(time_dim)):
                for channel_index in range(int(channel_dim)):
                    for y_start in range(0, int(y_dim), chunk_y):
                        y_stop = min(y_start + chunk_y, int(y_dim))
                        for x_start in range(0, int(x_dim), chunk_x):
                            x_stop = min(x_start + chunk_x, int(x_dim))
                            pending.add(
                                executor.submit(
                                    project_chunk,
                                    level_index,
                                    time_index,
                                    channel_index,
                                    y_start,
                                    y_stop,
                                    x_start,
                                    x_stop,
                                )
                            )
                            if len(pending) >= 2 * workers:
                                done, pending = wait(
                                    pending,
                                    return_when=FIRST_COMPLETED,
                                )
                                collect(done)
        while pending:
            done, pending = wait(
                pending,
                return_when=FIRST_COMPLETED,
            )
            collect(done)
    progress.close()
    processing_state.set_registered_max_projection(
        processed_path,
        max_projection_path=output_path,
    )
    return output_path


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
        Pixel size in physical units, supplied as either (y, x) or (z, y, x).
    reverse_stage_y : bool, default=True
        Reverse the stage-derived Y placement coordinate without flipping tile
        pixels. OPM stage motion is opposite image Y.
    reverse_stage_z : bool, default=True
        Convert physical stage Z into the opposite laboratory-Z placement
        coordinate before calculating its image-Y contribution.
    pad_yx : list of int, default  = [0, 0].
        Padding in y and x dimensions already applied to the dataset
    time_range: list of int, default = None
    """

    def __init__(
        self,
        ts_dataset: str | Path,
        tile_positions: list[float, float],
        output_path: str | Path,
        pixel_size: Sequence[float],
        spatial_offset_z_um: float = 0.0,
        source_position_indices: Sequence[int] | None = None,
        pad_yx: Sequence[int] = (0, 0),
        time_range: tuple[int, int] | None = None,
        blend_pixels: tuple[int, int] = (380, 380),
        chunk_size: int = 512,
        padding_multiple: int = 8,
        reverse_stage_y: bool = True,
        reverse_stage_z: bool = True,
        opm_angle_deg: float | None = None,
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
            Physical YX or ZYX pixel spacing.
        spatial_offset_z_um
            Physical Z coordinate represented by the singleton projection plane.
        source_position_indices
            Original acquisition position index for each input tile.
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
        reverse_stage_y
            Whether to reverse the stage-derived image-Y placement coordinate.
            Tile pixel arrays are not modified.
        reverse_stage_z
            Whether to reverse physical stage Z for laboratory-coordinate
            placement. Tile pixels and the acquired scan axis are not modified.
        opm_angle_deg
            OPM angle used to map relative stage Z into deskewed image Y when
            `tile_positions` contains ZYX coordinates.

        Returns
        -------
        None
            No value is returned.
        """
        self.pad_y = pad_yx[0]
        self.pad_x = pad_yx[1]

        self.ts_dataset = tuple(ts_dataset)

        self.reverse_stage_y = bool(reverse_stage_y)
        self.reverse_stage_z = bool(reverse_stage_z)
        placement_coordinates = stage_positions_to_image_coordinates(
            tile_positions,
            reverse_y=self.reverse_stage_y,
            reverse_z=self.reverse_stage_z,
            opm_angle_deg=opm_angle_deg,
        )
        self.tile_positions = placement_coordinates[:, -2:]
        if source_position_indices is None:
            source_position_indices = range(len(self.tile_positions))
        self.source_position_indices = tuple(
            int(value) for value in source_position_indices
        )
        if len(self.source_position_indices) != len(self.tile_positions):
            raise ValueError(
                "source_position_indices must contain one value per input tile"
            )
        self.output_path = Path(output_path)
        pixel_size = tuple(float(value) for value in pixel_size)
        if len(pixel_size) == 2:
            self.z_pixel_size = 1.0
            self.pixel_size = round_spatial_values(pixel_size)
        elif len(pixel_size) == 3:
            self.z_pixel_size = round_spatial(pixel_size[0])
            self.pixel_size = round_spatial_values(pixel_size[-2:])
        else:
            raise ValueError("pixel_size must contain YX or ZYX spacing")
        if any(value <= 0 for value in (*self.pixel_size, self.z_pixel_size)):
            raise ValueError("pixel sizes must be positive")
        self.spatial_offset_z_um = round_spatial(spatial_offset_z_um)

        self.time_dim, self.channels, self.z_dim, height, width = self.ts_dataset[
            0
        ].shape
        self.output_dtype = np.dtype(self.ts_dataset[0].dtype.numpy_dtype)
        if self.output_dtype not in (np.dtype(np.uint16), np.dtype(np.float32)):
            raise ValueError(
                "Maximum-projection fusion supports uint16 or float32 input"
            )
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

        return fused_shape, (float(min_y), float(min_x))

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
        if blend_pixels_y:
            blend_zone_y = np.linspace(0, np.pi, blend_pixels_y)
            feather_y = 0.5 * (1 - np.cos(blend_zone_y))

            y[:blend_pixels_y] = feather_y  # top
            y[-blend_pixels_y:] = feather_y[::-1]  # bottom

        # X blending
        if blend_pixels_x:
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
        factors = tuple(
            factor for factor in (1, 2, 4, 8, 16, 32) if factor <= min(self.fused_shape)
        )
        self.multiscale_factors_yx = factors
        axes = [
            {"name": "t", "type": "time"},
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ]
        datasets = []
        specs = []
        for level, factor in enumerate(factors):
            datasets.append(
                v05.Dataset(
                    path=str(level),
                    coordinateTransformations=[
                        v05.ScaleTransformation(
                            scale=round_tczyx_transform(
                                (
                                    1,
                                    1,
                                    self.z_pixel_size,
                                    self.pixel_size[0] * factor,
                                    self.pixel_size[1] * factor,
                                )
                            )
                        ),
                        v05.TranslationTransformation(
                            translation=round_tczyx_transform(
                                (
                                    0,
                                    0,
                                    self.spatial_offset_z_um,
                                    self.offset[0],
                                    self.offset[1],
                                )
                            )
                        ),
                    ],
                )
            )
            specs.append(
                (
                    (
                        time_to_use,
                        self.channels,
                        1,
                        (self.fused_shape[0] + factor - 1) // factor,
                        (self.fused_shape[1] + factor - 1) // factor,
                    ),
                    self.output_dtype,
                )
            )
        image = v05.Image(
            multiscales=[
                v05.Multiscale(
                    name="fused-max-projection",
                    axes=axes,
                    datasets=datasets,
                )
            ]
        )
        _, arrays = prepare_image(
            self.output_path,
            image,
            specs,
            extra_attributes={},
            chunks=(1, 1, 1, self.chunk_size, self.chunk_size),
            writer="tensorstore",
            overwrite=True,
        )
        self.multiscale_arrays = tuple(
            arrays[str(level)] for level in range(len(factors))
        )
        return arrays["0"]

    def write_multiscales(self) -> None:
        """Stream strided YX pyramid levels from the fused level zero."""
        for level, factor in enumerate(self.multiscale_factors_yx[1:], start=1):
            target = self.multiscale_arrays[level]
            target_height = int(target.shape[-2])
            target_width = int(target.shape[-1])
            source_height = int(self.fused_ts.shape[-2])
            source_width = int(self.fused_ts.shape[-1])
            output_step = max(1, self.chunk_size // factor)
            progress = tqdm(
                total=(
                    int(target.shape[0])
                    * math.ceil(target_height / output_step)
                    * math.ceil(target_width / output_step)
                ),
                desc=f"max pyramid {factor}x",
                leave=False,
                unit="chunk",
            )
            for time_index in range(int(target.shape[0])):
                for y_start in range(0, target_height, output_step):
                    y_stop = min(y_start + output_step, target_height)
                    for x_start in range(0, target_width, output_step):
                        x_stop = min(x_start + output_step, target_width)
                        source = np.asarray(
                            self.fused_ts[
                                time_index,
                                :,
                                0,
                                y_start * factor : min(y_stop * factor, source_height),
                                x_start * factor : min(x_stop * factor, source_width),
                            ]
                            .read()
                            .result()
                        )
                        reduced = downsample_yx(source, factor)
                        target[
                            time_index,
                            :,
                            0,
                            y_start:y_stop,
                            x_start:x_stop,
                        ].write(reduced).result()
                        progress.update()
            progress.close()

    def fuse_tiles(self):
        """Fuse tiles into bounded spatial chunks.

        Only chunks intersecting a tile are materialized. This keeps peak RAM
        proportional to ``channels * chunk_size**2`` rather than to the full
        mosaic canvas, which may include large empty gaps between stage positions.

        Parameters
        ----------
        None
            This callable has no parameters.

        Returns
        -------
        None
            No value is returned.
        """
        source_times = (
            range(self.time_range[0], self.time_range[1])
            if self.time_range is not None
            else range(self.time_dim)
        )
        source_times = tuple(source_times)

        tile_channel_nonzero: dict[tuple[int, int], np.ndarray] = {}
        zero_channel_count = 0
        for source_time in source_times:
            for tile_idx in range(len(self.ts_dataset)):
                tile = np.asarray(
                    self.ts_dataset[tile_idx][
                        source_time,
                        :,
                        0,
                        self.pad_y : self.pad_y + self.tile_shape[0],
                        self.pad_x : self.pad_x + self.tile_shape[1],
                    ]
                    .read()
                    .result()
                )
                present = np.any(tile != 0, axis=(-2, -1))
                tile_channel_nonzero[(source_time, tile_idx)] = present
                zero_channel_count += int(np.count_nonzero(~present))
        if zero_channel_count:
            print(
                f"Excluding {zero_channel_count} zero maximum-projection "
                "tile/channel images from fusion weights."
            )

        tile_bounds = []
        chunks_to_tiles: dict[tuple[int, int], list[int]] = {}
        fused_height, fused_width = self.fused_shape
        tile_height, tile_width = self.tile_shape
        for tile_idx, (y, x) in enumerate(self.tile_positions):
            y_start = int((y - self.offset[0]) / self.pixel_size[0])
            x_start = int((x - self.offset[1]) / self.pixel_size[1])
            y_end = y_start + tile_height
            x_end = x_start + tile_width
            tile_bounds.append((y_start, y_end, x_start, x_end))

            visible_y_start = max(0, y_start)
            visible_y_end = min(fused_height, y_end)
            visible_x_start = max(0, x_start)
            visible_x_end = min(fused_width, x_end)
            if visible_y_start >= visible_y_end or visible_x_start >= visible_x_end:
                continue
            first_chunk_y = (visible_y_start // self.chunk_size) * self.chunk_size
            first_chunk_x = (visible_x_start // self.chunk_size) * self.chunk_size
            for chunk_y in range(first_chunk_y, visible_y_end, self.chunk_size):
                for chunk_x in range(first_chunk_x, visible_x_end, self.chunk_size):
                    chunks_to_tiles.setdefault((chunk_y, chunk_x), []).append(tile_idx)

        covered_chunks = sorted(chunks_to_tiles)
        progress = tqdm(
            total=len(source_times) * len(covered_chunks),
            desc="max fusion",
            leave=True,
            unit="chunk",
        )
        for output_time, source_time in enumerate(source_times):
            for chunk_y, chunk_x in covered_chunks:
                chunk_y_end = min(chunk_y + self.chunk_size, fused_height)
                chunk_x_end = min(chunk_x + self.chunk_size, fused_width)
                chunk_height = chunk_y_end - chunk_y
                chunk_width = chunk_x_end - chunk_x
                accumulation = np.zeros(
                    (self.channels, chunk_height, chunk_width), dtype=np.float32
                )
                weight_sum = np.zeros(
                    (self.channels, chunk_height, chunk_width), dtype=np.float32
                )

                for tile_idx in chunks_to_tiles[(chunk_y, chunk_x)]:
                    channel_present = tile_channel_nonzero[(source_time, tile_idx)]
                    if not np.any(channel_present):
                        continue
                    y_start, y_end, x_start, x_end = tile_bounds[tile_idx]
                    overlap_y_start = max(chunk_y, y_start)
                    overlap_y_end = min(chunk_y_end, y_end)
                    overlap_x_start = max(chunk_x, x_start)
                    overlap_x_end = min(chunk_x_end, x_end)
                    if (
                        overlap_y_start >= overlap_y_end
                        or overlap_x_start >= overlap_x_end
                    ):
                        continue

                    tile_y_start = overlap_y_start - y_start
                    tile_y_end = overlap_y_end - y_start
                    tile_x_start = overlap_x_start - x_start
                    tile_x_end = overlap_x_end - x_start
                    output_y = slice(overlap_y_start - chunk_y, overlap_y_end - chunk_y)
                    output_x = slice(overlap_x_start - chunk_x, overlap_x_end - chunk_x)

                    tile_data = (
                        self.ts_dataset[tile_idx][
                            source_time,
                            :,
                            0,
                            self.pad_y + tile_y_start : self.pad_y + tile_y_end,
                            self.pad_x + tile_x_start : self.pad_x + tile_x_end,
                        ]
                        .read()
                        .result()
                        .astype(np.float32)
                    )
                    weights = self.weight_mask[
                        tile_y_start:tile_y_end,
                        tile_x_start:tile_x_end,
                    ]
                    accumulation[:, output_y, output_x] += tile_data * weights
                    weight_sum[:, output_y, output_x] += (
                        channel_present[:, None, None] * weights
                    )

                fused = np.divide(
                    accumulation,
                    weight_sum,
                    out=np.zeros_like(accumulation),
                    where=weight_sum > 0,
                )
                if self.output_dtype == np.dtype(np.float32):
                    fused_output = fused
                else:
                    fused_output = np.rint(
                        np.clip(fused, 0, np.iinfo(np.uint16).max)
                    ).astype(np.uint16)
                self.fused_ts[
                    output_time,
                    :,
                    0,
                    chunk_y:chunk_y_end,
                    chunk_x:chunk_x_end,
                ].write(fused_output).result()
                progress.update()
        progress.close()

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
        self.write_multiscales()
