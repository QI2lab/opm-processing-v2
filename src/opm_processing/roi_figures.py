"""ROI overviews rendered with the selected napari display."""

from __future__ import annotations

import json
import mmap
from contextlib import ExitStack
from tempfile import TemporaryFile
from typing import TYPE_CHECKING

import numpy as np
import psutil
from PIL import Image
from tqdm import tqdm

from opm_processing.dataio.position_collection import open_image_array
from opm_processing.export_ome_tiff import resolve_export_image

JPEG_QUALITY = 100

if TYPE_CHECKING:
    from pathlib import Path

    from napari import Viewer
    from napari.layers import Image as ImageLayer


def write_roi_overview(image: Image.Image, output: Path, metadata: dict) -> None:
    """Scale an annotated RGBA overview once and write PNG and JPEGli JPEG.

    Parameters
    ----------
    image : PIL.Image.Image
        Rendered tissue, ROI outlines and scale bar in the final orientation.
    output : pathlib.Path
        PNG destination; the JPEG uses the same stem with a .jpg suffix.
    metadata : dict
        Display settings, pixel spacing and native source-pixel ROI coordinates.
    """
    from imagecodecs import JPEGLI, jpegli_encode

    # JPEGli's shipped encoder rejects dimensions above 65500, including 65535.
    scale = min(1.0, 65500 / max(image.size))
    size = tuple(max(1, round(value * scale)) for value in image.size)
    rendered = image.resize(size, Image.Resampling.LANCZOS) if scale < 1 else image
    png = output.with_suffix(".tmp.png")
    jpeg = output.with_suffix(".tmp.jpg")
    try:
        tqdm.write(f"Writing PNG overview: {size[0]} x {size[1]} pixels")
        rendered.save(png, format="PNG", compress_level=1)
        tqdm.write("Writing JPEG overview with JPEGli")
        jpeg.write_bytes(
            jpegli_encode(
                np.asarray(rendered),
                level=JPEG_QUALITY,
                colorspace=JPEGLI.CS.EXT_RGBA,
                subsampling="444",
                optimize=False,
                progressive=False,
            )
        )
    except BaseException:
        png.unlink(missing_ok=True)
        jpeg.unlink(missing_ok=True)
        raise
    finally:
        if rendered is not image:
            rendered.close()
    # Keep completed temporary files if Windows cannot replace an open image.
    png.replace(output)
    jpeg.replace(output.with_suffix(".jpg"))
    metadata = {
        **metadata,
        "shape_yx": list(size[::-1]),
        "native_shape_yx": metadata.get("native_shape_yx", list(image.size[::-1])),
        "overview_scale": metadata.get("overview_scale", 1.0) * scale,
        "pixel_size_yx_um": (
            np.asarray(metadata["pixel_size_yx_um"]) * image.size[::-1] / size[::-1]
        ).tolist(),
        "jpeg": {
            "encoder": "jpegli",
            "quality": JPEG_QUALITY,
            "subsampling": "444",
        },
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def export_roi_overview(
    viewer: Viewer,
    layers: list[ImageLayer],
    source: Path,
    bounds: list[tuple[int, int, int, int]],
    output: Path,
) -> None:
    """Render at native resolution, then write scaled PNG and JPEGli overviews.

    Parameters
    ----------
    viewer : napari.Viewer
        Selection viewer whose channel settings, quarter turns and flips are retained.
    layers : list of napari.layers.Image
        Original channel layers, in their source channel order.
    source : pathlib.Path
        Calibrated maximum-Z preview store.
    bounds : list of tuple
        ROI rectangles on the native pixel grid, ordered y0, y1, x0, x1.
    output : pathlib.Path
        PNG path; a JPEG is also written. Adjacent JSON records display and scale.
    """
    import napari
    from napari.qt import get_qapp
    from qtpy.QtCore import Qt

    array = open_image_array(resolve_export_image(source))
    if len(array.shape) != 5 or array.shape[2] != 1:
        raise ValueError("ROI figures require a saved TCZYX maximum-Z projection")
    reference = layers[0]
    timepoint = int(viewer.dims.current_step[0])
    selected = [layer for layer in viewer.layers if layer in layers and layer.visible]
    if not selected:
        raise ValueError("Enable at least one image channel before exporting figures")
    channels = [layers.index(layer) for layer in selected]
    settings = [
        {
            "name": layer.name,
            "colormap": layer.colormap,
            "contrast_limits": tuple(layer.contrast_limits),
            "gamma": float(layer.gamma),
            "opacity": float(layer.opacity),
            "blending": layer.blending,
            "interpolation2d": layer.interpolation2d,
            **(
                {
                    "custom_interpolation_kernel_2d": layer.custom_interpolation_kernel_2d.tolist()
                }
                if layer.interpolation2d == "custom"
                else {}
            ),
        }
        for layer in selected
    ]
    source_height, source_width = array.shape[-2:]
    displayed = viewer.dims.displayed
    if viewer.dims.ndisplay != 2 or set(displayed) != {
        reference.ndim - 2,
        reference.ndim - 1,
    }:
        raise ValueError("ROI figures require the two-dimensional YX view")
    # Public napari transforms include the layer scale, rotation and affine.
    basis = np.zeros((3, reference.ndim))
    basis[1, -2] = basis[2, -1] = 1
    world = np.asarray([reference.data_to_world(point) for point in basis])
    for layer in selected:
        if not np.allclose(
            [layer.data_to_world(point) for point in basis], world, rtol=0, atol=1e-7
        ):
            raise ValueError("Overview channels must have matching spatial transforms")
    direction = np.asarray(
        [
            1 if value == default else -1
            for value, default in zip(
                viewer.camera.orientation2d, ("down", "right"), strict=True
            )
        ]
    )
    linear = (world[1:] - world[0])[:, displayed].T * direction[:, None]
    spacing = np.linalg.norm(linear, axis=1)
    if not np.all(np.isfinite(spacing) & (spacing > 0)):
        raise ValueError("Overview pixel spacing must be finite and positive")
    axes = np.argmax(np.abs(linear), axis=1)
    orientation = np.rint(linear / spacing[:, None]).astype(int)
    if len(set(axes)) != 2 or not np.allclose(
        linear, orientation * spacing[:, None], rtol=0, atol=1e-7
    ):
        raise ValueError("Overview export supports 90-degree rotations and flips")
    signs = orientation[np.arange(2), axes]
    height, width = np.asarray((source_height, source_width))[axes].tolist()
    source_spacing = spacing[np.argsort(axes)]
    length_um, bar_thickness_um = 2000.0, 80.0
    line_width_um, dash_um, gap_um = 20.0, 20.0, 10.0
    if length_um > 0.8 * width * spacing[1]:
        raise ValueError("The 2 mm scale bar must fit inside the displayed image")
    halo = max(
        8,
        *(
            max(np.shape(options.get("custom_interpolation_kernel_2d", [[1]]))) // 2
            for options in settings
        ),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    renderer = napari.Viewer(show=False)
    renderer.canvas.background_color_override = viewer.canvas.background_color
    rendered_layers = [
        renderer.add_image(
            np.zeros((2, 2), dtype=np.float32),
            rgb=False,
            multiscale=False,
            **options,
        )
        for options in settings
    ]
    metadata = {
        "source_path": str(source),
        "shape_yx": [height, width],
        "source_shape_yx": [source_height, source_width],
        "time_index": timepoint,
        "pixel_size_yx_um": spacing.tolist(),
        "display_axes_yx": axes.tolist(),
        "display_axis_signs": signs.tolist(),
        "scale_bar_um": float(length_um),
        "scale_bar_thickness_um": bar_thickness_um,
        "roi_line_width_um": line_width_um,
        "roi_dash_um": dash_um,
        "roi_gap_um": gap_um,
        "roi_bounds_coordinate_system": "source pixel grid",
        "roi_bounds_yx_pixels": bounds,
        "channels": [
            {
                **options,
                "channel_index": channel,
                "colormap": {
                    "name": options["colormap"].name,
                    "colors": options["colormap"].colors.tolist(),
                    "controls": options["colormap"].controls.tolist(),
                },
            }
            for channel, options in zip(channels, settings, strict=True)
        ],
    }
    try:
        # Qt initializes the OpenGL framebuffer on show, even for offscreen use.
        renderer.window._qt_window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
        renderer.show()
        get_qapp().processEvents()
        with renderer._layer_slicer.force_sync():
            with ExitStack() as buffers:
                if height * width * 4 <= psutil.virtual_memory().available:
                    pixels = np.empty((height, width, 4), np.uint8)
                else:
                    scratch = buffers.enter_context(TemporaryFile(dir=output.parent))
                    scratch.truncate(height * width * 4)
                    mapping = buffers.enter_context(
                        mmap.mmap(scratch.fileno(), 0, access=mmap.ACCESS_WRITE)
                    )
                    pixels = np.ndarray((height, width, 4), np.uint8, buffer=mapping)
                # A transposed/flipped view writes native blocks into the final
                # orientation without resampling or copying the whole image.
                native_pixels = pixels[
                    slice(None, None, int(signs[0])),
                    slice(None, None, int(signs[1])),
                ].transpose(*np.argsort(axes), 2)
                try:
                    # Halo samples retain the same interpolation at block edges.
                    for y0 in tqdm(
                        range(0, source_height, 2048),
                        desc="ROI overview",
                        unit="row of blocks",
                    ):
                        y1 = min(y0 + 2048, source_height)
                        for x0 in range(0, source_width, 2048):
                            x1 = min(x0 + 2048, source_width)
                            a, b = max(0, y0 - halo), min(source_height, y1 + halo)
                            c, d = max(0, x0 - halo), min(source_width, x1 + halo)
                            for channel, layer in zip(
                                channels, rendered_layers, strict=True
                            ):
                                layer.data = (
                                    array[timepoint, channel, 0, a:b, c:d]
                                    .read()
                                    .result()
                                )
                            rendered = renderer.export_figure(
                                scale_factor=1, flash=False
                            )
                            if rendered.shape != (b - a, d - c, 4):
                                raise ValueError(
                                    "Napari did not render the native pixel dimensions"
                                )
                            native_pixels[y0:y1, x0:x1] = rendered[
                                y0 - a : y1 - a, x0 - c : x1 - c
                            ]
                    line_width = np.maximum(
                        1, np.rint(line_width_um / source_spacing)
                    ).astype(int)
                    outline = (255, 255, 255, 255)
                    for y0, y1, x0, x1 in bounds:
                        xdash = (np.arange(x1 - x0) * source_spacing[1]) % (
                            dash_um + gap_um
                        ) < dash_um
                        ydash = (np.arange(y1 - y0) * source_spacing[0]) % (
                            dash_um + gap_um
                        ) < dash_um
                        for start, stop in (
                            (y0, min(y0 + line_width[0], y1)),
                            (max(y0, y1 - line_width[0]), y1),
                        ):
                            native_pixels[start:stop, x0:x1][:, xdash] = outline
                        for start, stop in (
                            (x0, min(x0 + line_width[1], x1)),
                            (max(x0, x1 - line_width[1]), x1),
                        ):
                            native_pixels[y0:y1, start:stop][ydash] = outline
                    # Paint the bar horizontally in the final displayed frame.
                    right, center_y = round(0.94 * width), round(0.93 * height)
                    left = right - round(length_um / spacing[1])
                    for thickness, color in (
                        (bar_thickness_um * 5 / 3, (0, 0, 0, 255)),
                        (bar_thickness_um, outline),
                    ):
                        bar_height = max(1, round(thickness / spacing[0]))
                        top = max(0, center_y - bar_height // 2)
                        pixels[top : min(height, top + bar_height), left:right] = color
                    image = Image.frombuffer(
                        "RGBA", (width, height), pixels, "raw", "RGBA", 0, 1
                    )
                    try:
                        write_roi_overview(image, output, metadata)
                    finally:
                        image.close()
                finally:
                    del native_pixels, pixels
    finally:
        renderer.close()
