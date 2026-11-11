"""Native-resolution ROI overviews rendered with the selected napari display."""

from __future__ import annotations

import json
import mmap
from tempfile import TemporaryFile
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from opm_processing.dataio.position_collection import open_image_array
from opm_processing.export_ome_tiff import resolve_export_image
from opm_processing.export_projections import annotation_font

if TYPE_CHECKING:
    from pathlib import Path

    from napari import Viewer
    from napari.layers import Image as ImageLayer


def export_roi_overview(
    viewer: Viewer,
    layers: list[ImageLayer],
    source: Path,
    bounds: list[tuple[int, int, int, int]],
    output: Path,
) -> None:
    """Write a native PNG without allocating a full-image GPU framebuffer.

    Parameters
    ----------
    viewer : napari.Viewer
        Selection viewer whose channel order and display settings are retained.
    layers : list of napari.layers.Image
        Original channel layers, in their source channel order.
    source : pathlib.Path
        Calibrated maximum-Z preview store.
    bounds : list of tuple
        ROI rectangles on the native pixel grid, ordered y0, y1, x0, x1.
    output : pathlib.Path
        Overview PNG path. Its adjacent JSON records display and scale settings.
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
    height, width = array.shape[-2:]
    halo = max(
        8,
        *(
            max(np.shape(options.get("custom_interpolation_kernel_2d", [[1]]))) // 2
            for options in settings
        ),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.png")
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
    try:
        # Qt initializes the OpenGL framebuffer on show, even for offscreen use.
        renderer.window._qt_window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
        renderer.show()
        get_qapp().processEvents()
        # Temporary disk state bounds RAM and is removed even if rendering fails.
        with (
            renderer._layer_slicer.force_sync(),
            TemporaryFile(dir=output.parent) as scratch,
        ):
            scratch.truncate(height * width * 4)
            with mmap.mmap(scratch.fileno(), 0, access=mmap.ACCESS_WRITE) as mapping:
                pixels = np.ndarray((height, width, 4), np.uint8, buffer=mapping)
                try:
                    # Halo samples retain the same interpolation at block edges.
                    for y0 in tqdm(
                        range(0, height, 2048),
                        desc="ROI overview",
                        unit="row of blocks",
                    ):
                        y1 = min(y0 + 2048, height)
                        for x0 in range(0, width, 2048):
                            x1 = min(x0 + 2048, width)
                            a, b = max(0, y0 - halo), min(height, y1 + halo)
                            c, d = max(0, x0 - halo), min(width, x1 + halo)
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
                            pixels[y0:y1, x0:x1] = rendered[
                                y0 - a : y1 - a, x0 - c : x1 - c
                            ]
                    line_width = max(1, round(min(height, width) / 800))
                    outline = (255, 255, 0, 255)
                    for y0, y1, x0, x1 in bounds:
                        pixels[y0 : min(y0 + line_width, y1), x0:x1] = outline
                        pixels[max(y0, y1 - line_width) : y1, x0:x1] = outline
                        pixels[y0:y1, x0 : min(x0 + line_width, x1)] = outline
                        pixels[y0:y1, max(x0, x1 - line_width) : x1] = outline
                    pixel_um = float(reference.scale[-1])
                    target = width * pixel_um / 5
                    power = 10 ** np.floor(np.log10(target))
                    length_um = max(
                        v * power
                        for v in (0.1, 0.2, 0.5, 1, 2, 5)
                        if v * power <= target
                    )
                    bar_width = max(1, round(length_um / pixel_um))
                    font_size = max(10, round(min(height, width) / 55))
                    font = annotation_font(font_size)
                    label = (
                        f"{length_um / 1000:g} mm"
                        if length_um >= 1000
                        else f"{length_um:g} µm"
                    )
                    label_box = font.getbbox(label)
                    margin = max(2, round(min(height, width) / 40))
                    bar_height = max(2, round(min(height, width) / 200))
                    patch_height = min(height, font_size * 2 + bar_height)
                    patch_width = min(width, max(bar_width, label_box[2]) + 2 * margin)
                    right, bottom = width - margin, height - margin
                    left, top = (
                        max(0, right - patch_width),
                        max(0, bottom - patch_height),
                    )
                    # Drawing on the full mapped PIL image would copy the entire mosaic.
                    patch = Image.fromarray(pixels[top:bottom, left:right].copy())
                    draw = ImageDraw.Draw(patch)
                    draw.text(
                        (margin, 0),
                        label,
                        font=font,
                        fill="white",
                        stroke_width=max(1, font_size // 20),
                        stroke_fill="black",
                    )
                    draw.rectangle(
                        (
                            margin,
                            font_size + 4,
                            margin + bar_width - 1,
                            font_size + 3 + bar_height,
                        ),
                        fill="white",
                    )
                    pixels[top:bottom, left:right] = np.asarray(patch)
                    patch.close()
                    image = Image.frombuffer(
                        "RGBA", (width, height), mapping, "raw", "RGBA", 0, 1
                    )
                    try:
                        image.save(temporary, format="PNG")
                    finally:
                        image.close()
                finally:
                    del pixels
        temporary.replace(output)
    finally:
        renderer.close()
        temporary.unlink(missing_ok=True)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "source_path": str(source),
                "shape_yx": [height, width],
                "time_index": timepoint,
                "pixel_size_yx_um": list(reference.scale[-2:]),
                "scale_bar_um": float(length_um),
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
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
