"""Recover only the unchanged tissue samples from a rendered viewing SVG."""

import base64
import hashlib
import io
from types import SimpleNamespace
from xml.etree import ElementTree

import numpy as np
import pytest
from PIL import Image
from scripts.ao_saved_viewing_background import read_saved_viewing_background
from scripts.ao_tissue_overlay import write_tissue_overlay

pytestmark = pytest.mark.unit

_SVG = "{http://www.w3.org/2000/svg}"
_XLINK = "{http://www.w3.org/1999/xlink}"


def _write_svg(path, images, *, transform="matrix(0.45 0 0 0.45 0 0)"):
    height, width = images[0].shape[:2]
    root = ElementTree.Element(
        f"{_SVG}svg", {"viewBox": f"0 0 {width * 0.45} {height * 0.45}"}
    )
    for image in images:
        encoded = io.BytesIO()
        Image.fromarray(image).save(encoded, format="PNG")
        ElementTree.SubElement(
            root,
            f"{_SVG}image",
            {
                "width": str(image.shape[1]),
                "height": str(image.shape[0]),
                "transform": transform,
                f"{_XLINK}href": "data:image/png;base64,"
                + base64.b64encode(encoded.getvalue()).decode("ascii"),
            },
        )
    path.write_bytes(ElementTree.tostring(root))


def test_real_overlay_svg_recovers_exact_rgb_without_pupils_or_bars(tmp_path):
    """Asymmetric RGB samples retain their orientation through the real writer."""
    height, width = 192, 256
    y, x = np.indices((height, width))
    image = np.stack((x * 3, y * 5, (x * 2 + y) % 256), axis=-1).astype(np.uint8)
    centers = {
        index: (32 + 64 * row, 40 + 84 * column)
        for index, (row, column) in enumerate(np.ndindex(3, 3))
    }
    geometry = SimpleNamespace(
        image=image,
        extent_yx_um=(0, height, 0, width),
        centers_by_depth_index={0: centers},
        depth_offsets={0: 0},
        provenance={},
    )
    regions = [
        SimpleNamespace(
            depth=0, index=index, pupil=np.ma.array([[0.0, 1.0], [2.0, -1.0]])
        )
        for index in centers
    ]
    page = write_tissue_overlay(
        regions,
        geometry,
        tmp_path,
        timepoint=0,
        pupil_diameter_um=32,
        scale_bar_um=40,
        output_shape_yx=(height, width),
    )["pages"][0]
    svg = tmp_path / "ao_tissue_overlay_t000_z000.svg"
    images = list(ElementTree.fromstring(svg.read_bytes()).iter(f"{_SVG}image"))
    assert len(images) >= 3  # Background, pupil raster and separate colorbar rasters.
    recovered, provenance = read_saved_viewing_background(svg, (height, width))
    np.testing.assert_array_equal(recovered, image)
    assert recovered.dtype == np.uint8 and recovered.flags.c_contiguous
    assert page["pixel_shape_yx"] == provenance["pixel_shape_yx"] == [height, width]
    assert (
        provenance["source_svg_sha256"] == hashlib.sha256(svg.read_bytes()).hexdigest()
    )
    embedded = base64.b64decode(images[0].get(f"{_XLINK}href").partition(",")[2])
    assert provenance["embedded_png_sha256"] == hashlib.sha256(embedded).hexdigest()
    assert provenance["svg_image_index"] == 0


def test_duplicate_full_canvas_images_are_ambiguous(tmp_path):
    """A second native-sized image cannot be mistaken for clean tissue."""
    path = tmp_path / "duplicate.svg"
    image = np.zeros((5, 7, 3), dtype=np.uint8)
    _write_svg(path, [image, image])
    with pytest.raises(ValueError, match="unique full-canvas"):
        read_saved_viewing_background(path, (5, 7))


def test_requested_shape_cannot_select_a_pupil_image(tmp_path):
    """Only the first image with the declared tissue dimensions is eligible."""
    path = tmp_path / "pupil.svg"
    _write_svg(
        path,
        [np.zeros((5, 7, 3), dtype=np.uint8), np.ones((3, 2, 3), dtype=np.uint8)],
    )
    with pytest.raises(ValueError, match="first embedded image"):
        read_saved_viewing_background(path, (3, 2))
    with pytest.raises(ValueError, match="unique full-canvas"):
        read_saved_viewing_background(path, (7, 5))


@pytest.mark.parametrize(
    "transform",
    (
        "matrix(0.45 0 0 -0.45 0 0)",
        "matrix(0.45 0 0 0.45 1 0)",
        "matrix(0.45 0.1 0 0.45 0 0)",
        "scale(0.5)",
    ),
)
def test_flipped_moved_or_incomplete_canvas_is_rejected(tmp_path, transform):
    """The recovered pixels must occupy the complete, unflipped saved canvas."""
    path = tmp_path / "transformed.svg"
    _write_svg(path, [np.zeros((5, 7, 3), dtype=np.uint8)], transform=transform)
    with pytest.raises(ValueError):
        read_saved_viewing_background(path, (5, 7))


def test_transparent_background_is_not_silently_composited(tmp_path):
    """Recovering the background must never blend it with another raster."""
    image = np.full((5, 7, 4), 255, dtype=np.uint8)
    image[2, 3, 3] = 254
    path = tmp_path / "transparent.svg"
    _write_svg(path, [image])
    with pytest.raises(ValueError, match="fully opaque"):
        read_saved_viewing_background(path, (5, 7))
