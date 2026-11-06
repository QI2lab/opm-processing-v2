"""Recover real production SVG backgrounds and reject invalid image metadata."""

import base64
import hashlib
import io
from copy import deepcopy
from xml.etree import ElementTree

import numpy as np
import pytest
from PIL import Image
from scripts.ao_saved_viewing_background import read_saved_viewing_background

pytestmark = pytest.mark.unit

_SVG = "{http://www.w3.org/2000/svg}"
_XLINK = "{http://www.w3.org/1999/xlink}"


def test_real_overlay_svg_recovers_exact_rgb_without_pupils_or_bars(saved_overlay_svg):
    """Known RGB samples retain their orientation through the real production writer."""
    case = saved_overlay_svg
    shape = case.pixels.shape[:2]
    images = list(ElementTree.fromstring(case.path.read_bytes()).iter(f"{_SVG}image"))
    assert len(images) >= 3
    recovered, provenance = read_saved_viewing_background(case.path, shape)
    np.testing.assert_array_equal(recovered, case.pixels)
    assert recovered.dtype == np.uint8 and recovered.flags.c_contiguous
    assert case.page["pixel_shape_yx"] == provenance["pixel_shape_yx"] == list(shape)
    assert (
        provenance["source_svg_sha256"]
        == hashlib.sha256(case.path.read_bytes()).hexdigest()
    )
    embedded = base64.b64decode(images[0].get(f"{_XLINK}href").partition(",")[2])
    assert provenance["embedded_png_sha256"] == hashlib.sha256(embedded).hexdigest()
    assert provenance["svg_image_index"] == 0


def test_duplicate_full_canvas_images_are_ambiguous(saved_overlay_svg):
    """A duplicate real background cannot be chosen arbitrarily."""
    case = saved_overlay_svg
    tree = ElementTree.parse(case.path)
    root = tree.getroot()
    root.append(deepcopy(next(root.iter(f"{_SVG}image"))))
    tree.write(case.path)
    with pytest.raises(ValueError, match="unique full-canvas"):
        read_saved_viewing_background(case.path, case.pixels.shape[:2])


def test_requested_shape_cannot_select_a_pupil_image(saved_overlay_svg):
    """A real saved pupil raster cannot be substituted for its tissue background."""
    case = saved_overlay_svg
    tree = ElementTree.parse(case.path)
    root = tree.getroot()
    images = list(root.iter(f"{_SVG}image"))
    parents = {child: parent for parent in root.iter() for child in parent}
    for image in images[2:]:
        parents[image].remove(image)
    tree.write(case.path)
    encoded = base64.b64decode(images[1].get(f"{_XLINK}href").partition(",")[2])
    with Image.open(io.BytesIO(encoded)) as pupil:
        pupil_shape = (pupil.height, pupil.width)
    assert pupil_shape != case.pixels.shape[:2]
    with pytest.raises(ValueError, match="first embedded image"):
        read_saved_viewing_background(case.path, pupil_shape)
    with pytest.raises(ValueError, match="unique full-canvas"):
        read_saved_viewing_background(case.path, case.pixels.shape[:2][::-1])


@pytest.mark.parametrize(
    "transform",
    (
        "matrix(0.45 0 0 -0.45 0 0)",
        "matrix(0.45 0 0 0.45 1 0)",
        "matrix(0.45 0.1 0 0.45 0 0)",
        "scale(0.5)",
    ),
)
def test_flipped_moved_or_incomplete_canvas_is_rejected(saved_overlay_svg, transform):
    """Changing actual saved canvas geometry must invalidate pixel recovery."""
    case = saved_overlay_svg
    tree = ElementTree.parse(case.path)
    next(tree.getroot().iter(f"{_SVG}image")).set("transform", transform)
    tree.write(case.path)
    with pytest.raises(ValueError):
        read_saved_viewing_background(case.path, case.pixels.shape[:2])


def test_transparent_background_is_not_silently_composited(saved_overlay_svg):
    """An alpha-modified real PNG must be rejected instead of blended."""
    case = saved_overlay_svg
    rgba = Image.fromarray(case.pixels).convert("RGBA")
    rgba.putpixel((3, 2), (*case.pixels[2, 3], 254))
    encoded = io.BytesIO()
    rgba.save(encoded, format="PNG")
    tree = ElementTree.parse(case.path)
    image = next(tree.getroot().iter(f"{_SVG}image"))
    image.set(
        f"{_XLINK}href",
        "data:image/png;base64," + base64.b64encode(encoded.getvalue()).decode("ascii"),
    )
    tree.write(case.path)
    with pytest.raises(ValueError, match="fully opaque"):
        read_saved_viewing_background(case.path, case.pixels.shape[:2])
