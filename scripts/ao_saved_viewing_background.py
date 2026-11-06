"""Recover an unchanged tissue background from an existing viewing-copy SVG."""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import re
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import numpy as np
from PIL import Image

_SVG = "{http://www.w3.org/2000/svg}"
_XLINK = "{http://www.w3.org/1999/xlink}"


def _numbers(value: str) -> tuple[float, ...]:
    try:
        numbers = tuple(float(item) for item in re.split(r"[\s,]+", value.strip()))
    except ValueError as error:
        raise ValueError("Invalid saved-background SVG geometry") from error
    if not numbers or not np.isfinite(numbers).all():
        raise ValueError("Invalid saved-background SVG geometry")
    return numbers


def _positive_scale(transform: str) -> tuple[float, float]:
    match = re.fullmatch(r"\s*(matrix|scale)\s*\(([^()]*)\)\s*", transform)
    if match is None:
        raise ValueError("Saved background must have an untranslated positive scale")
    values = _numbers(match[2])
    if match[1] == "matrix" and len(values) == 6:
        a, b, c, d, e, f = values
        if any(value != 0 for value in (b, c, e, f)):
            raise ValueError("Saved background must not be rotated, flipped or moved")
        scale = (a, d)
    elif match[1] == "scale" and len(values) in (1, 2):
        scale = (values[0], values[-1])
    else:
        raise ValueError("Unsupported saved-background SVG transform")
    if min(scale) <= 0:
        raise ValueError("Saved background must have an untranslated positive scale")
    return scale


def _validate_canvas(
    root: ElementTree.Element,
    element: ElementTree.Element,
    parents: dict[ElementTree.Element, ElementTree.Element],
    shape_yx: tuple[int, int],
) -> None:
    height, width = shape_yx
    scale_x, scale_y = _positive_scale(element.get("transform", "scale(1)"))
    for attribute, expected in (
        ("x", 0),
        ("y", 0),
        ("width", width),
        ("height", height),
    ):
        values = _numbers(element.get(attribute, "0"))
        if len(values) != 1 or values[0] != expected:
            raise ValueError("Saved background does not cover the full SVG canvas")
    ancestor = parents.get(element)
    while ancestor is not None:
        if ancestor.get("transform"):
            if _positive_scale(ancestor.get("transform", "")) != (1, 1):
                raise ValueError("Saved background has a transformed SVG ancestor")
        ancestor = parents.get(ancestor)
    viewport = _numbers(root.get("viewBox", ""))
    expected = (0, 0, width * scale_x, height * scale_y)
    if len(viewport) != 4 or not np.allclose(viewport, expected, rtol=1e-9, atol=1e-7):
        raise ValueError("Saved background does not cover the full SVG canvas")


def read_saved_viewing_background(
    svg_path: Path, shape_yx: tuple[int, int]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract the unique opaque full-canvas PNG without overlay compositing.

    Only the first embedded SVG image can be the tissue background. Its pixel
    shape, positive transform, and complete canvas coverage are checked before
    returning RGB samples in their original top-to-bottom orientation.
    """
    if len(shape_yx) != 2 or any(
        isinstance(size, bool) or int(size) != size or size <= 0 for size in shape_yx
    ):
        raise ValueError("Saved background shape must contain two positive integers")
    shape_yx = tuple(int(size) for size in shape_yx)
    svg_path = Path(svg_path).expanduser().resolve()
    source = svg_path.read_bytes()
    try:
        root = ElementTree.fromstring(source)
    except ElementTree.ParseError as error:
        raise ValueError("Cannot parse saved viewing-copy SVG") from error
    if root.tag != f"{_SVG}svg":
        raise ValueError("Saved viewing background must come from an SVG document")
    images = list(root.iter(f"{_SVG}image"))
    matches = []
    for index, element in enumerate(images):
        href = element.get(f"{_XLINK}href", element.get("href", ""))
        if not href.startswith("data:image/png;base64,"):
            raise ValueError("Saved viewing SVG must use embedded PNG images")
        try:
            payload = re.sub(r"\s+", "", href.partition(",")[2])
            encoded = base64.b64decode(payload, validate=True)
            with Image.open(io.BytesIO(encoded)) as candidate:
                if candidate.format != "PNG":
                    raise ValueError("Saved viewing SVG contains a non-PNG image")
                if candidate.size == (shape_yx[1], shape_yx[0]):
                    matches.append((index, element, encoded))
        except (binascii.Error, OSError) as error:
            raise ValueError("Cannot decode saved viewing-copy PNG") from error
    if len(matches) != 1:
        raise ValueError("Expected one unique full-canvas tissue background PNG")
    index, element, encoded = matches[0]
    if index != 0:
        raise ValueError(
            "Full-canvas tissue background must be the first embedded image"
        )
    parents = {child: parent for parent in root.iter() for child in parent}
    _validate_canvas(root, element, parents, shape_yx)
    with Image.open(io.BytesIO(encoded)) as background:
        rgba = np.asarray(background.convert("RGBA"))
    if not np.all(rgba[:, :, 3] == 255):
        raise ValueError("Saved tissue background must be fully opaque")
    rgb = np.array(rgba[:, :, :3], dtype=np.uint8, order="C", copy=True)
    return rgb, {
        "source_svg": str(svg_path),
        "source_svg_sha256": hashlib.sha256(source).hexdigest(),
        "embedded_png_sha256": hashlib.sha256(encoded).hexdigest(),
        "pixel_shape_yx": list(shape_yx),
        "svg_image_index": index,
        "image_transform": element.get("transform", "scale(1)"),
        "extraction": "original opaque tissue PNG only; pupils and colorbars excluded",
    }
