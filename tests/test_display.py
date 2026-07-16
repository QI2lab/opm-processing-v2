import importlib

import numpy as np
from napari_ome_zarr._reader import napari_get_reader
from ome_types import from_xml

from opm_processing.dataio.position_collection import create_position_collection

display_module = importlib.import_module("opm_processing.display")


class FakeLayer:
    def __init__(self):
        self.data = [np.zeros((2, 3, 4, 5), dtype=np.uint16)]
        self.multiscale = True
        self.visible = True
        self.translate = None
        self.contrast_limits = (0, 1)
        self.gamma = 1.0


class FakeViewer:
    def __init__(self, layers):
        self.layers = layers
        self.open_calls = []

    def open(self, path, *, plugin):
        self.open_calls.append((path, plugin))
        return self.layers


def _create_collection(tmp_path):
    raw_path = tmp_path / "sample.zarr"
    raw_path.mkdir()
    data_path = tmp_path / "sample_deskewed.ome.zarr"
    create_position_collection(
        data_path,
        (2, 2, 2, 3, 4, 5),
        (1.0, 0.3, 0.3),
        stage_positions=[[1, 2, 3], [4, 5, 6]],
        channels=["488nm", "561nm"],
    )
    return raw_path, data_path


def test_bf2raw_has_valid_ome_xml_and_root_plugin_reader(tmp_path):
    _, data_path = _create_collection(tmp_path)
    xml_path = data_path / "OME" / "METADATA.ome.xml"

    ome = from_xml(xml_path.read_text(encoding="utf-8"), validate=True)
    assert [image.id for image in ome.images] == ["Image:0", "Image:1"]
    assert ome.images[0].pixels.size_t == 2
    assert ome.images[0].pixels.size_c == 2
    assert [channel.name for channel in ome.images[0].pixels.channels] == [
        "488nm",
        "561nm",
    ]

    reader = napari_get_reader(str(data_path))
    assert reader is not None
    layer_data = reader()
    assert len(layer_data) == 2
    assert layer_data[0][0][0].shape == (2, 2, 3, 4, 5)


def test_display_delegates_root_open_to_napari_plugin(tmp_path, monkeypatch):
    raw_path, data_path = _create_collection(tmp_path)
    layers = [FakeLayer() for _ in range(4)]
    viewer = FakeViewer(layers)

    monkeypatch.setattr(display_module.napari, "Viewer", lambda: viewer)
    monkeypatch.setattr(display_module.napari, "run", lambda: None)
    monkeypatch.setattr(display_module, "link_layers", lambda *_args: None)

    display_module.display(
        raw_path,
        to_display="full",
        time_range=(1, 2),
        pos_range=(1, 2),
    )

    assert viewer.open_calls == [(str(data_path), "napari-ome-zarr")]
    assert layers[0].visible is False
    assert layers[2].visible is True
    assert layers[2].translate == (0.0, 4.0, 5.0, 6.0)
    assert layers[2].data[0].shape == (1, 3, 4, 5)


def test_resolve_data_path_prefers_ome_suffix_with_legacy_fallback(tmp_path):
    raw_path = tmp_path / "sample.zarr"
    legacy_path = tmp_path / "sample_deskewed.zarr"
    legacy_path.mkdir()

    assert display_module._resolve_data_path(raw_path, "full") == legacy_path

    ome_path = tmp_path / "sample_deskewed.ome.zarr"
    ome_path.mkdir()

    assert display_module._resolve_data_path(raw_path, "full") == ome_path
