"""Simulated disk-to-disk OME-BigTIFF exports with pixel and metadata truth."""

import json
from datetime import datetime, timezone

import numpy as np
import pytest
import tifffile
import zarr
from ome_types import from_xml, to_xml
from ome_types.model import CommentAnnotation, AnnotationRef
from typer.testing import CliRunner
from yaozarrs import open_group, v05
from yaozarrs.write.v05 import prepare_image

from opm_processing.dataio.position_collection import create_position_collection
from opm_processing.dataio.processing_state import ProcessingState
from opm_processing.export_ome_tiff import app


@pytest.mark.integration
@pytest.mark.parametrize("source_xml,workers", [(True, 1), (False, 3)])
def test_fused_pair_roundtrip_pixels_and_metadata(tmp_path, source_xml, workers):
    """Preserve a calibrated fluorescent object, channel settings, and provenance.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated on-disk acquisition and export directory.
    source_xml : bool
        Include a rich source OME companion, or use acquisition NGFF alone.
    workers : int
        Serial or threaded TIFF compression worker count.
    """
    dtype = np.uint16
    z, y, x = np.mgrid[:3, :515, :529]
    object_pixels = 900 * np.exp(
        -((z - 1) ** 2 + ((y - 251) / 45) ** 2 + ((x - 287) / 39) ** 2)
    )
    # Add known emitters at TIFF tile boundaries and at both partial-tile edges.
    object_pixels[:, 511:514, 510:516] += 233
    object_pixels[:, -1, -1] = (7, 11, 17)
    truth = np.stack(
        [
            np.stack([object_pixels, object_pixels * 2.3]),
            np.stack([object_pixels * 0.7, object_pixels * 1.4]),
        ]
    ).astype(dtype)
    projection = truth.max(axis=2, keepdims=True)
    raw_path = tmp_path / "sample.ome.zarr"
    raw = create_position_collection(
        raw_path,
        (2, 1, 2, 3, 5, 7),
        (0.4, 0.115, 0.115),
        channels=("488nm", "637nm"),
        dtype=dtype,
        attributes={
            "opm_v2": {
                "index_sizes": {"t": 2, "p": 1, "c": 2, "z": 3},
                "configuration": {
                    "acq_config": {
                        "opm_mode": "mirror",
                        "DAQ": {
                            "channel_states": [True, True],
                            "channel_exposures_ms": [10, 15],
                            "channel_powers": [12, 18],
                            "image_mirror_step_um": 0.4,
                        },
                    }
                },
            }
        },
    )
    raw.arrays[0].write(truth[..., :5, :7]).result()
    raw_group = zarr.open_group(raw_path, mode="a")["0"]
    image_metadata = (
        open_group(raw_path)["0"]
        .ome_metadata()
        .model_copy(
            update={
                "omero": v05.Omero(
                    channels=[
                        v05.OmeroChannel(
                            label=name,
                            color=color,
                            window=v05.OmeroWindow(start=0, end=2000, min=0, max=2000),
                        )
                        for name, color in (("488nm α", "33CC99"), ("637nm", "FF3300"))
                    ]
                )
            }
        )
    )
    raw_group.attrs["ome"] = image_metadata.model_dump(mode="json", exclude_none=True)
    companion = raw_path / "OME/METADATA.ome.xml"
    source_ome = from_xml(companion.read_text(encoding="utf-8"))
    if source_xml:
        source_ome.images[0].acquisition_date = datetime(
            2026, 10, 5, 19, 12, 32, tzinfo=timezone.utc
        )
        source_ome.images[0].pixels.time_increment = 8.5
        source_ome.images[0].pixels.channels[0].emission_wavelength = 525
        source_ome.structured_annotations.comment_annotations.append(
            CommentAnnotation(id="Annotation:sample", value="Simulated specimen μ")
        )
        source_ome.images[0].annotation_refs.append(
            AnnotationRef(id="Annotation:sample")
        )
        companion.write_text(to_xml(source_ome), encoding="utf-8")
    else:
        companion.unlink()

    fused_path = tmp_path / "sample_fused.ome.zarr"
    projection_path = tmp_path / "sample_max_z_fused.ome.zarr"
    spacing = (0.23, 0.115, 0.119)
    origin = (-40.5, 20.25, -11.75)
    for path, data, z_origin in (
        (fused_path, truth, origin[0]),
        (projection_path, projection, origin[0] + 0.23),
    ):
        dims = [
            v05.TimeAxis(name="t"),
            v05.ChannelAxis(name="c"),
            *[v05.SpaceAxis(name=axis, unit="micrometer") for axis in "zyx"],
        ]
        image = v05.Image(
            multiscales=[
                v05.Multiscale(
                    axes=dims,
                    datasets=[
                        v05.Dataset(
                            path="0",
                            coordinateTransformations=[
                                v05.ScaleTransformation(scale=[1, 1, *spacing]),
                                v05.TranslationTransformation(
                                    translation=[0, 0, z_origin, *origin[1:]]
                                ),
                            ],
                        ),
                        v05.Dataset(
                            path="1",
                            coordinateTransformations=[
                                v05.ScaleTransformation(
                                    scale=[1, 1, *[v * 2 for v in spacing]]
                                ),
                            ],
                        ),
                    ],
                )
            ]
        )
        _, arrays = prepare_image(
            path,
            image,
            [
                (data.shape, np.dtype(dtype)),
                (data[..., ::2, ::2, ::2].shape, np.dtype(dtype)),
            ],
            chunks=(1, 1, 1, 128, 128),
            writer="tensorstore",
            extra_attributes={"specimen": "Simulated α"},
            overwrite=True,
        )
        arrays["0"].write(data).result()
        arrays["1"].write(np.zeros(arrays["1"].shape, dtype=dtype)).result()
    processed = tmp_path / "sample_decon_deskewed.ome.zarr"
    state = ProcessingState.create(tmp_path / "sample.processing.json", raw_path)
    state.initialize_run(processed, configuration={"deconvolve": True}, overwrite=True)
    state.save_registration(processed, configuration={}, pairwise_metrics={})
    state.complete_registration(processed, fused_path=fused_path, tiles=[])
    state.set_registered_max_projection(processed, max_projection_path=projection_path)
    destination = tmp_path / "exports"
    runner = CliRunner()
    result = runner.invoke(
        app, [str(tmp_path), "--output", str(destination), "--workers", str(workers)]
    )
    assert result.exit_code == 0, result.exception
    for source, expected in ((fused_path, truth), (projection_path, projection)):
        target = destination / (source.name.removesuffix(".ome.zarr") + ".ome.tif")
        with tifffile.TiffFile(target) as tif:
            assert tif.is_bigtiff and tif.is_ome
            assert len(tif.series) == 1 and len(tif.series[0].levels) == 1
            np.testing.assert_array_equal(
                tif.asarray().reshape(expected.shape), expected
            )
            assert tif.series[0].dtype == np.dtype(dtype)
            assert len(tif.pages) == int(np.prod(expected.shape[:3]))
            assert all(
                page.aspage().is_tiled
                and page.aspage().compression == tifffile.COMPRESSION.ADOBE_DEFLATE
                for page in tif.pages
            )
            assert all(
                page.aspage().predictor == tifffile.PREDICTOR.HORIZONTAL
                for page in tif.pages
            )
            exported = from_xml(tif.ome_metadata, validate=True)
            image = exported.images[0]
            pixels = image.pixels
            assert (
                pixels.size_t,
                pixels.size_c,
                pixels.size_z,
                pixels.size_y,
                pixels.size_x,
            ) == expected.shape
            assert pixels.dimension_order.value == "XYZCT"
            assert pixels.metadata_only is None and not pixels.bin_data_blocks
            assert pixels.tiff_data_blocks[0].plane_count == len(tif.pages)
            assert (
                pixels.physical_size_z,
                pixels.physical_size_y,
                pixels.physical_size_x,
            ) == spacing
            assert pixels.physical_size_x_unit.value == "µm"
            assert [channel.name for channel in pixels.channels] == ["488nm α", "637nm"]
            assert [channel.excitation_wavelength for channel in pixels.channels] == [
                488,
                637,
            ]
            assert pixels.channels[0].color.as_hex() == "#3c9"
            if source_xml:
                assert image.acquisition_date == source_ome.images[0].acquisition_date
                assert pixels.channels[0].emission_wavelength == 525
                assert pixels.time_increment == 8.5
                assert (
                    exported.structured_annotations.comment_annotations[0].value
                    == "Simulated specimen μ"
                )
            z_origin = origin[0] if expected.shape[2] == 3 else origin[0] + 0.23
            assert image.stage_label.z == z_origin
            for index, plane in enumerate(pixels.planes):
                t, c, z_index = np.unravel_index(index, expected.shape[:3])
                assert (plane.the_t, plane.the_c, plane.the_z) == (t, c, z_index)
                assert (plane.position_z, plane.position_y, plane.position_x) == (
                    z_origin + z_index * spacing[0],
                    *origin[1:],
                )
                assert (
                    plane.exposure_time == (10, 15)[c]
                    and plane.exposure_time_unit.value == "ms"
                )
            annotations = exported.structured_annotations.map_annotations[-1].value
            assert json.loads(annotations["ProcessingState"]) == state.document
            assert json.loads(annotations["NGFF"]) == dict(open_group(source).attrs)
            assert json.loads(annotations["Acquisition"])["channel_names"] == [
                "488nm α",
                "637nm",
            ]
            assert json.loads(annotations["AcquisitionSettings"])["configuration"][
                "acq_config"
            ]["DAQ"]["channel_powers"] == [12, 18]
            x_resolution = tif.pages[0].tags["XResolution"].value
            assert x_resolution[0] / x_resolution[1] == pytest.approx(
                10_000 / spacing[2]
            )
    # Existing exports are left intact; an explicit overwrite still round-trips pixels.
    before = (destination / "sample_fused.ome.tif").read_bytes()
    rejected = runner.invoke(app, [str(fused_path), "--output", str(destination)])
    assert isinstance(rejected.exception, FileExistsError)
    assert (destination / "sample_fused.ome.tif").read_bytes() == before
    replaced = runner.invoke(
        app, [str(fused_path), "--output", str(destination), "--overwrite"]
    )
    assert replaced.exit_code == 0, replaced.exception
    np.testing.assert_array_equal(
        tifffile.imread(destination / "sample_fused.ome.tif").reshape(truth.shape),
        truth,
    )
    assert not list(destination.glob("*.tmp"))


@pytest.mark.integration
def test_standalone_planar_fused_pair(tmp_path):
    """A portable singleton TCZ image retains NGFF channels and calibrated time."""
    y, x = np.mgrid[:17, :19]
    truth = (100 * ((y - 8) ** 2 + (x - 9) ** 2 < 16)).astype(np.uint16)[
        None, None, None
    ]
    for name in ("sample_fused", "sample_max_z_fused"):
        image = v05.Image(
            multiscales=[
                v05.Multiscale(
                    axes=[
                        v05.TimeAxis(name="t", unit="second"),
                        v05.ChannelAxis(name="c"),
                        *[
                            v05.SpaceAxis(name=axis, unit="micrometer")
                            for axis in "zyx"
                        ],
                    ],
                    datasets=[
                        v05.Dataset(
                            path="0",
                            coordinateTransformations=[
                                v05.ScaleTransformation(scale=[3.5, 1, 0.3, 0.2, 0.2]),
                                v05.TranslationTransformation(
                                    translation=[9.75, 0, 0, 0, 0]
                                ),
                            ],
                        )
                    ],
                )
            ],
            omero=v05.Omero(
                channels=[
                    v05.OmeroChannel(
                        label="fluorescent disk",
                        color="00FF00",
                        window=v05.OmeroWindow(start=0, end=100, min=0, max=100),
                    )
                ]
            ),
        )
        _, arrays = prepare_image(
            tmp_path / f"{name}.ome.zarr",
            image,
            [(truth.shape, truth.dtype)],
            writer="tensorstore",
            chunks=(1, 1, 1, 17, 19),
            overwrite=True,
        )
        arrays["0"].write(truth).result()
    result = CliRunner().invoke(app, [str(tmp_path)])
    assert result.exit_code == 0, result.exception
    for name in ("sample_fused", "sample_max_z_fused"):
        with tifffile.TiffFile(tmp_path / f"{name}.ome.tif") as tif:
            np.testing.assert_array_equal(tif.asarray(), truth[0, 0, 0])
            assert tif.is_bigtiff and tif.is_ome
            pixels = from_xml(tif.ome_metadata, validate=True).images[0].pixels
            assert pixels.size_t == pixels.size_c == pixels.size_z == 1
            assert pixels.channels[0].name == "fluorescent disk"
            assert pixels.time_increment == 3.5
            assert pixels.planes[0].position_z == 0
            assert pixels.planes[0].delta_t == 9.75
