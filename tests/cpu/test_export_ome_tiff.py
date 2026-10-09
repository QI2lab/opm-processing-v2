"""Simulated disk-to-disk OME-BigTIFF exports with pixel and metadata truth."""

import json

import numpy as np
import pytest
import tifffile
from ome_types import from_xml
from typer.testing import CliRunner
from yaozarrs import open_group, v05

from opm_processing.export_ome_tiff import app


@pytest.mark.integration
@pytest.mark.parametrize("source_xml,workers", [(True, 1), (False, 3)])
def test_fused_pair_roundtrip_pixels_and_metadata(
    tmp_path, fused_export, source_xml, workers
):
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
    destination = tmp_path / "exports"
    runner = CliRunner()
    result = runner.invoke(
        app, [str(tmp_path), "--output", str(destination), "--workers", str(workers)]
    )
    assert result.exit_code == 0, result.exception
    for source, expected in (
        (fused_export.fused_path, fused_export.truth),
        (fused_export.projection_path, fused_export.projection),
    ):
        target = destination / (source.name.removesuffix(".ome.zarr") + ".ome.tif")
        with tifffile.TiffFile(target) as tif:
            assert tif.is_bigtiff and tif.is_ome
            assert 'PhysicalSizeXUnit="µm"' in tif.pages[0].description
            assert 'Name="488nm α"' in tif.pages[0].description
            assert len(tif.series) == 1 and len(tif.series[0].levels) == 1
            np.testing.assert_array_equal(
                tif.asarray().reshape(expected.shape), expected
            )
            assert tif.series[0].dtype == np.dtype(fused_export.dtype)
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
            ) == fused_export.spacing
            assert pixels.physical_size_x_unit.value == "µm"
            assert [channel.name for channel in pixels.channels] == ["488nm α", "637nm"]
            assert [channel.excitation_wavelength for channel in pixels.channels] == [
                488,
                637,
            ]
            assert pixels.channels[0].color.as_hex() == "#3c9"
            if source_xml:
                assert (
                    image.acquisition_date
                    == fused_export.source_ome.images[0].acquisition_date
                )
                assert pixels.channels[0].emission_wavelength == 525
                assert pixels.time_increment == 8.5
                assert (
                    exported.structured_annotations.comment_annotations[0].value
                    == "Simulated specimen μ"
                )
            z_origin = (
                fused_export.origin[0]
                if expected.shape[2] == 3
                else fused_export.origin[0] + 0.23
            )
            assert image.stage_label.z == z_origin
            for index, plane in enumerate(pixels.planes):
                t, c, z_index = np.unravel_index(index, expected.shape[:3])
                assert (plane.the_t, plane.the_c, plane.the_z) == (t, c, z_index)
                assert (plane.position_z, plane.position_y, plane.position_x) == (
                    z_origin + z_index * fused_export.spacing[0],
                    *fused_export.origin[1:],
                )
                assert (
                    plane.exposure_time == (10, 15)[c]
                    and plane.exposure_time_unit.value == "ms"
                )
            annotations = exported.structured_annotations.map_annotations[-1].value
            assert (
                json.loads(annotations["ProcessingState"])
                == fused_export.state.document
            )
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
                10_000 / fused_export.spacing[2]
            )
    # Existing exports are left intact; an explicit overwrite still round-trips pixels.
    before = (destination / "sample_fused.ome.tif").read_bytes()
    rejected = runner.invoke(
        app, [str(fused_export.fused_path), "--output", str(destination)]
    )
    assert isinstance(rejected.exception, FileExistsError)
    assert (destination / "sample_fused.ome.tif").read_bytes() == before
    replaced = runner.invoke(
        app, [str(fused_export.fused_path), "--output", str(destination), "--overwrite"]
    )
    assert replaced.exit_code == 0, replaced.exception
    np.testing.assert_array_equal(
        tifffile.imread(destination / "sample_fused.ome.tif").reshape(
            fused_export.truth.shape
        ),
        fused_export.truth,
    )
    # Reproduce the former ASCII entity encoding, then repair only OME tags.
    compressed_tiles = {}
    for name in ("sample_fused", "sample_max_z_fused"):
        target = destination / f"{name}.ome.tif"
        with tifffile.TiffFile(target, mode="r+") as tif:
            compressed_tiles[name] = []
            for page in tif.pages:
                for offset, count in zip(
                    page.dataoffsets, page.databytecounts, strict=False
                ):
                    tif.filehandle.seek(offset)
                    compressed_tiles[name].append(
                        (offset, count, tif.filehandle.read(count))
                    )
            tag = tif.pages[0].tags["ImageDescription"]
            tag.overwrite(tag.value.encode("ascii", "xmlcharrefreplace"))
    repaired = runner.invoke(
        app,
        [str(fused_export.fused_path), "--output", str(destination), "--metadata-only"],
    )
    assert repaired.exit_code == 0, repaired.exception
    for name, expected in (
        ("sample_fused", fused_export.truth),
        ("sample_max_z_fused", fused_export.projection),
    ):
        with tifffile.TiffFile(destination / f"{name}.ome.tif") as tif:
            assert 'PhysicalSizeXUnit="µm"' in tif.pages[0].description
            assert 'Name="488nm α"' in tif.pages[0].description
            np.testing.assert_array_equal(
                tif.asarray().reshape(expected.shape), expected
            )
            offsets = [
                (offset, count)
                for page in tif.pages
                for offset, count in zip(
                    page.dataoffsets, page.databytecounts, strict=False
                )
            ]
            assert offsets == [
                (offset, count) for offset, count, _ in compressed_tiles[name]
            ]
            for offset, count, encoded in compressed_tiles[name]:
                tif.filehandle.seek(offset)
                assert tif.filehandle.read(count) == encoded
    assert not list(destination.glob("*.tmp"))


@pytest.mark.integration
def test_standalone_planar_fused_pair(tmp_path, fused_store_factory):
    """A portable singleton TCZ image retains NGFF channels and calibrated time."""
    y, x = np.mgrid[:17, :19]
    truth = (100 * ((y - 8) ** 2 + (x - 9) ** 2 < 16)).astype(np.uint16)[
        None, None, None
    ]
    for name in ("sample_fused", "sample_max_z_fused"):
        fused_store_factory(
            truth,
            name=name,
            spacing=(0.3, 0.2, 0.2),
            time_step_s=3.5,
            time_origin_s=9.75,
            chunks=(1, 1, 1, 17, 19),
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
