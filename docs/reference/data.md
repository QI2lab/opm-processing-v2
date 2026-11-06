# Dataset layout and metadata

## Acquisition and image axes

The supported input is a current qi2lab opm-v2 filesystem-backed OME-Zarr
acquisition. Its manifest and OME/NGFF metadata provide calibration, channels,
stage positions, and scan geometry.

| Representation | Axes | Meaning |
| --- | --- | --- |
| Logical acquisition view | TPCZYX | Time, position, channel, scan plane, camera Y, camera X |
| Per-position image array | TCZYX | One positioned image series |
| Deskewed tile | TCZYX | Laboratory Z replaces the raw scan-plane axis |
| Fused image | TCZYX | Registered laboratory coordinates |
| Maximum-Z image | TCZYX | Z has length one |

Position collections retain separate positioned arrays and their transforms.
The OME-Zarr image metadata defines physical scales and translations; positions
in the acquisition description may use ZXY or XYZ ordering as named by their
fields. Do not interpret the raw scan-plane dimension as laboratory Z.

Processed stage labels retain the recorded stage reference without orientation
changes. Spatial values are stored to three decimal places.
These values are in micrometers; millimeter inputs are converted before rounding,
preserving the equivalent six decimal places in millimeters.
`opm_processing_stage_positions` declares their ZXY order;
OME StageLabels use physical XYZ. `opm_processing_coordinates` separately records
the placement reference and coordinate version. Local image translations include
the retained scan endpoint, deskew crop, and averaging offsets. Fused transforms
and ROI footprints use that same version. Older stage previews must be regenerated
before their absolute coordinates can be used for ROI selection or AO overlays.

## Output files

`<stem>` is the acquisition name. Files default to the acquisition directory,
or to the directory supplied with `process --output`.

| Output | Contents |
| --- | --- |
| `<stem>_deskewed.ome.zarr` | Calibrated, deskewed position collection |
| `<stem>_decon_deskewed.ome.zarr` | Deconvolved, deskewed position collection |
| `<stem>_max_z_deskewed.ome.zarr` | Per-tile maximum-Z images |
| `<stem>_max_z_decon_deskewed.ome.zarr` | Per-tile deconvolved maximum-Z images |
| `<stem>_fused.ome.zarr` | Registered fused multiscale image |
| `<stem>_max_z_fused.ome.zarr` | Registered fused maximum-Z multiscale image |
| `<stem>_flatfield.ome.tif` | Estimated illumination fields, when requested |
| `<stem>.processing.json` | Processing configuration, completion, source path, and registration state |
| `<stem>_depth_intensity_gains.json` | Fitted gains and depth-normalization provenance |
| `<stem>_roi.json` | Saved physical ROI selection |
| `<stem>_roi/` | Default ROI processing destination |

Native planar acquisitions use their planar output naming convention. Inspect
the command's printed destination and journal to identify its output stores.

`fuse --output-dir` places the fused volume and registered maximum-Z stores in
another folder or drive. Registration and the depth-intensity gain report
remain beside the processed input. External fused stores include an
`opm_processing.processing_state` link to that journal, using a relative path
when possible and an absolute path for cross-drive access. The link contains
no duplicate registration measurements.

## Processing journal

One journal describes the source and outputs for an acquisition. Image stores
contain OME/NGFF image metadata; mutable processing checkpoints are kept in the
journal. Run fingerprints include the settings and geometry needed for compatible
resume. Completion is recorded after writes are durable.

The journal also locates the raw acquisition when processing, fusion, inspection,
or export starts from an output directory. New journals record a relative source
link as well as the absolute provenance path. Keep the journal beside its processed
tile collections and preserve the relative raw/output layout when moving or copying a dataset.
An existing same-named raw store beside the current input takes precedence over
the recorded absolute location, including when the original copy is accessible.
Nested ROI journals can also resolve raw data through the matching acquisition
journal in a parent directory. Separately stored acquisitions on another drive
must remain accessible at their recorded location or be explicitly selected.

Stage-placed projection metadata stores relative links to its source collection.
Live manifests likewise prefer an existing raw store beside the current manifest
over their historical absolute `data_path`.

## Units and pyramids

Spatial transforms use micrometers. Time axes retain the units recorded in the
metadata. Command ranges are zero-based and stop-exclusive; registration shift
and blending options use processed-image pixels in ZYX order.

The fused OME-Zarr volume has reduced spatial levels. Its maximum-Z image is
generated from each registered fused level. OME-TIFF export currently writes
only level zero and preserves a singleton Z axis for the projection. OME-XML
stores channel metadata, physical units, and plane positions, with provenance
annotations when the journal and acquisition are available.

See [acquisition and state functions](python/dataio.md) for the corresponding
Python structures.
