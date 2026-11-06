# map-legacy-opm

Expose a legacy raw OPM Zarr v2 array as a Bio-Formats2Raw OME-Zarr acquisition,
using [refkv](https://github.com/d-v-b/refkv) references to its existing compressed
chunks. No raw pixels or original metadata are changed or copied.

```powershell
uv run map-legacy-opm "F:\20250513_humanOB"
uv run inspect-opm "F:\20250513_humanOB"
uv run process "F:\20250513_humanOB"
```

The first command creates `whole_OB_slice_polya.ome.zarr` beside the original
`whole_OB_slice_polya.zarr`. Acquisition discovery selects the mapped acquisition
when both are present. Processing writes ordinary modern outputs beside them.
Add processing options such as illumination correction as appropriate for the
acquisition; this command only supplies format compatibility.

| Argument or option | Description |
| --- | --- |
| `SOURCE` | Legacy raw store or a directory containing exactly one legacy array. |
| `--output PATH` | New mapping directory; default is `<source-stem>.ome.zarr`. Existing destinations are refused. |
| `--workers INTEGER` | Parallel filesystem checks, from 1 to 32; default 8. |

The mapper validates the complete frame index coverage, camera calibration,
channel settings, and stage trajectory before publishing a mapping. Stage
coordinates are stored in micrometers with three decimal places. Missing raw
chunk files cause an error, including legacy arrays whose fill value is null.

Supported legacy inputs are single-timepoint stage-X scans in unsigned 16-bit,
C-order TPCZYX arrays with Blosc compression, no filters, and singleton position
chunks. Unsupported geometry or inconsistent metadata is rejected. Recorded
hardware exposure fields are retained; the ordinary processor determines its
timing metadata from those fields using the existing acquisition contract.

## Referenced Bio-Formats2Raw layout

The mapping contains an OME series list, OME-XML, and one TCZYX image per position:

```text
whole_OB_slice_polya.ome.zarr/
  zarr.json
  OME/zarr.json
  OME/METADATA.ome.xml
  0/zarr.json
  0/0/zarr.json
  ...
  references.vzip
  mapping_report.json
```

`references.vzip` contains the same OME-Zarr hierarchy and references each
compressed chunk in the original store. Keep that raw store available. References
use relative paths on the same drive, so moving both sibling directories together
preserves them; different-drive outputs use absolute file URLs.

The acquisition reader opens these references as read-only TensorStore views,
including when the processor starts a flatfield worker in another process.
The mapping directory has a storage-transformer marker to make unsupported
filesystem readers fail instead of displaying empty arrays. To read with another
Zarr consumer, open the archive through refkv:

```python
import refkv
import refkv.zarr
import zarr

store = refkv.zarr.store(refkv.open("path/to/references.vzip"))
root = zarr.open_group(store, mode="r")
position_zero = root["0/0"]  # TCZYX
```

The archive, original data, mapping report, and generated diagnostic outputs are
local artifacts. The project ignores reference archives, OME-Zarr directories, and temporary test
output directories; these are not source files to commit.
