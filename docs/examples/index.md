# Deconvolution usage

The example processes an OME-Zarr using its recorded acquisition calibration:

```bash
uv run python scripts/deconvolve.py "/path/to/acquisition.ome.zarr" --output "/path/to/processed"
```

Its Python call is:

```python
from pathlib import Path
from opm_processing.process import process

process(
    root_path=Path("/path/to/acquisition.ome.zarr"),
    deconvolve=True,
    save_float32=True,
    output=Path("/path/to/processed"),
)
```

The pipeline selects each channel, timepoint and position using the stored axes.
Camera calibration and scan geometry come from the OME-Zarr; missing read-noise
calibration selects the no-read-noise model. Theoretical PSFs require recorded
optical calibration and emission wavelengths. Measured PSFs can be supplied
instead with `decon_psf_paths=[Path(...), ...]`, in channel order.

See [deconvolution parameters and metadata](../methods/deconvolution.md) for the
array API, units, required optical fields, illumination handling and FFT tiling.
