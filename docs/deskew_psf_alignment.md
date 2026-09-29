# Tilt investigation: withdrawn production changes

Production orthogonal deskew and PSF generation are restored byte-for-byte to
the repository versions (`opmtools.py` and `opmpsf.py`). The experimental
inverse-affine interpolation, Gaussian Z filtering, axial PSF spacing change,
and their processing-version fields have been withdrawn. No dataset was
modified or reprocessed during this investigation. Experimental scripts and
tests remain under ignored `diagnostics/`; they are not production fixes.

## What the earlier simulation did not establish

The comparison changed the interpolation kernel and assessed alignment with
the nearest Cartesian axis, accepting elongation along either Y or Z. The
change from 15.5 to 0.53 degrees did not establish more faithful reconstruction
or a defect in production orthogonal interpolation. The optical field and
reconstruction kernel also shared underlying vectorial-diffraction assumptions.
The simulation omitted measured light-sheet illumination, detector-area
integration, scanner dynamics, and the actual specimen motion. It is not an
independent validation of the instrument's forward model. The inference that
PSF axial spacing was defective also requires validation of the intended
optical-coordinate convention before any production change.

The cause of the observed live-data tilt remains unconfirmed. Motion blur and
an inadequate forward model remain possible explanations. Matched-model
reconstruction tests validate numerical consistency under their assumptions,
not those assumptions' applicability to the instrument.

## Read-only metadata audit

For `both_lasers.ome.zarr` and its existing processed output:

| Parameter | Recorded / passed value |
| --- | --- |
| Mode | mirror |
| Raw scan planes | 25 |
| Acquired scan step | 0.8 um |
| Camera pixel size | 0.115 um |
| OPM angle | 30 degrees |
| Scan orientation | positive, no reversal |
| Experimental upsample factor | 2 |
| Reconstruction planes | 49 |
| PSF and deskew scan step | 0.4 um |
| Output Z/Y/X voxel sizes | 0.23 / 0.115 / 0.115 um |
| Output Z/Y/X shape | 64 / 396 / 1600 |

The endpoint span is preserved: 24 * 0.8 = 48 * 0.4 = 19.2 um.
The raw NGFF `z` dimension represents scan index; the output `z` dimension is
Cartesian Z. All 2500 frame records agree on scan step, angle and orientation,
and are ordered as 100 timepoints with 25 scan planes each. Mirror positions
are null, so actual mirror travel cannot be verified independently from these
records. No metadata mismatch was identified in this audit.
