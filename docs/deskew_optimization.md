# CPU correction and orthogonal interpolation

Camera calibration, illumination correction, and orthogonal interpolation remain
separate processing stages. Camera calibration reads uint16 counts once and writes
float32 values once, including the optional detector-X gain. Illumination division
runs independently after empty-tile detection. Flatfield sample loading also uses
the shared camera calibration. Both kernels parallelize detector rows and
preserve NumPy broadcasting for other supported input shapes.

Orthogonal interpolation parallelizes output ZY rows. It computes the source
quartet once per row and interpolates X with scalar float32 arithmetic, avoiding
row-sized temporaries and repeated zero-intensity scans. Source coordinates remain
float64. The four-sample geometry, intensity normalization, clipping, padding,
reverse-Z behavior, and Z averaging convention are unchanged. The default Z
averaging factor remains **2**; factors other than 1 and 2 remain supported.

Direct volumes use a NumPy zero allocation and skip geometrically unsupported
rows. The optional empty-allocation path writes every row.
Allocation stays outside the Numba parallel kernel: Numba's parallel zero-fill
would touch the whole direct output and lose the benefit on small acquisitions.
There are no kernel factories, scheduling variants, reusable output buffers,
precomputed coordinate maps, or new processing options.

The CLI keeps its direct-deskew routing. The unused combined chunked wrapper
was removed during cleanup; the processing commands did not call it.

## Validation

Tests check known camera transfer functions, a diffraction point source with
illumination and shot noise, a laboratory-frame shell, X-coordinate invariance,
and Z averaging. Fusion integration tests process simulated objects from disk
inputs to verified disk outputs.

Performance measurements use one-off comparisons against implementations read
from git. They exclude compilation and warmup and validate output values.
Benchmark-only tests are not part of the test suite.

## Historical timings

Measured on Windows on 2026-09-29, with 24 logical CPUs and 128 GiB RAM, against
`main` at `b999cfdc7ea3d781c8f4e6eae45c4c8695baf0cd`. Values are median seconds,
excluding compilation and one full warmup per implementation: five trials for
small/large and three for chunked. Sampling is 0.4 um along scan, 0.115 um in the
camera plane, 30 degrees, and Z averaging 2. All 18 comparisons passed
`rtol=1e-6, atol=1e-5`; the maximum absolute difference was 0.00006103515625.

Small is 10 x 256 x 1900, large is 500 x 512 x 1900, and chunked is
5000 x 512 x 1900. Chunked timings include output assembly. The full three-stage
chunked case uses the actual production wrapper, including main's in-place
illumination division; the first two scopes use its default schedule with the
omitted stages removed. Do not subtract separate rows to estimate stage costs.

| Acquisition | Threads | Stages | Main (s) | Integrated (s) | Speedup |
| --- | ---: | --- | ---: | ---: | ---: |
| Small | 4 | Interpolation | 0.0597 | 0.0024 | 24.74x |
| Small | 16 | Interpolation | 0.0638 | 0.0021 | 29.95x |
| Small | 4 | Camera + interpolation | 0.0687 | 0.0062 | 11.14x |
| Small | 16 | Camera + interpolation | 0.0768 | 0.0050 | 15.45x |
| Small | 4 | Camera + illumination + interpolation | 0.0736 | 0.0077 | 9.61x |
| Small | 16 | Camera + illumination + interpolation | 0.0909 | 0.0073 | 12.47x |
| Large | 4 | Interpolation | 1.3138 | 0.2447 | 5.37x |
| Large | 16 | Interpolation | 1.0522 | 0.2281 | 4.61x |
| Large | 4 | Camera + interpolation | 2.4406 | 0.5123 | 4.76x |
| Large | 16 | Camera + interpolation | 2.1538 | 0.4776 | 4.51x |
| Large | 4 | Camera + illumination + interpolation | 2.8955 | 0.8205 | 3.53x |
| Large | 16 | Camera + illumination + interpolation | 2.5810 | 0.7099 | 3.64x |
| Chunked | 4 | Interpolation | 15.5175 | 6.2470 | 2.48x |
| Chunked | 16 | Interpolation | 13.8055 | 5.9601 | 2.32x |
| Chunked | 4 | Camera + interpolation | 27.3621 | 8.5954 | 3.18x |
| Chunked | 16 | Camera + interpolation | 25.4921 | 8.3875 | 3.04x |
| Chunked | 4 | Camera + illumination + interpolation | 28.5575 | 11.3179 | 2.52x |
| Chunked | 16 | Camera + illumination + interpolation | 25.7509 | 11.0732 | 2.33x |

At the time of these measurements, the correctness suite passed with
`OPM_REQUIRE_GPU=1`: **259 passed**, with 18 optional benchmarks skipped. Those
benchmark tests and the unused combined chunked wrapper were subsequently removed.

## Cleanup verification, 2026-10-07

Comparison against commit `1dfb002` confirms that all five deskew functions,
including their Numba decorators, retain identical executable syntax. All four
processing-command deskew calls retain the same arguments. Fusion accumulation,
normalization, region partitioning, block sizing, parallel scheduling, bounded
asynchronous writes, and pyramid generation also retain identical executable
syntax. The source-cache budget, registration read-ahead, and pinned GPU staging
remain in place.

The review found an unnecessary clipping allocation in the new depth-normalization
output conversion. Matching input/output dtypes now return the original array.
Unscaled uint16 tile interiors again use direct copying without an intermediate
pixel buffer; scaled float32 values still clip when converted to uint16. The
numerical unit test verifies saturation and preservation of the original buffer.

One-off timings on this Windows workstation exclude compilation and warmup and
report medians of five alternating trials. Deskew uses 16 Numba threads and the
same geometry as the historical measurements. Its reference is the implementation
before the speedup at `b999cfd`. Every output voxel agrees within
`rtol=1e-6, atol=1e-5`; maximum absolute difference is `0.0001220703125`.

| Deskew volume | Before speedup (s) | Current (s) | Speedup |
| --- | ---: | ---: | ---: |
| 10 x 256 x 1900 | 0.0680 | 0.0020 | 34.41x |
| 500 x 512 x 1900 | 1.1488 | 0.2679 | 4.29x |

Fusion compares the committed render/read methods with the current methods,
using the unchanged scheduling and accumulation kernels. It reads two simulated
three-channel 32 x 512 x 512 tiles from OME-Zarr and writes the fused level-zero
OME-Zarr array with four workers. Inputs use 8 x 128 x 128 spatial chunks;
outputs use 4 x 128 x 128 chunks, with eight-plane fusion slabs. Depth
normalization is disabled. Both implementations receive the same input handles
and perform a full warmup before timing, so these are warm-cache disk-to-disk
measurements. Registration, metadata creation, and pyramid generation are outside
the timed scope.

| Tile arrangement | Committed (s) | Current (s) |
| --- | ---: | ---: |
| 50% X overlap | 0.4274 | 0.4273 |
| Separated fields | 0.4640 | 0.4525 |

Both outputs were reopened and matched the simulated Gaussian object's known
intensities exactly, including empty canvas space. Direct-copy and blended-region
counts also matched. These measurements show no slowdown in the checked paths;
the small fusion timing differences are within run-to-run variation.

All 56 affected deskew, fusion, depth-normalization, projection, and registration
correctness tests passed with `OPM_REQUIRE_GPU=1`. Temporary timing data was removed.
