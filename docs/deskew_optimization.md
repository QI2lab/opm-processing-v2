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
rows. The existing chunk wrapper uses an empty allocation and writes every row.
Allocation stays outside the Numba parallel kernel: Numba's parallel zero-fill
would touch the whole direct output and lose the benefit on small acquisitions.
There are no kernel factories, scheduling variants, reusable output buffers,
precomputed coordinate maps, or new processing options.

The CLI keeps its existing routing. This change does not introduce automatic
chunking where the CLI previously called direct deskew. The existing chunked
wrapper retains its default threshold, overlaps, cropping, and assembly.

## Validation

Default tests check known camera transfer functions, a diffraction point source
with illumination and shot noise, a laboratory-frame shell, X-coordinate
invariance, Z averaging, and known intensities across the default chunk boundary
using a mocked store.

Performance checks are optional integration tests in `tests/benchmarks`, skipped
unless `--run-benchmarks` is given. See [the test instructions](../tests/README.md#optional-cpu-pipeline-benchmarks).
They load the local `main` implementation from git, validate every output voxel,
exclude compilation and warmup, and report medians without speed assertions.
Timing covers CPU correction and deskew, not disk I/O or deconvolution.

## Measured timings

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

The CPU-selected suite completed with 193 passed, 18 optional benchmarks skipped,
and 57 GPU-marked cases deselected. Nine existing fusion/registration integration
cases attempted CUDA execution and failed because CuPy could not find toolkit
headers. All nine failures reproduced against an isolated copy of the same main
commit. GPU solver validation remains limited by that environment issue.
