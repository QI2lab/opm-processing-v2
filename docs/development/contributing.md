# Contributing

Run development work from a repository checkout. Follow the
[installation guide](../getting-started/installation.md), then install development
dependencies:

```bash
uv sync --group dev
uv run prek install
```

## Code conventions

Use the existing module layout and NumPy-style docstrings. Explain parameters,
units, returned values, and meaningful algorithm conventions. Keep helpers and
state limited to what the calculation needs. The controlled acquisition format
is the input contract; support additional formats only when explicitly required.

Core processing belongs in `src/opm_processing`. Simulations, experiments, and
figure generators belong in `scripts/`, with a clear opening docstring and module
imports. Generated reports, figures, and temporary acquisitions belong under
ignored `diagnostics/` or pytest temporary directories.

## Validate changes

```bash
uv run prek run --all-files
uv run pytest --strict-markers
git diff --check
```

Tests must be numerical unit tests or simulated disk-to-disk integration tests.
Use the [testing guide](testing.md) for CUDA selection and physical acceptance.
Run performance measurements outside pytest and record the conditions and
comparison baseline. Deskew, fusion, and deconvolution optimizations must retain
their stated geometry and numerical behavior.

Prek runs Ruff linting and formatting, spelling, project metadata validation,
and GitHub Actions validation. The same hooks run on pushes and pull requests.
Ruff follows the MERFISH repository's rule families, with NumPy docstrings
required here. Automatic fixes use Ruff's safe fixes.

## Documentation

Update command references when options change and method pages when numerical
behavior changes. Mark experimental and withdrawn work explicitly. Keep the
README focused on installation and a first workflow; put explanations in the
site. See [building documentation](documentation.md).
