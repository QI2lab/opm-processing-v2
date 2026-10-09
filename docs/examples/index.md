# Simulation scripts

Repository simulations and figure generators live in `scripts/`. Run them as
modules from the repository root so package imports resolve. Their opening
docstrings describe inputs, calculations, and outputs. Generated data and figures
belong in ignored `diagnostics/` directories.

```bash
uv run python -m scripts.opm_simulation --output diagnostics/meridian_sphere
uv run --with matplotlib python -m scripts.plot_opm_simulation diagnostics/meridian_sphere/scan_0.2um
```

| Workflow | Description |
| --- | --- |
| [Meridian sphere](../meridian_sphere_simulation.md) | Known fluorescent object, optical forward model, and camera sampling |
| [Publication figure](../opm_publication_figure.md) | Acquisition and reconstruction panels at multiple scan spacings |
| [Fixed-plane series](../fixed_plane_sampling_series.md) | Object passage through a stationary oblique plane |

Each workflow page records the physical assumptions and commands used for its
figures. Experimental outcomes apply to those specified models and sampling;
they do not establish performance for every specimen or optical mismatch.
