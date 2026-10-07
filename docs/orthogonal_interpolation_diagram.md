# Orthogonal interpolation diagram

**Orthogonal interpolation from two adjacent camera planes to the lab grid.**
Panel A shows adjacent camera planes at scan positions s_i and s_(i+1),
separated by Δs along lab Y. Camera coordinates are u (column) and v (row).
The camera-column direction is transverse to the YZ scan geometry and remains
identical to lab X: X = u. The physical mapping is

    X = u
    Y = s + v cos(θ)
    Z = v sin(θ).

Panel B is a side view at fixed X. The target lab-grid point P is projected
perpendicularly onto each adjacent camera plane, giving Q_i and Q_(i+1).
Faint gray dots show the other lab-grid points awaiting interpolation;
lab-grid lines are omitted and only the active target P is starred.
Open circles mark these projection points; the four emphasized filled circles
are their neighboring camera-row samples. A shaded quadrilateral connects
these four pixels, and dotted spokes connect each contributing pixel to P.
The two colored edges represent interpolation within the camera planes.
Green arrows connect Q_i and Q_(i+1) to P, with green right-angle marks at
both planes. These arrows are normal to the camera planes: this is the
orthogonal construction. The quadrilateral edges and the individual
pixel-to-P spokes are not themselves required to be perpendicular.

The calculation, documented here rather than on the figure, follows the current production implementation. First, compute
s_* = Y − Z cot(θ) and select the adjacent scan positions bracketing s_*.
For each selected plane j, its orthogonal projection has row coordinate
v_j = (Y − s_j) cos(θ) + Z sin(θ). With camera pitch p, let
k_j = floor(v_j / p) and β_j = v_j / p − k_j. Interpolate within that plane:

    L_j = (1 − β_j) I_j[k_j] + β_j I_j[k_j + 1].

The existing production core combines the two values as

    I_out = (p / Δs) (L_i + L_(i+1)).

This expression deliberately preserves the implementation's intensity scale;
the sum is not replaced with distance-weighted interpolation between planes.
For constant input, the core's gain is 2p/Δs before optional Z averaging.
All four samples retain the same camera-column index u = X. Only the YZ
geometry requires interpolation. Origins are omitted for clarity; the diagram
uses 30° and illustrative spacing, not a particular acquisition's parameters.
Boundary handling, padding and optional output-Z averaging are omitted.

The drawing script checks that both Q points lie on their camera planes and
that P−Q is perpendicular to the camera-row direction. A numerical four-sample
example is evaluated by the actual `orthogonal_deskew` function with Z
downsampling disabled. The expected value is 19.5213548685; the float32 result
is 19.5213546753. The two untouched camera columns remain exactly zero.
These are diagram-consistency checks, not changes to the production algorithm.

Exports are PDF, editable-text SVG and 600-dpi PNG under
`diagnostics/orthogonal_interpolation_labeled`. The numerical check is saved
as `geometry_check.json`.

```powershell
uv run --no-sync --with matplotlib python -m scripts.plot_orthogonal_interpolation --output diagnostics/orthogonal_interpolation_labeled
```

Spatial axes are labeled in ?m; the numerical example uses illustrative p = 1 ?m and ?s = 3 ?m. The bottom key identifies camera planes and their pixels, orthogonal projection paths, other lab-grid points, and target P. Scan displacement s is explicitly labeled along lab Y.
