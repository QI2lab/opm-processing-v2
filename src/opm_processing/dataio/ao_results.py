"""Read saved opm-v2 AO coefficient states through native Zarr storage.

Both the historical Zarr v2 groups and current Zarr v3 groups written by
``opm_v2.utils.sensorless_ao.save_optimization_results`` store modal states in
``starting_coeffs`` and ``optimal_coeffs``. This reader does not estimate mirror
shape, reconstruct pupils, or alter coefficient units.
"""

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
from yaozarrs import open_group


@dataclass
class AOOptimizationResult:
    """One saved optimization's native coefficient states and array metadata."""

    zarr_path: Path
    time_index: int
    position_index: int

    def __post_init__(self) -> None:
        """Open the actual generic Zarr group without materializing images."""
        self.zarr_path = Path(self.zarr_path).expanduser().absolute()
        self._group = open_group(self.zarr_path)

    @cached_property
    def metadata(self) -> dict[str, Any]:
        """Return recorded optimization settings without inferring missing modes."""
        return dict(self._group.attrs)

    def _array(self, name: str) -> np.ndarray:
        """Read one named saved array with the installed TensorStore backend."""
        return np.asarray(self._group[name].to_tensorstore().read().result())

    @cached_property
    def starting_coefficients(self) -> np.ndarray:
        """Return the finite saved starting vector in its original OPD units."""
        values = np.asarray(self._array("starting_coeffs"), dtype=float)
        if values.ndim != 1 or not values.size or not np.isfinite(values).all():
            raise ValueError("Saved AO starting coefficients must be a finite vector")
        return values

    @cached_property
    def optimal_coefficients(self) -> np.ndarray:
        """Return one coefficient vector after each recorded modal update."""
        values = np.asarray(self._array("optimal_coeffs"), dtype=float)
        if (
            values.ndim != 2
            or values.shape[1] != self.starting_coefficients.size
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                "Saved AO coefficient history must use the starting modal basis"
            )
        return values

    @cached_property
    def final_coefficients(self) -> np.ndarray:
        """Return the last completed iteration, matching the saved producer state."""
        modes = self.metadata.get("modes_to_optimize", [])
        iterations = int(self.metadata.get("num_iterations", 1))
        if iterations < 1 or not isinstance(modes, (list, tuple)):
            raise ValueError("Saved AO optimization dimensions are invalid")
        expected = iterations * len(modes)
        if self.optimal_coefficients.shape[0] != expected:
            raise ValueError(
                "Saved AO history does not contain every recorded modal update"
            )
        if not expected:
            return self.starting_coefficients.copy()
        return self.optimal_coefficients[-1].copy()
