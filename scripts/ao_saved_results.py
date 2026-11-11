"""Portable saved-AO reader with the historical SJS pupil compatibility API.

New exports reconstruct the SDK-validated basis in export_ao_overview. The
``get_pupil`` method retains the historical SJS signed-mode ordering solely for
unversioned saved manifests; it delegates numerical evaluation to the existing
production WaveKit operator.
"""

from pathlib import Path
from typing import Any

import numpy as np

from opm_processing.dataio.ao_results import AOOptimizationResult as CoefficientResult

try:
    from scripts.ao_wavekit_phase import WAVEKIT_MODE_NM, phase_from_modal_coeff
except ModuleNotFoundError as error:
    if error.name != "scripts":
        raise
    from ao_wavekit_phase import WAVEKIT_MODE_NM, phase_from_modal_coeff


class AOOptimizationResult(CoefficientResult):
    """Read real saved states and preserve explicitly requested legacy rendering."""

    def get_pupil(
        self, *, coefficient_state: str = "final", **options: Any
    ) -> np.ma.MaskedArray:
        """Apply SJS's historical odd-azimuthal signed-mode lookup unchanged."""
        states = {
            "starting": self.starting_coefficients,
            "final": self.final_coefficients,
            "change": self.final_coefficients - self.starting_coefficients,
        }
        if coefficient_state not in states:
            raise ValueError("Unknown AO coefficient state")
        coefficients = states[coefficient_state]
        # SJS uses sine then cosine for odd |m| pairs, and cosine then sine
        # for even |m| pairs. SDK mode evaluation remains the shared operator.
        indices = [
            WAVEKIT_MODE_NM.index((n, -m if abs(m) % 2 else m)) + 1
            for n, m in WAVEKIT_MODE_NM[: coefficients.size]
        ]
        return phase_from_modal_coeff(coefficients, mode_indices=indices, **options)

    def plot_coefficients(
        self, *, save_path: Path, show: bool = False
    ) -> tuple[Any, Any]:
        """Plot saved starting/final OPD vectors and their signed difference."""
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(8, 10))
        indices = np.arange(self.starting_coefficients.size)
        for offset, values, label in (
            (-0.25, self.starting_coefficients, "Starting"),
            (0, self.final_coefficients, "Final"),
            (0.25, self.final_coefficients - self.starting_coefficients, "Change"),
        ):
            axis.barh(indices + offset, values, height=0.25, label=label)
        axis.set_xlabel("OPD (micrometers)")
        axis.legend()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path)
        if show:
            plt.show()
        return figure, axis
