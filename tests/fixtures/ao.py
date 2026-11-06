"""Known AO pupils and native, package-backed tissue geometry for unit tests."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest


@dataclass
class SavedAOCase:
    """Actual opm-v2 saved optimization plus its prescribed modal truth."""

    result: Any
    starting: np.ndarray
    final: np.ndarray
    reader_path: Path


@pytest.fixture
def ao_result_factory(tmp_path):
    """Persist AO-format units through the actual opm-v2 optimization writer."""
    import zarr
    from opm_v2.utils.sensorless_ao import save_optimization_results
    from scripts import ao_saved_results

    counter = iter(range(10000))

    def create(
        *,
        path=None,
        starting=None,
        final=None,
        time=0,
        index=0,
        history=None,
        iterations=1,
        modes=(0,),
        zarr_format=3,
    ):
        starting = np.asarray(
            np.zeros(32) if starting is None else starting, dtype=float
        )
        final = np.asarray(starting if final is None else final, dtype=float)
        if final.shape != starting.shape:
            raise ValueError("AO fixture modal vectors must share a basis")
        history = np.asarray(final[None] if history is None else history, dtype=float)
        directory = path or tmp_path / f"saved_ao_{next(counter)}"
        directory.mkdir(parents=True, exist_ok=True)
        image = np.arange(20, dtype=np.uint16).reshape(4, 5)
        with zarr.config.set({"default_zarr_format": zarr_format}):
            save_optimization_results(
                all_images=np.repeat(image[None], 1 + 2 * len(history), axis=0),
                all_metrics=1 + np.arange(1 + 2 * len(history), dtype=float),
                optimal_images=np.repeat(image[None], len(history), axis=0),
                optimal_metrics=3 + 2 * np.arange(len(history), dtype=float),
                optimal_coeffs=history,
                starting_coeffs=starting,
                starting_metric=1,
                starting_image=image,
                update_status=np.ones(len(history), dtype=bool),
                metadata={
                    "num_iterations": iterations,
                    "num_mode_samples": 1,
                    "modes_to_optimize": list(modes),
                },
                save_dir_path=directory,
            )
        result = ao_saved_results.AOOptimizationResult(
            directory / "ao_results.zarr", time, index
        )
        return SavedAOCase(
            result, starting.copy(), final.copy(), Path(ao_saved_results.__file__)
        )

    return create
