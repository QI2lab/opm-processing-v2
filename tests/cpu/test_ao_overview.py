"""Export saved AO pupils directly using the SJS publication presentation."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from scripts.ao_publication_style import write_publication_pupil_grid

pytestmark = pytest.mark.unit


@pytest.fixture
def pupil_reader():
    """Supply native saved coefficients and a deliberately forbidden old renderer."""

    class Result:
        def __init__(self, path, time, position):
            self.zarr_path = Path(path)
            self.time_index = time
            self.position_index = position
            self.starting_coefficients = np.linspace(-0.1234567, 0.2345678, 32)
            self.final_coefficients = self.starting_coefficients + 0.0123456
            self.pupil = np.ma.array(
                [[-np.pi, 0.1234567], [position + 0.2345678, 0]],
                mask=[[False, False], [False, True]],
            )
            self.calls = []

        def get_pupil(self, **kwargs):
            raise AssertionError("The legacy reader must only load saved coefficients")

    return SimpleNamespace(AOOptimizationResult=Result)


def test_publication_grid_uses_linear_phase_colors_and_reference_layout(
    tmp_path, pupil_reader, monkeypatch
):
    """Render every saved pupil unchanged with the reference's linear pi scale."""
    regions = [
        SimpleNamespace(
            index=index,
            pupil=pupil_reader.AOOptimizationResult(tmp_path, 0, index).pupil,
        )
        for index in range(9)
    ]
    originals = [region.pupil.copy() for region in regions]
    reference = tmp_path / "ao_grid_phase_plots.py"
    reference.write_text("# Publication formatting reference\n", encoding="utf-8")
    figures = []
    original_savefig = Figure.savefig

    def capture_figure(figure, *args, **kwargs):
        figures.append(figure)
        return original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture_figure)
    output = tmp_path / "publication"
    write_publication_pupil_grid(
        regions, output, timepoint=0, depth=0, reference_path=reference
    )
    assert len(list(output.glob("*.png"))) == 1
    assert len(list(output.glob("*.svg"))) == 1
    figure = figures[0]
    np.testing.assert_array_equal(figure.get_size_inches(), (12, 25))
    assert figure._suptitle is None
    pupil_axes = [axis for axis in figure.axes if axis.images]
    assert len(pupil_axes) == 9
    for index, axis in enumerate(pupil_axes):
        assert axis.get_subplotspec().get_gridspec().ncols == 2
        assert axis.get_title() == str(index)
        image = axis.images[0]
        assert image.origin == "lower"
        assert tuple(image.get_extent()) == (-1, 1, -1, 1)
        assert type(image.norm) is Normalize
        np.testing.assert_allclose(
            image.norm(np.asarray((-2, -1, 0, 1, 2)) * np.pi),
            (0, 0.25, 0.5, 0.75, 1),
        )
        np.testing.assert_array_equal(image.get_array(), originals[index])
        np.testing.assert_array_equal(regions[index].pupil, originals[index])
