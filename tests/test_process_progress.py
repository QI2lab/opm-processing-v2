"""Progress counters follow checkpointed tiles and acquisition order."""

import importlib

import pytest


@pytest.mark.unit
@pytest.mark.parametrize("outer_axis", ["t", "p"])
def test_nested_progress_counts_completed_tiles(monkeypatch, outer_axis):
    """Count resumed acquisition groups and close both nested progress bars."""
    processing = importlib.import_module("opm_processing.process")
    bars = []

    class Progress:
        def __init__(self, **options):
            self.options = options
            self.n = options["initial"]
            self.closed = False
            bars.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

        def update(self, count):
            self.n += count

    monkeypatch.setattr(processing, "tqdm", Progress)
    tiles = [(0, 0), (0, 1), (1, 0), (1, 1)]
    if outer_axis == "p":
        tiles.sort(key=lambda tile: (tile[1], tile[0]))
    iterator = processing.progress_tile_groups(
        iter(tiles[1:]), tiles, {tiles[0]}, (outer_axis,)
    )
    next(iterator)
    assert [bar.options["desc"] for bar in bars] == (
        ["time", "positions"] if outer_axis == "t" else ["positions", "time"]
    )
    assert bars[0].n == 0
    assert bars[1].options["total"] == 2
    assert bars[1].n == 1
    # The yielded tile is still in progress until the caller resumes iteration.
    next(iterator)
    assert bars[0].n == 1
    assert bars[1].n == 2
    assert bars[1].closed
    assert bars[2].n == 0
    list(iterator)
    assert bars[0].n == 2
    assert bars[2].n == 2
    assert all(bar.closed for bar in bars)
