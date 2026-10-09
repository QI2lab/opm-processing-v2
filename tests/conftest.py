"""Enforce test categories and load fixtures grouped by processing responsibility."""

from pathlib import Path

import pytest

pytest_plugins = (
    "tests.fixtures.hardware",
    "tests.fixtures.tensorstore",
    "tests.fixtures.acquisition",
    "tests.fixtures.tiling",
    "tests.fixtures.roi",
    "tests.fixtures.live",
    "tests.fixtures.optics",
    "tests.fixtures.fusion",
    "tests.fixtures.storage",
    "tests.fixtures.export",
    "tests.fixtures.state",
    "tests.fixtures.camera",
    "tests.fixtures.illumination",
)


def pytest_collection_modifyitems(items):
    """Require unit or disk-based integration tests during collection.

    Parameters
    ----------
    items : list of pytest.Item
        Collected correctness tests to validate.

    Raises
    ------
    pytest.UsageError
        If categories or hardware placement are inconsistent, a forbidden
        category is used, or an integration test has no temporary disk storage.
    """
    for item in items:
        execution = item.path.relative_to(Path(__file__).parent).parts[0]
        if execution not in ("cpu", "gpu"):
            raise pytest.UsageError(
                f"{item.nodeid}: tests must be in tests/cpu or tests/gpu"
            )
        opposite = "gpu" if execution == "cpu" else "cpu"
        if item.get_closest_marker(opposite):
            raise pytest.UsageError(
                f"{item.nodeid}: hardware marker contradicts its directory"
            )
        item.add_marker(getattr(pytest.mark, execution))
        if execution == "cpu" and "cupy_gpu" in item.fixturenames:
            raise pytest.UsageError(f"{item.nodeid}: CUDA fixtures belong in tests/gpu")
        if any(
            item.get_closest_marker(name)
            for name in ("smoke", "api", "regression", "benchmark")
        ):
            raise pytest.UsageError(
                f"{item.nodeid}: smoke, API, regression, and benchmark tests are not permitted"
            )
        categories = [
            name for name in ("unit", "integration") if item.get_closest_marker(name)
        ]
        if len(categories) != 1:
            raise pytest.UsageError(
                f"{item.nodeid} must have exactly one of unit or integration"
            )
        if categories == ["integration"] and not {
            "tmp_path",
            "tmp_path_factory",
        }.intersection(item.fixturenames):
            raise pytest.UsageError(
                f"{item.nodeid}: integration tests require simulated input and "
                "verified output on disk"
            )
