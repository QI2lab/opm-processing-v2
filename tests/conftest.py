"""Enforce test categories and load fixtures grouped by processing responsibility."""

from pathlib import Path

import pytest

pytest_plugins = (
    "tests.fixtures.hardware",
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
    "tests.fixtures.pipeline",
    "tests.fixtures.cli",
)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Retain outcomes so contract checks do not mask actual failures or CUDA skips.

    Parameters
    ----------
    item : pytest.Item
        Current test.
    call : pytest.CallInfo
        Execution phase being reported.

    Yields
    ------
    None
        Control to pytest's normal report implementation.
    """
    outcome = yield
    report = outcome.get_result()
    setattr(item, f"pipeline_report_{report.when}", report)


def pytest_collection_modifyitems(items):
    """Require unit or integration tests and consistent execution hardware.

    Parameters
    ----------
    items : list of pytest.Item
        Collected tests to validate.

    Raises
    ------
    pytest.UsageError
        If a test lacks an allowed classification, uses a forbidden category, or
        requires CUDA from the CPU suite.
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
                f"{item.nodeid}: only unit and integration tests are permitted"
            )
        categories = [
            name for name in ("unit", "integration") if item.get_closest_marker(name)
        ]
        if len(categories) != 1:
            raise pytest.UsageError(
                f"{item.nodeid}: select exactly one of unit or integration"
            )
        if categories == ["integration"] and not {
            "simulated_camera",
            "pipeline_optical_data",
            "independent_photon_scene_factory",
        }.intersection(item.fixturenames):
            raise pytest.UsageError(
                f"{item.nodeid}: integration tests require shared forward-model ground truth"
            )
        if (
            categories == ["integration"]
            and "opm_v2_acquisition_factory" not in item.fixturenames
        ):
            raise pytest.UsageError(
                f"{item.nodeid}: every integration must start with OpmDataHandler output"
            )
        if categories == ["integration"] and "acquisition_factory" in item.fixturenames:
            raise pytest.UsageError(
                f"{item.nodeid}: synthetic acquisition layouts are forbidden in integration tests"
            )
