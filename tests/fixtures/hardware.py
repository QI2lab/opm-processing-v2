"""CPU thread limits and production reconstruction observers."""

from __future__ import annotations

import pytest


@pytest.fixture
def small_cpu_kernels():
    """Bound worker overhead for small parallel numerical unit-test arrays."""
    from numba import get_num_threads, set_num_threads

    previous = get_num_threads()
    set_num_threads(min(2, previous))
    yield
    set_num_threads(previous)


@pytest.fixture
def production_forward_predictions(monkeypatch, request):
    """Observe accepted acquired-camera predictions without rebuilding the optics.

    Returns
    -------
    list of dict
        One record per public RLGC call, containing actual acquired-grid
        predictions and the production-selected iteration count. The baseline
        is the same-grid input prediction or the production initializer.
        CPU tests receive an empty list without importing the CUDA solver.
    """
    calls = []
    if request.node.get_closest_marker("gpu") is None:
        return calls

    import importlib
    import io
    import logging
    import re
    from collections import deque

    import cupy as cp
    import numpy as np

    solver = importlib.import_module("opm_processing.imageprocessing.rlgc")
    reconstruct, iterate = solver.rlgc, solver._iterate_rlgc
    active = []

    def observe_iterations(observed, shape, forward, *args, **kwargs):
        """Retain owned samples from calls to the unchanged production forward."""
        if not active:
            return iterate(observed, shape, forward, *args, **kwargs)
        run = active[-1]
        attempts = re.findall(r"attempt_rlgc_crop_scan=(\d+)", run["trace"].getvalue())
        attempt = attempts[-1] if attempts else None
        if attempt != run.get("attempt"):
            run["chunks"].clear()
            run["attempt"] = attempt
        prefix = kwargs["log_prefix"]
        region = args[2]
        factor = run["factor"]
        layouts = re.findall(
            r"^(.*)raw_scan=\[(\d+):(\d+)\] retained_fine_scan=\[(\d+):(\d+)\]",
            run["trace"].getvalue(),
            re.MULTILINE,
        )
        layout = next((row for row in reversed(layouts) if row[0] == prefix), None)
        if layout is None:
            destination = owned = slice(None)
        else:
            _, source_start, _, start, stop = layout
            start, stop = int(start) // factor, (int(stop) - 1) // factor + 1
            destination = slice(start, stop)
            owned = slice(start - int(source_start), stop - int(source_start))
        entry = run["chunks"].setdefault(
            prefix,
            {
                "destination": destination,
                "history": deque(maxlen=2),
                "visits": 0,
                "observed_shape": observed.shape,
                "object_shape": shape,
                "updates": [],
                "shared": kwargs.get("_checkpoint") is not None,
            },
        )
        entry["visits"] += 1
        if observed.shape == shape and "baseline_prediction" not in entry:
            values = forward(observed)[region][owned]
            entry["baseline_prediction"] = (
                cp.asnumpy(values) if isinstance(values, cp.ndarray) else values.copy()
            )
            del values
        state = kwargs.get("_checkpoint")
        completed = 0 if state is None else state.iterations
        callback = kwargs["callback"]

        def observe_forward(estimate):
            """Call the actual operator and keep only its two latest predictions."""
            predicted = forward(estimate)
            if state is None and "baseline_prediction" in entry:
                return predicted
            values = predicted[region][owned]
            captured = (
                cp.asnumpy(values) if isinstance(values, cp.ndarray) else values.copy()
            )
            if state is not None:
                entry["history"].append((completed, captured))
            if completed == 0 and "baseline_prediction" not in entry:
                entry["baseline_prediction"] = captured
            return predicted

        def observe_update(number, estimate):
            """Observe final full-state predictions before production crops them."""
            nonlocal completed
            completed = number
            entry["updates"].append(number)
            if state is not None and (
                (state.first_kl_stop is not None and not kwargs.get("_finalize"))
                or number == kwargs["max_iterations"]
            ):
                values = forward(estimate)[region][owned]
                entry["tail"] = (
                    number,
                    cp.asnumpy(values)
                    if isinstance(values, cp.ndarray)
                    else values.copy(),
                )
            if callback is not None:
                callback(number, estimate)

        result = iterate(
            observed,
            shape,
            observe_forward,
            *args,
            **dict(kwargs, callback=observe_update),
        )
        if state is None:
            values = forward(result)[region][owned]
            entry["final"] = (
                cp.asnumpy(values) if isinstance(values, cp.ndarray) else values.copy()
            )
        return result

    def observe_reconstruction(image, *args, **kwargs):
        """Assemble actual owned predictions at the production-reported budget."""
        if not kwargs.get("gradient_consensus", True):
            return reconstruct(image, *args, **kwargs)
        trace = io.StringIO()
        logger = kwargs.get("logger") or logging.getLogger(__name__)
        level = logger.level
        handler = logging.StreamHandler(trace)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        run = {
            "trace": trace,
            "chunks": {},
            "factor": kwargs.get("scan_upsample_factor", 1),
        }
        active.append(run)
        try:
            result = reconstruct(image, *args, **dict(kwargs, logger=logger))
        finally:
            active.pop()
            logger.removeHandler(handler)
            logger.setLevel(level)
        stops = re.findall(r"stop=\S+ iterations=(\d+)", trace.getvalue())
        accepted = int(stops[-1]) if stops else 0
        record = {
            name: np.zeros(np.shape(image), dtype=np.float32)
            for name in ("prediction", "baseline_prediction")
        }
        record["iterations"] = accepted
        record["chunks"] = [
            {
                name: entry[name]
                for name in (
                    "destination",
                    "visits",
                    "observed_shape",
                    "object_shape",
                    "updates",
                    "shared",
                )
            }
            for entry in run["chunks"].values()
        ]
        for entry in run["chunks"].values():
            final = entry.get("final")
            if final is None:
                final = dict((*entry["history"], entry.get("tail", (-1, None))))[
                    accepted
                ]
            for name in ("prediction", "baseline_prediction"):
                target = record[name] if np.ndim(image) == 3 else record[name][None]
                target[entry["destination"]] = (
                    final if name == "prediction" else entry[name]
                )
        calls.append(record)
        return result

    monkeypatch.setattr(solver, "_iterate_rlgc", observe_iterations)
    monkeypatch.setattr(solver, "rlgc", observe_reconstruction)
    return calls
