"""TensorStore unit-test doubles backed by known NumPy or Zarr pixels."""

from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import tensorstore as ts
import zarr


def completed_operation(operation):
    """Execute an operation and return its value or error through a future.

    Parameters
    ----------
    operation : callable
        Read or write operation whose result must follow Future.result semantics.

    Returns
    -------
    concurrent.futures.Future
        Completed future retaining an operation's value or original exception.
    """
    future = Future()
    try:
        future.set_result(operation())
    except Exception as error:
        future.set_exception(error)
    return future


@pytest.fixture
def tensorstore_dataset():
    """Return a spec-constrained TensorStore mock factory for known pixel arrays.

    Returns
    -------
    callable
        Factory providing shape, dtype, domain labels, chunks, sliced reads and
        writes with completed futures. Writes update the backing pixels; reads
        return independent arrays. Processing algorithms are never replaced.
    """

    def create(data, *, chunks=None, labels=None):
        """Wrap known pixels with the TensorStore operations used by unit tests.

        Parameters
        ----------
        data : numpy.ndarray or zarr.Array
            Backing pixels; writes update this array.
        chunks : tuple of int or None
            Read chunk dimensions; omitted chunks span the array.
        labels : sequence of str or None
            Dimension labels; omitted dimensions are unlabeled.

        Returns
        -------
        unittest.mock.MagicMock
            TensorStore double constrained to the installed TensorStore interface.
        """
        store = MagicMock(spec_set=ts.TensorStore)
        store.shape = data.shape
        store.rank = data.ndim
        store.dtype = ts.dtype(data.dtype)
        store.domain = SimpleNamespace(labels=tuple(labels or ("",) * data.ndim))
        store.chunk_layout = SimpleNamespace(
            read_chunk=SimpleNamespace(shape=tuple(chunks or data.shape))
        )
        store.read.side_effect = lambda: completed_operation(lambda: np.array(data))
        store.write.side_effect = lambda value: completed_operation(
            lambda: data.__setitem__(Ellipsis, value)
        )

        def select(selection):
            """Return a readable selection whose writes update the parent pixels.

            Parameters
            ----------
            selection : tuple, slice or int
                NumPy-style dimensions selecting known pixels.

            Returns
            -------
            unittest.mock.MagicMock
                Selected TensorStore double sharing the parent write destination.
            """
            # NumPy silently clips slice bounds; TensorStore rejects oversized
            # requests. Keep that behavior so edge-chunk tests remain meaningful.
            indices = selection if isinstance(selection, tuple) else (selection,)
            axis = 0
            for index in indices:
                if index is Ellipsis:
                    axis += data.ndim - len(indices) + 1
                elif index is not None:
                    if isinstance(index, slice) and (
                        (index.start is not None and index.start > data.shape[axis])
                        or (index.stop is not None and index.stop > data.shape[axis])
                    ):
                        raise IndexError("Selection exceeds the TensorStore domain")
                    axis += 1
            selected = create(np.asarray(data[selection]))
            selected.read.side_effect = lambda: completed_operation(
                lambda: np.array(data[selection])
            )
            selected.write.side_effect = lambda value: completed_operation(
                lambda: data.__setitem__(selection, value)
            )
            return selected

        store.__getitem__.side_effect = select
        return store

    return create


@pytest.fixture(autouse=True)
def mock_tensorstore_datasets(request, monkeypatch, tensorstore_dataset):
    """Replace all TensorStore dataset entry points for isolated unit tests.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Test category determining whether storage is mocked.
    monkeypatch : pytest.MonkeyPatch
        Restore real TensorStore entry points after the test.
    tensorstore_dataset : callable
        Shared pixel-backed, spec-constrained dataset double factory.

    Returns
    -------
    None
        Unit tests use NumPy/Zarr-backed mocks. Integration tests retain real
        TensorStore reads and writes from acquisition files to saved outputs.
    """
    if request.node.get_closest_marker("integration"):
        return

    def open_dataset(spec):
        """Open a controlled file-backed Zarr specification without TensorStore.

        Parameters
        ----------
        spec : dict
            File kvstore specification, optionally including creation schema.

        Returns
        -------
        concurrent.futures.Future
            Dataset double or the original Zarr opening error.
        """

        def open_array():
            """Create or reopen the fixture's Zarr pixels and wrap their metadata."""
            path = spec["kvstore"]["path"]
            if spec.get("create"):
                schema = spec["schema"]
                layout = schema["chunk_layout"]
                chunk_shape = layout.get("chunk", layout.get("write_chunk"))["shape"]
                array = zarr.create_array(
                    path,
                    shape=tuple(schema["domain"]["shape"]),
                    dtype=schema["dtype"],
                    chunks=tuple(chunk_shape),
                    dimension_names=schema["domain"].get("labels"),
                    overwrite=spec["delete_existing"],
                )
            else:
                array = zarr.open_array(path, mode="r+")
            return tensorstore_dataset(
                array, chunks=array.chunks, labels=array.metadata.dimension_names
            )

        return completed_operation(open_array)

    def stack_datasets(arrays, *, axis):
        """Stack known unit-test pixels and preserve their dimension labels.

        Parameters
        ----------
        arrays : sequence
            TensorStore doubles describing the per-position images.
        axis : int
            New unlabeled position dimension.

        Returns
        -------
        unittest.mock.MagicMock
            TPCZYX dataset double with the known stacked pixel values.
        """
        labels = list(arrays[0].domain.labels)
        labels.insert(axis, "")
        return tensorstore_dataset(
            np.stack([array.read().result() for array in arrays], axis=axis),
            labels=labels,
        )

    monkeypatch.setattr(ts, "array", tensorstore_dataset)
    monkeypatch.setattr(ts, "open", open_dataset)
    monkeypatch.setattr(ts, "stack", stack_datasets)
