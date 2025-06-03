"""
Reader + Napari plugin for OME-NGFF v0.5 (Zarr V3) datasets using TensorStore.
Supports coordinate transforms, lazy multiscale images, without Dask.
"""
import json
from pathlib import Path

import tensorstore as ts
from napari_plugin_engine import napari_hook_implementation

class OmeZarrReader:
    """
    Reader for OME-NGFF v0.5 stores (Zarr V3) using TensorStore.
    Provides multiscale metadata, coordinate transforms, and lazy access.
    """

    def __init__(self, root_path):
        """
        Initialize the reader with the root directory of the Zarr V3 store.

        Parameters
        ----------
        root_path : str or pathlib.Path
            Path to the root of the OME-NGFF Zarr V3 store (contains "zarr.json").
        """
        self.root = Path(root_path)
        self._load_metadata()

    def _load_metadata(self):
        """
        Load and parse the root group metadata from zarr.json.

        Raises
        ------
        FileNotFoundError
            If the zarr.json file is not found in the root directory.
        ValueError
            If the OME metadata is missing from zarr.json.
        """
        meta_file = self.root / "zarr.json"
        if not meta_file.exists():
            raise FileNotFoundError(f"Zarr V3 metadata not found at {meta_file}")

        with open(meta_file, 'r') as f:
            data = json.load(f)

        attrs = data.get('attributes', {})
        ome_meta = attrs.get('ome')
        if ome_meta is None:
            raise ValueError(f"No OME metadata found in {meta_file}")

        self.multiscales = ome_meta['multiscales']

    def get_axes(self, scale_index=0):
        """
        Return axis definitions for a given multiscale level.

        Parameters
        ----------
        scale_index : int, optional
            Index of the multiscale level (0 by default).

        Returns
        -------
        list of dict
            List of axis definitions with keys 'name' and 'type'.
        """
        return self.multiscales[scale_index]['axes']

    def list_datasets(self, scale_index=0):
        """
        List dataset array paths for a given multiscale level.

        Parameters
        ----------
        scale_index : int, optional
            Index of the multiscale level (0 by default).

        Returns
        -------
        list of str
            List of dataset paths within the store.
        """
        datasets = self.multiscales[scale_index]['datasets']
        return [d['path'] for d in datasets]

    def get_coordinate_transformations(
        self, scale_index=0, dataset_path=None
    ):
        """
        Fetch coordinate transformation entries for a dataset.

        Parameters
        ----------
        scale_index : int, optional
            Index of the multiscale level.
        dataset_path : str or None, optional
            Path of the dataset. If None and only one dataset exists,
            it will be selected automatically.

        Returns
        -------
        list of dict
            Coordinate transformation objects with keys 'type',
            and either 'scale' or 'translation'.

        Raises
        ------
        ValueError
            If multiple datasets exist and dataset_path is not provided,
            or if the specified dataset_path is not found.
        """
        ds_list = self.multiscales[scale_index]['datasets']

        if dataset_path is None:
            if len(ds_list) != 1:
                paths = [d['path'] for d in ds_list]
                raise ValueError(
                    f"Multiple datasets at level {scale_index}; "
                    f"specify one of: {paths}"
                )
            ds = ds_list[0]
        else:
            ds = next((d for d in ds_list if d['path'] == dataset_path), None)
            if ds is None:
                raise ValueError(
                    f"Dataset '{dataset_path}' not found at level {scale_index}"
                )

        return ds.get('coordinateTransformations', [])

    def index_to_physical(
        self, idx, scale_index=0, dataset_path=None
    ):
        """
        Convert array index coordinates to physical space using transforms.

        Parameters
        ----------
        idx : sequence of int
            Index coordinates matching the dataset dimensionality.
        scale_index : int, optional
            Index of the multiscale level.
        dataset_path : str or None, optional
            Dataset path within the level.

        Returns
        -------
        list of float
            Physical coordinates after applying scale and translation.
        """
        coords = list(idx)
        transforms = self.get_coordinate_transformations(
            scale_index, dataset_path
        )

        for t in transforms:
            ttype = t.get('type')
            if ttype == 'scale':
                coords = [c * s for c, s in zip(coords, t['scale'])]
            elif ttype == 'translation':
                coords = [c + o for c, o in zip(coords, t['translation'])]

        return coords

    def open_scale(
        self, scale_index=0, dataset_path=None, **open_kwargs
    ):
        """
        Return a lazy TensorStore array for a specified level.

        Parameters
        ----------
        scale_index : int, optional
            Index of the multiscale level.
        dataset_path : str or None, optional
            Dataset path within the level.
        **open_kwargs : dict
            Additional arguments to pass to ``ts.open``.

        Returns
        -------
        tensorstore.TensorStore
            A lazy array-like object. Data is not read until ``.read()``
            or slicing is executed.

        Raises
        ------
        ValueError
            If multiple datasets exist and dataset_path is not provided,
            or if the specified dataset_path is not available.
        """
        ds_list = self.multiscales[scale_index]['datasets']
        paths = [d['path'] for d in ds_list]

        if dataset_path is None:
            if len(paths) != 1:
                raise ValueError(
                    f"Multiple datasets at level {scale_index}; "
                    f"specify one of: {paths}"
                )
            dataset_path = paths[0]
        elif dataset_path not in paths:
            raise ValueError(
                f"Dataset '{dataset_path}' not available; "
                f"available: {paths}"
            )

        spec = {
            'driver': 'zarr3',
            'kvstore': {'driver': 'file', 'path': str(self.root)},
            'metadata': {'path': dataset_path}
        }
        spec.update(open_kwargs)
        return ts.open(spec).result()

    def get_lazy(self, scale_index=0, dataset_path=None):
        """
        Alias for ``open_scale`` to obtain lazy access.

        See Also
        --------
        open_scale : Return a lazy TensorStore array.
        """
        return self.open_scale(scale_index, dataset_path)

    def read_region(
        self, scale_index=0, dataset_path=None, index_slices=None
    ):
        """
        Read a sub-region defined by ``index_slices`` into memory.

        Parameters
        ----------
        scale_index : int, optional
            Index of the multiscale level.
        dataset_path : str or None, optional
            Dataset path within the level.
        index_slices : tuple of slices or int
            A tuple specifying the region to read in array coordinates.

        Returns
        -------
        numpy.ndarray
            The requested sub-region as an array.

        Raises
        ------
        ValueError
            If ``index_slices`` is not provided.
        """
        if index_slices is None:
            raise ValueError("index_slices must be provided")

        store = self.open_scale(scale_index, dataset_path)
        return store[index_slices].read().result()


@napari_hook_implementation
def napari_get_reader(path):
    """
    Napari hook to load OME-NGFF v0.5 (Zarr V3) stores as multiscale images.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the dataset directory (must contain "zarr.json").

    Returns
    -------
    callable or None
        A function that accepts the dataset path and returns a list of
        layer data tuples, or None if the path is not an OME-NGFF v0.5 store.
    """
    root = Path(path)
    if not (root.is_dir() and (root / "zarr.json").exists()):
        return None

    def reader(path_to_store):
        """
        Internal reader for Napari.

        Parameters
        ----------
        path_to_store : str or pathlib.Path
            Path to the Zarr V3 store directory.

        Returns
        -------
        list of tuple
            Each tuple contains (levels, layer_args, layer_type).
        """
        reader = OmeZarrReader(path_to_store)
        levels = [reader.get_lazy(i) for i in range(len(reader.multiscales))]
        name = Path(path_to_store).stem
        scales = []
        for m in reader.multiscales:
            # assume first dataset has scale transform
            transforms = m['datasets'][0].get('coordinateTransformations', [])
            scale = next(
                (t['scale'] for t in transforms if t.get('type') == 'scale'),
                None
            )
            scales.append(scale)

        layer_args = {
            'name': name,
            'multiscale': True,
            'scale': scales
        }
        return [(levels, layer_args, 'image')]

    return reader