"""Real CLI invocation shared by simulated acquisition integration scenarios."""

import importlib
import tomllib
from pathlib import Path

import pytest
from typer.main import get_command
from typer.testing import CliRunner


@pytest.fixture
def cli_file_layout(request, fs):
    """Create mocked discovery inputs without replacing any package APIs.

    Returns
    -------
    pathlib.Path
        Acquisition directory containing the requested stores or frame names.
    """
    root = Path.cwd() / "cli-files"
    fs.create_dir(root)
    for name in request.param:
        path = root / name
        if name.endswith(".zarr"):
            fs.create_dir(path)
        else:
            fs.create_file(path)
    return root


@pytest.fixture
def encoder_frames(tmp_path):
    """Write calibrated display pixels for encoder input-validation units.

    Returns
    -------
    list of pathlib.Path
        One TIFF with a known acquired volume interval.
    """
    import numpy as np
    import tifffile

    path = tmp_path / "projection_t0.tiff"
    tifffile.imwrite(
        path, np.zeros((8, 10), np.uint8), metadata={"volume_interval_ms": 500}
    )
    return [path]


@pytest.fixture
def export_image_input(request, acquisition_factory, fused_store_factory):
    """Persist small real OME-Zarr inputs for individual TIFF writer units.

    Returns
    -------
    types.SimpleNamespace
        Source path, independently known pixels, and expected image-group path.
    """
    from types import SimpleNamespace

    import numpy as np

    kind = request.param
    pixels = np.arange(3 * 4 * 5, dtype=np.uint16).reshape(1, 1, 3, 4, 5)
    if kind == "image":
        path = fused_store_factory(pixels, name="sample_fused")
        image = path
    else:
        positions = 2 if kind == "multiple" else 1
        dataset = acquisition_factory(
            np.repeat(pixels[:, None], positions, axis=1), name="sample_deskewed"
        )
        path, image = dataset.path, dataset.path / "0"
    return SimpleNamespace(path=path, image=image, pixels=pixels, kind=kind)


@pytest.fixture
def acquisition_link(request, export_image_input, tmp_path):
    """Persist source-link variants using the production processing journal.

    Returns
    -------
    types.SimpleNamespace
        Processed path, source store and optional explicit acquisition override.
    """
    from types import SimpleNamespace

    from opm_processing.dataio.processing_state import ProcessingState

    kind = request.param
    source = export_image_input.path
    processed = tmp_path / "linked_deskewed.ome.zarr"
    if kind not in ("missing", "override"):
        count = 2 if kind == "ambiguous" else 1
        for index in range(count):
            state = ProcessingState.create(
                tmp_path / f"linked_{index}.processing.json", source
            )
            state.initialize_run(processed, configuration={}, overwrite=True)
            if kind == "relative":
                state.document["source"]["path"] = source.name
                state.save()
    return SimpleNamespace(processed=processed, source=source, kind=kind)


@pytest.fixture
def cli_export_options(request):
    """Pair display, scheduling, time-selection and annotation options.

    Returns
    -------
    dict
        Real projection CLI values; selection retains acquisition timestamps.
    """
    selected = request.param == "selected-color"
    return {
        "depth_color": selected,
        "depth_colormap": "viridis" if selected else "turbo",
        "workers": 2 if selected else 1,
        "timepoints": (1, 2) if selected else None,
        "scale_bar_um": 1.0 if selected else None,
    }


@pytest.fixture
def invoke_cli():
    """Invoke a registered CLI with its real parser and processing implementation.

    Returns
    -------
    callable
        Accept a command name, positional arguments and named option values.
        Tuple options retain their arity and list options repeat their flag.
        Successful execution returns the captured Click result; failures retain
        their original traceback.
    """
    registry = tomllib.loads(
        (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text("utf-8")
    )["project"]["scripts"]

    def invoke(name, *arguments, **options):
        """Parse actual CLI arguments and execute every numerical/storage boundary."""
        application = importlib.import_module(registry[name].split(":")[0]).app
        parameters = {p.name: p for p in get_command(application).params}
        args = [str(value) for value in arguments]
        for key, value in options.items():
            if value is None:
                continue
            parameter = parameters[key]
            if isinstance(value, bool):
                flags = parameter.opts if value else parameter.secondary_opts
                if flags:
                    args.append(flags[0])
                elif value != parameter.default:
                    raise ValueError(f"{name}: {key} has no flag for {value}")
            elif isinstance(value, list):
                for item in value:
                    args.extend((parameter.opts[0], str(item)))
            else:
                values = value if isinstance(value, tuple) else (value,)
                args.extend((parameter.opts[0], *(str(v) for v in values)))
        result = CliRunner().invoke(application, args, catch_exceptions=False)
        assert result.exit_code == 0, result.output
        return result

    return invoke
