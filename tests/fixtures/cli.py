"""Real CLI invocation shared by simulated acquisition integration scenarios."""

import importlib
import tomllib
from pathlib import Path

import pytest
from typer.main import get_command
from typer.testing import CliRunner


@pytest.fixture
def cli_runner():
    """Share a real Click terminal and registered-command runner."""
    return CliRunner()


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
def relocate_acquisition(tmp_path, integration_pipeline_records):
    """Move or copy actual acquisition artifacts within the test workspace.

    Returns
    -------
    callable
        Relocate a completed acquisition directory without rewriting provenance.
    """
    import shutil

    def relocate(root, *, copy=False):
        """Preserve original journals while relocating real stores and sidecars."""
        root = root.resolve()
        destination = root / "relocated"
        workspace = tmp_path.resolve()
        assert root.is_relative_to(workspace) and destination.is_relative_to(workspace)
        assert not destination.exists()
        artifacts = tuple(root.iterdir())
        targets = [(artifact, destination / artifact.name) for artifact in artifacts]
        assert all(
            source.resolve().parent == root and target.is_relative_to(destination)
            for source, target in targets
        )
        destination.mkdir()
        for source, target in targets:
            if copy:
                if source.is_dir():
                    shutil.copytree(source, target)
                else:
                    shutil.copy2(source, target)
            else:
                source.rename(target)
        integration_pipeline_records["relocations"].append((root, destination))
        return destination

    return relocate


@pytest.fixture
def invoke_cli(cli_runner):
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
        result = cli_runner.invoke(application, args, catch_exceptions=False)
        assert result.exit_code == 0, result.output
        return result

    return invoke
