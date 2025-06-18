import os
import shlex
import stat
import subprocess
import shutil
from pathlib import Path
import typer

app = typer.Typer()
app.pretty_exceptions_enable = False

# Packages to install via pip (pure-Python dependencies)
PIP_DEPS = [
    "numpy",
    "numba",
    "llvmlite",
    "tbb",
    "tqdm",
    "ryomen",
    "tensorstore",
    "ml_dtypes",
    "typer",
    "cmap",
    "napari[all]",
    "zarr>=3.0.8",
    "psfmodels",
    "tifffile>=2025.6.1",
    "basicpy @ git+https://github.com/QI2lab/BaSiCPy.git@main",
    "ome-zarr @ git+https://github.com/ome/ome-zarr-py.git@refs/pull/404/head",
]

# CUDA and related packages to install via conda (RAPIDS.ai recommendation)
CONDA_CUDA_PKGS = [
    "cucim=25.06",
    "'cuda-version=12.8'",
    "'cuda-toolkit=12.8'",
    "cupy",
    "'cudnn=8.8'",
    "cutensor",
    "nccl"
]


def run(command: str):
    """
    Run a shell command, echoing it and aborting on error.
    """
    typer.echo(f"$ {command}")
    subprocess.run(command, shell=True, check=True)


@app.command()
def setup_cuda():
    """
    1) Installs CUDA 12.8 and matching GPU packages via conda (RAPIDS.ai channel).
    2) Writes activation hook to override CUDA_ROOT and NVRTC flags.
    3) Installs pure-Python dependencies via pip (no-deps).
    """
    # Ensure Conda env is active
    prefix = os.environ.get("CONDA_PREFIX")
    if not prefix:
        typer.echo("Error: activate your conda environment first.", err=True)
        raise typer.Exit(1)

    # Choose installer
    installer = shutil.which("mamba") or shutil.which("conda")
    if not installer:
        typer.echo("Error: neither mamba nor conda found.", err=True)
        raise typer.Exit(1)

    # 1) Install CUDA packages via RAPIDS.ai recommended channels
    pkgs = " ".join(CONDA_CUDA_PKGS)
    run(f"{installer} install -y -c rapidsai -c conda-forge -c nvidia {pkgs}")
    
    # 2) Clear existing hooks and write activation hook
    activate_dir = Path(prefix) / "etc" / "conda" / "activate.d"
    activate_dir.mkdir(parents=True, exist_ok=True)
    for script in activate_dir.glob("*.sh"):
        try:
            script.unlink()
        except OSError:
            pass

    # 3) Write the new activation hook
    hook_file = activate_dir / "cuda_override.sh"
    cuda_root = f"{prefix}/targets/x86_64-linux"
    env_lib = f"{prefix}/lib"
    hook_contents = f"""#!/usr/bin/env sh
# 1) Point at the conda-installed CUDA toolkit
export CUDA_PATH="{cuda_root}"
export CUDA_HOME="$CUDA_PATH"
export PATH="$CUDA_PATH/bin:$PATH"

# 2) Prepend only the conda toolkit lib & env lib
export LD_LIBRARY_PATH="$CUDA_PATH/lib:{env_lib}${{LD_LIBRARY_PATH:+:${{LD_LIBRARY_PATH}}}}"

# 3) NVRTC must compile with C++17 and ignore deprecated dialect
export NVRTC_OPTIONS="--std=c++17"
export CCCL_IGNORE_DEPRECATED_CPP_DIALECT="1"
"""
    hook_file.write_text(hook_contents)
    hook_file.chmod(hook_file.stat().st_mode | stat.S_IEXEC)

    # 4) Pip install pure-Python deps without pulling CUDA wheels
    deps = " ".join(shlex.quote(d) for d in PIP_DEPS)
    run(f"pip install --no-deps {deps}")

    typer.echo("\nsetup complete!  Please 'conda deactivate' then 'conda activate {env_lib}' to apply changes.")


def main():
    app()


if __name__ == "__main__":
    main()
