import os
import shlex
import stat
import subprocess
import shutil
import platform
from pathlib import Path
import typer
import sys

app = typer.Typer()
app.pretty_exceptions_enable = False

# Base pip deps (pure-Python)
BASE_PIP_DEPS = [
    "numpy",
    "numba",
    "llvmlite",
    "tbb",
    "tqdm",
    "ryomen",
    "tensorstore",
    "cmap",
    "napari[all]",
    "zarr>=3.0.8",
    "psfmodels",
    "tifffile>=2025.6.1",
    "nvidia-cuda-runtime-cu12==12.8.*",
    "napari-ome-zarr",
    "simpleitk",
    "basicpy @ git+https://github.com/QI2lab/BaSiCPy.git@main",
    "ome-zarr"
]

# CUDA conda pkgs
LINUX_CONDA_CUDA_PKGS = [
    "'cuda-version=12.8'",
    "'cuda-toolkit=12.8'",
    "cuda-cudart",
    "cupy",
    "scikit-image",
    "'cucim=25.06'",
    "cudnn",
    "cutensor",
    "nccl"
]

WINDOWS_CONDA_CUDA_PKGS = [
    "cuda-version=12.8",
    "cuda-toolkit=12.8",
    "cuda-cudart",
    "cudnn",
    "cutensor",
    "cuda-nvcc",
]

LINUX_JAX_LIB = {
    "jax[cuda12_local]==0.4.38"
}

WINDOWS_OTHER_PIP_DEPS = {
    "cupy-cuda12x",
    "scikit-image",
}

# WINDOWS_CUCIM_GIT = (
#     "git+https://github.com/rapidsai/cucim.git@v25.06.00#egg=cucim-cu12&subdirectory=python/cucim"
# )

def run(command: str):
    typer.echo(f"$ {command}")
    subprocess.run(command, shell=True, check=True)

@app.command()
def setup_cuda():
    """
    1) Installs CUDA packages via conda (RAPIDS.ai channel), using OS-specific lists.
    2) Writes a single activation hook (either .sh on Linux or .bat on Windows).
    3) Installs all pip deps (on Windows including the extra cucim Git URL) in one call.
    """
    prefix = os.environ.get("CONDA_PREFIX")
    if not prefix:
        typer.echo("Error: activate your conda environment first.", err=True)
        raise typer.Exit(1)

    installer = shutil.which("mamba") or shutil.which("conda")
    if not installer:
        typer.echo("Error: neither mamba nor conda found.", err=True)
        raise typer.Exit(1)

    is_windows = platform.system() == "Windows"
    if is_windows:
        run(f"{installer} install -y -c rapidsai -c conda-forge -c nvidia {' '.join(WINDOWS_CONDA_CUDA_PKGS)}")
    else:
        run(f"{installer} install -y -c rapidsai -c conda-forge -c nvidia {' '.join(LINUX_CONDA_CUDA_PKGS)}")

    # Clear existing hooks
    activate_dir = Path(prefix) / "etc" / "conda" / "activate.d"
    activate_dir.mkdir(parents=True, exist_ok=True)
    for script in activate_dir.glob("*"):
        try:
            script.unlink()
        except OSError:
            pass


    if is_windows:

        # path to conda activation hooks
        activate_dir = os.path.join(prefix, "etc", "conda", "activate.d")
        os.makedirs(activate_dir, exist_ok=True)

        # .bat file that will run on env activation
        bat_path = os.path.join(activate_dir, "set_cuda_path.bat")
        with open(bat_path, "w") as f:
            f.write("@echo off\n")
            f.write("rem — set CUDA_PATH so CuPy finds the DLLs in %CONDA_PREFIX%\\Library\n")
            f.write('set "CUDA_PATH=%CONDA_PREFIX%\\Library"\n')

        system_where = Path(os.environ["WINDIR"]) / "System32" / "where.exe"
        dest_which = Path(prefix) / "Scripts" / "which.exe"

        # only copy if it doesn't already exist
        if not dest_which.exists():
            shutil.copy(system_where, dest_which)
    else:

        # Linux shell hook only
        sh_hook = activate_dir / "cuda_override.sh"
        env_lib = f"{prefix}/lib"
        linux_cuda_root = f"{prefix}/targets/x86_64-linux"
        sh_hook.write_text(f"""#!/usr/bin/env sh
# Point at the conda-installed CUDA toolkit
export CUDA_PATH="{linux_cuda_root}"
export CUDA_HOME="$CUDA_PATH"
export PATH="$CUDA_PATH/bin:$PATH"

# Prepend only the conda toolkit lib & env lib
export LD_LIBRARY_PATH="$CUDA_PATH/lib:{env_lib}${{LD_LIBRARY_PATH:+:${{LD_LIBRARY_PATH}}}}"
""")
        sh_hook.chmod(sh_hook.stat().st_mode | stat.S_IEXEC)

    # Single pip install for all deps
    if is_windows:
        
        subprocess.run(
            [sys.executable, "-m", "pip", "install", *BASE_PIP_DEPS],
            check=True
        )

        subprocess.run(
            [sys.executable, "-m", "pip", "install", *WINDOWS_OTHER_PIP_DEPS],
            check=True
        )
        
    else:
        pip_deps = BASE_PIP_DEPS.copy()
        deps_str = " ".join(shlex.quote(d) for d in pip_deps)
        run(f"pip install {deps_str}")
        
        linux_dep_str = " ".join(shlex.quote(d) for d in LINUX_JAX_LIB)
        run(f"pip install {linux_dep_str}")
    if is_windows:
        typer.echo(f"\nsetup complete!  Please 'conda deactivate' then 'conda activate {prefix}' to apply changes.")
    else:
        typer.echo(f"\nsetup complete!  Please 'conda deactivate' then 'conda activate {env_lib}' to apply changes.")


def main():
    app()


if __name__ == "__main__":
    main()