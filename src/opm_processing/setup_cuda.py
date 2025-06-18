import os
import shlex
import stat
import subprocess
import shutil
import platform
from pathlib import Path
import typer

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
    "ml_dtypes",
    "typer",
    "cmap",
    "napari[all]",
    "zarr>=3.0.8",
    "psfmodels",
    "cupy-cuda12x",
    "fastrlock",
    "tifffile>=2025.6.1",
    "nvidia-cuda-runtime-cu12==12.8.*",
    "basicpy @ git+https://github.com/QI2lab/BaSiCPy.git@main",
    "ome-zarr @ git+https://github.com/ome/ome-zarr-py.git@refs/pull/404/head",
]

# CUDA conda pkgs
CONDA_CUDA_PKGS = [
    "'cuda-version=12.8'",
    "'cuda-toolkit=12.8'",
]

# Extra cucim Git URL for Windows
LINUX_CUCIM_GIT = (
    "cucim-cu12"
)

# Extra cucim Git URL for Windows
WINDOWS_CUCIM_GIT = (
    "git+https://github.com/rapidsai/cucim.git@v25.06.00#egg=cucim&subdirectory=python/cucim"
)

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
    run(f"{installer} install -y -c rapidsai -c conda-forge -c nvidia {' '.join(CONDA_CUDA_PKGS)}")

    # Clear existing hooks
    activate_dir = Path(prefix) / "etc" / "conda" / "activate.d"
    activate_dir.mkdir(parents=True, exist_ok=True)
    for script in activate_dir.glob("*"):
        try:
            script.unlink()
        except OSError:
            pass

    if is_windows:
        # Windows batch hook only
        bat_hook = activate_dir / "cuda_override.bat"
        bat_hook.write_text(f"""@echo off
REM Point at the conda-installed CUDA toolkit
set "CUDA_PATH={prefix}\\Library\\bin"
set "CUDA_HOME=%CUDA_PATH%"

REM Prepend CUDA bin and lib to PATH
set "PATH=%CUDA_PATH%;{prefix}\\Library\\lib;%PATH%"

REM NVRTC must compile with C++17 and ignore deprecated dialect
set "NVRTC_OPTIONS=--std=c++17"
set "CCCL_IGNORE_DEPRECATED_CPP_DIALECT=1"
""")
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

# NVRTC must compile with C++17 and ignore deprecated dialect
export NVRTC_OPTIONS="--std=c++17"
export CCCL_IGNORE_DEPRECATED_CPP_DIALECT="1"
""")
        sh_hook.chmod(sh_hook.stat().st_mode | stat.S_IEXEC)

    # Single pip install for all deps
    pip_deps = BASE_PIP_DEPS.copy()
    if is_windows:
        pip_deps.append(WINDOWS_CUCIM_GIT)
    else:
        pip_deps.append(LINUX_CUCIM_GIT)
    deps_str = " ".join(shlex.quote(d) for d in pip_deps)
    run(f"pip install --no-deps {deps_str}")
    
    run("python -m cupyx.tools.install_library --cuda 12.x --library cutensor")
    run("python -m cupyx.tools.install_library --cuda 12.x --library nccl")
    run("python -m cupyx.tools.install_library --cuda 12.x --library cudnn")

    typer.echo(f"\nsetup complete!  Please 'conda deactivate' then 'conda activate {env_lib}' to apply changes.")


def main():
    app()


if __name__ == "__main__":
    main()