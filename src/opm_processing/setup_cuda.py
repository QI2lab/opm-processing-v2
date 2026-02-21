import os
import platform
import shlex
import shutil
import stat
import subprocess
import sys
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
    "cmap",
    "napari[all]",
    "zarr>=3.0.8",
    "psfmodels",
    "tifffile>=2025.6.1",
    "numcodecs",
    "nvidia-cuda-runtime-cu12==12.8.*",
    "cuda-bindings==12.8.*",
    "napari-ome-zarr",
    "simpleitk",
    "basicpy @ git+https://github.com/QI2lab/BaSiCPy.git@main",
    "ome-zarr",
]

# CUDA conda pkgs
LINUX_CONDA_CUDA_PKGS = [
    "cuda-version=12.8",
    "cuda-toolkit=12.8",
    "cuda-bindings=12.8.*",
    "cuda-cudart",
    "cuda-nvrtc",
    "cuda-nvvm",
    "cuda-nvcc",
    "cucim",
    "cuvs",
    "cupy",
    "scikit-image",
    "cudnn",
    "cutensor",
    "nccl",
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
    "jax[cuda12_local]==0.4.38",
}

WINDOWS_OTHER_PIP_DEPS = {
    "cupy-cuda12x",
    "scikit-image",
}


def run(command: str) -> None:
    typer.echo(f"$ {command}")
    subprocess.run(command, shell=True, check=True)


def _find_conda_installer(is_windows: bool) -> str:
    """
    Return a usable path to micromamba/mamba/conda.

    Preference order:
    1) micromamba/mamba/conda on PATH
    2) CONDA_EXE env var (often set even when conda isn't on PATH)
    3) common install locations
    """
    for name in ("micromamba", "mamba", "conda"):
        exe = shutil.which(name)
        if exe:
            return exe

    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe and Path(conda_exe).exists():
        return conda_exe

    candidates: list[Path] = []
    if is_windows:
        user = os.environ.get("USERPROFILE", "")
        candidates += [
            Path(user) / "miniconda3" / "condabin" / "conda.bat",
            Path(user) / "miniconda3" / "Scripts" / "conda.exe",
            Path(user) / "anaconda3" / "condabin" / "conda.bat",
            Path(user) / "anaconda3" / "Scripts" / "conda.exe",
            Path("C:/ProgramData/Miniconda3/condabin/conda.bat"),
            Path("C:/ProgramData/Miniconda3/Scripts/conda.exe"),
            Path("C:/ProgramData/Anaconda3/condabin/conda.bat"),
            Path("C:/ProgramData/Anaconda3/Scripts/conda.exe"),
        ]
    else:
        home = Path.home()
        candidates += [
            home / "miniconda3" / "bin" / "conda",
            home / "miniforge3" / "bin" / "conda",
            home / "mambaforge" / "bin" / "conda",
            home / "anaconda3" / "bin" / "conda",
            Path("/opt/conda/bin/conda"),
            Path("/usr/local/miniconda/bin/conda"),
        ]

    for p in candidates:
        if p.exists():
            return str(p)

    raise FileNotFoundError(
        "Could not find conda/mamba/micromamba. "
        "Open a conda-initialized shell, or ensure conda is installed and reachable."
    )


def _quote_exe(path_or_name: str, is_windows: bool) -> str:
    # For shell=True: Windows uses cmd.exe (single quotes don't quote reliably),
    # so use double quotes. POSIX shells accept shlex.quote.
    if is_windows:
        if any(ch in path_or_name for ch in (" ", "\t")) and not (
            path_or_name.startswith('"') and path_or_name.endswith('"')
        ):
            return f'"{path_or_name}"'
        return path_or_name
    return shlex.quote(path_or_name)


@app.command()
def setup_cuda() -> None:
    """
    1) Installs CUDA packages via conda/mamba/micromamba (OS-specific lists).
    2) Writes one activation hook (.sh on Linux, .bat on Windows).
    3) Installs pip deps using the active env's Python.
    """
    prefix = os.environ.get("CONDA_PREFIX")
    if not prefix:
        typer.echo("Error: activate your conda environment first.", err=True)
        raise typer.Exit(1)

    is_windows = platform.system() == "Windows"

    try:
        installer = _find_conda_installer(is_windows=is_windows)
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from e

    installer_q = _quote_exe(installer, is_windows=is_windows)

    # Install CUDA stack via conda/mamba
    pkgs = WINDOWS_CONDA_CUDA_PKGS if is_windows else LINUX_CONDA_CUDA_PKGS
    run(f"{installer_q} install -y -c rapidsai -c conda-forge -c nvidia {' '.join(pkgs)}")

    # Activation hook(s)
    activate_dir = Path(prefix) / "etc" / "conda" / "activate.d"
    activate_dir.mkdir(parents=True, exist_ok=True)

    # Remove only hooks we manage (do NOT delete other packages' hooks)
    sh_hook = activate_dir / "cuda_override.sh"
    bat_hook = activate_dir / "set_cuda_path.bat"
    for p in (sh_hook, bat_hook):
        try:
            p.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass

    if is_windows:
        # Keep it simple: point CUDA_* at env root/Library, and set XLA_FLAGS as well.
        bat_hook.write_text(
            "@echo off\n"
            "rem Point CUDA vars at the conda env so JAX/XLA can find nvvm/libdevice.\n"
            'set "CUDA_DIR=%CONDA_PREFIX%"\n'
            'set "CUDA_HOME=%CONDA_PREFIX%"\n'
            'set "CUDA_PATH=%CONDA_PREFIX%\\Library"\n'
            "\n"
            "rem Tell XLA where CUDA “data” lives.\n"
            'if defined XLA_FLAGS (\n'
            '  set "XLA_FLAGS=--xla_gpu_cuda_data_dir=%CONDA_PREFIX% %XLA_FLAGS%"\n'
            ") else (\n"
            '  set "XLA_FLAGS=--xla_gpu_cuda_data_dir=%CONDA_PREFIX%"\n'
            ")\n"
        )

        # Optional: provide a `which.exe` shim inside the env
        try:
            system_where = Path(os.environ["WINDIR"]) / "System32" / "where.exe"
            dest_which = Path(prefix) / "Scripts" / "which.exe"
            dest_which.parent.mkdir(parents=True, exist_ok=True)
            if system_where.exists() and not dest_which.exists():
                shutil.copy(system_where, dest_which)
        except KeyError:
            pass

    else:
        # Linux: critical fix for your error:
        # - CUDA_DIR must be the env root (contains nvvm/libdevice/libdevice.10.bc)
        # - XLA_FLAGS must point at the env root for CUDA data discovery
        cuda_root = prefix
        sh_hook.write_text(
            "#!/usr/bin/env sh\n"
            "# Conda CUDA + JAX/XLA configuration.\n"
            "\n"
            f'CUDA_ROOT="{cuda_root}"\n'
            "\n"
            "# XLA error messages reference CUDA_DIR specifically.\n"
            'export CUDA_DIR="$CUDA_ROOT"\n'
            'export CUDA_HOME="$CUDA_ROOT"\n'
            'export CUDA_PATH="$CUDA_ROOT"\n'
            "\n"
            "# Tools like ptxas/nvlink usually live in $CONDA_PREFIX/bin.\n"
            'export PATH="$CUDA_ROOT/bin:$PATH"\n'
            "\n"
            "# Libraries can be in both $CONDA_PREFIX/lib and targets/x86_64-linux/lib.\n"
            'export LD_LIBRARY_PATH="$CUDA_ROOT/lib:$CUDA_ROOT/targets/x86_64-linux/lib'
            '${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"\n'
            "\n"
            "# Tell XLA where CUDA “data” (nvvm/libdevice, etc.) lives.\n"
            'if [ -n "${XLA_FLAGS:-}" ]; then\n'
            '  export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_ROOT ${XLA_FLAGS}"\n'
            "else\n"
            '  export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_ROOT"\n'
            "fi\n"
        )
        sh_hook.chmod(sh_hook.stat().st_mode | stat.S_IEXEC)

    # Pip installs (always via the env python)
    subprocess.run([sys.executable, "-m", "pip", "install", *BASE_PIP_DEPS], check=True)

    if is_windows:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", *sorted(WINDOWS_OTHER_PIP_DEPS)],
            check=True,
        )
    else:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", *sorted(LINUX_JAX_LIB)],
            check=True,
        )

    typer.echo(
        f"\nsetup complete!  Please 'conda deactivate' then 'conda activate {prefix}' to apply changes."
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()