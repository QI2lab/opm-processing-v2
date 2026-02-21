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
    "'cuda-version=12.8'",
    "'cuda-toolkit=12.8'",
    "'cuda-bindings=12.8.*'",
    "cuda-cudart",
    "cuda-nvrtc",
    "cucim",
    "cupy",
    "scikit-image",
    "cudnn",
    "cutensor",
    "nccl",
    "'numpy==1.26.4'",
    "'scipy=1.12.0'",
]

WINDOWS_CONDA_CUDA_PKGS = [
    "cuda-version=12.8",
    "cuda-toolkit=12.8",
    "'cuda-bindings=12.8.*'",
    "cuda-cudart",
    "cudnn",
    "cutensor",
    "cuda-nvcc",
    "'numpy==1.26.4'",
    "'scipy=1.12.0'",
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

def _find_conda_installer() -> str:
    """
    Return a usable path to an environment installer executable.

    Preference order:
    1) micromamba, mamba, conda on PATH
    2) CONDA_EXE env var (if set)
    3) common install locations (miniconda/anaconda) on the current OS
    """
    # 1) PATH search
    for name in ("micromamba", "mamba", "conda"):
        exe = shutil.which(name)
        if exe:
            return exe

    # 2) CONDA_EXE is often set even when conda isn't on PATH
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe and Path(conda_exe).exists():
        return conda_exe

    # 3) Common install roots
    system = platform.system()
    candidates: list[Path] = []

    if system == "Windows":
        # Typical Windows installs
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
        # Typical Linux/macOS installs
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
        "Activate a conda-initialized shell, or ensure conda is installed and reachable."
    )

def run(command: str):
    typer.echo(f"$ {command}")
    subprocess.run(command, shell=True, check=True)

@app.command()
def setup_cuda() -> None:
    """
    1) Installs CUDA packages via conda/mamba/micromamba (OS-specific lists).
    2) Writes a single activation hook (either .sh on Linux or .bat on Windows).
    3) Installs all pip deps (and OS-specific extras) using the active env's Python.
    """
    prefix = os.environ.get("CONDA_PREFIX")
    if not prefix:
        typer.echo("Error: activate your conda environment first.", err=True)
        raise typer.Exit(1)

    is_windows = platform.system() == "Windows"

    def _find_conda_installer() -> str:
        # 1) PATH search
        for name in ("micromamba", "mamba", "conda"):
            exe = shutil.which(name)
            if exe:
                return exe

        # 2) CONDA_EXE is often set even if conda isn't on PATH
        conda_exe = os.environ.get("CONDA_EXE")
        if conda_exe and Path(conda_exe).exists():
            return conda_exe

        # 3) Common install roots
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

    def _quote_exe(path_or_name: str) -> str:
        # For shell=True: Windows uses cmd.exe (single quotes don't quote reliably),
        # so use double quotes. POSIX shells accept shlex.quote.
        if is_windows:
            if any(ch in path_or_name for ch in (" ", "\t")) and not (
                path_or_name.startswith('"') and path_or_name.endswith('"')
            ):
                return f'"{path_or_name}"'
            return path_or_name
        return shlex.quote(path_or_name)

    try:
        installer = _find_conda_installer()
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from e

    installer_q = _quote_exe(installer)

    # Install CUDA stack via conda/mamba
    if is_windows:
        pkgs = " ".join(WINDOWS_CONDA_CUDA_PKGS)
    else:
        pkgs = " ".join(LINUX_CONDA_CUDA_PKGS)

    run(f"{installer_q} install -y -c rapidsai -c conda-forge -c nvidia {pkgs}")

    # Activation hook(s)
    activate_dir = Path(prefix) / "etc" / "conda" / "activate.d"
    activate_dir.mkdir(parents=True, exist_ok=True)

    # Remove only the hooks we manage (avoid nuking other packages' hooks)
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
        bat_hook.write_text(
            "@echo off\n"
            "rem Set CUDA_PATH so CuPy finds the DLLs inside %CONDA_PREFIX%\\Library\n"
            'set "CUDA_PATH=%CONDA_PREFIX%\\Library"\n'
            'set "CUDA_HOME=%CUDA_PATH%"\n'
        )

        # Provide a `which.exe` shim inside the env (optional convenience)
        try:
            system_where = Path(os.environ["WINDIR"]) / "System32" / "where.exe"
            dest_which = Path(prefix) / "Scripts" / "which.exe"
            dest_which.parent.mkdir(parents=True, exist_ok=True)
            if system_where.exists() and not dest_which.exists():
                shutil.copy(system_where, dest_which)
        except KeyError:
            pass
    else:
        env_lib = f"{prefix}/lib"
        linux_cuda_root = f"{prefix}/targets/x86_64-linux"
        sh_hook.write_text(
            "#!/usr/bin/env sh\n"
            "# Point at the conda-installed CUDA toolkit\n"
            f'export CUDA_PATH="{linux_cuda_root}"\n'
            'export CUDA_HOME="$CUDA_PATH"\n'
            'export PATH="$CUDA_PATH/bin:$PATH"\n'
            "\n"
            "# Prepend only the conda toolkit lib & env lib\n"
            f'export LD_LIBRARY_PATH="$CUDA_PATH/lib:{env_lib}${{LD_LIBRARY_PATH:+:${{LD_LIBRARY_PATH}}}}"\n'
        )
        sh_hook.chmod(sh_hook.stat().st_mode | stat.S_IEXEC)

    # Pip installs (use env python explicitly)
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

def main():
    app()


if __name__ == "__main__":
    main()