import os
import sys
import stat
import subprocess
from pathlib import Path
import typer

app = typer.Typer()
app.pretty_exceptions_enable = False

@app.command()
def setup_activation():
    """
    Create a conda activation script that adds NVIDIA lib directories
    to LD_LIBRARY_PATH on Linux/macOS or to PATH on Windows.
    On Windows, also install the cuCIM package from GitHub.
    """
    # Get current conda environment prefix
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if not conda_prefix:
        typer.echo("Error: CONDA_PREFIX is not set. Are you inside a conda environment?", err=True)
        raise typer.Exit(code=1)

    activate_dir = Path(conda_prefix) / 'etc' / 'conda' / 'activate.d'
    activate_dir.mkdir(parents=True, exist_ok=True)

    if sys.platform == "win32":
        # Windows: prepend each nvidia\lib directory to %PATH%
        script_path = activate_dir / 'nvidia_path.bat'
        script_content = r"""@echo off
REM Automatically added: prepend NVIDIA lib directories to %%PATH%%
FOR /F "delims=" %%D IN ('dir /b /s "%CONDA_PREFIX%\Lib\site-packages\nvidia\lib"') DO (
    set "PATH=%%D;%%PATH%%"
)
"""
        # Write and echo
        script_path.write_text(script_content)
        typer.echo(f"Activation script created at {script_path}")

        # Install cuCIM from the specified GitHub branch on Windows only
        try:
            typer.echo("Installing cuCIM from GitHub (Windows only)...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-e",
                "git+https://github.com/rapidsai/cucim.git@v25.06.00#egg=cucim&subdirectory=python/cucim"
            ], check=True)
            typer.echo("cuCIM installation complete.")
        except subprocess.CalledProcessError as e:
            typer.echo(f"Error during cuCIM installation: {e}", err=True)
            raise typer.Exit(code=1)

    else:
        # Linux/macOS: prepend each nvidia/lib dir to LD_LIBRARY_PATH
        py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
        script_path = activate_dir / 'nvidia_ld_library_path.sh'
        script_content = f"""#!/usr/bin/env sh
# Automatically added: prepend NVIDIA lib directories to LD_LIBRARY_PATH
for d in $(find "$CONDA_PREFIX/lib/{py_ver}/site-packages/nvidia" -type d -name lib); do
    export LD_LIBRARY_PATH="$d:$LD_LIBRARY_PATH"
done
"""
        # Write and chmod
        script_path.write_text(script_content)
        mode = script_path.stat().st_mode if script_path.exists() else 0
        script_path.chmod(mode | stat.S_IEXEC)
        typer.echo(f"Activation script created at {script_path}")

def main():
    app()

if __name__ == "__main__":
    main()
