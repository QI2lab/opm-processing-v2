import os
import sys
import subprocess
from pathlib import Path
import typer

app = typer.Typer()
app.pretty_exceptions_enable = False

@app.command()
def setup_activation():
    """
    Create a conda activation script that:
      • on Linux/macOS, prepends NVIDIA lib dirs to LD_LIBRARY_PATH
      • on Windows, deletes any old .bat, then prepends Conda Scripts, Library\\bin
        and all NVIDIA DLL dirs to %PATH%, sets CUDA_PATH correctly so nvrtc64_120_0.dll
        (and friends) become discoverable by CuPy.
    """
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        typer.echo("Error: CONDA_PREFIX is not set. Are you inside a conda environment?", err=True)
        raise typer.Exit(code=1)

    activate_dir = Path(conda_prefix) / "etc" / "conda" / "activate.d"
    activate_dir.mkdir(parents=True, exist_ok=True)

    if sys.platform == "win32":
        site_pkgs = Path(conda_prefix) / "Lib" / "site-packages"

        # 1) Find all NVIDIA DLL directories
        dll_dirs = sorted({
            str(p.parent)
            for p in site_pkgs.rglob("*.dll")
            if "nvidia" in map(str.lower, p.parts)
        })

        # 2) Locate the NVRTC root (one level above 'bin')
        nvrtc_root_dirs = {
            str(p.parent.parent)
            for p in site_pkgs.rglob("nvrtc64_*.dll")
        }
        nvrtc_root = next(iter(nvrtc_root_dirs), None)

        # 3) Remove any existing batch file, then write nvidia_path.bat
        script_path = activate_dir / "nvidia_path.bat"
        if script_path.exists():
            script_path.unlink()

        lines = [
            "@echo off",
            "REM — Prepend Conda Scripts & Library\\bin so executables are visible",
            'set "PATH=%CONDA_PREFIX%\\Scripts;%CONDA_PREFIX%\\Library\\bin;%PATH%"',
            ""
        ]
        if nvrtc_root:
            lines += [
                "REM — Point CUDA_PATH at the NVRTC root (so CuPy adds \\bin correctly)",
                f'set "CUDA_PATH={nvrtc_root}"',
                "REM — Also make the NVRTC bin directory visible right away",
                'set "PATH=%CUDA_PATH%\\bin;%PATH%"',
                ""
            ]
        else:
            lines += [
                "REM — Warning: no nvrtc64_*.dll found under site-packages!",
                ""
            ]
        lines += ["REM — Now prepend every NVIDIA DLL directory:"]
        for d in dll_dirs:
            lines.append(f'set "PATH={d};%PATH%"')

        script_path.write_text("\r\n".join(lines))
        typer.echo(f"Activation script written to {script_path}")

        # 4) Register everything in this Python session immediately
        scripts_dir = Path(conda_prefix) / "Scripts"
        lib_bin_dir = Path(conda_prefix) / "Library" / "bin"
        os.environ["PATH"] = str(scripts_dir) + os.pathsep + str(lib_bin_dir) + os.pathsep + os.environ.get("PATH", "")

        if nvrtc_root:
            os.environ["CUDA_PATH"] = nvrtc_root
            bin_dir = Path(nvrtc_root) / "bin"
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ["PATH"]
            typer.echo(f"Set CUDA_PATH={nvrtc_root}")

        for d in dll_dirs:
            os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]
        typer.echo(f"Added {len(dll_dirs)} NVIDIA DLL dirs to PATH for this session.")

        # 5) Verify visibility
        if nvrtc_root:
            try:
                out = subprocess.run(
                    ["where", "nvrtc64_120_0.dll"],
                    capture_output=True, text=True, check=True
                )
                typer.echo(f"nvrtc DLL found at:\n{out.stdout.strip()}")
            except subprocess.CalledProcessError:
                typer.echo("Warning: nvrtc64_120_0.dll still not found!", err=True)
    else:
        # Linux/macOS: prepend NVIDIA lib dirs to LD_LIBRARY_PATH
        py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
        script_path = activate_dir / "nvidia_ld_library_path.sh"
        if script_path.exists():
            script_path.unlink()
        script_content = f"""#!/usr/bin/env sh
# Automatically added: prepend NVIDIA lib directories to LD_LIBRARY_PATH
for d in $(find "$CONDA_PREFIX/lib/{py_ver}/site-packages" -type f -path "*/nvidia*/*.so" -printf "%h\\n" | sort -u); do
    export LD_LIBRARY_PATH="$d:$LD_LIBRARY_PATH"
done
"""
        script_path.write_text(script_content)
        script_path.chmod(script_path.stat().st_mode | 0o111)
        typer.echo(f"Activation script written to {script_path}")

def main():
    app()

if __name__ == "__main__":
    main()
