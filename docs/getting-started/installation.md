# Installation

Run these commands from a checkout of this repository.

Python 3.12 and [`uv`](https://docs.astral.sh/uv/) are required.

```bash
uv sync
```

On Windows and Linux, this includes the default `gpu` dependency group, which
selects the CUDA extra for deconvolution and registration. Plain `uv sync` and
`uv run` keep CuPy and its NVIDIA runtime dependencies installed; you do not
need to repeat `--extra gpu`. The explicit extra remains supported.

To deliberately omit the default GPU group (for a non-CUDA environment):

```bash
uv sync --no-group gpu
```

Native Windows also requires the pure-Python `cucim.skimage` package from a
local cuCIM checkout because RAPIDS does not publish its standard wheel there.

## cuCIM on Windows

Install cuCIM from the tagged source checkout using an editable installation:

1. In the Start menu, right-click **Miniforge Prompt** and select
   **Run as Administrator** so Git can create symbolic links.
2. Enable symbolic links globally for Git:

   ```bat
   git config --global --add core.symlinks true
   ```

3. Change to this project's directory and activate the environment created by
   `uv sync`:

   ```bat
   cd /d C:\Users\qi2lab\Documents\github\opm-processing-v2
   .venv\Scripts\activate.bat
   ```

4. Install cuCIM:

   ```bat
   pip install -e "git+https://github.com/rapidsai/cucim.git@v26.08.00#egg=cucim-cu12&subdirectory=python/cucim"
   ```

Continue with the [first workflow](quickstart.md).
