# Building documentation

The site uses Material for MkDocs, with search, light/dark modes, code-copy
buttons, sidebar navigation, and source-generated Python reference pages. Its
theme and GitHub Pages deployment follow
[merfish3d-analysis](https://qi2lab.github.io/merfish3d-analysis/). Configuration lives in
`mkdocs.yml`; documentation dependencies are in the `docs` group.

## Separate environment

Run from the repository root. This environment contains only documentation
dependencies and does not install the processing package or GPU dependencies.

```bash
uv venv .venv-docs --python 3.12
uv --no-config pip install --python .venv-docs --group docs --index-url https://pypi.org/simple
uv run --python .venv-docs --no-project mkdocs serve
```

The local preview is served at `http://127.0.0.1:8000`. Stop it with Ctrl+C.
Build the complete static site with strict validation:

```bash
uv run --python .venv-docs --no-project mkdocs build --strict
```

Generated HTML goes to ignored `site/`. Reference generation parses the Python
source with inspection disabled, so it does not run processing or initialize CUDA.

## Page conventions

| Page type | Contents |
| --- | --- |
| CLI reference | Purpose, arguments, concise option descriptions, one example |
| Guide | A practical task, required inputs, steps, and resulting outputs |
| Method | Geometry, numerical conventions, assumptions, and validation scope |
| Python reference | Selected functions rendered from source docstrings |

Use relative links between documentation pages. Link source files through their
repository URLs. Add every published page to navigation and check the strict
build after moving files. Keep audit notes, experiment reports, and performance
measurements under ignored `diagnostics/`, outside the published documentation.

## CI and publication

The [published documentation](https://qi2lab.github.io/opm-processing-v2/)
uses GitHub Actions as its Pages source. Automatic builds run only when newly
pushed commits change `docs/`, `tests/CONTRIBUTING.md`, or `mkdocs.yml`. The workflow
checks the revisions from each push so unrelated commits do not rebuild earlier
documentation changes in the same PR. Successful builds on `main` deploy to
GitHub Pages; pull requests only validate the site. Manual runs can refresh
generated Python references or force a build, with deployment limited to `main`.

Build and deploy use only documentation dependencies. A local build produces
`site/` without publishing it.
