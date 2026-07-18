"""Enforce complete NumPy-style callable docstrings across the repository."""

import ast
import re
from pathlib import Path


def test_all_callables_document_parameters_and_returns() -> None:
    """Require complete ``Parameters`` and ``Returns`` sections.

    Parameters
    ----------
    None
        This callable has no parameters.

    Returns
    -------
    None
        No value is returned.
    """
    failures = []
    section_pattern = re.compile(
        r"(?ms)^([A-Za-z][A-Za-z ]+)\n-{3,}\n(.*?)(?=^[A-Za-z][A-Za-z ]+\n-{3,}\n|\Z)"
    )

    repository_root = Path(__file__).parents[1]
    for source_root in (repository_root / "src", repository_root / "tests"):
        for path in source_root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue

                location = (
                    f"{path.relative_to(repository_root)}:{node.lineno}:{node.name}"
                )
                docstring = ast.get_docstring(node, clean=True) or ""
                sections = dict(section_pattern.findall(docstring))
                for required_section in ("Parameters", "Returns"):
                    if not sections.get(required_section, "").strip():
                        failures.append(f"{location} missing {required_section}")

                parameter_body = sections.get("Parameters", "")
                arguments = (
                    *node.args.posonlyargs,
                    *node.args.args,
                    *node.args.kwonlyargs,
                )
                parameter_names = [
                    argument.arg
                    for argument in arguments
                    if argument.arg not in {"self", "cls"}
                ]
                if node.args.vararg is not None:
                    parameter_names.append(node.args.vararg.arg)
                if node.args.kwarg is not None:
                    parameter_names.append(node.args.kwarg.arg)

                for parameter_name in parameter_names:
                    entry_pattern = rf"(?m)^\s*{re.escape(parameter_name)}\s*(?::|$)"
                    if re.search(entry_pattern, parameter_body) is None:
                        failures.append(
                            f"{location} missing parameter {parameter_name!r}"
                        )

    assert not failures, "\n" + "\n".join(failures)
