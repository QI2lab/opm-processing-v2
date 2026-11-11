"""Review or apply a uniform stage-coordinate metadata repair without pixel IO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from opm_processing.dataio.coordinate_repair import (
    apply_stage_coordinate_repair,
    plan_stage_coordinate_repair,
)


def main() -> None:
    """Save a dry-run plan, or apply exactly the reviewed plan."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?")
    parser.add_argument(
        "--additional-root",
        type=Path,
        action="append",
        default=[],
        help="Additional explicit acquisition/output directory for split-drive data.",
    )
    parser.add_argument(
        "--plan", type=Path, help="Write a reviewable dry-run JSON plan"
    )
    parser.add_argument(
        "--apply", type=Path, help="Apply a previously reviewed JSON plan"
    )
    args = parser.parse_args()
    if args.apply:
        if args.root is not None or args.plan is not None or args.additional_root:
            parser.error("--apply takes its acquisition root from the reviewed plan")
        result = apply_stage_coordinate_repair(
            json.loads(args.apply.read_text("utf-8"))
        )
    else:
        if args.root is None or args.plan is None:
            parser.error("Dry-run usage requires ROOT --plan PATH")
        plan = plan_stage_coordinate_repair(
            args.root, additional_roots=args.additional_root
        )
        args.plan.parent.mkdir(parents=True, exist_ok=True)
        args.plan.write_text(
            json.dumps(plan, indent=2, ensure_ascii=False) + "\n", "utf-8"
        )
        result = {
            key: value
            for key, value in plan.items()
            if key not in ("files", "read_dependencies")
        }
        result["input_metadata_files"] = len(plan.get("read_dependencies", []))
        result["metadata_files"] = len(plan["files"])
        result["plan_path"] = str(args.plan.resolve())
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
