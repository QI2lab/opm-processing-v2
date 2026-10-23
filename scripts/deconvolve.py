"""Process an OME-Zarr using its recorded camera and microscope calibration.

Run: uv run python scripts/deconvolve.py /path/to/acquisition.ome.zarr
"""

import argparse
from pathlib import Path

from opm_processing.process import process


def main() -> None:
    """Run metadata-driven acquisition processing with RLGC enabled."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("acquisition", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()
    process(
        root_path=args.acquisition,
        deconvolve=True,
        save_float32=True,
        decon_gpu_id=args.gpu_id,
        output=args.output,
    )


if __name__ == "__main__":
    main()
