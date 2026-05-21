#!/usr/bin/env python3
import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_transformation.transformation import TRANSFORM_KEYS, run_single, run_batch

def main():
    parser = argparse.ArgumentParser(
        description="Apply 6 image transformations to leaf images."
    )
    parser.add_argument(
        "image", nargs="?", default=None,
        help="Path to a single image (display mode)."
    )
    parser.add_argument(
        "-src", dest="src", default=None,
        help="Source directory containing images (batch mode)."
    )
    parser.add_argument(
        "-dst", dest="dst", default=None,
        help="Destination directory for transformed images."
    )
    for key, name in TRANSFORM_KEYS.items():
        parser.add_argument(
            f"-{key}", dest=key, action="store_true",
            help=f"Apply only '{name}' (combinable)."
        )
    args = parser.parse_args()

    selected = {k for k in TRANSFORM_KEYS if getattr(args, k)}
    if not selected:
        selected = set(TRANSFORM_KEYS.keys())

    if args.image:
        run_single(args.image, selected)
    elif args.src and args.dst:
        run_batch(args.src, args.dst, selected)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
