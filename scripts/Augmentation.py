#!/usr/bin/env python3
import sys
import os
import argparse

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

from src.data_analysis.augmentation import augment_image

def main():
    parser = argparse.ArgumentParser(
        description="Apply 6 augmentations to a single image."
    )
    parser.add_argument("image_path", help="Path to the image to augment")
    args = parser.parse_args()

    if os.path.isfile(args.image_path):
        results_dir = os.path.join(ROOT_DIR, "results")
        augment_image(args.image_path, results_dir=results_dir)
    else:
        print(f"Error: {args.image_path} is not a valid file.")
        sys.exit(1)

if __name__ == "__main__":
    main()
