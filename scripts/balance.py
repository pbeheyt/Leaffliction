#!/usr/bin/env python3
import sys
import os
import argparse

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

from src.data_analysis.balance import balance_directory

def main():
    parser = argparse.ArgumentParser(
        description="Balance a dataset by augmenting minority classes."
    )
    parser.add_argument("dataset_dir", help="Path to the dataset directory")
    parser.add_argument(
        "--output",
        default=os.path.join(ROOT_DIR, "data/augmented"),
        help="Output directory for the balanced dataset",
    )
    args = parser.parse_args()

    if os.path.isdir(args.dataset_dir):
        balance_directory(args.dataset_dir, args.output)
        print("\nDataset balancing completed successfully!")
    else:
        print(f"Error: {args.dataset_dir} is not a valid directory.")
        sys.exit(1)

if __name__ == "__main__":
    main()
