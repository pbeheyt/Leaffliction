import argparse
import os

from src.data_analysis.augmentation import augment_image
from src.data_analysis.balance import balance_directory


def run(path, output):
    if os.path.isfile(path):
        augment_image(path)
        return

    if os.path.isdir(path):
        balance_directory(path, output)
        print("\nDataset generation completed successfully!")
        return

    print(f"Error: {path} is not a valid file or directory.")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Unified data analysis entrypoint for single-image "
            "augmentation or dataset balancing."
        )
    )
    parser.add_argument("path", help="Path to an image or a dataset directory")
    parser.add_argument(
        "--output",
        default="data/augmented",
        help="Output directory used when the input is a dataset directory",
    )
    args = parser.parse_args()
    run(args.path, args.output)


if __name__ == "__main__":
    main()
