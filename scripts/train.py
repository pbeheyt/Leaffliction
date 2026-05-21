#!/usr/bin/env python3
import sys
import os
import argparse

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

from src.classification.train import train_model

def main():
    parser = argparse.ArgumentParser(
        description="Train classifier with stratified split, validation metrics, and artifact export."
    )
    parser.add_argument("directory", help="Path to the dataset directory")
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Validation ratio for stratified split (default: 0.2)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for deterministic split (default: 42)",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=os.path.join(ROOT_DIR, "models"),
        help="Directory where trained artifacts are saved (default: models).",
    )
    parser.add_argument(
        "--report-dir",
        default=os.path.join(ROOT_DIR, "results/classification"),
        help=(
            "Directory where validation reports are saved "
            "(default: results/classification)."
        ),
    )
    parser.add_argument(
        "--min-val-samples",
        type=int,
        default=100,
        help=(
            "Minimum recommended validation samples for evaluation "
            "evidence (default: 100)."
        ),
    )
    args = parser.parse_args()

    train_model(
        directory=args.directory,
        test_size=args.test_size,
        random_state=args.random_state,
        artifacts_dir=args.artifacts_dir,
        report_dir=args.report_dir,
        min_val_samples=args.min_val_samples,
    )

if __name__ == "__main__":
    main()
