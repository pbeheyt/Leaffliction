#!/usr/bin/env python3
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_analysis.distribution import analyze_dataset

def main():
    if len(sys.argv) < 2:
        print("Usage: ./Distribution.py [directory]")
        sys.exit(1)
    
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_dir = os.path.join(root_dir, "results")
    analyze_dataset(sys.argv[1], output_dir=output_dir)

if __name__ == "__main__":
    main()
