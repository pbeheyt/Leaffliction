import argparse
import os
# import matplotlib.pyplot as plt


def analyze_dataset(directory):
    """
    Parcourt le dossier principal pour compter le nombre d'images
    par sous-dossier (variétés/maladies) et affiche les graphiques.
    """
    print(f"Analyzing dataset in {directory}...")
    # TODO: Parcourir le dossier, compter les fichiers
    # TODO: Afficher le Pie chart
    # TODO: Afficher le Bar chart
    pass


def main():
    parser = argparse.ArgumentParser(
        description="Analyses the distribution of images in a dataset."
    )
    parser.add_argument("directory", help="Path to the dataset directory")
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a valid directory.")
        return

    analyze_dataset(args.directory)


if __name__ == "__main__":
    main()
