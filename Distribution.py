import argparse
import os
import matplotlib.pyplot as plt

def analyze_dataset(directory):
    """
    Parcourt le dossier principal pour compter le nombre d'images
    par sous-dossier (variétés/maladies) et affiche les graphiques.
    """
    print(f"Analyzing dataset in {directory}...")
    
    # 1. Collecter les données
    plant_data = {}
    total_images = 0
    
    # Extensions d'images supportées
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

    for root, dirs, files in os.walk(directory):
        # On ignore le dossier parent s'il n'a pas lui-même d'images
        if root == directory:
            continue
            
        folder_name = os.path.basename(root)
        image_count = sum(1 for file in files if os.path.splitext(file)[1].lower() in valid_extensions)
        
        if image_count > 0:
            plant_data[folder_name] = image_count
            total_images += image_count

    if not plant_data:
        print("No images found in the subdirectories.")
        return

    # 2. Préparer les graphiques
    labels = list(plant_data.keys())
    counts = list(plant_data.values())
    
    # Extraire le nom du dossier racine pour le titre (ex: "Apple")
    dataset_name = os.path.basename(os.path.normpath(directory))

    # Configuration de la figure avec 2 sous-graphiques (1 ligne, 2 colonnes)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"Distribution of {dataset_name} dataset ({total_images} total images)", fontsize=16)

    # --- Pie Chart : Proportion ---
    ax1.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired(range(len(labels))))
    ax1.set_title("Proportion of diseases/varieties")
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # --- Bar Chart : Quantité ---
    bars = ax2.bar(labels, counts, color=plt.cm.Paired(range(len(labels))))
    ax2.set_title("Number of images per disease/variety")
    ax2.set_xlabel("Disease / Variety")
    ax2.set_ylabel("Number of images")
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right") # Rotation pour lisibilité

    # Ajouter les valeurs exactes au-dessus des barres
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, yval + (max(counts) * 0.01), int(yval), ha='center', va='bottom')

    # Ajuster la mise en page
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 3. Afficher les graphiques
    plt.show()

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
