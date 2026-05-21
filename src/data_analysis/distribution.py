import os

import matplotlib.pyplot as plt

from src.common.load_image import VALID_IMAGE_EXTENSIONS


def analyze_dataset(directory, output_dir="results"):
    """
    Analyze the dataset by counting images per subdirectory
    and generating pie and bar charts.
    """
    print(f"Analyzing dataset in {directory}...")

    plant_data = {}
    total_images = 0
    valid_extensions = VALID_IMAGE_EXTENSIONS

    for root, _, files in os.walk(directory):
        if root == directory:
            continue

        folder_name = os.path.basename(root)
        image_count = sum(
            1
            for file in files
            if os.path.splitext(file)[1].lower() in valid_extensions
        )

        if image_count > 0:
            plant_data[folder_name] = image_count
            total_images += image_count

    if not plant_data:
        print("No images found in the subdirectories.")
        return

    labels = list(plant_data.keys())
    samples = list(plant_data.values())
    dataset_name = os.path.basename(os.path.normpath(directory))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        f"Distribution of {dataset_name} dataset "
        f"({total_images} total images)",
        fontsize=16,
    )

    ax1.pie(
        samples,
        labels=labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=plt.cm.Paired(range(len(labels))),
    )
    ax1.set_title("Proportion of diseases/varieties")
    ax1.axis("equal")

    bars = ax2.bar(labels, samples, color=plt.cm.Paired(range(len(labels))))
    ax2.set_title("Number of images per disease/variety")
    ax2.set_xlabel("Disease / Variety")
    ax2.set_ylabel("Number of images")
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")

    for bar in bars:
        yval = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            yval + (max(samples) * 0.01),
            int(yval),
            ha="center",
            va="bottom",
        )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"distribution_{dataset_name}.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"Distribution chart saved to: {out_path}")
