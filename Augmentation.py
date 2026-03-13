import argparse
import os
# import cv2


def augment_image(image_path):
    """
    Génère 6 transformations pour une image donnée et les enregistre
    dans le même dossier avec les suffixes correspondants.
    """
    print(f"Applying data augmentation on {image_path}...")
    # TODO: Algorithmes des 6 transformations demandées
    # - Flip
    # - Rotate
    # - Skew
    # - Shear
    # - Crop
    # - Distortion
    
    # TODO: Afficher les 6 images (matplotlib ou cv2.imshow)
    # TODO: Sauvegarder les images (ex: image (1)_Flip.JPG)
    pass


def main():
    parser = argparse.ArgumentParser(
        description="Applies data augmentation to a single leaf image."
    )
    parser.add_argument("image_path", help="Path to the source image")
    args = parser.parse_args()

    if not os.path.isfile(args.image_path):
        print(f"Error: {args.image_path} is not a valid file.")
        return

    augment_image(args.image_path)


if __name__ == "__main__":
    main()
