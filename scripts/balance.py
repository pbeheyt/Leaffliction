import argparse
import os
import random
import shutil
import cv2
import augmentation

def balance_directory(dataset_dir, output_dir):
    """Balances the dataset using augmentations on minority classes."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    class_counts = {}
    
    for root, dirs, files in os.walk(dataset_dir):
        if root == dataset_dir:
            continue
        
        folder_name = os.path.basename(root)
        images = [f for f in files if os.path.splitext(f)[1].lower() in valid_extensions]
        class_counts[folder_name] = images

    if not class_counts:
        print(f"No valid dataset found in {dataset_dir}")
        return

    max_images = max([len(imgs) for imgs in class_counts.values()])
    print(f"Target count per class for balancing: {max_images} images.")
    
    augmentations = {
        "Flip": augmentation.flip_image,
        "Rotate": lambda img: augmentation.rotate_image(img, 45),
        "Skew": augmentation.skew_image,
        "Shear": augmentation.shear_image,
        "Crop": augmentation.crop_image,
        "Distortion": lambda img: augmentation.distort_image(img, -0.2)
    }

    for class_name, image_list in class_counts.items():
        src_path = os.path.join(dataset_dir, class_name)
        dst_path = os.path.join(output_dir, class_name)
        
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

        current_count = len(image_list)
        print(f"[{class_name}] Processing {current_count} files...")

        for img_name in image_list:
            shutil.copy2(os.path.join(src_path, img_name), os.path.join(dst_path, img_name))

        images_to_add = max_images - current_count
        if images_to_add <= 0:
            continue

        print(f"[{class_name}] Augmenting {images_to_add} files to reach balance...")
        while images_to_add > 0:
            rand_img_name = random.choice(image_list)
            rand_img_path = os.path.join(src_path, rand_img_name)
            img = cv2.imread(rand_img_path)
            
            if img is None: continue

            trans_name, trans_func = random.choice(list(augmentations.items()))
            modified_img = trans_func(img)

            orig_name, ext = os.path.splitext(rand_img_name)
            
            new_filename = f"{orig_name}_{trans_name}_{images_to_add}{ext}"
            cv2.imwrite(os.path.join(dst_path, new_filename), modified_img)
            
            images_to_add -= 1


def main():
    parser = argparse.ArgumentParser(description="Balances a dataset by generating augmented images.")
    parser.add_argument("directory", help="Path to the original dataset directory")
    parser.add_argument("--output", default="augmented_directory", help="Path where the balanced dataset will be saved.")
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a valid directory.")
        return

    balance_directory(args.directory, args.output)
    print("\nDataset generation completed successfully!")


if __name__ == "__main__":
    main()
