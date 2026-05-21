import os
import random
import shutil

import cv2

from src.common.load_image import VALID_IMAGE_EXTENSIONS, load_image
from .augmentation import AUGMENTATION_REGISTRY


def balance_directory(dataset_dir, output_dir, seed=None):
    """Balance the dataset using augmentations on minority classes."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if seed is not None:
        random.seed(seed)

    valid_extensions = VALID_IMAGE_EXTENSIONS
    class_samples = {}

    for root, _, files in os.walk(dataset_dir):
        if root == dataset_dir:
            continue

        folder_name = os.path.basename(root)
        images = [
            f
            for f in files
            if os.path.splitext(f)[1].lower() in valid_extensions
        ]
        class_samples[folder_name] = images

    if not class_samples:
        print(f"No valid dataset found in {dataset_dir}")
        return

    max_images = max(len(imgs) for imgs in class_samples.values())
    print(f"Target count per class for balancing: {max_images} images.")

    augmentations = AUGMENTATION_REGISTRY

    for class_name, image_list in class_samples.items():
        if not image_list:
            print(f"[{class_name}] Skipped (no valid images found).")
            continue

        src_path = os.path.join(dataset_dir, class_name)
        dst_path = os.path.join(output_dir, class_name)

        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

        current_count = len(image_list)
        print(f"[{class_name}] Processing {current_count} files...")

        for img_name in image_list:
            shutil.copy2(
                os.path.join(src_path, img_name),
                os.path.join(dst_path, img_name),
            )

        images_to_add = max_images - current_count
        if images_to_add <= 0:
            continue

        print(
            f"[{class_name}] Augmenting {images_to_add} files "
            "to reach balance..."
        )
        failures = 0
        max_failures = max(10, len(image_list) * 2)
        while images_to_add > 0 and failures < max_failures:
            rand_img_name = random.choice(image_list)
            rand_img_path = os.path.join(src_path, rand_img_name)
            try:
                img = load_image(rand_img_path)
            except ValueError:
                failures += 1
                continue

            trans_name, trans_func = random.choice(list(augmentations.items()))
            modified_img = trans_func(img)

            orig_name, ext = os.path.splitext(rand_img_name)
            new_filename = f"{orig_name}_{trans_name}_{images_to_add}{ext}"
            cv2.imwrite(os.path.join(dst_path, new_filename), modified_img)

            images_to_add -= 1

        if failures >= max_failures:
            print(
                f"[{class_name}] Stopped early due to repeated read failures."
            )
