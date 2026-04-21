import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


def flip_image(img):
    return cv2.flip(img, 1)


def rotate_image(img, angle=45):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]

    return cv2.warpAffine(img, rotation_matrix, (new_w, new_h))


def skew_image(img):
    h, w = img.shape[:2]
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

    offset = int(w * 0.2)
    pts2 = np.float32([[offset, 0], [w - offset, 0], [0, h], [w, h]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, matrix, (w, h))


def shear_image(img):
    h, w = img.shape[:2]
    shear_factor = 0.3
    matrix = np.float32([[1, shear_factor, 0], [0, 1, 0]])

    new_w = int(w + (h * shear_factor))
    return cv2.warpAffine(img, matrix, (new_w, h))


def crop_image(img):
    h, w = img.shape[:2]
    target_h, target_w = int(h * 0.5), int(w * 0.5)
    start_y, start_x = (h - target_h) // 2, (w - target_w) // 2
    return img[start_y : start_y + target_h, start_x : start_x + target_w]


def distort_image(img, power=0.5):
    h, w = img.shape[:2]
    k_fx, k_fy = w * 0.5, h * 0.5
    k_cx, k_cy = w / 2, h / 2
    camera_matrix = np.array(
        [[k_fx, 0, k_cx], [0, k_fy, k_cy], [0, 0, 1]], dtype=np.float32
    )

    dist_coeffs = np.array([power, power, 0, 0, 0], dtype=np.float32)
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )
    return cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)


def augment_image(image_path):
    print(f"Applying data augmentation on {image_path}...")

    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read image at {image_path}")
        return

    orig_name, ext = os.path.splitext(image_path)

    transformations = {
        "Flip": flip_image(img),
        "Rotate": rotate_image(img, 45),
        "Skew": skew_image(img),
        "Shear": shear_image(img),
        "Crop": crop_image(img),
        "Distortion": distort_image(img, -0.2),
    }

    fig, axes = plt.subplots(3, 2, figsize=(10, 8))
    fig.suptitle(f"Data Augmentation for {os.path.basename(image_path)}", fontsize=14)
    axes_flat = axes.flatten()

    for idx, (trans_name, modified_img) in enumerate(transformations.items()):
        out_filename = f"{orig_name}_{trans_name}{ext}"
        cv2.imwrite(out_filename, modified_img)
        print(f"Saved: {out_filename}")

        img_rgb = cv2.cvtColor(modified_img, cv2.COLOR_BGR2RGB)
        ax = axes_flat[idx]
        ax.imshow(img_rgb)
        ax.set_title(trans_name)
        ax.axis("off")

    plt.tight_layout()

    os.makedirs("results", exist_ok=True)
    base_name = os.path.basename(orig_name)
    out_path = f"results/augmentation_plot_{base_name}.png"

    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"Summary plot saved to: {out_path}")
