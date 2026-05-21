from plantcv import plantcv as pcv
import matplotlib.pyplot as plt
import numpy as np
import sys
import cv2
import os
from src.common.load_image import load_image
from src.common.mask import build_mask

TRANSFORM_KEYS = {
    "blur": "Gaussian blur",
    "mask": "Mask",
    "roi": "ROI objects",
    "analyze": "Analyze object",
    "landmarks": "Pseudolandmarks",
    "hist": "Color histogram",
}
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def color_histogram_draw(img, mask, is_extracting_features=False):
    def _draw(ax):
        total = int(np.count_nonzero(mask))
        channels = [
            ("Blue", img, 0, "blue", 256),
            ("Green", img, 1, "green", 256),
            ("Red", img, 2, "red", 256),
        ]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        channels += [
            ("Hue", hsv, 0, "purple", 180),
            ("Saturation", hsv, 1, "cyan", 256),
            ("Value", hsv, 2, "orange", 256),
            ("Lightness", lab, 0, "gray", 256),
            ("Green-Magenta", lab, 1, "magenta", 256),
            ("Blue-Yellow", lab, 2, "gold", 256),
        ]
        n_features = sum(vmax for *_, vmax in channels)
        if total == 0:
            return [0.0] * n_features if is_extracting_features else None

        hist_features = []
        for name, src, ch, color, vmax in channels:
            hist = cv2.calcHist(
                [src], [ch], mask, [vmax], [0, vmax]).flatten()
            hist = hist / total
            if is_extracting_features:
                hist_features.extend(hist.astype(float).tolist())
            else:
                ax.plot(hist * 100.0, color=color, label=name)

        if not is_extracting_features:
            ax.set_xlim([0, 256])
            ax.set_xlabel("Pixel intensity")
            ax.set_ylabel("Proportion of pixels (%)")
            ax.legend(fontsize=7)
        else:
            return hist_features

    return _draw


def pseudolandmarks_transform(img, mask):
    out = img.copy()
    pcv.params.debug = None

    top, bottom, center_v = pcv.homology.x_axis_pseudolandmarks(
        img=img, mask=mask
    )
    left, right, center_h = pcv.homology.y_axis_pseudolandmarks(
        img=img, mask=mask
    )

    groups = [
        (top, (0, 0, 255)),
        (bottom, (255, 0, 255)),
        (center_v, (255, 0, 0)),
        (left, (0, 255, 255)),
        (right, (255, 255, 0)),
        (center_h, (0, 255, 0)),
    ]
    for points, color in groups:
        for p in points:
            x, y = int(p[0][0]), int(p[0][1])
            cv2.circle(out, (x, y), radius=4, color=color, thickness=-1)
    return out


def analyze_transform(img, mask):
    img_h, img_w = img.shape[:2]
    roi = pcv.roi.rectangle(img=img, x=0, y=0, h=img_h, w=img_w)
    labeled_mask, n = pcv.create_labels(
        mask=mask, rois=roi, roi_type="partial"
    )
    pcv.params.debug = None
    return pcv.analyze.size(img=img, labeled_mask=labeled_mask, n_labels=n)


def roi_transform(img, mask):
    out = img.copy()

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(out, contours, -1, (0, 255, 0), thickness=3)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        pad = 6
        img_h, img_w = img.shape[:2]
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(img_w - 1, x + w + pad)
        y1 = min(img_h - 1, y + h + pad)
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 0, 0), thickness=4)
    return out


def display_transformations(transformations):
    n = len(transformations)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    fig.suptitle("Transformations", fontsize=14)
    axes_flat = np.atleast_1d(axes).flatten()

    for idx, (name, value) in enumerate(transformations.items()):
        ax = axes_flat[idx]
        if callable(value):
            value(ax)
            ax.set_title(name)
        else:
            if value.ndim == 2:
                ax.imshow(value, cmap="gray")
            else:
                ax.imshow(cv2.cvtColor(value, cv2.COLOR_BGR2RGB))
            ax.set_title(name)
            ax.axis("off")

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].axis("off")

    plt.tight_layout()
    plt.show()


def compute_transformations(img, selected):
    mask = build_mask(img)
    all_steps = {
        "blur": ("Gaussian blur",
                 lambda: pcv.gaussian_blur(
                     img=mask, ksize=(5, 5), sigma_x=0)),
        "mask": ("Mask",
                 lambda: pcv.apply_mask(
                     img=img, mask=mask, mask_color="white")),
        "roi": ("ROI objects", lambda: roi_transform(img, mask)),
        "analyze": ("Analyze object", lambda: analyze_transform(img, mask)),
        "landmarks": ("Pseudolandmarks",
                      lambda: pseudolandmarks_transform(img, mask)),
        "hist": ("Color histogram", lambda: color_histogram_draw(img, mask)),
    }

    out = {"Original": img}
    for key in TRANSFORM_KEYS:
        if key in selected:
            name, builder = all_steps[key]
            out[name] = builder()
    return out


def save_histogram(value, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    value(ax)
    ax.set_title("Color histogram")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_transformations(transformations, dst_dir, base_name, ext):
    os.makedirs(dst_dir, exist_ok=True)
    suffix_map = {v: k for k, v in TRANSFORM_KEYS.items()}
    for name, value in transformations.items():
        if name == "Original":
            continue
        key = suffix_map.get(name, name.lower().replace(" ", "_"))
        if callable(value):
            out_path = os.path.join(dst_dir, f"{base_name}_{key}.png")
            save_histogram(value, out_path)
        else:
            out_path = os.path.join(dst_dir, f"{base_name}_{key}{ext}")
            cv2.imwrite(out_path, value)
        print(f"  saved {out_path}")


def run_single(image_path, selected):
    img = load_image(image_path)
    if img is None:
        sys.exit(1)
    transformations = compute_transformations(img, selected)
    display_transformations(transformations)


def run_batch(src_dir, dst_dir, selected):
    if not os.path.isdir(src_dir):
        print(f"Error: {src_dir} is not a directory", file=sys.stderr)
        sys.exit(1)
    found = False
    for root, dirs, files in os.walk(src_dir):
        dirs.sort()
        for fname in sorted(files):
            if not fname.lower().endswith(IMAGE_EXTS):
                continue
            found = True
            src_path = os.path.join(root, fname)
            rel_dir = os.path.relpath(root, src_dir)
            out_dir = dst_dir if rel_dir == "." else os.path.join(
                dst_dir, rel_dir
            )
            print(f"Processing {src_path}")
            img = load_image(src_path)
            if img is None:
                continue
            base_name, ext = os.path.splitext(fname)
            transformations = compute_transformations(img, selected)
            save_transformations(transformations, out_dir, base_name, ext)
    if not found:
        print(f"No images found in {src_dir}")
