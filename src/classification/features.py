import os

import cv2
import numpy as np

from src.data_transformation.transformation import color_histogram_draw
from src.common.mask import (
    DEFAULT_LOWER_GREEN,
    DEFAULT_UPPER_GREEN,
    build_mask,
)


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

DEFAULT_IMAGE_SIZE = 224


LOWER_GREEN = DEFAULT_LOWER_GREEN
UPPER_GREEN = DEFAULT_UPPER_GREEN


def extract_features(image_path):
    image_bgr = cv2.imread(image_path)

    image_bgr = cv2.resize(
        image_bgr,
        (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    mask = build_mask(image_bgr)
    hsv_channels = cv2.split(hsv)

    color_stats = []
    for channel in hsv_channels:
        color_stats.extend([float(np.mean(channel)), float(np.std(channel))])

    hist_features = []
    factory = color_histogram_draw(
        image_bgr, mask, is_extracting_features=True)
    hist_features.extend(factory(None))

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    area_ratio = float(np.count_nonzero(mask)) / float(mask.size)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter > 0:
            compactness = float((4.0 * np.pi * area) / (perimeter * perimeter))
        else:
            compactness = 0.0
    else:
        area = 0.0
        perimeter = 0.0
        compactness = 0.0

    features = np.array(
        color_stats
        + hist_features
        + [lap_var, area_ratio, area, perimeter, compactness],
        dtype=np.float32,
    )
    return features


def iter_labeled_images(dataset_dir):
    for class_name in sorted(os.listdir(dataset_dir)):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        for filename in sorted(os.listdir(class_dir)):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in VALID_EXTENSIONS:
                continue
            yield os.path.join(class_dir, filename), class_name


def load_dataset(dataset_dir, log_every=200):
    samples = list(iter_labeled_images(dataset_dir))
    total = len(samples)
    if total == 0:
        raise ValueError("No valid images found in dataset.")

    print(
        f"[load_dataset] start: found {total} image(s) in {dataset_dir}",
        flush=True,
    )

    features = []
    labels = []
    processed = 0

    for image_path, class_name in samples:
        try:
            vector = extract_features(image_path)
        except ValueError as error:
            print(f"[load_dataset] skip: {error}", flush=True)
            continue

        features.append(vector)
        labels.append(class_name)
        processed += 1

        if processed % log_every == 0 or processed == total:
            print(f"[load_dataset] progress: {processed}/{total}", flush=True)

    if not features:
        raise ValueError(
            "No valid images found in dataset after feature extraction.")

    print(f"[load_dataset] done: kept {len(features)} sample(s)", flush=True)

    return np.vstack(features), np.array(labels)
