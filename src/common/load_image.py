import cv2


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img
