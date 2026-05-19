import cv2

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img
