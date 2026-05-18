import cv2
import numpy as np


DEFAULT_LOWER_GREEN = np.array([20, 25, 20], dtype=np.uint8)
DEFAULT_UPPER_GREEN = np.array([100, 255, 255], dtype=np.uint8)


def build_leaf_mask(image_bgr, lower_green, upper_green, kernel_size=5, hsv=None):
    if hsv is None:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask
