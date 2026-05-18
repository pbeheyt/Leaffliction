from plantcv import plantcv as pcv
import numpy as np


DEFAULT_LOWER_GREEN = np.array([20, 25, 20], dtype=np.uint8)
DEFAULT_UPPER_GREEN = np.array([100, 255, 255], dtype=np.uint8)


def build_mask(img):
    gray = pcv.rgb2gray_lab(rgb_img=img, channel="a")
    binary = pcv.threshold.binary(
        gray_img=gray, threshold=120, object_type="dark"
    )
    cleaned = pcv.fill(bin_img=binary, size=50)
    cleaned = pcv.fill_holes(bin_img=cleaned)
    return cleaned
