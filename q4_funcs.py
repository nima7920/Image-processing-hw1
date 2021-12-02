import math
import cv2
import matplotlib.pyplot as plt
import numpy as np


def convert_from_float32_to_uint8(img):
    max_index, min_index = np.max(img), np.min(img)
    a = 255 / (max_index - min_index)
    b = 255 - a * max_index
    result = (a * img + b).astype(np.uint8)
    return result


def gaussian_low_pass_filter(shape, sigma):
    low_pass = np.zeros(shape)
    x0, y0 = int(shape[0] / 2), int(shape[1] / 2)
    for i in range(shape[0]):
        for j in range(shape[1]):
            value = np.exp(-((i - x0) ** 2 + (j - y0) ** 2) / (2 * sigma ** 2))
            low_pass[i, j] = value
    return low_pass


def gaussian_highpass_filter(shape, sigma):
    return 1 - gaussian_low_pass_filter(shape, sigma)

