import cv2
import numpy as np
from scipy import signal


def find_ncc(img, template):
    ''' image '''
    kernel = np.ones(template.shape, dtype=np.float32)
    kernel = kernel / np.sum(kernel)
    img_mean = signal.correlate2d(img, kernel, mode='same')  # f*I
    img_sqr_mean = signal.correlate2d(img * img, kernel, mode='same')  # f^2 * I

    ''' template '''
    template_normal = (template - np.mean(template)) / np.sqrt(np.sum(np.square(template - np.mean(template))))
    ''' final result '''
    result = signal.correlate2d(img, template_normal, mode='same') - img_mean * np.sum(template_normal)
    img_demo = np.sqrt(template.size * (img_sqr_mean - img_mean * img_mean))
    result = result / img_demo
    return result


def find_template_ncc(img, template):
    ncc = find_ncc(img, template)
    mask = np.where(ncc > 0.4)
    return mask


def draw_box(img, mask, template_size):
    x, y = template_size[0], template_size[1]
    mask_x = np.copy(mask[1])
    mask_x = np.sort(mask_x)
    j = 0
    for i in mask_x:
        if j + 15 < i:
            indices = np.where(mask[1] == i)[0]
            if indices.size > 10:
                cv2.rectangle(img, (i - 100, mask[0][indices[0]] - 100), (i + x, mask[0][indices[0]] + y), (0, 0, 255),
                              2)
            j = i
