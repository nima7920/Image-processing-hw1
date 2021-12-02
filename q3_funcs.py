import numpy as np
import cv2


def warp(img, h, size):
    ''' working inverse : generating destination matrix and calculating its pixels from origin '''
    h_inverse = np.linalg.inv(h)
    result = np.zeros((size[1], size[0], 3), dtype='uint8')
    for x in range(size[1]):
        for y in range(size[0]):
            dest_point = np.matmul(h_inverse, np.array([y, x, 1]))
            x1, y1 = int(dest_point[1]), int(dest_point[0])
            a = dest_point[1] - x1
            b = dest_point[0] - y1
            a_array, b_array = np.array([[1 - a, a]]), np.transpose(np.array([[1 - b, b]]))
            for color in range(3):
                points = np.array([[img[x1, y1, color], img[x1, y1 + 1, color]],
                                   [img[x1 + 1, y1, color], img[x1 + 1, y1 + 1, color]]])
                value = np.matmul(np.matmul(a_array, points), b_array)
                result[x, y, color] = int(value)
    return result


def get_output_points(input_points):
    width = int(np.sqrt(np.sum(np.square(input_points[0] - input_points[1]))))
    height = int(np.sqrt(np.sum(np.square(input_points[0] - input_points[3]))))
    output_points = np.array([[0, 0], [width, 0], [width, height], [0, height]])
    return output_points, width, height
