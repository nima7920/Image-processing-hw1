import numpy as np
import cv2
import q3_funcs

input_path = "inputs/books.jpg"
output_paths = ["outputs/res16.jpg", "outputs/res17.jpg", "outputs/res18.jpg"]

img = cv2.imread(input_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

''' yellow book '''
input_points = np.array([[665, 210], [600, 400], [315, 295], [380, 105]])
output_points, width, height = q3_funcs.get_output_points(input_points)
''' finding homography '''
h, status = cv2.findHomography(input_points, output_points)
print(h)
result_yellow = q3_funcs.warp(img, h, (width, height))
result_yellow = cv2.cvtColor(result_yellow, cv2.COLOR_RGB2BGR)
cv2.imwrite(output_paths[0], result_yellow)

''' black book '''
input_points = np.array([[812, 969], [610, 1099], [419, 796], [621, 666]])
output_points, width, height = q3_funcs.get_output_points(input_points)
''' finding homography '''
h, status = cv2.findHomography(input_points, output_points)
print(h)

result_black = q3_funcs.warp(img, h, (width, height))
result_black = cv2.cvtColor(result_black, cv2.COLOR_RGB2BGR)
cv2.imwrite(output_paths[1], result_black)

''' black and white book '''
input_points = np.array([[362, 744], [153, 710], [204, 427], [410, 464]])
output_points, width, height = q3_funcs.get_output_points(input_points)
''' finding homography '''
h, status = cv2.findHomography(input_points, output_points)
print(h)

result_black_white = q3_funcs.warp(img, h, (width, height))
result_black_white = cv2.cvtColor(result_black_white, cv2.COLOR_RGB2BGR)
cv2.imwrite(output_paths[2], result_black_white)
