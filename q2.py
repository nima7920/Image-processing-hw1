import cv2
import numpy as np

import q2_funcs

img_input_path = "inputs/Greek-ship.jpg"
template_input_path = "inputs/patch.png"
output_path = "outputs/res15.jpg"
''' reading images '''
img = cv2.imread(img_input_path)
template = cv2.imread(template_input_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)

''' gray image '''
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
gray_template = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)

''' template pyramid '''
division_ratio = 0.8
dim1 = (int(template.shape[1] * division_ratio), int(template.shape[0] * division_ratio))
dim2 = (int(dim1[0] * division_ratio), int(dim1[1] * division_ratio))
gray_template2 = cv2.resize(gray_template, dim1)
gray_template3 = cv2.resize(gray_template, dim2)

dim_narrow = (int(template.shape[1] * 0.5), int(template.shape[0] * 0.8))
template_narrow = cv2.resize(gray_template, dim_narrow)

# dim_narrow_tall = (int(template.shape[1] * 1.2), int(template.shape[0]))
# template_narrow_tall = cv2.resize(gray_template, dim_narrow_tall)

''' resizing '''
ratio = 8
img_resized = gray_img[::ratio, ::ratio]
template_resized = gray_template[::ratio, ::ratio]
template_resized2 = gray_template2[::ratio, ::ratio]
template_resized3 = gray_template3[::ratio, ::ratio]
template_narrow = template_narrow[::ratio, ::ratio]
# template_narrow_tall = template_narrow_tall[::ratio, ::ratio]

''' taking images to floating points '''
img_float = np.float32(img_resized / 255)
template_float = np.float32(template_resized / 255)
template_float2 = np.float32(template_resized2 / 255)
template_float3 = np.float32(template_resized3 / 255)
template_narrow_float = np.float32(template_narrow / 255)
# template_narrow__tall_float = np.float32(template_narrow_tall / 255)

ncc1 = q2_funcs.find_template_ncc(img_float, template_float)
ncc2 = q2_funcs.find_template_ncc(img_float, template_float2)
ncc3 = q2_funcs.find_template_ncc(img_float, template_float3)
ncc4 = q2_funcs.find_template_ncc(img_float, template_narrow_float)
# ncc5 = q2_funcs.find_template_ncc(img_float, template_narrow__tall_float)

threshold = np.zeros(img_float.shape)
threshold[ncc1] = 1
threshold[ncc2] = 1
threshold[ncc3] = 1
threshold[ncc4] = 1
# threshold[ncc5] = 1

threshold = cv2.resize(threshold, (threshold.shape[1] * ratio, threshold.shape[0] * ratio))
mask = np.where(threshold == 1)
q2_funcs.draw_box(img, mask, (template.shape[1], template.shape[0]))
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite(output_path, img)
