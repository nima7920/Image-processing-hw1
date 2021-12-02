import matplotlib.pyplot as plt
import numpy as np
import cv2
import q4_funcs

input_path_near = "outputs/res19-near.jpg"
input_path_far = "outputs/res20-far.jpg"

output_paths = np.array([
    "outputs/res21-near.jpg", "outputs/res22-far.jpg", "outputs/res23-dft-near.jpg"
    , "outputs/res24-dft-far.jpg", "outputs/res25-highpass-35.jpg", "outputs/res26-lowpass-10.jpg",
    "outputs/res27-highpassed.jpg", "outputs/res28-lowpassed.jpg", "outputs/res29-hybrid.jpg",
    "outputs/res30-hybrid-near.jpg", "outputs/res31-hybrid-far.jpg"])

img_near = cv2.imread(input_path_near)
img_far = cv2.imread(input_path_far)

img_near = cv2.cvtColor(img_near, cv2.COLOR_BGR2RGB)
img_far = cv2.cvtColor(img_far, cv2.COLOR_BGR2RGB)

''' adjusting images '''
first_points = np.float32([[820, 1445], [1480, 1457], [853, 2305], [1359, 2305]])
second_points = np.float32([[827, 1406], [1601, 1410], [922, 2225], [1500, 2212]])
h = cv2.getPerspectiveTransform(first_points, second_points)
img_far = cv2.warpPerspective(img_far, h, (img_near.shape[1], img_near.shape[0]))

# cutting 20% from bottom
new_height = int(0.8 * img_far.shape[0])
img_far = img_far[0:new_height, :, :]
img_near = img_near[0:new_height, :, :]
img_near_result = cv2.cvtColor(img_near, cv2.COLOR_RGB2BGR)
img_far_result = cv2.cvtColor(img_far, cv2.COLOR_RGB2BGR)
cv2.imwrite(output_paths[0], img_near_result)
cv2.imwrite(output_paths[1], img_far_result)

'''  ######## taking images to frequency domain ########## '''

img_near_fft = np.fft.fft2(img_near[:, :, 2])
img_near_fft = np.fft.fft2(img_near, axes=(0, 1))
img_near_shifted = np.fft.fftshift(img_near_fft)
img_near_fft_log = np.log(np.abs(img_near_shifted))
img_near_fft_log = q4_funcs.convert_from_float32_to_uint8(img_near_fft_log)
img_near_fft_log = cv2.cvtColor(img_near_fft_log, cv2.COLOR_RGB2BGR)
cv2.imwrite(output_paths[2], img_near_fft_log)

img_far_fft = np.fft.fft2(img_far, axes=(0, 1))
img_far_shifted = np.fft.fftshift(img_far_fft)
img_far_fft_log = np.log(np.abs(img_far_shifted))
img_far_fft_log = q4_funcs.convert_from_float32_to_uint8(img_far_fft_log)
img_far_fft_log = cv2.cvtColor(img_far_fft_log, cv2.COLOR_RGB2BGR)
cv2.imwrite(output_paths[3], img_far_fft_log)

''' ######## constructing low-pass and high-pass filters ##########'''
s, r = 10, 35
low_pass = q4_funcs.gaussian_low_pass_filter(img_far_fft.shape, s)
high_pass = q4_funcs.gaussian_highpass_filter(img_near_shifted.shape, r)

plt.imsave(output_paths[4], high_pass)
plt.imsave(output_paths[5], low_pass)

''' ####### applying high-pass and low-pass filters ########'''
img_near_highpass = img_near_shifted * high_pass
img_near_highpass = np.fft.ifftshift(img_near_highpass)

# saving
img_near_highpass_result = np.fft.ifft2(img_near_highpass, axes=(0, 1))
img_near_highpass_result = np.real(img_near_highpass_result)
img_near_highpass_result = q4_funcs.convert_from_float32_to_uint8(img_near_highpass_result)
plt.imsave(output_paths[6], img_near_highpass_result)


img_far_lowpass = img_far_shifted * low_pass
img_far_lowpass = np.fft.ifftshift(img_far_lowpass)

# saving
img_far_lowpass_result = np.fft.ifft2(img_far_lowpass, axes=(0, 1))
img_far_lowpass_result = np.real(img_far_lowpass_result)
img_far_lowpass_result = q4_funcs.convert_from_float32_to_uint8(img_far_lowpass_result)

plt.imsave(output_paths[7], img_far_lowpass_result)

''' ###### combining resulting images ######## '''
result = (img_far_lowpass + img_near_highpass) / 2

# saving
result2 = np.log(np.abs(result))
result2 = q4_funcs.convert_from_float32_to_uint8(result2)
plt.imsave(output_paths[8], result2)

''' taking result into spatial domain '''
result = np.fft.ifft2(result, axes=(0, 1))
result = np.real(result)
result = q4_funcs.convert_from_float32_to_uint8(result)

result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
cv2.imwrite(output_paths[9], result)

''' smaller result '''
result = cv2.resize(result, (int(result.shape[1] / 15), int(result.shape[0] / 15)))
cv2.imwrite(output_paths[10], result)
