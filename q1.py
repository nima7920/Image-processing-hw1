import numpy as np
import cv2
import matplotlib.pyplot as plt
import q1_funcs

''' defining paths '''
input_path = "inputs/flowers.blur.png"
output_1_path = "outputs/res01.jpg"
output_2_path = "outputs/res02.jpg"
output_3_path = "outputs/res03.jpg"
output_4_path = "outputs/res04.jpg"
output_5_path = "outputs/res05.jpg"
output_6_path = "outputs/res06.jpg"
output_7_path = "outputs/res07.jpg"
output_8_path = "outputs/res08.jpg"
output_9_path = "outputs/res09.jpg"
output_10_path = "outputs/res10.jpg"
output_11_path = "outputs/res11.jpg"
output_12_path = "outputs/res12.jpg"
output_13_path = "outputs/res13.jpg"
output_14_path = "outputs/res14.jpg"

''' opening image '''
img = cv2.imread(input_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_float = np.float32(img / 255)

''' defining kernel '''
sigma, kernel_size = 3, 7
alpha = 0.5
kernel = q1_funcs.gaussianKernel(sigma, kernel_size)
kernel_img = q1_funcs.kernel_img(kernel)
plt.imsave(output_1_path, kernel_img)

''' method 1 '''
img_blurred = cv2.filter2D(img_float, -1, kernel)  ## f*g : convolving image with kernel
# saving result
img_blurred_result = q1_funcs.convert_from_float32_to_uint8(img_blurred)
img_blurred_result = cv2.cvtColor(img_blurred_result, cv2.COLOR_RGB2BGR)
cv2.imwrite(output_2_path, img_blurred_result)

unsharp_mask = np.subtract(img_float, img_blurred)
# saving result
unsharp_mask_result = q1_funcs.convert_from_float32_to_uint8(unsharp_mask)
unsharp_mask_result = cv2.cvtColor(unsharp_mask_result, cv2.COLOR_RGB2BGR)
cv2.imwrite(output_3_path, unsharp_mask_result)

img_sharpened = np.add(img_float, alpha * unsharp_mask)
# saving result
img_sharpened_result = q1_funcs.convert_from_float32_to_uint8(img_sharpened)
img_sharpened_result = cv2.cvtColor(img_sharpened_result, cv2.COLOR_RGB2BGR)
cv2.imwrite(output_4_path, img_sharpened_result)

''' method 2 '''
''' laplacian kernel '''
k = 0.5
laplacian_kernel = q1_funcs.laplacian(kernel)
# saving result
laplacian_kernel_result = q1_funcs.kernel_img(laplacian_kernel)
plt.imsave(output_5_path, laplacian_kernel_result)

unsharp_mask = cv2.filter2D(img_float, -1, laplacian_kernel)
# saving result
unsharp_mask_result = q1_funcs.convert_from_float32_to_uint8(unsharp_mask)
unsharp_mask_result = cv2.cvtColor(unsharp_mask_result, cv2.COLOR_RGB2BGR)
cv2.imwrite(output_6_path, unsharp_mask_result)

img_sharpened = img_float - k * unsharp_mask
# saving result
img_sharpened_result = q1_funcs.convert_from_float32_to_uint8(img_sharpened)
img_sharpened_result = cv2.cvtColor(img_sharpened_result, cv2.COLOR_RGB2BGR)
cv2.imwrite(output_7_path, img_sharpened_result)

''' method 3 '''
k = 0.4
# taking fourier transform
img_fourier = np.fft.fft2(img_float, axes=(0, 1))
img_fourier = np.fft.fftshift(img_fourier)
# saving fouerier image
img_fourier_result = np.log(np.abs(img_fourier))
img_fourier_result = q1_funcs.convert_from_float32_to_uint8(img_fourier_result)
img_fourier_result = cv2.cvtColor(img_fourier_result, cv2.COLOR_RGB2BGR)
cv2.imwrite(output_8_path, img_fourier_result)

''' high pass filter '''
highpass = q1_funcs.ideal_highpass_filter((img.shape[0], img.shape[1]), 100)
# saving
highpass_result = q1_funcs.convert_from_float32_to_uint8(highpass)
cv2.imwrite(output_9_path, highpass_result)

img_sharpened_fourier = q1_funcs.apply_fourier_mul(img_fourier, 1 + k * highpass)
# saving
img_sharpened_fourier_result = np.log(np.abs(img_sharpened_fourier))
img_sharpened_fourier_result = q1_funcs.convert_from_float32_to_uint8(img_sharpened_fourier_result)
cv2.imwrite(output_10_path, img_sharpened_fourier_result)

''' final result '''
result3 = np.fft.ifftshift(img_sharpened_fourier)
result3 = np.fft.ifft2(result3, axes=(0, 1))
result3 = np.real(result3)
result3 = q1_funcs.convert_from_float32_to_uint8(result3)
result3 = cv2.cvtColor(result3, cv2.COLOR_RGB2BGR)
cv2.imwrite(output_11_path, result3)

'''  
  method 4
'''
k = 0.00001
img_laplacian = q1_funcs.apply_laplacian_fourier(img_fourier)
# saving
img_laplacian_result = np.abs(img_laplacian)
img_laplacian_result = q1_funcs.convert_from_float32_to_uint8(img_laplacian_result)
img_laplacian_result = cv2.cvtColor(img_laplacian_result, cv2.COLOR_RGB2BGR)
cv2.imwrite(output_12_path, img_laplacian_result)

img_laplacian_inverse = np.fft.ifft2(img_laplacian, axes=(0, 1))
img_laplacian_inverse = np.real(img_laplacian_inverse)
# saving
img_laplacian_inverse_result = q1_funcs.convert_from_float32_to_uint8(img_laplacian_inverse)
img_laplacian_inverse_result = cv2.cvtColor(img_laplacian_inverse_result, cv2.COLOR_RGB2BGR)
cv2.imwrite(output_13_path, img_laplacian_inverse_result)

''' final result '''
result4 = img + k * img_laplacian_inverse
result4 = q1_funcs.convert_from_float32_to_uint8(result4)
result4 = cv2.cvtColor(result4, cv2.COLOR_RGB2BGR)
cv2.imwrite(output_14_path, result4)
