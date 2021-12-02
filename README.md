## Image processing homework 2
My codes for second homework of Image processing course, Sharif university of technology , fall 2021

For each question , there is a file with the main code of the problem , and a funcs file containing the functions used in main code.

### Question 1 : Image sharpening

Given an image, sharp it using each of the following four methods :

1- Subtracting an unsharp mask achieved from subtracting image and it's gaussian filtered from the original image

2- method 1 , with gaussian filter replaced with laplacian of gauss filter

3- applying a highpass filter in frequency domain 

4- applying laplacian filter in frequency domain

### Question 2: template matching 

Given an image (Greek-ship image ) and a template (patch image) find all instances of template in the image and draw a box around them.

Template matching is done using ncc method. To find all matchings ( with different sizes )template is resized with four different ratios and ncc is applies on all these copies.

### Question 3: Homography and Image Warping

Given an image with three books inside it. Output 3 different images each being one of these books

### Question 4: Hybrid Images

Given two images , generate a hybrid image from them.



