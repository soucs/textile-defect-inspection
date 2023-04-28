import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv
import h5py
import pywt

def resize_img(img):
    dim = (500, 500)
    return cv.resize(img,dim,interpolation=cv.INTER_AREA)
def show(imgs):
    for i, img in enumerate(imgs):
        cv.imshow('img'+str(i),img)
    cv.waitKey(0)
    cv.destroyAllWindows()

# label_dict = {'good':0, 'holes_cuts':1, 'threaderror':2, 'oilstains_colorerror':3, 'wrinkles':4, 'foreignbodies':5}
file_path = r'/home/soucs/Python/textile-defect-inspection/dataset/textile_defect_data.hdF5'
imgs = h5py.File(file_path)['jute_defect_imgs'][:]
labels = h5py.File(file_path)['jute_defect_labels'][:]

img = imgs[90]
# img = resize_img(cv.blur(img,(5,5)))

# # Apply Canny edge detector
# edges = cv.Canny(img, 100, 200)
# kernel = np.ones((3, 3), np.uint8) # Create 3x3 kernel for dilation
# dilated = cv.dilate(edges, kernel, iterations=2) # Dilate edges using kernel

# # Define Gabor filter parameters
# ksize = 10  # Kernel size
# sigma = 4  # Standard deviation of the Gaussian envelope
# theta = 0  # Orientation of the Gabor filter
# lambd = 10  # Wavelength of the sinusoidal function
# gamma = 0.5  # Spatial aspect ratio
# psi = 0  # Phase offset of the sinusoidal function
# # Create Gabor filter
# kernel = cv.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi)
# # Apply Gabor filter to image
# texture = cv.filter2D(dilated, cv.CV_8UC3, kernel)
# show([img,edges,dilated, texture])

# sift = cv.SIFT_create()
# kp = sift.detect(img, None)
# sift_img = cv.drawKeypoints(img, kp, 0, (0, 255, 0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# show([img,sift_img])

def coMat(img):
    # Define the number of gray levels
    levels = 256

    # Define the distance and angle offsets for co-occurrence
    d = 1
    theta = 0

    # Compute the co-occurrence matrix
    co_mat = np.zeros((levels, levels), dtype=np.uint32)
    for i in range(d, img.shape[0]):
        for j in range(d, img.shape[1]):
            i_index = img[i, j]
            j_index = img[i-d, j+theta]
            co_mat[i_index, j_index] += 1

    # Normalize the co-occurrence matrix
    co_mat = co_mat.astype(np.float64)
    co_mat /= np.sum(co_mat)

    # Print the co-occurrence matrix
    return co_mat.max()



# Compute the Laplacian
# log = cv.Laplacian(img, cv.CV_64F)
# show([img,log])

# Perform 2D wavelet transform using Haar wavelet
coeffs = pywt.dwt2(img, 'haar')

# Split the coefficients into sub-bands
cA, (cH, cV, cD) = coeffs


# Define the kernel
# kernel = img[:50,:50]
# # kernel_img  = np.tile(kernel, (10, 10))
# # result = img-kernel_img

# max_li = set()
# for i in range(0,450,10):
#     for j in range(0,450,10):
#         kernel = img[i:50+i,j:50+j]
#         max_li.add(coMat(kernel))
#         print(i,j)
# print(max_li)


# # Calculate histogram
# hist0 = cv.calcHist([kernel],[0],None,[256],[0,256])
# hist1 = cv.calcHist([img[150:200,350:400]],[0],None,[256],[0,256])

# show([img])
# # Plot histogram
# plt.plot(hist0)
# plt.plot(hist1)
# plt.xlim([0, 256])
# plt.xlabel('Gray level')
# plt.ylabel('Frequency')
# plt.show()
# # show([img,kernel,result])

# # calculate correlation coefficient between the two distributions
# corr_coef = np.corrcoef(hist0.reshape(-1), hist1.reshape(-1))[0, 1]
# print(corr_coef)



# Display the resulting sub-bands
cv.imshow('Img', img)
cv.imshow('Approximation (low-pass)', cA)
cv.imshow('Horizontal detail (high-pass)', cH)
cv.imshow('Vertical detail (high-pass)', cV)
cv.imshow('Diagonal detail (high-pass)', cD)
cv.waitKey(0)