import numpy as np
import cv2
import matplotlib.pyplot as plt


def gaussian_kernel(n, sigma):

    '''
    Create Gaussian kernel
    :param n: size of the kernel
    :param sigma: sigma of Gaussian
    :return: 2D kernel matrix
    '''
    kernel_1D = np.linspace(-(n//2), n//2, n)
    for i in range(n):
        kernel_1D[i] = normalize(kernel_1D[i], sigma, 0)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
    # ensure the central value is always 0
    kernel_2D /= kernel_2D.max()
    return kernel_2D


def normalize(x, sigma, mu):
    '''
    normalize the kernel using Univariate Normal Distribution
    :param x: object of calculation
    :param sigma: sigma of Gaussian
    :param mu: mean value
    :return: gaussian value
    '''
    g = 1/(sigma * np.sqrt(2*np.pi)) * np.e**(-0.5*((x-mu)/sigma)**2)
    return g


def convolution(image, kernel):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    row, col = image.shape
    pad_size = kernel.shape[0] // 2
    output = np.zeros(image.shape)
    pad_image = np.zeros((row+2*pad_size, col+2*pad_size))
    pad_image[pad_size:pad_image.shape[0]-pad_size, pad_size:pad_image.shape[1]-pad_size] = image
    for r in range(row):
        for c in range(col):
            output[r, c] = np.sum(kernel*pad_image[r:r+kernel.shape[0], c:c+kernel.shape[0]])
    return output


kernel_size = 7
sigma = np.sqrt(kernel_size)
img = cv2.imread('C:/Users/Trista/PycharmProjects/Computer Vision Projects/segmentation_project/images/bunny.bmp')
kernel = gaussian_kernel(kernel_size, sigma)
img_blurred = convolution(img, kernel)
plt.imshow(img_blurred)

