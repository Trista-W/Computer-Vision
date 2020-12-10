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


def l_o_g(x, y, sigma):
    '''
    Calculate the LoG at a given point and with a give para
    :param x: position x
    :param y: position y
    :param sigma: sigma of gaussian
    :return: LoG
    '''
    norm = x**2 + y**2 - 2*(sigma**2)
    denorm = 2 * np.pi * sigma**6
    expo = np.exp(-(x**2 + y**2) / 2*(sigma**2))
    return norm*expo/denorm


def log_kernel(sigma, size):
    w = int(np.ceil(size * float(sigma)))
    if w % 2 == 0:
        w += 1
    mask = []
    width = int(w // 2)
    for i in range(-width, width+1):
        for j in range(-width, width+1):
            mask.append(l_o_g(i, j, sigma))
    mask = np.array(mask)
    mask = np.reshape(mask, (w, w))
    return mask


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


def zero_crossing(image):
    row, col = image.shape
    pad_image = np.zeros((row+2, col+2))
    pad_image[1:row+1, 1:col+1] = image
    output = np.zeros(pad_image.shape)
    for r in range(1, row+1):
        for c in range(1, col+1):
            neg_count = 0
            pos_count = 0
            # A zero-crossing is represented by two neighboring pixels that change from positive to negative.
            # Of the two pixels, the one closest to zero is used to represent the zero-crossing.
            for i in range(-1, 2, 2):
                for j in range(-1, 2, 2):
                    print(pad_image[r+i, c+j])
                    if pad_image[r+i, c+j] < 0:
                        neg_count += 1
                    elif pad_image[r+i, c+j] >= 0:
                        pos_count += 1
            if pos_count > 0 and neg_count > 0:
                output[r, c] = 1
    return output


kernel_size = 7
sigma = np.sqrt(kernel_size)
img = cv2.imread('images/hand.bmp')
kernel = gaussian_kernel(kernel_size, sigma)
img_blurred = convolution(img, kernel)
log = log_kernel(sigma, kernel_size)
img_log = convolution(img_blurred, log)
img_log = zero_crossing(img_log)
plt.imshow(img_blurred)
