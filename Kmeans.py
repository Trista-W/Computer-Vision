import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread('C:/Users/Trista/PycharmProjects/Computer Vision Projects/segmentation_project/images/bunny.bmp')
# cv2.imshow('image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
pixels = image.reshape((-1, 3))
pixels = pixels / 255


def init_centroids(X, k):
    m, n = X.shape
    centroids = np.zeros((k, n))
    idx = np.random.randint(0, m, k)
    for i in range(k):
        centroids[i, :] = X[idx[i], :]
    return centroids


def find_closest_centroids(X, centroids):
    m = X.shape[0]
    k = centroids.shape[0]
    idx = np.zeros(m)

    for i in range(m):
        min_dist = 1000000
        for j in range(k):
            dist = np.sum((X[i, :] - centroids[j, :])**2)
            if dist < min_dist:
                min_dist = dist
                idx[i] = j
    return idx


def compute_centroids(X, idx, k):
    m, n = X.shape
    centroids = np.zeros((k, n))
    for i in range(k):
        indices = np.where(idx == i)
        centroids[i, :] = (np.sum(X[indices, :], axis=1) / len(indices[0])).ravel()
    return centroids


def k_means(X, init_centroids, max_iters):
    m, n = X.shape
    k = init_centroids.shape[0]
    idx = np.zeros(m)
    centroids = init_centroids
    for i in range(max_iters):
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, k)

    return idx, centroids


def kernel_kmeans(X, sigma):
    m = X.shape[0]
    k = [[0] * m for i in range(m)]
    for i in range(m):
        for j in range(i, m):
            if i != j:
                # RBF kernel: K(xi,xj) = e ( (-|xi-xj|**2) / (2sigma**2)
                dist = np.sum((X[i, :] - X[j, :])**2)
                k[i][j] = np.exp(-dist / (2 * sigma ** 2))
                k[j][i] = k[i][j]
    return k


init = init_centroids(pixels, 6)
idx, centroids = k_means(pixels, init, 10)

idx1 = find_closest_centroids(pixels, centroids)
pixels_recovered = centroids[idx.astype(int), :]
pixels_recovered = np.reshape(pixels_recovered, (image.shape[0], image.shape[1], image.shape[2]))
plt.imshow(pixels_recovered)
