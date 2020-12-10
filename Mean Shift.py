import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth


image = cv2.imread('images/bunny.bmp')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
pixels = image.reshape((-1, 3))
pixels = pixels / 255
rows, cols, dim = image.shape
meandist = np.array([[1000.0 for r in range(cols)] for c in range(rows)])
labels = np.array([[-1 for r in range(cols)] for c in range(rows)])
m, S, threshold = 1, 5, 1.0
bandwidth = 0.25
seg_img = image


def meanShift(img, radius):
    rows, cols, dim = img.shape
    means = []
    for r in range(0, rows, radius):
        for c in range(0, cols, radius):
            centroid = np.array([r, c, img[r][c][0], img[r][c][1], img[r][c][2]])
            x, y = centroid[0], centroid[1]
            r1 = max(0, x-radius)
            r2 = min(r1+2*radius, rows)
            c1 = max(0, y-radius)
            c2 = min(c1+2*radius, cols)
            kernel = []
            for n in range(15):
                for i in range(r1, r2):
                    for j in range(c1, c2):
                        dc = np.linalg.norm([img[i][j], centroid[2:]])
                        ds = (np.linalg.norm([np.array([i, j]) - centroid[:2]])) * m/S
                        D = np.linalg.norm([dc, ds])
                        if D < bandwidth:
                            kernel.append([i, j, img[i][j][0], img[i][j][1], img[i][j][2]])
                if kernel:
                    kernel = np.array([kernel])
                    mean = np.mean(kernel, axis=0, dtype=np.int64)
                    # update window
                    dc = np.linalg.norm([centroid[2:] - mean[2:]])
                    ds = (np.linalg.norm([centroid[:2] - mean[:2]])) * m / S
                    dsm = np.linalg.norm([dc, ds])
                    centroid = mean
                    if dsm <= threshold:
                        break
            means.append(centroid)
    return means


def groupMeans(means):
    flags = [1 for mean in means]
    n = len(means)
    for i in range(n):
        if flags[i] == 1:
            w = 1.0
            j = i + 1
            while j < n:
                dc = np.linalg.norm([means[i][2:] - means[j][2:]])
                ds = (np.linalg.norm([means[i][:2] - means[j][:2]])) * m / S
                dsm = np.linalg.norm([dc, ds])
                if dsm < bandwidth:
                    means[i] += means[j]
                    w += 1.0
                    flags[j] = 0
                j += 1
            means[i] = means[i] / w
    converged_means = []
    for i in range(n):
        if flags[i] == 1:
            converged_means.append(means[i])
    converged_means = np.array([converged_means])
    return converged_means


def constructImage(means, img):
    rows, cols, dim = img.shape
    for r in range(rows):
        for c in range(cols):
            for j in range(len(means)):
                dc = np.linalg.norm([img[r][c] - means[j][2:]])
                ds = (np.linalg.norm([np.array([r, c]) - means[j][:2]])) * m / S
                D = np.linalg.norm([dc, ds])
                if D < meandist[r][c]:
                    meandist[r][c] = D
                    labels[r][c] = j
            seg_img[r][c] = means[labels[r][c]][2:]
    return seg_img


means = meanShift(image, 40)
converged_means = groupMeans(means)
num_centroids = converged_means.shape[1]
converged_means = np.array(converged_means)
converged_means = np.reshape(converged_means, (num_centroids, 5))
reconstructed_img = constructImage(converged_means, image)
plt.imshow(reconstructed_img)


# sklearn
# bandwidth = estimate_bandwidth(pixels, quantile=0.2, n_samples=500)
# ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
# ms.fit(pixels)
# labels = ms.labels_
# cluster_centers = ms.cluster_centers_
#
# labels_unique = np.unique(labels)
# n_clusters_ = len(labels_unique)
# pixels_recovered = cluster_centers[labels.astype(int), :]
# pixels_recovered = np.reshape(pixels_recovered, (image.shape[0], image.shape[1], image.shape[2]))
# plt.imshow(pixels_recovered)
