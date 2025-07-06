from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.ndimage import convolve
import numpy as np
import cv2

def get_gmm_from_labels(X, Y, g):
    """
    Fit GMMs to image regions defined by X. Works for both grayscale and RGB.
    """
    

    h, w = X.shape
    X_flat = X.flatten()
    # Don't reshape if already in (65536, 3) format
    if Y.shape[0] == h * w:
        Y_flat = Y
    else:
        if Y.ndim == 2:
            Y_flat = Y.reshape(h * w, 1)
        elif Y.ndim == 3 and Y.shape[2] == 3:
            Y_flat = Y.reshape(h * w, 3)
        else:
            raise ValueError("Unexpected Y shape: " + str(Y.shape))

    
    k = np.max(X) + 1
    GMMs = []
    for i in range(k):
        mask = X_flat == i
        pixels = Y_flat[mask, :]  # Correct boolean mask over rows
        if pixels.shape[0] == 0:
            raise ValueError(f"No pixels found for label {i}")
        gmm = GaussianMixture(n_components=g, covariance_type='full', reg_covar=1e-3)
        gmm.fit(pixels)
        GMMs.append(gmm)

    return GMMs


def mrf_map(X, Y, GMMs, k, g, MAP_iter, beta):
    m, n = X.shape
    c = Y.shape[2] if Y.ndim == 3 else 1
    U = np.zeros((k, m, n))

    for label in range(k):
        pdf = GMMs[label].score_samples(Y.reshape(-1, c))
        U[label] = -pdf.reshape(m, n)

    for _ in range(MAP_iter):
        for label in range(k):
            mask = (X == label).astype(int)
            penalty = convolve(mask, np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]), mode='constant')
            U[label] += beta * penalty

        X = np.argmin(U, axis=0)

    total_U = np.sum(np.min(U, axis=0))
    return X, total_U

def hmrf_em(X, Y, GMMs, k, g, EM_iter, MAP_iter, beta):
    Y_flat = Y.reshape(-1, Y.shape[2]) if Y.ndim == 3 else Y.reshape(-1, 1)
    sum_U = []
    for it in range(EM_iter):
        X, total_U = mrf_map(X, Y, GMMs, k, g, MAP_iter, beta)
        GMMs = get_gmm_from_labels(X, Y_flat, g)
        sum_U.append(total_U)
        if it >= 2 and np.std(sum_U[-3:]) < 0.01:
            break
    return X, Y, GMMs

def initialize_kmeans_gmm(image, k=2, g=2):
    """
    Performs K-means clustering and GMM initialization on a 3-channel image.
    Automatically converts grayscale to 3-channel.
    """
    if image.ndim == 2:
        image_rgb = np.stack([image] * 3, axis=-1)
    else:
        image_rgb = image.copy()

    h, w, _ = image_rgb.shape
    flat_rgb = image_rgb.reshape(-1, 3)

    km = KMeans(n_clusters=k, random_state=0).fit(flat_rgb)
    X = km.labels_.reshape(h, w)

    GMMs = []
    for i in range(k):
        indices = np.where(X == i)
        pixels = flat_rgb[X.flatten() == i]
        gmm = GaussianMixture(n_components=g, covariance_type='full', reg_covar=1e-3)
        gmm.fit(pixels)
        GMMs.append(gmm)

    return X, image_rgb, GMMs

def segment_with_hmrf(image, k=2, g=2, beta=1, EM_iter=10, MAP_iter=10):
    """
    Full segmentation pipeline: KMeans init + HMRF EM refinement.
    Works on grayscale or RGB input.
    """
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    X_init, Y_rgb, GMMs = initialize_kmeans_gmm(image, k, g)
    X_final, _, _ = hmrf_em(X_init, Y_rgb, GMMs, k, g, EM_iter, MAP_iter, beta)
    segmented = (X_final * (255 // (k - 1))).astype(np.uint8)
    return segmented
'''
def getbinary(a, X, centroid=180):
    """
    Replicates MATLAB's getbinary().
    Converts grayscale 'a' and sets pixels in 'X' to white if > centroid.
    """
    if a.ndim == 3:
        a_gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    else:
        a_gray = a.copy()

    binary_mask = a_gray > centroid

    if X.ndim == 2:
        X_color = np.stack([X]*3, axis=-1)
    else:
        X_color = X.copy()

    X_color[binary_mask] = [255, 255, 255]
    return X_color


def getbinary(image, label_map, centroid=180):
    """
    Overlays white (255,255,255) onto image wherever original grayscale intensity > threshold.
    This mimics MATLAB-style region highlighting.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # create binary mask
    mask = gray > centroid

    if label_map.ndim == 2:
        label_map = np.stack([label_map]*3, axis=-1)

    output = label_map.copy()
    output[mask] = [255, 255, 255]
    return output
'''

import cv2
import numpy as np

def getbinary(image, X, centroid=180):
    """
    Faithful equivalent of your MATLAB getbinary().
    
    image: original RGB or grayscale image (a in MATLAB)
    X: segmentation map (but only used to match output shape)
    centroid: threshold value for grayscale
    
    Returns: image with white region where original grayscale > centroid.
    """
    # Convert image to grayscale if needed
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Initialize output image
    if X.ndim == 2:
        out = np.stack([X]*3, axis=-1).astype(np.uint8)
    else:
        out = X.copy().astype(np.uint8)

    # Apply the threshold exactly like your MATLAB code
    mask = gray > centroid
    out[mask] = [255, 255, 255]  # force white where gray > centroid

    return out


def threshold_label_map(image, centroid=180):
    """
    Applies MATLAB-style thresholding to produce a binary lesion mask.
    Works independent of the segmentation output.
    
    Parameters:
    - image: original grayscale or RGB image
    - centroid: intensity threshold

    Returns:
    - binary_mask: 0/255 label image (np.uint8)
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    mask = (gray > centroid).astype(np.uint8) * 255
    return mask
