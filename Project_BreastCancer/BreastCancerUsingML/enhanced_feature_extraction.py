
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
import pywt
from sklearn.decomposition import PCA

def extract_glcm_features(gray_img):
    img = img_as_ubyte(gray_img)
    glcm = graycomatrix(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    return [
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'dissimilarity')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0],
        graycoprops(glcm, 'ASM')[0, 0],
        -np.sum(glcm * np.log2(glcm + (glcm == 0)))  # entropy
    ]

def extract_color_histogram(image, bins=32):
    chans = cv2.split(image)
    features = []
    for chan in chans:
        hist = cv2.calcHist([chan], [0], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)
    return features

def extract_shape_features(gray_img):
    contours, _ = cv2.findContours(gray_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        moments = cv2.moments(cnt)
        hu = cv2.HuMoments(moments).flatten()
        return hu.tolist()
    return [0]*7

def extract_wavelet_features(gray_img):
    coeffs = pywt.dwt2(gray_img, 'haar')
    cA, (cH, cV, cD) = coeffs
    stats = lambda x: [np.mean(x), np.std(x), np.min(x), np.max(x)]
    return stats(cA) + stats(cH) + stats(cV) + stats(cD)

def extract_all_features(image, mask=None):
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    if mask is not None:
        image = image.copy()
        gray = gray.copy()
        image[mask != 255] = 0
        gray[mask != 255] = 0

    features = []
    features += extract_glcm_features(gray)
    features += extract_color_histogram(image)
    features += extract_shape_features(gray)
    features += extract_wavelet_features(gray)
    return features
print(extract_all_features)

#PCA added to improve feature selection

def select_features_with_pca(feature_matrix, variance_threshold=0.95):
    """
    Applies PCA to reduce dimensionality of the feature matrix while retaining specified variance.
    
    Parameters:
    - feature_matrix (ndarray): 2D array (samples x features)
    - variance_threshold (float): % of variance to retain (e.g., 0.95 for 95%)
    
    Returns:
    - reduced_features (ndarray): PCA-reduced feature matrix
    - pca_model (PCA object): fitted PCA model (can be reused for test data)
    """
    pca = PCA(n_components=variance_threshold)
    reduced_features = pca.fit_transform(feature_matrix)
    return reduced_features, pca
