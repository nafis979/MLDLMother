import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import cv2
import numpy as np

def gmm_segmentation(image, n_components=2):
    """
    Apply Gaussian Mixture Model segmentation to a grayscale image.
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    flat_image = image.reshape(-1, 1)
    gmm = GaussianMixture(n_components=n_components, covariance_type='tied', random_state=0)
    gmm.fit(flat_image)
    labels = gmm.predict(flat_image)
    segmented_image = labels.reshape(image.shape)
    return segmented_image

def improved_gmm_segmentation(image, n_components=2):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Crop out bottom black border using threshold
    mask = image > 20  # filter out the black background
    masked_image = image * mask

    # Equalize contrast inside breast region
    equalized = cv2.equalizeHist(masked_image.astype(np.uint8))

    # Normalize and reshape
    flat_image = equalized[mask].reshape(-1, 1).astype(np.float32) / 255.0

    gmm = GaussianMixture(n_components=n_components, covariance_type='tied', random_state=0)
    gmm.fit(flat_image)

    full_flat = equalized.reshape(-1, 1).astype(np.float32) / 255.0
    labels = gmm.predict(full_flat)

    # Reconstruct full segmented image
    segmented = labels.reshape(image.shape)

    # Stretch to 0-255 for visualization
    segmented = (segmented * (255 // (n_components - 1))).astype(np.uint8)
    return segmented





def kmeans_segmentation(image, n_clusters=2):
    """
    Apply KMeans clustering segmentation to a grayscale image.
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    flat_image = image.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
    kmeans.fit(flat_image)
    labels = kmeans.predict(flat_image)
    segmented_image = labels.reshape(image.shape)
    return segmented_image

# Example plotting utility for visual inspection
def show_segmentation_results(original, gmm_result, kmeans_result):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(gmm_result, cmap='jet')
    plt.title('GMM Segmentation')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(kmeans_result, cmap='jet')
    plt.title('K-Means Segmentation')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
