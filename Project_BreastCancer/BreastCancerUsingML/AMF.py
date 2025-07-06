import numpy as np
import cv2

def adaptive_median_filter(image, max_kernel_size=7):
    """
    Apply Adaptive Median Filter to a grayscale image.
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Cast to int16 to avoid overflow
    image = image.astype(np.int16)

    padded_image = np.pad(image, max_kernel_size // 2, mode='edge')
    filtered_image = np.zeros_like(image)

    rows, cols = image.shape

    for i in range(rows):
        for j in range(cols):
            kernel_size = 3
            found = False
            while kernel_size <= max_kernel_size and not found:
                half = kernel_size // 2
                window = padded_image[i:i + kernel_size, j:j + kernel_size]
                Zmed = np.median(window)
                Zmin = window.min()
                Zmax = window.max()
                A1 = Zmed - Zmin
                A2 = Zmed - Zmax
                if A1 > 0 and A2 < 0:
                    B1 = image[i, j] - Zmin
                    B2 = image[i, j] - Zmax
                    if B1 > 0 and B2 < 0:
                        filtered_image[i, j] = image[i, j]
                    else:
                        filtered_image[i, j] = Zmed
                    found = True
                else:
                    kernel_size += 2
            if not found:
                filtered_image[i, j] = Zmed

    # Convert back to uint8 for image processing
    return np.clip(filtered_image, 0, 255).astype(np.uint8)
