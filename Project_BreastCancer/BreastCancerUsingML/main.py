import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter

from enhanced_feature_extraction import select_features_with_pca, extract_all_features
from AMF import adaptive_median_filter
from segmentation import segment_with_hmrf, getbinary, threshold_label_map

# === CONFIG ===
input_folders = [
    r"C:\Users\Nafis\Desktop\VSCodeFiles\MLDLMother\Project_BreastCancer\BreastCancerUsingML\mammogram_old",
    r"C:\Users\Nafis\Desktop\VSCodeFiles\MLDLMother\Project_BreastCancer\BreastCancerUsingML\mammogram_augmented"
]

output_folder = r"C:\Users\Nafis\Desktop\VSCodeFiles\MLDLMother\Project_BreastCancer\BreastCancerUsingML\processed_new\segmented"
feature_output = r"C:\Users\Nafis\Desktop\VSCodeFiles\MLDLMother\Project_BreastCancer\BreastCancerUsingML\processed_new\features_augmented__pca.csv"

Path(output_folder).mkdir(parents=True, exist_ok=True)

features_list = []
labels = []

for folder in input_folders:
    for filename in os.listdir(folder):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        # Label assignment
        fname_lower = filename.lower()
        if "benign" in fname_lower:
            label = "Benign"
        elif "malign" in fname_lower:
            label = "Malignant"
        elif "normal" in fname_lower:
            label = "Normal"
        else:
            print(f"⚠️ Skipping unlabeled image: {filename}")
            continue

        image_path = os.path.join(folder, filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠️ Skipping unreadable image: {filename}")
            continue

        # === PREPROCESSING
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = adaptive_median_filter(gray)

        # === DEBUG PREVIEW (Optional)
        if filename.lower().startswith("benign001"):
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(gray, cmap='gray')
            axs[0].set_title("Original Grayscale")
            axs[1].imshow(denoised, cmap='gray')
            axs[1].set_title("After AMF")
            axs[2].imshow(enhanced, cmap='gray')
            axs[2].set_title("CLAHE Output")
            for ax in axs: ax.axis('off')
            plt.tight_layout()
            plt.show()

        # === SEGMENTATION
        segmented = segment_with_hmrf(denoised, k=2, g=2, beta=1, EM_iter=10, MAP_iter=10)
        mask = threshold_label_map(gray, centroid=180)
        segmented_combined = np.logical_and(segmented == 255, mask == 255).astype(np.uint8) * 255
        overlay = getbinary(image, segmented_combined, centroid=180)

        # === FEATURE EXTRACTION
        features = extract_all_features(gray, segmented_combined)
        features.append(label)
        features_list.append(features)
        labels.append(label)

        # === Save overlay image
        cv2.imwrite(os.path.join(output_folder, filename), overlay)

        print(f"✅ Processed: {filename} → {label}")

# === Summary
print("\n🔍 Final label distribution:", Counter(labels))

# === PCA + SAVE
features_only = np.array(features_list)[:, :-1].astype(float)
features_labels = [row[-1] for row in features_list]

features_pca, pca_model = select_features_with_pca(features_only, variance_threshold=0.95)
features_with_labels = np.column_stack([features_pca, features_labels])

columns = [f"pca_feature_{i}" for i in range(features_pca.shape[1])] + ["Label"]
df_pca = pd.DataFrame(features_with_labels, columns=columns)
df_pca.to_csv(feature_output, index=False)

print(f"✅ PCA features saved to: {feature_output}")
