import pandas as pd
from sklearn.utils import resample

# Load your PCA feature CSV
df = pd.read_csv(r"C:\Users\Nafis\Desktop\VSCodeFiles\MLDLMother\Project_BreastCancer\BreastCancerUsingML\features_augmented__pca.csv")  # path to your file

# Separate classes
benign_df = df[df["Label"] == "Benign"]
malignant_df = df[df["Label"] == "Malignant"]

# Find the smaller class size
min_size = min(len(benign_df), len(malignant_df))

# Undersample both to the same size (optional: oversample the minority instead)
benign_bal = resample(benign_df, replace=False, n_samples=min_size, random_state=42)
malignant_bal = resample(malignant_df, replace=False, n_samples=min_size, random_state=42)

# Combine and shuffle
balanced_df = pd.concat([benign_bal, malignant_bal]).sample(frac=1, random_state=42)

# Save balanced data to a new CSV
balanced_df.to_csv("features_augmented_pca_balanced.csv", index=False)
print("✅ Balanced dataset saved to features_augmented_pca_balanced.csv")
