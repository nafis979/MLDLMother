import pandas as pd

# Load the CSV file
df = pd.read_csv(r"C:\Users\Nafis\Desktop\VSCodeFiles\MLDLMother\Project_BreastCancer\BreastCancerUsingML\processed_new\features_augmented__pca.csv")  # Change path as needed

# Count each label
label_counts = df["Label"].value_counts()

# Print all label counts
print("Label Distribution:\n", label_counts)

# Optional: Get only malignant count
malignant_count = label_counts.get("Malignant", 0)
print(f"\nMalignant count: {malignant_count}")
