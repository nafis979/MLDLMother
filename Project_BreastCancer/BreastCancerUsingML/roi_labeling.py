import os
import pandas as pd

# ✅ Set your folder path here
image_folder = r"C:\Users\Nafis\Desktop\VSCodeFiles\MLDLMother\Project_BreastCancer\BreastCancerUsingML\mammogram_old"  # Path where your .jpg images are stored

# ✅ Create a list of (filename, label)
data = []
for filename in os.listdir(image_folder):
    if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg") or filename.lower().endswith(".png"):
        label = (
            "Benign" if "benign" in filename.lower()
            else "Malignant" if "malign" in filename.lower()
            else "Normal" if "normal" in filename.lower()
            else None
        )
        if label:
            data.append((filename, label))

# ✅ Save to CSV
df = pd.DataFrame(data, columns=["filename", "label"])
df.to_csv("roi_labels.csv", index=False)
print(f"Saved roi_labels.csv with {len(df)} entries.")
