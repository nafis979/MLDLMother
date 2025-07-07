import os
import shutil

src_dir = r"C:\Users\Nafis\Desktop\VSCodeFiles\MLDLMother\Project_BreastCancer\BreastCancerUsingML\processed_new\segmented"
dest_dir = r"C:\Users\Nafis\Desktop\VSCodeFiles\MLDLMother\Project_BreastCancer\BreastCancerUsingML\processed_new\baselineCNN_input"

os.makedirs(dest_dir, exist_ok=True)

for fname in os.listdir(src_dir):
    fname_lower = fname.lower()
    if "benign" in fname_lower:
        label = "Benign"
    elif "malign" in fname_lower:
        label = "Malignant"
    else:
        continue

    label_dir = os.path.join(dest_dir, label)
    os.makedirs(label_dir, exist_ok=True)

    shutil.copy(os.path.join(src_dir, fname), os.path.join(label_dir, fname))
