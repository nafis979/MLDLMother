import os
import cv2
import albumentations as A

# === Folders ===
input_folder = r"C:\Users\Nafis\Desktop\VSCodeFiles\MLDLMother\Project_BreastCancer\BreastCancerUsingML\mammogram_old"
output_folder = r"C:\Users\Nafis\Desktop\VSCodeFiles\MLDLMother\Project_BreastCancer\BreastCancerUsingML\mammogram_augmented"
os.makedirs(output_folder, exist_ok=True)

# === Albumentations Augmentation Pipeline ===
augmenter = A.Compose([
    A.HorizontalFlip(p=0.5),                # Horizontal flip
    A.Rotate(limit=20, p=0.7),              # Random rotation (higher p to reflect imgaug behavior)
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0, p=0.8)  # Brightness variation
])

# === Loop through images ===
for filename in os.listdir(input_folder):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    img_path = os.path.join(input_folder, filename)
    image = cv2.imread(img_path)

    if image is None:
        print(f"Skipping unreadable: {filename}")
        continue

    # Convert BGR to RGB for augmentation
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create 2 augmentations per image
    for i in range(2):
        augmented = augmenter(image=image_rgb)
        aug_image = augmented["image"]

        # Convert back to BGR for saving
        aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)

        name, ext = os.path.splitext(filename)
        new_filename = f"{name}_aug{i+1}{ext}"
        output_path = os.path.join(output_folder, new_filename)

        cv2.imwrite(output_path, aug_image_bgr)
        print(f"✅ Saved: {new_filename}")
