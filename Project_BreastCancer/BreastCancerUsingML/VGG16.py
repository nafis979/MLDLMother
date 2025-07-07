import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os

# === CONFIG ===
data_dir = r"/kaggle/input/baselinecnn-input/baselineCNN_input"  # ✅ your dataset path
num_classes = 2
batch_size = 16
learning_rate = 1e-4
epochs = 15
k_folds = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Transform ===
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(num_output_channels=3),  # VGG expects 3 channels
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

# === Dataset and Fold Split ===
dataset = ImageFolder(data_dir, transform=transform)
X = list(range(len(dataset)))
y = [dataset[i][1] for i in range(len(dataset))]

skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

def run_model(model_type="vgg16"):
    print(f"\n📊 Running {model_type.upper()} Cross-Validation...")
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        train_ds = Subset(dataset, train_idx)
        val_ds = Subset(dataset, val_idx)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # === Load Model with Updated Weights API ===
        if model_type == "vgg16":
            from torchvision.models import vgg16, VGG16_Weights
            weights = VGG16_Weights.DEFAULT
            model = vgg16(weights=weights)
            for p in model.features.parameters():
                p.requires_grad = False
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

        elif model_type == "alexnet":
            from torchvision.models import alexnet, AlexNet_Weights
            weights = AlexNet_Weights.DEFAULT
            model = alexnet(weights=weights)
            for p in model.features.parameters():
                p.requires_grad = False
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

        model = model.to(device)

        # === Class Weights ===
        weights_array = compute_class_weight('balanced', classes=np.unique(y), y=np.array(y)[train_idx])
        class_weights = torch.tensor(weights_array, dtype=torch.float).to(device)

        # === Training Setup ===
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # === Training ===
        for epoch in range(epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

        # === Evaluation ===
        model.eval()
        preds, actuals = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                outputs = model(xb)
                pred = torch.argmax(outputs, dim=1).cpu().numpy()
                preds.extend(pred)
                actuals.extend(yb.numpy())

        acc = accuracy_score(actuals, preds)
        prec = precision_score(actuals, preds, average='macro', zero_division=0)
        rec = recall_score(actuals, preds, average='macro', zero_division=0)
        fold_scores.append([acc, prec, rec])
        print(f" Fold {fold+1}: Acc={acc:.2f}, Prec={prec:.2f}, Rec={rec:.2f}")

    # === Summary ===
    avg = np.mean(fold_scores, axis=0)
    print(f"\n✅ {model_type.upper()} Summary:")
    print(f" Avg Accuracy : {avg[0]*100:.2f}%")
    print(f" Avg Precision: {avg[1]*100:.2f}%")
    print(f" Avg Recall   : {avg[2]*100:.2f}%")

# === Run VGG16 Baseline ===
run_model("vgg16")
