import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np

# === CONFIG ===
data_dir = r"/kaggle/input/baselinecnn-input/baselineCNN_input"
num_classes = 2
batch_size = 16
learning_rate = 1e-4
epochs = 10
k_folds = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === TRANSFORMATIONS ===
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(num_output_channels=3),  # ResNet expects 3 channels
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# === LOAD DATASET ===
dataset = ImageFolder(data_dir, transform=transform)
X = list(range(len(dataset)))
y = [dataset[i][1] for i in range(len(dataset))]

skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

# === METRIC STORAGE ===
acc_list, prec_list, rec_list = [], [], []

# === FOLD LOOP ===
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n📂 Fold {fold+1}/{k_folds}")
    
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # === LOAD PRETRAINED RESNET18 ===
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Freeze all layers except final FC
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace FC layer
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes)
    )
    
    model = model.to(device)

    # === LOSS + OPTIMIZER ===
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

    # === TRAINING LOOP ===
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    # === EVALUATION ===
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            out = model(xb)
            pred = torch.argmax(out, dim=1).cpu().numpy()
            preds.extend(pred)
            actuals.extend(yb.numpy())

    acc = accuracy_score(actuals, preds)
    prec = precision_score(actuals, preds, average='macro', zero_division=0)
    rec = recall_score(actuals, preds, average='macro', zero_division=0)

    acc_list.append(acc)
    prec_list.append(prec)
    rec_list.append(rec)

    print(f"✅ Fold Accuracy: {acc*100:.2f}%, Precision: {prec*100:.2f}%, Recall: {rec*100:.2f}%")

# === Statistical Summary ===
print("\n📊 Final Stratified K-Fold Summary (ResNet18):")
print(f"Accuracy : {np.mean(acc_list)*100:.2f}% ± {np.std(acc_list)*100:.2f}%")
print(f"Precision: {np.mean(prec_list)*100:.2f}% ± {np.std(prec_list)*100:.2f}%")
print(f"Recall   : {np.mean(rec_list)*100:.2f}% ± {np.std(rec_list)*100:.2f}%")