import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# === CONFIG ===
data_dir = r"/kaggle/input/baselinecnn-input/baselineCNN_input"
num_classes = 2
batch_size = 16
learning_rate = 1e-4
epochs = 10
k_folds = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Transformations ===
transform = transforms.Compose([
    transforms.Resize((299, 299)),  # works for both DenseNet and Inception
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# === Dataset Setup ===
dataset = ImageFolder(data_dir, transform=transform)
X = list(range(len(dataset)))
y = [dataset[i][1] for i in range(len(dataset))]
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

# === Model Training Function ===
def run_model(model_type):
    print(f"\n📊 Running {model_type.upper()}...")
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n🔁 Fold {fold+1}/{k_folds}")

        train_ds = Subset(dataset, train_idx)
        val_ds = Subset(dataset, val_idx)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # === Model Selection ===
        if model_type == 'densenet121':
            from torchvision.models import densenet121, DenseNet121_Weights
            weights = DenseNet121_Weights.IMAGENET1K_V1
            model = densenet121(weights=weights)
            for p in model.features.parameters():
                p.requires_grad = False
            model.classifier = nn.Sequential(
                nn.Linear(model.classifier.in_features, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )

        elif model_type == 'inceptionv3':
            from torchvision.models import inception_v3, Inception_V3_Weights
            weights = Inception_V3_Weights.IMAGENET1K_V1
            model = inception_v3(weights=weights, aux_logits=True)
            for p in model.parameters():
                p.requires_grad = False
            model.fc = nn.Sequential(
                nn.Linear(model.fc.in_features, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )
            model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)

        else:
            raise ValueError("Only 'densenet121' and 'inceptionv3' are supported.")

        model = model.to(device)

        # === Weighted Loss & Optimizer ===
        weights_array = compute_class_weight('balanced', classes=np.unique(y), y=np.array(y)[train_idx])
        class_weights = torch.tensor(weights_array, dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # === Training Loop ===
        for epoch in range(epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                if model_type == "inceptionv3":
                    output, aux_output = model(xb)
                    loss = criterion(output, yb) + 0.4 * criterion(aux_output, yb)
                else:
                    output = model(xb)
                    loss = criterion(output, yb)
                loss.backward()
                optimizer.step()

        # === Evaluation ===
        model.eval()
        preds, actuals = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                out = model(xb)
                if isinstance(out, tuple):  # Inception's output
                    out = out[0]
                pred = torch.argmax(out, dim=1).cpu().numpy()
                preds.extend(pred)
                actuals.extend(yb.numpy())

        acc = accuracy_score(actuals, preds)
        prec = precision_score(actuals, preds, average='macro', zero_division=0)
        rec = recall_score(actuals, preds, average='macro', zero_division=0)
        cm = confusion_matrix(actuals, preds)
        fold_scores.append([acc, prec, rec])

        print(f" ✅ Fold {fold+1}: Acc={acc:.2f}, Prec={prec:.2f}, Rec={rec:.2f}")

    # === Summary ===
    avg = np.mean(fold_scores, axis=0)
    std = np.std(fold_scores, axis=0)
    print(f"\n📊 {model_type.upper()} Cross-Validation Summary:")
    print(f" Accuracy : {avg[0]*100:.2f}% ± {std[0]*100:.2f}%")
    print(f" Precision: {avg[1]*100:.2f}% ± {std[1]*100:.2f}%")
    print(f" Recall   : {avg[2]*100:.2f}% ± {std[2]*100:.2f}%")
    print(f" Confusion Matrix (last fold):\n{cm}")

# === Run the models ===
run_model("densenet121")
# run_model("inceptionv3")  # Uncomment to run InceptionV3
