from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv(r"C:\Users\Nafis\Desktop\VSCodeFiles\MLDLMother\Project_BreastCancer\BreastCancerUsingML\features_augmented_pca_balanced.csv")
X = df.drop(columns=["Label"]).values
y = LabelEncoder().fit_transform(df["Label"])

# Random Forest CV
clf = RandomForestClassifier(n_estimators=50,random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
acc = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
print(f"✅ Random Forest Accuracy: {np.mean(acc):.2f} ± {np.std(acc):.2f}")


from sklearn.svm import SVC

# SVM CV
clf = SVC(kernel="rbf", gamma="scale")
acc = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
print(f"✅ SVM Accuracy: {np.mean(acc):.2f} ± {np.std(acc):.2f}")


import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np

class EnhancedNN(nn.Module):
    def __init__(self, input_dim, hidden=64, output_dim=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, output_dim)
        )

    def forward(self, x):
        return self.model(x)

# Prepare data
X_scaled = StandardScaler().fit_transform(X)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for train_idx, val_idx in cv.split(X_tensor, y_tensor):
    model = EnhancedNN(X_tensor.shape[1], output_dim=len(np.unique(y)))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(30):
        model.train()
        optimizer.zero_grad()
        output = model(X_tensor[train_idx])
        loss = criterion(output, y_tensor[train_idx])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_tensor[val_idx]).argmax(1)
        acc = accuracy_score(y_tensor[val_idx], preds)
        scores.append(acc)

print(f"✅ Enhanced NN Accuracy: {np.mean(scores):.2f} ± {np.std(scores):.2f}")


# PNN pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pnn", RadiusNeighborsClassifier(radius=5, weights="uniform", outlier_label='most_frequent'))
])

# Stratified CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")

print(f"✅ PNN Accuracy: {np.mean(scores):.2f} ± {np.std(scores):.2f}")