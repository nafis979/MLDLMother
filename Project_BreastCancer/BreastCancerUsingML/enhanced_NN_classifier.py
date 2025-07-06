import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv(r"C:\Users\Nafis\Desktop\VSCodeFiles\MLDLMother\Project_BreastCancer\BreastCancerUsingML\features_augmented_pca_balanced.csv")

# Encode labels
le = LabelEncoder()
df["Label"] = le.fit_transform(df["Label"])

# Split features and labels
X = df.drop("Label", axis=1).values
y = df["Label"].values

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save scaler and encoder
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Define enhanced neural network
class EnhancedNN(nn.Module):
    def __init__(self, input_size):
        super(EnhancedNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
    def forward(self, x):
        return self.model(x)

# Initialize model
model = EnhancedNN(input_size=X.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Train model
for epoch in range(300):
    model.train()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/300, Loss: {loss.item():.4f}")

# Evaluate model
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    predictions = outputs.argmax(dim=1)
    
    # Convert tensors to numpy arrays
    y_true = y_test_tensor.numpy()
    y_pred = predictions.numpy()
    
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n✅ Test Accuracy: {accuracy * 100:.2f}%")

    # Classification report
    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=le.classes_))

    # Confusion Matrix
    print("🧮 Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    # Additional metrics
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
'''
    # Predicted probabilities for AUC
    probs = torch.softmax(outputs, dim=1).numpy()
    y_true_bin = label_binarize(y_true, classes=np.unique(y))

    try:
        auc_score = roc_auc_score(y_true_bin, probs, average='macro', multi_class='ovr')
    except Exception as e:
        auc_score = None
        print("⚠️ AUC calculation failed:", e)

    print(f"✅ Precision (macro): {precision:.4f}")
    print(f"✅ Recall (macro): {recall:.4f}")
    print(f"✅ F1 Score (macro): {f1:.4f}")
    if auc_score is not None:
        print(f"✅ AUC Score (macro, OvR): {auc_score:.4f}")
'''
# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names=None, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()

class_names = ["Benign", "Malignant", "Normal"]
plot_confusion_matrix(y_true, y_pred, class_names)

# Save model
torch.save(model.state_dict(), "enhanced_nn_model.pth")
