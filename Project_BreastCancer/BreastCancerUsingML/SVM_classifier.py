import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.svm import SVC
import joblib

# Load extracted features
df = pd.read_csv(r"C:\Users\Nafis\Desktop\VSCodeFiles\MLDLMother\Project_BreastCancer\BreastCancerUsingML\features_augmented_pca_balanced.csv")  # or use features_top10.csv or features_pca.csv

# Encode labels
le = LabelEncoder()
df["Label"] = le.fit_transform(df["Label"])

# Split X and y
X = df.drop("Label", axis=1).values
y = df["Label"].values

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler and encoder
joblib.dump(scaler, "svm_scaler.pkl")
joblib.dump(le, "svm_label_encoder.pkl")

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# SVM classifier
#svm = SVC(probability=True)

# Assuming best params from GridSearchCV
best_svm_params = {'C': 0.1, 'gamma': 0.01, 'kernel': 'rbf'}
svm = SVC(**best_svm_params, probability=True)

svm.fit(X_train, y_train)

# Predict
y_pred = svm.predict(X_test)

# Evaluation metrics
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("✅ Precision:", precision_score(y_test, y_pred, average="macro"))
print("✅ Recall:", recall_score(y_test, y_pred, average="macro"))
print("✅ Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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
plot_confusion_matrix(y_test, y_pred, class_names)


# --- AUC and F1 ---
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import label_binarize

# Predict probabilities
y_proba = svm.predict_proba(X_test)

# Binarize y_test for AUC (multiclass)
y_test_bin = label_binarize(y_test, classes=np.unique(y))

# AUC Score (macro, OvR)
try:
    auc_score = roc_auc_score(y_test_bin, y_proba, average='macro', multi_class='ovr')
    print("✅ AUC Score (macro, OvR):", auc_score)
except Exception as e:
    print("⚠️ Could not compute AUC score:", e)

# F1 Score
f1 = f1_score(y_test, y_pred, average='macro')
print("✅ F1 Score (macro):", f1)




# Save model
joblib.dump(svm,"svm_model.pkl")
