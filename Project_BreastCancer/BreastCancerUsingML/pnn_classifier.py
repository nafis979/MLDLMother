import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.neighbors import RadiusNeighborsClassifier
import joblib

# Load extracted features
df = pd.read_csv(r"C:\Users\Nafis\Desktop\VSCodeFiles\MLDLMother\Project_BreastCancer\BreastCancerUsingML\features_top10.csv")  # or use features_top10.csv or features_pca.csv

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
joblib.dump(scaler, "pnn_scaler.pkl")
joblib.dump(le, "pnn_label_encoder.pkl")

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define a PNN-like classifier using RadiusNeighborsClassifier
'''
pnn = RadiusNeighborsClassifier(radius=2.0, weights='distance', outlier_label='most_frequent')
pnn.fit(X_train, y_train)

#Hyperparameter_tuning for PNN
'''


pnn = RadiusNeighborsClassifier(radius=1, weights='distance', outlier_label='most_frequent')


pnn.fit(X_train, y_train)




# Predict
y_pred = pnn.predict(X_test)

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


# Save model
joblib.dump(pnn, "pnn_model.pkl")
