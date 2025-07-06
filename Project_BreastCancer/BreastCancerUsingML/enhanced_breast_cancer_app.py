
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import joblib
import torch
from enhanced_feature_extraction import extract_all_features
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import RadiusNeighborsClassifier

# Load enhanced model
class EnhancedNN(torch.nn.Module):
    def __init__(self, input_size):
        super(EnhancedNN, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 3)
        )

    def forward(self, x):
        return self.model(x)

# Load tools
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("label_encoder.pkl")
model = EnhancedNN(input_size=scaler.mean_.shape[0])
model.load_state_dict(torch.load("glcm_nn_model.pth", map_location=torch.device("cpu")))
model.eval()

# Load other classifiers
pnn_model = joblib.load("pnn_model.pkl")
pnn_scaler = joblib.load("pnn_scaler.pkl")
pnn_encoder = joblib.load("pnn_label_encoder.pkl")

svm_model = joblib.load("svm_model.pkl")
svm_scaler = joblib.load("svm_scaler.pkl")
svm_encoder = joblib.load("svm_label_encoder.pkl")

rf_model = joblib.load("rf_model.pkl")
rf_scaler = joblib.load("rf_scaler.pkl")
rf_encoder = joblib.load("rf_label_encoder.pkl")

# Streamlit UI
st.title("Enhanced Breast Cancer Detection")

uploaded_file = st.file_uploader("Upload a JPG image", type=["jpg", "jpeg"])

classifier_option = st.selectbox("Choose Classifier", ["Enhanced NN", "PNN", "SVM", "Random Forest"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    st.image(image_np, caption="Uploaded Image", use_column_width=True)

    if st.button("Classify"):
        try:
            features = extract_all_features(image_np)

            if classifier_option == "Enhanced NN":
                X_scaled = scaler.transform([features])
                input_tensor = torch.tensor(X_scaled, dtype=torch.float32)
                with torch.no_grad():
                    output = model(input_tensor)
                    pred = output.argmax(dim=1).item()
                    label = encoder.inverse_transform([pred])[0]

            elif classifier_option == "PNN":
                X_scaled = pnn_scaler.transform([features])
                pred = pnn_model.predict(X_scaled)[0]
                label = pnn_encoder.inverse_transform([pred])[0]

            elif classifier_option == "SVM":
                X_scaled = svm_scaler.transform([features])
                pred = svm_model.predict(X_scaled)[0]
                label = encoder.inverse_transform([pred])[0]

            elif classifier_option == "Random Forest":
                X_scaled = rf_scaler.transform([features])
                pred = rf_model.predict(X_scaled)[0]
                label = encoder.inverse_transform([pred])[0]

            st.success(f"Prediction: **{label.upper()}**")

        except Exception as e:
            st.error(f"Classification failed: {e}")
