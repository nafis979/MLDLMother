# Breast Cancer Detection using ML

This project uses classical Machine Learning and Deep Learning techniques to classify breast cancer from mammogram images. The analysis is performed using a dataset of **440 mammogram images**, sourced from both private and publicly available portions of the **CBIS-DDSM dataset**.

> **Note:** The dataset is not provided publicly here. However, it can be shared privately upon request.

---

## ⚙️ Workflow Overview:

The project follows a structured workflow:

- **Data Collection**: 
  - 440 mammogram images from private and CBIS-DDSM datasets.
  
- **Data Augmentation**:
  - Augmented mammogram images using techniques like rotation, flipping, and brightness variations for improved model robustness.

- **Preprocessing**:
  - Adaptive Median Filtering (AMF) for noise reduction.
  - CLAHE (Contrast Limited Adaptive Histogram Equalization) for image enhancement.

- **Segmentation**:
  - Hidden Markov Random Field (HMRF) combined with binary thresholding for precise tumor segmentation.

- **Feature Extraction**:
  - Gray-Level Co-Occurrence Matrix (GLCM) texture features.
  - Wavelet transformation features.
  - Color histogram and shape-based features.
  
- **Dimensionality Reduction**:
  - Principal Component Analysis (PCA) for optimized feature selection.

- **Model Training and Validation**:
  - Enhanced Neural Network (PyTorch-based), Random Forest, SVM, and Probabilistic Neural Network (PNN).
  - Hyperparameter tuning and cross-validation.

- **Deployment**:
  - Streamlit-based user interface for real-time predictions.

---

## 📌 Project Files:
- `main.ipynb` contains **all the steps** (preprocessing, segmentation, feature extraction, PCA, and classification), excluding the Streamlit deployment part.
- Streamlit interface (`enhanced_breast_cancer_app.py`) is provided separately.
