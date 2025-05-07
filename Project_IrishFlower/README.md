# Iris Classification Project

This project uses the Iris dataset to build and evaluate machine learning models for species classification.

## Overview
- **Dataset**: Iris dataset (150 samples, 4 features: sepal length, sepal width, petal length, petal width; 3 species).
- **Objective**: Classify Iris species using multiple machine learning algorithms.
- **Tools**: Python, Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib.

## Features
- Data exploration with descriptive statistics and visualizations (boxplots, swarmplots, violinplots, correlation heatmap, pairplot, 3D scatter).
- Models implemented:
  - Support Vector Machine (SVM)
  - Logistic Regression
  - Decision Tree
  - Random Forest
- Model evaluation using accuracy, classification report, and confusion matrix.
- Prediction on new samples with confidence scores, nearest neighbor analysis, z-scores, and Euclidean distance from dataset mean.

## Usage
1. Install dependencies: `pip install numpy pandas scikit-learn seaborn matplotlib`
2. Run the Jupyter notebook or Python script to explore data, train models, and test predictions.

## Key Findings
- Random Forest achieved high accuracy (~0.97) on the test set.
- Visualizations reveal clear feature separations between species, especially petal length and width.
- New sample predictions include confidence and statistical analysis for interpretability.

## File Structure
- `iris_classification.ipynb`: Main notebook with code and visualizations.
- `README.md`: This file.

## License
MIT License
