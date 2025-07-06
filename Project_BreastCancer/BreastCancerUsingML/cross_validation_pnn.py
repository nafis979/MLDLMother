from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import pandas as pd

# Load data
df = pd.read_csv(r"C:\Users\Nafis\Desktop\VSCodeFiles\MLDLMother\Project_BreastCancer\BreastCancerUsingML\features_top10.csv")  # or your PCA/selected feature file
X = df.drop(columns=["Label"]).values
y = LabelEncoder().fit_transform(df["Label"].values)

# Define pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pnn", RadiusNeighborsClassifier(outlier_label='most_frequent'))
])

# Grid of radius values
param_grid = {
    "pnn__radius": [0.5, 1.0, 1.5, 2.0, 5.0],
    "pnn__weights": ["uniform", "distance"]
}

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring="accuracy")
grid.fit(X, y)

# Best result
print("✅ Best Params:", grid.best_params_)
print("✅ Best CV Accuracy:", grid.best_score_)
