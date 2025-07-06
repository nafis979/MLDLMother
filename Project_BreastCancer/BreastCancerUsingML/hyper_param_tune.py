import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Load CSV
df = pd.read_csv(r"C:\Users\Nafis\Desktop\VSCodeFiles\MLDLMother\Project_BreastCancer\BreastCancerUsingML\features_augmented_pca_balanced.csv")
X = df.drop(columns=["Label"]).values
y = LabelEncoder().fit_transform(df["Label"])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# SVM
svm_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC())
])
svm_params = {
    "svm__C": [0.1, 1, 10],
    "svm__gamma": [0.01, 0.1, 1],
    "svm__kernel": ["rbf"]
}
svm_grid = GridSearchCV(svm_pipe, svm_params, cv=cv, scoring="accuracy")
svm_grid.fit(X, y)
print("✅ SVM Best:", svm_grid.best_params_)

# Random Forest
rf_grid = GridSearchCV(
    RandomForestClassifier(),
    {"n_estimators": [50, 100, 200], "max_depth": [5, 10, 20, None]},
    cv=cv, scoring="accuracy"
)
rf_grid.fit(X, y)
print("✅ RF Best:", rf_grid.best_params_)

# Enhanced Neural Network
nn_grid = GridSearchCV(
    MLPClassifier(),
    {
        "hidden_layer_sizes": [(64,), (64, 32), (128,)],
        "learning_rate_init": [0.001, 0.0001],
        "max_iter": [300]
    },
    cv=cv, scoring="accuracy"
)
nn_grid.fit(X, y)
print("✅ NN Best:", nn_grid.best_params_)
