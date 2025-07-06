import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

# Load CSV (use raw string path)
df = pd.read_csv(r"C:\Users\Nafis\Desktop\VSCodeFiles\MLDLMother\Project_BreastCancer\BreastCancerUsingML\features_segmented_wo_pca.csv")

# Encode labels to numeric values
df['Label_encoded'] = LabelEncoder().fit_transform(df['Label'])

# Drop the string label column before computing correlation
correlation_matrix = df.drop(columns=["Label"]).corr()

# Correlation of each feature with the encoded label
label_correlation = correlation_matrix['Label_encoded'].drop('Label_encoded')


# Select top N features (e.g., N = 10)
N = 10
top_features = label_correlation.abs().sort_values(ascending=False).head(N).index.tolist()

# Add the label column back
top_features.append("Label")

# Slice dataframe to keep only top N features + label
df_top = df[top_features]

# Save to CSV
df_top.to_csv("processed/features_top10.csv", index=False)

print(f"✅ Saved top {N} features based on correlation to: processed/features_top10.csv")

# Plot
plt.figure(figsize=(14, 6))
label_correlation.abs().plot(kind='bar', color='skyblue')
plt.title("Correlation of Each Feature with Cancer Type")
plt.ylabel("Absolute Correlation")
plt.xlabel("Feature Index")
plt.grid(True)
plt.tight_layout()
plt.show()


# 🔥 Plot heatmap of correlations among top features
plt.figure(figsize=(10, 8))
heatmap_data = df_top.drop(columns=["Label"])
sns.heatmap(heatmap_data.corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar_kws={"shrink": 0.8})
plt.title("Heatmap of Top 10 Features (Correlation Matrix)")
plt.tight_layout()
plt.show()