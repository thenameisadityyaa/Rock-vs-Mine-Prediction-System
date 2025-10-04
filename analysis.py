# analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# =========================
# Load Dataset
# =========================
file_path = input("Enter path to your CSV dataset: ")
df = pd.read_csv(file_path, header=None)

print("\n=== Dataset Preview ===")
print(df.head())

# =========================
# Dataset Info
# =========================
print("\n=== Dataset Info ===")
print(df.info())

print("\n=== Statistical Summary ===")
print(df.describe())

# =========================
# Missing Values
# =========================
print("\n=== Missing Values ===")
print(df.isnull().sum())

# =========================
# Target Column Distribution
# =========================
target_col = df.columns[-1]
print(f"\n=== Target Distribution ({target_col}) ===")
print(df[target_col].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(x=df[target_col], palette="bwr")
plt.title("Target Distribution")
plt.show()

# =========================
# Correlation Heatmap
# =========================
plt.figure(figsize=(12,10))
corr = df.iloc[:, :-1].corr()
sns.heatmap(corr, cmap="viridis")
plt.title("Correlation Heatmap of Features")
plt.show()

# =========================
# Feature Distributions
# =========================
plt.figure(figsize=(15,10))
for i in range(60):
    plt.subplot(10,6,i+1)
    sns.histplot(df[i], bins=10, color='skyblue')
    plt.title(f"Signal {i}")
    plt.tight_layout()
plt.show()

# =========================
# PCA 2D Visualization
# =========================
X = df.iloc[:, :-1]
y = df[target_col].map({'R':0,'M':1})

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8,6))
scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='bwr', alpha=0.7)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA 2D Projection")
plt.legend(handles=scatter.legend_elements()[0], labels=['Rock','Mine'])
plt.show()

print("\n=== Analysis Complete ===")
