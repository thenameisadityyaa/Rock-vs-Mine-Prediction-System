# ===================================================================
# analysis.py
# Complete analysis for Sonar AI: Rock vs. Mine
# ===================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import tensorflow as tf

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ===================================================================
# 1. LOAD DATA
# ===================================================================
print("Loading Sonar dataset...")

dataset_path = os.path.join("sonar.csv")  # Adjust if using Kaggle download
df = pd.read_csv(dataset_path, header=None)

# Features and target
X = df.drop(columns=60)
y = df[60].map({'R': 0, 'M': 1})  # Rock=0, Mine=1

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

# ===================================================================
# 2. RANDOM FOREST MODEL (with class balance)
# ===================================================================
print("\nTraining Random Forest...")

pipeline_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    ))
])

pipeline_rf.fit(X_train, y_train)

# Predict & evaluate
y_pred_rf = pipeline_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Test Accuracy: {accuracy_rf:.2%}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf, target_names=['Rock', 'Mine']))

# Confusion matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(7,5))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=['Rock','Mine'], yticklabels=['Rock','Mine'])
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Feature importance
importances = pipeline_rf.named_steps['classifier'].feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,8))
sns.barplot(x=importances[indices][:15], y=[f'Signal {i}' for i in indices[:15]], palette='viridis')
plt.title("Top 15 Feature Importances (Random Forest)")
plt.xlabel("Importance Score")
plt.show()

# ===================================================================
# 3. NEURAL NETWORK MODEL (Optional)
# ===================================================================
print("\nTraining Neural Network...")

scaler_nn = StandardScaler()
X_train_scaled = scaler_nn.fit_transform(X_train)
X_test_scaled = scaler_nn.transform(X_test)

nn_model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = nn_model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test),
                       epochs=150, batch_size=16, verbose=0)

loss, accuracy_nn = nn_model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Neural Network Test Accuracy: {accuracy_nn:.2%}")

# Training history plot
plt.figure(figsize=(10,6))
pd.DataFrame(history.history)[['accuracy','val_accuracy']].plot()
plt.title("Neural Network Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

# ===================================================================
# 4. PCA VISUALIZATION
# ===================================================================
print("\nPerforming PCA 2D projection...")

scaler_pca = StandardScaler()
X_scaled = scaler_pca.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='coolwarm', alpha=0.7)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA 2D Projection of Sonar Data")
plt.legend(handles=scatter.legend_elements()[0], labels=['Rock','Mine'])
plt.show()

# ===================================================================
# 5. SHAP EXPLAINABILITY FOR RANDOM FOREST
# ===================================================================
print("\nGenerating SHAP explainability plots...")

rf_model = pipeline_rf.named_steps['classifier']
X_test_scaled_rf = pipeline_rf.named_steps['scaler'].transform(X_test)

explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test_scaled_rf)

feature_names = [f"Signal {i}" for i in range(X.shape[1])]

# SHAP summary plot
shap.summary_plot(shap_values[1], X_test, feature_names=feature_names, show=True)

# SHAP force plot for first test sample
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], X_test.iloc[0,:], feature_names=feature_names)
