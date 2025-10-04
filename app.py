# ===================================================================
# SONAR AI: ROCK VS MINE PREDICTION (Updated)
# ===================================================================

import streamlit as st
import pandas as pd
import numpy as np
import os
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA

sns.set_style("whitegrid")

# ===================================================================
# APP CONFIGURATION
# ===================================================================
st.set_page_config(
    page_title="Sonar AI | Rock vs Mine",
    page_icon="🌊",
    layout="wide"
)

# ===================================================================
# LOAD MODEL & DATA (CACHED)
# ===================================================================
@st.cache_resource
def load_model_and_data():
    """Downloads Sonar dataset, trains Random Forest model, returns pipeline & visuals"""
    try:
        path = kagglehub.dataset_download("rupakroy/sonarcsv")
        file_path = os.path.join(path, "sonar.csv")
        df = pd.read_csv(file_path, header=None)
    except Exception as e:
        st.error(f"Failed to download dataset: {e}")
        return None, None, None, None, None, None, None

    # Features and target
    X = df.drop(columns=60, axis=1)
    Y = df[60].map({'R':0,'M':1})  # Encode Rock=0, Mine=1

    # Train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

    # Random Forest pipeline with tuning
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ))
    ])
    pipeline.fit(X_train, Y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(Y_test, y_pred)
    cm = confusion_matrix(Y_test, y_pred)
    feature_importances = pipeline.named_steps['classifier'].feature_importances_

    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    return pipeline, acc, cm, feature_importances, X_train.columns, X_pca, Y

# Load data
model_pipeline, model_accuracy, cm_data, importances, feature_names, X_pca_all, y_all = load_model_and_data()

# ===================================================================
# MAIN APP UI
# ===================================================================
st.title("🌊 Sonar AI: Rock vs Mine Prediction System")
if model_pipeline:
    st.metric(label="Model Accuracy", value=f"{model_accuracy:.2%}")
    st.markdown("Predict whether an underwater object is a Rock or a Mine using sonar signals.")

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Batch Prediction (CSV)", "Model Insights", "Manual Prediction"])

    # -----------------------------
    # TAB 1: Batch CSV Prediction
    # -----------------------------
    with tab1:
        st.header("Upload CSV for Batch Prediction")
        uploaded_file = st.file_uploader("CSV file (60 sonar features, no header)", type="csv")
        if uploaded_file is not None:
            try:
                input_df = pd.read_csv(uploaded_file, header=None)
                if input_df.shape[1] != 60:
                    st.error(f"CSV must have exactly 60 columns, found {input_df.shape[1]}")
                else:
                    preds = model_pipeline.predict(input_df)
                    probs = model_pipeline.predict_proba(input_df)
                    input_df['Predicted_Object'] = ['Mine' if p==1 else 'Rock' for p in preds]
                    input_df['Confidence'] = [f"{prob.max()*100:.2f}%" for prob in probs]
                    st.success("Predictions Complete!")
                    st.dataframe(input_df)

                    # Download results
                    @st.cache_data
                    def convert_df_to_csv(df):
                        return df.to_csv(index=False).encode('utf-8')

                    csv_results = convert_df_to_csv(input_df)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_results,
                        file_name="sonar_predictions.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"Error processing file: {e}")

    # -----------------------------
    # TAB 2: Model Insights
    # -----------------------------
    with tab2:
        st.header("Model Performance & Insights")

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(6,5))
        sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', xticklabels=['Rock','Mine'], yticklabels=['Rock','Mine'], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # Feature Importance
        st.subheader("Top 15 Feature Importances")
        indices = np.argsort(importances)[::-1][:15]
        fig, ax = plt.subplots(figsize=(8,6))
        sns.barplot(x=importances[indices], y=[f"Signal {i}" for i in indices], palette="viridis", ax=ax, dodge=False)
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Feature")
        st.pyplot(fig)

        # PCA 2D Scatter
        st.subheader("PCA 2D Projection of All Data")
        fig, ax = plt.subplots(figsize=(8,6))
        colors = y_all.map({0:'blue',1:'red'})  # Rock=blue, Mine=red
        ax.scatter(X_pca_all[:,0], X_pca_all[:,1], c=colors)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("PCA Scatter Plot of Sonar Signals")
        st.pyplot(fig)

    # -----------------------------
    # TAB 3: Manual Single Prediction
    # -----------------------------
    with tab3:
        st.header("Manual Input for Single Prediction")
        input_features = []
        with st.expander("Adjust the 60 Sonar Signals"):
            col1, col2, col3 = st.columns(3)
            for i in range(60):
                if i<20:
                    val = col1.slider(f"Signal {i+1}", 0.0, 1.0, 0.5, key=f's{i}')
                elif i<40:
                    val = col2.slider(f"Signal {i+1}", 0.0, 1.0, 0.5, key=f's{i}')
                else:
                    val = col3.slider(f"Signal {i+1}", 0.0, 1.0, 0.5, key=f's{i}')
                input_features.append(val)

        if st.button("▶️ Predict Object", use_container_width=True):
            arr = np.array(input_features).reshape(1,-1)
            pred = model_pipeline.predict(arr)
            prob = model_pipeline.predict_proba(arr).max()
            if pred[0]==0:
                st.success(f"The object is a **ROCK** with {prob*100:.2f}% confidence.")
            else:
                st.error(f"The object is a **MINE** with {prob*100:.2f}% confidence.")

else:
    st.error("Model could not be loaded. Application cannot run.")
