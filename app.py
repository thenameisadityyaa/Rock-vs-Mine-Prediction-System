# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA

st.set_page_config(page_title="Sonar Shield | Rock vs Mine", page_icon="🌊", layout="wide")

st.title("🌊 Sonar Shield: Rock vs. Mine Prediction System")

# ----------------------------
# Function: Load and Train Model
# ----------------------------
@st.cache_resource
def train_model(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1].map({'R': 0, 'M': 1})  # Encode target
    
    # Handle imbalance
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=['Rock', 'Mine'], output_dict=True)
    feature_importances = clf.feature_importances_
    
    return clf, scaler, accuracy, cm, cr, feature_importances, X_train_scaled, X_test_scaled, y_train, y_test

# ----------------------------
# Sidebar: CSV Upload
# ----------------------------
st.sidebar.header("Upload Sonar Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV (60 sonar columns + target column 'R' or 'M')", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, header=None)
    st.write("### Dataset Preview")
    st.dataframe(df.head())
    
    # Train model on uploaded dataset
    with st.spinner("Training model on your dataset..."):
        model, scaler, acc, cm, cr, importances, X_train_scaled, X_test_scaled, y_train, y_test = train_model(df)
    
    st.success(f"Model trained! Accuracy: {acc:.2%}")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Dataset Analysis", "Model Performance", "Batch Prediction", "Manual Prediction"])
    
    # ----------------------------
    # Tab 1: Dataset Analysis
    # ----------------------------
    with tab1:
        st.header("📊 Dataset Analysis")
        st.subheader("Statistical Summary")
        st.dataframe(df.describe())
        
        st.subheader("Target Distribution")
        sns.countplot(x=df.iloc[:, -1])
        st.pyplot(plt.gcf())
        plt.clf()
        
        st.subheader("Correlation Heatmap")
        corr = df.iloc[:, :-1].corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, cmap='viridis')
        st.pyplot(plt.gcf())
        plt.clf()
    
    # ----------------------------
    # Tab 2: Model Performance
    # ----------------------------
    with tab2:
        st.header("🧪 Model Performance")
        st.subheader("Confusion Matrix")
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Rock','Mine'], yticklabels=['Rock','Mine'])
        st.pyplot(plt.gcf())
        plt.clf()
        
        st.subheader("Feature Importances")
        indices = np.argsort(importances)[::-1][:15]
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[indices], y=[f"Signal {i}" for i in indices], palette="viridis")
        st.pyplot(plt.gcf())
        plt.clf()
        
        st.subheader("Classification Report")
        st.dataframe(pd.DataFrame(cr).T)
        
        st.subheader("PCA 2D Projection")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(np.vstack([X_train_scaled, X_test_scaled]))
        y_combined = np.concatenate([y_train, y_test])
        plt.figure(figsize=(8,6))
        scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=y_combined, cmap='bwr', alpha=0.7)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend(handles=scatter.legend_elements()[0], labels=['Rock','Mine'])
        st.pyplot(plt.gcf())
        plt.clf()
    
    # ----------------------------
    # Tab 3: Batch Prediction
    # ----------------------------
    with tab3:
        st.header("📥 Batch Prediction for Uploaded Dataset")
        X_input = df.iloc[:, :-1]
        X_input_scaled = scaler.transform(X_input)
        predictions = model.predict(X_input_scaled)
        df['Predicted'] = ['Rock' if p==0 else 'Mine' for p in predictions]
        st.dataframe(df)
        
        csv_results = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions CSV", data=csv_results, file_name="sonar_predictions.csv", mime="text/csv")
    
    # ----------------------------
    # Tab 4: Manual Prediction
    # ----------------------------
    with tab4:
        st.header("🖊️ Manual Input for Single Prediction")
        input_features = []
        with st.expander("Adjust 60 sonar signal values"):
            cols = st.columns(3)
            for i in range(60):
                col = cols[i%3]
                val = col.slider(f"Signal {i+1}", 0.0, 1.0, 0.5)
                input_features.append(val)
        
        if st.button("Predict Object"):
            input_array = np.array(input_features).reshape(1,-1)
            input_scaled = scaler.transform(input_array)
            pred = model.predict(input_scaled)[0]
            st.subheader("Prediction Result")
            if pred == 0:
                st.success("✅ The object is a ROCK")
            else:
                st.error("💣 The object is a MINE")

else:
    st.info("Upload a CSV dataset to start.")

