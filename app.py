import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    ConfusionMatrixDisplay
)

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Water Potability App", page_icon="💧", layout="wide")

# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.title("💧 Water Potability App")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["🏠 Home", "📊 EDA", "🔬 Prediction", "ℹ️ About"])

# -------------------------------
# Upload Dataset
# -------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("water_potability.csv")

df = df.fillna(df.median())

# -------------------------------
# Train Model (shared across pages)
# -------------------------------
X = df.drop("Potability", axis=1)
y = df["Potability"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

st.sidebar.markdown("---")
st.sidebar.subheader("Choose Model")
model_choice = st.sidebar.selectbox("Select a model", ["Logistic Regression", "Random Forest"])

if model_choice == "Logistic Regression":
    model = LogisticRegression()
else:
    model = RandomForestClassifier()

model.fit(X_train, y_train)

# ================================================
# PAGE 1: HOME
# ================================================
if page == "🏠 Home":
    st.title("💧 Water Potability Prediction App")
    st.markdown("### Welcome!")
    st.write(
        "This app uses machine learning to predict whether water is safe to drink "
        "based on physicochemical properties. Use the sidebar to navigate between pages."
    )

    st.markdown("---")
    st.subheader("Dataset Overview")
    st.write(f"**Total samples:** {len(df)}  |  **Features:** {len(df.columns)-1}  |  "
             f"**Potable:** {int(df['Potability'].sum())}  |  "
             f"**Not Potable:** {int((df['Potability'] == 0).sum())}")
    st.dataframe(df.head())

# ================================================
# PAGE 2: EDA
# ================================================
elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")

    st.subheader("Dataset Statistics")
    st.dataframe(df.describe())

    st.markdown("---")
    st.subheader("Potability Distribution")
    fig, ax = plt.subplots()
    df["Potability"].value_counts().plot(kind="bar", ax=ax, color=["#e74c3c", "#2ecc71"], edgecolor="black")
    ax.set_xticklabels(["Not Potable", "Potable"], rotation=0)
    ax.set_ylabel("Count")
    ax.set_title("Potability Count")
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df
